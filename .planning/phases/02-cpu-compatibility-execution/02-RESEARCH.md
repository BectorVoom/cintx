# Phase 2 Research: CPU Compatibility Execution

**Phase:** 02
**Date:** 2026-03-14
**Primary inputs:**
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/ROADMAP.md`
- `.planning/phases/01-contracts-and-typed-foundations/*`
- `docs/libcint_detailed_design_resolved_en.md`
- `libcint-master/include/cint.h.in`
- `libcint-master/include/cint_funcs.h`
- `libcint-master/doc/program_ref.txt`
- `libcint-master/src/cint{1e,2e,2c2e,3c1e,3c2e}.c`
- `libcint-master/src/cint_bas.c`

## Objective and Phase Boundary

Phase 2 should deliver real CPU execution for stable families through both safe and raw interfaces, using Phase 1 contracts as fixed inputs.

In scope:
- CPU execution path for stable families: `1e`, `2e`, `2c2e`, `3c1e`, `3c2e`.
- Safe `evaluate`/`evaluate_into` APIs (`SAFE-03`) with representation-correct layout.
- Raw compatibility execution APIs that accept libcint-style `atm/bas/env`, `shls`, `dims`, `cache`, `opt` (`RAW-01`, `RAW-02`, `RAW-03`).
- Memory-limit behavior with chunking attempts and explicit typed failures (`MEM-01`, `MEM-02`).
- CPU as the correctness baseline backend (`EXEC-01`).

Out of scope for this phase:
- GPU/CubeCL execution (`EXEC-02`, `EXEC-03`).
- C ABI shim (`ABIC-01`).
- Manifest governance and full CI lock/oracle gates (`COMP-03`, `COMP-04`, `VERI-*`, `RAW-04`).
- Optional family rollout (`with-f12`, `with-4c1e`) beyond stable baseline.

## Requirement Targets (Explicit)

| Requirement | What must exist by end of Phase 2 |
|---|---|
| `COMP-01` | Safe and raw execution for stable families (`1e/2e/2c2e/3c1e/3c2e`) with oracle-tolerance numerical checks across supported cart/sph/spinor paths. |
| `RAW-01` | Public raw execution surface with libcint-compatible input contracts (`atm/bas/env`, `shls`, `dims`, `cache`, `opt`) and strict validation. |
| `RAW-02` | Raw workflow supports workspace query then execution; null-equivalent query semantics are preserved. |
| `RAW-03` | `dims` and output size mismatches fail before execution/write; no silent truncation and no partial writes returned as success. |
| `SAFE-03` | `evaluate` and `evaluate_into` typed APIs produce representation-correct output layout. |
| `MEM-01` | `memory_limit_bytes` leads to chunked execution when feasible, otherwise `MemoryLimitExceeded`. |
| `MEM-02` | Supported paths avoid unhandled OOM abort by using fallible allocation policy and typed allocation errors. |
| `EXEC-01` | CPU backend executes all Phase-2-supported operations as the reference runtime path. |

## Current Baseline and Gap Analysis

Phase 1 provides reusable contracts but no execution path yet:
- Exists: typed models/errors, deterministic query path, diagnostics envelope, shared query validator.
- Missing: execution planner, raw `atm/bas/env` validation layer, raw/safe evaluate APIs, FFI backend integration, no-partial-write writer, fallible workspace allocator, chunk scheduler.

Critical mismatch to fix before execution planning:
- `src/runtime/validator.rs` currently computes natural dims from shell angular momentum only. It does not incorporate `NCTR` and does not apply libcint spinor `kappa` rules.
- For Phase 2 execution this must be replaced with libcint-compatible contracted dimension logic:
  - cart: `(l+1)(l+2)/2 * nctr`
  - sph: `(2l+1) * nctr`
  - spinor: `spinor_len(l,kappa) * nctr`

## Standard Stack

Use this stack for Phase 2 implementation (no optional exploration):
- `thiserror` (already present): typed public errors.
- `tracing` (already present): execution/chunking/failure diagnostics.
- `num-complex`: safe spinor output typing and layout conversion support.
- `smallvec`: shell tuple and dims/stride internals with reduced heap churn.
- `rayon`: chunk-level CPU parallelism only where determinism/ordering are preserved.
- `cc` (build-dependency): compile vendored libcint CPU sources for hermetic CPU backend.
- `bindgen` (dev/build tooling): generate stable FFI declarations for oracle/dev and reduce handwritten signature drift.
- `approx` (dev-dependency): tolerance assertions for oracle comparisons.
- `proptest` (dev-dependency): property tests for raw dims/layout/shape invariants.

Do not make GPU/runtime dependencies part of this phase implementation path.

## Architecture Patterns

1. Shared Planner for Safe and Raw Surfaces
- Both safe and raw evaluation must build one canonical execution plan.
- Safe and raw differ only in input decoding and output adapters.
- Planner owns: validated operator/family/representation, logical dims, required bytes, and chunk plan.

2. CPU Backend Adapter over Vendored libcint
- Implement CPU backend through a narrow FFI adapter layer.
- For supported APIs, backend invocation should call corresponding libcint entry points.
- This minimizes numerical risk and accelerates compatibility delivery.

3. Separate Query vs Execute Contracts
- Preserve explicit two-step model:
  - `query_workspace*` computes required bytes/dims.
  - `evaluate*` executes with validated output/workspace.
- Raw execution also supports sentinel query semantics compatible with libcint (`out == NULL`).

4. Raw Validator as Unsafe Boundary
- Introduce dedicated raw view/validator module for:
  - slot-width checks (`ATM_SLOTS=6`, `BAS_SLOTS=8`),
  - pointer-offset range checks into `env`,
  - shell tuple arity validation,
  - dims length/value checks per family arity.
- Never cast offsets to `usize` before sign/range validation.

5. Canonical Dims and Buffer Contract
- In compat layer, `dims == NULL` means natural dims.
- Non-null dims must match contract arity (`2`, `3`, or `4` depending on family).
- Treat invalid dims or undersized outputs as typed errors before backend execution.
- No partial writes are allowed on error returns.

6. Writer Strategy that Prevents Silent Partial Writes
- Use pre-validated required element count before any backend write path.
- If provided output is insufficient, fail with typed error without backend call.
- If backend must write into temporary buffer, only copy to user output after complete success.

7. Fallible Allocation and Memory-Limit Enforcement
- Add central allocator abstraction for internal workspace (`try_alloc_uninit` style).
- For allocations above trivial threshold, route through this allocator only.
- Enforce `memory_limit_bytes` at plan/build time and at execution-chunk boundaries.

8. Chunking as Planner Concern
- Planner attempts chunking when full-plan workspace exceeds limit.
- If no legal chunk strategy exists for the request, return `MemoryLimitExceeded`.
- Chunk metadata must be surfaced in tracing and execution stats.

## Implementation Blueprint for Planning

Recommended plan slices for Phase 2:

1. CPU FFI and Operator Routing
- Add FFI bindings for selected stable-family symbols.
- Build resolver from `(family, operator kind, representation)` to libcint function pointers.
- Add typed unsupported checks where mappings do not exist.

2. Raw Layout and Contract Validation
- Add `RawAtmView`, `RawBasView`, `RawEnvView`, `CompatDims`, and `RawShellTuple` validators.
- Validate slot widths and offsets to prevent UB before FFI calls.
- Define raw query/eval entry points with explicit safety docs.

3. Execution Planner and Output Layout Engine
- Replace query-only shape logic with contracted, representation-correct dimension derivation.
- Implement output shape/layout normalization for safe tensor views and raw flat buffers.
- Add no-partial-write guard path.

4. Safe Evaluate APIs
- Implement `evaluate_into` first (caller-owned output, minimal allocation risk).
- Implement `evaluate` convenience wrapper with fallible output allocation.
- Keep safe API free of raw `dims` parameters.

5. Memory and OOM Guarantees
- Add fallible workspace allocator.
- Implement chunk planner with explicit infeasible-path errors.
- Ensure raw path never depends on libcint internal malloc for supported flows unless through controlled fallback policy.

6. Phase-2 Compatibility Tests
- Add requirement-mapped tests for safe/raw execution across families and representations.
- Add dims/buffer mismatch tests proving zero partial writes.
- Add memory-limit chunk or fail tests and fallible-allocation failure-path tests.

## Validation Architecture

This section is intentionally prescriptive so Phase 2 planning can map tasks directly to tests.

### 1. Test Layers

- Layer A: Contract and validation tests.
- Layer B: Safe vs raw execution equivalence tests.
- Layer C: Oracle-tolerance correctness tests.
- Layer D: Failure semantics tests (`dims`, buffer, memory, allocation).

### 2. Requirement-to-Test Matrix

| Requirement | Required validation |
|---|---|
| `COMP-01` | For each stable family and representation supported in Phase 2, compare safe/raw outputs against oracle values within fixed tolerance policy. |
| `RAW-01` | Raw API integration tests with libcint-style `atm/bas/env/shls/dims/cache/opt` inputs, including validator rejection cases. |
| `RAW-02` | Workspace query then execute flow tests; raw sentinel query behavior tests (`out/cache` null-equivalent paths). |
| `RAW-03` | Invalid dims and undersized output tests that assert typed failure and unchanged output buffer bytes. |
| `SAFE-03` | `evaluate`/`evaluate_into` shape/layout correctness tests for cart/sph/spinor. |
| `MEM-01` | Memory-limit tests that assert chunked execution path where feasible, otherwise `MemoryLimitExceeded`. |
| `MEM-02` | Fail-allocator tests proving typed allocation failure instead of process abort. |
| `EXEC-01` | CPU backend end-to-end tests for all supported Phase 2 operator/family/rep combinations. |

### 3. Mandatory Negative Tests

- Invalid `atm` length not divisible by `ATM_SLOTS`.
- Invalid `bas` length not divisible by `BAS_SLOTS`.
- `env` offsets out of range.
- Shell tuple arity mismatch by family.
- `dims` length mismatch by family (`2/3/4`).
- `dims` values inconsistent with natural dims policy.
- Output buffer too small.
- Memory limit too small with no legal chunk.
- Allocator returns failure.

### 4. Oracle Strategy for Phase 2

- Use vendored libcint as oracle source for numerical comparisons.
- Avoid tautological tests where wrapper and oracle are the same call path with identical adapters.
- Keep at least one independent oracle path per family/representation in tests (for example direct oracle invocation fixture path) to verify writer/layout logic.

### 5. Trace and Diagnostics Assertions

All failure tests should assert diagnostics completeness at minimum:
- `api`
- `representation`
- `shell_tuple`
- `dims`
- `required_bytes`
- `provided_bytes`
- `memory_limit_bytes`
- `backend_candidate`
- `feature_flags`

## Don't Hand-Roll

Do not custom-build these in Phase 2:
- Numerical integral kernels for stable families (use vendored libcint CPU backend for baseline execution).
- Per-API ad hoc dims formulas (use one central dims/required-elements contract).
- Multiple allocators spread across modules (use one fallible workspace allocator boundary).
- Independent safe/raw planners (one planner only).
- String-only error classification (keep typed variants + diagnostics payload).

## Common Pitfalls

1. Treating current Phase 1 natural dims as execution-ready
- Current logic ignores `NCTR` and spinor `kappa`; using it for execution will produce wrong shapes.

2. Letting backend write before buffer-size validation
- This causes silent partial writes or output corruption semantics.

3. Relying on libcint internal `malloc` in supported paths
- This undermines `MEM-02` guarantees and can cause process abort behavior outside typed control.

4. Assuming all upstream spinor stable-family paths are safe
- `libcint-master/src/cint3c1e.c` contains a `not implemented` spinor driver that exits.
- Phase plan must explicitly decide how to satisfy `COMP-01` expectation for `3c1e` spinor paths.

5. Coupling memory-limit behavior to only query-time checks
- Execution-time chunk feasibility and allocation failures must be checked too.

6. Tautological oracle tests
- If oracle and tested path share the same adapter bugs, compatibility issues will pass undetected.

## Planning Risks Requiring Explicit Decisions

1. `3c1e` spinor compatibility risk (highest)
- Upstream `CINT3c1e_spinor_drv` exits with "not implemented".
- Required planning decision: either implement an alternate supported path for Phase 2 or explicitly constrain/reclassify this envelope with requirement-owner approval.

Decision (2026-03-14, revision pass):
- Phase 2 will **implement** this envelope (no re-scope): add a dedicated `3c1e` spinor adapter route in plan `02-06` and enforce it via routing + matrix/oracle tests in `02-05`.
- Unsupported-path typed errors remain valid only for out-of-phase envelopes, not for stable-family `3c1e` spinor requests covered by `COMP-01`.

2. Scope of stable operator set inside each family
- Current `OperatorKind` enum is minimal (`Overlap`, `Kinetic`, `NuclearAttraction`, `ElectronRepulsion`).
- Required planning decision: define exact symbol set covered by `COMP-01` in this phase and test matrix accordingly.

3. Chunking feasibility definition
- Some requests may not admit chunking under strict layout/output ownership constraints.
- Required planning decision: formalize "chunkable" vs "must fail" conditions for `MEM-01`.

4. Memory accounting policy
- Required planning decision: whether `memory_limit_bytes` applies to internal workspace only or workspace + convenience-output allocations.

## Code Examples

```rust
// Safe API: caller-owned output avoids hidden output allocations.
pub fn evaluate_into<T: OutputElement>(
    basis: &BasisSet,
    operator: Operator,
    representation: Representation,
    shell_tuple: &[usize],
    options: &ExecutionOptions,
    out: &mut OutputTensorMut<'_, T>,
) -> QueryResult<ExecutionStats> {
    let plan = runtime::planner::build_safe_plan(basis, operator, representation, shell_tuple, options)?;
    runtime::executor::execute_plan_into(plan, out)
}
```

```rust
// Raw API: libcint-compatible query/execute flow with strict validation.
pub unsafe fn eval_raw(
    api: RawApiId,
    out: Option<&mut [f64]>,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
    cache: Option<&mut [f64]>,
) -> Result<RawEvalSummary, LibcintRsError> {
    let validated = compat::validator::validate_raw_eval(api, out.as_deref(), dims, shls, atm, bas, env, opt, cache.as_deref())?;
    runtime::executor::execute_raw(validated)
}
```

```rust
// Memory-limit behavior: chunk if possible, else explicit typed failure.
fn build_chunk_plan(req: &ExecutionRequest, limit: Option<usize>) -> Result<ChunkPlan, LibcintRsError> {
    let full = estimate_workspace(req)?;
    if limit.is_none_or(|bytes| full.required_bytes <= bytes) {
        return Ok(ChunkPlan::single(full));
    }

    match chunker::split(req, limit.unwrap()) {
        Some(plan) => Ok(plan),
        None => Err(LibcintRsError::MemoryLimitExceeded {
            required_bytes: full.required_bytes,
            limit_bytes: limit.unwrap(),
        }),
    }
}
```

## Phase 2 Exit Checklist for Planning

- [ ] CPU backend executes all Phase-2-supported stable-family/representation combinations.
- [ ] Safe `evaluate`/`evaluate_into` implemented with representation-correct output layout.
- [ ] Raw `query/eval` supports libcint-compatible contracts for `atm/bas/env/shls/dims/cache/opt`.
- [ ] Invalid dims/buffer cases fail pre-write with typed errors and zero partial-write semantics.
- [ ] Memory-limit behavior demonstrates chunking path or explicit `MemoryLimitExceeded`.
- [ ] Supported execution paths use fallible allocation and return typed allocation failures.
- [ ] Requirement-mapped tests exist for `COMP-01`, `RAW-01`, `RAW-02`, `RAW-03`, `SAFE-03`, `MEM-01`, `MEM-02`, `EXEC-01`.

## Confidence

- Overall confidence: **Medium-High**.
- High confidence on architecture and contract patterns.
- Medium confidence on full spinor-family envelope due to known upstream `3c1e` spinor limitation that must be explicitly resolved during planning.
