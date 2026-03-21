# Phase 2: Execution & Compatibility Stabilization - Research

**Researched:** 2026-03-21
**Domain:** Raw libcint compatibility, shared execution planning, CubeCL backend integration
**Confidence:** MEDIUM

<user_constraints>
## User Constraints

No Phase 2 `CONTEXT.md` exists. The planner must inherit the locked constraints below from `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `docs/design/cintx_detailed_design.md`, and `AGENTS.md`.

### Locked Scope
- Address exactly `COMP-01`, `COMP-02`, `COMP-03`, `COMP-05`, `EXEC-02`, `EXEC-03`, `EXEC-04`, and `EXEC-05`.
- Build on Phase 1 foundations; do not redesign the `core` / `ops` / `runtime` split first.
- Keep upstream libcint `6.1.3` result compatibility as the target.
- Keep CubeCL as the only compute backend being planned for execution work.

### Locked Behavior
- Safe Rust API first, raw compat API second, optional C ABI shim third. Phase 2 is about the shared runtime plus raw compat surface, not the C ABI shim.
- Public library errors use `thiserror`; oracle, xtask, benchmarks, and other tooling use `anyhow`.
- Memory pressure must chunk safely or fail with typed errors and no partial writes.
- Coverage claims must remain tied to the canonical compiled manifest lock and helper/transform parity gates.

### Deferred / Out Of Scope
- `COMP-04`, `EXEC-01`, `OPT-01`, `OPT-02`, `OPT-03`
- Public C ABI shim
- Safe facade ergonomics and builder polish
- GTG
- Promoting optional or unstable-source APIs beyond the fixed Phase 2 requirement set
- 4c1e execution work, even though the current manifest lock already contains experimental `4c1e` entries
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COMP-01 | Compat caller can invoke raw APIs using `atm`, `bas`, `env`, `shls`, `dims`, `opt`, and `cache` inputs that match documented layout contracts. | Architecture Patterns 1-3, Don't Hand-Roll rows 1-3, Common Pitfalls 2-5 |
| COMP-02 | Compat caller can query required output sizes and workspace requirements without performing a full evaluation or writing output buffers. | Summary, Architecture Pattern 2, Don't Hand-Roll row 2, Code Examples `query_workspace_raw` flow |
| COMP-03 | Compat caller can use helper, transform, optimizer-lifecycle, and legacy wrapper APIs that are included in the upstream compatibility scope. | Summary, Standard Stack, Architecture Pattern 3, Open Question 1 |
| COMP-05 | Compat caller receives typed validation failures or explicit `UnsupportedApi` errors instead of silent truncation, partial writes, or undefined behavior. | Summary, Architecture Patterns 1-2, Common Pitfalls 3-5, State of the Art |
| EXEC-02 | Rust or compat caller can evaluate supported 1e, 2e, 2c2e, 3c1e, and 3c2e families through the shared planner and CubeCL backend. | Standard Stack, Architecture Pattern 4, Open Questions 2-3 |
| EXEC-03 | Caller can enforce memory limits so large evaluations chunk safely or fail with typed memory-limit or allocation errors and no partial writes. | Summary, Architecture Pattern 2, Don't Hand-Roll row 3, Common Pitfall 4 |
| EXEC-04 | Caller receives outputs with upstream-compatible cart, sph, and spinor shapes, ordering, and complex-layout semantics. | Architecture Pattern 2, Common Pitfall 5, Code Examples `required_elems_from_dims` and CubeCL launch/write split |
| EXEC-05 | Caller gets numerically equivalent results within accepted tolerance regardless of whether optimizer support is enabled. | Architecture Pattern 5, Common Pitfall 6, State of the Art |
</phase_requirements>

## Summary

Phase 2 is not ready to start with kernels. The codebase already has the right high-level boundaries, but the actual Phase 2 crates are still mostly disconnected: `cintx-runtime` owns a validated `query_workspace()` / `evaluate()` contract, while `cintx-compat`, `cintx-cubecl`, `cintx-oracle`, `cintx-rs`, and `cintx-capi` are still stubs and are not part of the active workspace build. The current canonical manifest lock contains only 22 operator entries across `1e`, `2e`, `2c2e`, `3c1e`, `3c2e`, and experimental `4c1e`; it does not currently encode the helper, transform, optimizer-lifecycle, or legacy-wrapper entries that `COMP-03` requires. The public `cintxRsError` enum is also still missing the raw-layout, env-offset, and buffer-size failures that the design calls for in `COMP-05`.

The raw compat contract is stricter than the current Rust surface. Upstream libcint accepts sentinel arguments such as `out == NULL`, `cache == NULL`, and `dims == NULL`; the design resolves that by making `query_workspace()` the canonical internal API, then mapping raw sentinel behavior onto that split. `dims` belongs only to compat/C layers, its length depends on family arity, its values exclude the component axis, and both undersized and oversized overrides must fail instead of truncating. Spinor buffers are flat interleaved doubles, not a separate layout family. Any path that writes output before layout validation, workspace validation, and allocation success is a contract bug.

The right Phase 2 plan is therefore a staged hardening phase: first make the manifest and workspace honest about what is actually in scope, then build a single shared raw-to-runtime pipeline, then wire CubeCL execution and optimizer parity on top of that. Keep this phase on base families only: `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e`. Treat `4c1e`, F12/STG/YP, GTG, the safe facade, and the C ABI shim as out of scope here even if some scaffolding already exists.

**Primary recommendation:** Start Phase 2 with a Wave 0 that adds the Phase 2 crates to the workspace, extends the canonical manifest to compat/helper coverage, and finishes the raw error/layout contract before any CubeCL execution code lands.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cintx-core` | local `0.1.0` | Typed atoms, shells, basis, tensor metadata, and public error enums | Phase 1 already validated this crate as the typed source of truth; raw compat should convert into it, not bypass it. |
| `cintx-ops` | local `0.1.0` | Canonical manifest lock and resolver metadata | All family/representation dispatch is supposed to be manifest-driven. |
| `cintx-runtime` | local `0.1.0` | Validation, workspace estimation, chunking, and query/evaluate contract | Phase 1 already proved this is the shared planning layer; Phase 2 should extend it, not duplicate it. |
| `cintx-compat` | local `0.1.0` | Raw `atm/bas/env` compat API, helper APIs, legacy wrappers, flat output writer | This is the design-authoritative boundary for `COMP-01` through `COMP-03`. |
| `cintx-cubecl` + `cubecl` | local `0.1.0` + `0.9.0` stable | Backend executor, transfers, transforms, specialization, device cache | `cargo info cubecl` and docs.rs both show `0.9.0` as the current stable release; `0.10.0-pre.2` exists but is a prerelease, so Phase 2 should stay on `0.9.0` unless a verified blocker forces a change. |
| `cintx-oracle` | local `0.1.0` | Vendored upstream libcint comparison harness | Result compatibility is the project’s primary gate, so the oracle crate is not optional for Phase 2 planning even if heavy CI comes later. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tracing` | workspace pin `0.1.41` (latest `0.1.44`) | Planner, chunking, dispatch, transfer, and OOM diagnostics | Keep the existing `0.1.x` line for Phase 2; upgrade only as a deliberate dependency refresh. |
| `thiserror` | `2.0.18` | Public typed library errors | Required for the raw validation and no-partial-write contract. |
| `anyhow` | `1.0.102` | Oracle, xtask, benches, and dev-only tooling errors | Use at the app boundary only; not in the library API. |
| `smallvec` | workspace uses `^1.13` (latest stable `1.15.1`) | Shell tuples, extents, and stride/dims storage without avoidable heap churn | Keep the stable `1.x` line; do not jump to `2.0.0-alpha.*` in this phase. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cubecl 0.9.0` stable | `cubecl 0.10.0-pre.2` | Only justified if a verified `0.9.0` blocker exists; otherwise it adds prerelease churn to the same phase that already has major surface-area risk. |
| Shared `cintx-runtime` pipeline for compat execution | Separate compat-only planner/executor path | Faster to prototype, but it guarantees drift between safe/compat semantics and makes later parity work harder. |
| Manifest-resolved helper/legacy scope | Hand-maintained wrapper lists | Easier short term, but it breaks the project’s “compiled manifest is the API source of truth” rule. |

**Workspace enablement before coding:**
```toml
[workspace]
members = [
    ".",
    "crates/cintx-core",
    "crates/cintx-ops",
    "crates/cintx-runtime",
    "crates/cintx-compat",
    "crates/cintx-cubecl",
    "crates/cintx-oracle",
]
```

**Version verification:**
- `cargo info cubecl` reports `0.9.0` with `0.10.0-pre.2` as the latest prerelease.
- `cargo info tracing` reports `0.1.44`; the workspace currently pins `0.1.41`.
- `cargo info thiserror` reports `2.0.18`.
- `cargo info anyhow` reports `1.0.102`.
- `cargo info smallvec` reports stable `1.15.1`; the workspace currently uses `^1.13`.

## Architecture Patterns

### Recommended Project Structure
```text
crates/
├── cintx-core/      # Typed atoms/shells/basis/tensor/error primitives
├── cintx-ops/       # Canonical manifest lock + resolver metadata
├── cintx-runtime/   # Validation, workspace query, chunking, dispatch decisions
├── cintx-compat/    # Raw atm/bas/env adapters, helpers, legacy wrappers, flat writer
├── cintx-cubecl/    # Backend executor, transfers, transforms, specialization, device cache
└── cintx-oracle/    # Vendored libcint comparison harness and fixtures
```

### Pattern 1: Manifest-First Compat Dispatch
**What:** Resolve every compat call through the canonical manifest and current `Resolver` metadata before choosing arity, representation, kernel family, or unsupported-path behavior.
**When to use:** Every raw compat function, helper dispatcher, optimizer lifecycle entrypoint, and legacy wrapper.
**Example:**
```rust
// Adapted from design §§4.6, 5.5, and 7.2 plus the current Resolver/runtime contracts.
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
) -> Result<RawEvalSummary, cintxRsError> {
    let raw = RawInput::new(api, dims, shls, atm, bas, env, opt)?;
    let typed = raw.validate_against_manifest()?;
    let query = typed.query_workspace()?;
    if out.is_none() {
        return Ok(RawEvalSummary::workspace_only(query));
    }
    typed.execute_into_compat_buffer(out.unwrap(), cache, query)
}
```

### Pattern 2: Central `CompatDims` and Layout Writer
**What:** Validate `dims`, compute required flat element counts, and write cart/sph/spinor outputs through one shared compat writer. The component axis comes from manifest metadata, never from `dims`.
**When to use:** Any path that accepts `dims`, writes to a flat `double*`, or handles spinor/complex layout.
**Example:**
```rust
// Adapted from design §3.6.1 and libcint's program_ref ordering rules.
fn required_elems_from_dims(
    arity: usize,
    component_count: usize,
    dims: &[usize],
    complex_interleaved: bool,
) -> Result<usize, cintxRsError> {
    if dims.len() != arity {
        return Err(cintxRsError::InvalidDims {
            expected: arity,
            provided: dims.len(),
        });
    }
    let base = dims.iter().product::<usize>() * component_count;
    Ok(if complex_interleaved { base * 2 } else { base })
}
```

### Pattern 3: Thin Legacy Wrappers
**What:** Reproduce upstream `cNAME_*` / `_optimizer` wrappers as thin forwards into the raw compat pipeline, matching the macro behavior in `src/misc.h`.
**When to use:** `COMP-03` legacy wrapper coverage.
**Example:**
```rust
// Adapted from libcint's ALL_CINT / ALL_CINT1E macros in src/misc.h.
pub unsafe fn cint2e_sph_legacy(
    out: &mut [f64],
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<i32, cintxRsError> {
    eval_raw(RawApiId::Int2eSph, Some(out), None, shls, atm, bas, env, opt, None)
        .map(|summary| summary.not0)
}
```

### Pattern 4: Runtime/Backend Separation
**What:** Keep `cintx-runtime` backend-neutral. It validates, estimates workspace, decides chunking, and records dispatch metadata. `cintx-cubecl` owns client creation, device buffers, kernel launch, transforms, and transfer planning.
**When to use:** All evaluation work for `EXEC-02` and `EXEC-03`.
**Example:**
```rust
// Adapted from design §5.2.
pub trait BackendExecutor {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool;
    fn execute(&self, plan: &ExecutionPlan<'_>, io: &mut ExecutionIo<'_>)
        -> Result<ExecutionStats, cintxRsError>;
}
```

### Pattern 5: Immutable Optimizer Handles With Shared Output Path
**What:** Build optimizer state once, then make optimized and non-optimized execution use the same output writer and the same observable tensor/layout contract.
**When to use:** `COMP-03` optimizer lifecycle work and `EXEC-05` parity work.
**Example:**
```rust
// Adapted from design §§3.8 and 7.8.
pub struct RawOptimizerHandle {
    inner: Arc<OptimizerCache>,
}

pub fn init_optimizer(raw: &RawInput<'_>) -> Result<RawOptimizerHandle, cintxRsError> {
    Ok(RawOptimizerHandle { inner: Arc::new(build_cache(raw)?) })
}
```

### Anti-Patterns to Avoid
- **Compat-specific execution logic:** Do not let `cintx-compat` choose kernels or chunking policy on its own; it should call into `cintx-runtime`.
- **Ad hoc `dims` math per API:** One bug in one family becomes a silent layout regression. Centralize it.
- **Writing output before validation finishes:** This violates `COMP-05` immediately.
- **Pulling `4c1e` into Phase 2 because it is in the lock:** The roadmap still places it in Phase 3 optional-family work.
- **Adding helper/legacy APIs without manifest coverage:** That would satisfy callability but break the project’s verification model.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Raw `atm` / `bas` / `env` parsing | Scattered index arithmetic in each compat function | Central `RawAtmView`, `RawBasView`, `RawEnvView`, and slot constants | Upstream slot widths and offsets are easy to misread; one parser bug poisons every API. |
| `dims` and output-size calculation | Per-function manual buffer formulas | A single `CompatDims` + `required_elems_from_dims()` + layout writer | Prevents partial writes, over-acceptance of large dims, and component-axis confusion. |
| OOM handling | Direct `Vec`, `vec![0; n]`, `malloc`, or retry loops on hot paths | `WorkspaceAllocator` + `FallibleBuffer` + planner chunking | The design explicitly forbids abort-path allocation and partial-write fallback. |
| Legacy wrappers | Hand-authored wrapper functions for every `cNAME_*` variant | Thin forwards generated from manifest metadata or a small wrapper macro layer | The wrapper surface is wide and mechanically repetitive. |
| Backend routing | Symbol-name switches in compat code | `Resolver` + `runtime::dispatch` + a backend trait | Keeps backend details out of the compat API and preserves future extensibility. |
| Compatibility goldens | Custom expected-value blobs | Vendored libcint oracle harness | The requirement is upstream result compatibility, not self-consistency. |

**Key insight:** In this phase, “small convenience paths” are the real compatibility surface. `dims`, flat spinor buffers, helper counts, legacy wrappers, optimizer handles, and `cache == NULL` are all correctness-critical. Centralize each one exactly once.

## Common Pitfalls

### Pitfall 1: Treating the Current Manifest as Phase-Complete
**What goes wrong:** Planning assumes `COMP-03` is already covered because the resolver and generated tables exist.
**Why it happens:** The current lock and generated CSV only contain 22 operator entries; helper/transform/optimizer/legacy entries are absent.
**How to avoid:** Make manifest extension a Wave 0 task, and block helper/legacy implementation on it.
**Warning signs:** `rg` over the generated manifest finds only `HelperKind::Operator`, or helper APIs exist in code but not in generated artifacts.

### Pitfall 2: Forgetting That Phase 2 Crates Are Outside the Active Workspace
**What goes wrong:** Compat/CubeCL/oracle code is written, but CI and `cargo test --workspace` never build it.
**Why it happens:** The current workspace includes only the root package, `cintx-core`, `cintx-ops`, and `cintx-runtime`.
**How to avoid:** First wave adds `cintx-compat`, `cintx-cubecl`, and `cintx-oracle` to workspace members and default verification commands.
**Warning signs:** `cargo test --workspace` stays green even after heavy edits to Phase 2 crates because those crates are still excluded.

### Pitfall 3: Misreading `dims`
**What goes wrong:** Compat paths treat `dims` as including the component axis, accept oversized overrides, or silently truncate to fit.
**Why it happens:** The raw C API is permissive-looking, but the design intentionally narrows it.
**How to avoid:** Enforce the design rule: `dims` length equals family arity, values exclude `comp`, and any mismatch is `InvalidDims`.
**Warning signs:** Buffer-size math varies by API, or tests try to “round up” dimensions.

### Pitfall 4: Writing Partial Results on Allocation or Memory-Limit Failure
**What goes wrong:** A chunk starts writing, then OOM or size mismatch occurs mid-path.
**Why it happens:** Compat writers often look like simple memcpy destinations, so it is easy to validate too late.
**How to avoid:** Validate layout, compute required bytes, complete allocation, and finalize chunk plan before any caller-visible writes.
**Warning signs:** Output slices are borrowed early, or `cache == NULL` causes allocation after output writing has started.

### Pitfall 5: Getting Spinor Layout Wrong
**What goes wrong:** Flat output length, ordering, or interleaving is wrong even when numerical values are correct.
**Why it happens:** Spinor is still a flat `double*`-compatible buffer in legacy/raw layers, but it represents complex values and uses representation-specific AO counts.
**How to avoid:** Keep spinor counts and writer semantics centralized and test them against libcint’s documented ordering and `CINTlen_spinor` / `CINTcgto_spinor` formulas.
**Warning signs:** Spinor buffer sizes are computed with real-only counts, or writer code treats spinor like sph/cart.

### Pitfall 6: Optimizer Path Drift
**What goes wrong:** Optimized and non-optimized calls return different shapes, `not0` summaries, or numeric outputs.
**Why it happens:** Optimizer plumbing gets added as a separate execution path instead of as optional immutable input to the same pipeline.
**How to avoid:** Keep optimizer use as a cache input to the same planner/backend/writer path, then compare on/off behavior.
**Warning signs:** Separate writer code for optimized calls, or tests cover only “optimizer exists” and not parity.

### Pitfall 7: Over-Scoping Into Optional Families
**What goes wrong:** Phase 2 gets dragged into `4c1e`, F12/STG/YP, or GTG bring-up.
**Why it happens:** The manifest and design mention those families, and `4c1e` already appears in the current lock.
**How to avoid:** Keep this phase on base families plus compat/helper/optimizer parity only. Return explicit `UnsupportedApi` for out-of-scope cases.
**Warning signs:** Plan tasks mention `with-f12`, `with-4c1e`, or GTG without an explicit roadmap exception.

## Code Examples

Verified patterns adapted from primary sources:

### Raw Compat Query/Evaluate Split
```rust
// Adapted from design §§5.5 and 7.2 plus libcint's out/cache sentinel behavior.
pub unsafe fn query_workspace_raw(
    api: RawApiId,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<WorkspaceQuery, cintxRsError> {
    let raw = RawInput::new(api, dims, shls, atm, bas, env, opt)?;
    raw.validate_against_manifest()?.query_workspace()
}
```
Source: design §5.5, design §7.2, `libcint-master/src/cint1e.c`, `libcint-master/src/cint2e.c`

### Thin Legacy Wrapper Forwarding
```rust
// Adapted from libcint's wrapper macros in src/misc.h.
pub unsafe fn cint1e_ovlp_cart_legacy(
    out: &mut [f64],
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<i32, cintxRsError> {
    eval_raw(RawApiId::Int1eOvlpCart, Some(out), None, shls, atm, bas, env, None, None)
        .map(|summary| summary.not0)
}
```
Source: `libcint-master/src/misc.h`

### CubeCL Launch Skeleton
```rust
// Adapted from the CubeCL README/docs.rs launch pattern.
unsafe {
    kernel::launch_unchecked::<R>(
        &client,
        CubeCount::Static(grid_x, grid_y, 1),
        CubeDim::new_1d(block_x),
        input_arg,
        output_arg,
    );
}
```
Source: `/tracel-ai/cubecl` Context7 docs, docs.rs `cubecl 0.9.0`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Handwritten symbol lists and header-only API assumptions | Canonical compiled manifest lock across the support matrix | Design 0.4-resolved and Phase 1 on 2026-03-21 | Phase 2 should extend the lock, not create side registries for helpers or wrappers. |
| C-style sentinel API as the only public model | Internal `query_workspace()` / `evaluate()` split, with compat mapping back to sentinels | Design 0.4-resolved and Phase 1 on 2026-03-21 | Raw compat should translate sentinel behavior into the shared runtime contract. |
| Best-effort allocation and implicit partial write tolerance | Fallible allocation plus typed OOM / memory-limit errors and no partial writes | Design 0.4-resolved and Phase 1 on 2026-03-21 | Layout validation and allocation must complete before caller-visible writes. |
| Separate per-surface execution behavior | Shared planner and backend path with compat/safe facades as adapters | Design 0.4-resolved | Compat work that bypasses `cintx-runtime` is architectural debt, not progress. |

**Deprecated / outdated:**
- GTG in the public surface: still explicitly out of scope.
- Assuming helper coverage from headers alone: the compiled manifest lock is the authoritative audit artifact.
- Treating `dims > natural` as acceptable padding: the design explicitly rejects larger values rather than silently tolerating them.

## Open Questions

1. **Should Wave 0 expand the canonical manifest to include helper/transform/optimizer-lifecycle/legacy entries?**
   - What we know: The roadmap requires `COMP-03`, upstream `include/cint.h.in` defines those APIs, and the current canonical lock/generated artifacts do not contain them.
   - What's unclear: Whether Phase 1 intentionally deferred that lock extension or simply stopped at operator families.
   - Recommendation: Treat manifest expansion as mandatory Wave 0 work for Phase 2.

2. **Which CubeCL runtime should be the first fully supported bring-up target for local development and CI?**
   - What we know: CubeCL `0.9.0` supports `wgpu`, `cuda`, `hip`, and `cpu` feature/runtime combinations, and the design requires CubeCL-backed execution rather than a separate non-CubeCL fallback.
   - What's unclear: Which runtime is actually available in the project’s target CI/dev environments.
   - Recommendation: Keep `runtime::dispatch` generic, but choose one concrete runtime for initial bring-up and one CI-smoke path early in planning instead of leaving this implicit.

3. **Should 3c1e/3c2e work be planned as a later wave than 1e/2e/2c2e?**
   - What we know: The design’s project plan separates “CubeCL backend foundation (1e/2e/2c2e)” from “3c1e/3c2e + optimizer”, and the current manifest already shows narrower representation support for `3c1e` than for `3c2e`.
   - What's unclear: Whether Phase 2 should still be a single execution phase or split internally into two plans/waves.
   - Recommendation: Yes; keep 1e/2e/2c2e and raw compat contract stabilization ahead of 3c1e/3c2e and optimizer parity.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` - locked project constraints and Phase 2 context
- `.planning/REQUIREMENTS.md` - exact Phase 2 requirement definitions
- `.planning/ROADMAP.md` - Phase 2 goal, success criteria, and dependency order
- `.planning/STATE.md` - verified Phase 1 completion state
- `docs/design/cintx_detailed_design.md` - authoritative architecture, raw API, OOM, feature-matrix, and testing design
- `crates/cintx-runtime/src/planner.rs` - current query/evaluate contract and option-drift guard
- `crates/cintx-runtime/src/workspace.rs` - current chunking and fallible allocation behavior
- `crates/cintx-core/src/error.rs` - current public error surface
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - current canonical manifest scope
- `crates/cintx-ops/src/generated/api_manifest.rs` - generated runtime manifest table
- `crates/cintx-ops/build.rs` - current manifest generation capabilities and defaults
- `cargo metadata --no-deps --format-version 1` - active workspace membership and excluded Phase 2 crates
- `cargo test --workspace` - current automated baseline only covers Phase 1 crates
- `libcint-master/doc/program_ref.txt` - upstream layout, ordering, and raw slot contracts
- `libcint-master/include/cint_funcs.h` - canonical raw integral function signatures
- `libcint-master/include/cint.h.in` - helper, optimizer, transform, and legacy wrapper surface
- `libcint-master/src/cint1e.c` - `out == NULL`, `cache == NULL`, and `dims == NULL` 1e behavior
- `libcint-master/src/cint2e.c` - `out == NULL`, `cache == NULL`, and `dims == NULL` 2e behavior
- `libcint-master/src/cint_bas.c` - cart/sph/spinor AO count formulas and offsets
- `libcint-master/src/misc.h` - legacy wrapper macro behavior
- `/tracel-ai/cubecl` - current CubeCL runtime/client/kernel launch patterns via Context7
- https://docs.rs/crate/cubecl/latest - current stable CubeCL release line and published dates
- https://github.com/tracel-ai/cubecl - official CubeCL source repository
- https://github.com/sunqm/libcint - official upstream libcint repository and README
- https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/ - current stable Rust release
- https://docs.rs/crate/tracing/latest - current tracing release line and published dates
- https://docs.rs/crate/anyhow/latest - current anyhow release line and published dates

### Secondary (MEDIUM confidence)
- `cargo info cubecl` - stable-vs-prerelease feature details for CubeCL
- `cargo info tracing` - current stable tracing details
- `cargo info thiserror` - current stable thiserror details
- `cargo info anyhow` - current stable anyhow details
- `cargo info smallvec` - latest stable smallvec details

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified against current Cargo manifests, `cargo info`, docs.rs, and the Rust release blog.
- Architecture: HIGH - grounded in the design document, current Phase 1 code, and upstream libcint contracts.
- Pitfalls: MEDIUM - most are directly evidenced, but backend-runtime choice and wave boundaries still require explicit project decisions.

**Research date:** 2026-03-21
**Valid until:** 2026-03-28
