# Phase 6: Fix raw eval staging retrieval and capability fingerprint propagation - Research

**Researched:** 2026-04-02
**Domain:** Rust runtime data flow, compat raw path, wgpu bootstrap capability token propagation
**Confidence:** HIGH

## Project Constraints (from CLAUDE.md)

- **Compatibility target:** libcint 6.1.3 result compatibility — oracle comparisons are the correctness gate.
- **Architecture:** CubeCL is the primary compute backend; CPU stays control-plane only.
- **API Surface:** Safe Rust API first, raw compat second, optional C ABI shim third.
- **Error Handling:** `thiserror` v2 for public library errors; `anyhow` for CLI/xtask/benchmark/oracle harness.
- **Verification:** Full API coverage claims must be backed by compiled manifest lock and oracle comparison gates.
- **Artifacts:** Deliverables written to `/mnt/data` are a mandatory part of the design and verification workflow.
- **GSD Workflow:** All file changes must flow through GSD commands to keep planning artifacts in sync.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COMP-01 | Compat caller can invoke raw APIs using `atm`, `bas`, `env`, `shls`, `dims`, `opt`, and `cache` inputs that match documented layout contracts. | Fix to `eval_raw()` ensures output buffer is populated from the real executor staging, not from a zero-filled local buffer. |
| COMP-04 | C integrator can enable an optional C ABI shim that returns integer status codes and exposes thread-local last-error details. | The C ABI shim (`cintx-capi`) calls `eval_raw()` from `cintx-compat`; the shim itself is correct, but its results are zeros until COMP-01 is fixed. |
| COMP-05 | Compat caller receives typed validation failures or explicit `UnsupportedApi` errors instead of silent truncation, partial writes, or undefined behavior. | No new validation errors are needed for this phase; the existing typed error surface is complete. |
| EXEC-02 | Rust or compat caller can evaluate supported 1e, 2e, 2c2e, 3c1e, and 3c2e families through the shared planner and CubeCL backend. | Fixing `eval_raw()` wires the compat path to the real CubeCL staging output for all supported families. |
| EXEC-04 | Caller receives outputs with upstream-compatible cart, sph, and spinor shapes, ordering, and complex-layout semantics. | The representation transform (`apply_representation_transform`) already runs before staging output is captured; fix ensures the transformed staging is used, not a fresh zero-filled buffer. |
| EXEC-05 | Caller gets numerically equivalent results within accepted tolerance regardless of whether optimizer support is enabled. | Once staging retrieval is fixed, the optimizer-on/off equivalence tests can produce real numeric comparison results. |
| VERI-01 | Maintainer can compare stable and enabled optional APIs against vendored upstream libcint through oracle tests with family-appropriate tolerances. | The oracle compare path calls `eval_raw()`; zeros will always fail parity gates until staging retrieval is fixed. |
</phase_requirements>

## Summary

Phase 6 closes two concrete bugs introduced in Phase 5 that were identified in the v1.0 milestone audit. Both bugs are in the data flow between the CubeCL executor's staging buffer and the callers that need to consume produced integral values.

**Bug 1 — eval_raw() writes zeros instead of executor output.**
In `crates/cintx-compat/src/raw.rs`, `eval_raw()` calls `evaluate()` correctly (which dispatches through `CubeClExecutor` and populates a staging buffer owned by the runtime scheduler), but then discards all executor output and creates a fresh zero-filled `staging` vector (line 420-426) before writing to the caller's `out` buffer. The staging data produced by the executor lives inside the runtime scheduler's loop in `crates/cintx-runtime/src/planner.rs::evaluate()` and is dropped at end of scope — it is never returned to the caller. The safe facade (`cintx-rs`) works around this by wrapping `CubeClExecutor` in a `RecordingExecutor` that captures `io.staging_output()` values mid-execute; `eval_raw()` must adopt the same pattern.

**Bug 2 — BackendCapabilityToken fingerprint stays zero.**
`BackendCapabilityToken::default()` initialises `capability_fingerprint` to `0`. When `query_workspace()` is called (in both `cintx-runtime` and `cintx-compat`) the options are passed directly, so the token's fingerprint is whatever the caller supplies — which is `0` from the default. The wgpu bootstrap (in `crates/cintx-cubecl/src/runtime_bootstrap.rs::bootstrap_wgpu_runtime()`) computes a real FNV-1a fingerprint from adapter name, backend API, device type, vendor/device IDs, features, and limits, but the result is used only inside the executor's `preflight_wgpu()` for capability gating — it is never written back to `BackendCapabilityToken`. As a consequence, the drift detection comparison in `planning_matches()` always compares `0` against `0`, and there is no real adapter identity anchor for reproducibility (D-04).

**Primary recommendation:** Fix `eval_raw()` to use `RecordingExecutor` wrapping `CubeClExecutor`, exactly matching the safe facade pattern. Fix fingerprint propagation by calling `bootstrap_wgpu_runtime()` inside `query_workspace()` (or the compat/safe wrappers that call it) and writing the resulting fingerprint into `ExecutionOptions::backend_capability_token` before planning proceeds.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.9.0` | CubeCL compute backend, already in workspace | Locked by Phase 5; no version change needed. |
| `cubecl-wgpu` | `0.9.0` | wgpu runtime bootstrap and adapter inspection | Already used in `cintx-cubecl`; provides `bootstrap_wgpu_runtime()` via capability snapshot. |
| `thiserror` | `2.0.18` | Public error surface | Project policy for library crates. |
| `tracing` | `0.1.41` | Structured diagnostics | Already in workspace; no change. |

No new library dependencies are required. This phase is pure logic correction within existing crate boundaries.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| RecordingExecutor pattern (copy from safe facade) | Refactor runtime `evaluate()` to return staging data | Refactoring the runtime `evaluate()` signature would change a stable contract and require planner/dispatch/metrics changes across multiple crates. RecordingExecutor is already proven and self-contained. |
| Call bootstrap at query time | Call bootstrap at eval_raw call site before evaluate | query_workspace is the right hook because the capability token is stored in WorkspaceQuery and compared at evaluate time. |

## Architecture Patterns

### Recommended Fix Structure

```
crates/cintx-compat/src/raw.rs
  eval_raw()
    — add RecordingExecutor<CubeClExecutor> wrapping (mirrors cintx-rs::api::SessionQuery::evaluate)
    — replace manual zero-fill staging with owned_values from RecordingExecutor

crates/cintx-cubecl/src/lib.rs (or bootstrap module)
  — expose bootstrap_wgpu_runtime() or a fingerprint helper to cintx-compat/cintx-runtime callers

crates/cintx-runtime/src/planner.rs  (or crates/cintx-compat/src/raw.rs)
  query_workspace()
    — call bootstrap_wgpu_runtime() to obtain WgpuPreflightReport
    — populate opts.backend_capability_token.capability_fingerprint from report.fingerprint
    — populate opts.backend_capability_token.adapter_name and backend_api from report.snapshot
```

### Pattern 1: RecordingExecutor for staging capture

**What:** Wraps the real executor; intercepts `execute()` to copy `io.staging_output()` values into a `Mutex<Vec<f64>>` after the inner executor runs.
**When to use:** Any site that calls `evaluate()` and needs to retrieve the produced staging values.
**Existing implementation:** `crates/cintx-rs/src/api.rs` lines 374-423 (verbatim reuse possible, or move to a shared location).

```rust
// Source: crates/cintx-rs/src/api.rs RecordingExecutor
impl<E: BackendExecutor> BackendExecutor for RecordingExecutor<E> {
    fn execute(&self, plan: &ExecutionPlan<'_>, io: &mut ExecutionIo<'_>)
        -> Result<ExecutionStats, cintxRsError>
    {
        let stats = self.inner.execute(plan, io)?;
        let mut staged_values = self.staged_values.lock().unwrap_or_else(|p| p.into_inner());
        staged_values.extend_from_slice(io.staging_output());
        Ok(stats)
    }
}
```

**Placement decision:** Either:
- (a) Move `RecordingExecutor` to `cintx-runtime` as a public utility (cleanest, avoids duplication), OR
- (b) Duplicate the pattern locally in `cintx-compat/src/raw.rs` (minimal change footprint).

Option (b) is lower risk for Phase 6 scope. Option (a) is preferable if a later phase will need it elsewhere.

### Pattern 2: Fingerprint propagation at query time

**What:** Before planning proceeds, call the bootstrap to get the real adapter fingerprint and write it into the capability token.
**When to use:** At `query_workspace()` in the compat raw path (and optionally the runtime planner path when wgpu intent is selected).

```rust
// Source: crates/cintx-cubecl/src/runtime_bootstrap.rs + options.rs
// Inside execution_options_from_opt() or prepare_raw_call():
let report = bootstrap_wgpu_runtime(&options.backend_intent)?;
options.backend_capability_token = BackendCapabilityToken {
    adapter_name: report.snapshot.adapter_name.clone(),
    backend_api: report.snapshot.backend_api.clone(),
    capability_fingerprint: report.fingerprint,
};
```

**Placement note:** `cintx-compat` already depends on `cintx-cubecl` (CubeClExecutor is imported in `raw.rs`). `bootstrap_wgpu_runtime` can be imported from `cintx_cubecl::runtime_bootstrap`. No new crate dependency needed.

**Fail-closed behaviour:** If `bootstrap_wgpu_runtime()` returns an error (no adapter), it means the entire call will fail at execute time anyway. The fingerprint call at query time can therefore propagate the same `UnsupportedApi` error, which is consistent with D-01/D-02 and surfaces the failure early rather than late.

### Anti-Patterns to Avoid

- **Manual zero-fill after evaluate:** `eval_raw()` must not allocate a new `Vec` after `evaluate()` and write zeros; the staging data flows through `io.staging_output()` inside the execute call and must be captured at that point.
- **Fingerprint computed post-evaluate:** If the fingerprint is populated only after `evaluate()` completes, the drift check in `planning_matches()` for that same call is meaningless (query and evaluate use the same options object). The fingerprint must be set before `query_workspace()` so the stored `WorkspaceQuery::backend_capability_token` reflects the real adapter.
- **Duplicate bootstrap calls in the hot path:** The `bootstrap_wgpu_runtime()` for `AdapterSelector::Auto` already caches via `OnceLock`; repeated calls are safe and cheap.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Capturing staging values from executor | New buffer passing mechanism through runtime evaluate() | RecordingExecutor (already in cintx-rs/src/api.rs) | Existing pattern is tested, avoids runtime contract changes, follows the ownership model. |
| Computing adapter fingerprint | Ad-hoc string hashing at call sites | `capability_fingerprint()` in `cintx-cubecl::capability` | FNV-1a implementation already exists; reuse avoids drift in hash semantics. |
| Adapter selection and setup | Direct wgpu calls | `bootstrap_wgpu_runtime()` in `cintx-cubecl::runtime_bootstrap` | Already handles selector parsing, panic catch_unwind, OnceLock caching, and typed error mapping. |

## Common Pitfalls

### Pitfall 1: eval_raw() staging allocated after evaluate() returns
**What goes wrong:** Calling `evaluate()` and then creating a fresh zero-filled `Vec` ignores the real output.
**Why it happens:** The runtime scheduler allocates staging per-chunk internally and drops it after each chunk loop iteration; nothing returns it to `eval_raw()`.
**How to avoid:** Wrap `CubeClExecutor` in `RecordingExecutor` before passing to `evaluate()`. Retrieve `owned_values` after `evaluate()` returns.
**Warning signs:** Output buffer contains all zeros after a successful `eval_raw()` call. The `not0` field in `RawEvalSummary` returns a non-zero value from stats but the output buffer is zero.

### Pitfall 2: Fingerprint propagation does not cover the compat raw path
**What goes wrong:** `execution_options_from_opt()` only sets `profile_label` and `memory_limit_bytes`; `backend_capability_token` stays at default (all-zero fingerprint).
**Why it happens:** Phase 5 wired the token contract in the runtime planner and safe facade but did not update `execution_options_from_opt()` in `cintx-compat`.
**How to avoid:** Populate `backend_capability_token` in `execution_options_from_opt()` or at the top of `prepare_raw_call()` by calling `bootstrap_wgpu_runtime()`.
**Warning signs:** `WorkspaceQuery::backend_capability_token.capability_fingerprint == 0` in all compat raw paths. The drift detection check in `planning_matches()` always passes vacuously (0 == 0).

### Pitfall 3: bootstrap called after query_workspace has already stored the zero fingerprint
**What goes wrong:** Fingerprint is populated on `ExecutionOptions` after `query_workspace()` is called; the `WorkspaceQuery` already has `capability_fingerprint: 0` copied from opts.
**Why it happens:** `query_workspace()` copies `opts.backend_capability_token` at the time it is called (line 126 in `crates/cintx-runtime/src/planner.rs`).
**How to avoid:** Populate `backend_capability_token` on `options` before calling `query_workspace()`, not after.
**Warning signs:** Token in `WorkspaceQuery` has fingerprint 0 even though `ExecutionOptions` has a non-zero fingerprint.

### Pitfall 4: RecordingExecutor placed after the unsafe eval_raw preflight
**What goes wrong:** If `RecordingExecutor` is added but the planning still uses the old zero-fill path, the final write will still be zeros.
**Why it happens:** The `prepared.compat_dims.write(out, &staging)` call uses the old zero-filled local `staging`, not `owned_values` from the executor.
**How to avoid:** Replace `staging` completely with `owned_values` from `RecordingExecutor`. Remove the manual zero-fill `Vec` and `staging.resize(…, 0.0)` lines entirely.
**Warning signs:** Test shows non-zero `owned_values.len()` but `out` still contains zeros.

### Pitfall 5: staging element count mismatch for multi-chunk evaluations
**What goes wrong:** `RecordingExecutor` appends staging from each chunk, so total `owned_values.len()` may exceed `required_elements` when multiple chunks are used.
**Why it happens:** `required_elements` is the total output size but chunks may interleave differently.
**How to avoid:** Check that `owned_values.len() == required_elements` before writing; use the same guard that the safe facade uses (`output_layout.staging_elements`).
**Warning signs:** `write()` panics or returns an incorrect `bytes_written` count.

## Code Examples

Verified patterns from codebase:

### eval_raw() fix — use RecordingExecutor (mirrors safe facade)

```rust
// Source: crates/cintx-rs/src/api.rs lines 139-142, 421
// Replace in crates/cintx-compat/src/raw.rs eval_raw():
let executor = RecordingExecutor::new(CubeClExecutor::new());
let mut allocator = HostWorkspaceAllocator::default();
let stats = evaluate(plan, &prepared.options, &mut allocator, &executor)?;
let owned_values = executor.owned_values()?;  // captures staging from all chunks

// Replace zero-fill section:
// REMOVE: let mut staging = Vec::new(); staging.resize(required_elements, 0.0);
// USE:
if owned_values.len() != required_elements {
    return Err(cintxRsError::ChunkPlanFailed {
        from: "eval_raw",
        detail: format!(
            "staging output length mismatch: expected={required_elements} got={}",
            owned_values.len()
        ),
    });
}
let out = out.expect("checked out.is_some()");
let written_elements = prepared.compat_dims.write(out, &owned_values)?;
```

### Fingerprint propagation — populate before query_workspace

```rust
// Source: crates/cintx-cubecl/src/runtime_bootstrap.rs bootstrap_wgpu_runtime()
// and crates/cintx-runtime/src/options.rs BackendCapabilityToken
// Add to execution_options_from_opt() in crates/cintx-compat/src/raw.rs:
fn execution_options_from_opt(opt: Option<&RawOptimizerHandle>) -> Result<ExecutionOptions, cintxRsError> {
    let mut options = ExecutionOptions::default();
    options.profile_label = Some(active_manifest_profile());
    if let Some(opt) = opt {
        options.memory_limit_bytes = opt.workspace_hint_bytes();
    }
    // Propagate real wgpu adapter fingerprint so planning_matches() drift check has a real anchor.
    let report = bootstrap_wgpu_runtime(&options.backend_intent)?;
    if report.is_capable() {
        options.backend_capability_token = BackendCapabilityToken {
            adapter_name: report.snapshot.adapter_name.clone(),
            backend_api: report.snapshot.backend_api.clone(),
            capability_fingerprint: report.fingerprint,
        };
    }
    Ok(options)
}
```

Note: `execution_options_from_opt()` currently returns `ExecutionOptions` (not `Result`). The signature must change to `Result<ExecutionOptions, cintxRsError>` and call sites in `prepare_raw_call()` must propagate the error with `?`.

### RecordingExecutor location — reuse or relocate

The `RecordingExecutor` struct is currently private to `cintx-rs/src/api.rs`. For Phase 6 scope, two options:

**(a) Duplicate locally in `cintx-compat/src/raw.rs`** — minimal change, no cross-crate refactor.
```rust
// Add near top of raw.rs (same structure as cintx-rs/src/api.rs):
use std::sync::Mutex;
struct RecordingExecutor { inner: CubeClExecutor, staged_values: Mutex<Vec<f64>> }
// ... impl BackendExecutor ...
```

**(b) Move to `cintx-runtime` as a public utility** — cleaner long-term, eliminates duplication.
```rust
// crates/cintx-runtime/src/recording.rs (new file)
pub struct RecordingExecutor<E> { ... }
```

The planner should recommend option (b) if the pattern is already used in two places, or option (a) if minimising Phase 6 surface is the priority.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (nextest available) |
| Config file | none (workspace default) |
| Quick run command | `cargo test -p cintx-compat -- raw::tests` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COMP-01 | eval_raw() writes real executor output (non-zero for 1e family on GPU, wgpu-capability error on no-GPU) | unit | `cargo test -p cintx-compat -- eval_raw_writes_executor_output` | ❌ Wave 0 |
| COMP-04 | C ABI shim result via eval_raw reflects real executor output | integration | `cargo test -p cintx-capi -- shim::tests::query_and_eval_wrappers_succeed_and_clear_tls_error` | ✅ (existing) |
| COMP-05 | eval_raw returns typed error on invalid inputs; no partial write | unit | `cargo test -p cintx-compat -- raw::tests` | ✅ (existing) |
| EXEC-02 | eval_raw executes 1e, 2e, 2c2e, 3c1e, 3c2e families | unit | `cargo test -p cintx-compat -- eval_raw_all_base_families` | ❌ Wave 0 |
| EXEC-04 | cart/sph/spinor shapes in eval_raw output match layout contract | unit | `cargo test -p cintx-compat -- eval_raw_representation_layouts` | ❌ Wave 0 |
| EXEC-05 | optimizer on/off produces equivalent results for eval_raw | unit | `cargo test -p cintx-compat -- eval_raw_optimizer_on_off_equivalence` | ❌ Wave 0 |
| VERI-01 | Oracle compare path gets real numerics from eval_raw | integration | `cargo test -p cintx-oracle` | ✅ (existing — will now get real data) |

Additional regression tests needed:
- `eval_raw_output_is_not_all_zeros`: assert at least one non-zero element when GPU available (mirrors D-15 in safe facade).
- `query_workspace_raw_fingerprint_is_nonzero_when_gpu_available`: assert `WorkspaceQuery::backend_capability_token.capability_fingerprint != 0` after `query_workspace_raw()` on GPU environments.
- `fingerprint_matches_between_query_and_evaluate`: assert that `planning_matches()` returns `true` when same options used.

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-compat -- raw::tests`
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_writes_executor_output` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_all_base_families` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_output_is_not_all_zeros` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `query_workspace_raw_fingerprint_is_nonzero_when_gpu_available` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_representation_layouts` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_optimizer_on_off_equivalence` test

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `eval_raw()` creates fresh zero-filled staging after evaluate() | Must use RecordingExecutor to capture real staging from io.staging_output() | Phase 6 target | Without fix: all oracle comparisons return wrong results; COMP-01/VERI-01 are blocked. |
| `BackendCapabilityToken.capability_fingerprint` defaults to 0 | Must be populated from wgpu bootstrap before query_workspace() | Phase 6 target | Without fix: drift detection in planning_matches() is vacuous (0 == 0); D-04 reproducibility is not satisfied. |
| `execution_options_from_opt()` returns `ExecutionOptions` | Must return `Result<ExecutionOptions, cintxRsError>` to propagate bootstrap failure | Phase 6 target | Signature change is necessary; all callers in `prepare_raw_call()` must propagate the error. |

**Deprecated/outdated:**
- Manual zero-fill staging construction in `eval_raw()` (lines 420-426 of `cintx-compat/src/raw.rs`).
- `capability_fingerprint: 0` as the effective fingerprint for any real wgpu execution path.

## Open Questions

1. **Where should RecordingExecutor live?**
   - What we know: It exists only in `cintx-rs/src/api.rs` (private). `eval_raw()` in `cintx-compat` needs the same pattern.
   - What's unclear: Whether Phase 6 scope justifies the refactor to move it to `cintx-runtime`, or whether local duplication is the right call.
   - Recommendation: Local duplication in `cintx-compat/src/raw.rs` for Phase 6 scope; create a follow-up note in `deferred_ideas` for the shared-utility refactor.

2. **Should fingerprint propagation also be applied to the safe facade (cintx-rs)?**
   - What we know: The safe facade also calls `query_workspace()` with `BackendCapabilityToken::default()` (fingerprint 0), so the same issue applies.
   - What's unclear: Whether VERI-01 and COMP-01/EXEC-02 are the primary targets and the safe facade path can wait.
   - Recommendation: Fix fingerprint propagation in both the compat raw path AND the safe facade path in the same phase, since the fix site is small (`execution_options_from_opt()` and `SessionRequest::query_workspace()`).

3. **What does eval_raw return when the bootstrap fails on a no-GPU environment?**
   - What we know: `bootstrap_wgpu_runtime()` returns `UnsupportedApi { requested: "wgpu-capability:missing_adapter" }` on no-GPU hosts.
   - What's unclear: Whether `eval_raw()` should propagate this error immediately at query time, or remain consistent with the current fail-closed behavior where it surfaces at execute time.
   - Recommendation: Propagate early (at `execution_options_from_opt()` / `prepare_raw_call()`). This is consistent with the executor's `preflight_wgpu()` and gives callers the typed error without allocating or planning unnecessarily.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust 1.94.0 toolchain | All crates | ✓ | pinned in rust-toolchain.toml | — |
| cargo | Build | ✓ | from toolchain | — |
| wgpu adapter | Fingerprint + staging tests | environment-dependent | — | Tests accept `wgpu-capability:missing_adapter` error as valid fail-closed outcome |
| cubecl 0.9.0 | CubeCL backend | ✓ | locked in Cargo.lock | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- wgpu GPU adapter — tests must accept both the GPU-success path and the `wgpu-capability:missing_adapter` fail-closed path (matching existing test patterns in executor.rs and raw.rs).

## Sources

### Primary (HIGH confidence)
- Local codebase (direct read, line-level evidence):
  - `crates/cintx-compat/src/raw.rs` — `eval_raw()` bug at lines 420-441; `execution_options_from_opt()` at lines 698-705; `prepare_raw_call()` at lines 576-641
  - `crates/cintx-rs/src/api.rs` — `RecordingExecutor` pattern at lines 374-423; `SessionQuery::evaluate()` at lines 100-175
  - `crates/cintx-cubecl/src/executor.rs` — `CubeClExecutor::execute()` confirms staging is written to `io.staging_output()` before return
  - `crates/cintx-cubecl/src/runtime_bootstrap.rs` — `bootstrap_wgpu_runtime()` returns `WgpuPreflightReport` with `fingerprint` field
  - `crates/cintx-cubecl/src/capability.rs` — `WgpuPreflightReport::fingerprint` populated by `capability_fingerprint(snapshot)` FNV-1a
  - `crates/cintx-runtime/src/options.rs` — `BackendCapabilityToken::default()` initialises `capability_fingerprint: 0`
  - `crates/cintx-runtime/src/planner.rs` — `query_workspace()` copies `opts.backend_capability_token` verbatim (line 126); `evaluate()` dispatcher at lines 130-216
  - `crates/cintx-runtime/src/dispatch.rs` — `ExecutionIo::staging_output()` exposes staging slice to executor
  - `.planning/REQUIREMENTS.md` — COMP-01, COMP-04, COMP-05, EXEC-02, EXEC-04, EXEC-05, VERI-01 traceability
  - `.planning/STATE.md` — Phase 5 decisions D-04, D-06, D-08 locked
  - `.planning/ROADMAP.md` — Phase 6 gap closure description

### Secondary (MEDIUM confidence)
- CLAUDE.md technology stack table — confirms `thiserror v2`, `anyhow 1.0.102`, `cubecl 0.9.0` as standard.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Bug identification: HIGH — bugs identified by direct code reading with line-level evidence.
- Fix approach: HIGH — RecordingExecutor pattern is already implemented and tested in the safe facade; fingerprint propagation follows the existing bootstrap API contract.
- Test gaps: HIGH — gaps are determinable from requirements traceability and existing test file inspection.

**Research date:** 2026-04-02
**Valid until:** 2026-04-16
