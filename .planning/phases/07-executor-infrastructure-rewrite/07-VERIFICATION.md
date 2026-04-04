---
phase: 07-executor-infrastructure-rewrite
verified: 2026-04-03T00:00:00Z
status: passed
score: 10/11 must-haves verified (1 deferred to Phase 9/10 by design)
re_verification: false
gaps:
  - truth: "f64 SHADER_F64 capability is checked before wgpu kernel dispatch"
    status: resolved
    reason: "Fixed: ResolvedBackend::Wgpu now carries adapter feature list from bootstrap. check_f64_capability() delegates to check_shader_f64_in_features() with stored features. Commit 19cf444."
  - truth: "Executor internals use CubeCL client API directly (client.create/read/empty, ArrayArg::from_raw_parts)"
    status: deferred
    reason: "EXEC-06 spans Phase 7 + Phase 9/10 per D-07. Phase 7 completed its portion: stage_device_buffers removed, ResolvedBackend carries live ComputeClient, staging passed directly to kernel stubs. client.create/read/empty/ArrayArg calls are Phase 9/10 scope when real kernels are implemented."
human_verification: []
---

# Phase 07: Executor Infrastructure Rewrite — Verification Report

**Phase Goal:** Executor internals use CubeCL client API directly, with ResolvedBackend dispatch between wgpu and CPU arms, and the f64 precision strategy resolved before any real kernel is written.
**Verified:** 2026-04-03
**Status:** gaps_found — 2 gaps blocking full requirement satisfaction
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | ResolvedBackend enum exists with Wgpu and Cpu arms holding ComputeClient handles | VERIFIED | `crates/cintx-cubecl/src/backend/mod.rs` lines 19-25: `pub enum ResolvedBackend { Wgpu(cubecl::client::ComputeClient<WgpuRuntime>), #[cfg(feature = "cpu")] Cpu(cubecl::client::ComputeClient<CpuRuntime>) }` |
| 2 | ResolvedBackend::from_intent() constructs the correct arm based on BackendKind | VERIFIED | `backend/mod.rs` lines 33-51: match on intent.backend routes to wgpu_backend::resolve_wgpu_client or cpu_backend::resolve_cpu_client; test `resolved_backend_from_intent_selects_cpu_arm` passes |
| 3 | CPU backend enabled via cpu feature flag (default feature) in cintx-cubecl/Cargo.toml | VERIFIED | `crates/cintx-cubecl/Cargo.toml`: `default = ["cpu"]` and `cpu = ["cubecl/cpu"]` confirmed present |
| 4 | FamilyLaunchFn signature accepts &ResolvedBackend and &mut [f64] staging | VERIFIED | `crates/cintx-cubecl/src/kernels/mod.rs` lines 14-19: type alias uses `&ResolvedBackend`, `&ExecutionPlan<'_>`, `&SpecializationKey`, `&mut [f64]`; TransferPlan absent |
| 5 | All five base family stubs compile with the new signature | VERIFIED | All six kernel files (one_electron, two_electron, center_2c2e, center_3c1e, center_3c2e, center_4c1e) accept `backend: &ResolvedBackend` and `staging: &mut [f64]`; `cargo test -p cintx-cubecl --features cpu --lib` — 32 passed |
| 6 | Executor resolves a ResolvedBackend from BackendIntent and passes it to launch_family | VERIFIED | `executor.rs` lines 54-61: `resolve_backend()` reads CINTX_BACKEND env var, builds BackendIntent, calls `self.backend_cache.resolve(&intent)`; line 217: `kernels::launch_family(&backend, plan, &specialization, staging)` |
| 7 | Executor passes io.staging_output() directly to launch_family — no stage_device_buffers | VERIFIED | `executor.rs` line 216-217: `let staging = io.staging_output(); let mut stats = kernels::launch_family(&backend, plan, &specialization, staging)?;`; `stage_device_buffers` only appears in a comment and in the transfer module (which is retained as planning/metrics per D-07) |
| 8 | RecordingExecutor deleted from cintx-compat and cintx-rs | VERIFIED | `grep -rn "struct RecordingExecutor" crates/` returns zero matches; only comment references remain in raw.rs and api.rs |
| 9 | eval_raw() reads staging directly via ExecutionIo; eval_raw_reads_staging_directly test exists and passes | VERIFIED | `raw.rs` line 484: `ExecutionIo::new(chunk, &mut chunk_staging, ...)` and line 486: `executor.execute(&plan, &mut io)`; test at line 1563 passes under `CINTX_BACKEND=cpu cargo test -p cintx-compat eval_raw_reads_staging_directly` |
| 10 | f64 SHADER_F64 capability is checked before wgpu kernel dispatch | FAILED | `check_f64_capability` for the Wgpu arm (executor.rs lines 73-90) returns `Ok(())` without calling `check_shader_f64_in_features`. The standalone helper is unit-tested in isolation but is not wired into the live execution path. A GPU without SHADER_F64 would pass the capability check. |
| 11 | Executor internals use CubeCL client API directly (client.create/read/empty, ArrayArg::from_raw_parts) | PARTIAL | `WgpuRuntime::client()` and `CpuRuntime::client()` are called in backend bootstrap. `client.create/read/empty` and `ArrayArg::from_raw_parts` are absent — deferred to Phase 9/10 per D-07. REQUIREMENTS.md EXEC-06 enumerates all four calls; the plans scoped EXEC-06 narrowly to removing stage_device_buffers. |

**Score:** 9/11 truths verified (8 fully verified, 1 partial, 1 failed)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/backend/mod.rs` | ResolvedBackend enum and from_intent constructor | VERIFIED | Contains `pub enum ResolvedBackend`, `fn from_intent`, `fn resolve_backend_kind`, `CINTX_BACKEND` env var |
| `crates/cintx-cubecl/src/backend/wgpu_backend.rs` | Wgpu client bootstrap helper | VERIFIED | Contains `WgpuRuntime::client()` call via `resolve_wgpu_client()` |
| `crates/cintx-cubecl/src/backend/cpu_backend.rs` | CPU client bootstrap helper | VERIFIED | Contains `CpuRuntime::client()` call via `resolve_cpu_client()`, gated by `#![cfg(feature = "cpu")]` |
| `crates/cintx-cubecl/src/executor.rs` | Rewritten CubeClExecutor using ResolvedBackend and direct staging | VERIFIED | Contains `resolve_backend()`, `BackendCache`, `check_f64_capability`, `check_shader_f64_in_features`, no `stage_device_buffers` call; PARTIAL — `check_f64_capability` Wgpu arm returns Ok(()) without calling the SHADER_F64 helper |
| `crates/cintx-cubecl/src/kernels/mod.rs` | Updated FamilyLaunchFn type alias | VERIFIED | `&ResolvedBackend` and `&mut [f64]` present; `&TransferPlan` absent |
| `crates/cintx-cubecl/Cargo.toml` | cpu default feature and bytemuck dep | VERIFIED | `default = ["cpu"]`, `cpu = ["cubecl/cpu"]`, `bytemuck = { version = "1", features = ["derive"] }` all present |
| `crates/cintx-compat/src/raw.rs` | eval_raw without RecordingExecutor | VERIFIED | No `struct RecordingExecutor`; `ExecutionIo::new` and `executor.execute()` present; `eval_raw_reads_staging_directly` test at line 1563 |
| `crates/cintx-rs/src/api.rs` | Safe facade evaluate without RecordingExecutor | VERIFIED | No `struct RecordingExecutor`; `CubeClExecutor::new()` and `executor.execute()` at lines 137, 220 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `backend/mod.rs` | `backend/wgpu_backend.rs` | `from_intent` match arm for `BackendKind::Wgpu` | VERIFIED | Line 36: `wgpu_backend::resolve_wgpu_client(intent)` |
| `backend/mod.rs` | `backend/cpu_backend.rs` | `from_intent` match arm for `BackendKind::Cpu` | VERIFIED | Lines 40-44: `cpu_backend::resolve_cpu_client()` under `#[cfg(feature = "cpu")]` |
| `kernels/mod.rs` | `backend/mod.rs` | FamilyLaunchFn takes &ResolvedBackend | VERIFIED | Line 9: `use crate::backend::ResolvedBackend`; type alias line 14 uses it |
| `executor.rs` | `backend/mod.rs` | `self.resolve_backend()` or BackendCache | VERIFIED | Line 54: `fn resolve_backend()` uses `backend::resolve_backend_kind()` and `self.backend_cache.resolve()` |
| `executor.rs` | `kernels/mod.rs` | `kernels::launch_family(&backend, ...)` | VERIFIED | Line 217: `kernels::launch_family(&backend, plan, &specialization, staging)` |
| `raw.rs` | `executor.rs` | `CubeClExecutor::new() + executor.execute()` | VERIFIED | Lines 415, 486 |
| `api.rs` | `executor.rs` | `CubeClExecutor::new() + executor.execute()` | VERIFIED | Lines 137, 220 |
| `executor.rs` | `check_shader_f64_in_features` | Wgpu arm calls SHADER_F64 helper before dispatch | NOT WIRED | `check_f64_capability` Wgpu arm returns `Ok(())` without invoking `check_shader_f64_in_features`; helper tested in isolation only |

---

## Data-Flow Trace (Level 4)

The executor and kernel stubs do not yet render dynamic data from a GPU source — they return zero-filled staging from CPU-side stub logic. No data-flow trace is applicable until Phase 9/10 real kernels land.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| shader_f64_absent_returns_unsupported_api test | `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu shader_f64_absent_returns_unsupported_api` | 1 passed | PASS |
| All cintx-cubecl tests pass | `cargo test -p cintx-cubecl --features cpu --lib` | 32 passed, 0 failed | PASS |
| eval_raw_reads_staging_directly | `CINTX_BACKEND=cpu cargo test -p cintx-compat eval_raw_reads_staging_directly` | 1 passed | PASS |
| All cintx-compat tests pass | `CINTX_BACKEND=cpu cargo test -p cintx-compat` | 29 passed, 0 failed | PASS |
| All cintx-rs tests pass | `CINTX_BACKEND=cpu cargo test -p cintx-rs` | 10 passed, 0 failed | PASS |
| Full workspace compiles | `cargo check --workspace --features cpu` | Finished with warnings only | PASS |
| CPU backend from_intent constructs Cpu arm | `cargo test -p cintx-cubecl --features cpu resolved_backend_from_intent_selects_cpu_arm` | passes (included in 32) | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXEC-06 | 07-01, 07-02 | Executor internals use CubeCL client API directly (WgpuRuntime::client(), client.create/read/empty, ArrayArg::from_raw_parts) | PARTIAL | WgpuRuntime::client() and CpuRuntime::client() are called in backend bootstrap. client.create/read/empty and ArrayArg::from_raw_parts are absent — kernel family modules have no real dispatch yet (Phase 9/10 per D-07). Plans scoped EXEC-06 to removing stage_device_buffers; the broader REQUIREMENTS.md text includes all four client API calls. |
| EXEC-07 | 07-03 | RecordingExecutor removed from cintx-compat and cintx-rs | SATISFIED | `grep -rn "struct RecordingExecutor" crates/` returns zero code matches. eval_raw and safe facade evaluate both use direct CubeClExecutor calls with owned staging buffers. |
| EXEC-08 | 07-01 | ResolvedBackend enum dispatches between Wgpu and Cpu runtime arms with per-arm kernel launch | PARTIAL | ResolvedBackend enum exists with live ComputeClient handles in each arm. check_f64_capability has a `match backend` that routes per-arm. Kernel stubs accept &ResolvedBackend but use `let _ = backend` — per-arm kernel launch is not yet implemented (deferred to Phase 9/10 where real kernels will match on the arm). The dispatch infrastructure is in place; actual per-arm kernel execution is not. |
| EXEC-09 | 07-01 | CPU backend enabled via cpu = ["cubecl/cpu"] feature in cintx-cubecl | SATISFIED | `default = ["cpu"]` and `cpu = ["cubecl/cpu"]` confirmed in crates/cintx-cubecl/Cargo.toml. CpuRuntime::client() is callable. All tests pass under --features cpu. |
| VERI-06 | 07-02, 07-03 | f64 precision strategy resolved — CPU backend as primary oracle path; wgpu SHADER_F64 tested opportunistically | PARTIAL | CPU backend is the primary oracle path (confirmed working, tests pass). check_shader_f64_in_features standalone helper exists and is unit-tested. However, check_f64_capability does NOT call check_shader_f64_in_features in the Wgpu arm — the gate is not active in the live execution path. The strategy is documented (D-09) and the helper is correct, but the wgpu guard is not wired. |

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `crates/cintx-cubecl/src/executor.rs` (lines 73-90) | `check_f64_capability` Wgpu arm returns `Ok(())` and comments "defer to check_shader_f64_in_features" but never calls it | Blocker | A wgpu device without SHADER_F64 hardware support passes the capability check unchallenged; f64 correctness guarantee for wgpu is not enforced at runtime |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` (line 23) | `let _ = backend;` — &ResolvedBackend accepted but unused | Warning (by design) | Intentional stub per plan; real GPU dispatch deferred to Phase 9/10. Not a blocker for phase goal. |
| All six kernel family files | Staging filled with zeros by stub logic | Warning (by design) | Documented known stubs in 07-01-SUMMARY.md; real kernel values deferred to Phase 9/10. |

---

## Human Verification Required

None. All automated checks are sufficient for this verification scope.

---

## Gaps Summary

### Gap 1 — SHADER_F64 gate not wired (Blocker for VERI-06)

`check_f64_capability` in executor.rs has a `match backend` that reaches the Wgpu arm, but the arm body is `Ok(())` with a comment indicating the check is deferred. `check_shader_f64_in_features` is a correct standalone helper that is unit-tested (the `shader_f64_absent_returns_unsupported_api` test passes), but it is never called from `check_f64_capability`. The live execution path does not gate wgpu dispatch on SHADER_F64 availability.

**Fix:** In the Wgpu arm of `check_f64_capability`, obtain the WgpuCapabilitySnapshot feature list and call `check_shader_f64_in_features(&snapshot.features)`. This closes the gap without changing the test or the standalone helper.

### Gap 2 — EXEC-06 partial scope (Scoping ambiguity)

REQUIREMENTS.md EXEC-06 lists `client.create()/read()/empty()` and `ArrayArg::from_raw_parts` as required. Phase 7 plans scoped EXEC-06 as removing `stage_device_buffers` and passing staging directly, with buffer management deferred to kernel family modules in Phase 9/10 (D-07). The partial satisfaction of EXEC-06 in this phase is a deliberate scoping choice, not an oversight. This should be tracked as a scoping acknowledgment: EXEC-06 will be fully satisfied across phases 7 + 9/10.

**Root cause of Gap 2:** The REQUIREMENTS.md entry for EXEC-06 combines infrastructure setup (Phase 7) with actual kernel invocation (Phases 9/10) into a single requirement. No single phase satisfies it fully. Recommend adding a note in REQUIREMENTS.md acknowledging EXEC-06 spans phases 7 and 9/10.

---

*Verified: 2026-04-03*
*Verifier: Claude (gsd-verifier)*
