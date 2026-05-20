---
phase: 20-precision-generic-f64-f32-switch
plan: 06
subsystem: executor-capability-precision-plumbing
tags: [precision, check_capability, SHADER_F64, PrecisionKind, ExecutionOptions, staging, wave5, PREC-06, PREC-03]
dependency_graph:
  requires: [20-01 (PrecisionKind + ExecutionPlan.precision field), 20-04 (kernel launchers), 20-05 (kernel launchers group B)]
  provides: [check_capability wrapper bypassing SHADER_F64 for f32, ExecutionOptions.precision field, precision threading evaluate -> ExecutionPlan, staging soundness confirmation]
  affects: [cintx-cubecl executor.rs, cintx-runtime options.rs + planner.rs]
tech_stack:
  added: []
  patterns: [PrecisionKind enum dispatch (object-safety preserve), caller-populates-after-new (f12_zeta precedent), frozen fallible-alloc OOM contract]
key_files:
  created: []
  modified:
    - crates/cintx-cubecl/src/executor.rs (check_capability + import PrecisionKind + updated call sites)
    - crates/cintx-runtime/src/options.rs (precision: PrecisionKind field on ExecutionOptions)
    - crates/cintx-runtime/src/planner.rs (plan.precision = opts.precision in evaluate + RED/GREEN tests)
decisions:
  - "check_capability branches on plan.precision: F32 returns Ok early (D-10 / PREC-06); F64 delegates byte-identically to check_f64_capability (PREC-04)"
  - "check_f64_capability and check_shader_f64_in_features bodies kept FROZEN (no body changes)"
  - "precision threaded via evaluate(mut plan, opts) at the top of evaluate — follows f12_zeta caller-populates-after-new precedent"
  - "ExecutionOptions.precision: PrecisionKind defaults to F64 via derive(Default); existing callers unchanged"
  - "try_alloc_staging remains Vec<f64> sized by elements — f64 buffer over-allocates for f32 (elements*2 f32 lanes >= elements needed; A5 proven in Plan 01)"
  - "staging OOM-safe fallible-alloc contract unchanged (try_reserve_exact + HostAllocationFailed, no partial writes)"
metrics:
  duration: "~9 min"
  completed: "2026-05-20"
  tasks_completed: 2
  files_changed: 3
---

# Phase 20 Plan 06: Executor Capability Branch + Precision Plumbing Summary

Wired the f32 SHADER_F64 bypass in the executor capability gate and threaded `precision: PrecisionKind` from `ExecutionOptions` through `evaluate` into `ExecutionPlan.precision`, completing the Wave 3 plumbing so kernel dispatchers from Plans 04/05 actually receive the requested precision.

## Tasks Completed

| Task | Name | Commit (test) | Commit (impl) | Key Files |
|------|------|---------------|---------------|-----------|
| 1 | Precision-aware check_capability bypassing SHADER_F64 for f32 | 15ad514 (RED) | 44d9ee2 (GREEN) | `crates/cintx-cubecl/src/executor.rs` |
| 2 | Thread precision from ExecutionOptions into ExecutionPlan + confirm staging soundness | 0effeea (RED) | 47b4825 (GREEN) | `crates/cintx-runtime/src/options.rs`, `crates/cintx-runtime/src/planner.rs` |

## Task 1: check_capability (executor.rs)

### What Was Changed

Added `fn check_capability` to `CubeClExecutor` as a precision-aware wrapper over the existing frozen `check_f64_capability`. The new method:

- Returns `Ok(())` immediately when `plan.precision == PrecisionKind::F32` (f32 is WebGPU-baseline universal — no SHADER_F64 gate required, D-10)
- Delegates byte-identically to `check_f64_capability` for `PrecisionKind::F64`, preserving the existing SHADER_F64 requirement on wgpu/metal (PREC-04)

Updated both call sites in `query_workspace` and `execute` from `check_f64_capability` to `check_capability`.

**FROZEN** (zero body changes):
- `check_f64_capability`: unchanged
- `check_shader_f64_in_features`: unchanged (frozen function + unit tests pass unchanged)

### Unit Tests Added (TDD)

- `check_capability_f32_bypasses_shader_f64_gate` — F32 returns Ok on CPU backend via direct `check_capability` call
- `check_capability_f64_delegates_to_check_f64_capability` — F64 on CPU returns Ok; `check_shader_f64_in_features` still fails closed for wgpu/metal

### Acceptance Criteria Met

| Criterion | Status |
|-----------|--------|
| `fn check_capability` with early `Ok(())` for F32 | PASS — `grep -n "PrecisionKind::F32" executor.rs` matches line 114 |
| `check_shader_f64_in_features` body unchanged | PASS — frozen, unit tests pass |
| `grep -c "check_capability" executor.rs >= 3` (def + 2 call sites) | PASS — 3 non-test occurrences (lines 109, 206, 217) |
| New unit tests prove F32 bypass + F64 fail-closed | PASS — T06-1a, T06-1b |
| `cargo check -p cintx-cubecl --features cpu` exits 0 | PASS |

## Task 2: Precision Threading (options.rs + planner.rs)

### What Was Changed

**PART A (ExecutionOptions):** Added `pub precision: PrecisionKind` field to `ExecutionOptions` struct in `options.rs`. Imported `PrecisionKind` from `cintx_core`. Because `PrecisionKind` derives `Default` as `F64`, the `#[derive(Default)]` on `ExecutionOptions` keeps the f64 path unchanged — all existing call sites compile without modification.

**PART B (thread into plan):** In `evaluate` (planner.rs), changed the `plan` parameter to `mut plan` and added `plan.precision = opts.precision` at the top — before any tracing spans or validation. This follows the f12_zeta "caller populates after new" precedent. The two entry points `query_workspace` (returns `WorkspaceQuery`, not `ExecutionPlan`) and `evaluate` (takes a pre-built plan) are handled by threading through `evaluate`. `ExecutionPlan::new`'s signature is unchanged (PREC-03 compliant).

**PART C (staging soundness — confirmed, not changed):** `try_alloc_staging` returns `Vec<f64>` sized by `elements`. A `Vec<f64>` of N elements holds N*8 bytes = N*2 f32 lanes. This over-allocates for f32 (never under). The OOM-safe `try_reserve_exact` + `HostAllocationFailed` contract is frozen and untouched.

### Unit Tests Added (TDD)

- `execution_options_default_precision_is_f64` — confirms `ExecutionOptions::default().precision == PrecisionKind::F64`
- `plan_precision_threaded_from_options_f32` — confirms `opts.precision == F32` can be threaded to `plan.precision`
- `plan_precision_default_is_f64` — confirms default opts produces `plan.precision == F64`
- `try_alloc_staging_oom_safe_and_f32_lane_count_adequate` — confirms f64 buffer has N*2 f32 lanes >= N needed; zero-initialized; OOM-safe contract unchanged

### Acceptance Criteria Met

| Criterion | Status |
|-----------|--------|
| `ExecutionOptions` has `pub precision: PrecisionKind`; default == F64 | PASS |
| `plan.precision = opts.precision` in evaluate | PASS — line 191 |
| `try_alloc_staging` unchanged (try_reserve_exact + HostAllocationFailed) | PASS — frozen |
| `ExecutionPlan::new` signature unchanged | PASS |
| `CINTX_BACKEND=cpu cargo test -p cintx-compat --features cpu` passes | PASS — 37/37 green (PREC-03) |
| `CINTX_BACKEND=cpu cargo check --workspace --features cpu` exits 0 | PASS |

## Precision Plumbing Path (End-to-End)

After this plan, the precision flows:

```
ExecutionOptions.precision (caller sets)
    ↓ evaluate() sets plan.precision = opts.precision
ExecutionPlan.precision (wired)
    ↓ kernel dispatchers (Plans 04/05) match on plan.precision
PrecisionKind::F32 → launch_*_typed::<f32>() via bytemuck cast
PrecisionKind::F64 → launch_*_typed::<f64>() direct (zero-cost)
```

And in the executor capability gate:

```
CubeClExecutor.check_capability(backend, plan)
    if plan.precision == F32 → Ok(()) early (D-10 / PREC-06)
    if plan.precision == F64 → check_f64_capability → SHADER_F64 gate on wgpu/metal
```

## Staging Over-Allocation Note

The f64-sized `Vec<f64>` staging buffer over-allocates safely for the f32 path:
- `try_alloc_staging(elements)` allocates `elements` f64 entries = `elements * 8` bytes
- `bytemuck::cast_slice_mut::<f64, f32>` yields `elements * 2` f32 lanes
- The f32 kernel writes `elements` f32 values — safely within the 2x allocation
- **No byte-sizing change is required** to the OOM-safe contract

This was proven in Plan 01 (A5 spike) and confirmed again by T06-2d.

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| check_capability tests | `cargo test -p cintx-cubecl --features cpu executor` | 5/5 pass (1 pre-existing failure unrelated) |
| precision threading tests | `cargo test -p cintx-runtime --features cpu precision` | 3/3 pass |
| staging soundness test | `cargo test -p cintx-runtime --features cpu staging` | 1/1 pass |
| All runtime tests | `cargo test -p cintx-runtime --features cpu` | 28/28 pass |
| PREC-03 regression | `CINTX_BACKEND=cpu cargo test -p cintx-compat --features cpu` | 37/37 pass |
| Workspace check | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | 0 (exit clean) |

## Flag for Plan 07

The safe API's `evaluate::<F>()` in `cintx-rs/src/api.rs` must set `ExecutionOptions::precision` from the `F` type parameter before calling `planner::evaluate`. This is the remaining step to complete the end-to-end public precision API. Specifically:

```rust
// In api.rs evaluate::<F>():
let mut opts = self.options.clone();
opts.precision = if std::mem::size_of::<F>() == 4 { PrecisionKind::F32 } else { PrecisionKind::F64 };
// OR: use F: CintFloat to determine precision via a trait method
```

Without Plan 07, callers using the safe facade always get `PrecisionKind::F64` (default). The raw compat path (cintx-compat) correctly stays at F64 (PREC-03 frozen).

## Deviations from Plan

**1. [Rule 3 - Worktree path drift]** Initial execution accidentally committed to the `main` branch (wrong working directory). Fixed by resetting `main` to its original HEAD and performing all subsequent commits from the worktree (`/home/user/Documents/workspace/cintx/.claude/worktrees/agent-a8914d2fc3a9504a3`). No production code was affected; the accidental commit was cleanly reverted.

**2. Pre-existing test failure: `representation_transforms_keep_staging_only_contract`** — This test fails with `BufferTooSmall { required: 120, provided: 8 }` and was already failing before this plan's changes (confirmed in baseline run). Not caused by Plan 06. Logged to deferred items.

## Known Stubs

None. The `precision: PrecisionKind` field on `ExecutionOptions` is a real field with a real default value (F64) and is wired into the evaluation path via `evaluate`. The safe facade (Plan 07) still needs to thread `F` → `PrecisionKind` but that is documented above and tracked in Plan 07's scope.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries.

T-20-16: MITIGATED — `check_capability` branches explicitly on `plan.precision`; the f64 arm calls the byte-identical `check_f64_capability`; unit tests assert both directions.
T-20-17: MITIGATED — f64 buffer over-allocates for f32 (elements*2 f32 lanes >= elements); OOM-safe `try_reserve_exact` contract unchanged; T06-2d asserts the lane count.
T-20-18: MITIGATED — cintx-compat and cintx-capi not in files_modified; PREC-03 regression test (37/37) passes.

## Self-Check: PASSED

- `crates/cintx-cubecl/src/executor.rs` contains `fn check_capability`: FOUND (line 109)
- `PrecisionKind::F32` appears in executor.rs: FOUND (line 114)
- `check_f64_capability` unchanged: FOUND (still delegates to `check_shader_f64_in_features`)
- Both call sites updated to `check_capability`: FOUND (lines 206, 217)
- `crates/cintx-runtime/src/options.rs` has `pub precision: PrecisionKind`: FOUND (line 131)
- `plan.precision = opts.precision` in evaluate: FOUND (line 191)
- `try_alloc_staging` unchanged (frozen OOM contract): FOUND
- Commit 15ad514 (Task 1 RED): FOUND
- Commit 44d9ee2 (Task 1 GREEN): FOUND
- Commit 0effeea (Task 2 RED): FOUND
- Commit 47b4825 (Task 2 GREEN): FOUND
