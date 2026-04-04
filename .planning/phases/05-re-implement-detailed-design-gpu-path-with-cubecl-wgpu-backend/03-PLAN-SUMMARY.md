---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 03
subsystem: cubecl
tags: [rust, cubecl, wgpu, executor, kernel, bootstrap, unsupported-taxonomy, validated-4c1e, d-11, d-12]

# Dependency graph
requires:
  - phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
    plan: 02
    provides: bootstrap_wgpu_runtime, WgpuPreflightReport, CapabilityReason taxonomy for executor consumption

provides:
  - CubeClExecutor without CUBECL_RUNTIME_PROFILE constant or fill_cartesian_staging synthetic helper (D-05)
  - preflight_wgpu() calling bootstrap_wgpu_runtime at query/execute entry with wgpu-capability: typed errors (D-01, D-02)
  - ensure_validated_4c1e using wgpu capability preflight instead of cpu-profile string check (D-11)
  - ensure_supported_family returning unsupported_family:<canonical_family> and unsupported_representation:<repr> taxonomy prefixes (D-12)
  - kernels::resolve_family returning unsupported_family:<canonical_family> and unsupported_representation:<repr> via UnsupportedApi (D-12)
  - BackendStagingOnly and CompatFinalWrite ownership checks preserved throughout real CubeCL launch path (D-06)
  - Transfer adapter label from backend_intent.selector propagated to stage_device_buffers (D-04)
  - TDD regression tests preventing reintroduction of synthetic staging fill or cpu-profile gate

affects:
  - 05-04 (kernel GPU compute path consumes the wgpu bootstrap and ownership contract established here)
  - cintx-compat (raw.rs validated_4c1e check aligned with new wgpu-based gate)
  - Any caller relying on unsupported family/representation error text — now uses taxonomy prefix format

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "unsupported_family:<canonical_family> and unsupported_representation:<repr> in both executor and kernels for D-12 taxonomy"
    - "preflight_wgpu() at query/execute entry provides D-01/D-02 fail-closed gate before any family/representation check"
    - "ensure_validated_4c1e gated under cfg(feature = with-4c1e) with wgpu capability preflight replacing cpu-profile string check"
    - "Tests use match executor.execute() to accept either GPU-success or wgpu-capability:missing_adapter fail-closed path"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/executor.rs
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-compat/src/raw.rs

key-decisions:
  - "Gate ensure_validated_4c1e and validated_4c1e_error under cfg(feature = with-4c1e) to eliminate dead_code warnings in default builds"
  - "Gate Representation import under cfg(feature = with-4c1e) since it is only used in ensure_validated_4c1e and with-4c1e test code"
  - "kernels::resolve_family now returns UnsupportedApi with unsupported_representation:<repr> instead of UnsupportedRepresentation struct to keep taxonomy consistent"
  - "Transfer adapter label sourced from backend_intent.selector rather than a static runtime_profile string per D-04 reproducibility"

patterns-established:
  - "D-11: Validated4C1E now enforced via wgpu capability preflight — cpu-profile string check is removed"
  - "D-12: Both executor and kernels use unsupported_family:<family> and unsupported_representation:<repr> taxonomy prefixes"
  - "TDD anti-regression: executor tests match on both Ok (GPU present) and Err(UnsupportedApi{wgpu-capability:...}) (no GPU) to remain valid in CI"

requirements-completed:
  - EXEC-02
  - EXEC-03
  - COMP-05

# Metrics
duration: 29min
completed: 2026-04-02
---

# Phase 05 Plan 03: Executor and Kernel Real CubeCL Launch Path Summary

**Real CubeCL chunk execution path without synthetic staging fill, with fail-closed wgpu preflight and D-12 unsupported taxonomy in executor and kernels**

## Performance

- **Duration:** 29 min
- **Started:** 2026-04-02T08:18:23Z
- **Completed:** 2026-04-02T08:47Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Removed `fill_cartesian_staging()` and `CUBECL_RUNTIME_PROFILE` constant from executor — no synthetic CPU-side staging fill path remains (D-05)
- Added `preflight_wgpu()` calling `bootstrap_wgpu_runtime` at query/execute entry with `wgpu-capability:` typed errors (D-01, D-02)
- Updated `ensure_validated_4c1e` to require wgpu capability preflight success instead of cpu-profile string check, gated under `cfg(feature = "with-4c1e")` (D-11)
- Updated `ensure_supported_family` and `kernels::resolve_family` to return `unsupported_family:<canonical_family>` and `unsupported_representation:<repr>` taxonomy prefixes in `UnsupportedApi` errors (D-12)
- Preserved `BackendStagingOnly` and `CompatFinalWrite` ownership checks throughout real kernel launch path (D-06)
- Added anti-regression TDD tests covering: monotonic stub removal, wgpu bootstrap at execute entry, D-11 wgpu capability gate for Validated4C1E, D-12 family and representation taxonomy

## Task Commits

Task 1 was completed by a parallel agent (worktree-agent-a9801704) and merged before Task 2 execution:

1. **Task 1 RED: Add failing tests for wgpu bootstrap and stub removal** - `7b4bf4e` (test) [parallel agent]
2. **Task 1 GREEN: Rewire executor to real CubeCL launch path; remove synthetic staging fill** - `17fd33a` (feat) [parallel agent]
3. **Task 2: Implement unsupported taxonomy and Validated4C1E capability gates** - `5f6dfe4` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/executor.rs` - Removed fill_cartesian_staging/CUBECL_RUNTIME_PROFILE, added preflight_wgpu(), updated ensure_validated_4c1e (D-11), ensure_supported_family (D-12), added Task 2 regression tests (656 lines)
- `crates/cintx-cubecl/src/kernels/mod.rs` - Updated resolve_family to return unsupported_family:/unsupported_representation: taxonomy (D-12), added Task 2 tests (302 lines)
- `crates/cintx-compat/src/raw.rs` - Removed cpu-profile gate from validated_4c1e check (parallel agent, Task 1)

## Decisions Made

- Gate `ensure_validated_4c1e` and `validated_4c1e_error` under `cfg(feature = "with-4c1e")` to eliminate dead_code warnings in default builds
- Gate `Representation` import under `cfg(feature = "with-4c1e")` since it is only used in the `with-4c1e`-gated methods and tests
- `kernels::resolve_family` now returns `UnsupportedApi { requested: "unsupported_representation:<repr>" }` instead of `UnsupportedRepresentation` struct to maintain consistent D-12 taxonomy across both executor and kernels layers
- Transfer adapter label sourced from `backend_intent.selector` rather than the removed static `runtime_profile` string per D-04 reproducibility

## Deviations from Plan

### Parallel Agent Coordination

**1. [Rule 3 - Blocking] Task 1 already completed by parallel agent**
- **Found during:** Initial worktree setup
- **Issue:** Worktree `worktree-agent-ac1205e0` was at commit `95c3cbf` (pre-Phase-05 merge point). Task 1 commits (`7b4bf4e`, `17fd33a`) existed on `worktree-agent-a9801704` branch.
- **Fix:** Fast-forward merged `worktree-agent-a9801704` into current worktree branch before starting Task 2.
- **Files modified:** (structural — brought in Plan 01, 02, and 03 Task 1 commits)
- **Verification:** `cargo test -p cintx-cubecl` passes 30 tests after merge.

### Auto-fixed Issues

**2. [Rule 1 - Bug] Dead code warnings from cfg-gated functions**
- **Found during:** Task 2 (build verification)
- **Issue:** `ensure_validated_4c1e` and `validated_4c1e_error` produce dead_code warnings in non-`with-4c1e` builds because they are only called inside `#[cfg(feature = "with-4c1e")]` blocks
- **Fix:** Added `#[cfg(feature = "with-4c1e")]` attribute to both functions; gated `Representation` import under the same feature flag
- **Files modified:** `crates/cintx-cubecl/src/executor.rs`
- **Verification:** `cargo build -p cintx-cubecl` shows no cintx-cubecl warnings; all 30 tests pass

---

**Total deviations:** 2 (1 coordination, 1 dead_code auto-fix)
**Impact on plan:** Both handled correctly. No scope creep.

## Issues Encountered

- Task 1 was already implemented by a parallel agent running in a sibling worktree. Merged cleanly via fast-forward — no conflicts. Task 2 proceeded as planned.

## Known Stubs

The kernel launch functions (`one_electron::launch_one_electron`, `two_electron::launch_two_electron`, etc.) still return `ExecutionStats` with `not0 = 1` (based on staging buffer being non-empty) but do not run actual GPU compute. These are intentional stubs pending real CubeCL kernel implementation in Phase 05 Plan 04/05. The ownership contract and preflight path are real and correct; only the integral compute values are stub zeros.

## Next Phase Readiness

- Executor now correctly calls `bootstrap_wgpu_runtime` at query/execute entry with fail-closed wgpu-capability errors (D-01, D-02)
- Unsupported scope uses explicit taxonomy reasons in both executor and kernels layers (D-12)
- Validated4C1E gate is now wgpu-capability-based, not cpu-profile-string-based (D-11)
- Ownership contracts (`BackendStagingOnly` → `CompatFinalWrite`) remain enforced (D-06)
- Plan 04 can build on these foundations to implement real CubeCL integral kernel compute paths

---
*Phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend*
*Completed: 2026-04-02*
