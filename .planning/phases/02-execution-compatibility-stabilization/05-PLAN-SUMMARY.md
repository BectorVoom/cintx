---
phase: 02-execution-compatibility-stabilization
plan: 05
subsystem: execution
tags: [rust, cubecl, runtime, kernels, transfer-planning]
requires:
  - phase: 02-execution-compatibility-stabilization
    provides: runtime backend contract and planner dispatch ownership model from plan 04
provides:
  - Concrete `CubeClExecutor` with explicit CPU runtime-profile baseline and typed unsupported-family gating
  - `SpecializationKey`, `DeviceResidentCache`, and staging-only `TransferPlan` flow for backend execution
  - Kernel family registry and launch wiring for canonical `1e`, `2e`, and `2c2e` paths
affects: [compat-dispatch, cubecl-backend, oracle-execution, phase-02-plan-06]
tech-stack:
  added: []
  patterns:
    - canonical-family kernel registry dispatch
    - staging-only backend output ownership (CompatFinalWrite boundary)
    - basis/device resident metadata caching
key-files:
  created:
    - crates/cintx-cubecl/src/executor.rs
    - crates/cintx-cubecl/src/resident_cache.rs
    - crates/cintx-cubecl/src/specialization.rs
    - crates/cintx-cubecl/src/transfer.rs
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  modified:
    - crates/cintx-cubecl/src/lib.rs
    - crates/cintx-cubecl/src/executor.rs
key-decisions:
  - "Pinned the initial executable CubeCL profile to `CUBECL_RUNTIME_PROFILE = \"cpu\"` and exposed a concrete constructor through `CubeClExecutor::new`."
  - "Kept backend execution fail-closed to canonical `1e`/`2e`/`2c2e` registry entries and returned `UnsupportedApi` for follow-on families."
  - "Preserved planner output ownership contract as `BackendStagingOnly -> CompatFinalWrite`; transfer planning only stages metadata/workspace/output buffers."
patterns-established:
  - "Registry-driven dispatch: choose launch path from manifest canonical family + representation metadata, not raw symbol names."
  - "Transfer planning is allocation-aware and maps staging/device allocation failures to typed `HostAllocationFailed` / `DeviceOutOfMemory` errors."
requirements-completed: [EXEC-02, EXEC-03]
duration: 10min
completed: 2026-03-21
---

# Phase 2 Plan 05: CubeCL Executor Core and Base Family Launch Slice Summary

**Concrete CPU-profile CubeCL executor with resident cache and staging transfer planning, plus canonical-family launch coverage for `1e`, `2e`, and `2c2e`.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-21T10:40:10Z
- **Completed:** 2026-03-21T10:49:50Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Implemented a concrete `CubeClExecutor` that satisfies `BackendExecutor`, exports an explicit constructor, and anchors Phase 2 runtime baseline to `cpu`.
- Added specialization, resident metadata cache, and transfer-planning modules to keep host work in validation/marshaling/copy orchestration and staging-only ownership.
- Implemented and tested kernel registry launch wiring for canonical `1e`, `2e`, and `2c2e`, with explicit rejection of unsupported follow-on families.

## Task Commits

1. **Task 1: Implement the CubeCL executor core, specialization keys, resident cache, and transfer planner** - `540efdc` (feat)
2. **Task 2: Implement the `1e`, `2e`, and `2c2e` CubeCL family registry and launch path** - `95e140c` (feat)

## Files Created/Modified
- `crates/cintx-cubecl/src/lib.rs` - Exported concrete executor/cache/specialization/transfer surfaces.
- `crates/cintx-cubecl/src/executor.rs` - Added `CubeClExecutor` implementation, runtime profile pin, typed support checks, and kernel-registry dispatch.
- `crates/cintx-cubecl/src/resident_cache.rs` - Added basis/device-scoped `DeviceResidentCache` keyed by basis hash + representation.
- `crates/cintx-cubecl/src/specialization.rs` - Added `SpecializationKey` with canonical family, representation, component rank, and shell angular-momentum tuple.
- `crates/cintx-cubecl/src/transfer.rs` - Added staging-only `TransferPlan` and typed allocation/transfer failure mapping.
- `crates/cintx-cubecl/src/kernels/mod.rs` - Added canonical family registry (`1e`, `2e`, `2c2e`) and unsupported-family rejection tests.
- `crates/cintx-cubecl/src/kernels/one_electron.rs` - Added `1e` launch entry returning typed `ExecutionStats`.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` - Added `2e` launch entry returning typed `ExecutionStats`.
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` - Added `2c2e` launch entry returning typed `ExecutionStats`.

## Decisions Made
- Kept `center_4c1e`, `center_3c1e`, and `center_3c2e` outside the active registry and returned typed `UnsupportedApi` for unresolved families.
- Ensured transfer/output contract remains explicit (`BackendStagingOnly` and `CompatFinalWrite`) at executor and transfer-plan boundaries.
- Used canonical-family registry resolution in kernels and executor to avoid symbol-name branching in backend dispatch.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CubeCL backend now has an executable core path and deterministic runtime profile baseline for further compat and oracle wiring.
- Follow-on plans can add `3c1e`/`3c2e` launch support and broader transform coverage without revisiting executor ownership contracts.

## Self-Check: PASSED

- FOUND: `.planning/phases/02-execution-compatibility-stabilization/05-PLAN-SUMMARY.md`
- FOUND: `540efdc`
- FOUND: `95e140c`

---
*Phase: 02-execution-compatibility-stabilization*
*Completed: 2026-03-21*
