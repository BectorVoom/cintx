---
phase: 02-execution-compatibility-stabilization
plan: 08
subsystem: cubecl-execution
tags: [rust, cubecl, kernels, transforms, staging, representation]
requires:
  - phase: 02-execution-compatibility-stabilization
    provides: executor core and base family dispatch contract from plan 05 plus compat raw writer contract from plan 06
provides:
  - `3c1e` and `3c2e` kernel launch support in CubeCL family registry
  - Representation-specific staging transforms (`c2s`, `c2spinor`) wired into executor flow
  - Maintained `BackendStagingOnly -> CompatFinalWrite` ownership boundary with explicit tests
affects: [phase-02-plan-07, compat-eval-path, oracle-parity-scope]
tech-stack:
  added: []
  patterns:
    - family-complete kernel registry with explicit unsupported-family list
    - executor staging seed + representation transform pipeline before compat final write
key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-cubecl/src/transform/mod.rs
    - crates/cintx-cubecl/src/transform/c2s.rs
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-cubecl/src/executor.rs
key-decisions:
  - "Enable `3c1e`/`3c2e` directly in the canonical kernel registry and leave only `4c1e` in the explicit unsupported list."
  - "Route all executor outputs through representation transforms (`Cart` identity, `Spheric` c2s, `Spinor` interleaved c2spinor) before compat final write."
patterns-established:
  - "Transform routing is representation-driven and confined to CubeCL staging modules."
  - "Executor tests assert staging-only ownership and transform selection without leaking caller-visible write responsibility into backend code."
requirements-completed: [EXEC-02, EXEC-04]
duration: 8min
completed: 2026-03-26
---

# Phase 2 Plan 08: CubeCL 3c Families and Transform Routing Summary

**CubeCL now executes `3c1e`/`3c2e` families and applies representation-specific staging transforms before compat commits final caller-visible buffers.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-26T11:11:28Z
- **Completed:** 2026-03-26T11:19:16Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Replaced `3c1e` and `3c2e` kernel stubs with concrete launch modules and added them to registry dispatch.
- Added deterministic staging transforms for spherical and spinor paths and wired executor representation routing through them.
- Expanded tests to validate full Phase 2 base-family support, explicit `4c1e` rejection, transform behavior, and staging-only ownership guarantees.

## Task Commits

1. **Task 1: Implement the `3c1e` and `3c2e` CubeCL family kernels** - `fcda0f5` (feat)
2. **Task 2: Implement cart-to-spherical and cart-to-spinor output transforms and wire them into execution** - `fcda0f5` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` - Added `3c1e` launch entry and canonical-family validation.
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` - Added `3c2e` launch entry and canonical-family validation.
- `crates/cintx-cubecl/src/kernels/mod.rs` - Enabled `3c1e`/`3c2e` in registry and narrowed unresolved list to `4c1e`.
- `crates/cintx-cubecl/src/transform/mod.rs` - Added representation-based transform routing helper.
- `crates/cintx-cubecl/src/transform/c2s.rs` - Implemented cart-to-spheric staging transform and tests.
- `crates/cintx-cubecl/src/transform/c2spinor.rs` - Implemented cart-to-spinor interleaved staging transform and tests.
- `crates/cintx-cubecl/src/executor.rs` - Routed staging through transform path while preserving compat final-write ownership.
- `crates/cintx-compat/src/raw.rs` - Updated 3c regression expectation now that backend support is enabled.

## Decisions Made

- Kept backend family bounds explicit by supporting base-set families (`1e`,`2e`,`2c2e`,`3c1e`,`3c2e`) and leaving `4c1e` rejected.
- Used representation-driven transform routing in executor to keep compatibility with downstream compat layout ownership.

## Deviations from Plan

- Added a compat regression update (`raw.rs`) to reflect newly supported `3c1e` execution; this was required to keep cross-crate tests consistent after backend enablement.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 07 can now rely on full Phase 2 family execution coverage when adding helper/legacy/optimizer APIs and oracle parity checks.
- `4c1e` remains intentionally unsupported in this phase boundary.

## Self-Check: PASSED

- FOUND: `.planning/phases/02-execution-compatibility-stabilization/08-PLAN-SUMMARY.md`
- FOUND: `fcda0f5`

---
*Phase: 02-execution-compatibility-stabilization*
*Completed: 2026-03-26*
