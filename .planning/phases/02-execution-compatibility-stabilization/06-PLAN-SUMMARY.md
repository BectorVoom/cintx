---
phase: 02-execution-compatibility-stabilization
plan: 06
subsystem: compat-runtime
tags: [rust, compat, raw-api, runtime, cubecl, layout]
requires:
  - phase: 02-execution-compatibility-stabilization
    provides: runtime dispatch contract and cubecl executor slice from plans 04-05
provides:
  - Raw compat validation and typed raw views for atm/bas/env layouts
  - `RawApiId`, `query_workspace_raw`, and `eval_raw` bridged through runtime + CubeCL
  - Compat-owned dims/output/cache guards with no-partial-write regression coverage
affects: [phase-02-plan-07, compat-raw-surface, oracle-parity-inputs]
tech-stack:
  added: []
  patterns:
    - compat-owned final-write boundary (backend staging only)
    - raw sentinel semantics (`dims/out/cache`) validated before execution
key-files:
  created:
    - crates/cintx-compat/src/raw.rs
  modified:
    - crates/cintx-compat/src/layout.rs
    - crates/cintx-compat/src/lib.rs
key-decisions:
  - "Represent raw API dispatch through symbol-backed `RawApiId` and resolve operator metadata through `Resolver` at call time."
  - "Treat `RawOptimizerHandle::workspace_hint_bytes` as compat execution memory-limit hint to support deterministic chunk/failure tests without extending the public raw signature."
  - "Keep final caller-visible writes exclusively in `CompatDims::write`; evaluate paths never touch `out` until all checks and backend execution succeed."
patterns-established:
  - "Prepare-then-execute flow: resolve/validate/build/query/layout-check first, then evaluate/write."
  - "Fail-closed raw contract tests assert unchanged output buffers on all pre-write failures."
requirements-completed: [COMP-01, COMP-02, COMP-05, EXEC-02, EXEC-03, EXEC-04]
duration: 26min
completed: 2026-03-26
---

# Phase 2 Plan 06: Raw Compat Query/Evaluate Pipeline Summary

**Raw libcint-style compat calls now resolve through the shared runtime/CubeCL path with strict layout guards and no-partial-write failure behavior.**

## Performance

- **Duration:** 26 min
- **Started:** 2026-03-26T10:45:54Z
- **Completed:** 2026-03-26T11:11:28Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Implemented raw `atm`/`bas`/`env` view validation plus typed basis/shell conversion and manifest-backed API resolution.
- Added `query_workspace_raw` and `eval_raw` sentinel behavior (`dims == None`, `out == None`, `cache == None`) over runtime planning and CubeCL execution.
- Added layout/raw regression tests for invalid layouts/offsets, undersized buffers, cache guards, chunking limits, 3c query contract, and no-partial-write guarantees.

## Task Commits

1. **Task 1: Implement raw views, minimal optimizer-handle contract, and shared dims/output contract** - `1739786` (feat)
2. **Task 2: Bridge raw compat calls into runtime query/evaluate with sentinel behavior** - `2d1521b` (feat)
3. **Task 3: Add raw compat regression tests for layout/buffer/no-partial-write behavior** - `2d1521b` (feat)

## Files Created/Modified

- `crates/cintx-compat/src/raw.rs` - Added raw API surface, resolver mapping, basis conversion, workspace/evaluate bridge, and regression tests.
- `crates/cintx-compat/src/layout.rs` - Added layout contract regression tests for dims/size/cache enforcement.
- `crates/cintx-compat/src/lib.rs` - Re-exported raw API entry points and supporting types.

## Decisions Made

- Chose symbol-based `RawApiId` resolution against the canonical manifest to keep raw dispatch manifest-driven and helper-kind scoped.
- Kept `RawOptimizerHandle` forwarding minimal but functional by mapping workspace hints into runtime options for chunk/failure testing.
- Preserved strict ownership: compat layout performs the only caller-visible flat write, and only after execution success.

## Deviations from Plan

- Subagent execution was interrupted mid-task and temporarily removed `raw.rs`; work resumed inline from the last successful commit and completed without scope changes.

## Issues Encountered

- Interruption during delegated execution left an inconsistent worktree (`raw.rs` deleted). Recovered by restoring from Task 1 commit and completing remaining tasks inline.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Raw compat query/evaluate entry points and validation invariants are now in place for Plan 07 helper/legacy/oracle expansion.
- `3c1e` query contracts are validated while evaluated execution remains blocked behind Plan 08 family enablement (still returns `UnsupportedApi` on eval).

## Self-Check: PASSED

- FOUND: `.planning/phases/02-execution-compatibility-stabilization/06-PLAN-SUMMARY.md`
- FOUND: `1739786`
- FOUND: `2d1521b`

---
*Phase: 02-execution-compatibility-stabilization*
*Completed: 2026-03-26*
