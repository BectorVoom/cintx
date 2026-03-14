---
phase: 02-cpu-compatibility-execution
plan: "08"
subsystem: api
tags: [memory-policy, raw-compat, allocation-failure, diagnostics, regression-tests]
requires:
  - phase: 02-cpu-compatibility-execution
    provides: centralized runtime memory policy and raw query/evaluate routing from 02-04 and 02-07
provides:
  - Shared executor memory-policy outcomes reused by raw compatibility query/evaluate flows
  - Raw query->execute contract checks that include memory-policy metadata parity under limits
  - API-level typed AllocationFailure regression coverage for safe and raw evaluate entrypoints
affects: [phase-2-close-matrix, raw-api-memory-threading, diagnostics-contracts]
tech-stack:
  added: []
  patterns: [shared memory-policy outcome, query-execute contract locking, simulated allocation-failure gate]
key-files:
  created:
    - tests/phase2_allocation_failures.rs
  modified:
    - src/runtime/executor.rs
    - src/runtime/raw/query.rs
    - src/runtime/raw/evaluate.rs
    - src/runtime/raw/validator.rs
    - tests/phase2_memory_limits.rs
key-decisions:
  - "Expose memory-policy planning as a shared executor outcome so safe and raw surfaces cannot drift on memory limits."
  - "Persist raw query memory-policy metadata and require execute-time parity for memory required bytes, working set, scratch, and chunking."
  - "Use an explicit feature-flag simulation gate to deterministically assert typed AllocationFailure diagnostics at API boundaries."
patterns-established:
  - "Raw query and raw execute must validate both shape/layout and memory-policy invariants before dispatch."
  - "Allocation-failure API regressions are tested via deterministic simulation flags instead of probabilistic allocator pressure."
requirements-completed: [MEM-01, MEM-02, RAW-02]
duration: 12 min
completed: 2026-03-14
---

# Phase 2 Plan 08: API Memory Threading and Allocation Failure Contracts Summary

**Safe and raw API execution paths now share one runtime memory-policy contract, with typed allocation-failure and raw query->execute memory-limit regressions locked in tests.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-14T07:26:09Z
- **Completed:** 2026-03-14T07:38:09Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added `MemoryPolicyOutcome` + shared planner helper in `runtime::executor` and reused it across safe and raw execution surfaces.
- Extended raw compatibility query/evaluate contracts to carry and enforce memory-policy metadata consistency (`required`, `working_set`, `scratch`, `chunking`).
- Added API-visible allocation-failure coverage in `phase2_allocation_failures` and expanded memory-limit regression coverage in `phase2_memory_limits`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Thread memory options through safe/raw API entrypoints** - `b157019` (feat)
2. **Task 2: Add API-visible allocation failure regression tests** - `47cf467` (fix)
3. **Task 3: Lock RAW-02 + MEM contract integration under memory limits** - `dd8c5c9` (test)

## Files Created/Modified

- `src/runtime/executor.rs` - Added shared memory-policy outcome/planning helper and simulation gate for allocation-failure regression paths.
- `src/runtime/raw/validator.rs` - Added raw validation metadata needed for shared memory-policy planning (`shell_angular_momentum`, `primitive_count`).
- `src/runtime/raw/query.rs` - Added raw workspace memory-policy metadata and memory-policy failure handling at query time.
- `src/runtime/raw/evaluate.rs` - Enforced raw query->execute memory-policy parity and chunk-aware staged output writes through fallible allocation.
- `tests/phase2_memory_limits.rs` - Added safe/raw API threading and raw query->execute memory-limit contract regression tests.
- `tests/phase2_allocation_failures.rs` - Added typed allocation-failure and diagnostics assertions through safe/raw API entrypoints.

## Decisions Made

- Treat memory-policy parity (not just dims/layout parity) as part of RAW-02 query->execute contract enforcement.
- Keep raw compatibility `required_bytes` as output contract bytes while carrying separate memory-policy bytes for limit enforcement and diagnostics.
- Simulate allocation failures with an explicit feature flag to avoid flaky allocator-pressure tests and preserve deterministic CI behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reconciled plan-state metadata after `state advance-plan` parser failure**
- **Found during:** Post-task state update
- **Issue:** `state advance-plan` returned `Cannot parse Current Plan or Total Plans in Phase from STATE.md`, which prevented automated position/session updates.
- **Fix:** Updated `.planning/STATE.md` and `.planning/ROADMAP.md` directly to reflect completed `02-08` progress and session continuity.
- **Files modified:** `.planning/STATE.md`, `.planning/ROADMAP.md`
- **Verification:** Re-read both files and confirmed `02-08` is checked, phase progress is `7/8`, and state progress is `9/10` (90%).

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Metadata/state bookkeeping only; implementation scope and verification outputs were unchanged.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 API memory threading and allocation-failure semantics are now regression-protected across safe/raw surfaces.
- Remaining Phase 2 closeout work (`02-05`) can consume these tests as compatibility evidence for MEM/RAW behavior under constraints.

---
*Phase: 02-cpu-compatibility-execution*
*Completed: 2026-03-14*

## Self-Check: PASSED

- FOUND: .planning/phases/02-cpu-compatibility-execution/02-08-SUMMARY.md
- FOUND COMMIT: b157019
- FOUND COMMIT: 47cf467
- FOUND COMMIT: dd8c5c9
