---
phase: 02-cpu-compatibility-execution
plan: "06"
subsystem: runtime
tags: [libcint, cpu-backend, routing, spinor]
requires:
  - phase: 02-cpu-compatibility-execution
    provides: deterministic CPU linkage baseline and shared execution request contracts from 02-01
provides:
  - Typed CPU FFI symbol catalog for phase-2 stable families
  - Family/operator/representation backend router with typed unsupported outcomes
  - Dedicated 3c1e spinor adapter route that avoids unsupported short-circuiting
  - Stable-family routing matrix tests enforcing full phase-2 envelope coverage
affects: [phase-2-routing, cpu-backend, compatibility-tests, execution-planner]
tech-stack:
  added: []
  patterns: [typed route-key dispatch, adapter-backed compatibility routing]
key-files:
  created:
    - src/runtime/backend/mod.rs
    - src/runtime/backend/cpu/mod.rs
    - src/runtime/backend/cpu/ffi.rs
    - src/runtime/backend/cpu/router.rs
    - src/runtime/backend/cpu/spinor_3c1e.rs
  modified:
    - src/runtime/mod.rs
    - src/lib.rs
    - tests/phase2_cpu_backend_routing.rs
key-decisions:
  - "Use a typed CpuRouteKey (family, operator, representation) as the single dispatch input for CPU routing."
  - "Treat out-of-phase route envelopes as explicit UnsupportedApi errors before any unsafe backend call."
  - "Route 3c1e spinor through an explicit adapter metadata path backed by the supported spherical 3c1e kernel."
patterns-established:
  - "Backend routing tests must assert both full in-scope coverage and typed unsupported behavior for out-of-scope envelopes."
  - "Special-case compatibility gaps (like 3c1e spinor) are represented as explicit adapter route variants, not implicit fallbacks."
requirements-completed: [EXEC-01, COMP-01]
duration: 11 min
completed: 2026-03-14
---

# Phase 2 Plan 06: CPU Router and 3c1e Spinor Envelope Summary

**Typed CPU routing now covers all phase-2 stable-family envelopes with an explicit 3c1e spinor adapter path instead of unsupported dispatch.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-14T06:27:37Z
- **Completed:** 2026-03-14T06:38:10Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Added the `src/runtime/backend/cpu` module stack with typed symbol bindings for stable-family CPU kernels.
- Implemented deterministic router dispatch keyed by `(family, operator, representation)` with typed unsupported errors for out-of-scope envelopes.
- Added dedicated `3c1e` spinor adapter metadata and wired router behavior so in-scope requests route successfully.
- Expanded route tests to lock complete stable-family coverage and assert typed unsupported outcomes only for out-of-scope keys.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement typed CPU FFI bindings and stable-family router map** - `7450f55` (feat)
2. **Task 2: Implement `3c1e` spinor adapter path and wire it as required route** - `37a8eb5` (feat)
3. **Task 3: Expand routing tests to enforce full stable-family matrix** - `886b2b3` (test)

## Files Created/Modified

- `src/runtime/backend/mod.rs` - Introduced runtime backend namespace.
- `src/runtime/backend/cpu/mod.rs` - Added CPU backend module exports.
- `src/runtime/backend/cpu/ffi.rs` - Added typed symbol catalog and C symbol bindings.
- `src/runtime/backend/cpu/router.rs` - Added typed routing key and dispatch resolution logic.
- `src/runtime/backend/cpu/spinor_3c1e.rs` - Added explicit `3c1e` spinor adapter route metadata.
- `src/runtime/mod.rs` - Re-exported backend route types/functions in runtime surface.
- `src/lib.rs` - Re-exported runtime backend routing API at crate root.
- `tests/phase2_cpu_backend_routing.rs` - Added route matrix and `3c1e` adapter coverage tests.

## Decisions Made

- A typed route key is the canonical dispatch input to keep safe/raw callers on one deterministic backend map.
- Unsupported results are constrained to out-of-phase envelopes and returned as `LibcintRsError::UnsupportedApi`.
- `3c1e` spinor dispatch is represented by an explicit adapter variant to make the upstream gap visible and testable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reconciled STATE/ROADMAP updates after gsd-tools parser mismatch**
- **Found during:** Post-task state update step
- **Issue:** `state advance-plan` returned `Cannot parse Current Plan or Total Plans in Phase from STATE.md`, and `roadmap update-plan-progress` reported success without mutating plan completion rows.
- **Fix:** Applied direct markdown updates to `.planning/STATE.md` and `.planning/ROADMAP.md` to reflect completed plan `02-06`, updated progress values, and corrected phase plan checkboxes.
- **Files modified:** `.planning/STATE.md`, `.planning/ROADMAP.md`
- **Verification:** Re-read both files and confirmed `Phase 2` progress now reflects `2/8` with `02-01` and `02-06` checked.

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Documentation/state bookkeeping only; implementation scope and verification outcomes were unchanged.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CPU backend dispatch contracts are now explicit and test-locked for all stable-family phase-2 envelopes.
- Downstream execution/evaluation plans can consume `route`/`route_request` without re-encoding family/operator/representation maps.

---
*Phase: 02-cpu-compatibility-execution*
*Completed: 2026-03-14*

## Self-Check: PASSED
