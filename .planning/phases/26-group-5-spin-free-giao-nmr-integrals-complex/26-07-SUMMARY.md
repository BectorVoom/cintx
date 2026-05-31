---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 07
subsystem: testing
tags: [giao, vendor-parity, oracle, test-quality, moment_common, refactor]

# Dependency graph
requires:
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    provides: "giao_2e_parity.rs (GIAO-02 4-family vendor parity, plan 26-03)"
provides:
  - "giao_2e_parity.rs reuses moment_common's shared tolerance/mismatch helpers (single source of truth across both GIAO parity files)"
  - "self-documenting cross-center quartet selection guarded by an angular-momentum assertion (refactor-safe against fixture shell reordering)"
affects: [phase-26-verification, future-2e-giao-families]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parity tolerance/mismatch helpers live ONCE in moment_common; every parity file imports via #[path = \"moment_common.rs\"] mod moment_common"
    - "Index-literal shell selections are guarded by an angular-momentum assert_eq read back from the fixture, so shell-ordering drift fails loudly"

key-files:
  created: []
  modified:
    - "crates/cintx-oracle/tests/giao_2e_parity.rs"

key-decisions:
  - "Switched the two count_mismatches call sites to moment_common's 4-arg signature (ref, obs, ATOL, RTOL) rather than keeping a 2-arg local wrapper — preserves identical atol=1e-12/rtol=0 semantics with no divergent local tolerance copy"
  - "IN-01 resolved via approach (b): no shared quartet helper exists in moment_common, so an in-line assert_eq!(quartet_l, [0,1,0,1]) guards the quartet, reading each shell's ANG_OF from the fixture before consumption"

patterns-established:
  - "GIAO parity files share moment_common helpers (ATOL, RTOL, ncart, nsph, count_mismatches, assert_any_nonzero)"
  - "Hard-coded shell-index literals in parity tests must be guarded by a fixture-sourced angular-momentum assertion"

requirements-completed: [GIAO-02]

# Metrics
duration: 6min
completed: 2026-05-31
---

# Phase 26 Plan 07: GIAO-02 2e parity test-quality hardening Summary

**giao_2e_parity.rs now imports moment_common's shared tolerance/mismatch helpers (single source of truth) and guards its cross-center quartet with a fixture-sourced [0,1,0,1] angular-momentum assertion — closing WR-03 + IN-01 without changing what the GIAO-02 gate tests.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-05-31
- **Completed:** 2026-05-31
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- WR-03 closed: deleted the local duplicate `ATOL`/`RTOL` consts and the local `matches_with_tol`, `count_mismatches`, `ncart`, `nsph`, `assert_any_nonzero` re-implementations; all now imported from `moment_common`, so a tolerance change tracks both GIAO parity files (and the Phase 24 moment scaffolds) instead of silently drifting (T-26-11).
- IN-01 closed: the `[3, 2, 3, 2]` quartet literal is now guarded by `assert_eq!(quartet_l, [0, 1, 0, 1], ...)` reading each shell's `ANG_OF` from the live fixture, so a future shell-reordering in `build_h2o_sto3g_common_orig` fails loudly rather than silently selecting different angular momenta (T-26-12).
- All 4 GIAO-02 vendor parity tests (`int2e_g1`, `int2e_ig1`, `int2e_gg1`, `int2e_g1g2`) still pass byte-identical to vendored libcint at atol=1e-12 with the refactored helpers.

## Task Commits

Each task was committed atomically:

1. **Task 1: Import shared moment_common helpers; delete local duplicates** - `b54a54e` (refactor)
2. **Task 2: Make the cross-center quartet self-documenting (IN-01)** - `8d8418d` (test)

## Files Created/Modified
- `crates/cintx-oracle/tests/giao_2e_parity.rs` - Imports `ATOL, RTOL, assert_any_nonzero, count_mismatches, ncart, nsph` from `moment_common` (via `#[path = "moment_common.rs"] mod moment_common;`); removed local duplicates; added a fixture-sourced angular-momentum assertion on the quartet.

## Decisions Made
- **count_mismatches signature:** moment_common's `count_mismatches` is the 4-arg `(reference, observed, atol, rtol)` form, whereas the deleted local copy was 2-arg with hard-coded tolerances. Updated both call sites to pass `ATOL, RTOL` explicitly. Semantics are byte-identical (atol=1e-12, rtol=0); the goal was one source of truth, not a tolerance change. The minor logging-format difference (moment_common logs every mismatch with a threshold; the local copy capped at 16 with a rel field) is cosmetic and does not affect the pass/fail outcome.
- **IN-01 approach (b):** `moment_common` exposes no 2e quartet helper (only `non_square_shell_pair` / `cross_center_non_square_shell_pair`, both `(usize, usize)` pairs). Rather than add a quartet helper to a module shared with Phase 24 1e scaffolds, the plan's approved fallback (b) was used: an in-line `assert_eq!(..., [0, 1, 0, 1], ...)` guard at the point of use in `giao_2e_vendor_parity`.
- **assert_any_nonzero kept shared:** moment_common already exports `assert_any_nonzero` with identical semantics, so the local copy was deleted and the import added (one more shared helper than the plan's `giao_1e_parity.rs:24-33` reference, which did not need it).

## Deviations from Plan

None - plan executed exactly as written. Both tasks followed the plan's prescribed approach (Task 1 import-and-delete; Task 2 approach (b) AM assertion).

## Issues Encountered
None. Build passed on the first attempt for both tasks; all 4 vendor parity tests pass.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The GIAO-02 parity gate (`giao_2e_parity.rs`) is now refactor-safe: tolerance drift and fixture shell-ordering drift both fail loudly.
- No blockers. This plan touched only `giao_2e_parity.rs`, with no overlap with any other Phase 26 gap-closure plan.

## Self-Check: PASSED

- `crates/cintx-oracle/tests/giao_2e_parity.rs` modified and committed.
- SUMMARY.md present at `.planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-07-SUMMARY.md`.
- Commits verified in git log: `b54a54e` (Task 1), `8d8418d` (Task 2), `0c716cb` (SUMMARY).
- All 4 GIAO-02 vendor parity tests pass at atol=1e-12 with the refactored helpers.

---
*Phase: 26-group-5-spin-free-giao-nmr-integrals-complex*
*Completed: 2026-05-31*
