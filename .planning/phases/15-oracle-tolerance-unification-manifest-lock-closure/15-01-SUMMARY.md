---
phase: 15-oracle-tolerance-unification-manifest-lock-closure
plan: 01
subsystem: testing
tags: [oracle, tolerance, manifest, fixtures, xtask]

# Dependency graph
requires:
  - phase: 14-unstable-source-api-families
    provides: unstable-source families implemented and oracle-wired
provides:
  - tolerance_for_family returns FamilyTolerance directly with catch-all for any family string
  - manifest_oracle_families() derives oracle-eligible families from compiled manifest lock
  - is_oracle_eligible_family() replaces hardcoded PHASE4_ORACLE_FAMILIES check
  - xtask manifest_audit.rs uses manifest-driven oracle eligibility
affects: [16-remaining-phases, oracle-harness, manifest-audit]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Catch-all tolerance with Box::leak for 'static str lifetime on unknown families"
    - "Manifest-driven oracle eligibility: parse compiled_manifest.lock.json at runtime instead of hardcoded const"

key-files:
  created: []
  modified:
    - crates/cintx-oracle/src/compare.rs
    - crates/cintx-oracle/src/fixtures.rs
    - crates/cintx-oracle/src/lib.rs
    - xtask/src/manifest_audit.rs

key-decisions:
  - "tolerance_for_family drops Result wrapper and returns FamilyTolerance directly with catch-all arm using Box::leak for unknown families at unified atol=1e-12"
  - "manifest_oracle_families() reads COMPILED_MANIFEST_LOCK_JSON at call time and collects families with stability stable or optional — replaces PHASE4_ORACLE_FAMILIES constant"
  - "PHASE4_ORACLE_FAMILIES retained as deprecated constant (not removed) to preserve external compat; internal is_phase4_oracle_family delegates to is_oracle_eligible_family"
  - "xtask manifest_audit delegates is_phase4_oracle_family to shared is_oracle_eligible_family from cintx-oracle::fixtures"

patterns-established:
  - "Manifest-driven eligibility: oracle family gates read the compiled lock JSON rather than maintain parallel hardcoded lists"
  - "Infallible tolerance: tolerance_for_family can never fail — catch-all ensures any new family gets unified atol=1e-12 automatically"

requirements-completed: [ORAC-01, ORAC-04]

# Metrics
duration: 8min
completed: 2026-04-06
---

# Phase 15 Plan 01: Oracle Tolerance Unification Summary

**Catch-all tolerance_for_family returning FamilyTolerance directly, PHASE4_ORACLE_FAMILIES replaced by manifest-driven manifest_oracle_families() using compiled lock JSON**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-05T23:44:00Z
- **Completed:** 2026-04-05T23:52:31Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `tolerance_for_family` now returns `FamilyTolerance` directly (no `Result` wrapper) with a catch-all arm that uses `Box::leak` to produce a `&'static str` for any unknown family at unified `atol=1e-12`
- Added `manifest_oracle_families()` in fixtures.rs that parses `COMPILED_MANIFEST_LOCK_JSON` and collects all families with stability `"stable"` or `"optional"` — replaces the hardcoded `PHASE4_ORACLE_FAMILIES` constant for internal eligibility checks
- Added `is_oracle_eligible_family()` as the canonical public check; `is_phase4_oracle_family()` in both fixtures.rs and xtask now delegates to it
- Updated `xtask/src/manifest_audit.rs` to import `is_oracle_eligible_family` instead of `PHASE4_ORACLE_FAMILIES`; both crates compile and all oracle tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor tolerance_for_family and manifest-driven oracle families** - `f7737e6` (feat)
2. **Task 2: Update xtask manifest_audit.rs to use manifest-driven oracle families** - `2017703` (feat)

## Files Created/Modified
- `crates/cintx-oracle/src/compare.rs` - tolerance_for_family signature changed to infallible; call site updated to remove Result match
- `crates/cintx-oracle/src/fixtures.rs` - added manifest_oracle_families(), is_oracle_eligible_family(); is_phase4_oracle_family delegates to new function; oracle_families JSON field uses manifest derivation
- `crates/cintx-oracle/src/lib.rs` - re-exports manifest_oracle_families and is_oracle_eligible_family
- `xtask/src/manifest_audit.rs` - imports is_oracle_eligible_family instead of PHASE4_ORACLE_FAMILIES; local function delegates to shared impl

## Decisions Made
- `tolerance_for_family` drops `Result` wrapper: the function can never fail since the catch-all arm covers all inputs; callers are simplified by removing the `?` operator and dead error branch
- `PHASE4_ORACLE_FAMILIES` kept as a deprecated constant for external callers; internal eligibility logic uses manifest-driven derivation
- `Box::leak` used for the catch-all arm to satisfy the `&'static str` field in `FamilyTolerance`; this occurs at most once per unique unknown family string during a process run

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Oracle tolerance is now fully open: any new family added to the manifest lock will automatically receive `atol=1e-12` tolerance without code changes
- Oracle eligibility is manifest-driven: xtask audit and fixture builders now derive eligible families from the compiled lock JSON
- Ready for Plan 15-02: manifest lock closure and full oracle coverage validation

## Self-Check: PASSED

- compare.rs: FOUND
- fixtures.rs: FOUND
- lib.rs: FOUND
- manifest_audit.rs: FOUND
- SUMMARY.md: FOUND
- Commit f7737e6: FOUND
- Commit 2017703: FOUND

---
*Phase: 15-oracle-tolerance-unification-manifest-lock-closure*
*Completed: 2026-04-06*
