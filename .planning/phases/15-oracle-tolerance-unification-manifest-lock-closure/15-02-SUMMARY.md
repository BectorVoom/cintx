---
phase: 15-oracle-tolerance-unification-manifest-lock-closure
plan: 02
subsystem: testing
tags: [oracle, manifest, xtask, parity, coverage]

# Dependency graph
requires:
  - phase: 15-01
    provides: tolerance_for_family unified to atol=1e-12, manifest_oracle_families() from lock

provides:
  - oracle-covered-update xtask command that stamps oracle_covered=true after parity confirmation
  - manifest-audit --check-lock validates oracle_covered completeness for all stable entries
  - compiled_manifest.lock.json regenerated with oracle_covered=true on all 110 stable/optional entries

affects:
  - phase 15-03 (manifest lock closure verification)
  - CI manifest_drift_gate (now also gates on oracle_covered completeness)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - oracle-covered-update reads lock, runs generate_profile_parity_report for all 4 profiles, stamps entries
    - check_oracle_coverage in manifest-audit is gated by check_lock flag to avoid noise in non-lock mode

key-files:
  created:
    - xtask/src/oracle_covered_update.rs
  modified:
    - xtask/src/main.rs
    - xtask/src/manifest_audit.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json

key-decisions:
  - "oracle-covered-update stamps helper/transform/optimizer/legacy entries unconditionally as covered because verify_helper_surface_coverage passes as part of generate_profile_parity_report"
  - "manifest-audit check_oracle_coverage only checks stability=stable entries, not optional or unstable_source, matching D-07 intent"
  - "should_fail condition in manifest-audit now includes !uncovered_stable.is_empty() so --check-lock is a hard gate on oracle_covered completeness"

patterns-established:
  - "oracle_covered stamp pattern: call generate_profile_parity_report per profile (bails on any mismatch), collect passing fixture symbols, stamp entries"

requirements-completed: [ORAC-02, ORAC-03]

# Metrics
duration: 7min
completed: 2026-04-06
---

# Phase 15 Plan 02: Oracle-Covered Write-Back and Manifest Lock Closure Summary

**oracle-covered-update xtask stamps all 110 stable/optional manifest lock entries oracle_covered=true after confirming parity across all 4 profiles at atol=1e-12; manifest-audit --check-lock now gates on completeness**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-05T23:56:10Z
- **Completed:** 2026-04-06T00:02:44Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `oracle-covered-update` xtask command that runs oracle parity for all four approved profiles and stamps `oracle_covered=true` on every passing entry in the manifest lock
- Added `check_oracle_coverage` function to `manifest-audit` that gates on `oracle_covered` completeness for all stable entries when `--check-lock` is passed
- Regenerated `compiled_manifest.lock.json` with 110 entries stamped as `oracle_covered=true` (was 0 of 130)
- `manifest-audit --check-lock` now exits 0 with zero uncovered stable entries

## Task Commits

Each task was committed atomically:

1. **Task 1: Create oracle-covered-update xtask and add oracle_covered check to manifest-audit** - `e504be3` (feat)
2. **Task 2: Run oracle-covered-update to stamp the lock and verify manifest-audit passes** - `8c1685b` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `xtask/src/oracle_covered_update.rs` - New xtask sub-command: reads lock, runs 4-profile parity, stamps oracle_covered=true
- `xtask/src/main.rs` - Added OracleCoveredUpdate enum variant, command dispatch, and help text
- `xtask/src/manifest_audit.rs` - Added check_oracle_coverage function, uncovered_stable list in report, should_fail condition updated
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - Regenerated with oracle_covered=true on all 110 stable/optional entries

## Decisions Made
- Helper/transform/optimizer/legacy entries stamped unconditionally because `generate_profile_parity_report` calls `verify_helper_surface_coverage` internally and bails on any mismatch — they are effectively parity-confirmed
- check_oracle_coverage checks only `stability == "stable"` entries per D-07 (not optional or unstable_source)
- `should_fail` uses `!uncovered_stable.is_empty()` so manifest-audit is a hard CI gate against oracle_covered regression

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- oracle_covered=true on all stable entries — ORAC-02 and ORAC-03 closed
- manifest-audit --check-lock is a hard CI gate for oracle coverage regression
- Phase 15-03 can proceed with final verification and manifest lock closure confirmation

## Self-Check

**Files exist:**
- `xtask/src/oracle_covered_update.rs`: FOUND
- `xtask/src/manifest_audit.rs`: FOUND (check_oracle_coverage present)
- `crates/cintx-ops/generated/compiled_manifest.lock.json`: FOUND

**Commits exist:**
- e504be3: FOUND
- 8c1685b: FOUND

## Self-Check: PASSED

---
*Phase: 15-oracle-tolerance-unification-manifest-lock-closure*
*Completed: 2026-04-06*
