---
phase: 04-verification-release-automation
plan: 02
subsystem: infra
tags: [xtask, oracle, manifest-lock, ci-gates, oom]
requires:
  - phase: 04-verification-release-automation-01
    provides: profile-aware oracle fixture/report APIs for gate runners
provides:
  - executable xtask command router for VERI-02 gate commands
  - lock-driven manifest drift/profile coverage audit report
  - oracle/helper/OOM gate runners with persisted artifacts
affects: [ci, phase-04-plan-03, verification]
tech-stack:
  added: [xtask+cintx-compat(with-f12,with-4c1e)]
  patterns: [fail-closed gate CLI, profile-scoped artifact persistence, /mnt/data fallback metadata]
key-files:
  created:
    - .planning/phases/04-verification-release-automation/deferred-items.md
  modified:
    - xtask/Cargo.toml
    - xtask/Cargo.lock
    - xtask/src/main.rs
    - xtask/src/manifest_audit.rs
    - xtask/src/oracle_update.rs
key-decisions:
  - "Keep gate commands fail-closed and return non-zero on drift/parity/OOM regressions."
  - "Scope manifest lock-vs-generated symbol diff to oracle operator/source rows to avoid helper/legacy noise."
  - "Persist profile-specific oracle artifacts even when individual profile parity fails."
patterns-established:
  - "Pattern: command entrypoint parses deterministic profile matrix defaults and delegates to gate modules."
  - "Pattern: artifact writers attempt /mnt/data first, then CINTX_ARTIFACT_DIR fallback with metadata."
requirements-completed: [VERI-02]
duration: 21m
completed: 2026-03-28
---

# Phase 4 Plan 02: xtask Gate Surface Summary

**Manifest audit, oracle parity, helper/legacy parity, and OOM contract checks are now runnable from one xtask CLI surface with profile-scoped reports and fail-closed exits.**

## Performance

- **Duration:** 21m
- **Started:** 2026-03-28T10:58:54Z
- **Completed:** 2026-03-28T11:20:16Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Replaced xtask stub entrypoint with concrete gate command contracts for `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, and `oom-contract-check`.
- Implemented `run_manifest_audit` with strict profile-scope enforcement and lock-vs-generated symbol diff output at `/mnt/data/cintx_phase_04_manifest_audit.json` (with fallback metadata).
- Implemented oracle/helper/OOM gate runners, including exact OOM regression invocations and per-profile oracle artifact persistence.

## Task Commits

1. **Task 1: Replace xtask stubs with concrete gate CLI contracts** - `5805d4a` (feat)
2. **Task 2: Implement lock-driven manifest drift and profile coverage audit command** - `d0d0c09` (feat)
3. **Task 3: Implement oracle parity, helper/legacy parity, and OOM contract gate runners** - `b26d270` (feat)

## Files Created/Modified

- `xtask/Cargo.toml` - Added xtask gate dependencies, optional-profile compat feature activation, and standalone workspace table.
- `xtask/Cargo.lock` - Captured deterministic dependency resolution for xtask.
- `xtask/src/main.rs` - Added subcommand parser/router with deterministic profile defaults and fail-closed exit handling.
- `xtask/src/manifest_audit.rs` - Added manifest lock scope validation, profile coverage diffing, and machine-readable audit report emission.
- `xtask/src/oracle_update.rs` - Added oracle compare/profile artifact persistence, helper parity gate, and explicit OOM contract test runner.
- `.planning/phases/04-verification-release-automation/deferred-items.md` - Logged out-of-scope optional-profile parity drift surfaced by fail-closed oracle gate.

## Decisions Made

- Added an empty `[workspace]` table to `xtask/Cargo.toml` so `cargo run --manifest-path xtask/Cargo.toml` works from the repo root without workspace-membership failure.
- Kept `oracle-compare` fail-closed even when optional profiles currently report mismatches; this preserves D-08 gate semantics.
- Activated `with-f12` and `with-4c1e` for xtask’s compat graph to ensure optional-profile execution paths are available during gate runs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed xtask workspace boundary for direct manifest-path execution**
- **Found during:** Task 1
- **Issue:** `cargo run --manifest-path xtask/Cargo.toml` failed because xtask was under the repo workspace root but not declared as a workspace member.
- **Fix:** Added empty `[workspace]` to `xtask/Cargo.toml`.
- **Files modified:** `xtask/Cargo.toml`
- **Verification:** Task 1 `--help` command succeeded after the fix.
- **Committed in:** `5805d4a`

**2. [Rule 1 - Bug] Corrected manifest audit diff scope to oracle operator/source symbols**
- **Found during:** Task 2
- **Issue:** Initial lock diff included helper/transform/optimizer rows, causing false `missing_in_generated` drift.
- **Fix:** Scoped lock-symbol comparison to Resolver operator/source rows in approved oracle families.
- **Files modified:** `xtask/src/manifest_audit.rs`
- **Verification:** `manifest-audit --check-lock` passed with zero false drift.
- **Committed in:** `d0d0c09`

**3. [Rule 3 - Blocking] Enabled optional-profile compat features in xtask dependency graph**
- **Found during:** Task 3
- **Issue:** Optional-profile oracle checks ran without optional compat features in the xtask graph, producing avoidable execution mismatches.
- **Fix:** Added direct xtask dependency on `cintx-compat` with `with-f12` and `with-4c1e` features.
- **Files modified:** `xtask/Cargo.toml`, `xtask/Cargo.lock`
- **Verification:** Optional-profile mismatch counts reduced and profile artifacts persisted correctly.
- **Committed in:** `b26d270`

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 bug)
**Impact on plan:** All deviations were correctness/operability fixes needed to make the gate command surface executable and auditable.

## Issues Encountered

- `oracle-compare` remains fail-closed on existing optional-profile parity drift (`with-f12`, `with-4c1e`, `with-f12+with-4c1e`) and exits non-zero by design.
- The out-of-scope parity deltas were logged to `.planning/phases/04-verification-release-automation/deferred-items.md`.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03 can wire CI required checks directly to the implemented xtask gates.
- Deferred optional-profile parity deltas should be addressed in a dedicated follow-up before strict merge enforcement for those profiles.

## Self-Check: PASSED

- FOUND: `.planning/phases/04-verification-release-automation/02-PLAN-SUMMARY.md`
- FOUND: `.planning/phases/04-verification-release-automation/deferred-items.md`
- FOUND commits: `5805d4a`, `d0d0c09`, `b26d270`

---
*Phase: 04-verification-release-automation*
*Completed: 2026-03-28*
