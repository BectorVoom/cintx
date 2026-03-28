---
phase: 04-verification-release-automation
plan: 03
subsystem: infra
tags: [ci, github-actions, xtask, verification, profile-matrix]
requires:
  - phase: 04-verification-release-automation-02
    provides: executable xtask gate commands for manifest, oracle, helper/legacy, and OOM checks
provides:
  - concrete CI gate templates for oracle and feature matrix verification
  - required PR workflow jobs for merge-blocking manifest/oracle/helper/OOM gates
  - mandatory profile set enforcement across PR governance automation
affects: [ci, branch-protection, veri-02]
tech-stack:
  added: []
  patterns: [non-fail-fast evidence matrix, required gate job naming, toolchain pin from rust-toolchain.toml]
key-files:
  created: []
  modified:
    - ci/oracle-compare.yml
    - ci/feature-matrix.yml
    - .github/workflows/compat-governance-pr.yml
key-decisions:
  - "Keep all required PR gates as explicit top-level jobs: manifest_drift_gate, oracle_parity_gate, helper_legacy_parity_gate, and oom_contract_gate."
  - "Resolve the Rust channel from rust-toolchain.toml inside each required job to avoid toolchain drift."
  - "Run helper/legacy and OOM checks across all required profiles via deterministic profile loops."
patterns-established:
  - "Pattern: CI templates under ci/ define workflow_call profile matrices with fail-fast disabled for full evidence collection."
  - "Pattern: PR governance wiring calls xtask commands directly through cargo run --manifest-path xtask/Cargo.toml -- <gate>."
requirements-completed: [VERI-02]
duration: 2m
completed: 2026-03-28
---

# Phase 4 Plan 03: PR Governance Gate Wiring Summary

**Merge-blocking PR automation now runs concrete xtask manifest, oracle parity, helper/legacy parity, and OOM gate commands across the required feature-profile envelope.**

## Performance

- **Duration:** 2m
- **Started:** 2026-03-28T20:24:50+09:00
- **Completed:** 2026-03-28T11:26:44Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Replaced CI template stubs with executable profile-matrix templates in `ci/oracle-compare.yml` and `ci/feature-matrix.yml`.
- Enforced the required profile set (`base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`) with `fail-fast: false` for evidence-complete runs.
- Rewired `.github/workflows/compat-governance-pr.yml` to four explicit required jobs invoking only Phase 4 xtask gates.

## Task Commits

1. **Task 1: Replace CI template stubs with concrete Phase 4 gate templates** - `f422d26` (feat)
2. **Task 2: Rewire PR governance workflow to enforce required merge-blocking gates** - `562526c` (feat)

## Files Created/Modified

- `ci/oracle-compare.yml` - Added workflow-call oracle gate template with profile matrix, non-fail-fast execution, helper/legacy parity invocation, and artifact upload paths.
- `ci/feature-matrix.yml` - Added workflow-call manifest/oom gate template with required profile matrix and non-fail-fast execution.
- `.github/workflows/compat-governance-pr.yml` - Replaced obsolete phase-3 checks with required xtask governance jobs and pinned-toolchain setup from `rust-toolchain.toml`.

## Decisions Made

- Kept the required profile set centralized as `CINTX_REQUIRED_PROFILES` in PR governance to avoid drift across job commands.
- Executed `helper-legacy-parity` and `oom-contract-check` per profile using explicit loops so every required profile is exercised.
- Kept all required jobs fail-closed with no `continue-on-error` bypass.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 4 Plan 04 can build on these required PR checks for release and verification closure.
- Branch protection can map required checks directly to the new gate job names.

## Self-Check: PASSED

- FOUND: `.planning/phases/04-verification-release-automation/03-PLAN-SUMMARY.md`
- FOUND commit: `f422d26`
- FOUND commit: `562526c`

---
*Phase: 04-verification-release-automation*
*Completed: 2026-03-28*
