---
phase: 15-oracle-tolerance-unification-manifest-lock-closure
plan: "03"
subsystem: CI / xtask oracle gates
tags: [ci, matrix, oracle, xtask, github-actions]
dependency_graph:
  requires: ["15-02"]
  provides: ["matrix oracle_parity_gate", "single-profile oracle-compare"]
  affects: [".github/workflows/compat-governance-pr.yml", "xtask/src/oracle_update.rs"]
tech_stack:
  added: []
  patterns: ["GitHub Actions matrix strategy", "fail-fast: false per-profile reporting"]
key_files:
  created: []
  modified:
    - xtask/src/oracle_update.rs
    - .github/workflows/compat-governance-pr.yml
decisions:
  - "Accept any non-empty subset of standard profiles in validate_required_profile_scope; CI matrix covers full coverage across parallel jobs"
  - "Use fail-fast: false so all four profile jobs report independently even when one fails"
metrics:
  duration: 2
  completed_date: "2026-04-06"
requirements: [ORAC-03]
---

# Phase 15 Plan 03: Oracle CI Matrix Strategy Summary

**One-liner:** Parallelized oracle_parity_gate across four profiles via GitHub Actions matrix with single-profile xtask invocations accepted.

## What Was Built

Switched the `oracle_parity_gate` CI job from a single sequential invocation covering all four profiles to a GitHub Actions matrix strategy that runs each profile as a parallel job. Complemented this by relaxing the `validate_required_profile_scope` function in xtask to accept any non-empty subset of standard profiles instead of requiring all four simultaneously.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Relax oracle-compare profile validation to accept single profiles | 0678abf | xtask/src/oracle_update.rs |
| 2 | Update CI workflow to matrix strategy for oracle_parity_gate | 70bd798 | .github/workflows/compat-governance-pr.yml |

## Key Changes

### xtask/src/oracle_update.rs

- Removed the `BTreeSet` difference check that required all four standard profiles to be passed together
- Removed the "profile scope mismatch" bail that blocked single-profile invocations
- New logic accepts any non-empty subset of standard profiles; CI matrix ensures full coverage across parallel jobs
- Preserved the unstable-source standalone enforcement (D-02)
- Removed now-unused `PHASE4_APPROVED_PROFILES` import and `std::collections::BTreeSet` import

### .github/workflows/compat-governance-pr.yml

- Added `strategy.fail-fast: false` to `oracle_parity_gate` job so all four profile jobs report independently
- Added `matrix.profile` with four values: `[base, with-f12, with-4c1e, "with-f12+with-4c1e"]` (quoted due to `+`)
- Updated job name to `oracle_parity_gate (${{ matrix.profile }})` for per-profile identification
- Changed `--profiles "${CINTX_REQUIRED_PROFILES}"` to `--profiles "${{ matrix.profile }}"` for single-profile per job
- All other jobs (`manifest_drift_gate`, `helper_legacy_parity_gate`, `oom_contract_gate`) left unchanged

## Verification

- `cargo check --manifest-path xtask/Cargo.toml` exits 0 with no warnings
- `oracle_parity_gate` job contains `matrix:`, `fail-fast: false`, four profile values, and `${{ matrix.profile }}` in the run step
- All other CI gates remain unchanged

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused imports after removing BTreeSet diff validation**
- **Found during:** Task 1
- **Issue:** Removing the BTreeSet validation left `PHASE4_APPROVED_PROFILES` and `std::collections::BTreeSet` as unused imports causing compiler warnings
- **Fix:** Removed both unused imports from the use declarations
- **Files modified:** xtask/src/oracle_update.rs
- **Commit:** 0678abf

## Known Stubs

None — both changes are complete functional implementations.

## Self-Check: PASSED
