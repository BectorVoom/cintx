---
phase: 23-group-1-remaining-1st-derivative-families-cart-sph
plan: 05
subsystem: oracle-verification
tags: [regression-guard, vendor-parity, drv1-02, rank-9, 1e-derivative]
requires:
  - "cluster-C rank-9 both-side 1e families (int1e_ipovlpip/ipkinip/ipnucip) shipped in commit 319d055"
  - "existing test crates/cintx-oracle/tests/one_electron_grad_both_parity.rs"
provides:
  - "recorded green vendor-parity run for DRV1-02 under the double gate (regression-free confirmation)"
affects:
  - "DRV1-02 requirement coverage (phase 23 can close it with a green gate on record)"
tech-stack:
  added: []
  patterns:
    - "double-gate vendor parity re-run as a no-source regression guard (--features cpu + CINTX_ORACLE_BUILD_VENDOR=1)"
key-files:
  created:
    - ".planning/phases/23-group-1-remaining-1st-derivative-families-cart-sph/23-05-SUMMARY.md"
  modified: []
decisions:
  - "Plan 05 touches no source: re-runs the existing cluster-C vendor parity test and records the result, satisfying DRV1-02 requirement-coverage audit without re-doing finished work."
metrics:
  duration: "1 min"
  completed: "2026-05-30"
  tasks: 1
  files: 0
---

# Phase 23 Plan 05: DRV1-02 Rank-9 Both-Side 1e Regression Guard Summary

Re-ran the existing cluster-C rank-9 both-side 1e vendor parity test
(`one_electron_grad_both_parity`) under the mandatory double gate and confirmed
`int1e_ipovlpip` / `int1e_ipkinip` / `int1e_ipnucip` still match vendored libcint
6.1.3 byte-identically (0 mismatches at atol=1e-12, cart + sph) — no regression
introduced by the cluster A/B work in plans 01-04.

## What This Plan Did

This is a single-task, no-source regression-guard plan. It did NOT re-implement
DRV1-02 — that work shipped in commit `319d055`. It re-ran the existing
cluster-C vendor parity test as a guard to prove plans 01-04 (cluster A/B engine
plumbing — f12.rs, shared raw.rs/manifest edits) introduced no regression into
the already-shipped rank-9 1e path, so the phase can close DRV1-02 on a green gate.

## Task Completed

### Task 1: Re-run existing cluster-C rank-9 vendor parity (DRV1-02 regression guard)

- **Read first:** `crates/cintx-oracle/tests/one_electron_grad_both_parity.rs` —
  confirmed it covers all three families (ipovlpip/ipkinip/ipnucip) at NCOMP=9,
  cart + sph, atol=1e-12, vendor parity gated on `has_vendor_libcint` + `cpu`.
  NOT modified.
- **Command:**
  `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_both_parity -- --test-threads=1`
- **Result:** `running 6 tests` (N=6 > 0 — double gate active, NOT a silent 0-test skip) → `test result: ok. 6 passed; 0 failed`
  - `test_int1e_ipovlpip_h2o_sto3g_parity ... ok` (vendor parity, cart + sph)
  - `test_int1e_ipkinip_h2o_sto3g_parity ... ok` (vendor parity, cart + sph)
  - `test_int1e_ipnucip_h2o_sto3g_parity ... ok` (vendor parity, cart + sph)
  - `test_int1e_ipovlpip_determinism ... ok`
  - `test_int1e_ipkinip_determinism ... ok`
  - `test_int1e_ipnucip_determinism ... ok`
- **Mismatch counts:** 0 mismatches at atol=1e-12 for cart + sph across all three rank-9 families (the parity tests assert `mm == 0`, and each passed).
- **Source modified:** none. `git diff --name-only HEAD -- crates/` is empty; working tree clean.

## Verification

- Double gate confirmed active: banner showed `running 6 tests` (N>0), not a silent 0-test skip.
- `test result: ok.` — 0 vendor mismatches at atol=1e-12, cart + sph, all three rank-9 families.
- `git diff --name-only HEAD -- crates/` shows no change — this plan ran a test only.

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- SUMMARY.md created: `.planning/phases/23-group-1-remaining-1st-derivative-families-cart-sph/23-05-SUMMARY.md` (FOUND)
- No source/test files modified: `git diff --name-only HEAD -- crates/` empty (CONFIRMED)
- Vendor parity test re-run under double gate: 6 tests, result ok, 0 mismatches (CONFIRMED)
