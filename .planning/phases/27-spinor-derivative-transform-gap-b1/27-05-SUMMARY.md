---
phase: 27-spinor-derivative-transform-gap-b1
plan: 05
subsystem: testing
tags: [oracle-coverage, manifest-lock, spinor, vendor-parity, no-silent-skip]

# Dependency graph
requires:
  - phase: 27-01
    provides: D-08 adversarial spinor fixture + RED spinor_deriv_parity scaffold
  - phase: 27-02
    provides: cart_to_spinor_sf_derivative_2d/_3c2e wrappers (ncomp fold, internal transpose)
  - phase: 27-02a
    provides: SPHERICAL aux-k reconciliation (360-element buffer, not 720)
  - phase: 27-03
    provides: sf_2d launcher rewire + 1e vendor parity GREEN (ranks 3/9/81)
  - phase: 27-04
    provides: sf_3c2e launcher rewire + int3c2e_ip1 vendor parity GREEN (360-element buffer)
provides:
  - oracle_covered=true recorded for 20 vendor-backed sf-derivative spinor families
  - completed D-10 no-silent-skip manifest-coverage assertion (FLIPPED/DEFERRED split)
  - D-12 vendor-stub deferral note in xtask/src/oracle_covered_update.rs
  - manifest-audit green; full vendor parity suite green under both gate flags
affects: [phase-28-gap-b2-c2s-si, phase-29-relativistic-sigma, FND-04-closure, follow-up-FD-verification-4-stub-arms]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Coverage-flip records (does not itself verify) parity proven in prior waves; depends_on the parity-proving plans is load-bearing"
    - "no-silent-skip assertion reads runtime MANIFEST_ENTRIES (build.rs-regenerated from the lock) to defend the exact flipped/deferred split, gated under the live-vendor cfg"

key-files:
  created:
    - .planning/phases/27-spinor-derivative-transform-gap-b1/27-05-SUMMARY.md
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs (build.rs-regenerated)
    - crates/cintx-ops/src/generated/api_manifest.csv (build.rs-regenerated)
    - xtask/src/oracle_covered_update.rs
    - crates/cintx-oracle/tests/spinor_deriv_parity.rs

key-decisions:
  - "Flip oracle_covered=true for EXACTLY the 20 vendor-backed sf-derivative spinor families (18 arity-2 1e sf_2d ranks 3/9/27/81 + int3c2e_ip1/ip2 sf_3c2e rank-3); the 4 D-12 vendor-stub arms + 6 D-03 int2e + 1 D-04 ECP stay false"
  - "Verify each component_rank against the rank-tier table BEFORE flipping (rank guard in the flip script); no rank edits made (all already correct)"
  - "Record D-12 in the oracle_covered_update.rs deferral note (vendor stubs return 0 / exit(1), no byte-identity reference, FD verification deferred); retain the if fixture.skipped {continue;} guard"
  - "test_no_silent_skip reads cintx_ops::generated::MANIFEST_ENTRIES at runtime and asserts FLIPPED=true / DEFERRED=false, gated under the vendor-cpu cfg so the claim ties to a live vendor run"

patterns-established:
  - "Pattern: Coverage-flip plan with a runtime manifest-coverage assertion that fails (not skips) on flip-propagation failure or accidental deferred-family stamping"

requirements-completed: [FND-04]

# Metrics
duration: 9min
completed: 2026-05-31
---

# Phase 27 Plan 05: Spinor-Derivative Coverage Flip + No-Silent-Skip Assertion Summary

**Flipped oracle_covered=true for the 20 vendor-backed sf-derivative spinor families (18 arity-2 1e sf_2d ranks 3/9/27/81 + int3c2e_ip1/ip2 sf_3c2e rank-3) and completed the D-10 no-silent-skip manifest-coverage assertion that defends the exact flipped-true / deferred-false split; the 4 D-12 vendor-stub arms + D-03/D-04 deferrals stay false.**

## Performance

- **Duration:** ~9 min
- **Tasks:** 2
- **Files modified:** 5 (2 of which are build.rs-regenerated artifacts)

## Accomplishments
- Flipped `oracle_covered` false→true for exactly the 20 vendor-backed sf-derivative spinor families in `compiled_manifest.lock.json`, with each `component_rank` verified against the rank-tier table (3/9/27/81) before the flip — no rank edits required.
- Kept the 11 deferred families false: the 4 D-12 vendor-stub arms (int2c2e_ip1/ip2_spinor, int3c1e_ip1/iprinv_spinor), the 6 D-03 arity-4 int2e_ip* families, and the 1 D-04 int1e_ecp_iprinv_spinor.
- Updated the deferral note in `xtask/src/oracle_covered_update.rs` to record D-12 (libcint 6.1.3 ships those 4 arms as `return 0` / `exit(1)` stubs, so no byte-identity reference is achievable; FD verification is a deferred follow-up); retained the `if fixture.skipped { continue; }` guard.
- Completed the D-10 `test_no_silent_skip` assertion: added `FLIPPED` (20) and `DEFERRED` (11) const arrays and a runtime loop over `cintx_ops::generated::MANIFEST_ENTRIES` asserting every flipped family reads true and every deferred family reads false, retaining the existing live-vendor / not-skipped nonzero-output asserts.
- `manifest-audit` green; full vendor parity suite green under both gate flags: 6 passed, 0 failed, 3 ignored (the D-12 arm tests stay #[ignore]'d), `running 9 tests`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Flip oracle_covered=true for the 20 vendor-backed families + update D-12 deferral note + manifest-audit green** — `e9e5f63` (feat)
2. **Task 2: Complete the D-10 no-silent-skip assertion (flipped/deferred split) + run the full vendor parity suite** — `688a08e` (test)

## Files Created/Modified
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — oracle_covered flipped true for the 20 vendor-backed families; false retained for the 11 deferred families (exactly 20 lines: false→true).
- `crates/cintx-ops/src/generated/api_manifest.rs` / `.csv` — build.rs-regenerated MANIFEST_ENTRIES reflecting the flip (verified int1e_ipovlp_spinor=true, int3c2e_ip2_spinor=true, int3c1e_ip1_spinor=false).
- `xtask/src/oracle_covered_update.rs` — deferral note updated to record D-12; skipped guard retained.
- `crates/cintx-oracle/tests/spinor_deriv_parity.rs` — FLIPPED/DEFERRED const arrays + runtime MANIFEST_ENTRIES coverage assertion added to test_no_silent_skip.

## Decisions Made
None beyond the plan as specified — the plan's interfaces block enumerated the exact 20 flip symbols and 11 deferred symbols and the rank-tier table; all were confirmed correct before flipping.

## Deviations from Plan

None - plan executed exactly as written.

The plan's "all 20 component_rank already correct" claim was verified true (no rank edits needed). All 20 flip families were `oracle_covered=false` before the flip; all 11 deferred families remain `false`.

## Issues Encountered
None. The pre-existing unrelated baseline failure `test_f32_int3c2e_sph_parity` did not surface in the `spinor_deriv_parity` suite (it lives in a different suite), and the dead-code / snake-case warnings observed during builds are pre-existing and out of scope.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 27 success criterion 3 (vendor parity executes under both flags, not skipped; manifest-audit green; no capi/legacy surface added) is satisfied.
- FND-04 (Gap B1 sf-derivative spinor transform) is recorded as covered for the 20 vendor-backed families.
- Follow-up: finite-difference verification of the 4 D-12 vendor-stub arms (deferred — libcint has no working driver) remains the outstanding item to eventually flip those arms under an FD-tolerance gate distinct from the vendor byte-identity gate. D-03 (arity-4 int2e_ip* spinor, needs sf_4d derivative wrapper) and D-04 (int1e_ecp_iprinv_spinor, relativistic track) are deferred to follow-up phases.

## Self-Check: PASSED

- FOUND: `.planning/phases/27-spinor-derivative-transform-gap-b1/27-05-SUMMARY.md`
- FOUND: commit `e9e5f63` (Task 1)
- FOUND: commit `688a08e` (Task 2)

---
*Phase: 27-spinor-derivative-transform-gap-b1*
*Completed: 2026-05-31*
