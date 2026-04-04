---
phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
plan: 04
subsystem: testing
tags: [rust, cubecl, libcint, rys, 3c2e, oracle]
requires:
  - phase: 10-01
    provides: real 2e-style recurrence foundations and c2s transform wiring used by 3c2e
provides:
  - real host-side 3c2e kernel with Rys fill, ij split, Cartesian contraction, and spherical transform
  - vendored libcint parity coverage for H2O STO-3G int3c2e at atol 1e-9
  - corrected ij-HRR behavior for li/lj branch asymmetry in 3c2e
affects: [oracle-gates, kernel-parity, 3c2e]
tech-stack:
  added: []
  patterns:
    - ibase-style ij handling via canonical li>=lj evaluation plus transpose-back
    - 2e-family oracle fixtures reserve libcint env globals with PTR_ENV_START
key-files:
  created:
    - crates/cintx-oracle/tests/center_3c2e_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
key-decisions:
  - "Use canonical li>=lj execution for 3c2e and transpose outputs back to caller shell order."
  - "Use PTR_ENV_START in 3c2e oracle fixtures to prevent libcint global env slot corruption."
patterns-established:
  - "3c2e ij split follows ibase HRR transfer (j-ladder) instead of direct combined-index remap."
  - "3c2e oracle parity fixtures mirror 2e-family env layout safeguards."
requirements-completed: [KERN-05, VERI-05]
duration: 8m
completed: 2026-04-03
---

# Phase 10 Plan 04: 3c2e Kernel and Oracle Parity Summary

**Real int3c2e host-kernel evaluation now matches vendored libcint 6.1.3 on H2O STO-3G at atol 1e-9, with explicit ibase-safe ij handling.**

## Performance

- **Duration:** 8m
- **Started:** 2026-04-03T11:51:35Z
- **Completed:** 2026-04-03T12:00:02Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Replaced the 3c2e stub pipeline with a working Rys/VRR/HRR/contraction/c2s host implementation.
- Added `center_3c2e` oracle parity tests for H2O STO-3G including vendored libcint comparison at `1e-9`.
- Closed parity gaps by fixing ij-HRR split behavior and li/lj ordering behavior for asymmetric shell pairs.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement 3c2e kernel — Rys-based G-fill with 2e machinery reuse** - `4f9612d` (feat)
2. **Task 2: Add 3c2e oracle parity test against vendored libcint** - `5fce2a0` (fix)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` - real 3c2e kernel, ibase-safe ij split, canonical li/lj handling, and output transpose-back path.
- `crates/cintx-oracle/tests/center_3c2e_parity.rs` - H2O STO-3G nonzero/idempotency and vendored parity coverage for `int3c2e`.

## Decisions Made
- Canonicalized runtime evaluation to `li >= lj` and transposed outputs back to original shell order to match libcint ibase behavior consistently.
- Corrected ij split to use j-HRR transfer from the ij-base ladder.
- Reserved libcint env global slots in the oracle fixture (`PTR_ENV_START`) for 2e-family correctness.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed 2e-family env slot corruption in 3c2e parity fixture**
- **Found during:** Task 2 (oracle parity verification)
- **Issue:** Fixture data started at `env[0]`, corrupting libcint reserved global slots and producing non-Coulomb reference behavior.
- **Fix:** Updated fixture initialization to start payload at `PTR_ENV_START`.
- **Files modified:** `crates/cintx-oracle/tests/center_3c2e_parity.rs`
- **Verification:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu -- center_3c2e --test-threads=1`
- **Committed in:** `5fce2a0` (part of Task 2 commit)

**2. [Rule 1 - Bug] Fixed ij-HRR transfer for asymmetric 3c2e shell pairs**
- **Found during:** Task 2 (vendor parity mismatch isolation)
- **Issue:** ij split logic incorrectly remapped combined indices and mis-handled `lj=0`/asymmetric-center cases, causing sign/magnitude parity failures.
- **Fix:** Implemented ibase-style j-HRR transfer and canonical `li>=lj` execution with transpose-back to original shell order.
- **Files modified:** `crates/cintx-cubecl/src/kernels/center_3c2e.rs`
- **Verification:** `cargo test -p cintx-cubecl --features cpu -- center_3c2e --test-threads=1` and oracle parity command above.
- **Committed in:** `5fce2a0` (part of Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both fixes were required to satisfy the plan’s correctness/parity acceptance criteria.

## Issues Encountered
- None remaining after parity-driven kernel corrections.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- 3c2e kernel and oracle parity gate are green at required tolerance.
- No additional manual verification or setup is needed for this plan.

## Self-Check: PASSED
- FOUND: `.planning/phases/10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure/10-04-SUMMARY.md`
- FOUND: `4f9612d`
- FOUND: `5fce2a0`

---
*Phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure*
*Completed: 2026-04-03*
