---
phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
plan: 05
subsystem: testing
tags: [rust, libcint, int2e, rys, oracle]
requires:
  - phase: 10-01
    provides: host-side Rys roots and c2s transforms required by real 2e execution
provides:
  - Real host-side `int2e_sph` kernel pipeline in `two_electron.rs`
  - Vendor parity coverage for `int2e_sph` over H2O and H2 STO-3G shell quartets
affects: [phase-10-oracle-gates, raw-int2e-compatibility]
tech-stack:
  added: []
  patterns:
    - Host-side 2e G-tensor fill with adaptive ibase/kbase + branch HRR
    - Full shell-quartet vendor parity loop with mixed atol/rtol matching
key-files:
  created:
    - crates/cintx-oracle/tests/two_electron_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/two_electron.rs
key-decisions:
  - "Use `PTR_ENV_START`-aligned env layout in parity fixtures to keep libcint global env slots valid."
  - "Apply missing primitive-pair Gaussian prefactors (`pdata_ij.fac * pdata_kl.fac`) in 2e accumulation for vendor parity."
patterns-established:
  - "2e kernel parity hard gate: vendor comparison across all shell quartets before plan close."
  - "Rule 1 correctness fix during parity bring-up is committed in-task and documented in summary."
requirements-completed: [KERN-02, VERI-05]
duration: 8min
completed: 2026-04-03
---

# Phase 10 Plan 05: 2e Kernel + Oracle Parity Summary

**Host-side `int2e_sph` now computes non-zero four-center ERIs via Rys quadrature and matches vendored libcint for H2O/H2 STO-3G at `atol=1e-12`, `rtol=1e-10`.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-03T11:47:00Z
- **Completed:** 2026-04-03T11:55:35Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced the `two_electron.rs` stub with a full host-side 2e kernel pipeline (`fill_g_tensor_2e` + HRR branches + Cartesian contraction + `cart_to_sph_2e`).
- Added `two_electron_parity.rs` with full vendor oracle parity loops for H2O STO-3G (`5^4` quartets) and H2 STO-3G (`2^4` quartets).
- Verified the kernel against vendored libcint with strict mixed tolerance checks (`1e-12` absolute, `1e-10` relative) and non-zero output assertions.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement 2e ERI kernel — full Rys quadrature G-fill with 4 centers** - `0c10aee` (feat)
2. **Task 2: Add 2e oracle parity test against vendored libcint** - `a8bdd1b` (fix)

## Files Created/Modified

- `crates/cintx-cubecl/src/kernels/two_electron.rs` - real 2e kernel implementation and primitive-pair prefactor fix.
- `crates/cintx-oracle/tests/two_electron_parity.rs` - vendor parity tests for H2O/H2 with mixed atol/rtol comparison.

## Decisions Made

- Kept adaptive `ibase`/`kbase` branch handling explicit in `two_electron.rs` to preserve libcint stride semantics.
- Used `PTR_ENV_START`-offset fixture construction in oracle tests to avoid corrupting libcint global env slots.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added missing primitive pair Gaussian prefactors in 2e accumulation**
- **Found during:** Task 2 (vendor parity execution)
- **Issue:** Vendor parity failed; `int2e_sph` values were systematically high because quartet accumulation omitted `exp(-ai*aj/aij*|A-B|^2)` and `exp(-ak*al/akl*|C-D|^2)` factors.
- **Fix:** Multiplied primitive quartet factor by `compute_pdata_host(...).fac` for both ij and kl pairs.
- **Files modified:** `crates/cintx-cubecl/src/kernels/two_electron.rs`
- **Verification:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu -- two_electron --test-threads=1`
- **Committed in:** `a8bdd1b` (task commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 bug)
**Impact on plan:** Correctness-critical fix required for oracle parity; no scope creep beyond plan objective.

## Issues Encountered

- Initial vendor parity run failed with large mismatch counts for both H2 and H2O shell quartets.
- Root cause was traced to missing primitive-pair prefactors in the 2e kernel accumulation path; fixed and re-verified.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `int2e_sph` now has a strict oracle gate and passing parity coverage, ready for downstream gate closure work.
- Remaining risk is performance tuning only; correctness gate for this plan is satisfied.

## Self-Check: PASSED

- Found summary file: `.planning/phases/10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure/10-05-SUMMARY.md`
- Found key artifacts: `crates/cintx-cubecl/src/kernels/two_electron.rs`, `crates/cintx-oracle/tests/two_electron_parity.rs`
- Verified commits exist: `0c10aee`, `a8bdd1b`

---
*Phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure*
*Completed: 2026-04-03*
