---
phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
plan: 02
subsystem: kernels
tags: [rys-quadrature, 2c2e, libcint, oracle, env-layout, PTR_ENV_START]

# Dependency graph
requires:
  - phase: 10-01
    provides: rys_roots_host dispatcher, cart_to_sph_2c2e, oracle vendor build with int2c2e_sph FFI

provides:
  - Real 2c2e kernel in center_2c2e.rs: fill_g_tensor_2c2e + launch_center_2c2e
  - Oracle parity test center_2c2e_parity.rs passing vs vendored libcint 6.1.3 at atol=1e-9
  - PTR_ENV_START=20 constant exported from cintx-compat/raw.rs with documented convention

affects:
  - All future parity tests: env must start user data at PTR_ENV_START=20 to avoid corrupting PTR_RANGE_OMEGA
  - 10-04 (int2e kernel and parity test) — same env layout fix applies
  - 10-05 (int3c2e) — same env layout fix needed

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2c2e Rys G-tensor fill: fac_env = common_factor * ci * ck (NO exponential); gz = w[irys] * fac1"
    - "libcint env[0..19] reserved for global params (PTR_EXPCUTOFF, PTR_RANGE_OMEGA, etc.); user data starts at PTR_ENV_START=20"
    - "G-tensor layout: g[axis*g_size + k*dm + i*dn + root]; three-axis flat array [gx|gy|gz]"
    - "VRR for 2c2e: c00=-tmp2*(ri-rk), c0p=tmp3*(ri-rk) because rijrx=rklrx=0 for 2-center integral"

key-files:
  created:
    - crates/cintx-oracle/tests/center_2c2e_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/center_2c2e.rs
    - crates/cintx-compat/src/raw.rs

key-decisions:
  - "env user data MUST start at PTR_ENV_START=20; any data placed at env[8] is misread as PTR_RANGE_OMEGA (omega for range-separated Coulomb), causing wrong 2e+ integral values"
  - "2c2e kernel is correct — the parity failure was entirely in the test data, not the kernel"
  - "common_factor = PI^3*2/SQRTPI * fac_sp(li) * fac_sp(lk) already includes fac_sp; no separate post-multiply needed for 2c2e"

patterns-established:
  - "env layout for 2e+ parity tests: always initialize env=vec![0.0; PTR_ENV_START] then push user data"

requirements-completed: [KERN-03, VERI-05]

# Metrics
duration: ~196min (including extended debugging session to find PTR_RANGE_OMEGA env collision)
completed: 2026-04-03
---

# Phase 10 Plan 02: 2c2e Kernel and Oracle Parity Summary

**Real Rys quadrature 2c2e kernel (fill_g_tensor_2c2e + VRR) passing vendor libcint 6.1.3 oracle parity at atol=1e-9 for H2O STO-3G after fixing PTR_RANGE_OMEGA env collision in test data**

## Performance

- **Duration:** ~196 min (includes extended root-cause investigation)
- **Started:** 2026-04-03T08:00:00Z
- **Completed:** 2026-04-03T11:16:02Z
- **Tasks:** 2
- **Files modified:** 3
- **Files created:** 1

## Accomplishments
- Implemented `fill_g_tensor_2c2e`: G-tensor VRR fill for s/p/d shells following libcint `g2e.c` CINTg0_2e + CINTg0_2e_2d algorithms
- Implemented `launch_center_2c2e`: full primitive loop + contraction + cart_to_sph_2c2e pipeline
- Created `center_2c2e_parity.rs` with idempotency and vendor oracle parity tests — both PASS
- Diagnosed and fixed critical env layout bug: PTR_RANGE_OMEGA=env[8] was being set to 1.1078 (H2 z-coord) causing libcint to compute range-separated Coulomb instead of standard Coulomb
- Exported `PTR_ENV_START=20` from `cintx-compat/raw.rs` to prevent recurrence in future tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement real 2c2e kernel with Rys quadrature G-tensor fill** - `90a0eee` (feat)
2. **Task 2: Add 2c2e oracle parity test and fix PTR_ENV_START env layout** - `8cbdfc7` (feat)

**Plan metadata:** (see final metadata commit)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` — Real 2c2e kernel: fill_g_tensor_2c2e (VRR G-tensor fill), launch_center_2c2e (primitive loop + contraction + c2s), unit tests for s-s and p-p
- `crates/cintx-oracle/tests/center_2c2e_parity.rs` — Idempotency test + vendor libcint 6.1.3 parity test for H2O STO-3G (all 25 shell pairs, atol=1e-9)
- `crates/cintx-compat/src/raw.rs` — Added `PTR_ENV_START = 20` constant with documentation of libcint reserved env layout

## Decisions Made

- **PTR_ENV_START=20 is mandatory for 2e+ integrals**: libcint reads `env[PTR_RANGE_OMEGA=8]` to decide whether to apply range separation. All test data must zero env[0..20] before placing user data. This is the root cause of the 4-29% parity failure.
- **Kernel algorithm is correct as committed**: The G-tensor VRR, fac1 formula, and c00/c0p displacement formulas all match libcint source. No changes were needed to the kernel itself.
- **common_factor includes fac_sp — no double-application**: The 2c2e `common_factor = PI^3*2/SQRTPI * fac_sp(li) * fac_sp(lk)` already includes fac_sp. Unlike the 1e kernel which applies fac_sp as a post-multiply, 2c2e includes it in common_factor (matching libcint `g2c2e.c` line 44-45 vs `g1e.c` which uses `common_factor=1` and applies fac_sp in `cint1e.c` line 120-121).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed PTR_RANGE_OMEGA env collision in parity test data**
- **Found during:** Task 2 (oracle parity test debugging)
- **Issue:** `build_h2o_sto3g()` placed atomic coordinates starting at env[0], but libcint reserves env[0..19] for global parameters including `PTR_RANGE_OMEGA=env[8]`. H2's z-coordinate (1.1078) at env[8] was read as omega=1.1078, causing libcint to compute long-range Coulomb (`erf(omega*r)/r`) instead of standard Coulomb (`1/r`). Result: 27 mismatches with ratios varying from 1.04 to 1.29 depending on shell exponents.
- **Root cause diagnosis**: Isolated test with correct env layout (zeros at [0..20], user data at [20+]) showed vendor gives 49.64 matching our kernel. The mismatch was entirely in test data.
- **Fix:** Changed `build_h2o_sto3g()` to pre-fill `env = vec![0.0; PTR_ENV_START]` before pushing user data. Exported `PTR_ENV_START=20` from `cintx-compat/raw.rs` for future tests.
- **Files modified:** `crates/cintx-oracle/tests/center_2c2e_parity.rs`, `crates/cintx-compat/src/raw.rs`
- **Verification:** Both idempotency and vendor parity tests pass (mismatch_count=0 at atol=1e-9)
- **Committed in:** 8cbdfc7 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test data env layout)
**Impact on plan:** Fix was necessary for correctness. Kernel itself required no changes. No scope creep beyond adding PTR_ENV_START export.

## Issues Encountered

The main challenge was diagnosing a subtle libcint env layout bug. The Rys quadrature algorithm, G-tensor layout, VRR recurrence coefficients, and c2s transform were all correct. The parity failure had the following misleading characteristics:
- Ratios cintx/vendor varied by shell type (s-s: ~1.04, p-p: ~1.29) suggesting a missing normalization factor
- Simple 1-atom test (without the H2O env layout) gave a different vendor result
- The 1e parity test used the same bad env layout but passed because 1e integrals don't read PTR_RANGE_OMEGA

The C test that ran the vendor with the full H2O env confirmed shell (4,4) = 47.61 (with omega=1.1078 applied), while a corrected env (zeros at [0..19]) gave 49.64 matching cintx.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 2c2e kernel fully validated against vendor libcint 6.1.3 at atol=1e-9
- `PTR_ENV_START=20` exported and documented — all future 2e+ parity tests must use this convention
- Plans 10-04 (int2e) and 10-05 (int3c2e) parity tests will need the same env layout fix
- The center_3c1e_parity.rs from 10-03 also likely has this env layout issue — should be checked

---
*Phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure*
*Completed: 2026-04-03*
