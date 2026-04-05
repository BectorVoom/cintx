---
phase: 13-f12-stg-yp-kernels
plan: "04"
subsystem: testing
tags: [f12, stg, yp, oracle, nabla, spherical-harmonics, derivative-kernels]

# Dependency graph
requires:
  - phase: 13-f12-stg-yp-kernels
    provides: F12 STG/YP kernel infrastructure, planner component_rank parsing, oracle harness for 10 sph symbols

provides:
  - Multi-component spherical harmonic transform for F12 derivative operators (ip1=3 comp, ipip1/ipvip1/ip1ip2=9 comp)
  - Correct component_rank in manifest for all 8 derivative F12 entries
  - Oracle parity at atol=1e-12 for all 10 F12 sph symbols
  - nabla1i_2e/nabla1j_2e/nabla1k_2e G tensor derivative functions in f12.rs
  - gout_ip1/gout_ipip1/gout_ipvip1/gout_ip1ip2 Cartesian contraction functions

affects: [f12-verification, phase-14, oracle-harness, manifest-audit]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ncomp>1 dispatch in f12_kernel_core — branch on variant.ncomp then (j_inc,k_inc) pair
    - G tensor ceil/base split — ceil angular momenta for G tensor shape, base for gout loops and sph transform
    - Column-major reordering in gout_ipip1 — matches libcint autocode output ordering exactly
    - eval_f12_sph_ncomp oracle helper — sizes buffer as ncomp*n_sph_elements for multi-component oracle eval

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-oracle/tests/f12_oracle_parity.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "nabla1i_2e reads g[off + n + di] at li_ceil headroom — base angular momentum used only for gout loops and sph transform, ceil for G tensor allocation"
  - "gout_ipip1 uses libcint-exact column-major reordering (s[0..8] written as transposed 3x3), ipvip1/ip1ip2 do not reorder"
  - "grad2.c and hess.c added to oracle build.rs — cint2e_f12.c only declares extern forward references, not definitions"
  - "f12_zeta_zero_rejected_all_10 buffer sized at (9*n).max(1) to prevent BufferTooSmall masking InvalidEnvParam"

patterns-established:
  - "Per-primitive gout + per-component cart_to_sph_2e: produce ncomp*nf_cart Cartesian values per primitive, then sph-transform each component slice separately"
  - "Manifest component_rank drives planner staging buffer size via parse_component_multiplier — must equal actual derivative component count"

requirements-completed: [F12-03]

# Metrics
duration: 90min
completed: 2026-04-05
---

# Phase 13 Plan 04: F12 Derivative Sph Transform Summary

**Closed F12-03 gap by implementing nabla/Hessian G tensor derivatives and per-component sph transforms for all 8 F12 derivative operators, achieving oracle parity at atol=1e-12 for all 10 F12 sph symbols against vendored libcint 6.1.3**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-04-05
- **Completed:** 2026-04-05
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Ported nabla1i_2e/nabla1j_2e/nabla1k_2e from libcint g2e.c — correct G tensor derivative accumulation with ceil angular momentum headroom
- Implemented gout_ip1 (3 comp), gout_ipip1/gout_ipvip1/gout_ip1ip2 (9 comp each) matching libcint autocode exactly, including ipip1 column-major reordering
- Refactored f12_kernel_core to dispatch on ncomp: ncomp==1 uses original path, ncomp>1 uses per-primitive gout + per-component sph transform
- Updated component_rank in api_manifest.rs/csv/lock.json for all 8 derivative entries (ip1: "3", ipip1/ipvip1/ip1ip2: "9")
- Replaced 8 idempotency tests with 8 oracle_parity_int2e_* tests covering SS/HH/SP shell quartets, all passing at atol=1e-12
- Added grad2.c and hess.c to oracle build.rs to provide linker symbols for derivative gout implementations

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement multi-component derivative contraction** - `72ab805` (feat)
2. **Task 2: Oracle parity tests for all 8 derivative F12 variants** - `ea01786` (feat)

**Plan metadata:** (pending — final docs commit)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/f12.rs` - nabla1i/j/k_2e functions, gout_ip1/ipip1/ipvip1/ip1ip2, ncomp dispatch in f12_kernel_core
- `crates/cintx-ops/src/generated/api_manifest.rs` - component_rank corrected for 8 derivative entries
- `crates/cintx-ops/src/generated/api_manifest.csv` - same component_rank corrections
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - same component_rank corrections
- `crates/cintx-oracle/tests/f12_oracle_parity.rs` - 8 oracle_parity_int2e_* tests, eval_f12_sph_ncomp helper, idempotency tests removed
- `crates/cintx-oracle/build.rs` - grad2.c and hess.c added to vendored libcint compilation

## Decisions Made
- Used ceil angular momenta (li + i_inc, lj + j_inc, lk + k_inc) for G tensor shape only; base angular momenta for gout contraction loop bounds and sph transforms — matches libcint's "headroom" design
- gout_ipip1 applies column-major transposition to match libcint autocode output order; gout_ipvip1 and gout_ip1ip2 do not (their autocode doesn't)
- grad2.c and hess.c added to oracle build rather than cint2e_f12.c — the latter only declares extern forward references, actual definitions live in autocode files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed gout_ipip1 s-accumulation using wrong g2y term**
- **Found during:** Task 1 (gout_ipip1 implementation)
- **Issue:** s[1] was written as `g2x * g2y * g0z` but correct libcint formula is `g2x * g1y * g0z` — g2 is nabla at li+0, g1 is nabla at li+1, both computed separately per axis
- **Fix:** Extracted separate g1x/g1y/g1z (nabla at li+1) variables alongside g2x/g2y/g2z (nabla at li+0) in the inner loop
- **Files modified:** crates/cintx-cubecl/src/kernels/f12.rs
- **Verification:** Oracle parity tests pass for stg_ipip1_sph and yp_ipip1_sph at atol=1e-12
- **Committed in:** 72ab805 (Task 1)

**2. [Rule 3 - Blocking] Added grad2.c and hess.c to oracle build.rs for linker symbols**
- **Found during:** Task 2 (oracle parity test verification)
- **Issue:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test` failed with "undefined symbol: CINTgout2e_int2e_ipip1" — cint2e_f12.c only provides extern declarations, not definitions
- **Fix:** Added both files to `rerun-if-changed` and `cc::Build` compilation in build.rs
- **Files modified:** crates/cintx-oracle/build.rs
- **Verification:** All 15 oracle parity tests pass after fix
- **Committed in:** ea01786 (Task 2)

**3. [Rule 1 - Bug] Fixed f12_zeta_zero_rejected_all_10 buffer allocation**
- **Found during:** Task 2 (oracle parity test verification)
- **Issue:** After component_rank changed from "" to "3"/"9", planner sized staging buffers as ncomp*n_sph_elements. Test was allocating only 1 element for all-s quartet, triggering BufferTooSmall before InvalidEnvParam
- **Fix:** Changed buffer size to `(9 * n).max(1)` so derivative symbols always have enough space for zeta validation to fire
- **Files modified:** crates/cintx-oracle/tests/f12_oracle_parity.rs
- **Verification:** f12_zeta_zero_rejected_all_10 passes and correctly tests all 10 symbols
- **Committed in:** ea01786 (Task 2)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All fixes necessary for correctness and test coverage. No scope creep.

## Issues Encountered
- libcint idx-based G tensor contraction (where ix/iy/iz are pre-offset) required careful translation to Rust's explicit offset-based indexing (gx_off=0, gy_off=g_size, gz_off=2*g_size)
- ipip1 column-major reordering is present in libcint autocode for ipip1 but absent for ipvip1/ip1ip2 — required careful reading of each autocode file separately

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 10 F12 sph symbols verified against libcint 6.1.3 at atol=1e-12
- F12-03 gap closed; derivative kernel infrastructure ready for any future operator additions
- Oracle harness correctly compiles all required autocode files

---
*Phase: 13-f12-stg-yp-kernels*
*Completed: 2026-04-05*
