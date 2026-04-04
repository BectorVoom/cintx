---
phase: 09-1e-real-kernel-and-cart-to-sph-transform
plan: 04
subsystem: testing
tags: [libcint, oracle, parity, ffi, bindgen, cc, kinetic, cart2sph, vendored]

requires:
  - phase: 09-1e-real-kernel-and-cart-to-sph-transform
    provides: "Real 1e host kernel (overlap, kinetic, nuclear) + cart-to-sph transform"
provides:
  - "Vendored libcint 6.1.3 compiled from source via cc crate in cintx-oracle build.rs"
  - "FFI wrappers for int1e_ovlp_sph / int1e_kin_sph / int1e_nuc_sph via bindgen"
  - "True vendor parity tests comparing cintx output against vendored libcint C binary"
  - "int1e_kin_sph kinetic formula corrected (D_j^2 steps ±2 j-levels, not ±2 i-levels)"
  - "C2S_L1 p-shell ordering fixed (identity px/py/pz, not py/pz/px)"
  - "CINTcommon_fac_sp normalization applied for s/p shells"
affects:
  - 09-1e-real-kernel-and-cart-to-sph-transform
  - future 2e kernel plans using same kinetic formula pattern

tech-stack:
  added:
    - "cc crate: libcint 6.1.3 vendored static library compilation (cintx-oracle)"
    - "bindgen: FFI bindings for int1e_ovlp/kin/nuc_sph + CINTcgto_spheric"
  patterns:
    - "CINTX_ORACLE_BUILD_VENDOR=1 env gate: fast default tests, vendor build only when needed"
    - "rerun-if-env-changed: build.rs must emit this for env-gated features to retrigger"
    - "cmake template processing in build.rs: replace @VAR@ and #cmakedefine in cint.h.in / cint_config.h.in"
    - "vendor parity test layout: collect_1e_sph_matrix_vendor handles column-major libcint output"

key-files:
  created:
    - "crates/cintx-oracle/src/vendor_ffi.rs - Safe Rust wrappers around vendored libcint FFI"
  modified:
    - "crates/cintx-oracle/build.rs - Full libcint 6.1.3 compilation + bindgen + rerun-if-env-changed"
    - "crates/cintx-oracle/src/lib.rs - Added vendor_ffi module registration"
    - "crates/cintx-oracle/tests/one_electron_parity.rs - Added 3 vendor parity tests"
    - "crates/cintx-cubecl/src/kernels/one_electron.rs - Kinetic formula fix + common_fac_sp"
    - "crates/cintx-cubecl/src/transform/c2s.rs - C2S_L1 p-shell ordering fix"
    - "crates/cintx-cubecl/tests/c2s_tests.rs - Updated p-shell coefficient tests"

key-decisions:
  - "D_j^2 derivative steps ±2 j-levels (±2*dj in flat index), not ±2 i-levels: second derivative requires g0[jx+2] and g0[jx-2], so HRR must produce lj+2 levels and nmax=li+lj+2"
  - "C2S_L1 is identity (px/py/pz), not (py/pz/px): libcint g_trans_cart2sph[] default p ordering is identity with normalization in CINTcommon_fac_sp, not embedded in transform coefficients"
  - "gnu89 required for cint_bas.c: K&R empty-param function pointers fail C99 strict mode"
  - "autocode/intor1.c required: int1e_kin_sph lives there, not in cint1e.c"
  - "rerun-if-env-changed missing from original build.rs: cargo won't retrigger build on env var change without it"

patterns-established:
  - "Vendor parity tests: use #[cfg(has_vendor_libcint)] gate, require --features cpu and CINTX_ORACLE_BUILD_VENDOR=1"
  - "Libcint column-major output: unpack with out[jj*ni+ii] not out[ii*nj+jj] for (bra=i, ket=j)"

requirements-completed: [KERN-01, VERI-05]

duration: ~3h
completed: 2026-04-03
---

# Phase 09 Plan 04: Vendor Oracle Parity Summary

**Vendored libcint 6.1.3 compiled from C source via cc crate; all three 1e spherical operators match upstream to atol=1e-11 after fixing kinetic D_j^2 formula and p-shell C2S ordering**

## Performance

- **Duration:** ~3h (including multi-round debugging of kinetic formula)
- **Started:** 2026-04-03T12:00:00Z
- **Completed:** 2026-04-03T15:50:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Compiled vendored libcint 6.1.3 from 16 C source files via cc crate in build.rs, including cmake template processing for `cint.h.in` and `cint_config.h.in`
- Generated bindgen FFI bindings for int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph, CINTcgto_spheric
- All 6 oracle parity tests pass: 3 idempotency + 3 vendor comparison (mismatch_count == 0 at atol=1e-11/rtol=1e-9 for H2O STO-3G 7x7 matrix)
- Fixed kinetic energy D_j^2 formula: derivative steps ±2 j-levels (not ±2 bra VRR levels)
- Fixed p-shell cart-to-sph ordering: identity (px,py,pz) matches libcint default
- Applied CINTcommon_fac_sp normalization for s/p shells (~4pi factor)

## Task Commits

1. **Task 1: Wire vendored libcint compilation and FFI module** - `37f2e8e` (feat)
2. **Task 2: Vendor parity tests + kinetic/p-shell bug fixes** - `f1ebbd8` (feat)

## Files Created/Modified

- `crates/cintx-oracle/src/vendor_ffi.rs` - Safe Rust wrappers around vendored libcint FFI (gated `#[cfg(has_vendor_libcint)]`)
- `crates/cintx-oracle/build.rs` - Full libcint compilation, cmake template processing, bindgen, rerun-if-env-changed
- `crates/cintx-oracle/src/lib.rs` - Added `pub mod vendor_ffi` registration
- `crates/cintx-oracle/tests/one_electron_parity.rs` - Added `collect_1e_sph_matrix_vendor` + 3 vendor parity tests
- `crates/cintx-cubecl/src/kernels/one_electron.rs` - Kinetic formula fix + common_fac_sp normalization
- `crates/cintx-cubecl/src/transform/c2s.rs` - C2S_L1 p-shell identity ordering fix
- `crates/cintx-cubecl/tests/c2s_tests.rs` - Updated p-shell coefficient assertions

## Decisions Made

- Kinetic derivative uses ±2 j-levels (D_j^2 expands as `jx*(jx-1)*g0[jx-2] - 2*aj*(2*jx+1)*g0[jx] + 4*aj^2*g0[jx+2]`), requiring HRR to lj+2 and nmax=li+lj+2
- p-shell C2S_L1 is identity matrix matching libcint's default non-PYPZPX ordering; normalization via CINTcommon_fac_sp(1)=0.4886 is separate
- gnu89 (-std=gnu89) required for cint_bas.c which uses K&R empty-param function pointer declarations
- autocode/intor1.c is mandatory: int1e_kin_sph is auto-generated there, not in cint1e.c

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed kinetic energy formula: derivative direction and step size**
- **Found during:** Task 2 (vendor parity tests)
- **Issue:** `contract_kinetic` applied D_i^2 (stepping ±2 in the VRR bra i-index) but libcint's `CINTgout1e_int1e_kin` uses D_j^2 (stepping ±2 j-levels in the HRR ket direction). The closed form is `jx*(jx-1)*g0[jx-2] - 2*aj*(2*jx+1)*g0[jx] + 4*aj^2*g0[jx+2]` where the ±2 refers to j-level offsets (±2*dj in flat index). Required HRR to lj+2 and nmax=li+lj+2.
- **Files modified:** `crates/cintx-cubecl/src/kernels/one_electron.rs`
- **Verification:** `test_int1e_kin_sph_h2o_sto3g_vendor_parity` passes (0 mismatches)
- **Committed in:** f1ebbd8

**2. [Rule 1 - Bug] Fixed p-shell cart-to-sph ordering in C2S_L1**
- **Found during:** Task 2 (overlap vendor parity failed with p-shell blocks)
- **Issue:** C2S_L1 used (py,pz,px) ordering. Libcint default uses (px,py,pz) identity transform; normalization is in CINTcommon_fac_sp, not in the transform matrix.
- **Files modified:** `crates/cintx-cubecl/src/transform/c2s.rs`, `crates/cintx-cubecl/tests/c2s_tests.rs`
- **Verification:** `test_int1e_ovlp_sph_h2o_sto3g_vendor_parity` passes with p-shell corrections
- **Committed in:** f1ebbd8

**3. [Rule 1 - Bug] Applied CINTcommon_fac_sp normalization to s/p shells**
- **Found during:** Task 2 (s-shell overlap off by ~4pi factor)
- **Issue:** Libcint applies `common_factor * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l)` in the primitive loop. For s: 0.2821, for p: 0.4886, for d+: 1.0. Without this, s/p integrals were ~4pi off.
- **Files modified:** `crates/cintx-cubecl/src/kernels/one_electron.rs`
- **Verification:** All 3 vendor parity tests pass (0 mismatches)
- **Committed in:** f1ebbd8

**4. [Rule 2 - Missing Critical] Added rerun-if-env-changed to build.rs**
- **Found during:** Task 2 (build cache not invalidating when CINTX_ORACLE_BUILD_VENDOR changed)
- **Issue:** Without `cargo:rerun-if-env-changed=CINTX_ORACLE_BUILD_VENDOR`, cargo doesn't retrigger build.rs when the env var is added/removed. Tests continued using stale binaries.
- **Files modified:** `crates/cintx-oracle/build.rs`
- **Committed in:** f1ebbd8

**5. [Rule 3 - Blocking] Added autocode/intor1.c to build.rs source file list**
- **Found during:** Task 1 (int1e_kin_sph undefined symbol at link time)
- **Issue:** Plan listed cint1e.c for int1e_kin_sph but it is actually auto-generated in autocode/intor1.c. Linker failed with undefined reference.
- **Files modified:** `crates/cintx-oracle/build.rs`
- **Committed in:** 37f2e8e

---

**Total deviations:** 5 auto-fixed (3 Rule 1 bugs, 1 Rule 2 missing critical, 1 Rule 3 blocking)
**Impact on plan:** All auto-fixes required for correctness. No scope creep. Bugs 1-3 were fundamental algorithm errors that would have blocked oracle parity indefinitely.

## Issues Encountered

- cmake template variables in `cint.h.in` and `cint_config.h.in` had to be processed manually in build.rs (replace `@cint_VERSION@`, `#cmakedefine I8`, etc.) since cmake is not available
- K&R function pointer declarations in cint_bas.c required `-std=gnu89` flag
- Cargo build cache invalidation: `rerun-if-env-changed` was missing, causing stale binaries; required manual `touch` to force rebuild during debugging
- Libcint output is column-major (Fortran order): `out[j*ni+i]`, requiring transpose when assembling the full matrix

## Next Phase Readiness

- ROADMAP SC1/SC2 satisfied: cintx matches vendored libcint 6.1.3 for all three 1e operators at atol=1e-11/rtol=1e-9
- Foundation for 2e integral oracle parity established (same pattern: vendor compile + bindgen + collect_matrix_vendor)
- All 09-04 plan requirements satisfied: KERN-01 (kernel correctness), VERI-05 (oracle verification)

## Self-Check: PASSED

- vendor_ffi.rs: FOUND
- one_electron_parity.rs: FOUND
- 09-04-SUMMARY.md: FOUND
- commit 37f2e8e (Task 1): FOUND
- commit f1ebbd8 (Task 2): FOUND

---
*Phase: 09-1e-real-kernel-and-cart-to-sph-transform*
*Completed: 2026-04-03*
