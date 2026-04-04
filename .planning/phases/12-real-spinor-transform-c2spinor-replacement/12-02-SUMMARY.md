---
phase: 12-real-spinor-transform-c2spinor-replacement
plan: 02
subsystem: transform, oracle
tags: [spinor, c2spinor, oracle, parity, 1e, vendor-ffi]
requirements: [SPIN-03]

dependency_graph:
  requires:
    - crates/cintx-cubecl/src/transform/c2spinor.rs (cart_to_spinor_sf_2d)
    - crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs (CG coefficient tables)
    - crates/cintx-oracle/src/vendor_ffi.rs (vendor_CINTcgto_spinor)
    - libcint-master/include/cint_funcs.h (int1e_ovlp_spinor etc. declarations)
  provides:
    - vendor FFI wrappers for int1e_ovlp_spinor, int1e_kin_spinor, int1e_nuc_spinor
    - oracle_gate_1e_spinor: all three 1e spinor operators pass at atol=1e-12
    - cart_to_spinor_sf_2d: full 2D bra+ket c2spinor sf transform
  affects:
    - eval_raw spinor path for 1e family (Representation::Spinor now applies real transform)
    - Phase 12-03 multi-center spinor oracle work

tech_stack:
  patterns:
    - Vendor FFI: CINTIntegralFunction typedef, double *out = double complex * layout-compatible
    - 2D c2spinor transform: bra step (conjugate convention) + ket step (complex multiply)
    - Oracle comparison: flat f64 buffer of interleaved re/im complex elements

key_files:
  created: []
  modified:
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/tests/oracle_gate_closure.rs

decisions:
  - "2D c2spinor sf transform: bra step uses conjugate convention (saI += -caI * v1) matching libcint a_bra_cart2spinor_sf; ket step uses complex multiply matching a_ket_cart2spinor"
  - "Output layout: column-major interleaved (j_spinor outer, i_spinor inner), matching libcint zcopy_ij: staging[(j*di+i)*2] = re, +1 = im"
  - "Spinor buffer size: ni_sp * nj_sp * 2 f64 (planner already sizes correctly with complex_multiplier=2)"
  - "Executor wiring: Representation::Spinor branch in one_electron.rs calls cart_to_spinor_sf_2d with shell.kappa from both shells"

metrics:
  duration_minutes: 5
  completed_date: "2026-04-04"
  tasks_completed: 2
  files_modified: 5
---

# Phase 12 Plan 02: 1e Spinor Oracle Parity Gate Summary

Vendor FFI wrappers for three 1e spinor integrals, 2D cart-to-spinor sf transform implementation, and oracle parity gate confirming int1e_ovlp_spinor/int1e_kin_spinor/int1e_nuc_spinor match vendored libcint 6.1.3 at atol=1e-12.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add vendor FFI wrappers for 1e spinor integrals | 9722561 | build.rs, vendor_ffi.rs |
| 2 | Add 1e spinor oracle parity gate test | 3be7b40 | c2spinor.rs, one_electron.rs, oracle_gate_closure.rs |

## What Was Built

**Task 1 — Vendor FFI wrappers:**
- Updated `build.rs` allowlist to include `int1e_ovlp_spinor|int1e_kin_spinor|int1e_nuc_spinor`
- Added `vendor_int1e_ovlp_spinor`, `vendor_int1e_kin_spinor`, `vendor_int1e_nuc_spinor` wrappers in `vendor_ffi.rs`
- Output buffer: `ni_sp * nj_sp * 2` f64 (interleaved re/im complex elements)
- All three functions declared in `cint_funcs.h` via `CINTIntegralFunction` typedef (same as sph variants)

**Task 2 — 2D c2spinor transform + oracle test:**

**c2spinor.rs** — new functions:
- `cart_to_spinor_sf_2d`: full 2D bra+ket scalar-field spinor transform
  - Bra step: `apply_bra_sf_block_all_kappa` / `apply_bra_block` — conjugate convention
  - Ket step: `apply_ket_transform` / `apply_ket_block` — complex coefficient multiply
  - kappa==0 case: GT block first (rows 0..nd_gt), then LT block (rows nd_gt..nd)
- `bra_coeff_refs`: returns flat coefficient slices for l=0..4

**one_electron.rs**:
- Added explicit `Representation::Spinor` arm calling `cart_to_spinor_sf_2d(staging, &cart_buf, li, kappa_i, lj, kappa_j)`
- Explicit `Representation::Cart` arm (copy Cartesian)
- Import `cart_to_spinor_sf_2d` from `crate::transform::c2spinor`

**oracle_gate_closure.rs**:
- Added `ATOL_SPINOR: f64 = 1e-12` constant (under `has_vendor_libcint`)
- Added `oracle_gate_1e_spinor` test: tests all three operators against vendored libcint on shells (0,1)

## Test Results

```
test oracle_gate_1e_spinor ... ok

1e spinor oracle: shells (0,1), ni_sp=2, nj_sp=2, nelems=8
  PASS: int1e_ovlp_spinor shells (0,1): mismatch_count=0, nonzero=2/8
  PASS: int1e_kin_spinor shells (0,1): mismatch_count=0, nonzero=2/8
  PASS: int1e_nuc_spinor shells (0,1): mismatch_count=0, nonzero=2/8
oracle_gate_1e_spinor: PASS — all three operators match vendored libcint at atol=1e-12

All existing oracle gate closure tests: ok (4 passed, 0 failed)
```

## Deviations from Plan

### Auto-added Missing Critical Functionality

**[Rule 2 - Missing Critical Functionality] Implemented 2D c2spinor sf transform**
- **Found during:** Task 2 (needed to pass oracle parity)
- **Issue:** The `one_electron.rs` Spinor arm was a no-op (copied Cartesian buffer). The Plan 01 SUMMARY documented this gap as a TODO. The 2D bra+ket transform was needed for oracle parity to pass.
- **Fix:** Implemented `cart_to_spinor_sf_2d` matching libcint `c2s_sf_1e` exactly: bra step uses conjugate convention (`saI += -caI * v1`), ket step uses complex multiply `((cR+i*cI)*(gR+i*gI))`. Both steps handle kappa==0 (GT+LT blocks).
- **Files modified:** `crates/cintx-cubecl/src/transform/c2spinor.rs`, `crates/cintx-cubecl/src/kernels/one_electron.rs`
- **Commits:** included in Task 2 commit 3be7b40

## Known Stubs

None — all three operators produce non-zero output matching libcint at atol=1e-12.

## Self-Check: PASSED
