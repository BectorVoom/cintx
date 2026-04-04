---
phase: 12-real-spinor-transform-c2spinor-replacement
plan: 04
subsystem: transform, kernels
tags: [spinor, c2spinor, multi-center, 2e, 2c2e, 3c2e, kernel-wiring]
requirements: [SPIN-02, SPIN-03, SPIN-04]

dependency_graph:
  requires:
    - crates/cintx-cubecl/src/transform/c2spinor.rs (cart_to_spinor_sf_2d, helper functions)
    - crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs (CG coefficient tables)
    - crates/cintx-cubecl/src/kernels/one_electron.rs (Spinor wiring pattern reference)
  provides:
    - cart_to_spinor_sf_4d: two-step 4-center spinor transform for 2e integrals
    - cart_to_spinor_sf_3c2e: sph-on-k then spinor-on-ij for 3c2e integrals
    - Representation::Spinor arm in launch_two_electron
    - Representation::Spinor arm in launch_center_2c2e
    - Representation::Spinor arm in launch_center_3c2e
    - apply_representation_transform: Spinor arm returns Err (not silent no-op)
  affects:
    - Phase 12-05 oracle un-ignore: multi-center spinor parity tests can now be activated

tech_stack:
  added: []
  patterns:
    - Two-step spinor transform: c2s_sf_2e1 (real input, conjugate bra) + c2s_sf_2e2 (complex input, zf complex multiply)
    - Bra-zf vs bra-sf distinction: step 2 of 4D uses complex multiply (not conjugate) matching libcint a_bra1_cart2spinor_zf
    - 3c2e: sph on k auxiliary then cart_to_spinor_sf_2d on (i,j) per k-sph slice
    - Explicit exhaustive match: Spheric/Spinor/Cart arms replace _ catch-all in all launchers

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-cubecl/src/transform/mod.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-cubecl/src/kernels/center_2c2e.rs
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs

key-decisions:
  - "cart_to_spinor_sf_4d uses two-step process: step 1 applies cart_to_spinor_sf_2d on (i,j) per kl-cart slice; step 2 applies bra-zf+ket-zf on complex intermediate using complex multiply convention"
  - "apply_bra_zf_block uses (cR*vR - cI*vI, cR*vI + cI*vR) matching libcint a_bra1_cart2spinor_zf — not the conjugate convention of step 1"
  - "cart_to_spinor_sf_3c2e applies cart-to-sph on k-index first, then reuses cart_to_spinor_sf_2d on each k-sph slice for (i,j)"
  - "apply_representation_transform Spinor arm returns UnsupportedApi error instead of silent no-op — closes Gap 2"
  - "cart_to_spinor_interleaved_staging deleted from c2spinor.rs — no callers remained"

metrics:
  duration_minutes: 5
  completed_date: "2026-04-05"
  tasks_completed: 2
  files_modified: 5
---

# Phase 12 Plan 04: Multi-Center Spinor Transform Wiring Summary

Multi-center spinor transform functions implemented (cart_to_spinor_sf_4d for 2e, cart_to_spinor_sf_3c2e for 3c2e), wired into all three multi-center kernel launchers as explicit Representation::Spinor arms, and apply_representation_transform no-op replaced with explicit error.

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-05T01:07:59Z
- **Completed:** 2026-04-05T01:13:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Implemented `cart_to_spinor_sf_4d` — two-step 4-center spinor transform matching libcint `c2s_sf_2e1` + `c2s_sf_2e2`
- Implemented `cart_to_spinor_sf_3c2e` — 3-center spinor transform: sph on k, then `cart_to_spinor_sf_2d` on each k-sph slice
- Added helper functions: `apply_2d_spinor_zf`, `apply_bra_zf_block_all_kappa`, `apply_bra_zf_block`, `apply_ket_zf_block_all_kappa`, `apply_ket_zf_block`, `c2s_k_coeff`
- Deleted `cart_to_spinor_interleaved_staging` no-op
- Fixed `apply_representation_transform` Spinor arm to return `Err(UnsupportedApi)` instead of silently succeeding
- Wired `Representation::Spinor` arm in `launch_two_electron` (calls `cart_to_spinor_sf_4d`)
- Wired `Representation::Spinor` arm in `launch_center_2c2e` (calls `cart_to_spinor_sf_2d`)
- Wired `Representation::Spinor` arm in `launch_center_3c2e` (calls `cart_to_spinor_sf_3c2e`)
- Replaced `_ =>` catch-all with explicit `Representation::Cart` arms in all three launchers
- 95 cintx-cubecl lib tests pass + all integration tests pass
- `cargo check --workspace` clean

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add multi-center spinor transforms, fix apply_representation_transform | 2a01812 | c2spinor.rs, mod.rs |
| 2 | Wire Representation::Spinor arms into 2e, 2c2e, 3c2e launchers | ca86866 | two_electron.rs, center_2c2e.rs, center_3c2e.rs |

## What Was Built

**c2spinor.rs** — new functions:
- `cart_to_spinor_sf_4d(staging, cart, li, kappa_i, lj, kappa_j, lk, kappa_k, ll, kappa_l)`:
  Step 1 calls `cart_to_spinor_sf_2d` on each (k_cart, l_cart) slice to get complex intermediate `opij`.
  Step 2 applies bra-zf + ket-zf (complex multiply convention) over the complex intermediate.
  Output layout: `staging[(((l_sp * dk + k_sp) * dj + j_sp) * di + i_sp) * 2]`.
- `cart_to_spinor_sf_3c2e(staging, cart, li, kappa_i, lj, kappa_j, lk)`:
  Applies c2s on k-index (cart → sph), then `cart_to_spinor_sf_2d` on each k-sph slice.
  Output layout: k-sph outer, then (j_sp, i_sp) inner, interleaved re/im.
- `c2s_k_coeff(l, m_row, cart_col)`: retrieves C2S coefficient for k auxiliary transform.
- Bra-zf and ket-zf helpers for step 2 complex transform.
- Deleted `cart_to_spinor_interleaved_staging`.

**mod.rs**: Spinor arm returns `Err(UnsupportedApi)` — closes Gap 2.

**Kernel launchers**: All three now have explicit `Representation::Spinor` arms:
- `launch_two_electron`: calls `cart_to_spinor_sf_4d` with kappa from all four shells
- `launch_center_2c2e`: calls `cart_to_spinor_sf_2d` with kappa from shell_i and shell_k
- `launch_center_3c2e`: calls `cart_to_spinor_sf_3c2e` with kappa from shell_i_in and shell_j_in

## Unit Tests Added

- `sf_4d_ssss_kappa_neg1_output_size` — output is 32 f64 for all-s quartet
- `sf_4d_ssss_kappa_neg1_nonzero` — non-zero output confirmed
- `sf_4d_pppp_kappa_neg1_output_size` — output is 512 f64 for all-p quartet, non-zero
- `sf_3c2e_sss_output_size` — output is 8 f64 for sss triple
- `sf_3c2e_sss_nonzero` — non-zero output confirmed
- `sf_3c2e_ssp_k_output_size` — output is 24 f64 for s,s,p-aux triple

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. The multi-center kernel Spinor arms are now wired. The oracle parity gates in
`crates/cintx-oracle/tests/oracle_gate_closure.rs` are still `#[ignore]` (added in Plan 03).
These can be un-ignored in Plan 05 once oracle correctness is confirmed.

---
*Phase: 12-real-spinor-transform-c2spinor-replacement*
*Completed: 2026-04-05*

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/transform/c2spinor.rs (contains cart_to_spinor_sf_4d and cart_to_spinor_sf_3c2e)
- FOUND: crates/cintx-cubecl/src/transform/mod.rs (Spinor arm returns Err)
- FOUND: crates/cintx-cubecl/src/kernels/two_electron.rs (Representation::Spinor arm)
- FOUND: crates/cintx-cubecl/src/kernels/center_2c2e.rs (Representation::Spinor arm)
- FOUND: crates/cintx-cubecl/src/kernels/center_3c2e.rs (Representation::Spinor arm)
- FOUND: commit 2a01812 (Task 1)
- FOUND: commit ca86866 (Task 2)
- NOT FOUND in c2spinor.rs: cart_to_spinor_interleaved_staging (deleted as required)
- Test result: 95 lib tests passed, 0 failed
- cargo check --workspace: clean
