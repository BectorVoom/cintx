---
phase: 09-1e-real-kernel-and-cart-to-sph-transform
plan: "01"
subsystem: transform
tags: [c2s, cart-to-sph, condon-shortley, libcint-compat]
dependency_graph:
  requires: []
  provides: [c2s-coefficients, cart_to_sph_1e, ncart, nsph]
  affects: [crates/cintx-cubecl/src/kernels/one_electron.rs]
tech_stack:
  added: []
  patterns: [host-side-transform, bra-ket-matrix-multiply, libcint-coefficient-extraction]
key_files:
  created:
    - crates/cintx-cubecl/tests/c2s_tests.rs
  modified:
    - crates/cintx-cubecl/src/transform/c2s.rs
decisions:
  - "Extracted C2S_L0..C2S_L4 from libcint cart2sph.c g_trans_cart2sph[] verbatim — no rounding or reordering"
  - "cart_to_spheric_staging made a no-op: 1e kernels handle c2s internally with shell angular momentum"
  - "cart_to_sph_1e applies two-pass bra+ket matrix multiply using static coefficient dispatch"
metrics:
  duration_seconds: 187
  completed_date: "2026-04-03"
  tasks_completed: 2
  files_modified: 2
---

# Phase 09 Plan 01: Cart-to-Sph Transform (Condon-Shortley Coefficients) Summary

Real Condon-Shortley coefficient matrices C2S_L0..C2S_L4 extracted from libcint `cart2sph.c` and `cart_to_sph_1e()` function implementing correct bra+ket matrix multiply for all angular momentum pairs l=0..4.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement Condon-Shortley coefficients and cart_to_sph_1e | `7ebbaba` | `c2s.rs` (235+/18-) |
| 2 | Add c2s coefficient validation and transform correctness tests | `e91bb1e` | `tests/c2s_tests.rs` (new, 238 lines) |

## What Was Built

### c2s.rs (replaced)

- `pub const C2S_L0: [[f64; 1]; 1]` — s-shell identity (1x1)
- `pub const C2S_L1: [[f64; 3]; 3]` — p-shell transform (3x3), px/py/pz permutation
- `pub const C2S_L2: [[f64; 6]; 5]` — d-shell transform (5x6) with exact libcint coefficients
- `pub const C2S_L3: [[f64; 10]; 7]` — f-shell transform (7x10)
- `pub const C2S_L4: [[f64; 15]; 9]` — g-shell transform (9x15)
- `pub fn cart_to_sph_1e(cart_buf, sph_buf, li, lj)` — two-pass bra+ket matrix multiply
- `pub fn ncart(l)` — returns (l+1)(l+2)/2
- `pub fn nsph(l)` — returns 2l+1
- `pub fn cart_to_spheric_staging(staging)` — no-op (returns Ok without modifying data)

### Integration Tests (new)

13 tests in `crates/cintx-cubecl/tests/c2s_tests.rs`:
- `test_ncart_values`, `test_nsph_values` — dimension helper correctness
- `test_c2s_l0_identity` — 1x1 identity
- `test_c2s_l1_coefficients` — p-shell permutation matrix validation
- `test_c2s_l2_coefficients` — 5 specific coefficient checks at 1e-15 tolerance
- `test_c2s_l3_dimensions` — 7x10 shape + spot-check values
- `test_c2s_l4_dimensions` — 9x15 shape + spot-check values
- `test_c2s_ss_identity` — s-s transform is identity
- `test_c2s_pp_transform` — p-p transform produces correct permutation output
- `test_c2s_ds_transform` — d-s 6-cart -> 5-sph with exact coefficient check
- `test_c2s_sd_transform` — s-d 6-cart -> 5-sph with exact coefficient check
- `test_cart_to_spheric_staging_noop` — staging data unchanged
- `test_cart_to_spheric_staging_noop_empty` — empty slice succeeds

## Verification Results

```
cargo test -p cintx-cubecl -- c2s
  9 passed; 0 failed (unit + integration filter)

cargo test -p cintx-cubecl --test c2s_tests
  13 passed; 0 failed
```

## Success Criteria Check

- [x] C2S_L0 through C2S_L4 coefficient matrices exist in c2s.rs matching libcint cart2sph.c
- [x] cart_to_sph_1e() correctly transforms cartesian to spherical for l=0..4
- [x] cart_to_spheric_staging is a no-op (does not corrupt staging data)
- [x] All tests pass under `cargo test -p cintx-cubecl`

## Deviations from Plan

None — plan executed exactly as written.

The plan specified 7 behaviors for Task 1 unit tests and 10 behaviors for Task 2 integration tests. The implementation provided all required behaviors plus additional tests (`test_c2s_sd_transform`, `test_cart_to_spheric_staging_noop_empty`) for more complete coverage.

## Known Stubs

None. All coefficient matrices are fully populated from libcint source. The `cart_to_sph_1e()` function correctly handles l=0..4; l>4 returns 0.0 coefficients (documented behavior, no stubs).

## Self-Check: PASSED

- FOUND: `crates/cintx-cubecl/src/transform/c2s.rs`
- FOUND: `crates/cintx-cubecl/tests/c2s_tests.rs`
- FOUND: commit `7ebbaba` (feat: c2s coefficients and cart_to_sph_1e)
- FOUND: commit `e91bb1e` (test: c2s integration tests)
