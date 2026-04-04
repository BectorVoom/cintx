---
phase: 12-real-spinor-transform-c2spinor-replacement
plan: 01
subsystem: transform
tags: [spinor, c2spinor, cg-coefficients, compat, tdd]
requirements: [SPIN-01, SPIN-02, SPIN-04]

dependency_graph:
  requires:
    - cintx-cubecl/transform/c2s.rs (ncart function)
    - libcint cart2sph.c (g_trans_cart2jR/g_trans_cart2jI arrays at documented offsets)
  provides:
    - c2spinor_coeffs.rs: CG coupling coefficient tables for l=0..4 (gt/lt x R/I)
    - c2spinor.rs: four transform variants (sf, iket_sf, si, iket_si)
    - cintx-compat transform.rs: compat entry points delegating to real transforms
  affects:
    - All spinor oracle parity work (phases 12-02, 12-03)

tech_stack:
  patterns:
    - TDD: RED (failing tests) → GREEN (implementation) → all 23+3 tests pass
    - CG coefficient extraction: Python script to parse C array offsets from g_c2s[] struct
    - Row layout: coeff[i*nf*2+n] = alpha[i][n], coeff[i*nf*2+nf+n] = beta[i][n]
    - kappa dispatch: kappa<0 → GT block, kappa>0 → LT block, kappa==0 → GT then LT
    - iket = multiply output by i: (re,im) → (-im,re)

key_files:
  created:
    - crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs
    - (replaces old c2spinor.rs stub entirely)
  modified:
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-cubecl/src/transform/mod.rs
    - crates/cintx-compat/src/transform.rs

decisions:
  - "Use Python to extract CG coefficient values from C source at documented offsets; verified against libcint g_c2s[] struct for l=0..4"
  - "Implement four separate code paths (sf, iket_sf, si, iket_si) with internal block helpers — no shared-core-with-flags per D-03"
  - "cart_to_spinor_interleaved_staging made into no-op (not deleted) to preserve staging API; executor wiring gap documented with TODO"
  - "si gcart layout in compat: four Pauli segments [v1, vx, vy, vz] each of size nf packed contiguously"

metrics:
  duration_minutes: 11
  completed_date: "2026-04-04"
  tasks_completed: 2
  files_modified: 4
---

# Phase 12 Plan 01: Real Spinor Transform (c2spinor Replacement) Summary

Extracted Clebsch-Gordan coefficient tables from vendored libcint cart2sph.c and implemented all four CINTc2s_*spinor* transform variants with correct kappa dispatch, replacing the amplitude-averaging stub.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | CG coefficient tables + four c2spinor variants | dd60d98 | c2spinor_coeffs.rs (new), c2spinor.rs (rewritten), mod.rs |
| 2 | Rewire compat spinor entry points to real transforms | 01b5eb4 | crates/cintx-compat/src/transform.rs |

## What Was Built

**c2spinor_coeffs.rs**: Complete CG coupling coefficient tables for l=0..4, both GT (j=l+1/2, kappa<0) and LT (j=l-1/2, kappa>0) blocks, real and imaginary parts. Extracted from `g_trans_cart2jR[]`/`g_trans_cart2jI[]` arrays in libcint cart2sph.c at the offsets documented in `g_c2s[]`. Row layout: `row[0..nf]` = alpha coefficients, `row[nf..2*nf]` = beta coefficients.

**c2spinor.rs**: Four public transform functions:
- `cart_to_spinor_sf`: scalar-field, `saR += caR*v1`, `saI += caI*v1`
- `cart_to_spinor_iket_sf`: multiply-by-i variant, `saR -= caI*v1`, `saI += caR*v1`
- `cart_to_spinor_si`: spin-included with Pauli coupling (v1, vx, vy, vz)
- `cart_to_spinor_iket_si`: multiply-by-i variant of si
- `spinor_len(l, kappa)`: kappa dispatch returning 2l+2/2l/4l+2
- `cart_to_spinor_interleaved_staging`: no-op (staging path; executor wiring gap documented)

**crates/cintx-compat/src/transform.rs**: Four compat entry points delegate to real transforms. si variants split the `gcart` buffer into four Pauli segments. nctr>1 contractions are handled iteratively.

## Test Results

- `cargo test --package cintx-cubecl --lib transform`: 42 passed, 0 failed
- `cargo test --package cintx-compat --lib transform`: 3 passed, 0 failed
- `cargo check --workspace`: compiles without errors

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as designed.

### Documentation Gap (Intentional)

`cart_to_spinor_interleaved_staging` was kept as a no-op (rather than deleted) because it is called from `apply_representation_transform` in mod.rs. This is a known gap: the executor dispatch path does not yet pass l/kappa to the spinor transform. A TODO comment was added. This does not affect correctness of the new transform functions — they are called directly from compat and tests.

## Known Stubs

- `cart_to_spinor_interleaved_staging` in c2spinor.rs is now a no-op placeholder (not the amplitude-averaging stub). The real transforms require explicit l/kappa from the executor — this is tracked as a TODO and will be resolved when oracle tests exercise the full spinor path (Phase 12-02/12-03).

## Self-Check: PASSED
