---
phase: 12-real-spinor-transform-c2spinor-replacement
plan: "05"
subsystem: spinor-oracle-gates
tags: [oracle-parity, spinor, 2e, 2c2e, 3c2e, bug-fix]
dependency_graph:
  requires: [12-04-SUMMARY.md]
  provides: [oracle_gate_2e_spinor, oracle_gate_2c2e_spinor, oracle_gate_3c2e_spinor]
  affects: [cintx-cubecl/src/executor.rs, cintx-cubecl/src/transform/c2spinor.rs]
tech_stack:
  added: []
  patterns: [libcint-kappa0-lt-gt-ordering, conj-multiply, real-split-vs-interleaved]
key_files:
  created: []
  modified:
    - crates/cintx-oracle/tests/oracle_gate_closure.rs
    - crates/cintx-cubecl/src/executor.rs
    - crates/cintx-cubecl/src/transform/c2spinor.rs
decisions:
  - "kappa=0 spinor block ordering: LT first then GT — matches libcint implicit memory layout where LT coeff pointer over-reads into GT table"
  - "executor.rs skips apply_representation_transform for Spinor: kernel launchers own the transform per Plan 04"
metrics:
  duration: ~90min (continued from previous session)
  completed: 2026-04-05
  tasks_completed: 1
  files_changed: 3
---

# Phase 12 Plan 05: Multi-Center Spinor Oracle Parity Gates Summary

Un-ignore three multi-center spinor oracle parity tests (2e, 2c2e, 3c2e) and fix two blocking bugs in the spinor transform pipeline, achieving 0-mismatch parity with vendored libcint 6.1.3 at atol=1e-12.

## What Was Built

Plan 05 activates the oracle parity gates for all resolvable multi-center spinor integral families. The plan had one task: remove the `#[ignore]` attributes from three oracle tests that were previously blocked on missing kernel wiring. Plan 04 wired the `Representation::Spinor` arms in the kernel launchers; Plan 05 proves correctness end-to-end.

In practice, two bugs were discovered and fixed before the tests could pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] executor.rs missing Spinor bypass for apply_representation_transform**

- **Found during:** Task 1, first test run
- **Issue:** Plan 04 added `Representation::Spinor` dispatch to kernel launchers and changed `apply_representation_transform` to return `Err(UnsupportedApi)` for Spinor. But executor.rs still called `apply_representation_transform` unconditionally after the kernel returned. This caused UnsupportedApi errors for ALL spinor tests including previously-passing 1e tests.
- **Fix:** Added `if !matches!(plan.representation, cintx_core::Representation::Spinor)` guard around the `apply_representation_transform` call in executor.rs. Spinor transforms are kernel-launcher responsibility (per Plan 04 design); the executor must skip the generic transform for Spinor.
- **Files modified:** `crates/cintx-cubecl/src/executor.rs`
- **Result after fix:** 1e, 2c2e, 3c2e spinor oracle tests pass; 2e spinor still fails with 36 mismatches.
- **Commit:** b322139

**2. [Rule 1 - Bug] kappa=0 spinor component ordering: LT first, then GT**

- **Found during:** Task 1, debugging oracle_gate_2e_spinor (36 mismatches after fix 1)
- **Issue:** All kappa=0 dispatch sites in c2spinor.rs applied GT block first (rows 0..2*l+1), then LT (rows 2*l+2..4*l+1). But libcint's `a_bra_cart2spinor_sf`, `a_bra1_cart2spinor_zf`, `a_ket_cart2spinor`, and `a_ket1_cart2spinor` use the LT coeff pointer for kappa >= 0. For kappa=0, `nd = 4*l+2` but only 2*l LT rows exist — so the function reads LT rows 0..2*l first, then continues reading past the LT table end directly into the GT table region (which immediately follows LT in the `g_trans_cart2jR` flat array). This is an intentional libcint memory-layout convention: **LT first, GT second** for kappa=0.
  - For l=0: LT has 0 rows and LT/GT point to the same offset → order irrelevant, tests passed.
  - For l=1 (p-shell, shell k in the 2e test): LT has 2 rows, GT has 4 rows. Wrong ordering placed GT spinors at positions 0-3 and LT at 4-5, but libcint expects LT at 0-1 and GT at 2-5. This caused 36 mismatches in the 96-element 2e output.
- **Fix:** In all six kappa=0 dispatch sites in c2spinor.rs, changed the block order to LT first then GT:
  - `cart_to_spinor_sf` (1D sf)
  - `cart_to_spinor_iket_sf` (1D iket_sf)
  - `cart_to_spinor_si` (1D si)
  - `cart_to_spinor_iket_si` (1D iket_si)
  - `apply_bra_sf_block_all_kappa` (bra step of 2D c2spinor_sf_2d)
  - `apply_ket_transform` (ket step of 2D c2spinor_sf_2d)
  - `apply_bra1_zf_block_all_kappa` (bra-zf step of 4D c2spinor_sf_4d step 2)
  - `apply_ket1_block_all_kappa` (ket step of 4D c2spinor_sf_4d step 2)
- **Files modified:** `crates/cintx-cubecl/src/transform/c2spinor.rs`
- **Result after fix:** oracle_gate_2e_spinor: 0 mismatches. All spinor oracle tests pass.
- **Commit:** b322139

## Test Results

All oracle gate closure tests pass:

```
oracle_gate_1e_spinor:   PASS — 0 mismatches, nonzero=2/8  (no regression)
oracle_gate_2e_spinor:   PASS — 0 mismatches, nonzero=20/96
oracle_gate_2c2e_spinor: PASS — 0 mismatches, nonzero=2/8
oracle_gate_3c2e_spinor: PASS — 0 mismatches, nonzero=2/16
oracle_gate_3c1e_spinor: IGNORED (int3c1e_spinor not in libcint 6.1.3)

oracle_gate_all_five_families: PASS (sph families unchanged)
```

## Known Stubs

None. All oracle parity tests are wired to real data with 0-mismatch tolerance.

## Self-Check

### Check created files exist

No new files were created.

### Check commits exist

- b322139: FOUND — fix(12-05): un-ignore multi-center spinor oracle parity tests and fix spinor transforms

## Self-Check: PASSED
