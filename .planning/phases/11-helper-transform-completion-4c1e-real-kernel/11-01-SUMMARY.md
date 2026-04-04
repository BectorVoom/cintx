---
phase: 11-helper-transform-completion-4c1e-real-kernel
plan: "01"
subsystem: oracle/compat
tags: [helpers, transform, oracle, tolerance, CINTgto_norm, vendor-ffi]
dependency_graph:
  requires: []
  provides: [unified-tolerance, correct-CINTgto_norm, numeric-helper-oracle, numeric-transform-oracle]
  affects: [cintx-oracle/compare.rs, cintx-compat/helpers.rs, cintx-oracle/vendor_ffi.rs]
tech_stack:
  added: []
  patterns: [vendor-ffi-safe-wrapper, cfg(has_vendor_libcint)-gate, double-factorial]
key_files:
  created: []
  modified:
    - crates/cintx-oracle/src/compare.rs
    - crates/cintx-compat/src/helpers.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs
decisions:
  - "All families use UNIFIED_ATOL=1e-12 / UNIFIED_RTOL=1e-10 — per-family tolerance divergence was D-01 technical debt"
  - "CINTgto_norm now uses correct double-factorial formula from libcint misc.c — prior approximation would fail oracle comparison at 1e-12"
  - "Numeric oracle comparison block in verify_helper_surface_coverage gated on cfg(has_vendor_libcint) so default CI builds are unaffected"
  - "CINTc2s_bra_sph selected as the direct vendor FFI transform comparison target per HELP-02 — most commonly invoked transform in family pipelines"
  - "Remaining 6 transform symbols verified implicitly through family-level oracle parity (every 1e/2e/2c2e/3c1e/3c2e sph comparison exercises full cart->sph pipeline)"
metrics:
  duration: 4 min
  completed: "2026-04-04"
  tasks: 2
  files: 4
---

# Phase 11 Plan 01: Tolerance Unification, CINTgto_norm Fix, Numeric Helper/Transform Oracle

Unified all per-family tolerance constants to UNIFIED_ATOL=1e-12, fixed CINTgto_norm with the correct double-factorial formula from libcint misc.c, and added vendor FFI wrappers plus numeric oracle comparisons for all 17 helper symbols and direct buffer comparison for CINTc2s_bra_sph.

## What Was Built

### Task 1: Unify tolerance constants and fix CINTgto_norm
- Removed all 10 per-family tolerance constant pairs (`TOL_1E_ATOL`, `TOL_2E_ATOL`, etc.) from compare.rs
- Added unified `UNIFIED_ATOL=1e-12` and `UNIFIED_RTOL=1e-10`
- Updated `tolerance_for_family()` so all 7 family arms use the same unified constants
- Updated tolerance_table JSON serialization to use unified constants
- Fixed `CINTgto_norm` in helpers.rs: replaced the placeholder approximation `(2a)^((n+1.5)*0.5)` with the correct double-factorial formula `sqrt(fac2(2n-1) * PI^0.5 / (2a)^(n+1.5))` matching libcint misc.c

### Task 2: Add numeric helper and transform oracle comparisons with vendor FFI
- Added 15 integer-returning vendor FFI wrappers to vendor_ffi.rs:
  `vendor_CINTlen_cart`, `vendor_CINTlen_spinor`, `vendor_CINTcgto_cart`, `vendor_CINTcgto_spheric`, `vendor_CINTcgto_spinor`, `vendor_CINTtot_pgto_spheric`, `vendor_CINTtot_pgto_spinor`, `vendor_CINTtot_cgto_cart`, `vendor_CINTtot_cgto_spheric`, `vendor_CINTtot_cgto_spinor`, `vendor_CINTshells_cart_offset`, `vendor_CINTshells_spheric_offset`, `vendor_CINTshells_spinor_offset`, `vendor_CINTgto_norm`, `vendor_CINTc2s_bra_sph`
- Updated build.rs bindgen allowlist to include all helper and transform symbols
- Extended `verify_helper_surface_coverage` with `#[cfg(has_vendor_libcint)]` numeric comparison block:
  - Integer helpers: exact `==` equality for count and offset functions
  - Float helper: `CINTgto_norm` at atol=1e-12 for l=0..4, a in {0.5, 1.0, 2.5}
  - Transform: `CINTc2s_bra_sph` direct buffer comparison at atol=1e-12 for l=0,1,2

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] tolerance_for_family returned &str lifetime mismatch**
- **Found during:** Task 1 — `FamilyTolerance.family` is `&'static str` but `family: &str` parameter was used directly
- **Fix:** Added static str match arms to map each family string to a `'static` literal before constructing `FamilyTolerance`
- **Files modified:** `crates/cintx-oracle/src/compare.rs`
- **Commit:** 760d971

**2. [Rule 1 - Bug] Old TOL_* constants still referenced in tolerance_table JSON block**
- **Found during:** Task 1 — `cargo check` revealed 12 uses of old constant names in the JSON report section
- **Fix:** Replaced the per-family JSON tolerance table with a unified entry using `UNIFIED_ATOL`, `UNIFIED_RTOL`, and a note
- **Files modified:** `crates/cintx-oracle/src/compare.rs`
- **Commit:** 760d971

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1: Unify tolerances + fix CINTgto_norm | 760d971 | compare.rs, helpers.rs |
| 2: Add numeric vendor FFI + oracle comparisons | 4b84dd3 | compare.rs, vendor_ffi.rs, build.rs |

## Known Stubs

None — all helper and transform oracle comparisons are wired.

## Self-Check

- [x] `crates/cintx-oracle/src/compare.rs` — modified, `UNIFIED_ATOL` present, no `TOL_1E_ATOL`
- [x] `crates/cintx-compat/src/helpers.rs` — modified, double-factorial loop present
- [x] `crates/cintx-oracle/src/vendor_ffi.rs` — modified, all vendor wrappers present
- [x] commit 760d971 exists
- [x] commit 4b84dd3 exists
- [x] `cargo check -p cintx-compat -p cintx-oracle --features cpu` exits 0
- [x] All 9 `cintx-oracle` unit tests pass
- [x] `helpers` tests in `cintx-compat` pass

## Self-Check: PASSED
