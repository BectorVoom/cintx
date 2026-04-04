---
phase: 11-helper-transform-completion-4c1e-real-kernel
plan: 04
subsystem: oracle
tags: [oracle, vendor-ffi, cart-integrals, HELP-03, gap-closure]
dependency_graph:
  requires: [11-03]
  provides: [HELP-03-complete, cart-legacy-oracle-comparison]
  affects: [verify_legacy_wrapper_parity, vendor_ffi]
tech_stack:
  added: []
  patterns: [vendor-ffi-wrapper-pattern, col-major-to-row-major-transpose]
key_files:
  created: []
  modified:
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/src/compare.rs
decisions:
  - "Cart variant supplemental header adds int2c2e_cart and int3c1e_cart only — the other 6 cart functions are already declared in cint_funcs.h"
  - "1e cart comparisons transpose vendor col-major output to row-major matching the sph 1e pattern"
  - "2e/2c2e/3c cart comparisons do NOT transpose — same as sph 2e+ pattern"
metrics:
  duration: 5 min
  completed: 2026-04-04
  tasks_completed: 2
  files_modified: 3
---

# Phase 11 Plan 04: Cart Legacy Oracle Comparison Summary

Cart variant vendor FFI wrappers and numeric oracle comparison blocks close HELP-03 gap for all 8 cart legacy integral symbols against vendored libcint at UNIFIED_ATOL=1e-12.

## Objective

Close the HELP-03 gap by adding vendor FFI wrappers for all 8 cart legacy integral symbols and numeric oracle comparison blocks in `verify_legacy_wrapper_parity`. The sph-only implementation from Plan 03 left cart symbols covered only by surface parity, not numeric oracle comparison.

## What Was Built

### Task 1: build.rs — Cart function allowlist and supplemental declarations

- Added `int2c2e_cart` and `int3c1e_cart` to the supplemental header (they are not in `cint_funcs.h`)
- Extended the `allowlist_function` regex with 8 cart integral names: `int1e_ovlp_cart`, `int1e_kin_cart`, `int1e_nuc_cart`, `int2e_cart`, `int2c2e_cart`, `int3c1e_cart`, `int3c1e_p2_cart`, `int3c2e_ip1_cart`

### Task 2: vendor_ffi.rs and compare.rs — Cart FFI wrappers and comparison blocks

**vendor_ffi.rs additions (8 new functions):**
- `vendor_int1e_ovlp_cart` — 2-shell integral
- `vendor_int1e_kin_cart` — 2-shell integral
- `vendor_int1e_nuc_cart` — 2-shell integral
- `vendor_int2e_cart` — 4-shell integral
- `vendor_int2c2e_cart` — 2-shell integral
- `vendor_int3c1e_cart` — 3-shell integral
- `vendor_int3c1e_p2_cart` — 3-shell integral
- `vendor_int3c2e_ip1_cart` — 3-shell integral

**compare.rs additions in verify_legacy_wrapper_parity:**
- Cart size variables using `CINTcgto_cart` (ni_c, nj_c, nk_c, ni4_c, nj4_c, nk4_c, nl4_c + derived sizes)
- 8 cart comparison blocks following the established sph patterns:
  - 1e cart: transpose vendor col-major to row-major before comparison (matching sph 1e)
  - 2e/2c2e/3c cart: no transpose (matching sph 2e+ pattern)
- Replaced misleading comment "All remaining legacy symbols (cart...) are covered by surface parity" with accurate comment

## Verification Results

All four feature profiles pass:
- `cargo test -p cintx-oracle --features "cpu" --lib` — 9/9 pass
- `cargo test -p cintx-oracle --features "cpu,with-4c1e" --lib` — 9/9 pass
- `cargo test -p cintx-oracle --features "cpu,with-f12" --lib` — 9/9 pass
- `cargo test -p cintx-oracle --features "cpu,with-f12,with-4c1e" --lib` — 9/9 pass

## Commits

| Task | Commit | Message |
|------|--------|---------|
| Task 1 | c7685ca | feat(11-04): add cart function allowlist and supplemental declarations to build.rs |
| Task 2 | d96d6ef | feat(11-04): add cart vendor FFI wrappers and numeric comparison blocks |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all 8 cart comparison blocks are wired to the actual vendor FFI and produce real comparisons. The `verify_legacy_wrapper_parity` function only activates when `has_vendor_libcint` is set (i.e., CINTX_ORACLE_BUILD_VENDOR=1), which is the correct behavior for tests that don't build the vendor library.

## Self-Check: PASSED

Files exist:
- crates/cintx-oracle/build.rs — FOUND (modified)
- crates/cintx-oracle/src/vendor_ffi.rs — FOUND (modified)
- crates/cintx-oracle/src/compare.rs — FOUND (modified)

Commits:
- c7685ca — FOUND
- d96d6ef — FOUND
