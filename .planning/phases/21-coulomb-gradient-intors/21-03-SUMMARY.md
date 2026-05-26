---
phase: 21-coulomb-gradient-intors
plan: "03"
subsystem: kernels/one_electron + oracle/vendor_ffi + oracle/tests
tags: [gradient, ipovlp, ipkin, nabla1i, bra-derivative, oracle-parity, GRAD-03, GRAD-04]
dependency_graph:
  requires: [21-02]
  provides: [int1e_ipovlp kernel (cart+sph), int1e_ipkin kernel (cart+sph), vendor_int1e_ipovlp wrappers, vendor_int1e_ipkin wrappers, one_electron_grad_parity tests]
  affects: [cintx-cubecl, cintx-oracle]
tech_stack:
  added: []
  patterns: [bra-nabla (∂/∂Ai) on 1e G-tensor, product-rule ∂T/∂Ai derivation, component-leading 3×ni×nj staging layout]
key_files:
  created:
    - crates/cintx-oracle/tests/one_electron_grad_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs
decisions:
  - "nmax headroom for ipovlp: nmax = li + lj + 1 (one extra bra level so nabla ix+1 access is valid)"
  - "nmax headroom for ipkin: nmax = li + lj + 3 (kinetic +2 for D_j^2 jx+2 access + nabla +1 for ix+1 access)"
  - "ipkin derivation: ∂T/∂Ax = -0.5*(D_j^2(nabla1i_gx)*g0y*g0z + nabla1i_gx*d2y*g0z + nabla1i_gx*g0y*d2z) — product rule with D_j^2/nabla1i commutation (they act on different indices)"
  - "Spinor gradient returns UnsupportedApi (R5/D-03) — guard placed before gradient compute path"
  - "Vendor build: added autocode/grad1.c to cc::Build and int1e_ipovlp/ipkin to bindgen allowlist — these symbols were missing from the earlier vendor build"
  - "Oracle tests use determinism checks (always run) + vendor parity tests (#[cfg(has_vendor_libcint)])"
  - "flip oracle_covered to true for int1e_ipovlp and int1e_ipkin pending plan 21-08 manifest-audit pass"
metrics:
  duration: "~25 min"
  completed: "2026-05-26"
  tasks: 2
  files_modified: 4
---

# Phase 21 Plan 03: int1e_ipovlp + int1e_ipkin Gradient Kernels SUMMARY

**One-liner:** int1e_ipovlp/ipkin 3-component bra-nabla kernels with product-rule ∂T/∂Ai derivation; vendor parity at atol=1e-12 confirmed vs libcint 6.1.3 for all 4 variants (cart+sph).

## What Was Built

### Task 1: 1e Gradient Branches in one_electron.rs

**`contract_grad_1e_bra`** — applies `CINTnabla1i_1e` (bra-derivative) to the overlap G-tensor:
- Formula per axis: `f[jx*dj+ix] = ix * g[jx*dj+(ix-1)] + (-2*ai) * g[jx*dj+(ix+1)]`
- Returns `Vec<f64>` of length `3 * nci * ncj` in component-leading layout
- Requires nmax = li + lj + 1 headroom in the G-tensor

**`contract_ipkin`** — applies `∂T/∂Ai` to the kinetic 1e G-tensor using the product rule:
- Derived from `T_ij = -0.5 * (d2x*gy*gz + gx*d2y*gz + gx*gy*d2z)` with `∂/∂Ax`
- Key insight: `D_j^2` and `nabla1i` commute (act on different indices: j vs i)
- Therefore: `∂T/∂Ax = -0.5*(D_j^2(g1x)*g0y*g0z + g1x*d2y*g0z + g1x*g0y*d2z)` where `g1x = nabla1i(g0x)`
- Requires nmax = li + lj + 3 (kinetic +2 AND nabla +1)

**Dispatcher extension** — added `is_ipovlp` and `is_ipkin` branches; spinor guard → `UnsupportedApi` (R5/D-03).

**3-component staging** — writes component-leading `staging[comp * ni*nj + n]` for both Spheric and Cart representations.

**Unit tests added (6 tests):**
- `test_ipovlp_component_count` — s-s → 3, p-s → 9
- `test_ipovlp_determinism` — bit-identical across two calls
- `test_ipovlp_z_component_z_displacement` — z-component nonzero, x/y components ~0 for z displacement
- `test_ipovlp_spinor_returns_unsupported` — UnsupportedApi for spinor
- `test_ipkin_component_count` — s-s → 3
- `test_ipkin_determinism` — bit-identical across two calls

### Task 2: Vendor FFI Wrappers + Oracle Parity Tests

**`vendor_ffi.rs`** — added 4 new wrappers:
- `vendor_int1e_ipovlp_sph`, `vendor_int1e_ipovlp_cart`
- `vendor_int1e_ipkin_sph`, `vendor_int1e_ipkin_cart`
Each takes `out: &mut [f64]` of size `3 * ni * nj` and calls the corresponding `ffi::` symbol.

**`build.rs`** — added:
- `autocode/grad1.c` to the cc::Build vendor library (contains `CINTgout1e_int1e_ipovlp`, etc.)
- `int1e_ipovlp_sph|int1e_ipovlp_cart|int1e_ipkin_sph|int1e_ipkin_cart` to bindgen allowlist

**`one_electron_grad_parity.rs`** — 8 tests (4 determinism + 4 vendor parity):
- Determinism: two calls are bit-identical (always run)
- Vendor parity: `count_mismatches(&vendor, &cintx, 1e-12, 0.0) == 0` for all 4 variants

## nmax Headroom Record

| Operator | nmax Formula | Reason |
|----------|-------------|--------|
| ipovlp   | li + lj + 1 | bra nabla accesses g[ix+1] → need 1 extra bra level |
| ipkin    | li + lj + 3 | kinetic accesses g[jx+2] (need +2) AND bra nabla accesses g[ix+1] (need +1) |

## Oracle Parity Results (H2O STO-3G, atol=1e-12)

All 4 vendor parity tests pass with 0 mismatches at atol=1e-12:

| Operator | Representation | Vendor build | Result |
|----------|----------------|--------------|--------|
| int1e_ipovlp | sph | libcint 6.1.3 | 0 mismatches |
| int1e_ipovlp | cart | libcint 6.1.3 | 0 mismatches |
| int1e_ipkin | sph | libcint 6.1.3 | 0 mismatches |
| int1e_ipkin | cart | libcint 6.1.3 | 0 mismatches |

The nonzero sentinel confirms the gradient matrix is not all-zero for all 4 variants.

## Component-Leading Layout Confirmed

Output is written as `staging[comp * ni*nj + n]` where `comp ∈ {0,1,2}` and `block_len = ni*nj`. This matches the `[comp * block_len + n]` F-order pattern from PATTERNS §Output component-leading F-order layout and matches vendor libcint's `int1e_ipovlp_sph` output confirmed by the parity tests.

## Note for Manifest Audit

Plans 21-02 set `oracle_covered: false` for int1e_ipovlp and int1e_ipkin. Now that the oracle parity tests are green, these entries should be flipped to `oracle_covered: true` in the manifest lock. This flip is appropriate in a follow-up plan (21-08 manifest-audit pass) once all Wave 2 plans are complete.

## Verification Results

- `cargo test -p cintx-cubecl --features cpu ipovlp` — 4 passed, 0 failed
- `cargo test -p cintx-cubecl --features cpu ipkin` — 2 passed, 0 failed
- `cargo test -p cintx-oracle --features cpu --test one_electron_grad_parity` — 4 passed (determinism), 0 failed
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_parity` — 8 passed (determinism + vendor parity), 0 failed
- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` — 0 errors

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] usize subtraction overflow in contract_ipkin**
- **Found during:** Task 1 (ipkin tests)
- **Issue:** `g[off + nx - 2 * dj]` with usize arithmetic panicked when j < 2 even though the coefficient `jf*(jf-1)` is zero (the multiplication happens at runtime but the index is evaluated first in debug mode)
- **Fix:** Changed to `let g0_lo = if j >= 2 { g[off + nx - 2 * dj] } else { 0.0 };` — guard before the index access
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Commit:** 64d5f6f (already in the same commit)

**2. [Rule 2 - Missing vendor build files] autocode/grad1.c not in cc::Build**
- **Found during:** Task 2 (vendor build attempt)
- **Issue:** The `int1e_ipovlp_sph` and `int1e_ipkin_sph` symbols are implemented in `autocode/grad1.c` which was not included in the vendor build — bindgen error `cannot find function int1e_ipovlp_sph in module ffi`
- **Fix:** Added `autocode/grad1.c` to cc::Build `.file()` list and to the rerun-if-changed triggers; added symbols to bindgen allowlist
- **Files modified:** crates/cintx-oracle/build.rs
- **Commit:** 1482244

**3. [Rule 1 - Bug] spinor test tried Spinor representation for workspace query**
- **Found during:** Task 1 (spinor reject test)
- **Issue:** The test tried to use `Representation::Spinor` for `query_workspace` which itself rejects spinor for ipovlp (not the kernel guard we want to test)
- **Fix:** Use `Representation::Spheric` for workspace query then force `plan.representation = Representation::Spinor` to exercise the kernel guard directly
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Commit:** 64d5f6f (already in the same commit)

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. The changes are:
1. In-process Rust kernel math (G-tensor manipulation)
2. Vendor FFI wrappers (unsafe but guarded by `#[cfg(has_vendor_libcint)]`)
3. Test-only file additions

T-21-03-01 (G-tensor headroom) is mitigated: nmax = li + lj + 1 for ipovlp, li + lj + 3 for ipkin; both have been verified via unit tests and oracle parity.

T-21-03-02 (zero-fill silent-wrong) is mitigated: the nonzero sentinel in each parity test rejects a zero-fill output.

T-21-03-03 (spinor gradient) is mitigated: `UnsupportedApi` guard confirmed by unit test.

## Self-Check: PASSED

Files confirmed present:
- crates/cintx-cubecl/src/kernels/one_electron.rs — `contract_grad_1e_bra`, `contract_ipkin`, `is_ipovlp`, `is_ipkin` dispatcher branches present
- crates/cintx-oracle/src/vendor_ffi.rs — `vendor_int1e_ipovlp_sph` at line 410, `vendor_int1e_ipkin_cart` at line 497
- crates/cintx-oracle/tests/one_electron_grad_parity.rs — 8 tests present
- crates/cintx-oracle/build.rs — `autocode/grad1.c` and allowlist additions present

Commits confirmed:
- 64d5f6f: feat(21-03): implement int1e_ipovlp + int1e_ipkin gradient branches (bra-derivative nabla)
- 1482244: feat(21-03): add vendor FFI wrappers + oracle parity tests for ipovlp/ipkin (cart+sph)
