---
phase: quick-260529-jtd
plan: 01
subsystem: 1e-integrals
tags: [spinor, gradient, int1e, c2spinor, oracle-parity, libcint-compat]
requires:
  - "On-device 3-component Cartesian 1e gradient kernels (quick 260529-j7d)"
  - "Host-side cart_to_spinor_sf_2d transform (c2s_sf_1e analogue)"
provides:
  - "int1e_ipovlp_spinor / int1e_ipkin_spinor / int1e_ipnuc_spinor / int1e_iprinv_spinor evaluate for nctr=1"
  - "Vendor parity coverage for the spinor int1e gradient family"
affects:
  - "crates/cintx-cubecl/src/kernels/one_electron.rs"
  - "crates/cintx-oracle/src/vendor_ffi.rs"
  - "crates/cintx-oracle/build.rs"
tech-stack:
  added: []
  patterns:
    - "Device-cart-compute -> host cart->spinor transform (per-component), mirroring scalar spinor 1e"
key-files:
  created:
    - "crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs"
  modified:
    - "crates/cintx-cubecl/src/kernels/one_electron.rs"
    - "crates/cintx-oracle/src/vendor_ffi.rs"
    - "crates/cintx-oracle/build.rs"
decisions:
  - "cart->spinor transform stays host-side per project convention; only the Cartesian gradient runs on-device"
  - "nctr>1 spinor gradient keeps the UnsupportedApi guard (same as the scalar spinor path)"
  - "nroots>5 nuclear spinor gradient keeps the fail-closed guard"
metrics:
  duration_min: 13
  completed: 2026-05-29
---

# Phase quick-260529-jtd Plan 01: Spinor int1e Gradient (ipovlp/ipkin/ipnuc/iprinv) Summary

Wired the four previously-UnsupportedApi spinor int1e gradient operators
(`int1e_{ipovlp,ipkin,ipnuc,iprinv}_spinor`) through the existing on-device
3-component Cartesian gradient and the host-side spin-free `cart_to_spinor_sf_2d`
transform (applied per component), and proved byte-level parity vs libcint 6.1.3
(0 mismatches at atol=1e-12 for all four operators).

## What changed

- **Removed** the Risk R5 / D-03 spinor-gradient rejection block in
  `launch_one_electron_typed` (`one_electron.rs`).
- **Implemented** the `Representation::Spinor` arm of the 1e gradient output-staging
  match: per-component call to `cart_to_spinor_sf_2d::<F>`, mirroring the scalar
  spinor 1e arm. Keeps the `nctr>1` `UnsupportedApi` guard and (upstream) the
  `nroots>5` fail-closed guard for the nuclear operators.
- **Added** four vendor FFI wrappers `vendor_int1e_ip{ovlp,kin,nuc,rinv}_spinor`
  (output `3 * ni_sp * nj_sp * 2` f64, component-leading interleaved complex) and
  added the four symbols to the bindgen allowlist in `build.rs`.
- **Added** `crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs`: a
  double-gated (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`) H2O/STO-3G spinor
  (kappa=0) parity test for all four operators plus always-on smoke tests; the
  iprinv test sweeps the rinv origin over all three nuclei.

## TDD flow

- **RED** (f9e830a): replaced the four `test_ip*_spinor_returns_unsupported` cubecl
  tests with positive-behavior tests (ipovlp/ipkin/ipnuc/iprinv evaluate for nctr=1)
  + an nctr>1 guard test; added an ipkin positive test (none previously existed).
  Confirmed the four positive tests fail against the still-present rejection.
- **GREEN** (fb02060): removed the rejection, implemented the spinor staging arm;
  the five cubecl tests pass and the full cintx-cubecl suite stays green (246 lib).

## Vendor parity (Task 3) — honest mismatch counts vs libcint 6.1.3

Ran with BOTH gates: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle
--features cpu --test one_electron_grad_spinor_parity`. The vendor was built and
`has_vendor_libcint` was active (the vendor-gated parity bodies executed).

| Operator              | Mismatches (atol=1e-12) |
| --------------------- | ----------------------- |
| int1e_ipovlp_spinor   | 0                       |
| int1e_ipkin_spinor    | 0                       |
| int1e_ipnuc_spinor    | 0                       |
| int1e_iprinv_spinor   | 0 (per origin, all 3 nuclei swept) |

First vendor run (before the transpose fix): ipovlp=0, ipkin=0, **ipnuc=32**,
**iprinv=58** mismatches — see Deviations below. After the fix: all 0.

No regression in the existing sph/cart gradient vendor parity:
`one_electron_grad_parity` (8 passed) and `one_electron_nuc_grad_parity` (6 passed).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cart block orientation for the spinor transform**
- **Found during:** Task 3 (vendor parity).
- **Issue:** The device gradient kernels emit each per-component Cartesian block
  ket-major / bra-fastest (`block[cj*nci + ci]`), but `cart_to_spinor_sf_2d` reads
  its `cart` argument bra-major / ket-fastest (`cart[bra*ncj + ket]`, per
  `c2spinor.rs apply_bra_block`: `cart[n*ncj + j]`). Feeding the kernel block
  directly produced an anti-hermitian-transpose error for the *asymmetric* nuclear
  operators (ipnuc=32, iprinv=58 mismatches); the symmetric-per-block ipovlp/ipkin
  happened to be transpose-invariant for these shells and passed, masking the issue.
- **Fix:** Transpose each per-component block into bra-major before the cart->spinor
  transform. All four operators now match libcint at 0 mismatches.
- **Files modified:** `crates/cintx-cubecl/src/kernels/one_electron.rs`
- **Commit:** e1dae40

## Out-of-scope observation (NOT fixed)

The SCALAR spinor 1e path (`one_electron.rs` ~line 2871) feeds the same ket-major
cart block to `cart_to_spinor_sf_2d` WITHOUT transposing. The existing scalar
spinor tests are idempotency-only (no vendor parity for `*_spinor` in
`safe_api_arity2_parity.rs`, which covers only cart/sph OperatorIds), so this latent
orientation question is untested for p-and-higher shells on the scalar path. This is
outside the scope of this task (the scalar path is not exercised here) and is NOT
modified. Logged here for a future scalar-spinor vendor-parity task.

## Known Stubs

None. All four operators compute real device-cart values and a real host transform;
no placeholder/empty-data paths were introduced.

## Commits

- f9e830a `test(quick-260529-jtd): spinor int1e gradient RED + vendor FFI/bindgen/parity scaffolding`
- fb02060 `feat(quick-260529-jtd): implement spinor int1e gradient staging arm (GREEN)`
- e1dae40 `fix(quick-260529-jtd): transpose cart block to bra-major for spinor gradient`

## Self-Check: PASSED

- Files: all 4 present (parity test created; 3 source files modified).
- Commits: f9e830a, fb02060, e1dae40 all present in git log.
