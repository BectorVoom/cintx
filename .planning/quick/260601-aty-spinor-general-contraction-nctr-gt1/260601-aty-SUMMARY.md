---
phase: quick-260601-aty
plan: 01
subsystem: cintx-cubecl spinor cart→spinor transform + cintx-oracle parity
tags: [spinor, general-contraction, nctr, cart-to-spinor, vendor-parity, byte-identity]
requires:
  - cart_to_spinor_sf_2d / cart_to_spinor_sf_4d / cart_to_spinor_sf_derivative_2d (already shipped)
  - device scalar 1e/2e kernels emitting per-(ci,cj[,ck,cl]) contraction-major cart blocks
provides:
  - int1e_{ovlp,kin,nuc}_spinor on general-contracted (nctr>1) bases, vendor byte-identical
  - int2e_spinor on general-contracted (nctr>1) bases, vendor byte-identical
  - spinor gradient (int1e_ipovlp_spinor) nctr>1 vendor byte-identity coverage
affects:
  - downstream pyscf-rs F-03 spinor surface (cc-pVDZ / 6-31G general contraction now unblocked)
tech-stack:
  added: []
  patterns:
    - "nctr>1 spinor scatter = per-(ci,cj[,ck,cl]) loop + contraction-major dst (i_global=ci*di+i_sp), reusing the same base index formula as the Spheric/Cart arms"
    - "1e sf_2d path needs the per-sub-block ket-major→bra-major transpose; 2e sf_4d path needs NO transpose (device block is already i-fastest, matching sf_4d's cart input)"
key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs
    - crates/cintx-oracle/tests/oracle_gate_closure.rs
decisions:
  - "Do NOT re-apply contraction coefficients in either spinor arm — the device scalar kernel already accumulated each (ci,cj[,ck,cl]) block with its own per-column coeffs; the arm only transforms+scatters."
  - "Cover the spinor gradient nctr>1 with a REAL vendor byte-identity gate (vendor_int1e_ipovlp_spinor is a genuine libcint driver, not a stub) rather than honest-deferring."
metrics:
  duration: ~30 min
  completed: 2026-06-01
  tasks: 3
  files_changed: 4
  commits: 3
---

# Quick Task 260601-aty: Spinor General Contraction (nctr>1) Summary

Wired general-contraction (nctr>1) support into the 1e and 2e spin-free spinor cart→spinor
arms — removing the two `UnsupportedApi` fail-closed guards — and proved byte-identity to
vendored libcint 6.1.3 at atol=1e-12 on NON-SQUARE general-contracted bases, including the
spinor gradient path.

## What changed

### Task 1 — 1e spinor arm (commit `8d6ebbc`, green)
- Removed the `if n_ctr_i != 1 || n_ctr_j != 1 { UnsupportedApi }` guard from the
  `one_electron.rs` scalar `Representation::Spinor` arm.
- Replaced the single-block transform with a per-`(ci,cj)` loop that mirrors the Spheric arm
  directly above it: each sub-block at `base = (ci*n_ctr_j+cj)*block_len` is transposed
  ket-major→bra-major (the 260529-jtd/kke orientation fix that `cart_to_spinor_sf_2d`
  requires, since unlike `cart_to_spinor_si_2d` it does NOT own the transpose), transformed
  into a per-`(ci,cj)` temp, then scattered contraction-major into the dense interleaved-complex
  staging: `dst = (j_global*ni_sp + i_global)*2`, `ni_sp = n_ctr_i*di`,
  `i_global = ci*di + i_sp`. No coefficient re-apply (the device scalar kernel already
  contracted each column). Added a fail-closed staging guard (T-aty-03) before any write.
- Test harness: `build_two_p_spinor_nctr2()` (p nctr=2 × d nctr=2, NON-SQUARE, distinct
  per-column coeffs, COLUMN-major env coeff block), nctr-aware cintx + vendor collectors,
  `assert_fixture_nctr_gt1` guard, and 3 vendor byte-identity parity tests
  (`test_int1e_{ovlp,kin,nuc}_spinor_nctr2_parity`).

### Task 2 — 2e spinor arm (commit `89fff71`, green)
- Removed the `if n_ctr_i!=1||...||n_ctr_l!=1 { UnsupportedApi }` guard from the
  `two_electron.rs` `Representation::Spinor` arm.
- Replaced the single-block transform with a per-`(ci,cj,ck,cl)` loop mirroring the
  Spheric/Cart arms. CONFIRMED at the source that the device 4D block is i-fastest
  (`block[ic + nfi*(jc + nfj*(kc + nfk*lc))]`, scattered by the Cart arm with NO transpose)
  and that `cart_to_spinor_sf_4d` reads i-fastest cart — so the 2e spinor path needs NO
  transpose (the catch-all vendor parity at atol=1e-12 confirms this was correct). Per-quad
  contraction-major scatter:
  `dst = (((lidx*n2c_k+kidx)*n2c_j+jidx)*n2c_i+iidx)*2`. No coeff re-apply. Fail-closed
  staging guard added.
- Test harness: `build_two_center_spinor_nctr2()` (s/p/p/d, every shell nctr=2, NON-SQUARE
  (k,l) pair, l-sum=4 → nroots=3 ≤ device cap), `oracle_gate_2e_spinor_nctr2` vendor
  byte-identity gate + an always-on `oracle_gate_2e_spinor_nctr2_evaluates` smoke gate.

### Task 3 — spinor gradient nctr>1 (commit `69798aa`, green)
- CONFIRMED: the 1e spinor GRADIENT arms carry NO residual nctr>1 `UnsupportedApi` guard —
  they route through `cart_to_spinor_sf_derivative_2d`, which takes `n_ctr_i/n_ctr_j` and
  composes contraction-major internally (27-03/D-08). No kernel edit was needed.
  `test_ipovlp_spinor_grad_nctr_gt1_evaluates` (one_electron.rs unit test) still passes.
- COVERED (not deferred): `vendor_int1e_ipovlp_spinor` is a genuine libcint driver (NOT a
  return-0/exit(1) stub — contrast `int2c2e_ip1/ip2_spinor` and `CINT3c1e_spinor`), so added
  a real vendor byte-identity gate `test_int1e_ipovlp_spinor_grad_nctr2_parity` on the
  NON-SQUARE p(nctr=2)×d(nctr=2) pair, component-leading interleaved-complex (ncomp=3),
  atol=1e-12, 0 mismatches.

## Verification — double-gated parity actually executed (not skipped)

All parity tests are `#[cfg(has_vendor_libcint)]`, set ONLY by `CINTX_ORACLE_BUILD_VENDOR=1`,
and run under `--features cpu`. Both gates were active for every run below; the tests appear
in the run list with real mismatch/nonzero counts (not "0 tests run" / "ignored"):

- `one_electron_scalar_spinor_parity` (CINTX_ORACLE_BUILD_VENDOR=1, --features cpu):
  **13 passed; 0 failed; 0 ignored.** Includes the 3 nctr2 scalar parity tests, the gradient
  nctr2 parity test, AND the 3 pre-existing nctr==1 `_asym_parity` regression tests.
- `oracle_gate_closure` 2e spinor (double-gated): `oracle_gate_2e_spinor_nctr2` →
  **mismatch_count=0, nonzero=17980/23040**; pre-existing `oracle_gate_2e_spinor` (nctr==1)
  still 0 mismatches.
- `one_electron_grad_spinor_parity` (regression, nctr==1, double-gated): **8 passed; 0 failed.**
- Guard-removal grep: no active `spinor … general contraction` UnsupportedApi guard remains
  in either kernel (only a test comment matches).
- `cargo build -p cintx-cubecl --features cpu`: succeeds, no new errors.

Each nctr2 parity test reports a real nonzero count and a 0-mismatch assertion that executed
against vendored libcint — confirming parity ran, not silently skipped.

## Deviations from Plan

None for Rules 1–3. The plan was executed as written. The 2e arm's "needs NO transpose"
hypothesis (flagged in the plan as CONFIRM-at-source) was verified against the Cart arm and
then validated by the atol=1e-12 vendor gate.

## Deferred Issues (out of scope — see deferred-items.md)

**`oracle_gate_3c2e_spinor` fails — PRE-EXISTING, not a regression.** This test asserts
`INT3C2E_IP1_SPINOR` returns `UnsupportedApi`; it now returns `Ok`. Verified by reverting all
four of this task's files to the base commit `4d60e8d`, rebuilding, and re-running — it fails
identically at base. `INT3C2E_IP1_SPINOR` is a 3-center family routed through `center_3c2e.rs`
(`cart_to_spinor_sf_3c2e`), which this task NEVER touched (only `two_electron.rs` 4-center and
`one_electron.rs` 1e arms changed). Logged to `deferred-items.md` with a follow-up note. Not
fixed here per the scope boundary.

## Open question recorded (scope note, not expanded)

The todo's separate observation — that multi-shell-same-l SEGMENTED bases (e.g. 6-31g 3×s/2×p,
all nctr==1) produce an eigenvalue-identical but globally PERMUTED spinor AO ordering vs PySCF —
is a distinct global-assembly / `ao_loc_2c` ordering-convention concern, NOT this contraction
gap. The contraction-major fix here (`i_global = ci*di + i_sp` WITHIN a shell) governs the
intra-shell layout only; it does not change the inter-shell global stitch order, so it does
**not** by itself reconcile the cross-shell permutation. That needs its own follow-up item
(global `ao_loc_2c` ordering audit). Not expanded into this task's scope.

## Self-Check: PASSED

- Files: all 4 modified files FOUND.
- Commits: `8d6ebbc`, `89fff71`, `69798aa` all FOUND in git log.
