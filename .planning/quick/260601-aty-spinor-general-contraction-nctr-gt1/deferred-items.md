# Deferred / out-of-scope items — quick task 260601-aty

> **RESOLVED 2026-06-01 by quick task 260601-d7e** (both items below). The 3c2e gate
> was stale (the family is byte-identical to vendor — proven by the pre-existing
> `test_int3c2e_ip1_spinor_adversarial_parity`); the gate was reconciled to real vendor
> parity. The global-AO-ordering question was settled: cintx is libcint-faithful (0
> mismatches on a segmented `[s,p,s,p]` basis), so the pyscf-rs permutation is a PySCF
> `ao_loc_2c` convention concern owned by pyscf-rs — see
> `.planning/quick/260601-d7e-spinor-3c2e-gate-and-global-ao-order/CONCLUSION-ao_loc_2c.md`.

## Pre-existing failure (NOT a regression from this task)

**`oracle_gate_3c2e_spinor`** (crates/cintx-oracle/tests/oracle_gate_closure.rs:~1050)

- **Status:** FAILS at the clean base commit `4d60e8d` (verified by reverting all four
  of this task's modified files to base, rebuilding cintx-cubecl, and re-running the
  test — it fails identically: `int3c2e_ip1 spinor must return UnsupportedApi, got: Ok(...)`).
- **What it is:** the test asserts `INT3C2E_IP1_SPINOR` (a 3-center 2-electron derivative
  spinor family) returns `UnsupportedApi` (the R5 fail-closed expectation). It now returns
  `Ok(RawEvalSummary { not0: 4, ... })` — i.e. the family started evaluating, while the test
  still encodes the old "must reject" contract.
- **Why out of scope:** `INT3C2E_IP1_SPINOR` is routed through `center_3c2e.rs`
  (`cart_to_spinor_sf_3c2e` / `cart_to_spinor_sf_derivative_3c2e`), which this task NEVER
  touched. This task only edited the `two_electron.rs` (4-center int2e) and `one_electron.rs`
  (1e) `Representation::Spinor` arms. The 3c2e dispatch is independent.
- **Follow-up:** needs its own item — either the int3c2e_ip1 spinor family is now genuinely
  wired (in which case the test's `must return UnsupportedApi` assertion should be replaced
  with a vendor byte-identity gate against `vendor_int3c2e_ip1_spinor`), or the family
  should still fail-closed and the unexpected `Ok` is a separate latent bug. Out of scope
  for the nctr>1 contraction gap.
