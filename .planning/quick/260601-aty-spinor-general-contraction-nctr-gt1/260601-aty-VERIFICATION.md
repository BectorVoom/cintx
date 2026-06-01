---
phase: quick-260601-aty
verified: 2026-06-01T00:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Quick Task 260601-aty: Spinor General Contraction (nctr>1) Verification Report

**Task Goal:** Close the spinor general-contraction (nctr>1) gap — int1e_{ovlp,kin,nuc}_spinor and int2e_spinor must evaluate on a general-contracted basis and byte-match vendored libcint at atol=1e-12.
**Verified:** 2026-06-01
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | int1e_ovlp_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12 | VERIFIED | `test_int1e_ovlp_spinor_nctr2_parity` passed under double gate (CINTX_ORACLE_BUILD_VENDOR=1 + --features cpu); mismatch_count=0, nonzero>0 confirmed in 13/13 run |
| 2 | int1e_kin_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12 | VERIFIED | `test_int1e_kin_spinor_nctr2_parity` passed; same double-gated run |
| 3 | int1e_nuc_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12 | VERIFIED | `test_int1e_nuc_spinor_nctr2_parity` passed; same double-gated run |
| 4 | int2e_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12 | VERIFIED | `oracle_gate_2e_spinor_nctr2` passed; mismatch_count=0, nonzero=17980/23040 confirmed |
| 5 | The nctr>1 spinor output is contraction-major (i_global = ci*di + i_sp), byte-matching vendor CINTcgto_spinor AO ordering | VERIFIED | Code at one_electron.rs:11078 (`i_global = ci*di + i_sp`) and two_electron.rs:3661 (`iidx = ci*di + i_sp`) implement contraction-major scatter; vendor byte-identity at atol=1e-12 confirms ordering correctness |
| 6 | The two UnsupportedApi nctr>1 guards (1e + 2e) no longer reject general contraction | VERIFIED | Grep of both files returns no `n_ctr_i != 1` or "general contraction" UnsupportedApi guards in active code paths; only test comments and unrelated guards remain |
| 7 | Spinor gradient path under nctr>1 is confirmed evaluating and covered by a vendor parity assertion | VERIFIED | `test_int1e_ipovlp_spinor_grad_nctr2_parity` ran and passed under double gate; `test_ipovlp_spinor_grad_nctr_gt1_evaluates` unit test also passes; vendor_int1e_ipovlp_spinor is a genuine libcint driver (not a stub) |

**Score:** 7/7 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | 1e spinor arm with per-(ci,cj) contraction-major scatter; UnsupportedApi guard removed | VERIFIED | Lines 11015-11087: per-(ci,cj) loop present at line 11055; `cart_to_spinor_sf_2d` at line 11065; no nctr>1 UnsupportedApi guard |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | 2e spinor arm with per-(ci,cj,ck,cl) contraction-major scatter; UnsupportedApi guard removed | VERIFIED | Lines 3602-3677: per-(ci,cj,ck,cl) loop at line 3640; `cart_to_spinor_sf_4d` at line 3646; no nctr>1 UnsupportedApi guard |
| `crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs` | nctr>1 non-square (p×d) spinor fixture + 3 vendor parity tests (ovlp/kin/nuc) + gradient parity test | VERIFIED | `build_two_p_spinor_nctr2()` at line 150 (p×d NON-SQUARE confirmed by `assert_fixture_nctr_gt1`); 3 scalar parity tests at lines 617, 639, 660; gradient parity test at line 697; all gated `#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")]` |
| `crates/cintx-oracle/tests/oracle_gate_closure.rs` | nctr>1 int2e_spinor vendor parity gate | VERIFIED | `build_two_center_spinor_nctr2()` at line 1395; `oracle_gate_2e_spinor_nctr2` at line 1477; `oracle_gate_2e_spinor_nctr2_evaluates` at line 1546; s/p/p/d non-square quarlet |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| one_electron.rs Representation::Spinor arm | cart_to_spinor_sf_2d | per-(ci,cj) loop; per-sub-block ket-major→bra-major transpose; contraction-major scatter into n2c-dim interleaved-complex output | VERIFIED | Code confirmed at lines 11055-11086; transpose present at line 11059-11063 |
| two_electron.rs Representation::Spinor arm | cart_to_spinor_sf_4d | per-(ci,cj,ck,cl) loop; contraction-major scatter into n2c^4 interleaved-complex output; NO transpose (device block already i-fastest) | VERIFIED | Code confirmed at lines 3640-3677; no transpose applied per architecture note at line 3610 |
| parity fixture build (nctr>1) | vendor_CINTcgto_spinor / vendor_int1e_*_spinor | double-gated oracle (cpu + CINTX_ORACLE_BUILD_VENDOR=1), atol=1e-12, count_mismatches==0 | VERIFIED | All 4 parity tests (3 scalar + 1 gradient) ran with real nonzero counts and 0 mismatches; 2e nctr2 gate confirmed mismatch_count=0, nonzero=17980/23040 |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 1e spinor nctr2 scalar parity (ovlp/kin/nuc) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity` | 13 passed; 0 failed; 0 ignored | PASS |
| 1e spinor nctr2 gradient parity | included in above run | `test_int1e_ipovlp_spinor_grad_nctr2_parity ... ok` | PASS |
| 2e spinor nctr2 parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test oracle_gate_closure -- oracle_gate_2e_spinor` | 3 passed; 0 failed; mismatch_count=0, nonzero=17980/23040 | PASS |
| gradient smoke test (nctr>1 evaluates, no UnsupportedApi) | `cargo test -p cintx-cubecl --features cpu --lib -- one_electron::tests::test_ipovlp_spinor_grad_nctr_gt1_evaluates` | 1 passed | PASS |
| Double-gate confirmation (tests actually ran, not skipped) | nonzero counts in output | nonzero=17980/23040 (2e), nonzero counts in 1e output confirm real execution | PASS |

---

## Anti-Patterns Found

None. No TODO/FIXME/placeholder markers or stub return patterns found in the four modified files. Staging guards (fail-closed BufferTooSmall) are present in both new spinor arms, consistent with the OOM-safe stop contract.

---

## Deferred / Out-of-Scope Items

**`oracle_gate_3c2e_spinor` fails (pre-existing, not a regression).**

This test asserts `INT3C2E_IP1_SPINOR` returns `UnsupportedApi` but it returns `Ok(...)`. Confirmed pre-existing:

1. The four files changed by this task are exactly `one_electron.rs`, `two_electron.rs`, `one_electron_scalar_spinor_parity.rs`, and `oracle_gate_closure.rs` — `center_3c2e.rs` was not touched.
2. At the base commit `4d60e8d`, the same test text asserting `UnsupportedApi` was present, meaning the test was already encoding the wrong contract before this task began.
3. Documented in `deferred-items.md` with a follow-up note.

This failure is not actionable under this task's scope. It needs its own item to either update the test to a vendor byte-identity gate (if the 3c2e spinor family is correctly wired) or re-add the UnsupportedApi rejection if the evaluation is a latent bug.

---

## Human Verification Required

None. All acceptance criteria verified programmatically via double-gated oracle.

---

## Summary

The spinor general-contraction (nctr>1) gap is closed. Both UnsupportedApi guards are gone. The 1e and 2e spinor arms implement contraction-major per-(ci,cj[,ck,cl]) scatter using the proven template. Four new vendor parity tests ran against real libcint under the double gate and produced 0 mismatches at atol=1e-12 with nonzero output confirmed. The gradient path was confirmed (no residual guard, routes through `cart_to_spinor_sf_derivative_2d` which already handled nctr>1) and covered with a real vendor parity gate. Pre-existing nctr==1 regression tests continued to pass.

---

_Verified: 2026-06-01_
_Verifier: Claude (gsd-verifier)_
