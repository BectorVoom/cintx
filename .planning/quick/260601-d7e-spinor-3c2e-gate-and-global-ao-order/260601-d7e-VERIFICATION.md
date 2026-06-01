---
phase: quick-260601-d7e
verified: 2026-06-01T19:15:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Quick Task 260601-d7e Verification Report

**Task Goal:** (A) Fix the stale oracle_gate_3c2e_spinor must-reject test by reconciling against the real vendor int3c2e_ip1_spinor driver; (B) prove+document that cintx's global spinor AO (ao_loc_2c) ordering is libcint-faithful via a multi-shell vendor parity test, with NO PySCF-compat ordering mode added.
**Verified:** 2026-06-01T19:15:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | oracle_gate_3c2e_spinor no longer asserts the stale R5 UnsupportedApi must-reject contract; reconciled with test_int3c2e_ip1_spinor_adversarial_parity as primary byte-identity source; two gates CONSISTENT | VERIFIED | Function renamed to `oracle_gate_3c2e_ip1_spinor_vendor_parity`; old function body gone; doc-comment at lines 1330-1338 records history and cross-references adversarial parity; both gates assert byte-identity (consistent, not contradictory) |
| 2 | INT3C2E_IP1_SPINOR confirmed byte-identical to vendor (aux-k SPHERICAL = 2lk+1*nctr); test_int3c2e_ip1_spinor_adversarial_parity passes under double gate | VERIFIED | Test executed: `1 passed; 0 failed; 0 ignored` under `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu`; new gate also passes: `mismatches=0, vendor_nonzero=4/24, cintx_nonzero=4/24` |
| 3 | Multi-shell (>=3 shells), segmented same-l, spinor GLOBAL-assembly parity test exists comparing cintx full spinor matrix vs vendored libcint element-by-element | VERIFIED | `spinor_global_ao_order_parity.rs` (382 lines); 4-shell [s,p,s,p] fixture; shell-count-generic collectors; vendor parity test `test_spinor_global_ao_order_parity` + smoke `test_spinor_global_ao_order_evaluates` |
| 4 | Global-ordering test includes at least one l>0 shell so a non-square block participates and ordering is non-trivial | VERIFIED | Fixture has shells l=[0,1,0,1]; p-shells give dim=6 (4l+2=6); `assert_fixture_segmented_same_l` guard enforces >=3 shells, repeated l, >=1 l>0, all nctr==1; fixture guard called in both tests |
| 5 | CONCLUSION doc records the libcint-faithful disposition of cintx's global spinor AO ordering (prove + document; NO PySCF-compat ordering mode added) | VERIFIED | CONCLUSION-ao_loc_2c.md exists; states "libcint-faithful"; records "0 mismatches"; explicitly states "NO PySCF-compat ordering mode added (locked decision)"; references ao_loc_2c throughout |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-oracle/tests/oracle_gate_closure.rs` | oracle_gate_3c2e_spinor updated to honest consistent state | VERIFIED | Contains `oracle_gate_3c2e_ip1_spinor_vendor_parity` at line 1349; stale function removed; all checks pass |
| `crates/cintx-oracle/tests/spinor_deriv_parity.rs` | Pre-existing `test_int3c2e_ip1_spinor_adversarial_parity` not ignored | VERIFIED | Function at line 274; no `#[ignore]` attribute; confirmed passing under double gate |
| `crates/cintx-oracle/tests/spinor_global_ao_order_parity.rs` | Multi-shell global spinor AO ordering vendor parity test, min 80 lines | VERIFIED | 382 lines; [s,p,s,p] 4-shell fixture; generic collectors; fixture guard; vendor + smoke tests |
| `.planning/quick/260601-d7e-spinor-3c2e-gate-and-global-ao-order/CONCLUSION-ao_loc_2c.md` | Documented disposition of ao_loc_2c ordering question | VERIFIED | 58 lines; complete finding including mismatch count (0), libcint-faithful conclusion, no PySCF-compat mode |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| oracle_gate_3c2e_ip1_spinor_vendor_parity | test_int3c2e_ip1_spinor_adversarial_parity (spinor_deriv_parity.rs) | doc-comment cross-reference | VERIFIED | Lines 1337-1338 and 1430, 1438 reference the adversarial parity test by name |
| oracle_gate_3c2e_ip1_spinor_vendor_parity | vendor_ffi::vendor_int3c2e_ip1_spinor | byte-identity gate call with SPHERICAL aux-k | VERIFIED | Called at line 1369; aux-k sized via `vendor_CINTcgto_spheric` (line 1363); buffer length guard + count_mismatches inline |
| spinor_global_ao_order_parity.rs | INT1E_OVLP_SPINOR + vendor_int1e_ovlp_spinor | per-shell-pair eval_raw, stitch_block, count_mismatches==0 | VERIFIED | `count_mismatches` called at line 368; both global collectors loop all shell pairs; stitch_block at line 283/315; assertion at lines 369-374 |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| test_int3c2e_ip1_spinor_adversarial_parity passes under double gate | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_int3c2e_ip1_spinor_adversarial_parity` | `1 passed; 0 failed; 0 ignored; 8 filtered out` | PASS |
| oracle_gate_3c2e_ip1_spinor_vendor_parity passes under double gate | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test oracle_gate_closure oracle_gate_3c2e` | `1 passed; 0 failed; 0 ignored; 13 filtered out; mismatches=0, nonzero=4/24 both sides` | PASS |
| spinor_global_ao_order_parity passes under double gate | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_global_ao_order_parity` | `2 passed; 0 failed; 0 ignored; 0 filtered out; mismatches=0, n_sp=16, elems=512` | PASS |
| Parity tests actually RAN (not silently skipped) | Inspect output for real mismatch counts | All three outputs show real mismatch counts and nonzero element counts, 0 tests filtered/ignored on parity bodies | PASS |

---

## Regression Check

| Check | Status | Evidence |
|-------|--------|---------|
| Task did NOT touch two_electron.rs / one_electron.rs Spinor arms | VERIFIED | `git diff --name-only 6c2de94~1 59cb40d` shows exactly 3 files: oracle_gate_closure.rs, spinor_global_ao_order_parity.rs, CONCLUSION-ao_loc_2c.md |
| cintx-compat/src/helpers.rs CINTshells_spinor_offset NOT modified | VERIFIED | Not in changed file list; helpers.rs last touched before task commits (commit 8db9fcb, well prior) |
| No .planning/phases/** or .planning/research/** files modified | VERIFIED | Diff output contains only the 3 expected files |

---

## Anti-Patterns Found

None. The new test file uses substantive assertions (count_mismatches, assert_any_nonzero, fixture guard), real fixture data with distinct per-shell exponents/coefficients, and real vendor comparisons. No stub patterns, no hardcoded-empty data flowing to assertions.

---

## Human Verification Required

None. All byte-identity claims are verified by actual test execution under the double gate. The test output confirms parity assertions ran and produced real mismatch counts (not silently skipped).

---

## Summary

Task 260601-d7e is fully achieved. Both sub-problems are closed with vendor byte-identity evidence:

- **Sub-problem A:** The stale R5 `UnsupportedApi` must-reject gate is gone. `oracle_gate_3c2e_ip1_spinor_vendor_parity` replaces it with a real vendor byte-identity gate (aux-k SPHERICAL, buffer-length equality guard, count_mismatches==0, both sides nonzero) and cross-references `test_int3c2e_ip1_spinor_adversarial_parity` as the primary adversarial coverage. Both gates are consistent. Branch-3a confirmed: the family evaluates and is byte-identical to vendor.

- **Sub-problem B:** `spinor_global_ao_order_parity.rs` proves cintx's global spinor AO ordering is libcint-faithful on a segmented [s,p,s,p] basis (0 mismatches, n_sp=16, elems=512). CONCLUSION-ao_loc_2c.md documents this finding with the actual measured mismatch count and explicitly records that no PySCF-compat ordering mode was added, per the locked decision.

All three tests passed under the required double gate (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`). The task scope was contained to oracle test files and the planning doc — no dispatch code, no nctr>1 paths, no helpers were modified.

---

_Verified: 2026-06-01T19:15:00Z_
_Verifier: Claude (gsd-verifier)_
