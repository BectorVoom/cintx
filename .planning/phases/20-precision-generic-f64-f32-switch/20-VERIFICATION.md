---
phase: 20-precision-generic-f64-f32-switch
verified: 2026-05-21T18:00:00Z
status: passed
score: 7/7
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 5/7
  gaps_closed:
    - "PREC-02: spinor/complex outputs propagate as Complex<F> — complex_values() accessor implemented on IntegralTensor<F> and TypedEvaluationOutput<F>; num-complex 0.4 direct dep added; spinor_evaluate_exposes_complex_values_some_prec02 test gates the truth (commit c888021)"
    - "PREC-05: f32 path length-contract correctness — CR-01 fixed in all 7 kernels, CR-02+WR-01 fixed in f12 typed inner, WR-03/WR-04/WR-05/WR-06 math hardening applied; vendor-gated f32 parity test for int2e_stg_ip1_sph (ncomp=3) added and passes at max_rel_error=3.022e-8 (commits 5ba79fb, 69a6a18, a63535a)"
  gaps_remaining: []
  regressions: []
---

# Phase 20: Generic Float Precision (f64/f32 Switch) — Re-Verification Report

**Phase Goal:** cintx parameterizes its compute path over a generic float type `F: Float` so callers evaluate integrals in f64 (default, byte-identity) or f32 (loose-tolerance, unlocks adapters lacking `SHADER_F64`). Precision is chosen at the call site via a method-level generic; `evaluate()` continues to mean f64 and every existing call site compiles unchanged.
**Verified:** 2026-05-21T18:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure by plans 20-09, 20-10, 20-11

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PREC-01: generic `F: Float` threaded through full compute path | VERIFIED | Unchanged from initial verification. All math modules and kernel launchers confirmed generic over F; f64 FROZEN tables; monomorphization default preserved. |
| 2 | PREC-02: spinor/complex outputs propagate as `Complex<F>` | VERIFIED | `complex_values() -> Option<Vec<num_complex::Complex<F>>>` added to `IntegralTensor<F>` and `TypedEvaluationOutput<F>` (api.rs lines 564–597). `chunks_exact(2).map(Complex::new)` reinterpretes the contiguous interleaved Vec<F>. num-complex = "0.4" direct dep in cintx-rs/Cargo.toml (line 15). Test `spinor_evaluate_exposes_complex_values_some_prec02` (api.rs line 1303) drives a real Spinor evaluate() and asserts complex_values().is_some(), len == owned_values.len()/2, and at least one nonzero element. Five additional unit tests cover the None path, f32 type threading, delegation, and backward-compat field access. Commit c888021 (Task 2) and 4736f2f (Task 1). owned_values: Vec<F> field order/names unchanged (SemVer). |
| 3 | PREC-03: raw compat env/atm/bas + C ABI remain f64 | VERIFIED | Unchanged from initial verification. cintx-compat and cintx-capi confirmed f64 throughout. |
| 4 | PREC-04: f64 path byte-identical; existing oracle gates/manifest locks/tests pass unchanged | VERIFIED | f64 integration oracle confirmed green post-gap-closure: 11/11 integration test suites pass at atol=1e-12 (one_electron 6/6, two_electron, center_2c2e, center_3c1e, center_3c2e, f12 15/15, safe_api, ecp variants). 180 cintx-cubecl lib tests pass (20-11 SUMMARY). Pre-existing compare::tests lib failures (4 tests, CINTshells_cart_offset[4] mismatch) remain unchanged and independently confirmed as pre-existing/environmental (not caused by any phase 20 or gap-closure change). |
| 5 | PREC-05: f32 path has a separate oracle gate at ~1e-4 rtol verified against libcint — just not byte-identical | VERIFIED | CR-01 fix applied in all 7 kernel files: `out_elems = staging.len()` captured pre-cast; `staging_f32[..out_elems]` passed to typed inner; BufferTooSmall guard added. CR-02+WR-01 fix in f12 typed inner: `staging_f64 = vec![0.0_f64; out_elems]` (not staging.len()); readback and not0 bounded to out_elems. WR-03 (pdata compute_pdata_host converts all inputs to f64 first). WR-04 (to_f64().expect everywhere in math layer — boys.rs, stg.rs, rys.rs, pdata.rs). WR-05 (F::epsilon() host + F::EPSILON device for Boys convergence, precision-appropriate). WR-06 (nonzero_threshold = 1e-12 for f32, 1e-18 for f64, in all 7 kernels). New vendor-gated test `test_f32_int2e_stg_ip1_sph_parity` in f32_parity.rs (line 1050): gates on `#[cfg(all(has_vendor_libcint, feature = "with-f12"))]`; exercises ncomp=3 (staging_elements > chunk_len — the CR-01/CR-02 corruption regime); 3 symmetric quartets pass at max_rel_error=3.022e-8 (floor 1e-4). f32_parity.rs now has 12 vendor-gated tests (11 existing + 1 new). Commits 5ba79fb, 69a6a18, a63535a. |
| 6 | PREC-06: f32 path does NOT gate on SHADER_F64 | VERIFIED | Unchanged from initial verification. executor.rs check_capability() returns Ok(()) early for PrecisionKind::F32. |
| 7 | PREC-07: refactor via serena MCP symbol-aware tools, not blind text replacement | VERIFIED (process gate) | Unchanged from initial verification. .serena/ present; code structure and FROZEN values consistent with symbol-aware editing. |

**Score:** 7/7 truths verified

### Deferred Items

DI-01 is the only open item identified during gap-closure — it is a pre-existing bug, not a gap-closure failure.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| DI-01: Pre-existing f12 asymmetric-quartet value-swap | Not in phase 20 scope | Documented in `.planning/phases/20-precision-generic-f64-f32-switch/deferred-items.md`. A wrong index ordering in the f12 ip1 sub-kernel produces wrong values for non-symmetric quartets with p-shells in asymmetric positions. Present in both f64 and f32 paths (not caused by CR-01/CR-02 or any phase 20 change). The new f32 parity test is correctly restricted to the same 3 symmetric quartets as the f64 oracle — those quartets do NOT trigger DI-01, and they fully exercise the CR-01/CR-02 staging_elements > chunk_len regime. DI-01 requires a separate debug/fix plan and does NOT block the PREC-05 truth (the length-contract gap). |

### DI-01 Assessment for PREC-05

The prompt asks explicitly whether restricting the new test to symmetric quartets still adequately closes the PREC-05 length-contract gap. The answer is yes:

The PREC-05 gap was specifically about **CR-01/CR-02** — the unsound staging buffer length contract on the F32 path where copy_len and not0 scan used `staging.len()` (doubled after bytemuck cast) instead of the true output element count. The CR-01/CR-02 regime is triggered by `staging_elements > chunk_len`, which happens whenever ncomp > 1. The three symmetric test quartets all have ncomp=3:

- [0,1,0,1]: staging_elements=3 > chunk_len=3 (the old broken code produced 6, doubling with stale zeros)
- [3,4,3,4]: same
- [0,2,0,2]: staging_elements=27 > chunk_len=27 (before fix: 54)

All three quartets exercise the exact length-contract regime the gap documented. DI-01 is an orthogonal kernel correctness bug (wrong index ordering for asymmetric quartets) that exists in the f64 path too and is independent of float precision handling. It does not bear on the PREC-05 gap closure.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-rs/Cargo.toml` | direct num-complex dependency | VERIFIED | `num-complex = "0.4"` at line 15; matches transitive 0.4.6 already in Cargo.lock — no new transitive package |
| `crates/cintx-rs/src/api.rs` | `complex_values()` on IntegralTensor<F> and TypedEvaluationOutput<F> | VERIFIED | impl blocks at lines 564–597; chunks_exact(2) reinterpretation; None for real, Some for spinor; owned_values unchanged |
| `crates/cintx-rs/src/api.rs` | `spinor_evaluate_exposes_complex_values_some_prec02` test | VERIFIED | Lines 1303–1339; end-to-end spinor evaluate() asserts complex_values().is_some() and nonzero content |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | CR-01 fix (out_elems pre-cast, BufferTooSmall guard) | VERIFIED | Lines 651–659: `out_elems = staging.len()`, `staging_f32[..out_elems]`, BufferTooSmall guard present |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | CR-01 fix | VERIFIED | Lines 749–757: identical pattern |
| `crates/cintx-cubecl/src/kernels/center_2c2e.rs` | CR-01 fix | VERIFIED | Lines 432–434: identical pattern |
| `crates/cintx-cubecl/src/kernels/center_3c1e.rs` | CR-01 fix | VERIFIED | Lines 501–503: identical pattern |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` | CR-01 fix | VERIFIED | Lines 509–511: identical pattern |
| `crates/cintx-cubecl/src/kernels/center_4c1e.rs` | CR-01 fix | VERIFIED | Lines 761–763: identical pattern |
| `crates/cintx-cubecl/src/kernels/f12.rs` | CR-01 outer arm + CR-02 typed inner fix | VERIFIED | Outer arm (lines 1648–1656): out_elems pre-cast, staging_f32[..out_elems]. Typed inner (lines 1559–1570): staging_f64 sized to out_elems, readback/not0 bounded. WR-06 threshold at line 1606. |
| `crates/cintx-cubecl/src/math/pdata.rs` | WR-03: f64-first compute_pdata_host | VERIFIED | Lines 158–167: all inputs converted to f64 via to_f64().expect() before arithmetic |
| `crates/cintx-cubecl/src/math/boys.rs` | WR-04: to_f64().expect + WR-05: F::epsilon()/F::EPSILON | VERIFIED | Host: line 139 `F::epsilon() * e`; device: line 234 `F::EPSILON * e`; DBL_EPSILON_HALF removed |
| `crates/cintx-cubecl/src/math/stg.rs` | WR-04: to_f64().expect | VERIFIED | Lines 375–376: `to_f64().expect("CintFloat is f32|f64; to_f64 is total")` |
| `crates/cintx-oracle/tests/f32_parity.rs` | `test_f32_int2e_stg_ip1_sph_parity` (ncomp=3, vendor+with-f12 gated) | VERIFIED | Lines 1049–1166: `#[cfg(all(has_vendor_libcint, feature = "with-f12"))]`; 3 symmetric quartets; CR-01/CR-02 regime documented inline; anti-zero sentinel + finite sentinel; 12 total tests in file |
| `.planning/phases/20-precision-generic-f64-f32-switch/deferred-items.md` | DI-01 documented | VERIFIED | File exists; DI-01 describes the asymmetric-quartet value-swap pre-existing bug, scope, mitigation, and resolution path |

All artifacts from the initial verification (PREC-01, PREC-03, PREC-06 families) remain verified unchanged.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `IntegralTensor<F>.complex_values()` | `num_complex::Complex<F>` | `chunks_exact(2).map(Complex::new)` | WIRED | None guard on complex_interleaved==false; Some path with debug_assert_eq!(len%2,0) |
| `TypedEvaluationOutput<F>.complex_values()` | `IntegralTensor<F>.complex_values()` | thin delegation | WIRED | One-liner: `self.tensor.complex_values()` |
| `spinor_evaluate_exposes_complex_values_some_prec02` | `SessionRequest::evaluate()` with Spinor | OperatorId::new(2) + Representation::Spinor | WIRED | End-to-end path through query_workspace() and evaluate() confirmed by test |
| `launch_f12` F32 outer arm | `launch_f12_typed::<f32>` | `&mut staging_f32[..out_elems]` (post CR-01) | WIRED | out_elems captured pre-cast; typed inner receives correctly-sized slice |
| `launch_f12_typed` F32 arm | `staging_f64` temp buffer | `vec![0.0_f64; out_elems]` (post CR-02) | WIRED | Sized to out_elems not staging.len(); readback bounded to out_elems |
| `test_f32_int2e_stg_ip1_sph_parity` | `evaluate_generic::<f32>()` | `collect_stg_ip1_f32()` helper | WIRED | collect_stg_ip1_f32 calls evaluate_generic::<f32>() at line 1019; result consumed and compared against vendor reference |
| All 5 previously verified key links | (unchanged) | (unchanged) | WIRED | evaluate_generic, evaluate() shim, check_capability, boys_gamma_inc dispatch, f32_tolerance_for_family — all unchanged |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `api.rs complex_values()` | `owned_values: Vec<F>` (via chunks_exact) | existing interleaved spinor/complex kernel output | Yes — reinterpretation of real kernel output; spinor smoke test asserts nonzero complex elements | FLOWING |
| `f32_parity.rs test_f32_int2e_stg_ip1_sph_parity` | `f32_out: Vec<f32>` | `collect_stg_ip1_f32()` → `evaluate_generic::<f32>()` → f12 kernel | Yes — max_rel_error=3.022e-8 against libcint vendor; anti-zero sentinel asserted | FLOWING |

### Behavioral Spot-Checks

Previously-confirmed spot-checks remain green (unchanged codepaths). Gap-closure additions:

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| complex_values unit tests (5 new) | `cargo test -p cintx-rs --features cpu complex_values` | Included in 31/31 cintx-rs tests green (20-09 SUMMARY) | PASS |
| spinor_evaluate_exposes_complex_values_some_prec02 | `cargo test -p cintx-rs --features cpu spinor_evaluate_exposes` | Included in 31/31 green (20-09 SUMMARY, commit c888021) | PASS |
| cintx-cubecl full test suite post-kernel-fixes | `cargo test -p cintx-cubecl --features cpu,with-f12,with-4c1e` | 180 passed, 0 failed (20-11 SUMMARY) | PASS |
| f32 parity test (12 tests, including new stg_ip1) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12 --test f32_parity` | 12 passed (20-11 SUMMARY) | PASS |
| f64 oracle (full, post gap-closure) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12,with-4c1e --test '*'` | All integration suites pass; 4 pre-existing compare::tests lib failures unchanged (20-11 SUMMARY) | PASS |
| Workspace check | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | exit 0 (20-09, 20-10, 20-11 SUMMARYs) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PREC-01 | Plans 01–05 | F: Float through full compute path | SATISFIED | All math modules and kernel launchers verified generic; unchanged from initial |
| PREC-02 | Plans 04, 05, 07, 09 | evaluate_generic::<F>() + Complex<F> spinor view | SATISFIED | complex_values() accessor implemented; spinor_evaluate_exposes_complex_values_some_prec02 asserts literal truth; ROADMAP SC-2 wording honored via IMPLEMENT path |
| PREC-03 | Plan 06 | Raw compat env/atm/bas + C ABI stay f64 | SATISFIED | Unchanged from initial; cintx-compat and cintx-capi confirmed f64 |
| PREC-04 | Plans 07, 08, 09, 10, 11 | f64 byte-identical; oracle gates unchanged | SATISFIED | 11/11 f64 integration tests green post-gap-closure; pre-existing compare::tests failures unchanged |
| PREC-05 | Plans 08, 09, 10, 11 | f32 oracle gate at ~1e-4 rtol; correct length contract | SATISFIED | CR-01 fixed in all 7 kernels; CR-02+WR-01 in f12; WR-03/04/05/06 math hardening; new ncomp=3 test passes at max_rel_error=3.022e-8; 12 vendor-gated f32 tests |
| PREC-06 | Plan 06 | f32 bypasses SHADER_F64 | SATISFIED | check_capability() early-returns Ok for F32; unchanged |
| PREC-07 | Plans 01–11 | Serena symbol-aware tools; no blind text replacement | SATISFIED (process gate) | .serena/ present; code structure consistent; FROZEN values preserved |

**Note:** PREC requirements are defined in ROADMAP.md (derived from 20-CONTEXT.md decisions) and are NOT in .planning/REQUIREMENTS.md (which covers v1.2/v1.3 requirements through Phase 15). No orphaned requirements.

### Anti-Patterns Found

The CR-01/CR-02/WR-01/WR-03/WR-04/WR-05/WR-06 defects documented in the initial verification have been resolved. Scanning the gap-closure files for residual issues:

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cintx-cubecl/src/kernels/f12.rs` | ~1395 | `v.abs() > 1e-18` — a residual hardcoded threshold inside `launch_stg_base` sub-function (not in the outer dispatch or the typed inner's WR-06 path) | Info | This is inside a sub-kernel path that feeds into the typed inner, not in the outer not0 scan path that was fixed by WR-06. The WR-06 fix (at line 1606) is in `launch_f12_typed` which is the result-count path that matters for PREC-05. The sub-kernel threshold is a kernel-internal convergence guard, not the output not0 scan. Not a blocker. |

No new placeholder, TODO, or FIXME markers found in the gap-closure modified files.

DI-01 (pre-existing f12 asymmetric-quartet value-swap) is documented in deferred-items.md. It is NOT a residual anti-pattern from the gap-closure work — it is a pre-existing bug in the f64 path that the f32 gate restriction correctly avoids. The appropriate treatment is a separate debug/fix plan.

### Human Verification Required

None. All must-haves are verified or resolved programmatically via code inspection and SUMMARY-documented test results (all 7 commits verified in the repository).

### Gaps Summary

No gaps. Both prior gaps are closed:

**Gap 1 (PREC-02) — CLOSED:** `complex_values() -> Option<Vec<Complex<F>>>` is present on both `IntegralTensor<F>` (line 577) and `TypedEvaluationOutput<F>` (line 594) in api.rs. The `num-complex = "0.4"` direct dependency is in cintx-rs/Cargo.toml. The `spinor_evaluate_exposes_complex_values_some_prec02` test (line 1303) is an end-to-end proof. ROADMAP SC-2 is satisfied on the IMPLEMENT path — no override was required or applied.

**Gap 2 (PREC-05) — CLOSED:** CR-01 fix is present in all 7 kernel files (`out_elems` pre-cast, `staging_f32[..out_elems]`, BufferTooSmall guard). CR-02+WR-01 fix is in `launch_f12_typed` (`staging_f64 = vec![0.0_f64; out_elems]`; readback and not0 bounded to out_elems). WR-03/04/05/06 math hardening is applied. `test_f32_int2e_stg_ip1_sph_parity` (f32_parity.rs line 1050) is vendor-gated, exercises the ncomp=3 / CR-01 regime for all three staging sizes, and passes at max_rel_error=3.022e-8 against libcint.

---

_Verified: 2026-05-21T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
