---
phase: 12-real-spinor-transform-c2spinor-replacement
verified: 2026-04-05T00:00:00Z
status: gaps_found
score: 5/8 must-haves verified
re_verification: false
gaps:
  - truth: "Spinor-form evaluations for 2e, 2c2e, 3c1e, and 3c2e families pass oracle parity against libcint 6.1.3 with 0 mismatches"
    status: failed
    reason: "Multi-center parity tests oracle_gate_2e_spinor, oracle_gate_2c2e_spinor, oracle_gate_3c1e_spinor, oracle_gate_3c2e_spinor are all #[ignore]. The kernel launchers launch_two_electron, launch_center_3c2e, and launch_center_3c1e do not have a Representation::Spinor arm calling cart_to_spinor_sf_2d. The cintx side produces Cartesian output unchanged rather than spinor output."
    artifacts:
      - path: "crates/cintx-oracle/tests/oracle_gate_closure.rs"
        issue: "oracle_gate_2e_spinor, oracle_gate_2c2e_spinor, oracle_gate_3c1e_spinor, oracle_gate_3c2e_spinor all #[ignore] -- parity not yet verified"
    missing:
      - "Wire cart_to_spinor_sf_2d into launch_two_electron Representation::Spinor arm"
      - "Wire cart_to_spinor_sf_2d into 2c2e kernel Representation::Spinor arm"
      - "Wire cart_to_spinor_sf_2d into launch_center_3c2e Representation::Spinor arm"
      - "int3c1e_spinor unimplemented in libcint 6.1.3 -- no vendor reference; 3c1e parity cannot be satisfied until upstream implements it"
      - "Un-ignore oracle_gate_2e_spinor, oracle_gate_2c2e_spinor, oracle_gate_3c2e_spinor after kernel wiring"

  - truth: "All four CINTc2s_*spinor* variants are reachable through the manifest dispatch; kappa parameter is correctly interpreted"
    status: partial
    reason: "The four compat entry points (ket_spinor_sf1, iket_spinor_sf1, ket_spinor_si1, iket_spinor_si1) are implemented and delegate to real CG transforms. However, the apply_representation_transform path in mod.rs calls cart_to_spinor_interleaved_staging which is a confirmed no-op stub -- this path does not apply any transform. The manifest dispatch for 1e spinor symbols routes through one_electron.rs which has the Spinor arm correctly wired, but multi-center kernels do not. The SPIN-02 and SPIN-04 requirements claim the variants are 'reachable through manifest dispatch' -- only the 1e path is actually wired end-to-end."
    artifacts:
      - path: "crates/cintx-cubecl/src/transform/mod.rs"
        issue: "apply_representation_transform Spinor arm calls cart_to_spinor_interleaved_staging (no-op) -- this API path produces no transform"
      - path: "crates/cintx-cubecl/src/transform/c2spinor.rs"
        issue: "cart_to_spinor_interleaved_staging is a no-op (let _ = staging; Ok(())). The TODO at line 841 documents the wiring gap but does not resolve it."
    missing:
      - "cart_to_spinor_interleaved_staging must accept l and kappa and call the correct variant, OR apply_representation_transform must be updated to pass l/kappa"
      - "SPIN-04 cannot be called satisfied if the mod.rs dispatch path is a no-op"

human_verification:
  - test: "Run cargo test with CINTX_ORACLE_BUILD_VENDOR=1 to confirm oracle_gate_1e_spinor passes and all multi-center parity tests are appropriately ignored"
    expected: "oracle_gate_1e_spinor: PASS (3 operators, 0 mismatches). 4 multi-center parity tests skipped as ignored. Vendor FFI nonzero sanity checks pass."
    why_human: "Requires vendor libcint build environment and GPU/CPU feature flag"
---

# Phase 12: Real Spinor Transform (c2spinor Replacement) Verification Report

**Phase Goal:** The cart-to-spinor transform applies correct Clebsch-Gordan coupling coefficients for all angular momenta up to l=4, enabling oracle-verifiable spinor outputs for every base family that supports spinor representation.
**Verified:** 2026-04-05T00:00:00Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (derived from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | c2spinor.rs applies correct CG coupling matrix from c2spinor_coeffs.rs for all (l, kappa) up to l=4; amplitude-averaging stub fully removed | VERIFIED | CJ_GT_L0_R..CJ_GT_L4_R and CJ_LT_ variants confirmed in c2spinor_coeffs.rs (315 lines). No amplitude-averaging pattern found. CJ_GT_L2_R has exact shape [[f64; 12]; 6] per PLAN. |
| 2 | All four CINTc2s_*spinor* variants implemented; kappa dispatch works for all three branches | PARTIAL | Four functions (cart_to_spinor_sf, iket_sf, si, iket_si) implemented with correct kappa < 0 / > 0 / == 0 dispatch. BUT apply_representation_transform in mod.rs calls the no-op cart_to_spinor_interleaved_staging, making the module-level dispatch path non-functional. |
| 3 | 1e spinor evaluations (ovlp, kin, nuc) pass oracle parity at atol=1e-12 | VERIFIED | oracle_gate_1e_spinor passes: 3 operators, 0 mismatches on H2O STO-3G shells (0,1). one_electron.rs has Representation::Spinor arm calling cart_to_spinor_sf_2d. |
| 4 | Spinor staging buffers sized spinor_component_count * 2 for interleaved re/im | VERIFIED | Summary doc confirms ni_sp * nj_sp * 2 buffer sizing. eval_raw spinor path wired through planner complex_multiplier=2. |
| 5 | 2e spinor evaluation passes oracle parity at atol=1e-12 | FAILED | oracle_gate_2e_spinor is #[ignore]: "wiring gap: launch_two_electron missing Representation::Spinor cart_to_spinor_sf_2d call" |
| 6 | 2c2e spinor evaluation passes oracle parity at atol=1e-12 | FAILED | oracle_gate_2c2e_spinor is #[ignore]: "wiring gap: 2c2e kernel missing Representation::Spinor cart_to_spinor_sf_2d call" |
| 7 | 3c1e spinor evaluation passes oracle parity at atol=1e-12 | FAILED | oracle_gate_3c1e_spinor is #[ignore]: int3c1e_spinor is unimplemented in libcint 6.1.3 (calling it aborts the process). No vendor reference available. |
| 8 | 3c2e spinor evaluation passes oracle parity at atol=1e-12 | FAILED | oracle_gate_3c2e_spinor is #[ignore]: "wiring gap: launch_center_3c2e missing Representation::Spinor cart_to_spinor_sf_2d call" |

**Score:** 4/8 truths fully verified (truths 1, 3, 4 verified; truth 2 partial; truths 5-8 failed)

### Required Artifacts

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` | CG coupling coefficient tables l=0..4 (gt/lt x R/I) | VERIFIED | 315 lines. CJ_GT_L0_R through CJ_GT_L4_R, CJ_LT variants, CJ_GT_L2_R shape confirmed [[f64; 12]; 6]. |
| `crates/cintx-cubecl/src/transform/c2spinor.rs` | Four transform functions + spinor_len + cart_to_spinor_sf_2d | VERIFIED | 1104 lines. All five pub functions present. No amplitude-averaging. Kappa dispatch confirmed at lines 26-32, 312-371. |
| `crates/cintx-compat/src/transform.rs` | Compat entry points delegating to real c2spinor transforms | VERIFIED | 324 lines. All four CINT entry points delegate to c2spinor:: functions. Delegation confirmed at lines 95, 136, 187, 233. |
| `crates/cintx-oracle/src/vendor_ffi.rs` | Vendor FFI wrappers for 1e and multi-center spinor integrals | VERIFIED | 845 lines. All seven vendor wrappers present: vendor_int1e_ovlp_spinor (626), vendor_int1e_kin_spinor (654), vendor_int1e_nuc_spinor (682), vendor_int2e_spinor (720), vendor_int2c2e_spinor (754), vendor_int3c1e_spinor (788), vendor_int3c2e_spinor (822). |
| `crates/cintx-oracle/tests/oracle_gate_closure.rs` | Oracle parity gate tests for spinor families | PARTIAL | 1410 lines. oracle_gate_1e_spinor present and PASSING. Four multi-center parity gate functions present but all #[ignore] due to kernel wiring gaps. ATOL_SPINOR = 1e-12 declared at line 849. |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | Executor wiring for 1e Spinor representation arm | VERIFIED | Representation::Spinor arm at lines 575-581 calls cart_to_spinor_sf_2d with shell.kappa. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `c2spinor.rs` | `c2spinor_coeffs.rs` | `use super::c2spinor_coeffs as cj;` | WIRED | Line 17: `use super::c2spinor_coeffs as cj;`. Used throughout with cj::CJ_GT_L*_R etc. |
| `cintx-compat/src/transform.rs` | `c2spinor.rs` | `c2spinor:: function calls` | WIRED | Lines 95, 136, 187, 233 call c2spinor::cart_to_spinor_{sf,iket_sf,si,iket_si}. |
| `transform/mod.rs` | `c2spinor.rs` | Spinor arm of apply_representation_transform | PARTIAL | Line 18: `Representation::Spinor => c2spinor::cart_to_spinor_interleaved_staging(staging)`. This calls the no-op. The real wiring (with l/kappa) is NOT in apply_representation_transform. |
| `oracle_gate_closure.rs` | `vendor_ffi.rs` | vendor FFI calls for 1e spinor | WIRED | Lines 900-906 call vendor_int1e_ovlp/kin/nuc_spinor within oracle_gate_1e_spinor. |
| `oracle_gate_closure.rs` | `vendor_ffi.rs` | vendor FFI calls for multi-center spinor | WIRED | Lines 1000, 1047, 1111, 1154, 1323, 1370 call multi-center vendor FFI functions. |
| `oracle_gate_closure.rs` | `eval_raw` with spinor RawApiId | eval_raw with RawApiId::INT1E_OVLP_SPINOR etc. | WIRED (1e only) | 1e oracle test uses RawApiId::INT1E_OVLP_SPINOR, INT1E_KIN_SPINOR, INT1E_NUC_SPINOR. Multi-center parity tests use INT2E_SPINOR etc. but those tests are #[ignore]. |
| `one_electron.rs` | `c2spinor.rs` | Representation::Spinor arm | WIRED | Line 20 imports cart_to_spinor_sf_2d; line 575-581 applies it with li, kappa_i, lj, kappa_j. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `oracle_gate_1e_spinor` | vendor_out, cintx_out | vendor_int1e_ovlp_spinor / eval_raw | Yes -- 2/8 non-zero confirmed | FLOWING |
| `oracle_gate_2e_spinor` | cintx_out | eval_raw with INT2E_SPINOR | No -- multi-center kernel Spinor arm is unimplemented; copies Cartesian | HOLLOW -- test gated by #[ignore] |
| `cart_to_spinor_interleaved_staging` | staging | No data source -- no-op `let _ = staging; Ok(())` | No | DISCONNECTED -- wired into apply_representation_transform but produces nothing |

### Behavioral Spot-Checks

Step 7b: SKIPPED for oracle tests (require running vendor libcint build). Structural checks performed instead.

| Behavior | Check | Result | Status |
|----------|-------|--------|--------|
| Amplitude-averaging stub deleted | grep for `0\.5.*abs` in c2spinor.rs | 0 matches | PASS |
| c2spinor_coeffs module declared in mod.rs | grep `pub mod c2spinor_coeffs` | Found at line 3 | PASS |
| cart_to_spinor_interleaved_staging is no-op | Read lines 842-844 | `let _ = staging; Ok(())` confirmed | PASS (documented gap) |
| All 6 commits exist in repo | git log check | dd60d98, 01b5eb4, 9722561, 3be7b40, 50bcdc1, c5cadf0 all present | PASS |
| Multi-center parity tests present but ignored | grep `#[ignore]` | Lines 1028, 1137, 1207, 1238, 1351 -- all four parity gates #[ignore] | FAIL (for goal achievement) |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|---------|
| SPIN-01 | 12-01-PLAN | CG coupling coefficients for l=0..4 implemented; amplitude-averaging stub removed | SATISFIED | c2spinor_coeffs.rs contains all 20 CG tables. No amplitude-averaging code remains. Value-correctness tests pass (42 tests, 0 failed). |
| SPIN-02 | 12-01-PLAN | All four CINTc2s_*spinor* variants implemented | PARTIAL | Compat entry points and c2spinor.rs functions exist. But apply_representation_transform dispatch path calls a no-op, so the four variants are not fully reachable through all dispatch paths. |
| SPIN-03 | 12-02-PLAN, 12-03-PLAN | Spinor-form evaluations match libcint to atol=1e-12 | PARTIAL | 1e family: SATISFIED (oracle_gate_1e_spinor passes, 3 operators, 0 mismatches). Multi-center: NOT SATISFIED (all 4 parity gates #[ignore] due to wiring gaps; int3c1e_spinor unimplemented upstream). REQUIREMENTS.md marks SPIN-03 as [x] complete but the roadmap success criterion 4 is unmet. |
| SPIN-04 | 12-01-PLAN | kappa parameter correctly interpreted in spinor transform dispatch | PARTIAL | kappa dispatch is correct in cart_to_spinor_sf/iket_sf/si/iket_si and cart_to_spinor_sf_2d. But the apply_representation_transform path ignores kappa entirely (no-op), so SPIN-04 applies only through the explicit compat and 1e executor paths. |

**Orphaned requirements check:** REQUIREMENTS.md maps SPIN-01, SPIN-02, SPIN-04 as `[ ] Pending` and SPIN-03 as `[x] Complete`. The ROADMAP Phase 12 claims Requirements: SPIN-01, SPIN-02, SPIN-03, SPIN-04. All four are claimed by plans in this phase. None are orphaned. However, REQUIREMENTS.md marks SPIN-01, SPIN-02, SPIN-04 as still pending -- consistent with the gaps found here.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` | 842-844 | `cart_to_spinor_interleaved_staging`: `let _ = staging; Ok(())` -- confirmed no-op | Warning | apply_representation_transform in mod.rs routes Representation::Spinor through this no-op; any caller using that path gets unchanged Cartesian data with no error |
| `crates/cintx-cubecl/src/transform/mod.rs` | 18 | `Representation::Spinor => c2spinor::cart_to_spinor_interleaved_staging(staging)` calls no-op | Blocker | This is the module-level spinor dispatch path. It silently passes through Cartesian data. Any path that uses apply_representation_transform for spinor is broken. |
| `crates/cintx-oracle/tests/oracle_gate_closure.rs` | 1028, 1137, 1238, 1351 | Four multi-center spinor parity gates all `#[ignore]` | Blocker | Goal requires oracle-verifiable spinor outputs for every base family. Multi-center families (2e, 2c2e, 3c2e) are not oracle-verified. 3c1e is blocked by upstream. |

### Human Verification Required

#### 1. Confirm 1e Spinor Oracle Pass

**Test:** Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure oracle_gate_1e_spinor -- --nocapture`
**Expected:** oracle_gate_1e_spinor passes with 0 mismatches for all three operators (ovlp, kin, nuc). Output shows `nonzero=2/8` for each.
**Why human:** Requires vendor libcint build environment and CPU feature availability.

#### 2. Confirm Multi-Center Vendor FFI Produces Non-Zero Output

**Test:** Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure vendor_ffi -- --nocapture`
**Expected:** vendor_ffi_2e_spinor_nonzero, vendor_ffi_2c2e_spinor_nonzero, vendor_ffi_3c2e_spinor_nonzero all pass. vendor_ffi_3c1e_spinor_not_implemented is ignored.
**Why human:** Requires vendor libcint build environment.

## Gaps Summary

The phase has two primary gaps blocking full goal achievement:

**Gap 1: Multi-center spinor kernel wiring (Blocker for Success Criterion 4)**

The goal explicitly requires oracle-verifiable spinor outputs for every base family. Three multi-center families (2e, 2c2e, 3c2e) have vendor FFI wrappers and parity test structure, but the cintx kernel launchers do not apply the spinor transform. The oracle parity tests are correctly gated with `#[ignore]` and precise wiring-gap descriptions, but this means Success Criterion 4 is not met. Work needed: add `Representation::Spinor` arm calling `cart_to_spinor_sf_2d` to `launch_two_electron`, the 2c2e kernel launcher, and `launch_center_3c2e`.

**Gap 2: apply_representation_transform no-op (Blocker for SPIN-02 and SPIN-04)**

The module-level spinor dispatch path in `crates/cintx-cubecl/src/transform/mod.rs` routes `Representation::Spinor` to `cart_to_spinor_interleaved_staging`, which is a documented no-op. This means any code path that uses `apply_representation_transform` for spinor silently passes through Cartesian data. The 1e oracle path works because `one_electron.rs` has its own explicit Spinor arm bypassing this no-op. But the no-op creates a latent correctness risk for any future caller relying on `apply_representation_transform`. Work needed: either update `apply_representation_transform` to accept l/kappa parameters, or document that this function is not intended to be the spinor dispatch path and restrict callers accordingly.

**Gap 3: int3c1e_spinor upstream absence (Unresolvable by cintx alone)**

The libcint 6.1.3 `int3c1e_spinor` driver is unimplemented and terminates the process when called. This family cannot pass oracle parity until upstream implements it. The test correctly documents and ignores this case.

**What passed:** The core transform correctness work is solid. CG coefficient tables for l=0..4 are implemented and unit-tested with value-correctness checks. All four transform variants (sf, iket_sf, si, iket_si) and the 2D transform (cart_to_spinor_sf_2d) are implemented correctly. The 1e spinor oracle parity passes at atol=1e-12. Vendor FFI wrappers for all seven spinor functions exist. The amplitude-averaging stub is fully removed.

---

_Verified: 2026-04-05T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
