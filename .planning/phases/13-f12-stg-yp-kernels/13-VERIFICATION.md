---
phase: 13-f12-stg-yp-kernels
verified: 2026-04-05T12:00:00Z
status: passed
score: 4/4 success criteria verified
re_verification:
  previous_status: gaps_found
  previous_score: 3/4
  gaps_closed:
    - "SC3: All 10 with-f12 sph symbols now pass oracle parity against libcint 6.1.3 at atol=1e-12 (8 derivative variants upgraded from idempotency-only to full oracle comparison)"
    - "Manifest component_rank corrected for all 8 derivative F12 entries (ip1: '3', ipip1/ipvip1/ip1ip2: '9')"
    - "ncomp field in F12Variant is actively used in f12_kernel_core dispatch; no dead_code warning"
  gaps_remaining: []
  regressions: []
human_verification: []
---

# Phase 13: F12/STG/YP Kernels Verification Report

**Phase Goal:** STG and YP geminal two-electron kernels are implemented as separate dispatch paths with PTR_F12_ZETA env plumbing, covering all 10 with-f12 sph symbols at oracle parity against libcint 6.1.3 at atol=1e-12. Cart and spinor representations remain unsupported for F12 symbols (sph-only enforcement already in place from Phase 3).
**Verified:** 2026-04-05
**Status:** passed
**Re-verification:** Yes — after gap closure via plan 13-04

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | `kernels/f12.rs` implements STG and YP as separate kernel entry points; ibase/kbase routing handled; STG roots replicate `t = min(t, 19682.99)` clamp exactly | VERIFIED | `launch_stg_*` and `launch_yp_*` are distinct functions; `T_MAX: f64 = 19682.99_f64`; oracle parity tests pass at atol=1e-12 for all 10 symbols |
| SC2 | `ExecutionPlan` carries `operator_env_params` with `PTR_F12_ZETA` (env[9]); validator rejects F12 calls where env[9]==0.0 with typed `InvalidEnvParam` error | VERIFIED | `OperatorEnvParams { f12_zeta: Option<f64> }` on `ExecutionPlan`; `validate_f12_env_params` rejects None and Some(0.0); `f12_zeta_zero_rejected_all_10` covers all 10 symbols |
| SC3 | All 10 with-f12 sph symbols pass oracle parity against libcint 6.1.3 at family-appropriate tolerance; cart and spinor symbol counts confirmed zero | VERIFIED | All 15 tests in `f12_oracle_parity.rs` pass: 2 base oracle parity tests + 8 new derivative oracle parity tests (all at atol=1e-12 vs vendored libcint); `f12_sph_only_enforcement` confirms 0 cart, 0 spinor F12 entries |
| SC4 | Oracle fixtures validate that a call with zeta=0 is rejected by the validator or produces explicit Coulomb-equivalent result — not a silent wrong result | VERIFIED | `f12_zeta_zero_rejected_all_10` verifies all 10 symbols return `InvalidEnvParam { param: "PTR_F12_ZETA" }` for zeta=0 |

**Score:** 4/4 success criteria verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/math/stg.rs` | CINTstg_roots port with COS_14_14, Clenshaw/DCT, T_MAX clamp | VERIFIED | 466 lines; `T_MAX: f64 = 19682.99_f64`; `COS_14_14` 196-element array; Clenshaw algorithm implemented |
| `crates/cintx-cubecl/src/math/roots_xw_data.rs` | Static DATA_X and DATA_W arrays from roots_xw.dat | VERIFIED | Binary embedding via `include_bytes!` + `bytemuck::cast_slice` |
| `crates/cintx-core/src/error.rs` | `InvalidEnvParam` variant on `cintxRsError` | VERIFIED | Line 70: `InvalidEnvParam { param: &'static str, reason: String }` |
| `crates/cintx-runtime/src/planner.rs` | `OperatorEnvParams` struct and field on `ExecutionPlan` | VERIFIED | `pub struct OperatorEnvParams { pub f12_zeta: Option<f64> }` |
| `crates/cintx-runtime/src/validator.rs` | `validate_f12_env_params` rejecting zeta==0 for f12 family | VERIFIED | `validate_f12_env_params` rejects None and Some(0.0) |
| `crates/cintx-cubecl/src/kernels/f12.rs` | 10 F12 kernel entry points with `launch_f12` dispatcher; nabla/Hessian derivative functions; ncomp dispatch | VERIFIED | `nabla1i_2e`, `nabla1j_2e`, `nabla1k_2e`, `gout_ip1`, `gout_ipip1`, `gout_ipvip1`, `gout_ip1ip2` all present; `f12_kernel_core` dispatches on `ncomp` (1 or 3/9); `cart_to_sph_2e` called per component in ncomp>1 path (line 1372) |
| `crates/cintx-ops/src/generated/api_manifest.rs` | Correct component_rank for all 10 F12 entries | VERIFIED | Base STG/YP: `""` (scalar); ip1 variants: `"3"`; ipip1/ipvip1/ip1ip2 variants: `"9"` — all 8 derivative entries corrected from `""` |
| `crates/cintx-oracle/tests/f12_oracle_parity.rs` | Oracle parity tests for all 10 F12 sph symbols | VERIFIED | 571 lines; 10 `oracle_parity_int2e_*` tests (all `#[cfg(has_vendor_libcint)]`); `eval_f12_sph_ncomp` helper sizes buffer as `ncomp * n_sph_elements`; all 15 tests pass |
| `crates/cintx-oracle/src/vendor_ffi.rs` | Vendor FFI declarations for all 10 F12 symbols | VERIFIED | All 10 wrapper functions present |
| `crates/cintx-oracle/build.rs` | grad2.c and hess.c compiled for derivative gout symbols | VERIFIED | Lines 60-61 (rerun-if-changed) and lines 189-190 (cc::Build) include grad2.c and hess.c |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `kernels/f12.rs` | `math/stg.rs` | `stg_roots_host` call | WIRED | `use crate::math::stg::stg_roots_host`; called in STG path |
| `kernels/f12.rs` | `transform/c2s.rs` | `cart_to_sph_2e` per component in ncomp>1 path | WIRED | Line 1372: `cart_to_sph_2e(cart_slice, ...)` inside `for comp in 0..ncomp` loop |
| `validator.rs` | `error.rs` | `InvalidEnvParam` error variant | WIRED | `validate_f12_env_params` returns `Err(cintxRsError::InvalidEnvParam { param: "PTR_F12_ZETA", ... })` |
| `kernels/mod.rs` | `f12::launch_f12` | `"f12"` arm gated by `with-f12` feature | WIRED | `"f12" => Some(f12::launch_f12 as FamilyLaunchFn)` under `#[cfg(feature = "with-f12")]` |
| `compat/src/raw.rs` | `planner.rs` | `operator_env_params.f12_zeta` from env[9] | WIRED | `is_f12_family_symbol` + `env[PTR_F12_ZETA]` assignment |
| `api_manifest.rs` | `planner.rs` | `parse_component_multiplier` reads `component_rank` | WIRED | `component_rank: "3"` / `"9"` at manifest lines 1734, 1751, 1768, 1785, 1819, 1836, 1853, 1870 |
| `f12_oracle_parity.rs` | `vendor_ffi.rs` | 8 derivative `vendor_int2e_*` calls | WIRED | Lines 225, 247, 269, 291, 313, 335, 357, 379: all 8 derivative vendor wrappers called with libcint comparison |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `f12.rs: f12_kernel_core ncomp>1 path` | `gout_contracted` (ncomp*nf_base Cartesian) | `gout_ip1`/`gout_ipip1`/`gout_ipvip1`/`gout_ip1ip2` functions | Yes — oracle parity confirms matching libcint for SS, HH, SP shell quartets across all 8 derivative variants | FLOWING |
| `f12.rs: per-component sph transform` | `sph` (per-component sph slice written to staging) | `cart_to_sph_2e(cart_slice, ...)` for each of ncomp components | Yes — oracle parity at atol=1e-12 confirms correct sph output | FLOWING |
| `api_manifest.rs: component_rank` | Staging buffer size in planner | `parse_component_multiplier("3")` or `parse_component_multiplier("9")` | Yes — `f12_zeta_zero_rejected_all_10` uses `(9*n).max(1)` buffer; planner correctly sizes for derivative variants | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 15 F12 oracle parity tests pass | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu,with-f12 -p cintx-oracle --test f12_oracle_parity` | 15 passed, 0 failed, 0 ignored | PASS |
| Full workspace compiles clean with with-f12 | `cargo check --features cpu,with-f12` | No errors | PASS |
| cintx-cubecl compiles clean with with-f12 (no dead_code warning) | `cargo check --features cpu,with-f12 -p cintx-cubecl` | `Finished` with no warnings | PASS |
| Manifest 8 derivative entries have component_rank "3" or "9" | `grep "component_rank.*3\|component_rank.*9" api_manifest.rs` | 8 matches at lines 1734/1751/1768/1785/1819/1836/1853/1870 | PASS |
| `cart_to_sph_2e` called per component in f12.rs ncomp>1 path | `grep "cart_to_sph_2e" f12.rs` | Line 1372 inside `for comp in 0..ncomp` loop | PASS |
| grad2.c and hess.c present in oracle build.rs | `grep "grad2\|hess" build.rs` | Lines 59-61 (rerun-if-changed), 189-190 (cc::Build) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| F12-01 | 13-01, 13-02 | STG kernel implements modified Rys quadrature with tabulated polynomial roots matching libcint | SATISFIED | `stg_roots_host` ported with COS_14_14, Clenshaw, T_MAX clamp; oracle parity at atol=1e-12 for `int2e_stg_sph` and all STG derivative variants |
| F12-02 | 13-01, 13-02 | YP kernel implements correct routing distinct from STG path | SATISFIED | `launch_yp_*` entry points use `is_stg=false`; YP weight post-processing differs from STG; oracle parity at atol=1e-12 for all 5 YP symbols |
| F12-03 | 13-03, 13-04 | All 10 with-f12 sph symbols pass oracle parity against libcint at atol=1e-12 | SATISFIED | All 10 `oracle_parity_int2e_*` tests pass under `has_vendor_libcint`; gap closed by plan 13-04 implementing nabla/Hessian G tensor derivatives and per-component sph transform |
| F12-04 | 13-01, 13-02 | PTR_F12_ZETA (env[9]) correctly plumbed through ExecutionPlan to kernel launchers | SATISFIED | `OperatorEnvParams.f12_zeta` on `ExecutionPlan`; populated from `env[9]` in raw compat path and from `ExecutionOptions.f12_zeta` in safe API path |
| F12-05 | 13-01, 13-02 | Oracle fixtures validate zeta=0 is rejected or produces Coulomb-equivalent results explicitly | SATISFIED | `f12_zeta_zero_rejected_all_10` verifies all 10 symbols return `InvalidEnvParam { param: "PTR_F12_ZETA" }` for zeta=0; buffer sized at `(9*n).max(1)` prevents BufferTooSmall from masking the gate |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No anti-patterns found. No TODO/FIXME/placeholder comments in modified files. No dead_code warnings. No hardcoded empty returns in production paths. No idempotency-only tests masking missing oracle coverage.

### Human Verification Required

None. All must-haves are fully verifiable programmatically. The oracle parity tests against vendored libcint serve as the numerical correctness gate.

### Gaps Summary

No gaps. All three items from the previous verification were resolved by plan 13-04:

1. **SC3 oracle parity (8 derivative variants):** Closed by implementing `nabla1i_2e`, `nabla1j_2e`, `nabla1k_2e` G tensor derivative functions and `gout_ip1`/`gout_ipip1`/`gout_ipvip1`/`gout_ip1ip2` contraction functions. `f12_kernel_core` now dispatches on `ncomp`: the ncomp==1 path is unchanged; the ncomp>1 path accumulates `ncomp * nf_base` Cartesian values per primitive and applies `cart_to_sph_2e` once per component. All 8 derivative oracle parity tests pass at atol=1e-12 against vendored libcint 6.1.3 across SS, HH, and SP shell quartets.

2. **Manifest component_rank:** All 8 derivative F12 entries corrected: ip1 variants set to `"3"`, ipip1/ipvip1/ip1ip2 variants set to `"9"`. This ensures `parse_component_multiplier` in the planner correctly sizes staging buffers for multi-component output.

3. **ncomp dead_code warning:** The `ncomp` field of `F12Variant` is now actively read in `f12_kernel_core` (line 1206: `let ncomp = variant.ncomp`) and drives the dispatch branch, the `gout_contracted` buffer size, and the per-component sph transform loop. Cargo check produces zero warnings for `cintx-cubecl`.

Phase 13 goal is fully achieved.

---

_Verified: 2026-04-05_
_Verifier: Claude (gsd-verifier)_
