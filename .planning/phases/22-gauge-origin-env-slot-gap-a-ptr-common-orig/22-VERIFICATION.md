---
phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
verified: 2026-05-29T00:00:00Z
status: gaps_found
score: 6/7 must-haves verified
overrides_applied: 0
gaps:
  - truth: "A non-finite gauge origin (NaN/inf) when Some(...) is rejected with InvalidEnvParam{param:\"PTR_COMMON_ORIG\"}"
    status: failed
    reason: >
      The truth holds on the raw/compat path (raw.rs:634-637 calls
      validate_common_orig_env_params unconditionally after the env read) but FAILS on
      the safe-API path. api.rs:197-200 propagates common_orig from ExecutionOptions into
      the plan but never calls validate_common_orig_env_params. A caller using
      with_common_origin([f64::NAN, 0.0, 0.0]) has the NaN threaded silently into the
      plan with no InvalidEnvParam error returned. The builder doc comment (builder.rs:111)
      explicitly promises "finiteness-validated (NaN/inf rejected) by
      validate_common_orig_env_params" — that promise is currently false on the safe path.
      The truth's wording does not scope to a single path, so partial satisfaction = FAILED.
    artifacts:
      - path: "crates/cintx-rs/src/api.rs"
        issue: >
          Lines 197-200 propagate common_orig from options to plan but no
          validate_common_orig_env_params call follows. Compare with raw.rs:634-637
          where the call IS present. The validator and the module-path import pattern
          are available; the call site is simply absent.
    missing:
      - >
        After the common_orig propagation block (api.rs:200), add a validation call
        following the FacadeError mapping convention used elsewhere in the same
        function (see REVIEW WR-01 for the exact code fragment):
        cintx_runtime::validator::validate_common_orig_env_params(
            plan.descriptor.operator_name(),
            &plan.operator_env_params,
        ).map_err(FacadeError::from)?;
---

# Phase 22: PTR_COMMON_ORIG Gauge-Origin Env Slot Verification Report

**Phase Goal:** Plumb the `PTR_COMMON_ORIG` gauge-origin env slot (env[1..3]) end-to-end on the
`PTR_RINV_ORIG` precedent and add the non-zero gauge-origin oracle fixture that gates all moment
+ GIAO parity.
**Verified:** 2026-05-29T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A raw caller that sets env[1..3] gets those values on plan.operator_env_params.common_orig after eval_raw | VERIFIED | raw.rs:628-632: unconditional bounds-guarded read populates `plan.operator_env_params.common_orig = Some([x,y,z])` |
| 2 | A safe-API caller using .with_common_origin([x,y,z]) gets those values on the plan | VERIFIED | builder.rs:112-114 sets `self.options.common_orig = Some(origin)`; api.rs:197-200 propagates into the plan |
| 3 | common_orig defaults to None and None is accepted (defaults to [0,0,0]), not an error | VERIFIED | `OperatorEnvParams` derives Default (planner.rs:42); `validate_common_orig_env_params` returns Ok(()) when `params.common_orig` is None (validator.rs:214 — `if let Some` does not match None) |
| 4 | A non-finite gauge origin (NaN/inf) when Some(...) is rejected with InvalidEnvParam{param:"PTR_COMMON_ORIG"} | FAILED | Raw path: VERIFIED (raw.rs:634-637 calls validator). Safe-API path: FAILED — api.rs:197-200 propagates the value but no validator call exists; NaN passes silently into the plan. Truth is not scoped to one path, so partial = failed. |
| 5 | A committed gauge-origin fixture exists with non-zero env[1..3] (proving a zero origin can be distinguished from the default) | VERIFIED | fixtures.rs:151-168: `COMMON_ORIG_FIXTURE_ORIGIN = [0.5,-0.3,0.8]`, `build_h2o_sto3g_common_orig()` writes all three env slots |
| 6 | Loading that fixture and running eval_raw round-trips the non-zero env[1..3] into plan.operator_env_params.common_orig | VERIFIED | common_orig_roundtrip.rs:60-80: asserts `params.common_orig == Some(COMMON_ORIG_FIXTURE_ORIGIN)` and `!= Some([0,0,0])`; eval_raw returns Ok |
| 7 | The fixture builder is callable by Phases 24/26 as data infrastructure | VERIFIED | All fixture symbols are `pub`; `PTR_COMMON_ORIG` imported in fixtures.rs:5; `COMMON_ORIG_FIXTURE_ORIGIN`, `build_h2o_sto3g_common_orig`, `build_h2o_sto3g_common_orig_at` all public |

**Score:** 6/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-compat/src/raw.rs` | PTR_COMMON_ORIG const (=1) + eval_raw env[1..3] read | VERIFIED | Line 50: `pub const PTR_COMMON_ORIG: usize = 1;` with full doc. Lines 628-633: operator-agnostic bounds-guarded read. Lines 634-637: validator call via module path. |
| `crates/cintx-runtime/src/planner.rs` | OperatorEnvParams.common_orig field | VERIFIED | Lines 54-57: `pub common_orig: Option<[f64; 3]>` inside `OperatorEnvParams` struct (derives Default at line 42). |
| `crates/cintx-runtime/src/validator.rs` | validate_common_orig_env_params finiteness validator + D-01 unit tests | VERIFIED | Lines 201-223: validator body with `if let Some(origin)` + `!v.is_finite()` check. Lines 452-488: 4 unit tests (default_is_none, accepts_none, accepts_some_finite, rejects_non_finite). |
| `crates/cintx-runtime/src/options.rs` | ExecutionOptions.common_orig field | VERIFIED | Lines 123-127: `pub common_orig: Option<[f64; 3]>` inside ExecutionOptions. |
| `crates/cintx-rs/src/builder.rs` | with_common_origin setter | VERIFIED | Lines 107-115: `pub fn with_common_origin(mut self, origin: [f64; 3]) -> Self` sets `self.options.common_orig = Some(origin)`. |
| `crates/cintx-rs/src/api.rs` | ExecutionOptions.common_orig -> plan propagation block | VERIFIED (propagation only) | Lines 197-200: propagation present. Validator call absent — see FAILED truth #4. |
| `crates/cintx-oracle/src/fixtures.rs` | build_h2o_sto3g_common_orig fixture builder (non-zero env[1..3]) | VERIFIED | Lines 148-169: `COMMON_ORIG_FIXTURE_ORIGIN`, `build_h2o_sto3g_common_orig`, `build_h2o_sto3g_common_orig_at`. PTR_COMMON_ORIG imported at line 5. |
| `crates/cintx-oracle/tests/common_orig_roundtrip.rs` | raw<->plan round-trip test for the gauge-origin slot | VERIFIED | Full file: 2 plain `#[test]` functions under `#![cfg(feature = "cpu")]`; no `#[cfg(has_vendor_libcint)]` gate; imports via module path as required. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| crates/cintx-compat/src/raw.rs eval_raw | plan.operator_env_params.common_orig | env[PTR_COMMON_ORIG..PTR_COMMON_ORIG+3] read under bounds guard | WIRED | raw.rs:628-632: `if env.len() >= PTR_COMMON_ORIG + 3 { ... plan.operator_env_params.common_orig = Some([x,y,z]); }` |
| crates/cintx-compat/src/raw.rs eval_raw | validate_common_orig_env_params | module path `cintx_runtime::validator::...` | WIRED | raw.rs:634-637: unconditional call after read block, matching rinv/grids/f12 convention |
| crates/cintx-rs/src/api.rs | plan.operator_env_params.common_orig | options().common_orig propagation | WIRED | api.rs:197-200: `if let Some(origin) = self.request.options().common_orig { plan.operator_env_params.common_orig = Some(origin); }` |
| crates/cintx-rs/src/api.rs | validate_common_orig_env_params | module path call after propagation | NOT WIRED | No call to `validate_common_orig_env_params` exists anywhere in `api.rs`. This is the BLOCKER gap (WR-01). |
| crates/cintx-rs/src/builder.rs with_common_origin | self.options.common_orig | setter assignment | WIRED | builder.rs:113: `self.options.common_orig = Some(origin);` |
| crates/cintx-oracle/src/fixtures.rs build_h2o_sto3g_common_orig | env[PTR_COMMON_ORIG..PTR_COMMON_ORIG+3] | non-zero origin assignment | WIRED | fixtures.rs:165-167: writes origin[0], origin[1], origin[2] to env slots |
| crates/cintx-oracle/tests/common_orig_roundtrip.rs | plan.operator_env_params.common_orig | eval_raw populates slot from fixture env | WIRED | Round-trip test mirrors the internal eval_raw read and asserts equality with COMMON_ORIG_FIXTURE_ORIGIN |

### Data-Flow Trace (Level 4)

Not applicable — this phase delivers infrastructure (a slot, a validator, a fixture) rather than a component that renders dynamic data. The round-trip test verifies data flows end-to-end from fixture env through `eval_raw` into the plan field.

### Behavioral Spot-Checks

Environment note: `cargo build -p cintx-compat -p cintx-runtime -p cintx-rs -p cintx-oracle` builds clean per the verification environment notes. `cargo test -p cintx-runtime common_orig` passes 4/4. `cargo test -p cintx-oracle --features cpu --test common_orig_roundtrip` passes 2/2.

| Behavior | Evidence | Status |
|----------|----------|--------|
| 4 D-01 validator unit tests pass (default None, accepts None, accepts finite, rejects non-finite) | Provided: cargo test -p cintx-runtime common_orig → 4/4 pass | PASS |
| Round-trip integration tests pass | Provided: cargo test -p cintx-oracle --features cpu --test common_orig_roundtrip → 2/2 pass | PASS |
| Safe-API path validates NaN/inf on with_common_origin | No test exercises this; code inspection confirms no validator call in api.rs evaluate_generic | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FND-01 | 22-01, 22-02 | PTR_COMMON_ORIG gauge-origin env slot plumbed end-to-end; non-zero oracle fixture exists; env round-trip + validator unit tests pass | PARTIAL | All slot plumbing, fixture, and round-trip tests are present. The "validator gate" portion of FND-01 is only enforced on the raw path; the safe-API path lacks the validate call, leaving the "gauge-origin must be finite before kernel entry" contract unmet for safe-API callers. |

FND-01 is the only requirement ID declared in both plan frontmatter sections. No orphaned requirements exist — FND-01 is mapped to Phase 22 in REQUIREMENTS.md traceability table (line 194).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cintx-rs/src/api.rs` | 197-200 | common_orig propagated without validation; builder doc at builder.rs:111 promises finiteness validation that does not execute on this path | BLOCKER | NaN/inf gauge origin silently reaches the plan on the safe-API path; when Phase 24/26 kernels read common_orig the value will be non-finite with no typed error having been returned to the caller |
| `crates/cintx-oracle/tests/common_orig_roundtrip.rs` | 6, 23, 68 | Stale raw.rs line-number citations ("raw.rs:604-615") that refer to wrong lines in the current file (the common_orig block is at 628-633) | INFO | Misleads future readers; no functional impact |
| `crates/cintx-runtime/src/validator.rs` | 187-196 | `match params.rinv_orig { None => {...}, _ => {} }` single-arm match (clippy `single_match`); adjacent to Phase 22 work but pre-existing; new `validate_common_orig_env_params` correctly uses `if let` | INFO | Style inconsistency; potential `-D warnings` clippy failure |

### Human Verification Required

None. All gaps are observable programmatically via code inspection.

### Gaps Summary

One blocker gap prevents full goal achievement:

**WR-01 (BLOCKER): Safe-API path does not validate common_orig finiteness.**

The phase delivers all six structural pieces for PTR_COMMON_ORIG (const, plan field, validator, ExecutionOptions field, builder setter, api propagation). However, the api.rs `evaluate_generic` path propagates `common_orig` from `ExecutionOptions` into the plan without calling `validate_common_orig_env_params`. The raw path (`raw.rs:634-637`) does validate. This asymmetry means Truth #4 — "a non-finite gauge origin when Some(...) is rejected" — holds only for raw callers. A safe-API caller invoking `.with_common_origin([f64::NAN, 0.0, 0.0])` gets no error today, and when Phases 24/26 kernels begin consuming `common_orig`, that NaN will propagate into integral evaluation unchecked.

The fix is a single validator call after the propagation block at api.rs:200, following the `FacadeError::from` mapping convention already used in the same function. See REVIEW WR-01 for the exact code fragment.

Root cause: same as every other rinv/f12 precedent — the safe-API path was wired to propagate the option but the corresponding validate call was not added alongside the propagation.

---

_Verified: 2026-05-29T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
