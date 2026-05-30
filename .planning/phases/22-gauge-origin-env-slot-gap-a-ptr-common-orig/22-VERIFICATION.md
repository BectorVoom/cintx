---
phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
verified: 2026-05-29T12:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 6/7
  gaps_closed:
    - "A non-finite gauge origin (NaN/inf) when Some(...) is rejected with InvalidEnvParam{param:\"PTR_COMMON_ORIG\"}"
  gaps_remaining: []
  regressions: []
---

# Phase 22: PTR_COMMON_ORIG Gauge-Origin Env Slot Verification Report

**Phase Goal:** Plumb the `PTR_COMMON_ORIG` gauge-origin env slot (env[1..3]) end-to-end on the
`PTR_RINV_ORIG` precedent and add the non-zero gauge-origin oracle fixture that gates all moment
+ GIAO parity.
**Verified:** 2026-05-29T12:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (commit 4ecefa0)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A raw caller that sets env[1..3] gets those values on plan.operator_env_params.common_orig after eval_raw | VERIFIED | raw.rs:625-632: unconditional bounds-guarded read populates `plan.operator_env_params.common_orig = Some([x,y,z])` |
| 2 | A safe-API caller using .with_common_origin([x,y,z]) gets those values on the plan | VERIFIED | builder.rs:112-114 sets `self.options.common_orig = Some(origin)`; api.rs:197-200 propagates into the plan |
| 3 | common_orig defaults to None and None is accepted (defaults to [0,0,0]), not an error | VERIFIED | `OperatorEnvParams` derives Default (planner.rs:42); `validate_common_orig_env_params` returns Ok(()) when `params.common_orig` is None (validator.rs:214 — `if let Some` does not match None) |
| 4 | A non-finite gauge origin (NaN/inf) when Some(...) is rejected with InvalidEnvParam{param:"PTR_COMMON_ORIG"} | VERIFIED | Raw path: raw.rs:634-637 calls `cintx_runtime::validator::validate_common_orig_env_params` unconditionally after the read block. Safe-API path: api.rs:201-210 calls the same validator via `.map_err(FacadeError::from)?` immediately after the common_orig propagation block (commit 4ecefa0). Both paths now reject NaN/inf with a typed error. Two regression tests in api.rs tests module confirm this: `evaluate_rejects_non_finite_common_origin_on_safe_api_path` (asserts FacadeErrorKind::Validation + "PTR_COMMON_ORIG" in message) and `evaluate_accepts_finite_common_origin_on_safe_api_path` (confirms no false rejection). cargo test -p cintx-rs common_origin → 2/2 pass; full workspace → 469 passed, 0 failed. |
| 5 | A committed gauge-origin fixture exists with non-zero env[1..3] (proving a zero origin can be distinguished from the default) | VERIFIED | fixtures.rs:151-168: `COMMON_ORIG_FIXTURE_ORIGIN = [0.5,-0.3,0.8]`, `build_h2o_sto3g_common_orig()` writes all three env slots |
| 6 | Loading that fixture and running eval_raw round-trips the non-zero env[1..3] into plan.operator_env_params.common_orig | VERIFIED | common_orig_roundtrip.rs:60-80: asserts `params.common_orig == Some(COMMON_ORIG_FIXTURE_ORIGIN)` and `!= Some([0,0,0])`; eval_raw returns Ok |
| 7 | The fixture builder is callable by Phases 24/26 as data infrastructure | VERIFIED | All fixture symbols are `pub`; `PTR_COMMON_ORIG` imported in fixtures.rs:5; `COMMON_ORIG_FIXTURE_ORIGIN`, `build_h2o_sto3g_common_orig`, `build_h2o_sto3g_common_orig_at` all public |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-compat/src/raw.rs` | PTR_COMMON_ORIG const (=1) + eval_raw env[1..3] read | VERIFIED | Line 50: `pub const PTR_COMMON_ORIG: usize = 1;` with full doc. Lines 625-632: operator-agnostic bounds-guarded read. Lines 634-637: validator call via module path. |
| `crates/cintx-runtime/src/planner.rs` | OperatorEnvParams.common_orig field | VERIFIED | Lines 54-57: `pub common_orig: Option<[f64; 3]>` inside `OperatorEnvParams` struct (derives Default at line 42). |
| `crates/cintx-runtime/src/validator.rs` | validate_common_orig_env_params finiteness validator + D-01 unit tests | VERIFIED | Lines 201-223: validator body with `if let Some(origin)` + `!v.is_finite()` check. Lines 452-488: 4 unit tests (common_orig_default_is_none, validate_common_orig_accepts_none, validate_common_orig_accepts_some_finite, validate_common_orig_rejects_non_finite). |
| `crates/cintx-runtime/src/options.rs` | ExecutionOptions.common_orig field | VERIFIED | Lines 123-127: `pub common_orig: Option<[f64; 3]>` inside ExecutionOptions. |
| `crates/cintx-rs/src/builder.rs` | with_common_origin setter | VERIFIED | Lines 107-115: `pub fn with_common_origin(mut self, origin: [f64; 3]) -> Self` sets `self.options.common_orig = Some(origin)`. |
| `crates/cintx-rs/src/api.rs` | ExecutionOptions.common_orig -> plan propagation block + safe-path validator call | VERIFIED | Lines 197-200: propagation present. Lines 201-210: `cintx_runtime::validator::validate_common_orig_env_params(plan.descriptor.operator_name(), &plan.operator_env_params).map_err(FacadeError::from)?` — gap WR-01 closed by commit 4ecefa0. |
| `crates/cintx-oracle/src/fixtures.rs` | build_h2o_sto3g_common_orig fixture builder (non-zero env[1..3]) | VERIFIED | Lines 148-169: `COMMON_ORIG_FIXTURE_ORIGIN`, `build_h2o_sto3g_common_orig`, `build_h2o_sto3g_common_orig_at`. PTR_COMMON_ORIG imported at line 5. |
| `crates/cintx-oracle/tests/common_orig_roundtrip.rs` | raw<->plan round-trip test for the gauge-origin slot | VERIFIED | Full file: 2 plain `#[test]` functions under `#![cfg(feature = "cpu")]`; no `#[cfg(has_vendor_libcint)]` gate; imports via module path as required. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| crates/cintx-compat/src/raw.rs eval_raw | plan.operator_env_params.common_orig | env[PTR_COMMON_ORIG..PTR_COMMON_ORIG+3] read under bounds guard | WIRED | raw.rs:625-632: `if env.len() >= PTR_COMMON_ORIG + 3 { ... plan.operator_env_params.common_orig = Some([x,y,z]); }` |
| crates/cintx-compat/src/raw.rs eval_raw | validate_common_orig_env_params | module path `cintx_runtime::validator::...` | WIRED | raw.rs:634-637: unconditional call after read block, matching rinv/grids/f12 convention |
| crates/cintx-rs/src/api.rs | plan.operator_env_params.common_orig | options().common_orig propagation | WIRED | api.rs:197-200: `if let Some(origin) = self.request.options().common_orig { plan.operator_env_params.common_orig = Some(origin); }` |
| crates/cintx-rs/src/api.rs | validate_common_orig_env_params | module path call after propagation | WIRED | api.rs:201-210: `cintx_runtime::validator::validate_common_orig_env_params(plan.descriptor.operator_name(), &plan.operator_env_params).map_err(FacadeError::from)?` — gap WR-01 closed. |
| crates/cintx-rs/src/builder.rs with_common_origin | self.options.common_orig | setter assignment | WIRED | builder.rs:113: `self.options.common_orig = Some(origin);` |
| crates/cintx-oracle/src/fixtures.rs build_h2o_sto3g_common_orig | env[PTR_COMMON_ORIG..PTR_COMMON_ORIG+3] | non-zero origin assignment | WIRED | fixtures.rs:165-167: writes origin[0], origin[1], origin[2] to env slots |
| crates/cintx-oracle/tests/common_orig_roundtrip.rs | plan.operator_env_params.common_orig | eval_raw populates slot from fixture env | WIRED | Round-trip test mirrors the internal eval_raw read and asserts equality with COMMON_ORIG_FIXTURE_ORIGIN |

### Data-Flow Trace (Level 4)

Not applicable — this phase delivers infrastructure (a slot, a validator, a fixture) rather than a component that renders dynamic data. The round-trip test verifies data flows end-to-end from fixture env through `eval_raw` into the plan field.

### Behavioral Spot-Checks

Environment note: `cargo build -p cintx-compat -p cintx-runtime -p cintx-rs -p cintx-oracle` builds clean per the verification environment notes. `cargo test -p cintx-runtime common_orig` passes 4/4. `cargo test -p cintx-oracle --features cpu --test common_orig_roundtrip` passes 2/2. `cargo test -p cintx-rs common_origin` passes 2/2. Full `cargo test --workspace` passes 469/469.

| Behavior | Evidence | Status |
|----------|----------|--------|
| 4 D-01 validator unit tests pass (default None, accepts None, accepts finite, rejects non-finite) | Provided: cargo test -p cintx-runtime common_orig → 4/4 pass | PASS |
| Round-trip integration tests pass | Provided: cargo test -p cintx-oracle --features cpu --test common_orig_roundtrip → 2/2 pass | PASS |
| Safe-API path validates NaN/inf on with_common_origin | api.rs:201-210: validator call present after propagation block; regression tests `evaluate_rejects_non_finite_common_origin_on_safe_api_path` and `evaluate_accepts_finite_common_origin_on_safe_api_path` both pass (cargo test -p cintx-rs common_origin → 2/2 pass) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FND-01 | 22-01, 22-02 | PTR_COMMON_ORIG gauge-origin env slot plumbed end-to-end; non-zero oracle fixture exists; env round-trip + validator unit tests pass | SATISFIED | All slot plumbing, fixture, round-trip tests, and validator gate are present and passing on BOTH the raw path and the safe-API path. The "gauge-origin must be finite before kernel entry" contract now holds for all callers. 469/469 tests pass across the workspace. |

FND-01 is the only requirement ID declared in both plan frontmatter sections. No orphaned requirements exist — FND-01 is mapped to Phase 22 in REQUIREMENTS.md traceability table (line 194).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cintx-oracle/tests/common_orig_roundtrip.rs` | 6, 23, 68 | Stale raw.rs line-number citations ("raw.rs:604-615") that refer to wrong lines in the current file (the common_orig block is at 625-632) | INFO | Misleads future readers; no functional impact |
| `crates/cintx-runtime/src/validator.rs` | 187-196 | `match params.rinv_orig { None => {...}, _ => {} }` single-arm match (clippy `single_match`); pre-existing adjacent to Phase 22 work; new `validate_common_orig_env_params` correctly uses `if let` | INFO | Style inconsistency; potential `-D warnings` clippy failure under strict CI; pre-existing, not introduced by Phase 22 |

No blockers remain. The single blocker from the initial verification (WR-01: safe-API path missing validator call) is resolved.

### Human Verification Required

None. All truths are observable programmatically via code inspection and the provided test results.

### Gaps Summary

No gaps. All 7/7 must-have truths are verified.

The single blocker from the initial verification (WR-01) has been resolved by commit 4ecefa0:
- `api.rs:201-210` now calls `cintx_runtime::validator::validate_common_orig_env_params(plan.descriptor.operator_name(), &plan.operator_env_params).map_err(FacadeError::from)?` immediately after the `common_orig` propagation block.
- `cintxRsError::InvalidEnvParam { param, reason }` maps to `FacadeError::Validation` via the existing `From` impl in `error.rs:111-112`, so `FacadeErrorKind::Validation` and `"PTR_COMMON_ORIG"` both appear in the error — exactly what the two new regression tests assert.
- The safe-API path and the raw path now enforce the finiteness contract symmetrically.

---

_Verified: 2026-05-29T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
