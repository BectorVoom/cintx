---
phase: 18-sessionrequest-arity-ge3-dispatch
verified: 2026-05-12T12:30:00Z
status: human_needed
score: 5/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run 8 arity-3 + 4 arity-4 oracle parity tests on a host with vendored libcint (CINTX_ORACLE_BUILD_VENDOR=1)"
    expected: "All 12 tests pass at atol=1e-12, rtol=0.0 against libcint 6.1.3 with 0 mismatches; `test result: ok. 8 passed` for arity-3 + `test result: ok. 4 passed` for arity-4 under --features cpu,with-4c1e; the int3c2e_ip1_{cart,sph} tests in particular reference plain vendor_int3c2e_* and produce 0 mismatches"
    why_human: "This dev host lacks the vendored libcint build (has_vendor_libcint cfg OFF); all 12 new parity tests cfg-strip to 0 tests locally. SC#2 (byte-identity at atol=1e-12) requires CI matrix execution with CINTX_ORACLE_BUILD_VENDOR=1 across the four manifest profiles (base, with-f12, with-4c1e, with-f12+with-4c1e). Cannot be verified programmatically on this host."
  - test: "Verify F-order AO axis layout via the implicit oracle parity sweep on a vendor-built host"
    expected: "ARITY-03 (SC#3) is implicitly verified by SC#2 success: byte-identity vs vendored libcint with NO transpose means cintx is writing F-order by construction. If layout drifted, the first parity element would mismatch and `total_mismatches > 0` would fire."
    why_human: "Same vendor-build prerequisite. The F-order rustdoc claim on IntegralTensor is documentation; the verification mechanism is the oracle parity sweep that requires libcint."
deferred: []
---

# Phase 18: SessionRequest Arity ≥3 Dispatch Verification Report

**Phase Goal:** `SessionRequest::evaluate` dispatches arity-3 and arity-4 shell tuples through the existing operator catalog, returning tensors with F-order AO axes that match libcint memory layout. Covers `int2e_*` (SCF J/K hot path), `int3c1e`, `int3c1e_p2`, `int3c2e_ip1`, `int3c2e_sph`, `int3c2e_cart`, `int4c1e_sph`, `int4c1e_cart`.

**Verified:** 2026-05-12T12:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria — non-negotiable contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC#1 | `SessionRequest::evaluate` accepts arity-3 `(i,j,k)` and arity-4 `(i,j,k,l)` shell tuples and routes them to existing `cintx-ops` resolver entries (no parallel evaluator API). (ARITY-01) | VERIFIED | `crates/cintx-oracle/tests/safe_api_arity3_parity.rs:235-258` (`collect_safe_api_tuple_buffer`) builds a `ShellTuple::try_from_iter` from a 3-shell slice and calls `SessionRequest::new(...).query_workspace()?.evaluate()` — the existing arity-generic chunk loop. `safe_api_arity4_parity.rs:232-260` does the same with a 4-shell slice. All 8 arity-3 tests + 4 arity-4 tests use this path through `OperatorId::new(N)` against the resolver — no new evaluator API was introduced. Manifest entries at OperatorId 22/23 added by Plan 18-01; existing IDs 9/10/15-20 unchanged. |
| SC#2 | Arity-3 (`int3c1e`, `int3c1e_p2`, `int3c2e_ip1`, `int3c2e_sph`, `int3c2e_cart`) and arity-4 (`int2e_sph`, `int2e_cart`, `int4c1e_sph`, `int4c1e_cart`) round-trip through the safe API with byte-identity values against libcint at `atol=1e-12`. (ARITY-02) | VERIFIED (artifacts) / human-needed (runtime) | All 12 test functions exist, each iterating the full Cartesian sweep on H2O/STO-3G with `count_mismatches(..., atol=1e-12, rtol=0.0)` and `any_nonzero` sentinel. `grep -F 'const ATOL: f64 = 1e-12' crates/cintx-oracle/tests/safe_api_arity{3,4}_parity.rs` matches in both files. Length asserts (8 in arity-3, no explicit count in arity-4 but the panic from count_mismatches' `assert_eq!(reference.len(), observed.len(), ...)` would fire). **Runtime confirmation requires `CINTX_ORACLE_BUILD_VENDOR=1` host** — this host's `has_vendor_libcint` cfg is OFF so all 12 tests cfg-strip; CI matrix exercises them. See human_verification below. |
| SC#3 | Output tensors expose F-order AO axes consistent with libcint memory layout. (ARITY-03) | VERIFIED (documented + tests structured for verification) / human-needed (runtime) | `crates/cintx-rs/src/api.rs:452-480` carries the arity-aware F-order rustdoc on `IntegralTensor`. Arity-3 tests (`safe_api_arity3_parity.rs`) and arity-4 tests (`safe_api_arity4_parity.rs`) compare buffers byte-to-byte with NO transpose — if cintx wrote any other layout, `total_mismatches > 0` would fire. The verification mechanism is structurally correct; runtime confirmation requires vendor build (per SC#2). |
| SC#4 | Two-electron symmetry packing follows pyscf's `aosym` convention (`s1`, `s2ij`, `s2kl`, `s4`, `s8`) where supported, or returns a typed error documenting which packings are not yet implemented. (ARITY-04) | VERIFIED | `crates/cintx-core/src/operator.rs:22-48` defines `AoSymmetry { S1, S2ij, S2kl, S4, S8 }` with `#[default] S1`, `#[repr(u8)]`, and lowercase pyscf `Display`. `crates/cintx-runtime/src/options.rs:121` adds `pub aosym: Option<cintx_core::AoSymmetry>`. `crates/cintx-rs/src/api.rs:63-73` performs aosym preflight: `if let Some(aosym) = self.options.aosym { if aosym != cintx_core::AoSymmetry::S1 { return Err(FacadeError::UnsupportedAoSymmetry { requested: aosym.to_string() }); } }`. `crates/cintx-rs/src/error.rs:25-26` defines `FacadeError::UnsupportedAoSymmetry { requested: String }` with thiserror `#[error("unsupported aosym packing: {requested}")]`; `FacadeErrorKind::UnsupportedAoSymmetry` at line 12 and matching `kind()` arm at line 36 keep the enum exhaustive. Two new unit tests at `api.rs:780-833` verify (a) all four non-S1 variants return the typed error with the lowercase pyscf form in `requested`, and (b) both `None` and `Some(S1)` succeed through `query_workspace`. `cargo test -p cintx-rs --locked` exits 0 with 13 passed (including the two new tests). |
| SC#5 | Oracle parity tests for arity-3 and arity-4 dispatch are added to `cintx-oracle` and gate CI alongside the existing arity-2 parity tests. (ARITY-05) | VERIFIED (artifact-level) | `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` exists (33,318 bytes, 801 lines, 8 `#[test]` functions, all `#[cfg(has_vendor_libcint)]`); `safe_api_arity4_parity.rs` exists (25,689 bytes, 594 lines, 4 `#[test]` functions, 2 with extra `#[cfg(feature = "with-4c1e")]`). Both files use the same module gate `#![cfg(any(feature = "cpu", feature = "rocm"))]` as the existing `safe_api_arity2_parity.rs`. Per CONTEXT.md D-15, no new CI job needed — the existing `oracle_parity_gate` matrix automatically picks up the new test files. Confirmed both build cleanly under `--features cpu --locked --tests` and `--features cpu,with-4c1e --locked --tests`. |

**Score:** 5/5 truths verified at artifact + structure level; runtime byte-identity for SC#2/SC#3 requires CI vendor matrix.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-core/src/operator.rs` | `AoSymmetry` enum with 5 variants, `#[default] S1`, `#[repr(u8)]`, `Display` emitting lowercase pyscf form | VERIFIED | Lines 22-48; full enum + Display impl present (verified via Read). |
| `crates/cintx-core/src/lib.rs` | `AoSymmetry` re-export | VERIFIED | Line 18: `pub use operator::{AoSymmetry, OperatorId, Representation};` |
| `crates/cintx-rs/src/prelude.rs` | `AoSymmetry` re-export | VERIFIED | Line 31: `pub use cintx_core::AoSymmetry;` |
| `crates/cintx-runtime/src/options.rs` | `ExecutionOptions::aosym: Option<cintx_core::AoSymmetry>` field | VERIFIED | Line 121: `pub aosym: Option<cintx_core::AoSymmetry>,` after `f12_zeta` |
| `crates/cintx-rs/src/error.rs` | `FacadeError::UnsupportedAoSymmetry { requested: String }` + `FacadeErrorKind::UnsupportedAoSymmetry` + `kind()` arm | VERIFIED | Lines 12, 25-26, 36 — all three additive edits present; existing variants preserved in original order |
| `crates/cintx-rs/src/api.rs` | aosym preflight + F-order rustdoc + post-shift `INT4C1E_CART_OPERATOR_ID = 24` + two unit tests | VERIFIED | Lines 63-73 (preflight with `if let Some(aosym)` guard, fails fast before `runtime_query_workspace`); 452-480 (arity-aware F-order rustdoc with both Arity ≥ 3 / Arity 2 rows); 541 (`INT4C1E_CART_OPERATOR_ID: u32 = 24`); 780-833 (two new tests). |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | `int3c2e_cart` + `int3c2e_sph` operator-kind rows with stability stable, all 4 profiles, oracle_covered=true | VERIFIED | Lines 360 (`"symbol": "int3c2e_cart"`) and 376 (`"symbol": "int3c2e_sph"`). Lines 406 / 434 confirm `int4c1e_cart` / `int4c1e_sph` are present (post-shift positions). |
| `crates/cintx-ops/src/generated/api_manifest.rs` | Auto-regenerated; `int3c2e_cart` at MANIFEST_ENTRIES[22], `int3c2e_sph` at [23], `int4c1e_cart` at [24], `int4c1e_sph` at [25] | VERIFIED | awk over MANIFEST_ENTRIES confirms: 22=int3c2e_cart, 23=int3c2e_sph, 24=int4c1e_cart, 25=int4c1e_sph. |
| `crates/cintx-ops/src/resolver.rs` | `"int3c2e" => None` arm + early-continue in `legacy_wrapper_manifest_matches_misc` | VERIFIED | Lines 323 and 407 (`let Some(macro_kind) = misc_wrapper_macro(&base_symbol) else { continue; };`). |
| `crates/cintx-ops/src/lib.rs` | Duplicate fixture in second copy of the test got the same treatment | VERIFIED | Lines 21 and 44 in `crates/cintx-ops/src/lib.rs` (deviation 1 from Plan 18-01: a second copy of the test existed in lib.rs; fix applied symmetrically). |
| `crates/cintx-oracle/src/vendor_ffi.rs` | New wrappers `vendor_int3c1e_p2_sph` (R4 sph) and `vendor_int3c2e_cart` (R1) | VERIFIED | Lines 232 (`vendor_int3c2e_cart`) and 524 (`vendor_int3c1e_p2_sph`). NOTE: `vendor_int3c2e_ip1_sph` is intentionally ABSENT (correct per plan — `int3c2e_ip1_sph` parity reference reuses pre-existing `vendor_int3c2e_sph` at line 204). |
| `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` | 8 per-symbol parity tests, all `#[cfg(has_vendor_libcint)]`, OperatorIds 15/16/17/18/19/20/22/23, vendor functions including new `vendor_int3c1e_p2_sph` + `vendor_int3c2e_cart` | VERIFIED | 8 `#[test]` markers; 8 `#[cfg(has_vendor_libcint)]` markers; 8 length asserts; 2 kernel-misnomer NOTE comments; all 8 expected OperatorIds present; correct vendor wrappers used. Module gate `#![cfg(any(feature = "cpu", feature = "rocm"))]` confirmed. |
| `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` | 4 per-symbol parity tests, all `#[cfg(has_vendor_libcint)]`, 2 with extra `#[cfg(feature = "with-4c1e")]`, OperatorIds 9/10/24/25, post-shift values | VERIFIED | 4 `#[test]` markers; 4 `#[cfg(has_vendor_libcint)]` markers; 2 `#[cfg(feature = "with-4c1e")]` markers (correct count); 4 nested-quartet loops; no module-level `#![cfg(feature = "with-4c1e")]`; no pre-shift OperatorIds 22/23 leak through; correct vendor wrappers. |
| `crates/cintx-oracle/src/compare.rs` | `int3c2e_{cart,sph}` routed through `RawApiId::Symbol` dispatch (integration fix, commit 117f185) | VERIFIED | Lines 249-250 in `raw_api_for_symbol`; lines 339-340 in `eval_legacy_symbol`'s optional-families fallback. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `SessionRequest::query_workspace` | `FacadeError::UnsupportedAoSymmetry` | `if let Some(aosym) = self.options.aosym { if aosym != cintx_core::AoSymmetry::S1 { return Err(...) } }` | WIRED | `crates/cintx-rs/src/api.rs:63-73` — preflight runs BEFORE `runtime_query_workspace` call; `aosym.to_string()` uses Display impl from Plan 18-01. |
| `FacadeError::kind()` | `FacadeErrorKind::UnsupportedAoSymmetry` | match arm | WIRED | `crates/cintx-rs/src/error.rs:36` — `Self::UnsupportedAoSymmetry { .. } => FacadeErrorKind::UnsupportedAoSymmetry`. Exhaustive const fn. |
| `IntegralTensor` rustdoc | F-order layout invariant | rustdoc block on struct | WIRED | `crates/cintx-rs/src/api.rs:452-480` — both `Arity >= 3` and `Arity 2` rows present; cross-reference to oracle parity sweep files. |
| `safe_api_arity3_parity.rs::collect_safe_api_tuple_buffer` | `cintx_rs::SessionRequest::evaluate` | `SessionRequest::new(...).query_workspace()?.evaluate()` | WIRED | Lines 235-258 — full safe-API path; `ShellTuple::try_from_iter` from input slice; returns `output.tensor.owned_values`. |
| `safe_api_arity4_parity.rs::collect_safe_api_tuple_buffer` | `cintx_rs::SessionRequest::evaluate` | `SessionRequest::new(...).query_workspace()?.evaluate()` with 4 shells | WIRED | Lines 232-260 — same arity-agnostic pattern as arity-3; `ShellTuple::try_from_iter` accepts up to `SHELL_TUPLE_CAPACITY=4`. |
| arity-3 tests | `vendor_int3c1e_*` / `vendor_int3c2e_*` | `cintx_oracle::vendor_ffi::vendor_int3c*` calls | WIRED | Lines 334, 394, 459, 518 (cart) + 587, 648, 714, 773 (sph). Includes new wrappers `vendor_int3c1e_p2_sph` + `vendor_int3c2e_cart`. |
| arity-4 tests | `vendor_int2e_*` / `vendor_int4c1e_*` | `cintx_oracle::vendor_ffi::vendor_int{2e,4c1e}_*` calls | WIRED | Lines 352, 416, 502, 569. All vendor wrappers pre-existing (none added in Phase 18). |
| `compiled_manifest.lock.json` (entries 21, 22 plain int3c2e) | `api_manifest.rs` OPERATOR_DESCRIPTORS | `build.rs` regeneration | WIRED | Auto-regenerated; verified OperatorIds match positions 22/23 in MANIFEST_ENTRIES. |
| `ExecutionOptions::aosym` field | `cintx_core::AoSymmetry` | `Option<cintx_core::AoSymmetry>` type path | WIRED | `crates/cintx-runtime/src/options.rs:121`. |
| `cintx-rs::prelude` | `cintx-core::AoSymmetry` | `pub use cintx_core::AoSymmetry;` | WIRED | `crates/cintx-rs/src/prelude.rs:31`. |
| `compare.rs::raw_api_for_symbol` and `eval_legacy_symbol` | new `int3c2e_{cart,sph}` symbols | `RawApiId::Symbol` dispatch fallback | WIRED | Post-merge integration fix at commit `117f185` resolves the 3 oracle test failures that surfaced when the base-profile parity test picked up the new fixtures. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `SessionRequest::query_workspace` (post-preflight path) | `runtime_workspace` | `runtime_query_workspace(...)` (cintx-runtime) | Yes — drives the existing arity-generic chunk loop the Phase 17 work already validated | FLOWING |
| arity-3/4 parity tests | `safe_out` (Vec<f64>) | `SessionRequest::evaluate().tensor.owned_values` — populated by the real `CubeClExecutor` via the existing arity-generic path (Phase 17) | Yes — full path through `WorkspacePlan::from_runtime` → `ExecutionPlan::new` → real CubeCL kernel dispatch. Per Plan 18-04 Summary: "the safe-API ExecutionOptions::default() carries aosym = None, which the preflight treats as Some(S1) and lets through (verified by the tests succeeding to build and dispatch)." | FLOWING (structurally) / human-needed (runtime confirmation requires vendor build) |
| `IntegralTensor::owned_values` | Yes (dense F64 buffer of `extents.product()`) | Sourced from CubeCL kernel output via `TypedEvaluationOutput::tensor` | Yes per Phase 17 verification | FLOWING |
| `aosym_error_path_rejects_non_s1_with_typed_error` test | `err` (FacadeError) | `request.query_workspace()` with `Some(S2ij/S2kl/S4/S8)` | Yes — actual preflight returns the typed error with `requested = aosym.to_string()` | FLOWING |

**Note on SC#2 runtime confirmation:** The 12 new parity tests are `#[cfg(has_vendor_libcint)]`-gated. On this host (`has_vendor_libcint` cfg OFF), they cfg-strip to 0 tests. The byte-identity claim against libcint at atol=1e-12 is verified structurally (correct OperatorIds, correct vendor functions, correct buffer dimensions, correct tolerance constants) but not exercised at runtime here. CI matrix cells with `CINTX_ORACLE_BUILD_VENDOR=1` exercise the actual numeric verification.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace builds clean | `cargo build --workspace --locked` | `Finished dev [unoptimized + debuginfo] target(s) in 0.59s` | PASS |
| cintx-oracle builds with cpu + tests | `cargo build -p cintx-oracle --features cpu --locked --tests` | exit 0 (warnings on cfg-stripped dead code only) | PASS |
| cintx-oracle builds with cpu+with-4c1e + tests | `cargo build -p cintx-oracle --features cpu,with-4c1e --locked --tests` | exit 0 | PASS |
| cintx-rs builds clean across feature combos | `cargo build -p cintx-rs --features {with-f12,with-4c1e,unstable-source-api} --locked` | all exit 0 | PASS |
| Full workspace test suite passes | `cargo test --workspace --locked --no-fail-fast` | 33 test result blocks, all `test result: ok`, 0 FAILED | PASS |
| cintx-rs tests (including new aosym tests) pass | `cargo test -p cintx-rs --locked` | `13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out` | PASS |
| arity-3 parity tests cfg-strip cleanly without vendor | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity` | `test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out` | PASS (cfg-strip expected on this host) |
| arity-4 parity tests cfg-strip cleanly with with-4c1e | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu,with-4c1e --locked --test safe_api_arity4_parity` | `test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out` | PASS (cfg-strip expected on this host) |
| arity-3 + arity-4 test counts | grep `^#[test]` | arity-3: 8, arity-4: 4 (matches expected: 4 cart + 4 sph for arity-3; 2 int2e + 2 int4c1e for arity-4) | PASS |
| arity-3 OperatorIds present (15, 16, 17, 18, 19, 20, 22, 23) | grep `OperatorId::new(X)` | all 8 present, each exactly once at test call sites | PASS |
| arity-4 OperatorIds use post-shift values (9, 10, 24, 25), not pre-shift (22, 23) | grep `OperatorId::new(X)` | 9, 10, 24, 25 present at call sites; 22, 23 only in comments, not in tests | PASS |
| Anti-pattern: int3c2e_ip1 tests do NOT use `_ip1_` vendor wrappers | grep `vendor_int3c2e_ip1_(cart|sph)` in test bodies (not comments) | no matches at call sites; only in module-doc comments documenting the disposition | PASS |
| Anti-pattern: no module-level `#![cfg(feature = "with-4c1e")]` in arity-4 file | grep `^#![cfg(feature = "with-4c1e"` | no match | PASS |
| No naïve `.unwrap() != AoSymmetry::S1` | grep `.unwrap() != cintx_core::AoSymmetry::S1` | no match | PASS |
| Manifest lock has new entries | grep `"symbol": "int3c2e_(cart|sph)"` | both present at lines 360, 376 | PASS |
| No synthetic plain `cint3c2e_(cart|sph)` legacy entries | grep `"cint3c2e_cart"|"cint3c2e_sph"` excluding _ip1_ | no matches | PASS |

### Requirements Coverage

**IMPORTANT — REQUIREMENTS.md traceability gap (carried forward from Phase 17 and prior v1.3 phases):**

`.planning/REQUIREMENTS.md` only contains v1.2 sections (HELP, 4C1E, SPIN, F12, USRC, ORAC). It has NO v1.3 section and NO `ARITY-01..ARITY-05` IDs registered. The ROADMAP entry for Phase 18 explicitly states the IDs are "derived from success criteria 1-5 during `/gsd:spec-phase` or `/gsd:plan-phase`" — they were used as logical anchors in plan frontmatter without being committed to REQUIREMENTS.md. Phase 17 (also v1.3) has the same gap.

This is a **bookkeeping defect** that predates Phase 18 and is **NOT a blocker** for this phase per the orchestrator instructions. Requirements coverage is therefore mapped against the ROADMAP Success Criteria (the actual contract):

| Requirement ID | Source Plan(s) | Description (from ROADMAP SC text) | Status | Evidence |
|----------------|----------------|------------------------------------|--------|----------|
| ARITY-01 | 18-01, 18-02, 18-03, 18-04 | SC#1: `SessionRequest::evaluate` accepts arity-3/4 tuples and routes them to existing resolver entries (no parallel evaluator API) | SATISFIED (artifacts + structure) | Manifest expansion (Plan 18-01), `INT4C1E_CART_OPERATOR_ID = 24` shift (Plan 18-01 + 18-02), arity-3/4 test files driving `SessionRequest::evaluate` exclusively (Plans 18-03/04) |
| ARITY-02 | 18-01, 18-03, 18-04 | SC#2: 9 named operators round-trip with byte-identity at atol=1e-12 | SATISFIED (artifact-level) / NEEDS HUMAN (runtime) | 12 parity tests structured for atol=1e-12 verification; runtime requires vendor build (CI) |
| ARITY-03 | 18-02, 18-03, 18-04 | SC#3: F-order AO axis layout consistent with libcint | SATISFIED (artifact-level) / NEEDS HUMAN (runtime) | F-order rustdoc on IntegralTensor + parity tests use no-transpose buffer compare — runtime confirmation via SC#2 path |
| ARITY-04 | 18-01, 18-02 | SC#4: `aosym` packing follows pyscf convention where supported, typed error otherwise | SATISFIED | AoSymmetry enum + ExecutionOptions::aosym + preflight + FacadeError::UnsupportedAoSymmetry + 2 unit tests; aosym_error_path test passes on this host |
| ARITY-05 | 18-03, 18-04 | SC#5: Oracle parity tests for arity-3/4 added and gate CI | SATISFIED | Both new files added; module gate matches `safe_api_arity2_parity.rs`; CI `oracle_parity_gate` matrix automatically picks up new tests per CONTEXT.md D-15 |

### Anti-Patterns Found

None. Comprehensive scans for the anti-patterns listed in plan acceptance criteria confirm:

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| n/a | n/a | All anti-pattern greps cleared (no `vendor_int3c2e_ip1_cart/sph` call-site, no `ni * nj * nk * 3` buffer sizing, no `1e-11` pre-Phase-15 tolerance, no `eval_raw` in safe-API parity tests, no `spinor` in arity-3/4 tests, no module-level `with-4c1e` cfg, no pre-shift OperatorIds 22/23 in arity-4 tests, no synthetic `cint3c2e_{cart,sph}` legacy entries, no naïve `.unwrap()` against S1, no `has_vendor_libcint` in cintx-rs::api unit tests) | n/a | n/a |

Note: dead-code warnings on the inline test helpers (`arc_f64`, `build_h2o_sto3g`, etc.) when `has_vendor_libcint` is OFF are expected on this host — the helpers are referenced exclusively from cfg-stripped test bodies. Mentioned in Plan 18-04 SUMMARY ("8 dead-code warnings on the compiled-out helpers"). Not a defect.

### Human Verification Required

Two items requiring vendor-build matrix execution:

#### 1. Run all 12 new parity tests with vendored libcint enabled

**Test:** On a host with `CINTX_ORACLE_BUILD_VENDOR=1` and the vendored libcint 6.1.3 sources available, run:

```
CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity -- --test-threads=1
CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu,with-4c1e --locked --test safe_api_arity4_parity -- --test-threads=1
```

**Expected:**
- Arity-3 run: `test result: ok. 8 passed; 0 failed; 0 ignored`
- Arity-4 run: `test result: ok. 4 passed; 0 failed; 0 ignored`
- For each test, 0 mismatches at atol=1e-12 (the assertion `assert_eq!(total_mismatches, 0, ...)` does not fire), and `any_nonzero` sentinel fires (assertion does not abort with "all zeros over N triples/quartets").
- For the int3c2e_ip1_{cart,sph} tests in particular: 0 mismatches against plain `vendor_int3c2e_{cart,sph}` (the kernel-misnomer disposition is correct).
- Buffer-length asserts do not fire (cintx and vendor agree on output sizes).

**Why human:** This dev host lacks the vendored libcint build (`has_vendor_libcint` cfg OFF, confirmed by 8 dead-code warnings on the compiled-out helpers per Plan 18-04 SUMMARY). All 12 new tests cfg-strip to 0 tests locally. Programmatic verification requires the CI matrix or a manually configured vendor build environment.

#### 2. Verify F-order AO axis layout via the oracle parity sweep

**Test:** Same as item 1 — the F-order claim is documented by `IntegralTensor` rustdoc and **verified by construction** through the no-transpose direct buffer compare in the 12 new tests.

**Expected:** SC#2 success (item 1) implies SC#3 success. If cintx wrote any other AO axis layout, the first parity test would report `total_mismatches > 0`. There is no separate runtime check for F-order.

**Why human:** Same vendor-build prerequisite as item 1.

### Gaps Summary

No actionable gaps. The phase delivers every must-have at the artifact and structural level. The only outstanding work is **runtime byte-identity verification on a vendor-built host** — this is the standard CI matrix path documented in CONTEXT.md D-15 ("CI integration mirrors Phase 17 — no new CI job required"). Plan 18-03 and 18-04 SUMMARYs explicitly note that local runtime verification was not possible due to the missing vendor build; they correctly defer to CI.

### Traceability Bookkeeping (informational, not a gap)

`.planning/REQUIREMENTS.md` does not contain ARITY-01..ARITY-05 IDs (no v1.3 section exists). This bookkeeping defect predates Phase 18 and also affects Phase 17. The ROADMAP states the IDs are derived during `/gsd:spec-phase` or `/gsd:plan-phase`; plans correctly cite them; the contract is enforced via ROADMAP Success Criteria 1-5 which are explicitly tagged with the ARITY IDs. **Recommendation:** open a follow-up task to backfill v1.3 ARITY-* and HELP/EXP-* requirement IDs into REQUIREMENTS.md before v1.3 closes. NOT a Phase 18 blocker.

### Cross-Plan Integration Fix Verification (commit 117f185)

The post-merge integration fix in `crates/cintx-oracle/src/compare.rs` is present and correct:

- Line 249: `"int3c2e_cart" => Some(RawApiId::Symbol("int3c2e_cart"))`
- Line 250: `"int3c2e_sph" => Some(RawApiId::Symbol("int3c2e_sph"))`
- Lines 339-340: both new symbols added to the optional-families fallback in `eval_legacy_symbol`

This was necessary because Plan 18-01 added the new symbols to the base-profile manifest, which the existing parity-test fixture corpus then picked up — but the hard-coded `raw_api_for_symbol` map and `eval_legacy_symbol` arms did not yet handle them. The fix routes both through `RawApiId::Symbol` dispatch (no dedicated `cint3c2e_{cart,sph}` legacy wrapper exists in `cintx-compat`, matching the Phase 18 PATTERNS.md §Step 3 Option A discipline). All 3 previously-failing `compare::tests::*` tests now pass via the workspace test sweep.

---

## Status determination

- All 5 ROADMAP Success Criteria are SATISFIED at the artifact + structural level.
- All key links are WIRED.
- All anti-pattern scans clear.
- All build/test gates pass on this host (workspace builds clean; 33 test result blocks all OK; 0 FAILED).
- **Two human verification items exist** — both require the vendor-libcint build that is not available on this host. These items take precedence over `passed` per the Step 9 decision tree.

**Final status:** `human_needed`

The phase has been executed correctly. Byte-identity verification at atol=1e-12 against libcint 6.1.3 for the 12 new parity tests is the standard CI gate path. The cfg-gating is correct — the tests will run on every CI cell that sets `CINTX_ORACLE_BUILD_VENDOR=1`, which is the established Phase 15+ pattern.

---

*Verified: 2026-05-12T12:30:00Z*
*Verifier: Claude (gsd-verifier)*
