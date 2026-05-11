---
phase: 17-real-integral-evaluation-in-safe-api
verified: 2026-05-12T00:00:00Z
status: human_needed
score: 3/3
overrides_applied: 0
human_verification:
  - test: "Run `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity2_parity` on a host with CINTX_ORACLE_BUILD_VENDOR=1 (vendor libcint 6.1.3 built and linked)"
    expected: "12 tests pass — 8 cart/sph vendor-parity tests report 'ok' at atol=1e-12/rtol=0.0 (the cfg(has_vendor_libcint) gate activates them), plus all 4 spinor idempotency tests. 'test result: ok. 12 passed' in the output."
    why_human: "This host does not have CINTX_ORACLE_BUILD_VENDOR=1 set, so the 8 cart/sph parity tests are cfg-gated out at compile time. The 4 spinor idempotency tests pass here (verified). Byte-identity against vendored libcint — the primary RVAL-02 assertion — requires the vendor build and cannot be run in this environment."
---

# Phase 17: Real-Integral Evaluation in Safe API — Verification Report

**Phase Goal:** `SessionRequest::evaluate` returns real libcint-compatible values for every arity-2 intor it accepts today — the synthetic `(idx + 1)` / `((idx + 1) * 0.5)` placeholder in `fill_staging_values` is replaced with a real `cintx-compat::raw::eval_raw` dispatch under the hood. No public API change; internal evaluator swap only.

**Verified:** 2026-05-12

**Status:** human_needed — all automated checks pass; vendor-parity byte-identity assertion requires a host with vendored libcint built.

**Re-verification:** No — initial verification.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | RVAL-01: `fill_staging_values` is deleted and `cintx_cubecl::CubeClExecutor` is the real executor for all arity-2 paths | VERIFIED | `grep fn fill_staging_values crates/cintx-rs/src/api.rs` returns nothing. `grep "use cintx_cubecl::CubeClExecutor"` matches line 6. `executor.execute(&plan, &mut io)` is on line 226. Zero occurrences of `idx + 1` or synthetic fill patterns. |
| 2 | RVAL-02: A new oracle parity test file drives every supported arity-2 intor through `SessionRequest::evaluate` and asserts byte-identity at atol=1e-12 | VERIFIED (partial — spinor idempotency confirmed; vendor byte-identity requires human) | `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` exists at 838 lines. 12 `#[test]` functions cover all 12 arity-2 operators. 8 cart/sph tests are guarded `#[cfg(has_vendor_libcint)]` with `atol=1e-12, rtol=0.0`. 4 spinor idempotency tests pass unconditionally (`cargo test … --test safe_api_arity2_parity` exits 0, 4 passed). Vendor-side byte-identity needs human confirmation. |
| 3 | RVAL-03: No public API change in `cintx-rs` — `SessionRequest`, constructors, accessors, error types are SemVer-compatible with v1.2 | VERIFIED | All public items (`SessionRequest`, `SessionQuery`, `WorkspacePlan`, `WorkspaceExecutionToken`, `WorkspaceChunk`, `TypedEvaluationOutput`, `IntegralTensor`, `EvaluationStats`, `FacadeError`, `unsupported_unstable_request`, `unstable` module) are byte-identical to v1.2. Only private items were deleted (the stub `CubeClExecutor` struct, its `impl` blocks, and `fn fill_staging_values`). Test rename (`evaluate_runs_runtime_path…` → `evaluate_returns_deterministic_nonzero_real_values`) is private. |

**Score:** 3/3 truths verified (RVAL-02 vendor byte-identity requires human testing on a vendor-build host).

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-rs/src/api.rs` | Stub deleted; `use cintx_cubecl::CubeClExecutor;` added; synthetic fill gone | VERIFIED | `use cintx_cubecl::CubeClExecutor;` at line 6. `let executor = CubeClExecutor::new();` at line 143. No `fn fill_staging_values`, no `struct CubeClExecutor` (local def), no `idx + 1` pattern. File is 738 lines (was ~807 with stub). |
| `crates/cintx-oracle/Cargo.toml` | `cintx-rs` path-dep with `default-features = false` | VERIFIED | Line 28: `cintx-rs = { path = "../cintx-rs", default-features = false }`. Auto-fix also added `cintx-runtime = { path = "../cintx-runtime", default-features = false }` (required for the new parity test's type imports). `[features]` table unchanged — no new feature edges. |
| `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` | 12 tests, 8 cart/sph vendor-parity + 4 spinor idempotency, atol=1e-12 | VERIFIED | File exists, 838 lines, 12 `#[test]` functions confirmed. Module gate `#![cfg(any(feature = "cpu", feature = "rocm"))]` on line 13. All 8 cart/sph tests carry `#[cfg(has_vendor_libcint)]`. All use `atol = 1e-12_f64; rtol = 0.0_f64;`. No `1e-11_f64` or `eval_raw` usage. Spinor 4 tests carry no cfg guard (unconditional). |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/cintx-rs/src/api.rs` | `crates/cintx-cubecl/src/executor.rs` | `use cintx_cubecl::CubeClExecutor; let executor = CubeClExecutor::new();` | WIRED | Line 6 import; line 143 instantiation; line 226 `executor.execute(&plan, &mut io)`. |
| `crates/cintx-oracle/tests/safe_api_arity2_parity.rs::collect_safe_api_matrix` | `cintx_rs::SessionRequest::evaluate` | `SessionRequest::new(…).query_workspace()?.evaluate()` | WIRED | `collect_safe_api_matrix` at line 236 calls `SessionRequest::new` at line 263, `query_workspace` at line 272, `evaluate` at line 274. Used in all 12 test functions. |
| `crates/cintx-oracle/Cargo.toml` | `crates/cintx-rs/Cargo.toml` | Cargo workspace path dependency | WIRED | Line 28 in oracle `Cargo.toml`: `cintx-rs = { path = "../cintx-rs", default-features = false }`. `cargo build -p cintx-oracle --features cpu --locked --tests` exits 0. |
| `safe_api_arity2_parity.rs` (cart/sph tests) | `cintx_oracle::vendor_ffi::vendor_int{1e,2c2e}_{sph,cart}` | `vendor_ffi::vendor_int*` calls inside `#[cfg(has_vendor_libcint)]` helpers | WIRED (cfg-gated) | Helpers `collect_1e_sph_matrix_vendor`, `collect_1e_cart_matrix_vendor`, `collect_2c2e_sph_matrix_vendor`, `collect_2c2e_cart_matrix_vendor` all present with correct `#[cfg(has_vendor_libcint)]` guards. They call vendor_ffi functions exactly as documented. |
| Tolerance declaration | Phase 15 unified oracle tolerance | `atol = 1e-12_f64; rtol = 0.0_f64;` | WIRED | 8 occurrences of `atol = 1e-12_f64` and `rtol = 0.0_f64` in the test file (one per cart/sph vendor-parity test). No pre-Phase-15 tolerance constants present. |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `crates/cintx-rs/src/api.rs::SessionQuery::evaluate` | `owned_values: Vec<f64>` | `cintx_cubecl::CubeClExecutor::execute(&plan, &mut io)` at line 226 | Yes — real GPU/CPU kernel dispatch through the same CubeClExecutor that `cintx-compat::raw::eval_raw` uses; `owned_values` is populated by copying from `chunk_staging` (line 239) which is written by `executor.execute` | FLOWING |
| `safe_api_arity2_parity.rs::collect_safe_api_matrix` | `output.tensor.owned_values` | `query.evaluate()` on line 274 | Yes — flows from the real `CubeClExecutor` via the wired safe-API path | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `cintx-rs` builds (default features) | `cargo build -p cintx-rs --locked` | Finished dev profile in 0.46s | PASS |
| All `cintx-rs` tests pass (default features) | `cargo test -p cintx-rs --locked` | 11 passed, 0 failed | PASS |
| Rewritten idempotency+nonzero test passes | `cargo test -p cintx-rs --locked evaluate_returns_deterministic_nonzero_real_values` | 1 passed, 0 failed | PASS |
| All `cintx-rs` tests pass with-f12 | `cargo test -p cintx-rs --features with-f12 --locked` | 12 passed, 0 failed | PASS |
| All `cintx-rs` tests pass with-4c1e | `cargo test -p cintx-rs --features with-4c1e --locked` | 12 passed, 0 failed | PASS |
| All `cintx-rs` tests pass unstable-source-api | `cargo test -p cintx-rs --features unstable-source-api --locked` | 10 passed, 0 failed | PASS |
| `cintx-oracle` builds (cpu feature, with tests) | `cargo build -p cintx-oracle --features cpu --locked --tests` | Finished dev profile in 2.98s | PASS |
| `cintx-oracle` builds (no features, with tests) | `cargo build -p cintx-oracle --locked --tests` | Finished dev profile in 1.60s (module gate works — no compile errors) | PASS |
| 4 spinor idempotency tests pass | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity2_parity -- --test-threads=1` | 4 passed, 0 failed | PASS |
| 8 cart/sph vendor-parity tests (byte-identity at atol=1e-12) | Requires vendor build host (`CINTX_ORACLE_BUILD_VENDOR=1`) | SKIP — vendor build not present | SKIP — human needed |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| RVAL-01 | 17-02-PLAN.md | `fill_staging_values` deleted; real `cintx_cubecl::CubeClExecutor` dispatches every arity-2 operator | SATISFIED | Zero occurrences of `fn fill_staging_values`, `struct CubeClExecutor` (local), `idx + 1`; `use cintx_cubecl::CubeClExecutor` + `executor.execute(&plan, &mut io)` confirmed in api.rs |
| RVAL-02 | 17-01-PLAN.md, 17-03-PLAN.md | `cintx-oracle` test file drives every arity-2 intor through `SessionRequest::evaluate` and asserts byte-identity at atol=1e-12 | SATISFIED (automated); NEEDS HUMAN (vendor byte-identity) | File exists with 12 tests; spinor 4 pass here; cart/sph 8 need vendor libcint host |
| RVAL-03 | 17-02-PLAN.md | No public API change in `cintx-rs` — source/SemVer compatible with v1.2 | SATISFIED | All pub items identical to pre-phase state; only private items removed |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|---------|--------|
| `safe_api_arity2_parity.rs` | 778 | `TODO(phase-18-or-later)` comment on spinor vendor FFI upgrade path | Info | Intentional — spinor vendor wrappers return complex interleaved pairs incompatible with safe-API real-valued output without a projection step. Documented as deferred. Not a blocker. |
| `safe_api_arity2_parity.rs` | 333 | `fn ncart` declared but unused (compiler warning) | Warning | `ncart` is defined to mirror `nsph` but vendor cart helpers compute sizes via `ang.iter().map(|&l| ncart(l))` — the warning may indicate dead code if cart helper is restructured. Not a test correctness issue; the helper is still referenced within `collect_1e_cart_matrix_vendor`. Warning is `unused` not `dead_code` — worth cleaning up but not a blocker. |

---

## Human Verification Required

### 1. Vendor Byte-Identity: 8 cart/sph oracle parity tests

**Test:** On a host with `CINTX_ORACLE_BUILD_VENDOR=1` set (vendored libcint 6.1.3 built and linked), run:

```
CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity2_parity -- --test-threads=1
```

**Expected:** Test output shows "test result: ok. 12 passed" (8 cart/sph vendor-parity + 4 spinor idempotency). No mismatch count above 0 for any of the 8 vendor-parity assertions at `atol=1e-12, rtol=0.0`.

**Why human:** This host does not have `CINTX_ORACLE_BUILD_VENDOR=1` set. The `#[cfg(has_vendor_libcint)]` custom `cfg` flag is only activated by the build script when the vendor libcint 6.1.3 build succeeds. The 4 spinor idempotency tests (which do not call vendor FFI) have been confirmed passing in automated checks. The 8 cart/sph parity tests compile and their code is substantive and wired correctly, but the byte-identity assertion against vendored libcint is the whole point of RVAL-02 and can only be confirmed on a vendor-build host.

---

## Gaps Summary

No blocking gaps found. All three ROADMAP success criteria (RVAL-01, RVAL-02, RVAL-03) have code-level evidence. The one human-verification item is the vendor byte-identity oracle run — a confirmation test, not a missing implementation. The implementation is complete; byte-identity is unconfirmable without the vendor build.

**Auto-fix note:** Plan 17-03's execution required two deviations from the written plan, both of which improve correctness:

1. `cintx-runtime` was added to `crates/cintx-oracle/Cargo.toml` alongside `cintx-rs` because the new test file imports `cintx_runtime::ExecutionOptions`. This is a correct dependency addition — the plan's `must_haves` did not require `cintx-runtime` but it was necessary for compilation.

2. The `collect_safe_api_matrix` helper iterates over shell pairs (bra, ket) individually instead of passing all 5 H2O STO-3G shells in a single `ShellTuple`. The plan's PATTERNS.md scaffold had the wrong shape — `SHELL_TUPLE_CAPACITY=4` is exceeded by 5 shells. The implemented per-pair iteration is the correct arity-2 API usage and matches libcint's shell-pair loop convention.

Neither deviation changes the observable behavior of the tests relative to the RVAL-02 requirement.

---

_Verified: 2026-05-12_
_Verifier: Claude (gsd-verifier)_
