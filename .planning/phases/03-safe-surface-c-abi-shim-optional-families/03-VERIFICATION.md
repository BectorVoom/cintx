---
phase: 03-safe-surface-c-abi-shim-optional-families
verified: 2026-03-28T07:23:46Z
status: passed
score: 16/16 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 14/16
  gaps_closed:
    - "Cargo feature-gate artifacts for Phase 3 are substantive enough to satisfy the declared must-have contract."
    - "Safe facade ergonomic artifacts are substantive (builder + prelude) as declared in Plan 03 must_haves."
    - "Optional/unstable UnsupportedApi decisions propagate from compat raw path into safe facade mapping."
  gaps_remaining: []
  regressions: []
---

# Phase 3: Safe Surface, C ABI Shim & Optional Families Verification Report

**Phase Goal:** Expose the safe Rust facade, optional C shim, and gated optional families once the runtime is stable.
**Verified:** 2026-03-28T07:23:46Z
**Status:** passed
**Re-verification:** Yes - after gap closure

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | `cintx-rs` and `cintx-capi` are active workspace members. | ✓ VERIFIED | Workspace/default-members include both crates in `Cargo.toml`; `cargo metadata --no-deps` shows both in `workspace_members`. |
| 2 | Cargo feature gates exist for `with-f12`, `with-4c1e`, and `unstable-source-api` compile-time controls. | ✓ VERIFIED | Root + crate manifests contain all gates (`Cargo.toml`, `crates/cintx-rs/Cargo.toml`, `crates/cintx-capi/Cargo.toml`). |
| 3 | Stable `cintx-rs` exports remain default; unstable APIs are gated behind `unstable-source-api`. | ✓ VERIFIED | `crates/cintx-rs/src/lib.rs` and `crates/cintx-rs/src/api.rs` gate unstable exports via `#[cfg(feature = "unstable-source-api")]`. |
| 4 | `cintx-capi` remains stable-only in Phase 3. | ✓ VERIFIED | `crates/cintx-capi/src/lib.rs` exports only `errors`/`shim`; `CAPI_EXPOSES_UNSTABLE_SOURCE_API = false`. |
| 5 | `with-f12` only allows validated sph F12/STG/YP envelope and rejects unsupported representations explicitly. | ✓ VERIFIED | `validate_f12_envelope` in `crates/cintx-compat/src/raw.rs`; verified by `cargo test -p cintx-compat --lib --features with-f12` (29 passed). |
| 6 | `with-4c1e` only allows Validated4C1E envelope and rejects out-of-envelope inputs explicitly. | ✓ VERIFIED | `validate_4c1e_envelope` in `raw.rs` + `"outside Validated4C1E"` checks in `crates/cintx-cubecl/src/executor.rs`; validated by `cargo test -p cintx-compat --lib --features with-4c1e` (30 passed). |
| 7 | Source-only symbols fail explicitly when `unstable-source-api` is disabled. | ✓ VERIFIED | `validate_profile_and_source_gate` returns explicit source-only error in `raw.rs`; safe facade test `evaluate_rejects_source_only_symbols_via_compat_policy_gate` passes. |
| 8 | Manifest/resolver metadata is the source of truth for optional/unstable family availability. | ✓ VERIFIED | `compiled_manifest.lock.json` + generated manifest tables + resolver profile/source APIs (`compiled_in_profiles`, `is_source_only`). |
| 9 | Safe Rust callers can query workspace separately before evaluation. | ✓ VERIFIED | `SessionRequest::query_workspace()` returns `SessionQuery` with `WorkspacePlan` in `crates/cintx-rs/src/api.rs`; covered by passing unit tests. |
| 10 | Safe evaluation returns owned typed outputs from typed session/query contract. | ✓ VERIFIED | `SessionQuery::evaluate()` returns `TypedEvaluationOutput` with owned `Vec<f64>`; validated by `evaluate_runs_runtime_path_and_returns_owned_output`. |
| 11 | Optional/unstable `UnsupportedApi` decisions are wired from compat raw path into safe facade mapping. | ✓ VERIFIED | `cintx-rs` now depends on `cintx-compat`; `api.rs` imports/calls `enforce_safe_facade_policy_gate` before/after plan build. |
| 12 | Query/evaluate contract drift is explicitly rejected with fail-closed behavior preserved. | ✓ VERIFIED | `WorkspaceExecutionToken::ensure_matches` rejects drift; ownership checks enforce `BackendStagingOnly` + `CompatFinalWrite`. |
| 13 | C callers get `0` success and nonzero typed statuses on failure. | ✓ VERIFIED | `CintxStatus::Success = 0`; shim maps failures to status codes in `run_with_status`; `cargo test -p cintx-capi --lib` (13 passed). |
| 14 | C error details are thread-local and copy-out accessible. | ✓ VERIFIED | TLS `LAST_ERROR` and copy-out externs in `crates/cintx-capi/src/errors.rs`; thread isolation tests pass. |
| 15 | C shim wrappers are thin compat-style wrappers with fail-closed behavior. | ✓ VERIFIED | `cintrs_query_workspace`/`cintrs_eval` call `query_workspace_raw`/`eval_raw` directly with panic boundary + null pointer guards. |
| 16 | C ABI remains stable-only (no unstable source-only C symbols). | ✓ VERIFIED | `crates/cintx-capi/src/lib.rs` stable-only exports unchanged; no unstable-source C exports present. |

**Score:** 16/16 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `Cargo.toml` | Workspace/member activation and top-level Phase 3 feature declarations | ✓ VERIFIED | 62 lines (>=40); workspace/default-members and feature forwarding present. |
| `crates/cintx-rs/Cargo.toml` | Safe facade dependency and feature wiring | ✓ VERIFIED | 34 lines (>=20); includes explicit `cintx-compat` bridge + forwarded optional/unstable features. |
| `crates/cintx-capi/Cargo.toml` | Stable-only C ABI crate wiring with optional-family forwarding | ✓ VERIFIED | 26 lines (>=20); forwards `with-f12`/`with-4c1e`, no unstable-source forwarding. |
| `crates/cintx-rs/src/api.rs` | Typed session/query/evaluate facade and compat-policy gate wiring | ✓ VERIFIED | 746 lines (>=560); runtime query/evaluate + compat raw policy gate integration present. |
| `crates/cintx-rs/src/builder.rs` | Expanded typed builder API and tests | ✓ VERIFIED | 186 lines (>=120); includes composition, clear helpers, rebuild-from-request tests. |
| `crates/cintx-rs/src/prelude.rs` | Curated stable re-exports with unstable gate boundary | ✓ VERIFIED | 40 lines (>=30); re-exports builder/session/types/options with cfg-gated unstable namespace. |
| `crates/cintx-rs/src/error.rs` | Stable facade error enum and core-error category mapping | ✓ VERIFIED | 126 lines (>=80); typed categories + `From<cintxRsError>` mapping implemented. |
| `crates/cintx-capi/src/errors.rs` | C status taxonomy, TLS error storage, copy-out APIs | ✓ VERIFIED | 410 lines (>=150); extern copy-outs + thread-local state + tests present. |
| `crates/cintx-capi/src/shim.rs` | Thin extern C wrappers over compat raw with status mapping | ✓ VERIFIED | 659 lines (>=220); direct raw calls, status handling, panic boundary, guardrails present. |
| `crates/cintx-capi/src/lib.rs` | Stable C ABI export boundary | ✓ VERIFIED | 36 lines (>=30); stable-only exports and constant boundary checks present. |
| `crates/cintx-compat/src/raw.rs` | Optional/source gate policy helpers and envelope validation | ✓ VERIFIED | 1482 lines (>=1360); exposes `enforce_safe_facade_policy_gate` + profile/envelope checks + tests. |
| `crates/cintx-cubecl/src/executor.rs` | Validated4C1E envelope enforcement | ✓ VERIFIED | 437 lines (>=260); explicit `outside Validated4C1E` rejection path. |
| `crates/cintx-ops/src/resolver.rs` | Profile/source-aware resolver metadata APIs | ✓ VERIFIED | 506 lines (>=220); `compiled_in_profiles` and source-only helpers available. |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | Optional/unstable profile inventory | ✓ VERIFIED | 2955 lines (>=300); includes `with-f12`, `with-4c1e`, and unstable profile metadata. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `Cargo.toml` | `crates/cintx-rs/Cargo.toml` | Workspace and feature wiring for safe facade | WIRED | Root features forward to `cintx-rs` (`with-f12`, `with-4c1e`, `unstable-source-api`). |
| `Cargo.toml` | `crates/cintx-capi/Cargo.toml` | Workspace and feature wiring for optional stable C ABI | WIRED | Root `capi`/optional family forwarding aligns with crate-local features. |
| `crates/cintx-rs/src/lib.rs` | `crates/cintx-rs/src/api.rs` | Stable default exports and cfg-gated unstable namespace | WIRED | Matching `#[cfg(feature = "unstable-source-api")]` boundary in both files. |
| `compiled_manifest.lock.json` | `crates/cintx-ops/src/generated/api_manifest.rs` | Generated optional/unstable profile tables | WIRED | Both artifacts carry `with-f12`, `with-4c1e`, and unstable profile data. |
| `crates/cintx-ops/src/resolver.rs` | `crates/cintx-compat/src/raw.rs` | Resolver metadata drives profile/source gate behavior | WIRED | Raw path uses resolver descriptors and profile/source checks (`descriptor_by_symbol`, `is_compiled_in_profile`, `is_source_only`). |
| `crates/cintx-compat/src/raw.rs` | `crates/cintx-cubecl/src/executor.rs` | Shared Validated4C1E fail-closed envelope semantics | WIRED | Raw and executor both emit explicit `outside Validated4C1E` rejection reasons. |
| `crates/cintx-rs/src/api.rs` | `crates/cintx-runtime/src/planner.rs` | Safe query/evaluate uses runtime planner/evaluator | WIRED | Calls `runtime_query_workspace`, `ExecutionPlan::new`, `runtime_evaluate`. |
| `crates/cintx-rs/src/error.rs` | `crates/cintx-core/src/error.rs` | Stable facade mapping of core error categories | WIRED | `From<cintxRsError>` maps Unsupported/Layout/Memory/Validation variants. |
| `crates/cintx-rs/src/api.rs` | `crates/cintx-compat/src/raw.rs` | Compat-policy UnsupportedApi propagation in safe facade | WIRED | `api.rs` imports/calls `enforce_safe_facade_policy_gate` pre/post plan construction. |
| `crates/cintx-capi/src/shim.rs` | `crates/cintx-compat/src/raw.rs` | Thin raw wrapper calls | WIRED | Shim directly calls `query_workspace_raw` and `eval_raw`. |
| `crates/cintx-capi/src/shim.rs` | `crates/cintx-capi/src/errors.rs` | Status mapping and TLS last-error reporting | WIRED | Shim uses `set_last_error` and returns mapped status codes. |
| `crates/cintx-capi/src/lib.rs` | `crates/cintx-capi/src/shim.rs` | Stable-only C export surface | WIRED | `pub mod errors; pub mod shim;` with stable boundary constants/tests. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `EXEC-01` | `03-PLAN.md`, `05-PLAN.md`, `06-PLAN.md` | Rust caller can query workspace separately from evaluation via safe API. | ✓ SATISFIED | `SessionRequest::query_workspace` + `SessionQuery::evaluate` split and tests; `cargo test -p cintx-rs --lib` and feature profiles pass. |
| `COMP-04` | `04-PLAN.md` | Optional C ABI shim returns typed status codes and TLS last-error details. | ✓ SATISFIED | `CintxStatus`, TLS `LAST_ERROR`, copy-out externs, shim wrappers; `cargo test -p cintx-capi --lib` passes. |
| `OPT-01` | `02-PLAN.md`, `06-PLAN.md` | `with-f12` enables only validated sph envelope; out-of-envelope rejects explicitly. | ✓ SATISFIED | `validate_f12_envelope` + safe/compat tests; `cargo test -p cintx-compat --lib --features with-f12` passes. |
| `OPT-02` | `02-PLAN.md`, `06-PLAN.md` | `with-4c1e` constrained to validated bug envelope with explicit rejection out-of-envelope. | ✓ SATISFIED | `validate_4c1e_envelope` + executor checks and tests; `cargo test -p cintx-compat --lib --features with-4c1e` and `cargo test -p cintx-rs --lib --features with-4c1e` pass. |
| `OPT-03` | `01-PLAN.md`, `02-PLAN.md`, `05-PLAN.md`, `06-PLAN.md` | Source-only APIs stay behind `unstable-source-api` without stable GA drift. | ✓ SATISFIED | cfg-gated unstable namespace in `cintx-rs`; raw profile/source gate errors when feature disabled; capi remains stable-only. |

Phase-3 orphaned requirements check: none. `REQUIREMENTS.md` Phase 3 IDs (`COMP-04`, `EXEC-01`, `OPT-01`, `OPT-02`, `OPT-03`) are all claimed by plan frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| None | - | No blocker TODO/FIXME/placeholder/stub anti-patterns detected in phase-owned files. | - | - |

### Human Verification Required

None for this phase pass. Automated verification and targeted feature-matrix unit checks covered the declared must-haves and requirement IDs.

### Gaps Summary

No remaining gaps. The three prior blockers are closed, no regressions were found in previously passing areas, and Phase 3 goal achievement is verified.

---

_Verified: 2026-03-28T07:23:46Z_
_Verifier: Codex (gsd-verifier)_
