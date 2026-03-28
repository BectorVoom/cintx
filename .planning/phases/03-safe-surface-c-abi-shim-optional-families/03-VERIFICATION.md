---
phase: 03-safe-surface-c-abi-shim-optional-families
verified: 2026-03-28T01:15:35Z
status: gaps_found
score: 14/16 must-haves verified
gaps:
  - truth: "Cargo feature-gate artifacts for Phase 3 are substantive enough to satisfy the declared must-have contract."
    status: partial
    reason: "Both crate-local manifest artifacts are below their declared min_lines thresholds."
    artifacts:
      - path: "crates/cintx-rs/Cargo.toml"
        issue: "19 lines; must_haves requires min_lines: 20."
      - path: "crates/cintx-capi/Cargo.toml"
        issue: "19 lines; must_haves requires min_lines: 20."
    missing:
      - "Expand crate-local manifest wiring/documentation to meet the declared artifact substance thresholds."
  - truth: "Safe facade ergonomic artifacts are substantive (builder + prelude) as declared in Plan 03 must_haves."
    status: failed
    reason: "Builder and prelude artifacts are present and wired, but both are below declared min_lines thresholds."
    artifacts:
      - path: "crates/cintx-rs/src/builder.rs"
        issue: "56 lines; must_haves requires min_lines: 120."
      - path: "crates/cintx-rs/src/prelude.rs"
        issue: "18 lines; must_haves requires min_lines: 30."
    missing:
      - "Deliver the additional typed builder/prelude surface expected by Plan 03 must_haves."
  - truth: "Optional/unstable UnsupportedApi decisions propagate from compat raw path into safe facade mapping."
    status: failed
    reason: "Plan 03 key link `api.rs -> compat/raw.rs` is not wired; `cintx-rs` does not depend on `cintx-compat` and `api.rs` never calls raw compat APIs."
    artifacts:
      - path: "crates/cintx-rs/src/api.rs"
        issue: "No `cintx_compat`, `query_workspace_raw`, or `eval_raw` references."
      - path: "crates/cintx-rs/Cargo.toml"
        issue: "No `cintx-compat` dependency, so direct compat linkage is impossible."
    missing:
      - "Either wire safe API error propagation through compat raw as declared, or update must_haves/key_links to reflect the direct runtime path."
---

# Phase 3: Safe Surface, C ABI Shim & Optional Families Verification Report

**Phase Goal:** Expose the safe Rust facade, optional C shim, and gated optional families once the runtime is stable.
**Verified:** 2026-03-28T01:15:35Z
**Status:** gaps_found
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | `cintx-rs` and `cintx-capi` are active workspace members. | VERIFIED | `Cargo.toml` workspace/default-members include both crates; `cargo metadata --no-deps` lists both in `workspace_members`. |
| 2 | Cargo feature gates exist for `with-f12`, `with-4c1e`, and `unstable-source-api` compile-time controls. | FAILED | Gates exist, but must-have artifact depth contract is not met (`crates/cintx-rs/Cargo.toml` and `crates/cintx-capi/Cargo.toml` are below declared `min_lines`). |
| 3 | Stable `cintx-rs` exports remain default; unstable APIs are gated behind `unstable-source-api`. | VERIFIED | `crates/cintx-rs/src/lib.rs:19` and `crates/cintx-rs/src/api.rs:460` gate unstable exports/module with `cfg(feature = "unstable-source-api")`. |
| 4 | `cintx-capi` remains stable-only in Phase 3. | VERIFIED | `crates/cintx-capi/src/lib.rs:18` sets `CAPI_EXPOSES_UNSTABLE_SOURCE_API = false`; exports are limited to `errors` and `shim`. |
| 5 | `with-f12` only allows validated sph F12/STG/YP envelope and rejects unsupported representations explicitly. | VERIFIED | `crates/cintx-compat/src/raw.rs:485` (`validate_f12_envelope`) and tests around `raw.rs:1224`/`1246` enforce sph-only envelope and explicit `UnsupportedApi`. |
| 6 | `with-4c1e` only allows Validated4C1E envelope and rejects out-of-envelope inputs explicitly. | VERIFIED | `crates/cintx-compat/src/raw.rs:501` (`validate_4c1e_envelope`) + `crates/cintx-cubecl/src/executor.rs:39` (`ensure_validated_4c1e`) enforce bounds with `"outside Validated4C1E"` errors. |
| 7 | Source-only symbols fail explicitly when `unstable-source-api` is disabled. | VERIFIED | `crates/cintx-compat/src/raw.rs:647` rejects source-only symbols unless feature enabled; resolver marks source-only entries (`crates/cintx-ops/src/resolver.rs:129`). |
| 8 | Manifest/resolver metadata is the source of truth for optional/unstable family availability. | VERIFIED | Raw resolution uses resolver descriptor/profile checks (`crates/cintx-compat/src/raw.rs:623-651`) backed by generated manifest tables (`crates/cintx-ops/src/generated/api_manifest.rs`). |
| 9 | Safe Rust callers can query workspace separately before evaluation. | VERIFIED | `SessionRequest::query_workspace()` in `crates/cintx-rs/src/api.rs:62` returns `SessionQuery` with structured `WorkspacePlan`. |
| 10 | Safe evaluation returns owned typed outputs from typed session/query contract. | VERIFIED | `SessionQuery::evaluate()` in `crates/cintx-rs/src/api.rs:93` returns `TypedEvaluationOutput` with owned `Vec<f64>` and stats. |
| 11 | Optional/unstable `UnsupportedApi` decisions are wired from compat raw path into safe facade mapping. | FAILED | Declared key link is not wired: `cintx-rs` has no compat dependency and `api.rs` has no `query_workspace_raw`/`eval_raw` usage. |
| 12 | Query/evaluate contract drift is explicitly rejected with fail-closed behavior preserved. | VERIFIED | `WorkspaceExecutionToken::ensure_matches` rejects drift (`crates/cintx-rs/src/api.rs:184-211`); ownership checks enforce `BackendStagingOnly`/`CompatFinalWrite` (`api.rs:423-434`). |
| 13 | C callers get `0` success and nonzero typed statuses on failure. | VERIFIED | `CintxStatus` enum with `Success = 0` (`crates/cintx-capi/src/errors.rs:10`) and shim wrapper status returns in `crates/cintx-capi/src/shim.rs:203-221`. |
| 14 | C error details are thread-local and copy-out accessible. | VERIFIED | TLS `LAST_ERROR` (`crates/cintx-capi/src/errors.rs:84-86`) and copy-out APIs (`errors.rs:206-229`) are implemented and tested. |
| 15 | C shim wrappers are thin compat-style wrappers with fail-closed behavior. | VERIFIED | Shim calls `query_workspace_raw`/`eval_raw` directly (`crates/cintx-capi/src/shim.rs:300`, `372`) and only writes outputs on success. |
| 16 | C ABI remains stable-only (no unstable source-only C symbols). | VERIFIED | `crates/cintx-capi/src/lib.rs` exports stable modules only; no unstable-source C exports present. |

**Score:** 14/16 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `Cargo.toml` | Workspace/member activation and phase feature declarations | VERIFIED | 62 lines (>=40); workspace + feature wiring present and used by metadata. |
| `crates/cintx-rs/Cargo.toml` | Facade dependency/feature wiring | STUB | 19 lines (<20 min_lines). Exists and wired, but below declared depth threshold. |
| `crates/cintx-capi/Cargo.toml` | C ABI dependency/feature wiring (stable-only) | STUB | 19 lines (<20 min_lines). Exists and wired, but below declared depth threshold. |
| `crates/cintx-rs/src/api.rs` | Typed safe facade and query/evaluate flow | VERIFIED | 582 lines; runtime query/evaluate wiring, contract checks, and typed output implemented. |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | Optional/unstable profile inventory | VERIFIED | 2955 lines; includes `with-f12`/`with-4c1e`/unstable entries and profile lists. |
| `crates/cintx-ops/src/resolver.rs` | Profile-aware resolver and source-only helpers | VERIFIED | 473 lines; exposes profile/source checks consumed by raw layer. |
| `crates/cintx-compat/src/raw.rs` | Envelope validation + profile-gated raw rejection paths | VERIFIED | 1361 lines; contains F12 and Validated4C1E checks and explicit `UnsupportedApi` paths. |
| `crates/cintx-cubecl/src/executor.rs` | 4c1e feature-aware backend acceptance/rejection | VERIFIED | 437 lines; includes validated envelope enforcement and fail-closed ownership checks. |
| `crates/cintx-rs/src/error.rs` | Stable facade error enum + core mapping | VERIFIED | 126 lines; maps core unsupported/layout/memory/validation categories. |
| `crates/cintx-rs/src/builder.rs` | Typed builder APIs for safe session creation | STUB | 56 lines (<120 min_lines). Functional but below declared artifact depth. |
| `crates/cintx-rs/src/prelude.rs` | Curated stable safe-API re-exports | STUB | 18 lines (<30 min_lines). Functional but below declared artifact depth. |
| `crates/cintx-capi/src/errors.rs` | Status taxonomy + TLS error storage + copy-out APIs | VERIFIED | 366 lines; full status/TLS/copy-out implementation with tests. |
| `crates/cintx-capi/src/shim.rs` | Thin `extern "C"` wrappers + status mapping | VERIFIED | 581 lines; panic boundary, raw delegation, and TLS status behavior implemented. |
| `crates/cintx-capi/src/lib.rs` | Stable C ABI export boundary | VERIFIED | 32 lines (>=30); exports `errors`/`shim` and stable-only constants/tests. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `Cargo.toml` | `crates/cintx-rs/Cargo.toml` | Workspace/feature wiring for `cintx-rs` | WIRED | Root and crate-local features align (`with-f12`, `with-4c1e`, `unstable-source-api`). |
| `Cargo.toml` | `crates/cintx-capi/Cargo.toml` | Workspace/feature wiring for stable-only C ABI | WIRED | Root `capi` and optional-family feature forwarding present. |
| `crates/cintx-rs/src/lib.rs` | `crates/cintx-rs/src/api.rs` | Unstable exports compile-gated | WIRED | `cfg(feature = "unstable-source-api")` present in both files. |
| `compiled_manifest.lock.json` | `api_manifest.rs` | Generated optional/unstable profile tables | WIRED | Both artifacts contain optional profiles and unstable/source-only rows. |
| `resolver.rs` | `raw.rs` | Resolver metadata enforced in compat path | WIRED | `raw.rs` imports resolver types and enforces profile/source-only checks. |
| `raw.rs` | `executor.rs` | Validated4C1E fail-closed backend enforcement | WIRED | Raw envelope checks and CubeCL validated envelope checks share same rejection contract. |
| `crates/cintx-rs/src/api.rs` | `crates/cintx-runtime/src/planner.rs` | Safe query/evaluate uses runtime planner | WIRED | Uses `runtime_query_workspace`, `ExecutionPlan::new`, and `runtime_evaluate`. |
| `crates/cintx-rs/src/error.rs` | `crates/cintx-core/src/error.rs` | Stable mapping of core error categories | WIRED | `From<cintxRsError>` maps Unsupported/Layout/Memory/Validation classes. |
| `crates/cintx-rs/src/api.rs` | `crates/cintx-compat/src/raw.rs` | UnsupportedApi propagation from compat raw | NOT_WIRED | No compat dependency or raw API calls in `cintx-rs`; link declared in plan is absent. |
| `crates/cintx-capi/src/shim.rs` | `crates/cintx-compat/src/raw.rs` | Thin raw wrapper calls | WIRED | Direct calls to `query_workspace_raw` and `eval_raw`. |
| `crates/cintx-capi/src/shim.rs` | `crates/cintx-capi/src/errors.rs` | Status + TLS last-error reporting | WIRED | Uses `set_last_error`/`clear_last_error` and returns status codes. |
| `crates/cintx-capi/src/lib.rs` | `crates/cintx-capi/src/shim.rs` | Stable-only export surface | WIRED | `pub mod errors; pub mod shim;` and stable re-exports present. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `EXEC-01` | `03-PLAN.md` | Rust caller can query workspace separately from evaluation via safe API. | SATISFIED | `SessionRequest::query_workspace` and `SessionQuery::evaluate` split in `crates/cintx-rs/src/api.rs`; `cargo test -p cintx-rs --lib` passed. |
| `COMP-04` | `04-PLAN.md` | Optional C ABI shim returns status codes and TLS last-error details. | SATISFIED | `CintxStatus`, TLS `LAST_ERROR`, copy-out APIs, and shim wrappers in `crates/cintx-capi/src/errors.rs`/`shim.rs`; `cargo test -p cintx-capi --lib` passed. |
| `OPT-01` | `02-PLAN.md` | `with-f12` enables sph-only F12/STG/YP and rejects unsupported reps explicitly. | SATISFIED | `validate_f12_envelope` in `crates/cintx-compat/src/raw.rs`; feature tests passed under `--features with-f12`. |
| `OPT-02` | `02-PLAN.md` | `with-4c1e` is constrained to validated bug envelope and rejects out-of-envelope calls explicitly. | SATISFIED | `validate_4c1e_envelope` (`raw.rs`) and `ensure_validated_4c1e` (`executor.rs`); feature tests passed under `--features with-4c1e`. |
| `OPT-03` | `01-PLAN.md`, `02-PLAN.md` | Source-only APIs remain behind `unstable-source-api` without GA surface drift. | SATISFIED | `cfg(feature = "unstable-source-api")` in `cintx-rs`, source-only resolver metadata, and explicit raw rejection when feature disabled. |

Phase-3 orphaned requirements check (`REQUIREMENTS.md` traceability table): none. All `Phase 3` IDs (`COMP-04`, `EXEC-01`, `OPT-01`, `OPT-02`, `OPT-03`) are claimed by plan frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `crates/cintx-rs/src/api.rs` | 123, 307 | Facade output is synthesized via `fill_staging_values` instead of reading runtime-produced output buffers | WARNING | Safe-facade tensor values can drift from runtime/compat output semantics if backend output behavior changes. |

No blocker TODO/FIXME/placeholder stubs were found in phase-owned implementation files.

### Human Verification Required

None for this automated pass. Remaining blockers are code-level contract gaps captured above.

### Gaps Summary

Phase 3 is close, but it does not fully satisfy the declared must-have contract:

1. Four declared artifacts miss their own `min_lines` substance thresholds (`crates/cintx-rs/Cargo.toml`, `crates/cintx-capi/Cargo.toml`, `crates/cintx-rs/src/builder.rs`, `crates/cintx-rs/src/prelude.rs`).
2. One declared key link is not actually wired (`cintx-rs/api.rs -> cintx-compat/raw.rs`), so the plan’s stated propagation path for optional/unstable `UnsupportedApi` decisions is unmet.

Until these are resolved (or explicitly replanned/accepted), the phase cannot be marked fully achieved under must-have verification criteria.

---

_Verified: 2026-03-28T01:15:35Z_
_Verifier: Codex (gsd-verifier)_
