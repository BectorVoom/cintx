---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 01
subsystem: api
tags: [rust, cargo, feature-gates, capi, workspace]
requires:
  - phase: 02-execution-compatibility-stabilization
    provides: Runtime query/evaluate split, fail-closed compat execution, and manifest-driven operator metadata.
provides:
  - Activated `cintx-rs` and `cintx-capi` as workspace/default-members build targets.
  - Added explicit `with-f12`, `with-4c1e`, `unstable-source-api`, and `capi` feature topology.
  - Established stable-vs-unstable safe facade namespace scaffolding for Phase 3.
affects: [03-safe-surface-c-abi-shim-optional-families, 04-verification-release-automation]
tech-stack:
  added: []
  patterns:
    - Hyphenated workspace feature gates map explicitly to upstream libcint underscore features.
    - Source-only APIs are isolated behind `unstable-source-api` and excluded from stable exports.
key-files:
  created: []
  modified:
    - Cargo.toml
    - Cargo.lock
    - crates/cintx-rs/Cargo.toml
    - crates/cintx-capi/Cargo.toml
    - crates/cintx-rs/src/lib.rs
    - crates/cintx-rs/src/api.rs
    - crates/cintx-rs/src/builder.rs
    - crates/cintx-rs/src/prelude.rs
    - crates/cintx-capi/src/lib.rs
key-decisions:
  - "Top-level optional-family gates now explicitly forward to libcint `with_f12` and `with_4c1e` features."
  - "Unstable source APIs are exposed only through a cfg-gated `unstable` namespace in `cintx-rs`."
patterns-established:
  - "Workspace feature forwarding from root -> crate-local features -> upstream dependency features."
  - "Safe API scaffolding uses a typed `SessionRequest` -> `SessionQuery` boundary before behavior implementation."
requirements-completed: [OPT-03]
duration: 3 min
completed: 2026-03-28
---

# Phase 03 Plan 01: Workspace and Namespace Scaffolding Summary

**Activated Phase 3 feature topology and introduced a cfg-gated stable-vs-unstable facade scaffold across `cintx-rs` and `cintx-capi`.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-27T23:59:37Z
- **Completed:** 2026-03-28T00:02:45Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Added `crates/cintx-rs` and `crates/cintx-capi` to workspace members and default-members for normal Phase 3 verification.
- Added explicit root feature gates for `with-f12`, `with-4c1e`, `unstable-source-api`, and `capi` with direct upstream feature mapping (`with_f12`, `with_4c1e`).
- Replaced safe-surface stubs with typed request/query scaffolding and explicit unstable namespace gating; kept C ABI exports stable-only.

## Task Commits

Each task was committed atomically:

1. **Task 1: Activate Phase 3 crates and feature topology in workspace manifests** - `e05d26d` (feat)
2. **Task 2: Define stable and unstable namespace scaffolds for facade and C ABI boundaries** - `4751542` (feat)

## Files Created/Modified
- `Cargo.toml` - Activated Phase 3 workspace members and top-level feature forwarding.
- `Cargo.lock` - Regenerated lock entries for newly active workspace crates.
- `crates/cintx-rs/Cargo.toml` - Added facade dependencies and feature declarations (`with-f12`, `with-4c1e`, `unstable-source-api`).
- `crates/cintx-capi/Cargo.toml` - Added C ABI dependencies and stable-only feature declarations (`capi`, `with-f12`, `with-4c1e`).
- `crates/cintx-rs/src/lib.rs` - Added stable exports and cfg-gated unstable re-export.
- `crates/cintx-rs/src/api.rs` - Added typed `SessionRequest`/`SessionQuery` scaffold, explicit unsupported fallback, and unstable namespace module.
- `crates/cintx-rs/src/builder.rs` - Added typed `SessionBuilder` scaffolding for safe API construction.
- `crates/cintx-rs/src/prelude.rs` - Added stable prelude exports and cfg-gated unstable prelude export.
- `crates/cintx-capi/src/lib.rs` - Kept stable-only module exports and added explicit boundary marker.

## Decisions Made
- Kept hyphenated feature names at workspace/crate boundaries and mapped them directly to libcint underscore features at the root manifest to avoid profile drift.
- Introduced an explicit `unsupported_unstable_request()` helper so disabled source-only paths map to `UnsupportedApi` semantics in stable builds.
- Kept `cintx-capi` limited to stable export modules in this plan; no unstable-source C exports were introduced.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

- `crates/cintx-rs/src/api.rs:63` - `SessionQuery::evaluate()` intentionally returns `UnsupportedApi` scaffold text. This is intentional for Plan 01 boundary scaffolding; behavioral evaluation wiring is deferred to later Phase 3 plans.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 plan 02 can now implement manifest-driven optional-family/unstable-source runtime gating on top of explicit compile-time controls.
- No blockers detected for continuing this phase.

## Self-Check: PASSED

- FOUND: `.planning/phases/03-safe-surface-c-abi-shim-optional-families/01-PLAN-SUMMARY.md`
- FOUND: `e05d26d`
- FOUND: `4751542`

---
*Phase: 03-safe-surface-c-abi-shim-optional-families*
*Completed: 2026-03-28*
