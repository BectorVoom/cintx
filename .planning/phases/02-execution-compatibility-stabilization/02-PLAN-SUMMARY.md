---
phase: execution-compatibility-stabilization
plan: 02
subsystem: compat
tags: [rust, cargo-workspace, compat, cubecl, oracle]
requires:
  - phase: manifest-planner-foundation
    provides: typed core/ops/runtime crates and the shared runtime query/evaluate contract
provides:
  - workspace activation for `cintx-compat`, `cintx-cubecl`, and `cintx-oracle`
  - explicit `cintx-compat -> cintx-cubecl` and `cintx-oracle -> cintx-compat` dependency edges
  - crate-root smoke tests proving the newly activated phase-2 crates build in normal `cargo test` flows
affects:
  - execution-compatibility-stabilization
  - compatibility
  - execution
tech-stack:
  added: [cubecl 0.9.0, anyhow 1.0.102, serde_json 1.0.145, bindgen 0.71.1, cc 1.2.15]
  patterns:
    - workspace activation before implementation
    - direct crate-edge wiring for compat/backend/oracle call paths
    - crate-root smoke tests for cross-crate dependency edges
key-files:
  created:
    - crates/cintx-compat/Cargo.toml
    - crates/cintx-cubecl/Cargo.toml
    - crates/cintx-oracle/Cargo.toml
    - crates/cintx-compat/src/lib.rs
    - crates/cintx-cubecl/src/lib.rs
    - crates/cintx-oracle/src/lib.rs
  modified:
    - Cargo.toml
    - Cargo.lock
key-decisions:
  - "Keep Phase 2 workspace scope limited to core/ops/runtime/compat/cubecl/oracle; leave `cintx-rs` and `cintx-capi` out of this phase."
  - "Require explicit direct edges for `cintx-compat -> cintx-cubecl` and `cintx-oracle -> cintx-compat` to lock intended call routing."
  - "Pin `cintx-cubecl` kernels export to `kernels/mod.rs` to resolve module-path ambiguity during workspace activation."
patterns-established:
  - "Compat and oracle crates must prove dependency edges through compile-time smoke tests."
  - "Workspace membership and dependency floors are verified via `cargo metadata --no-deps` plus crate-scoped `cargo test -p ... --lib`."
requirements-completed: [EXEC-02, EXEC-03]
duration: 18 min
completed: 2026-03-21T10:17:35Z
---

# Phase 02 Plan 02: Workspace Activation Summary

**Activated `cintx-compat`, `cintx-cubecl`, and `cintx-oracle` as first-class workspace crates with explicit compat/backend/oracle dependency routing and crate-level smoke-test coverage.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-21T09:59:35Z
- **Completed:** 2026-03-21T10:17:35Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Added `cintx-compat`, `cintx-cubecl`, and `cintx-oracle` to workspace members and default-members so Phase 2 crates are built in standard workflows.
- Wired explicit manifest dependencies for the required `compat -> cubecl` and `oracle -> compat` crate edges and regenerated `Cargo.lock`.
- Replaced crate-root stubs with explicit module exports plus narrow smoke tests that assert cross-crate module references compile.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the Phase 2 crates to the workspace and wire the crate-level call path** - `57dac5f` (feat)
2. **Task 2: Make the newly activated crate roots pass library smoke tests** - `f3343a4` (feat)

**Plan metadata:** pending final docs commit

## Files Created/Modified
- `Cargo.toml` - expands workspace members/default-members to include compat, cubecl, and oracle crates.
- `Cargo.lock` - records the updated workspace dependency graph.
- `crates/cintx-compat/Cargo.toml` - adds explicit runtime/backend/core/ops dependencies plus `smallvec` and `tracing`.
- `crates/cintx-cubecl/Cargo.toml` - adds explicit runtime/core/ops dependencies plus `cubecl`, `smallvec`, and `tracing`.
- `crates/cintx-oracle/Cargo.toml` - adds explicit compat/core/ops dependencies and oracle build dependencies (`bindgen`, `cc`).
- `crates/cintx-compat/src/lib.rs` - preserves Phase 2 module exports and adds compat-to-cubecl smoke coverage.
- `crates/cintx-cubecl/src/lib.rs` - restores `transform` export, resolves kernels module path ambiguity, and adds crate export smoke coverage.
- `crates/cintx-oracle/src/lib.rs` - preserves module exports and adds oracle-to-compat smoke coverage.

## Decisions Made
- Kept Phase 2 scope constrained to workspace/dependency activation and smoke verification, deferring safe facade/C ABI scope per roadmap.
- Enforced direct crate-edge dependencies instead of implicit transitive wiring to keep future compat/oracle integration paths explicit.
- Treated the `kernels.rs` vs `kernels/mod.rs` conflict as a blocking activation issue and fixed it inline so the plan remains build-verifiable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Resolved ambiguous CubeCL kernels module layout**
- **Found during:** Task 2 (crate smoke test verification)
- **Issue:** `cintx-cubecl` had both `src/kernels.rs` and `src/kernels/mod.rs`, causing Rust module resolution error `E0761` when running `cargo test -p cintx-compat --lib`.
- **Fix:** Updated `crates/cintx-cubecl/src/lib.rs` to bind `kernels` explicitly to `#[path = "kernels/mod.rs"]`.
- **Files modified:** `crates/cintx-cubecl/src/lib.rs`
- **Verification:** `cargo test -p cintx-compat --lib && cargo test -p cintx-cubecl --lib && cargo test -p cintx-oracle --lib`
- **Committed in:** `f3343a4` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for build correctness; no scope creep beyond the plan's verification goals.

## Issues Encountered
- Initial Task 2 verification failed on a pre-existing CubeCL module layout ambiguity; fixed inline under Rule 3 and re-verified successfully.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 2 crates now participate in normal workspace metadata and test flows.
- Dependency edges needed for upcoming runtime/backend/compat implementation work are explicit and verified.

## Self-Check: PASSED
- FOUND: `.planning/phases/02-execution-compatibility-stabilization/02-PLAN-SUMMARY.md`
- FOUND: `57dac5f`
- FOUND: `f3343a4`

---
*Phase: execution-compatibility-stabilization*
*Completed: 2026-03-21*
