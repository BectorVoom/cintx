---
phase: manifest-planner-foundation
plan: 02
subsystem: runtime
tags: [rust, runtime, planner, workspace, tracing]
requires:
  - phase: manifest-planner-foundation
    provides: canonical manifest metadata and resolver descriptors from Plan 01
provides:
  - `cintx-runtime` workspace crate with execution options, validation, planner, and workspace modules
  - deterministic `query_workspace()` and `evaluate()` contract backed by stored chunk layouts
  - typed runtime validation and memory-limit errors routed through `cintxRsError`
affects:
  - execution-and-compatibility-stabilization
  - manifest-planner-foundation
tech-stack:
  added: []
  patterns:
    - query-then-evaluate workspace contracts
    - central fallible workspace allocation
    - manifest-driven runtime validation
key-files:
  created:
    - crates/cintx-runtime/Cargo.toml
    - crates/cintx-runtime/src/lib.rs
    - crates/cintx-runtime/src/options.rs
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-runtime/src/workspace.rs
  modified:
    - Cargo.toml
    - Cargo.lock
    - crates/cintx-core/src/error.rs
    - crates/cintx-core/src/lib.rs
key-decisions:
  - "Persist exact chunk layouts inside `WorkspaceQuery` and reject evaluate-time planning drift instead of silently replanning."
  - "Clamp `chunk_size_override` to the maximum work units that fit within the effective memory limit."
  - "Promote invalid shell atom indices to a dedicated typed runtime error."
patterns-established:
  - "Workspace queries are authoritative contracts reused by execution."
  - "Runtime validation failures surface through shared `cintxRsError` variants instead of planner-detail strings."
requirements-completed: [BASE-01, BASE-03]
duration: 9 min
completed: 2026-03-21T07:28:53Z
---

# Phase 01 Plan 02: Runtime Planner Contract Summary

**`cintx-runtime` now exposes a manifest-driven workspace/query/evaluate contract with typed validation failures, deterministic chunk planning, and tracing-backed planner diagnostics.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-21T07:19:32Z
- **Completed:** 2026-03-21T07:28:53Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Added the `cintx-runtime` workspace crate and exposed `ExecutionOptions`, `ExecutionPlan`, `query_workspace()`, `evaluate()`, validation helpers, and workspace allocator/chunk planner types.
- Made `WorkspaceQuery` carry the exact chunk contract so `evaluate()` reuses queried chunk layouts and rejects planning-option drift instead of silently replanning.
- Added typed `InvalidShellAtomIndex` handling plus regression tests for planner contract drift, chunk override clamping, and validation failures under memory limits.

## Task Commits

This resumed plan finalized as a single buildable code commit because the paused worktree already interleaved Task 1 and Task 2 runtime scaffolding:

1. **Task 1 + Task 2 finalization** - `0523f36` (fix)

**Plan metadata:** pending final summary/state/verification commit

## Files Created/Modified
- `Cargo.toml` - adds `crates/cintx-runtime` to the workspace and default members.
- `Cargo.lock` - records the new runtime crate in the workspace lockfile.
- `crates/cintx-core/src/error.rs` - adds typed runtime errors, including `InvalidShellAtomIndex`.
- `crates/cintx-runtime/src/options.rs` - defines execution memory-limit and tracing options.
- `crates/cintx-runtime/src/planner.rs` - implements manifest-aware `query_workspace()` / `evaluate()` plus execution stats.
- `crates/cintx-runtime/src/validator.rs` - validates shell tuples, representations, dims, and atom indices.
- `crates/cintx-runtime/src/workspace.rs` - implements fallible workspace allocation and deterministic chunk planning under memory caps.

## Decisions Made
- Treated the `query_workspace()` result as the canonical execution contract by storing chunk layouts inside `WorkspaceQuery` and refusing mismatched evaluate-time planning options.
- Kept chunk-size overrides as hints, not mandates, so runtime chunking still obeys the effective memory cap.
- Introduced `InvalidShellAtomIndex` rather than folding bad shell references into `ChunkPlanFailed`, keeping validation errors explicit for downstream callers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Resumed worktree required a single final code commit**
- **Found during:** Task 2 (plan resumption)
- **Issue:** The paused Plan 02 worktree already mixed Task 1 and Task 2 runtime scaffolding across the new `cintx-runtime` crate, so splitting commits non-interactively would have left broken intermediate states.
- **Fix:** Re-verified Task 1 acceptance criteria, finished the reviewer-blocking fixes, and committed the complete runtime contract as one buildable code snapshot.
- **Files modified:** `Cargo.toml`, `Cargo.lock`, `crates/cintx-core/src/{error,lib}.rs`, `crates/cintx-runtime/src/*`
- **Verification:** `cargo test -p cintx-core --lib`, `cargo test -p cintx-runtime --lib`, `cargo test --workspace`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo fmt --all -- --check`
- **Committed in:** `0523f36`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The deviation preserved a buildable history for resumed work without changing Plan 02 scope.

## Issues Encountered
- The sandbox blocks `.git/index.lock` creation, so the code commit had to be rerun with elevated git permissions.
- The built-in GSD artifact verifier could not parse this plan's nested `must_haves` structure, so phase verification used direct file/test evidence instead of the helper command.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 01 now includes the typed planner/workspace foundation that Phase 2 can wire into the compat and execution layers.
- No known blockers remain for moving into Phase 2 planning/discussion.

---
*Phase: manifest-planner-foundation*
*Completed: 2026-03-21*
