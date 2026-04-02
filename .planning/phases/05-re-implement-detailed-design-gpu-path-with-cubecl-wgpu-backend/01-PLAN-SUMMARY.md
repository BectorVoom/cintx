---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 01
subsystem: runtime
tags: [rust, backend-contract, cubecl, wgpu, planner, workspace, options]

# Dependency graph
requires:
  - phase: 04-verification-release-automation
    provides: stable runtime planner/evaluate contract and WorkspaceQuery type
provides:
  - BackendKind enum with Wgpu (primary) and Cpu (test) variants
  - BackendIntent struct carrying backend selection and adapter selector
  - BackendCapabilityToken struct with adapter name, api, and capability fingerprint
  - Extended ExecutionOptions with backend_intent and backend_capability_token fields
  - Extended WorkspaceQuery with backend contract fields (D-03 + D-08)
  - Updated planning_matches() enforcing four-field contract comparison
  - query_workspace() populates backend contract fields with tracing spans
  - evaluate() fails closed on backend-contract drift with explicit detail message
  - 18 passing unit tests including 4 new backend-contract regression tests
affects:
  - 05-02 (backend execution rewiring consumes stable backend-intent contract)
  - cintx-cubecl executor (backend_intent/capability_token will drive adapter selection)
  - cintx-compat raw path (query/evaluate drift detection now covers backend policy)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - BackendKind/BackendIntent/BackendCapabilityToken typed backend-contract propagation through options -> query -> evaluate
    - Four-field planning_matches contract comparison (memory, chunk_size, backend_intent, capability_token)
    - Fail-closed evaluate() on backend-contract drift with explicit drift-language error detail

key-files:
  created: []
  modified:
    - crates/cintx-runtime/src/options.rs
    - crates/cintx-runtime/src/workspace.rs
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-runtime/src/lib.rs
    - crates/cintx-runtime/src/scheduler.rs

key-decisions:
  - "BackendIntent defaults to BackendKind::Wgpu with selector 'auto' per D-03; Cpu variant kept for oracle/test use"
  - "BackendCapabilityToken fingerprint default is 0 (empty/uninitialized); later plans fill with real adapter context"
  - "planning_matches() compares all four fields atomically so any backend policy drift fails evaluate"
  - "Drift error detail now names all four contract fields to aid debugging without leaking internals"

patterns-established:
  - "Backend contract propagates from ExecutionOptions -> WorkspaceQuery -> evaluate() as typed fields (not hidden executor policy)"
  - "query_workspace span carries backend/selector/fingerprint fields for D-08 reproducibility"

requirements-completed:
  - EXEC-02
  - EXEC-03
  - COMP-05

# Metrics
duration: 3min
completed: 2026-04-02
---

# Phase 05 Plan 01: Runtime Backend Intent/Capability Contract Summary

**Typed BackendKind/BackendIntent/BackendCapabilityToken contract added to ExecutionOptions and WorkspaceQuery with four-field planning_matches drift detection and fail-closed evaluate enforcement (D-03, D-08)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-02T07:29:27Z
- **Completed:** 2026-04-02T07:33:12Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Defined `BackendKind` (Wgpu/Cpu), `BackendIntent`, and `BackendCapabilityToken` in `options.rs` with correct defaults (Wgpu, selector="auto")
- Extended `WorkspaceQuery` with backend contract fields and updated `planning_matches()` to compare all four contract fields
- Updated `query_workspace()` to copy backend fields from opts into query and emit tracing spans
- Updated `evaluate()` drift error detail to explicitly name backend-contract drift
- Added 4 new regression tests covering backend intent drift, capability token drift, and query field persistence; all 18 cintx-runtime tests pass

## Task Commits

Each task was committed atomically:

1. **Test: planning_matches_checks_backend_contract (RED)** - `c2cce94` (test)
2. **Task 1: Add runtime backend intent/capability contract types** - `0276f25` (feat)
3. **Task 2: Enforce backend contract drift checks in planner query/evaluate** - `cceb170` (feat)

## Files Created/Modified

- `crates/cintx-runtime/src/options.rs` - Added BackendKind, BackendIntent, BackendCapabilityToken; extended ExecutionOptions
- `crates/cintx-runtime/src/workspace.rs` - Extended WorkspaceQuery with backend fields; updated planning_matches(); added test
- `crates/cintx-runtime/src/planner.rs` - Updated query_workspace tracing, drift error detail, added 3 regression tests
- `crates/cintx-runtime/src/lib.rs` - Re-exported BackendCapabilityToken, BackendIntent, BackendKind
- `crates/cintx-runtime/src/scheduler.rs` - Updated WorkspaceQuery literal in test to include new fields (Rule 1 auto-fix)

## Decisions Made

- BackendIntent defaults to `BackendKind::Wgpu` with selector `"auto"` per D-03; `Cpu` variant kept for oracle/test use only
- `BackendCapabilityToken` fingerprint defaults to 0 (empty/uninitialized); later plans will fill with real adapter info when wiring real wgpu adapter selection
- `planning_matches()` compares all four fields atomically - any backend policy drift fails evaluate closed
- Drift error detail in `evaluate()` now explicitly names all four contract fields to improve debuggability

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed scheduler test missing new WorkspaceQuery fields**
- **Found during:** Task 1 (GREEN phase compilation)
- **Issue:** `scheduler.rs` test constructed `WorkspaceQuery` with struct literal; adding two new fields caused a compile error
- **Fix:** Added `backend_intent: BackendIntent::default()` and `backend_capability_token: BackendCapabilityToken::default()` to the test literal
- **Files modified:** `crates/cintx-runtime/src/scheduler.rs`
- **Verification:** All 18 cintx-runtime tests pass after fix
- **Committed in:** `0276f25` (Task 1 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - blocking compile error)
**Impact on plan:** Necessary fix to keep existing tests compiling. No scope creep.

## Issues Encountered

None - both TDD phases (RED->GREEN) worked as expected. Task 2 tests passed immediately because Task 1's `planning_matches()` four-field enforcement already satisfied the drift-rejection contract.

## Next Phase Readiness

- Backend contract is stable; Phase 5 Plan 02 can consume `BackendIntent` and `BackendCapabilityToken` for real wgpu adapter selection
- `BackendCapabilityToken.capability_fingerprint` currently defaults to 0; wgpu adapter query plan will populate with real device capability hash
- All 18 cintx-runtime tests pass with no regressions

---
*Phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend*
*Completed: 2026-04-02*

## Self-Check: PASSED

- FOUND: 01-PLAN-SUMMARY.md
- FOUND: options.rs
- FOUND: workspace.rs
- FOUND: planner.rs
- FOUND: c2cce94 (test RED commit)
- FOUND: 0276f25 (Task 1 feat commit)
- FOUND: cceb170 (Task 2 feat commit)
