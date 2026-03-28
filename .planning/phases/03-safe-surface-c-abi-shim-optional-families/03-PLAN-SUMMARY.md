---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 03
subsystem: api
tags: [rust, safe-api, runtime, facade, workspace-contract]
requires:
  - phase: 03-safe-surface-c-abi-shim-optional-families-01
    provides: Stable vs unstable namespace boundaries and feature-gate policy.
  - phase: 03-safe-surface-c-abi-shim-optional-families-02
    provides: Optional-family/runtime envelope gating and resolver profile constraints.
provides:
  - Safe typed session/query contracts with explicit execution-token metadata accessors.
  - Runtime-backed safe evaluate path that returns owned tensor output captured from backend staging.
  - Explicit query/evaluate contract-drift rejection before safe output publication.
affects: [cintx-rs, phase-03-plan-04, phase-04]
tech-stack:
  added: []
  patterns:
    - Safe evaluate uses runtime planner/evaluator and captures staged output through a backend wrapper.
    - Workspace execution token remains stable and inspectable through facade accessors.
key-files:
  created: []
  modified:
    - crates/cintx-rs/src/api.rs
key-decisions:
  - "Expose accessor methods on `WorkspaceExecutionToken` to keep contract metadata stable without exposing private fields."
  - "Capture owned safe output from runtime backend staging via `RecordingExecutor` instead of rebuilding post-evaluate buffers."
patterns-established:
  - "Facade contract pattern: query establishes token, evaluate verifies token + runtime planning contract, then publishes owned output."
  - "Fail-closed output pattern: owned result is returned only after successful runtime execution and output-shape contract validation."
requirements-completed: [EXEC-01]
duration: 9 min
completed: 2026-03-28
---

# Phase 03 Plan 03: Safe Rust Facade Summary

**Implemented a typed safe query/evaluate facade that preserves runtime contracts and returns owned output from the executed backend staging path.**

## Performance

- **Duration:** 9 min
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added stable accessor methods on `WorkspaceExecutionToken` and tightened safe-query contract assertions in tests.
- Wired `SessionQuery::evaluate()` to capture owned output from runtime execution via `RecordingExecutor<CubeClExecutor>`.
- Added owned-output contract drift validation (`staging_elements` vs captured output len) before returning `TypedEvaluationOutput`.

## Task Commits

1. **Task 1: Implement typed safe-session contracts and stable facade error/output types:** `2892d56` (`feat`)
2. **Task 2: Wire safe `query_workspace()` and `evaluate()` to runtime with fail-closed owned-output behavior:** `a7b9b9a` (`feat`)

## Files Created/Modified

- `crates/cintx-rs/src/api.rs` - session token accessor surface, runtime staging-output capture wrapper, owned-output contract checks, and test assertions.

## Deviations from Plan

None - plan executed exactly as written.

## Authentication Gates

None.

## Issues Encountered

- Concurrent edits from other agents were present in unrelated files; this execution intentionally staged and committed only plan-owned files.

## User Setup Required

None - no external services or credentials required.

## Next Phase Readiness

- Safe facade query/evaluate contract is typed, runtime-backed, and fail-closed for owned-output publication.
- Plan 04 C ABI work can consume the same runtime/facade contract behavior.

## Known Stubs

None.

## Self-Check: PASSED

- FOUND: `.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-PLAN-SUMMARY.md`
- FOUND: `2892d56`
- FOUND: `a7b9b9a`
