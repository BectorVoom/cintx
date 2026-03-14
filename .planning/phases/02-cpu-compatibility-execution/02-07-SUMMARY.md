---
phase: 02-cpu-compatibility-execution
plan: "07"
subsystem: runtime
tags: [raw-compat, cpu-routing, diagnostics, workspace-query]
requires:
  - phase: 02-cpu-compatibility-execution
    provides: RAW-01 contract validation boundary (02-02) and CPU route dispatch map (02-06)
provides:
  - Raw query module with null-equivalent out/cache sentinel normalization
  - Raw evaluate module that enforces query->execute buffer/cache contracts before CPU dispatch
  - RAW-02 integration tests covering sentinel query flow, successful handoff, and mismatch diagnostics
affects: [phase-2-raw-runtime, raw-api, memory-threading]
tech-stack:
  added: []
  patterns: [sentinel normalization, query-to-execute contract gating, diagnostics-preserving failure reporting]
key-files:
  created:
    - src/runtime/raw/query.rs
    - src/runtime/raw/evaluate.rs
    - tests/phase2_raw_query_execute.rs
  modified:
    - src/api/raw.rs
    - src/runtime/raw/mod.rs
    - src/runtime/mod.rs
    - src/lib.rs
    - src/diagnostics/report.rs
key-decisions:
  - "Kept legacy raw compatibility query entrypoint while adding a sentinel-aware request model for explicit out/cache null-equivalent behavior."
  - "Raw evaluate must validate queried metadata and buffer contracts before any dispatch or output writes."
  - "Diagnostics failure reporting now preserves explicitly supplied provided_bytes instead of always recomputing from dims."
patterns-established:
  - "Raw query and raw execute share one validated contract path and one canonical ExecutionRequest/CPU router dispatch path."
  - "RAW-02 regressions are locked with dedicated query-only, query->execute success, and mismatch diagnostics integration tests."
requirements-completed: [RAW-02]
duration: 11 min
completed: 2026-03-14
---

# Phase 2 Plan 07: Raw Query and Evaluate Integration Summary

**RAW-02 now has a deterministic raw query->execute flow with null-equivalent sentinel query semantics and CPU-routed execute contract enforcement.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-14T06:54:32Z
- **Completed:** 2026-03-14T07:06:08Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Added `runtime::raw::query` with explicit `out/cache` null-equivalent normalization so raw query mode behaves consistently for `None` and empty-slice sentinel inputs.
- Added `runtime::raw::evaluate` that re-validates raw inputs, checks query->execute invariants, constructs canonical `ExecutionRequest`, and dispatches through the 02-06 CPU router.
- Locked RAW-02 with integration coverage in `phase2_raw_query_execute` for sentinel query behavior, successful query->execute handoff, and mismatch diagnostics.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement raw workspace query flow with null-equivalent semantics** - `d01c332` (feat)
2. **Task 2: Implement raw evaluate path using shared execution request model** - `4e5161a` (feat)
3. **Task 3: Lock RAW-02 integration behavior with requirement tests** - `debd203` (fix)

## Files Created/Modified

- `src/runtime/raw/query.rs` - Added raw query contract entrypoint with sentinel normalization and diagnostics-backed workspace metadata.
- `src/runtime/raw/evaluate.rs` - Added raw execute path with query-contract checks, CPU route dispatch, and deterministic write behavior.
- `src/api/raw.rs` - Threaded API surface through runtime raw query/evaluate modules and exposed sentinel-aware query/evaluate entrypoints.
- `src/runtime/raw/mod.rs` - Registered and re-exported raw query/evaluate modules.
- `src/runtime/mod.rs` - Re-exported raw query/evaluate runtime symbols.
- `src/lib.rs` - Re-exported raw query/evaluate compatibility symbols at crate root.
- `tests/phase2_raw_query_execute.rs` - Added RAW-02 integration tests for query-only, query->execute success, and mismatch diagnostics.
- `src/diagnostics/report.rs` - Fixed diagnostics failure behavior to preserve explicit `provided_bytes`.

## Decisions Made

- Preserve backward compatibility for existing `raw::query_workspace_compat` callers while adding a dedicated sentinel-aware query request type for explicit raw compat behavior.
- Require execute-time contract validation against queried metadata (`dims`, required bytes/elements, cache contract) before dispatch to prevent drift between query and execute calls.
- Treat output/cached buffer mismatch diagnostics as first-class by preserving caller-supplied `provided_bytes` values on failures.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added the missing `phase2_raw_query_execute` test target during Task 1**
- **Found during:** Task 1 (raw query verification command)
- **Issue:** Plan verification required `tests/phase2_raw_query_execute.rs`, but the test target did not exist.
- **Fix:** Created `tests/phase2_raw_query_execute.rs` and added `raw_query_null_equivalent_contract` coverage.
- **Files modified:** `tests/phase2_raw_query_execute.rs`
- **Verification:** `cargo test --workspace --test phase2_raw_query_execute raw_query_null_equivalent_contract`
- **Committed in:** `d01c332`

**2. [Rule 1 - Bug] Preserved explicit buffer diagnostics on failure in Task 3**
- **Found during:** Task 3 (mismatch diagnostics verification)
- **Issue:** `QueryDiagnostics::record_failure` always recomputed `provided_bytes` from dims, overriding explicit output-buffer byte values.
- **Fix:** Updated diagnostics failure path to keep explicit `provided_bytes` when present and only derive from dims when missing.
- **Files modified:** `src/diagnostics/report.rs`
- **Verification:** `cargo test --workspace --test phase2_raw_query_execute`
- **Committed in:** `debd203`

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both deviations were required to satisfy RAW-02 verification semantics and did not introduce scope creep beyond plan intent.

## Issues Encountered

- Initial Task 3 diagnostics assertion failed because `provided_bytes` was overwritten by dims-derived bytes; fixed in diagnostics and re-verified.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Raw query/evaluate contracts are now wired and test-locked for RAW-02 semantics.
- Phase `02-08` can thread memory policy through this query->execute path without redefining raw contract checks.

---
*Phase: 02-cpu-compatibility-execution*
*Completed: 2026-03-14*

## Self-Check: PASSED
- FOUND: .planning/phases/02-cpu-compatibility-execution/02-07-SUMMARY.md
- FOUND COMMIT: d01c332
- FOUND COMMIT: 4e5161a
- FOUND COMMIT: debd203
