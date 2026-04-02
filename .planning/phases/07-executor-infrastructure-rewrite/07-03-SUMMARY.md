---
phase: 07-executor-infrastructure-rewrite
plan: "03"
subsystem: executor
tags: [rust, cubecl, executor, staging, compat, safe-facade]

requires:
  - phase: 07-executor-infrastructure-rewrite
    provides: "Plan 02: CubeClExecutor rewritten with direct staging writes via BackendExecutor"

provides:
  - "RecordingExecutor deleted from cintx-compat/src/raw.rs"
  - "RecordingExecutor deleted from cintx-rs/src/api.rs"
  - "eval_raw rewritten with manual chunk loop; staging read directly from ExecutionIo"
  - "Safe facade evaluate() rewritten with manual chunk loop; staging read directly from ExecutionIo"
  - "eval_raw_reads_staging_directly test verifying direct staging path end-to-end"

affects:
  - 07-executor-infrastructure-rewrite
  - cintx-compat
  - cintx-rs
  - cintx-capi

tech-stack:
  added: []
  patterns:
    - "Direct staging pattern: allocate Vec<f64> staging per chunk, pass to ExecutionIo::new(), read after executor.execute() returns"
    - "Chunk loop replication: replicate schedule_chunks/ExecutionIo loop at compat/safe-facade level to own staging buffer"

key-files:
  created: []
  modified:
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-rs/src/api.rs
    - crates/cintx-capi/src/shim.rs

key-decisions:
  - "Replicate chunk loop at caller level rather than using runtime::evaluate() so staging buffers can be owned and read after execution"
  - "Fix three incorrect all-zeros test assertions (Rule 1: eval_raw now correctly writes executor staging values to output)"
  - "cintx-oracle parity failures pre-existed and are out-of-scope; tracked in deferred-items"

patterns-established:
  - "Staging pattern: all eval paths must own their staging buffer and pass it into ExecutionIo, not depend on external capture wrappers"

requirements-completed: [EXEC-07, VERI-06]

duration: 6min
completed: 2026-04-02
---

# Phase 07 Plan 03: RecordingExecutor Removal Summary

**RecordingExecutor deleted from all call sites; eval_raw and safe facade evaluate() now allocate owned staging buffers and read executor output directly via manual chunk loop**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-02T23:03:22Z
- **Completed:** 2026-04-02T23:09:50Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Deleted `RecordingExecutor` struct and all impl blocks from `cintx-compat/src/raw.rs` and `cintx-rs/src/api.rs`
- Rewrote `eval_raw()` to replicate the runtime chunk loop with owned staging buffers; executor.execute() writes directly into the staging slice we own
- Rewrote safe facade `evaluate()` in `SessionQuery` to use the same direct chunk loop pattern
- Added `eval_raw_reads_staging_directly` test that proves the direct staging path is exercised and bytes_written > 0
- All 29 cintx-compat tests and 10 cintx-rs tests pass; full non-oracle workspace clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete RecordingExecutor from cintx-compat, wire eval_raw directly** - `f96208d` (feat)
2. **Task 2: Delete RecordingExecutor from cintx-rs safe facade, wire evaluate directly** - `093cf5f` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `crates/cintx-compat/src/raw.rs` - Removed `evaluate` import; added `BackendExecutor`, `ExecutionIo`, `schedule_chunks`, `WorkspaceAllocator`; rewrote eval_raw execution body with manual chunk loop; added `eval_raw_reads_staging_directly` test; updated two tests from incorrect all-zeros assertions to correct non-zero assertions
- `crates/cintx-rs/src/api.rs` - Removed `RecordingExecutor` struct and all impls; removed `evaluate as runtime_evaluate` and `std::sync::Mutex` imports; rewrote `SessionQuery::evaluate()` with manual chunk loop using owned staging buffers
- `crates/cintx-capi/src/shim.rs` - Updated shim test to assert non-zero output (same Rule 1 fix for incorrect zeros assertion)

## Decisions Made

- Replicate the runtime `evaluate()` chunk loop at the compat/safe-facade level so callers own the staging buffer and can read it after `executor.execute()` returns — this is simpler and more direct than adding a callback or return path to the runtime evaluate function
- Keep `cintx_runtime::evaluate()` available in the runtime (no removal needed) since it remains useful for other callers and the simple passthrough path

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect all-zeros test assertions in cintx-compat**
- **Found during:** Task 1 (Delete RecordingExecutor from cintx-compat)
- **Issue:** Two tests (`memory_limit_hint_can_chunk_successfully`, `three_center_contract_query_and_eval_work_for_supported_backend`) asserted `out.iter().all(|v| *v == 0.0)`, which was correct only because the old broken implementation allocated a separate zero-filled buffer instead of reading from executor staging. Now that staging is read correctly, the output contains non-zero stub values from `fill_cartesian_staging`.
- **Fix:** Updated assertions to `out.iter().any(|v| *v != 0.0)` reflecting the correct behavior
- **Files modified:** `crates/cintx-compat/src/raw.rs`
- **Verification:** 29 cintx-compat tests pass
- **Committed in:** `f96208d` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed incorrect all-zeros test assertion in cintx-capi**
- **Found during:** Task 2 (full workspace test run)
- **Issue:** `shim::tests::query_and_eval_wrappers_succeed_and_clear_tls_error` asserted all-zeros on eval output, same root cause as above — the capi shim calls through eval_raw which now writes non-zero stub values
- **Fix:** Updated assertion to `out.iter().any(|v| *v != 0.0)`
- **Files modified:** `crates/cintx-capi/src/shim.rs`
- **Verification:** All 13 cintx-capi tests pass
- **Committed in:** `093cf5f` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** All auto-fixes corrected tests that were asserting broken behavior. No scope creep.

## Issues Encountered

- `cintx-oracle` parity tests (`evaluated_output_parity_and_optimizer_equivalence_hold`, `parity_artifacts_are_written`) were pre-existing failures before this plan's changes — verified by git stash check. Out-of-scope per scope boundary rule; logged to deferred-items.

## Next Phase Readiness

- EXEC-07 complete: `RecordingExecutor` is fully removed from the codebase
- VERI-06 satisfied: `eval_raw_reads_staging_directly` test passes and proves direct staging path
- Staging values flow from executor kernel dispatch through owned staging buffers to output
- Plan 07-04 can proceed (if any)

---
*Phase: 07-executor-infrastructure-rewrite*
*Completed: 2026-04-02*
