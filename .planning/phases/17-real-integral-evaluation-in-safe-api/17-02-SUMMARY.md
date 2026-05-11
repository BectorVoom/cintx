---
phase: 17-real-integral-evaluation-in-safe-api
plan: 02
subsystem: api
tags: [executor-swap, safe-api, cintx-rs, real-integrals, unit-test-rewrite, cubecl]

# Dependency graph
requires:
  - phase: 16-multi-backend-support
    provides: real CubeClExecutor with BackendExecutor impl
  - phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
    provides: real 1e/2c2e kernel dispatch in cintx-cubecl
provides:
  - SessionRequest::evaluate dispatches to real CubeClExecutor for all arity-2 operators
  - Deterministic + nonzero unit test guarding against zero-fill stub regression
affects: [17-03, pyscf-rs-consumer, oracle-parity-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - executor-swap: delete local shadow struct + helper, add `use cintx_cubecl::CubeClExecutor;`
    - stats-accumulation: accumulate from both io.transfer_bytes() and chunk_stats.transfer_bytes

key-files:
  created: []
  modified:
    - crates/cintx-rs/src/api.rs

key-decisions:
  - "Accumulate chunk_stats.transfer_bytes in addition to io.transfer_bytes() so stats.transfer_bytes > 0 for real executor (the stub used io.record_transfer_bytes() but the real executor sets it only on returned stats)"
  - "Rename evaluate_runs_runtime_path_and_returns_owned_output to evaluate_returns_deterministic_nonzero_real_values to reflect what is actually asserted"
  - "Remove cintxRsError, OutputOwnership, WorkspaceBytes imports made unused by stub deletion"

patterns-established:
  - "executor-swap: add use cintx_cubecl::CubeClExecutor; at top of file; delete shadow struct + helper; existing call site resolves to real type"
  - "unit-test-rewrite: idempotency (two evaluate() calls agree) + nonzero (|v| > 1e-18) + preserved invariants = correct smoke for real executor"

requirements-completed: [RVAL-01, RVAL-03]

# Metrics
duration: 20min
completed: 2026-05-11
---

# Phase 17 Plan 02: Executor Swap and Unit Test Rewrite Summary

**Deleted fill_staging_values stub and local CubeClExecutor shadow from api.rs; safe API now dispatches to real cintx_cubecl::CubeClExecutor for all arity-2 operators; unit test rewritten to assert idempotency + nonzero**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-11T23:00:00Z
- **Completed:** 2026-05-11T23:20:06Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Deleted `fill_staging_values` helper (lines 465-490) and local stub `CubeClExecutor` struct + impls (lines 492-562) from `crates/cintx-rs/src/api.rs`
- Added `use cintx_cubecl::CubeClExecutor;` import; existing `let executor = CubeClExecutor::new();` call site unchanged and now resolves to the real executor
- Removed now-unused imports: `cintxRsError`, `OutputOwnership`, `WorkspaceBytes`
- Rewrote `evaluate_runs_runtime_path_and_returns_owned_output` to `evaluate_returns_deterministic_nonzero_real_values` with idempotency + nonzero + preserved invariant checks
- All 11 `cargo test -p cintx-rs --locked` tests pass; all four feature-flag builds pass

## Task Commits

1. **Task 1: Swap stub CubeClExecutor for real cintx_cubecl::CubeClExecutor** - `62a8bac` (feat)
2. **Task 2: Rewrite unit test to idempotency + nonzero** - `a63b8cd` (feat)

## Files Created/Modified
- `crates/cintx-rs/src/api.rs` - Deleted 100-line stub block, added one import, rewrote one test, fixed transfer_bytes accumulation

## Decisions Made
- Accumulate `chunk_stats.transfer_bytes` in addition to `io.transfer_bytes()`: the real executor sets `transfer_bytes` in the returned `ExecutionStats` struct but does not call `io.record_transfer_bytes()` (unlike the stub). Without this fix, `stats.transfer_bytes` would always be 0 for the real executor path.
- Rename the test: the old name implied it tested the runtime path (which was false for the stub). The new name honestly describes what's asserted.
- Remove unused imports: `cintxRsError`, `OutputOwnership`, `WorkspaceBytes` were only used by the deleted stub code.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed transfer_bytes accumulation to collect from chunk_stats**
- **Found during:** Task 2 (unit test rewrite)
- **Issue:** `api.rs` chunk loop accumulated `total_transfer_bytes` solely from `io.transfer_bytes()`. The real `CubeClExecutor::execute` returns `transfer_bytes` in the `ExecutionStats` result but does NOT call `io.record_transfer_bytes()`, causing `stats.transfer_bytes` to be 0 after executor swap.
- **Fix:** Changed accumulation from `io.transfer_bytes()` to `io.transfer_bytes().saturating_add(chunk_stats.transfer_bytes)` so stats correctly reflect the real executor's reported transfer.
- **Files modified:** crates/cintx-rs/src/api.rs
- **Verification:** `assert!(output1.stats.transfer_bytes > 0)` now passes in the rewritten test.
- **Committed in:** a63b8cd (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary for correctness — stats.transfer_bytes was silently 0 for the real executor. No scope creep.

## Issues Encountered
None beyond the transfer_bytes accumulation fix above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RVAL-01 satisfied: `fill_staging_values` removed; safe API now routes through real `CubeClExecutor` for all arity-2 operators
- RVAL-03 satisfied: no public API change; public surface byte-identical to v1.2
- Plan 17-03 (oracle parity tests for safe API arity-2 sweep) can proceed

## Self-Check

- [x] `crates/cintx-rs/src/api.rs` exists and contains `use cintx_cubecl::CubeClExecutor;`
- [x] No `fn fill_staging_values` in `crates/cintx-rs/src/api.rs`
- [x] No `struct CubeClExecutor` (private stub) in `crates/cintx-rs/src/api.rs`
- [x] No `owned_values[0], 1.0` assertion in `crates/cintx-rs/src/api.rs`
- [x] Task 1 commit `62a8bac` exists
- [x] Task 2 commit `a63b8cd` exists
- [x] All 11 `cargo test -p cintx-rs --locked` tests pass

## Self-Check: PASSED

---
*Phase: 17-real-integral-evaluation-in-safe-api*
*Completed: 2026-05-11*
