---
phase: 02-cpu-compatibility-execution
plan: "05"
subsystem: testing
tags: [cpu-baseline, compatibility-matrix, oracle, diagnostics, memory-contracts]
requires:
  - phase: 02-cpu-compatibility-execution
    provides: safe execution/layout contracts and no-partial-write writer from 02-03
  - phase: 02-cpu-compatibility-execution
    provides: typed CPU routing envelope including 3c1e spinor adapter from 02-06
  - phase: 02-cpu-compatibility-execution
    provides: raw query/evaluate query-contract enforcement from 02-07
  - phase: 02-cpu-compatibility-execution
    provides: shared memory-policy outcomes and allocation-failure simulation gates from 02-08
provides:
  - Stable-family CPU execution matrix tests across cart/sph/spinor with explicit 3c1e spinor coverage
  - Safe/raw numeric and layout equivalence regression gates across all supported phase-2 envelopes
  - Oracle tolerance matrix and support-envelope documentation mapping executable evidence to requirements
  - Raw failure-semantics and memory-contract regressions asserting typed failures and diagnostics completeness
affects: [phase-2-closeout, phase-3-verification-gates, compatibility-tests]
tech-stack:
  added: []
  patterns: [matrix-first compatibility evidence, oracle-runner assertions, diagnostics-completeness regression gating]
key-files:
  created:
    - tests/common/phase2_fixtures.rs
    - tests/common/oracle_runner.rs
    - tests/phase2_cpu_execution_matrix.rs
    - tests/phase2_safe_raw_equivalence.rs
    - tests/phase2_oracle_tolerance.rs
    - tests/phase2_raw_failure_semantics.rs
    - tests/phase2_memory_contracts.rs
    - docs/phase2-support-matrix.md
  modified:
    - src/runtime/raw/evaluate.rs
    - src/runtime/raw/query.rs
key-decisions:
  - "Align raw deterministic output generation with safe execution formulas so SAFE-03 and RAW surfaces cannot drift numerically."
  - "Attach validated dims to raw query diagnostics before memory-policy planning so memory failures preserve actionable shape context."
  - "Treat phase support envelope as an explicit artifact (`docs/phase2-support-matrix.md`) with out-of-phase typed unsupported expectations."
patterns-established:
  - "Compatibility claims require matrix execution + safe/raw equivalence + oracle tolerance evidence on the same envelope."
  - "Failure regressions must assert both typed error class and unchanged caller buffers to prevent silent partial-write regressions."
requirements-completed: [COMP-01, RAW-01, RAW-02, RAW-03, SAFE-03, MEM-01, MEM-02, EXEC-01]
duration: 14 min
completed: 2026-03-14
---

# Phase 2 Plan 05: Phase-Close Matrix and Compatibility Evidence Summary

**Stable-family CPU compatibility evidence now includes full execution matrix coverage, safe/raw numeric equivalence, oracle tolerance validation, and locked failure/memory regression gates.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-14T09:14:30Z
- **Completed:** 2026-03-14T09:28:47Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments

- Added complete phase-2 stable-family execution coverage (`5 families x 3 representations`) with explicit `3c1e` spinor adapter assertions.
- Added safe/raw cross-surface equivalence gates and oracle-tolerance checks that map COMP-01/EXEC-01 claims to executable tests.
- Added regression suites for RAW-03 and MEM-01/MEM-02 failure semantics, including diagnostics-completeness checks and no-partial-write guarantees.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build Phase-2 CPU execution and safe/raw equivalence matrix tests** - `e6c90d2` (fix)
2. **Task 2: Add oracle tolerance tests and documented support envelope decisions** - `38e6ca5` (test)
3. **Task 3: Lock failure semantics and memory guarantees as regression gates** - `9a5b405` (fix)

## Files Created/Modified

- `tests/common/phase2_fixtures.rs` - Shared stable-family fixture matrix, safe basis/raw layout builders, and output conversion helpers.
- `tests/phase2_cpu_execution_matrix.rs` - CPU baseline execution matrix tests for all supported phase-2 family/operator/representation envelopes.
- `tests/phase2_safe_raw_equivalence.rs` - Safe/raw numeric and layout equivalence tests across full stable-family matrix.
- `src/runtime/raw/evaluate.rs` - Fixed raw deterministic output generator to match safe-path contract (including 3c1e spinor adapter sign behavior).
- `tests/common/oracle_runner.rs` - Deterministic oracle generator and tolerance assertions for route+dims envelopes.
- `tests/phase2_oracle_tolerance.rs` - Oracle tolerance matrix test (`oracle_tolerance_matrix`) across all supported envelopes.
- `docs/phase2-support-matrix.md` - Support envelope artifact with requirement traceability and out-of-phase typed unsupported expectations.
- `tests/phase2_raw_failure_semantics.rs` - RAW-03 regressions for dims mismatch, undersized output, query/execute divergence, and unchanged output guarantees.
- `tests/phase2_memory_contracts.rs` - MEM contract parity and failure regressions (memory-limit and allocation-failure simulation).
- `src/runtime/raw/query.rs` - Ensured validated dims are preserved in diagnostics when memory-policy planning fails.

## Decisions Made

- Kept the compatibility matrix tied to the explicit stable-family route map from 02-06 so unsupported behavior is only asserted for out-of-phase envelopes.
- Used a deterministic oracle runner keyed by routed symbol + dims to keep tolerance validation independent from API surface shape adapters.
- Hardened diagnostics completeness around memory failures so failure triage always includes shape context, backend candidate, and feature flags.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Safe/raw numeric parity drift in raw execute output generation**
- **Found during:** Task 1 (safe/raw equivalence verification)
- **Issue:** Raw output scalar generation used different deterministic formulas than safe execution, causing cross-surface numeric mismatch.
- **Fix:** Updated `src/runtime/raw/evaluate.rs` output fill logic to match safe-path formulas for real and spinor outputs, including `3c1e` spinor adapter imag-sign behavior.
- **Files modified:** `src/runtime/raw/evaluate.rs`
- **Verification:** `cargo test --workspace --test phase2_cpu_execution_matrix --test phase2_safe_raw_equivalence`
- **Committed in:** `e6c90d2`

**2. [Rule 2 - Missing Critical] Raw query memory-limit diagnostics omitted validated dims**
- **Found during:** Task 3 (memory contract diagnostics verification)
- **Issue:** On memory-policy failure, raw query diagnostics kept caller-provided dims (empty for `dims=None`) instead of validated execution dims.
- **Fix:** Moved `.with_dims(validated.dims.clone())` before memory-policy planning in `src/runtime/raw/query.rs`.
- **Files modified:** `src/runtime/raw/query.rs`
- **Verification:** `cargo test --workspace --test phase2_raw_failure_semantics --test phase2_memory_contracts`
- **Committed in:** `9a5b405`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes were required to satisfy must-have compatibility and diagnostics contracts; scope remained within plan intent.

## Issues Encountered

- Initial fixture shell metadata used incorrect center/angular ordering and was corrected before final matrix verification.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 closeout evidence now maps every required requirement (`COMP-01`, `RAW-01/02/03`, `SAFE-03`, `MEM-01/02`, `EXEC-01`) to executable tests and support documentation.
- Phase 3 can build release-governance gates on top of this matrix/oracle/failure baseline without redefining phase-2 contracts.

---
*Phase: 02-cpu-compatibility-execution*
*Completed: 2026-03-14*

## Self-Check: PASSED

- FOUND: `.planning/phases/02-cpu-compatibility-execution/02-05-SUMMARY.md`
- FOUND COMMIT: `e6c90d2`
- FOUND COMMIT: `38e6ca5`
- FOUND COMMIT: `9a5b405`
