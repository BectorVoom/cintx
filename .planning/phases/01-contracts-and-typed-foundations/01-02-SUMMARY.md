---
phase: 01-contracts-and-typed-foundations
plan: "02"
subsystem: api
tags: [rust, workspace-query, diagnostics, tracing, validation]
requires:
  - phase: 01-01
    provides: typed domain contracts and SAFE-04 error taxonomy
provides:
  - deterministic workspace query contract shared by safe and raw query surfaces
  - MEM-03 diagnostics payload with correlated trace metadata on query failures
  - safe/raw workspace query APIs with dims validation and requirement-mapped tests
affects: [phase-2-cpu-compatibility-execution, raw-compatibility-contracts, diagnostics-observability]
tech-stack:
  added: [tracing]
  patterns: [shared validation outputs, query error plus diagnostics payload, safe/raw query separation]
key-files:
  created:
    - src/runtime/mod.rs
    - src/runtime/validator.rs
    - src/runtime/workspace_query.rs
    - src/diagnostics/mod.rs
    - src/diagnostics/report.rs
    - src/api/mod.rs
    - src/api/safe.rs
    - src/api/raw.rs
    - tests/phase1_workspace_query.rs
    - tests/phase1_diagnostics_contract.rs
  modified:
    - src/lib.rs
    - Cargo.toml
    - Cargo.lock
key-decisions:
  - "Return QueryError { error, diagnostics } so callers keep typed matching and machine-parseable context together."
  - "Keep safe query API free of dims while raw API accepts optional dims override that must exactly match natural shape."
patterns-established:
  - "Safe and raw query surfaces share validator outputs (ValidatedInputs and ValidatedShape) before workspace estimation."
  - "Query failures emit structured tracing fields keyed by deterministic correlation_id for trace/error joining."
requirements-completed: [SAFE-02, MEM-03]
duration: 8 min
completed: 2026-03-14
---

# Phase 1 Plan 2: Workspace Query and Diagnostics Contract Summary

**Deterministic safe/raw `query_workspace` contracts now share a typed validator pipeline and return typed failures with structured MEM-03 diagnostics metadata.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T01:43:49Z
- **Completed:** 2026-03-14T01:51:27Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments
- Implemented shared runtime validation (`ValidatedInputs`, `ValidatedShape`) and deterministic workspace estimation with stable alignment/dims/bytes outputs.
- Added structured diagnostics contracts (`QueryDiagnostics`, `QueryError`) and correlated tracing metadata across validation/workspace failure boundaries.
- Wired `api::safe::query_workspace` and `api::raw::query_workspace`, then locked determinism, dims rejection, and diagnostics completeness with integration tests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build shared validator outputs and deterministic workspace estimator** - `542db04` (feat)
2. **Task 2: Define structured diagnostics contract and tracing metadata** - `9818f87` (feat)
3. **Task 3: Wire safe/raw query APIs and add contract tests** - `52fbd84` (feat)
4. **Post-plan hardening: Resolve Rust review blockers (diagnostics bytes semantics + operator matrix strictness)** - `f164ed7` (fix)

## Files Created/Modified
- `src/runtime/validator.rs` - Canonical safe/raw validation into deterministic `ValidatedInputs` and `ValidatedShape`.
- `src/runtime/workspace_query.rs` - Deterministic `WorkspaceQuery` estimation and failure mapping to diagnostics wrapper errors.
- `src/diagnostics/report.rs` - MEM-03 payload and `QueryError` wrapper that preserves typed errors plus machine-parseable context.
- `src/contracts/operator.rs` - Explicit operator/family allow-list to enforce typed API compatibility contracts.
- `src/api/safe.rs` - Safe workspace query API that does not expose raw dims override.
- `src/api/raw.rs` - Raw-compatible workspace query API with deterministic dims override validation.
- `tests/phase1_workspace_query.rs` - Determinism and invalid-dims pre-execution contract tests.
- `tests/phase1_diagnostics_contract.rs` - Diagnostics payload completeness test for query-path failures.
- `tests/phase1_typed_contracts.rs` - Operator/family compatibility matrix regression coverage.

## Decisions Made
- Standardized query failure returns as `QueryError` to avoid losing typed variant matching while attaching structured diagnostics.
- Canonicalized feature flags in validation to keep diagnostics correlation IDs stable for identical effective inputs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `tracing` crate for required structured correlation events**
- **Found during:** Task 2
- **Issue:** Query-path tracing events required by MEM-03 could not be emitted without a tracing dependency.
- **Fix:** Added `tracing` to dependencies and wired structured `error`/`debug` events in diagnostics reporting.
- **Files modified:** `Cargo.toml`, `Cargo.lock`, `src/diagnostics/report.rs`
- **Verification:** `cargo test --workspace --test phase1_diagnostics_contract diagnostics_fields_complete`
- **Committed in:** `9818f87`

**2. [Rule 2 - Missing Critical] Resolved Rust review blockers before phase verification**
- **Found during:** Post-plan quality gate (`rust-reviewer`)
- **Issue:** Diagnostics `provided_bytes` could be stale, operator/family matrix was under-constrained, and regression tests missed edge cases.
- **Fix:** Recomputed diagnostics bytes from effective dims, treated unknown dims as `None`, boxed query errors for clippy size warnings, enforced explicit operator/family compatibility matrix, and added targeted regression tests.
- **Files modified:** `src/diagnostics/report.rs`, `src/runtime/workspace_query.rs`, `src/contracts/operator.rs`, `tests/phase1_workspace_query.rs`, `tests/phase1_diagnostics_contract.rs`, `tests/phase1_typed_contracts.rs`
- **Verification:** `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`
- **Committed in:** `f164ed7`

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing critical)
**Impact on plan:** Fixes tightened contract correctness and quality-gate compliance with no scope expansion.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SAFE-02 workspace introspection and MEM-03 diagnostics contracts are now stable for Phase 2 execution-path implementation.
- Raw dims rejection and diagnostics completeness are covered with requirement-mapped tests, reducing contract churn risk for CPU kernels.

## Self-Check: PASSED

- Verified all key created files and summary output exist on disk.
- Verified all task and hardening commits (`542db04`, `9818f87`, `52fbd84`, `f164ed7`) exist in git history.

---
*Phase: 01-contracts-and-typed-foundations*
*Completed: 2026-03-14*
