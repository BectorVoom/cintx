---
phase: 01-contracts-and-typed-foundations
plan: "01"
subsystem: api
tags: [rust, typed-contracts, thiserror, validation, error-taxonomy]
requires:
  - phase: none
    provides: project initialization baseline
provides:
  - typed Atom/Shell/Basis/Operator/Representation domain contracts with constructor invariants
  - public LibcintRsError taxonomy with stable SAFE-04 categories
  - requirement-mapped tests for SAFE-01 construction and SAFE-04 variant matching
affects: [phase-2-cpu-compatibility-execution, plan-01-02-workspace-diagnostics]
tech-stack:
  added: []
  patterns: [typed constructor validation, explicit public error variants, test-level variant matching]
key-files:
  created:
    - src/lib.rs
    - src/contracts/mod.rs
    - src/contracts/atom.rs
    - src/contracts/shell.rs
    - src/contracts/basis.rs
    - src/contracts/operator.rs
    - src/contracts/representation.rs
    - src/errors/mod.rs
    - src/errors/libcint_error.rs
    - tests/phase1_typed_contracts.rs
    - tests/phase1_error_taxonomy.rs
  modified:
    - src/lib.rs
    - src/contracts/mod.rs
    - src/contracts/atom.rs
    - src/contracts/shell.rs
    - src/contracts/basis.rs
    - src/contracts/operator.rs
    - tests/phase1_typed_contracts.rs
    - tests/phase1_error_taxonomy.rs
key-decisions:
  - "Keep constructors pointer-free and fail-fast with typed Result returns."
  - "Represent SAFE-04 with explicit LibcintRsError variants rather than string-classified errors."
patterns-established:
  - "Contracts return Result<T, LibcintRsError> directly with no anyhow leakage in public APIs."
  - "Requirement tests assert concrete error variants using matches! instead of error-message parsing."
requirements-completed: [SAFE-01, SAFE-04]
duration: 5 min
completed: 2026-03-14
---

# Phase 1 Plan 1: Typed Domain Contracts and Error Taxonomy Summary

**Typed Atom/Shell/Basis domain builders and a public LibcintRsError taxonomy now lock SAFE-01 and SAFE-04 behavior for downstream execution plans.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T01:32:24Z
- **Completed:** 2026-03-14T01:38:21Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments
- Added a library surface in `src/lib.rs` and split typed contract modules for atoms, shells, basis sets, operators, and representations.
- Introduced `LibcintRsError` as the public typed failure taxonomy, including unsupported/input/layout/dims/memory/allocation/backend categories.
- Added and expanded requirement-mapped tests to verify constructor invariants and stable error-variant matching across representative invalid inputs.

## Task Commits

Each task was committed atomically:

1. **Task 1: Introduce typed contract modules for atoms, shells, basis, operators, and representations** - `3238eee` (feat)
2. **Task 2: Define and wire the public typed error taxonomy** - `7d554b3` (feat)
3. **Task 3: Add requirement-mapped contract and error tests** - `106d6f7` (test)

## Files Created/Modified
- `src/lib.rs` - Library entry point and public re-exports for contracts/errors.
- `src/contracts/mod.rs` - Contract module wiring and shared dimensions validator.
- `src/contracts/atom.rs` - Typed Atom model with atomic-number and coordinate invariants.
- `src/contracts/shell.rs` - Shell and primitive builders enforcing contraction metadata validity.
- `src/contracts/basis.rs` - BasisSet constructor validating atom/shell linkage invariants.
- `src/contracts/operator.rs` - Operator/family model with unsupported combination checks.
- `src/contracts/representation.rs` - Representation enum for cartesian/spherical/spinor selection.
- `src/errors/libcint_error.rs` - Public SAFE-04 error taxonomy (`LibcintRsError`).
- `tests/phase1_typed_contracts.rs` - SAFE-01 typed construction and invariant tests.
- `tests/phase1_error_taxonomy.rs` - SAFE-04 category and variant-differentiation tests.

## Decisions Made
- Constructors own their contract data and reject invalid state immediately to keep the safe surface deterministic.
- Variant matching is part of the contract: tests assert exact `LibcintRsError` variants so downstream callers can branch without parsing strings.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `01-01` outputs are complete and stable for `01-02` workspace-query and diagnostics contract work.
- No blockers identified for continuing Phase 1.

## Self-Check: PASSED

- Verified all key created files exist on disk.
- Verified all task commits (`3238eee`, `7d554b3`, `106d6f7`) exist in git history.

---
*Phase: 01-contracts-and-typed-foundations*
*Completed: 2026-03-14*
