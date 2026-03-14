---
phase: 03-verification-and-compatibility-governance
plan: "01"
subsystem: testing
tags: [compatibility, helper-parity, transforms, diagnostics, runtime]
requires:
  - phase: 02-cpu-compatibility-execution
    provides: stable-family route matrix, safe/raw fixtures, and oracle helper baselines
provides:
  - runtime helper API for AO counts, shell offsets, normalization metadata, and deterministic transform scalars
  - phase-3 helper parity regression suite for counts/offsets/normalization and cart/sph/spinor transforms
  - typed helper failure-semantics coverage for malformed layouts and out-of-envelope route requests
affects:
  - 03-02 manifest governance
  - 03-03 optimizer equivalence
  - 03-04 CI compatibility gates
tech-stack:
  added: []
  patterns:
    - deterministic helper surfaces exported from runtime/lib
    - matrix-driven compatibility parity checks
    - typed diagnostics assertions for failure contracts
key-files:
  created:
    - src/runtime/helpers.rs
    - tests/common/phase3_helper_cases.rs
    - tests/phase3_helper_transform_parity.rs
  modified:
    - src/runtime/mod.rs
    - src/lib.rs
    - tests/common/phase2_fixtures.rs
key-decisions:
  - "Expose helper parity functions through runtime and crate-root re-exports so compatibility gates and downstream plans use one public surface."
  - "Keep helper validation failures typed via existing LibcintRsError variants rather than adding parallel error classes."
  - "Use stable-family matrix fixtures + oracle parity as the canonical transform regression gate, including explicit 3c1e spinor adapter coverage."
patterns-established:
  - "Helper parity tests pull dims from safe query contracts and compare deterministic helper transform scalars to oracle expectations."
  - "Negative-path helper tests assert both error variant and field-specific diagnostics text."
requirements-completed: [COMP-02, VERI-03]
duration: 3min
completed: 2026-03-14
---

# Phase 3 Plan 01: Helper Transform Parity Foundation Summary

**Runtime helper parity APIs now provide deterministic AO count/offset/normalization and transform outputs, with phase-3 matrix tests proving cart/sph/spinor compatibility and typed failure semantics.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T11:57:30Z
- **Completed:** 2026-03-14T12:00:45Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Added `runtime::helpers` with deterministic AO count/offset helpers, primitive normalization metadata, and transform scalar generation.
- Added matrix-driven phase-3 helper transform parity tests across the full phase-2 stable-family envelope, including explicit `3c1e` spinor adapter behavior.
- Locked helper failure semantics with typed/diagnostics assertions for malformed `bas/env`, invalid dims/exponents, and out-of-envelope routes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement deterministic helper parity surface for AO counts, offsets, and normalization** - `5df468a` (feat)
2. **Task 2: Add transform parity matrix tests for cart/sph/spinor helper behavior** - `8ae96c1` (test)
3. **Task 3: Lock helper failure semantics and diagnostics contracts** - `7031e8a` (test)

**Plan metadata:** Pending final docs commit in this execution run.

## Files Created/Modified
- `src/runtime/helpers.rs` - New helper parity API for component counts, shell offsets, normalization metadata, and deterministic transform scalar generation.
- `src/runtime/mod.rs` - Added helper module and public re-exports.
- `src/lib.rs` - Re-exported helper parity API at crate root.
- `tests/common/phase2_fixtures.rs` - Added expected helper count/offset fixtures and phase-3 helper options fixture.
- `tests/common/phase3_helper_cases.rs` - Added helper normalization expectations and malformed helper input builders.
- `tests/phase3_helper_transform_parity.rs` - Added COMP-02/VERI-03 parity and failure-semantics regression tests.

## Decisions Made
- Used the existing `RawBasView`/`RawEnvView` validation path inside helper APIs so error typing/diagnostics stay aligned with phase-1/2 contract style.
- Kept deterministic transform helper generation seed-compatible with existing oracle/executor contracts to prevent drift.
- Asserted explicit `3c1e` spinor adapter routing/sign behavior inside the parity matrix to preserve the phase-2 special-case contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bootstrapped the phase-3 helper test target during Task 1**
- **Found during:** Task 1 (Implement deterministic helper parity surface for AO counts, offsets, and normalization)
- **Issue:** Task-1 verification command targeted `tests/phase3_helper_transform_parity.rs`, which did not exist yet.
- **Fix:** Created initial phase-3 helper parity fixtures/test target in Task 1 so the required verification command could run immediately.
- **Files modified:** `tests/common/phase2_fixtures.rs`, `tests/common/phase3_helper_cases.rs`, `tests/phase3_helper_transform_parity.rs`
- **Verification:** `cargo test --workspace --test phase3_helper_transform_parity helper_counts_offsets_normalization_parity`
- **Committed in:** `5df468a`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** No scope creep; the auto-fix was required to satisfy task-level verification ordering.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 now has executable helper/transform compatibility evidence (COMP-02) and typed helper failure contract coverage (VERI-03).
- Manifest governance and CI-gate plans can now depend on this parity suite as the helper compatibility gate input.

---
*Phase: 03-verification-and-compatibility-governance*
*Completed: 2026-03-14*

## Self-Check: PASSED
