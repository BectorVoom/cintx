---
phase: 11-helper-transform-completion-4c1e-real-kernel
plan: 02
subsystem: kernels
tags: [rust, cubecl, 4c1e, gaussian, polynomial-recurrence, HRR, spinor-validation]

# Dependency graph
requires:
  - phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
    provides: TwoEShape, HRR functions, cart_to_sph_2e, 2e/3c1e/3c2e real kernel patterns
provides:
  - Real 4c1e Gaussian overlap kernel using polynomial 1D recurrence (g4c1e.c algorithm)
  - Spinor-first validation ordering in both center_4c1e.rs and raw.rs (D-05)
  - fill_4c1e_g_tensor function with nroots=1 (no Rys quadrature)
  - build_4c1e_shape helper with forced nroots=1
affects: [oracle-gate, 4c1e-integration-tests, validate_4c1e_envelope callers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "4c1e G-tensor fill: polynomial 1D recurrence instead of Rys quadrature (nroots=1 always)"
    - "Spinor-first validation: D-05 requires spinor check before feature gate in both kernel and compat layers"
    - "Shape4c1e: local struct mirrors TwoEShape but with nroots forced to 1"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/center_4c1e.rs
    - crates/cintx-compat/src/raw.rs

key-decisions:
  - "4c1e G-tensor uses polynomial 1D recurrence (buf[i+1] = 0.5*i/aijkl*buf[i-1] - r1r12*buf[i]) with nroots=1 hardcoded, not Rys quadrature — matches g4c1e.c algorithm"
  - "fac prefactor applied only to z-axis initial value (buf[0] = fac/(aijkl*sqrt(aijkl))); x and y axes start at 1.0 — z-axis product forms the integral normalization"
  - "Spinor check placed FIRST in both ensure_validated_4c1e (center_4c1e.rs) and validate_4c1e_envelope (raw.rs) per D-05 — spinor 4c1e returns UnsupportedApi with 'spinor' message before feature gate check"
  - "Shape4c1e is a local struct (not reusing TwoEShape) to allow nroots=1 override without changing two_electron.rs interface"
  - "HRR reuses the same 4-branch pattern (ibase/kbase selection) from the 2e kernel — same strides, same branch logic, just nroots=1"

patterns-established:
  - "Polynomial-recurrence G-tensor fill: apply fac only on axis==2 initial value; x/y start at 1.0"
  - "nroots=1 layout: di=1, stride layout same as 2e but g_size is smaller (no root dimension > 1)"

requirements-completed: [4C1E-01, 4C1E-03]

# Metrics
duration: 7min
completed: 2026-04-04
---

# Phase 11 Plan 02: 4c1e Real Kernel Summary

**Real 4c1e Gaussian overlap kernel with polynomial 1D recurrence (g4c1e.c), nroots=1, and spinor-first validation in both center_4c1e.rs and raw.rs**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-04T08:04:16Z
- **Completed:** 2026-04-04T08:11:36Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Replaced 4c1e stub in center_4c1e.rs with a real polynomial recurrence kernel matching g4c1e.c
- Implemented fill_4c1e_g_tensor using buf[i+1] = 0.5*i/aijkl*buf[i-1] - r1r12*buf[i] with nroots=1 (no Rys quadrature)
- Fixed spinor-first validation ordering in both ensure_validated_4c1e (center_4c1e.rs) and validate_4c1e_envelope (raw.rs) per D-05
- Added 3 unit tests covering G-tensor fill, nroots=1 invariant, and spinor error message content

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix spinor-first validation and implement 4c1e real kernel** - `554a379` (feat)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` - Real polynomial recurrence kernel replacing stub; spinor check moved to first position in ensure_validated_4c1e
- `crates/cintx-compat/src/raw.rs` - Spinor check moved to first position in validate_4c1e_envelope (before feature gate)

## Decisions Made
- Used a local `Shape4c1e` struct instead of reusing `TwoEShape` from two_electron.rs to allow nroots=1 override without modifying the 2e interface. The shape layout (ibase/kbase, strides di/dk/dl/dj, g_size) is identical to TwoEShape but computed with nroots=1.
- The fac prefactor in fill_4c1e_g_tensor is applied only to the z-axis initial value (`buf[0] = fac / (aijkl * aijkl.sqrt())`); x and y axes start at 1.0. This matches g4c1e.c where `fac` is the full integral normalization and is applied once in the z-direction product.
- HRR reuses the same 4-branch selection (ibase/kbase) copied from two_electron.rs. With nroots=1, the root loop in the HRR reduces to a single stride-1 pass.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. The polynomial recurrence algorithm from g4c1e.c translated cleanly. The center_4c1e module is gated behind `#[cfg(feature = "with-4c1e")]`, so tests require `--features cpu,with-4c1e` to run — verified that all 3 unit tests pass under that feature set.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 4c1e kernel is complete and produces non-zero output for valid cart/sph evaluations
- Spinor 4c1e requests correctly return UnsupportedApi with "spinor" in the message before any feature gate check in both kernel and compat layers
- Ready for oracle comparison against upstream libcint int4c1e_sph in subsequent plans

---
*Phase: 11-helper-transform-completion-4c1e-real-kernel*
*Completed: 2026-04-04*
