---
phase: 08-gaussian-primitive-infrastructure-and-boys-function
plan: "04"
subsystem: math
tags: [rys, boys, host-wrapper, integration-test, requirements]
dependency_graph:
  requires: [08-03]
  provides: [rys_root1_host, math-integration-crosscheck]
  affects: [cintx-cubecl/math/rys.rs, math_integration_tests.rs, REQUIREMENTS.md]
tech_stack:
  added: []
  patterns: [host-wrapper-pattern]
key_files:
  created: []
  modified:
    - crates/cintx-cubecl/src/math/rys.rs
    - crates/cintx-cubecl/tests/math_integration_tests.rs
    - .planning/REQUIREMENTS.md
decisions:
  - "Add rys_root1_host as a pure-Rust host wrapper replicating #[cube] rys_root1 branching logic exactly"
  - "Wire Rys-Boys weight-sum identity crosscheck at large/moderate/small x domains with appropriate tolerances"
  - "Mark MATH-03 Complete in requirements tracking after host wrapper and test wiring confirmed"
metrics:
  duration: "8min"
  completed: "2026-04-03T02:43:34Z"
  tasks_completed: 2
  files_changed: 3
---

# Phase 08 Plan 04: Rys-Boys Cross-validation and MATH-03 Closure Summary

**One-liner:** Host wrapper `rys_root1_host` added to rys.rs and wired into Rys-Boys weight-sum identity crosscheck test, closing MATH-03 verification gap.

## Objective

Close two verification gaps from 08-VERIFICATION.md:
1. Wire rys_root1 into the math integration test for Rys-Boys cross-validation (Gap 1)
2. Update REQUIREMENTS.md MATH-03 checkbox to reflect completed status

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add rys_root1_host wrapper and wire Rys-Boys crosscheck | a531f6e | rys.rs, math_integration_tests.rs |
| 2 | Update REQUIREMENTS.md MATH-03 status to Complete | 9e2c470 | REQUIREMENTS.md |

## What Was Built

### Task 1: rys_root1_host wrapper

Added `pub fn rys_root1_host(x: f64) -> (f64, f64)` to `crates/cintx-cubecl/src/math/rys.rs` before the `rys_roots` dispatch function. The host wrapper:

- Replicates the exact branching logic from the `#[cube]` `rys_root1` function
- Uses pure Rust (no CubeCL runtime required)
- Returns `(root, weight)` tuple corresponding to `u[0]` and `w[0]` from the cube function
- Follows the host wrapper pattern established in Phase 8 (D-12) for `boys_gamma_inc_host` and `compute_pdata_host`

Updated `math_integration_rys_boys_crosscheck` in `math_integration_tests.rs` to:
- Import `rys_root1_host` from `cintx_cubecl::math::rys`
- Actually call `rys_root1_host(x)` and compare against `boys_gamma_inc_host(x, 0)`
- Test the weight-sum identity across large x (epsilon=1e-12), moderate x (epsilon=1e-8), and small x (epsilon=1e-10) domains

### Task 2: REQUIREMENTS.md MATH-03 update

- Changed `- [ ] **MATH-03**` to `- [x] **MATH-03**` (line 65)
- Changed `| MATH-03 | Phase 8 | Pending |` to `| MATH-03 | Phase 8 | Complete |` (line 134)

## Verification Results

```
cargo test -p cintx-cubecl --features cpu
```

All test suites pass:
- lib unit tests: 32 passed
- boys_tests: 6 passed
- math_integration_tests: 4 passed (including rys_boys_crosscheck)
- os_tests: 7 passed
- pdata_tests: 3 passed
- rys_tests: 8 passed
- doc-tests: 0 tests
- Total: 60 passed, 0 failed

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- [x] `crates/cintx-cubecl/src/math/rys.rs` contains `pub fn rys_root1_host`
- [x] `crates/cintx-cubecl/tests/math_integration_tests.rs` contains `rys_root1_host`
- [x] `.planning/REQUIREMENTS.md` contains `[x] **MATH-03**` and `Complete`
- [x] Commits a531f6e and 9e2c470 exist
- [x] All 60 math tests pass
