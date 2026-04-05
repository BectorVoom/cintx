---
phase: 13-f12-stg-yp-kernels
plan: "02"
subsystem: compute
tags: [f12, stg, yp, cubecl, kernels, raw-compat, safe-api, PTR_F12_ZETA]

# Dependency graph
requires:
  - phase: 13-01
    provides: stg_roots_host function, OperatorEnvParams.f12_zeta, validate_f12_env_params, stub launch_f12
provides:
  - Full 10-entry-point F12 kernel implementation (5 STG + 5 YP variants) with shared VRR/HRR pipeline
  - PTR_F12_ZETA=9 constant and env[9] extraction in raw compat path (eval_raw)
  - validate_f12_env_params call in eval_raw before dispatch (typed error for zero/missing zeta)
  - f12_zeta field in ExecutionOptions with propagation through safe API SessionQuery
  - f12_zeta() builder method on SessionBuilder for ergonomic safe API zeta setting
affects: [cintx-compat, cintx-rs, cintx-cubecl, cintx-runtime, f12 oracle harness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "F12 nroots formula: (L_tot + 3) / 2 (ceiling division) — differs from 2e formula at odd L_tot"
    - "STG weight post-processing: w *= (1-u) * 2*ua/zeta; YP: w *= u — transforms stg_roots output to quadrature weights"
    - "F12 symbol detection by operator_symbol() prefix (int2e_stg/int2e_yp) not canonical_family (which is 2e)"
    - "launch_f12 dispatcher uses operator_name() prefix: stg/yp (no int2e_ prefix)"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-runtime/src/options.rs
    - crates/cintx-rs/src/builder.rs
    - crates/cintx-rs/src/api.rs
    - crates/cintx-rs/src/error.rs

key-decisions:
  - "manifest canonical_family for STG/YP operators is '2e' not 'f12'; F12 detection must use symbol prefix"
  - "launch_f12 passes 'f12' explicitly to validate_f12_env_params since canonical_family is '2e'"
  - "operator_name() returns 'stg'/'yp' (not 'int2e_stg'/'int2e_yp'); dispatcher strips 'stg'/'yp' prefix for variant suffix"
  - "InvalidEnvParam added to FacadeError From<cintxRsError> as Validation kind"

patterns-established:
  - "Raw compat F12 path: detect by operator_symbol() prefix, extract env[PTR_F12_ZETA], validate, then dispatch"
  - "Safe API F12 path: f12_zeta() builder -> ExecutionOptions.f12_zeta -> propagated in SessionQuery to operator_env_params"

requirements-completed: [F12-01, F12-02, F12-04]

# Metrics
duration: 90min
completed: 2026-04-05
---

# Phase 13 Plan 02: F12 Kernel Implementation and PTR_F12_ZETA Wiring Summary

**10 F12/STG/YP kernel entry points with shared VRR/HRR pipeline, and PTR_F12_ZETA=9 wired through both raw compat (env[9]) and safe API (ExecutionOptions.f12_zeta) paths with typed InvalidEnvParam validation**

## Performance

- **Duration:** ~90 min (cross-session)
- **Started:** 2026-04-05
- **Completed:** 2026-04-05
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Replaced 24-line `launch_f12` stub with ~1100-line full F12 kernel implementing 10 entry points (5 STG + 5 YP variants) using `stg_roots_host` and the shared 2e VRR/HRR pipeline
- Wired `PTR_F12_ZETA` (env[9]) through the raw compat path: `eval_raw` detects F12 symbols by `operator_symbol()` prefix, extracts `env[9]`, sets `operator_env_params.f12_zeta`, and calls `validate_f12_env_params` before dispatch
- Wired `f12_zeta` through the safe API path: `ExecutionOptions.f12_zeta` field + `SessionBuilder.f12_zeta()` method + propagation in `SessionQuery` to `operator_env_params`

## Task Commits

Each task was committed atomically:

1. **Task 1: F12 kernel implementation** - `8071ef3` (feat)
2. **Task 2: PTR_F12_ZETA wiring** - `c0ec466` (feat)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/f12.rs` - Full F12 kernel: F12Shape, f12_kernel_core, 10 entry point functions, launch_f12 dispatcher, 3 smoke tests
- `crates/cintx-compat/src/raw.rs` - PTR_F12_ZETA=9 constant, env[9] extraction in eval_raw, 2 tests (gated with-f12)
- `crates/cintx-runtime/src/options.rs` - f12_zeta: Option<f64> field on ExecutionOptions
- `crates/cintx-rs/src/builder.rs` - f12_zeta() builder method on SessionBuilder; 1 test
- `crates/cintx-rs/src/api.rs` - f12_zeta propagation from options to operator_env_params in SessionQuery
- `crates/cintx-rs/src/error.rs` - InvalidEnvParam variant added to FacadeError From<cintxRsError> (auto-fix)

## Decisions Made
- F12 symbol detection in raw.rs uses `operator_symbol()` (full symbol like "int2e_stg_sph") not `operator_name()` (which returns "stg") or `canonical_family` (which is "2e"), because `is_f12_family_symbol` checks "int2e_stg"/"int2e_yp" prefixes
- `launch_f12` passes "f12" explicitly to `validate_f12_env_params` rather than relying on `canonical_family` ("2e")
- `launch_f12` strips "stg"/"yp" prefix from `operator_name()` to get variant suffix rather than stripping "int2e_stg"/"int2e_yp" from the full symbol name

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FacadeError missing InvalidEnvParam variant coverage**
- **Found during:** Task 2 (PTR_F12_ZETA wiring)
- **Issue:** `From<cintxRsError> for FacadeError` had a non-exhaustive match — `InvalidEnvParam` added in Plan 13-01 was not covered, causing a compile error
- **Fix:** Added `cintxRsError::InvalidEnvParam { param, reason } => Self::Validation { ... }` arm
- **Files modified:** `crates/cintx-rs/src/error.rs`
- **Verification:** `cargo test -p cintx-rs` compiles and all 12 tests pass
- **Committed in:** `c0ec466` (Task 2 commit)

**2. [Rule 1 - Bug] launch_f12 dispatcher used wrong operator_name prefix for detection**
- **Found during:** Task 2 (test validation revealed zero-zeta test passing when it should fail)
- **Issue:** `launch_f12` checked `operator_name.starts_with("int2e_stg")` but `operator_name()` returns "stg" (no "int2e_" prefix). The STG/YP branch was never entered; the unrecognized operator error path would be hit instead.
- **Fix:** Changed detection to `operator_name.strip_prefix("stg")` / `strip_prefix("yp")` so variant suffix is derived correctly from the actual operator_name format
- **Files modified:** `crates/cintx-cubecl/src/kernels/f12.rs`
- **Verification:** All 105 cubecl tests pass; zero-zeta test correctly returns InvalidEnvParam
- **Committed in:** `c0ec466` (Task 2 commit)

**3. [Rule 1 - Bug] launch_f12 called validate_f12_env_params with wrong canonical_family**
- **Found during:** Task 2 (traced root cause of dispatcher issue)
- **Issue:** `validate_f12_env_params(plan.descriptor.entry.canonical_family, ...)` used "2e" (the actual manifest value) — the validator only checks when `canonical_family == "f12"`, so validation was silently skipped
- **Fix:** Changed to pass "f12" explicitly: `validate_f12_env_params("f12", &plan.operator_env_params)`
- **Files modified:** `crates/cintx-cubecl/src/kernels/f12.rs`
- **Verification:** zero-zeta test returns InvalidEnvParam
- **Committed in:** `c0ec466` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3x Rule 1 bugs)
**Impact on plan:** All three fixes were prerequisite for correct F12 dispatch. No scope creep. Deviations 2 and 3 were co-located in f12.rs from Task 1 and discovered during Task 2 test validation.

## Issues Encountered
- The `canonical_family` field for STG/YP operators in the compiled manifest is "2e" (the 2-electron family), not a dedicated "f12" family. This affects F12 symbol detection in two places: raw.rs eval_raw and the launch_f12 dispatcher. Both were fixed to detect by symbol/operator_name prefix instead.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- F12 kernel dispatch is complete end-to-end for all 10 entry points
- Both API paths (raw compat and safe API) correctly wire PTR_F12_ZETA to operator_env_params
- Ready for oracle comparison testing (Phase 13 UAT) which will validate actual integral values against upstream libcint
- F12 oracle fixtures and comparison harness are the natural next step

---
*Phase: 13-f12-stg-yp-kernels*
*Completed: 2026-04-05*

## Self-Check: PASSED
- FOUND: crates/cintx-cubecl/src/kernels/f12.rs
- FOUND: crates/cintx-compat/src/raw.rs
- FOUND: crates/cintx-rs/src/builder.rs
- FOUND: .planning/phases/13-f12-stg-yp-kernels/13-02-SUMMARY.md
- FOUND: commit 8071ef3 (Task 1)
- FOUND: commit c0ec466 (Task 2)
