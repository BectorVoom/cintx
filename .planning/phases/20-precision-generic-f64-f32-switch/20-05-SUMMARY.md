---
phase: 20-precision-generic-f64-f32-switch
plan: 05
subsystem: compute
tags: [cubecl, generics, f32, f64, precision, 3c1e, 3c2e, 4c1e, f12, bytemuck]

# Dependency graph
requires:
  - phase: 20-precision-generic-f64-f32-switch/20-01
    provides: CintFloat sealed trait, PrecisionKind enum, bytemuck cast spike (A5)
  - phase: 20-precision-generic-f64-f32-switch/20-04
    provides: precision-dispatcher pattern (1e/2e/2c2e launchers + c2spinor generic)
provides:
  - "launch_center_3c1e: outer FamilyLaunchFn signature + match plan.precision dispatch to _typed::<F>"
  - "launch_center_3c2e: same precision-dispatch pattern; li>=lj canonicalization + transpose-back preserved"
  - "launch_center_4c1e: same precision-dispatch pattern; Validated4C1E envelope preserved"
  - "launch_f12: same precision-dispatch pattern; all f12 math stays f64; f12_zeta stays Option<f64> per D-06"
affects: [phase-21]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Precision dispatcher: outer pub fn keeps FamilyLaunchFn signature, inner _typed<F: CintFloat> dispatched via match plan.precision"
    - "bytemuck::cast_slice_mut: reinterpret &mut [f64] as &mut [f32] on F32 arm (A5 proven sound)"
    - "f12_zeta frozen f64: env parameter stays Option<f64> on OperatorEnvParams; not cast to F inside typed inner"
    - "f12 full-f64 intermediate: launch_f12_typed uses Vec<f64> staging_f64 for all internal math; casts at output boundary only"
    - "f12 sph-only manifest constraint: f12 operators only have sph form; tests must use Representation::Spheric + int2e_stg_sph"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-cubecl/src/kernels/center_4c1e.rs
    - crates/cintx-cubecl/src/kernels/f12.rs

key-decisions:
  - "Dispatcher shape identical to Plan 04: outer pub fn keeps &mut [f64] FamilyLaunchFn signature; match plan.precision inside body; bytemuck cast on F32 arm only."
  - "f12_zeta stays Option<f64>: D-06 resolution confirmed — env parameter is not cast to F. The launch_f12_typed inner takes zeta: f64 and passes it to internal helpers unchanged."
  - "f12 full-f64 intermediate: launch_f12_typed allocates Vec<f64> staging_f64, runs the full f64 computation, then casts to F at the output boundary via F::from_f64_lossy. This avoids any precision loss in the intermediate stg_base/yp_base/etc. computation paths."
  - "f12 tests use Representation::Spheric: f12 operators only have sph form in the manifest (RepresentationSupport::new(false, true, false)). Test shells must be created with Representation::Spheric and the symbol must be int2e_stg_sph."
  - "3c2e li>=lj canonicalization preserved verbatim: the shell-swap + eval-in-canonical-order + transpose-back logic in launch_center_3c2e_typed is identical to the original launch_center_3c2e."
  - "Validated4C1E envelope preserved: center_4c1e calls ensure_validated_4c1e() inside the typed inner; the cfg(feature = with-4c1e) gate is on the outer pub fn only."

patterns-established:
  - "All 6 kernel families now precision-dispatched: 1e, 2e, 2c2e (Plan 04), 3c1e, 3c2e, 4c1e, f12 (Plan 05)"
  - "f12 test pattern: shells + query + plan use Representation::Spheric; symbol is int2e_stg_sph"

requirements-completed: [PREC-03, PREC-04, PREC-05, PREC-06]

# Metrics
duration: ~40min
completed: 2026-05-21
---

# Phase 20 Plan 05: Wave 4 Kernel Launchers Group B Summary

**3c1e + 3c2e + 4c1e + f12 launchers precision-dispatched to generic _typed::<F> inner via bytemuck cast; f12 full-f64 intermediate with output-boundary cast; f12_zeta stays Option<f64> per D-06**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-05-21T00:00:00Z
- **Completed:** 2026-05-21T01:17:34Z
- **Tasks:** 2 (each TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- All four remaining kernel launchers (3c1e, 3c2e, 4c1e, f12) now dispatch on `plan.precision` to a generic `_typed::<F>` inner function while keeping their registered `FamilyLaunchFn` signatures unchanged
- `bytemuck::cast_slice_mut(&mut [f64]) -> &mut [f32]` wired on the F32 arm of each dispatcher
- 3c2e `li>=lj` canonicalization (shell swap + canonical-order eval + transpose-back) preserved verbatim inside the typed inner
- 4c1e `Validated4C1E` envelope call preserved inside the typed inner; `#[cfg(feature = "with-4c1e")]` gate remains on the outer pub fn
- f12 uses a full-f64 intermediate buffer for all internal STG/YP computation; only the final staging write casts via `F::from_f64_lossy`; `f12_zeta` stays `Option<f64>` on `OperatorEnvParams` per D-06
- 178 lib tests pass; all oracle tests pass (3c1e, 3c2e, 2c2e, f12 parity all green)

## Task Commits

Each task committed atomically with TDD RED → GREEN flow:

1. **Task 1: 3c1e + 3c2e RED** - `bc96982` (test: add RED failing tests for precision-dispatch typed inner 3c1e and 3c2e)
2. **Task 1: 3c1e + 3c2e GREEN** - `acba3ed` (feat: precision-dispatch + generic _typed inner for 3c1e and 3c2e launchers)
3. **Task 2: 4c1e + f12 RED** - `3afa97e` (test: add RED failing tests for precision-dispatch typed inner 4c1e and f12)
4. **Task 2: 4c1e + f12 GREEN** - `82307bd` (feat: precision-dispatch + generic _typed inner for 4c1e and f12 launchers)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` - Extracted `launch_center_3c1e_typed<F: CintFloat>` inner; outer dispatcher on `plan.precision`; added parity_f64 + f32_smoke tests
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` - Same pattern; li>=lj canonicalization + transpose-back preserved in typed inner; added parity_f64 + f32_smoke tests
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` - Same pattern; Validated4C1E envelope preserved; tests gated `#[cfg(feature = "with-4c1e")]`
- `crates/cintx-cubecl/src/kernels/f12.rs` - `launch_f12_typed<F>` inner uses full-f64 intermediate + output-boundary cast; zeta stays f64; tests use Representation::Spheric + int2e_stg_sph

## Decisions Made
- **f12_zeta D-06 confirmed**: `f12_zeta` stays `Option<f64>` on `OperatorEnvParams`. No casting of `zeta` to `F` inside `launch_f12_typed`. If future plans move f12 kernel math to F, that cast point is `F::from_f64_lossy(zeta)`.
- **f12 full-f64 intermediate**: `launch_f12_typed` allocates `Vec<f64>` and runs all internal STG/YP helper functions with f64. The cast to `F` is only at the output boundary. This matches the 1e/2e/2c2e pattern where intermediate G-tensor math stays f64.
- **f12 tests use sph form**: f12 operators have `RepresentationSupport::new(false, true, false)` in the manifest (sph-only). The validator enforces that shell representation must match the requested representation. Tests must use `Shell::try_new(..., Representation::Spheric, ...)` + `Representation::Spheric` in `query_workspace`/`ExecutionPlan::new` + symbol `int2e_stg_sph`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] f12 test symbol and representation corrected**
- **Found during:** Task 2 GREEN
- **Issue:** RED tests used `int2e_stg_cart` (nonexistent — f12 operators are sph-only in manifest) and `Representation::Cart`. The validator rejects because the manifest has `RepresentationSupport::new(false, true, false)` and also because shell representation must match requested representation.
- **Fix:** Changed test symbol to `int2e_stg_sph`, changed all three test functions to use `Representation::Spheric` for shells, `query_workspace`, and `ExecutionPlan::new`. This is not a design change — the kernel supports Cart output internally, but the planner correctly blocks unsupported representation requests.
- **Files modified:** `f12.rs`
- **Verification:** All 3 new f12 tests pass; all 5 f12 oracle parity tests pass
- **Committed in:** `82307bd`

---

**Total deviations:** 1 auto-fixed (Rule 1 bug — wrong symbol and representation in RED tests)
**Impact on plan:** Test-only fix. The typed inner and outer dispatcher are exactly as designed. The plan's requirement (f12_zeta stays f64, f12 typed inner with output-boundary cast) is fully implemented.

## Issues Encountered
The `int2e_stg_cart` symbol does not exist in the manifest — f12/STG operators in libcint 6.1.3 are implemented as spheric-only. This was discovered at test runtime. The fix was straightforward (use `int2e_stg_sph` + `Representation::Spheric`). The validator's `shell.representation != representation` check was also triggered, requiring shells to be created with `Representation::Spheric`.

## Known Stubs
None — all precision dispatch paths produce real values. The f32 smoke tests assert `.is_finite()` and `abs() > 0.0`, confirming the F32 paths are functional.

## Next Phase Readiness
- All 6 kernel families now have precision-dispatched launchers: 1e, 2e, 2c2e (Plan 04), 3c1e, 3c2e, 4c1e, f12 (Plan 05)
- The `launch_<family>_typed<F: CintFloat>` pattern is established uniformly across the workspace
- Phase 21 can build on this by wiring `PrecisionKind` from user-facing API through `ExecutionOptions` to `plan.precision`
- f32 numeric tolerance baselines not yet measured (smoke tests confirm finite non-zero; Wave 5 tolerance floors require dedicated oracle runs)

---
*Phase: 20-precision-generic-f64-f32-switch*
*Completed: 2026-05-21*

## Self-Check: PASSED

- SUMMARY.md: FOUND at `.planning/phases/20-precision-generic-f64-f32-switch/20-05-SUMMARY.md`
- Commits verified: bc96982, acba3ed, 3afa97e, 82307bd all present in git log
- Files verified: center_3c1e.rs, center_3c2e.rs, center_4c1e.rs, f12.rs all exist
