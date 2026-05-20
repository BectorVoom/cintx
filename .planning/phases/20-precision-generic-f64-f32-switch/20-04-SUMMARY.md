---
phase: 20-precision-generic-f64-f32-switch
plan: 04
subsystem: compute
tags: [cubecl, generics, f32, f64, precision, spinor, c2spinor, bytemuck]

# Dependency graph
requires:
  - phase: 20-precision-generic-f64-f32-switch/20-01
    provides: CintFloat sealed trait, PrecisionKind enum, bytemuck cast spike (A5)
  - phase: 20-precision-generic-f64-f32-switch/20-02
    provides: generic obara_saika, pdata, stg over F: CintFloat
  - phase: 20-precision-generic-f64-f32-switch/20-03
    provides: generic rys roots, c2s cart-to-sph transforms
provides:
  - "launch_one_electron: outer FamilyLaunchFn signature + match plan.precision dispatch to _typed::<F>"
  - "launch_two_electron: same precision-dispatch pattern"
  - "launch_center_2c2e: same precision-dispatch pattern"
  - "cart_to_spinor_sf, cart_to_spinor_iket_sf, cart_to_spinor_si, cart_to_spinor_iket_si: generic over F: CintFloat"
  - "cart_to_spinor_sf_2d<F: CintFloat>: generic staging write via from_f64_lossy at output boundary"
  - "cart_to_spinor_sf_4d<F: CintFloat>: generic, uses ::<f64> for opij intermediate, F at final write"
  - "cart_to_spinor_sf_3c2e<F: CintFloat>: generic, passes F staging slices to cart_to_spinor_sf_2d"
affects: [20-05, phase-21]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Precision dispatcher: outer pub fn keeps FamilyLaunchFn signature, inner _typed<F: CintFloat> dispatched via match plan.precision"
    - "bytemuck::cast_slice_mut: reinterpret &mut [f64] as &mut [f32] on F32 arm (8-byte aligned, Pod)"
    - "Frozen f64 tables: CG coefficient tables in c2spinor stay f64; cast to F at accumulation write site only"
    - "Two-phase spinor genericization: intermediate Vec<f64> buffers stay f64; from_f64_lossy at final staging write"
    - "4D spinor intermediate: cart_to_spinor_sf_4d calls ::<f64> for opij step-1 buffer, ::<F> at final write"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-cubecl/src/kernels/center_2c2e.rs
    - crates/cintx-cubecl/src/transform/c2spinor.rs

key-decisions:
  - "Dispatch shape: outer pub fn launch_<family> keeps &mut [f64] FamilyLaunchFn signature; inner launch_<family>_typed<F: CintFloat> takes &mut [F]. Enables BackendExecutor / fn-pointer registry object-safety (Pitfall 3)."
  - "bytemuck cast on F32 arm only: F64 arm passes staging directly (zero-cost no-op); F32 arm calls bytemuck::cast_slice_mut(&mut [f64]) -> &mut [f32] (Plan 01 A5 proven sound)."
  - "CG coefficient tables FROZEN f64: only the final write to gsp: &mut [F] uses F::from_f64_lossy; all CG coefficient arithmetic stays in f64 for oracle parity."
  - "Step-1 opij intermediate in cart_to_spinor_sf_4d stays Vec<f64>: uses cart_to_spinor_sf_2d::<f64> for correctness; final staging write is the only F conversion point."
  - "Simplified kernel spinor arms: after c2spinor genericization, the temp Vec<f64> + element-by-element copy loops in 1e/2e/2c2e spinor arms were replaced with direct cart_to_spinor_sf_2d::<F>/4d::<F> calls."

patterns-established:
  - "Launcher precision dispatcher: copy this exact pattern for Plan 05 remaining kernels (3c2e, kinetic, etc.)"
  - "Spinor output always interleaved: staging[(j*di+i)*2]=re, +1=im — never num_complex::Complex<F> buffer"
  - "F32 smoke test pattern: assert staging_f32[i].is_finite() + count nonzero > 0 (no numeric literals needed)"

requirements-completed: [PREC-01, PREC-02, PREC-07]

# Metrics
duration: ~35min
completed: 2026-05-20
---

# Phase 20 Plan 04: Wave 3 Kernel Launchers Group A Summary

**1e/2e/2c2e launchers precision-dispatched to generic _typed::<F> inner via bytemuck cast; c2spinor transforms (sf/iket_sf/si/iket_si/2d/4d/3c2e) generic over F: CintFloat with CG tables frozen f64**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-20T21:40:00Z
- **Completed:** 2026-05-20T22:20:09Z
- **Tasks:** 2 (each TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Three kernel launchers (1e, 2e, 2c2e) now dispatch on `plan.precision` to a generic `_typed::<F>` inner function while keeping their registered `FamilyLaunchFn` signature unchanged — the fn-pointer registry and BackendExecutor object safety are preserved
- `bytemuck::cast_slice_mut(&mut [f64]) -> &mut [f32]` wired on the F32 arm of each dispatcher (Plan 01 A5 proven sound); F64 arm is zero-cost (no cast, passes staging directly)
- All seven c2spinor public functions made generic over `F: CintFloat`: `cart_to_spinor_sf`, `iket_sf`, `si`, `iket_si`, `sf_2d`, `sf_4d`, `sf_3c2e` — CG coefficient tables stay frozen `f64`, cast to `F` only at the final staging write via `F::from_f64_lossy`
- Simplified kernel spinor arms: replaced temp `Vec<f64>` + element-by-element copy loops with direct `cart_to_spinor_sf_2d::<F>` / `sf_4d::<F>` calls
- 163 lib tests pass; all oracle tests pass; full workspace compiles

## Task Commits

Each task was committed atomically with TDD RED → GREEN flow:

1. **Task 1: 1e + 2e RED** - `de0de12` (test: RED tests for precision-dispatch typed inner 1e)
2. **Task 1: 1e + 2e GREEN** - `1d73511` (feat: precision-dispatch + generic _typed inner for 1e and 2e launchers)
3. **Task 2: 2c2e RED** - `f44ecc9` (test: RED tests for precision-dispatch typed inner 2c2e)
4. **Task 2: 2c2e launcher GREEN** - `e88a051` (feat: precision-dispatch + generic _typed inner for center_2c2e launcher GREEN)
5. **Task 2: c2spinor RED** - `843c147` (test: RED failing tests for generic cart_to_spinor_sf_2d)
6. **Task 2: c2spinor GREEN** - `8abfbcd` (feat: genericize c2spinor transforms over F: CintFloat + simplify kernel callers GREEN)

_Note: Task 2 split into two RED/GREEN cycles — 2c2e launcher and c2spinor are separate concerns._

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/one_electron.rs` - Extracted `launch_one_electron_typed<F: CintFloat>` inner; outer dispatcher on `plan.precision`; simplified spinor arm to call `cart_to_spinor_sf_2d::<F>` directly
- `crates/cintx-cubecl/src/kernels/two_electron.rs` - Same pattern; spinor arm uses `cart_to_spinor_sf_4d::<F>`
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` - Same pattern; spinor arm uses `cart_to_spinor_sf_2d::<F>`
- `crates/cintx-cubecl/src/transform/c2spinor.rs` - Six public and four internal apply_* block functions made generic; CG tables frozen; `CintFloat` import added

## Decisions Made
- **Dispatch shape preserved**: outer `pub fn` keeps `&mut [f64]` because `FamilyLaunchFn` is a concrete fn-pointer type; making it generic would break object safety. The `match plan.precision` dispatch is hidden inside the outer body.
- **CG tables stay f64**: the CG coefficients in `apply_sf_block`, `apply_bra_block`, etc. are read from `&[f64]` static tables. All arithmetic accumulates in `f64`. Only the write `gsp[idx] = F::from_f64_lossy(acc)` uses `F`. This preserves bit-identical f64 output.
- **4D spinor step-1 intermediate**: `cart_to_spinor_sf_4d` computes a `Vec<f64> opij` intermediate via `cart_to_spinor_sf_2d::<f64>`. The final staging write is the only F conversion point. This avoids double-precision loss in the intermediate step.
- **Task 2 split into two TDD cycles**: center_2c2e launcher RED/GREEN committed separately from c2spinor RED/GREEN — each is a distinct concern with its own compilation gate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Enhancement] Simplified kernel spinor arms after c2spinor genericization**
- **Found during:** Task 2 GREEN (c2spinor genericization)
- **Issue:** The original spinor arms in 1e/2e/2c2e used `Vec<f64>` temp buffers + element-by-element copy loops because `cart_to_spinor_sf_2d` only accepted `&mut [f64]`. After making c2spinor generic, the temp buffers became unnecessary overhead.
- **Fix:** Replaced temp buffer pattern with direct `cart_to_spinor_sf_2d::<F>(staging, ...)` calls in all three kernel spinor arms.
- **Files modified:** `one_electron.rs`, `two_electron.rs`, `center_2c2e.rs`
- **Verification:** All 163 tests pass; workspace compiles
- **Committed in:** `8abfbcd` (c2spinor GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 enhancement — eliminated now-unnecessary indirection)
**Impact on plan:** Positive simplification. The temp buffer approach was the correct initial step before c2spinor was generic; removing it after the genericization is the correct cleanup.

## Issues Encountered
None — the plan's approach was correct. The key design judgment (outer fn with `&mut [f64]` + inner generic typed fn + bytemuck cast on F32 arm) compiled and worked on the first attempt for all three launchers.

## Known Stubs
None — all precision dispatch paths produce real values. The f32 smoke tests assert `.is_finite()` and `> 0.0` (or count nonzero), confirming the F32 path is functional.

## Next Phase Readiness
- Plan 05 (remaining kernels: 3c2e, ecp, kinetic, nuc, etc.) can copy the exact dispatcher shape established here: `fn launch_<family>(staging: &mut [f64]) { match plan.precision { F64 => typed::<f64>(staging), F32 => { let s32 = bytemuck::cast_slice_mut(staging); typed::<f32>(s32) } } }`
- The c2spinor 3c2e function is also generic (`cart_to_spinor_sf_3c2e<F>`) so Plan 05 3c2e launchers can call it directly with `&mut [F]` staging
- f32 error magnitude not independently measured here (the smoke tests confirm finite non-zero output; Wave 5 tolerance floors require dedicated oracle runs)

---
*Phase: 20-precision-generic-f64-f32-switch*
*Completed: 2026-05-20*

## Self-Check: PASSED

- SUMMARY.md: FOUND at `.planning/phases/20-precision-generic-f64-f32-switch/20-04-SUMMARY.md`
- Commits verified: de0de12, 1d73511, f44ecc9, e88a051, 843c147, 8abfbcd all present in git log
