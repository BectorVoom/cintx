---
phase: quick-260529-imi
plan: 01
subsystem: cubecl
tags: [cubecl, one-electron, overlap, kinetic, nuclear-attraction, device-kernel, rocm, generic-float]

# Dependency graph
requires:
  - phase: quick-260529-r2g
    provides: center_2c2e generic-float #[cube(launch)] device-kernel + run_::<R> + on-backend dispatch template
  - phase: quick-260529-i2q
    provides: f12 base-Cartesian #[cube(launch)] + run_on_backend dispatch pattern
provides:
  - one_electron_scalar_kernel<F: Float + CubeElement> #[cube(launch)] device kernel for scalar 1e overlap/kinetic/nuclear-attraction
  - run_1e_scalar_device<R: Runtime> + 5-arm run_1e_scalar_on_backend dispatch (Cpu/Wgpu/Cuda/Rocm-HipRuntime/Metal)
  - one_electron_random_rocm_parity.rs randomized ROCm idempotency oracle for int1e_{ovlp,kin,nuc}_sph
affects: [phase-23-group-1-1st-deriv, gradient-1e-device-port, spinor-1e-device-port]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "comptime op_kind (0=ovlp/1=kin/2=nuc) selects the operator branch at JIT specialization time, mirroring 2c2e's comptime nroots"
    - "per-axis #[cube] sub-block helpers (one_electron_vrr_axis / vrr2e_axis / hrr_axis / kin_d2) take an explicit base offset since vrr_step/hrr_step only operate from index 0"
    - "pi_const passed as a runtime F scalar (not F::new) so the f64 path keeps full PI precision (F::new only accepts f32)"

key-files:
  created:
    - crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs

key-decisions:
  - "Scalar-at-minimum scoping (no silent narrowing): only overlap/kinetic/nuclear ported on-device; gradient (ipovlp/ipkin/ipnuc/iprinv) + spinor arms stay host-side with an explicit in-code rationale at the scalar/gradient fork, matching how 3c2e/ECP staged their device-kernel ports"
  - "Kernel computes pair data (zeta/fac/aij2/P) IN-KERNEL in F so the whole arithmetic core is genuinely generic over F (host compute_pdata returns f64-typed PairData which would force f64)"
  - "Explicit `let mut irys: u32 = 0u32` type annotation required in the nuclear root loop — CubeCL macro inference fails on a bare `0u32` whose only use is the loop bound when the loop body is complex (NativeExpand E0283)"
  - "MAX_DEVICE_NROOTS=5 guard on the nuclear arm (li+lj<=8) mirrors the 2c2e device kernel; H2O/STO-3G stays well within nroots<=5"

patterns-established:
  - "Operator-comptime device-kernel port: a single #[cube(launch)] kernel handles multiple related operators via #[comptime] op_kind, keeping the host scalar/gradient fork as the dispatch boundary"

requirements-completed: []

# Metrics
duration: 24min
completed: 2026-05-29
---

# Phase quick-260529-imi Plan 01: Refactor one_electron.rs scalar arms to a CubeCL device kernel Summary

**Scalar 1e overlap/kinetic/nuclear-attraction now compute through a generic-float `#[cube(launch)]` device kernel (`one_electron_scalar_kernel<F>`) dispatched at f64 on the resolved backend including ROCm HipRuntime, verified mismatch_count=0 over 48 random H2O cases on a gfx1152 GPU.**

## Performance

- **Duration:** ~24 min
- **Tasks:** 2
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments
- Ported the three scalar 1e operators (overlap, kinetic, nuclear-attraction) from the host VRR/HRR/contraction pipeline onto a single `#[cube(launch)]` device kernel generic over `F: Float + CubeElement`, with `#[comptime] op_kind` + `#[comptime] nroots` operator/Rys selection — exactly the proven center_2c2e.rs template.
- Added `run_1e_scalar_device<R: Runtime>` (f64-internal, contraction-major / bra-fastest output readback) and the 5-arm `run_1e_scalar_on_backend` dispatch (Cpu/Wgpu/Cuda/Rocm-`HipRuntime`/Metal), wired into the scalar path of `launch_one_electron_typed`.
- Kept the gradient (ipovlp/ipkin/ipnuc/iprinv) and spinor 1e arms host-side with an explicit in-code rationale at the scalar/gradient fork (no silent narrowing).
- Added a randomized ROCm idempotency oracle (`int1e_{ovlp,kin,nuc}_sph` twice per random H2O/STO-3G case via `eval_raw`) and RAN it on real ROCm hardware: **mismatch_count=0 across 48 cases, any_nonzero=true**.

## Task Commits

1. **Task 1: Port scalar 1e operators to a #[cube(launch)] device kernel** - `23eb85d` (feat)
2. **Task 2: Add and RUN the randomized ROCm oracle parity test** - `e400b26` (test)

_Note: Task 1 is the TDD task — the device-vs-host cross-check + f32 genericity tests were written and made to pass within the single feat commit (the existing f64-dispatch + general-contraction parity tests serve as the regression gate)._

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/one_electron.rs` - Added `one_electron_scalar_kernel` `#[cube(launch)]` + per-axis `#[cube]` helpers (`one_electron_vrr_axis`, `one_electron_vrr2e_axis`, `one_electron_hrr_axis`, `one_electron_kin_d2`), `run_1e_scalar_device<R>`, `run_1e_scalar_on_backend`; rewired the scalar path of `launch_one_electron_typed` to dispatch the device kernel; gated host `contract_overlap`/`contract_kinetic`/`contract_nuclear` as `#[cfg(test)]` cross-check references; added device-vs-host cross-check tests (ovlp/kin/nuc) and the f32 genericity smoke test.
- `crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs` - New `#![cfg(feature = "rocm")]` + `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`-gated randomized idempotency oracle driving `int1e_{ovlp,kin,nuc}_sph` twice per random H2O/STO-3G case via `eval_raw`.

## Decisions Made
- **Scalar-at-minimum scoping (explicit, not silent narrowing):** This task ports only the three scalar operators on-device. The four gradient kernels (each a distinct nabla1i/D_j^2 mixing pipeline with +1/+2 angular headroom) and the spinor transform keep their host code paths and continue to pass their existing tests; the on-device port of those arms is deferred to a follow-up quick task. The rationale is recorded in code at the scalar/gradient fork.
- **In-kernel pair data in `F`:** The host `compute_pdata` returns an f64-typed `PairData`; calling it would force f64 even on the f32 monomorphization. The kernel recomputes zeta/fac/aij2/P inline in `F` (same approach 2c2e uses for its products) so the arithmetic core is genuinely generic.
- **`pi_const` as a runtime scalar:** `F::new` only accepts an `f32`, so passing PI through `F::new(std::f64::consts::PI)` would truncate the f64 path. PI is passed as a runtime `F` scalar argument (like `sqrtpi`/`pie4`), preserving full f64 precision.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Explicit `u32` type annotation on the nuclear root loop counter**
- **Found during:** Task 1 (device kernel implementation)
- **Issue:** The `#[cube(launch)]` macro failed with E0282/E0283 (`cannot infer type for NativeExpand<_>`) on the nuclear-arm `let mut irys = 0u32;` whose only use is the `while irys < nrys` bound — the macro could not infer the type inside the deeply-nested comptime/atom/Rys loop body.
- **Fix:** Annotated explicitly as `let mut irys: u32 = 0u32;`. Also hoisted `let nrys = nroots;` to the top of the kernel body (mirroring 2c2e) so the comptime `nroots` is compared against a runtime binding.
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Verification:** `cargo build -p cintx-cubecl --features cpu` compiles clean; all 28 `kernels::one_electron` tests pass.
- **Committed in:** `23eb85d` (Task 1 commit)

**2. [Rule 3 - Blocking] `pi_const` passed as a runtime scalar instead of `F::new`**
- **Found during:** Task 1 (device kernel implementation)
- **Issue:** `F::new(std::f64::consts::PI)` failed to compile (`expected f32, found f64`) — CubeCL `F::new` takes an `f32`, which would also silently truncate PI on the f64 device path.
- **Fix:** Added a `pi_const: F` runtime kernel argument and pass `std::f64::consts::PI` (full f64) at the launch site, `std::f64::consts::PI as f32` in the f32 genericity test.
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Verification:** Compiles clean; nuclear device-vs-host cross-check passes at atol=1e-12/rtol=1e-10.
- **Committed in:** `23eb85d` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking compile issues). Both are CubeCL-authoring mechanics (macro type inference + f32-only `F::new`), resolved exactly as prior device-kernel ports handle them. No scope creep.

## Issues Encountered
- `cargo fmt -p cintx-cubecl` reformatted the entire crate (cosmetic whitespace across ~33 unrelated files). To keep the Task 1 commit atomic (code changes only), the fmt-only noise in files I did not intend to change was reverted via `git restore`; only `one_electron.rs` was committed.
- Project-wide `cargo clippy -p cintx-cubecl --features cpu` reports 42 pre-existing errors (e.g. `approx_constant` on physics constants like `PIE4`/`FRAC_1_SQRT_2`, `excessive precision` on coefficient tables). My added `PIE4`/`SQRTPI` constants mirror the identical accepted pattern in `center_2c2e.rs:65` (same `approx_constant` lint in the baseline). These are out-of-scope pre-existing lint categories, not introduced regressions; clippy is not a clean gate in this crate.

## Verification
- `cargo test -p cintx-cubecl --features cpu --lib kernels::one_electron` — 28 passed (device-vs-host ovlp/kin/nuc cross-check, f32 genericity, f64 dispatch, general-contraction parity).
- `cargo test -p cintx-oracle --features cpu --test one_electron_parity` — 3 passed (H2O STO-3G ovlp/kin/nuc through the new device path).
- `cargo test -p cintx-oracle --features rocm --no-run` — new oracle test compiles under rocm; `cargo build -p cintx-oracle` confirms it is a no-op without rocm.
- **`cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`** — RAN on ROCm gfx1152 hardware; `test_int1e_scalar_random_rocm_idempotency ... ok`, explicit print: `PASS: rocm int1e_{ovlp,kin,nuc}_sph random idempotency mismatch_count=0 across 48 cases (any_nonzero=true) at atol=1e-12/rtol=1e-10`.

## Next Phase Readiness
- Scalar 1e family is now on the CubeCL device backend (CubeCL-primary-backend constraint satisfied for the scalar arms). The gradient and spinor 1e arms remain a documented follow-up device-kernel port.

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/one_electron.rs
- FOUND: crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs
- FOUND: .planning/quick/260529-imi-.../260529-imi-SUMMARY.md
- FOUND commit: 23eb85d (Task 1 - feat)
- FOUND commit: e400b26 (Task 2 - test)

---
*Phase: quick-260529-imi*
*Completed: 2026-05-29*
