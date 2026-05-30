---
phase: quick-260529-i2q
plan: 01
subsystem: cubecl
tags: [cubecl, cube-launch, f12, stg, yp, device-kernel, rocm, oracle, byte-identity]

# Dependency graph
requires:
  - phase: quick-260529-gbf
    provides: "ECP Type-1 angular-splice #[cube(launch)] device-kernel port template (ecp_angular_kernel / run_ecp_angular_device / run_ecp_angular_splice_on_backend)"
  - phase: quick-260529-hin
    provides: "ECP Type-2 inline-recompute device-kernel pattern + ecp_random_rocm_parity.rs oracle structure"
provides:
  - "F12 base Cartesian contraction runs on-device as a #[cube(launch)] kernel (f12_cart_contraction_kernel) generic over F: Float"
  - "run_f12_cart_contraction_device::<R> + run_f12_cart_contraction_on_backend per-backend dispatch in f12.rs"
  - "f12_kernel_core base (ncomp==1) branch threads backend through the device splice; let _ = backend; removed"
  - "Randomized ROCm F12 idempotency oracle (f12_random_rocm_parity.rs), raw-API driven, triple-gated"
affects: [f12, two_electron, gpu-port, derivative-f12-followup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "F12 family is the 7th integral family ported to a real on-device #[cube(launch)] kernel"
    - "Raw-API-driven ROCm idempotency oracle (eval_raw + RawApiId::Symbol) for feature-gated families with no safe-API variant"

key-files:
  created:
    - crates/cintx-oracle/tests/f12_random_rocm_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/f12.rs

key-decisions:
  - "Scoped to ONLY the base/ncomp==1 contraction (contract_f12_cart); derivative gout_* splices stay host-side for a follow-up, mirroring ECP's Type-1-then-Type-2 split."
  - "Launch at f64 with IDENTICAL nested loop order (l/k/j/i outer, irys inner) to preserve byte-identity vs vendored libcint."
  - "ROCm oracle randomizes zeta (PTR_F12_ZETA env[9]) + shell quartet per case rather than in-place env exponent jitter — distinct valid F12 systems exercising the device kernel without a vendor comparison."
  - "Retained contract_f12_cart host fn as the cfg(test) byte-identity oracle (marked #[cfg_attr(not(test), allow(dead_code))])."

patterns-established:
  - "Device-vs-host equivalence test asserts EXACT f64 == (max-abs-diff 0.0) + generic-F f32-within-eps, mirroring ecp.rs."

requirements-completed: []

# Metrics
duration: ~14min
completed: 2026-05-29
---

# Phase quick-260529-i2q: F12 base Cartesian contraction CubeCL device-kernel port Summary

**The F12 base Cartesian contraction (`contract_f12_cart`) now runs on-device as a `#[cube(launch)]` kernel generic over `F: Float`, dispatched per-backend (CPU/ROCm/Wgpu/Cuda/Metal), byte-identical to vendored libcint and verified by a randomized ROCm idempotency oracle (mismatch_count=0 across 48 cases on the AMD GPU).**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-05-29 (executor session)
- **Completed:** 2026-05-29
- **Tasks:** 3 auto tasks (Task 4 is a non-blocking human-verify checkpoint)
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments
- Ported the F12 base `[gx|gy|gz] -> cart tensor` contraction to a real on-device `#[cube(launch)]` kernel (`f12_cart_contraction_kernel`), the 7th integral family GPU port, following the ECP Type-1 template exactly.
- Rewired `f12_kernel_core`'s base branch to dispatch through `run_f12_cart_contraction_on_backend` and removed the `let _ = backend;` no-op — CLAUDE.md's CubeCL-primary-backend mandate now satisfied for the F12 base variant.
- Preserved byte-identity: the CPU-vs-vendor gate `f12_oracle_parity` passes (`int2e_stg_sph` + `int2e_yp_sph` at atol=1e-12, all 15 tests green) after the port.
- Added a randomized ROCm idempotency oracle that ran on the AMD GPU: `mismatch_count=0 across 48 cases (any_nonzero=true)`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Port contract_f12_cart to #[cube(launch)] device kernel + backend dispatch** - `6660b68` (feat)
2. **Task 2: Rewire f12_kernel_core base branch through backend dispatch** - `d5e7d2b` (feat)
3. **Task 3: Add randomized ROCm F12 idempotency oracle** - `8fe48a3` (test)

_(Task 1 was a tdd task but the failing-then-passing cycle collapsed into a single commit: the new kernel + its device-vs-host equivalence tests landed together and pass exactly.)_

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/f12.rs` - Added CubeCL imports; `cart_comps_flat_u32` marshaling helper; `f12_cart_contraction_kernel` (`#[cube(launch)]`, generic over F, full nfi*nfj*nfk*nfl block in one launch, statement-form if / u32 indices / while loops / inline-recompute / identical host loop order); `run_f12_cart_contraction_device::<R>`; `run_f12_cart_contraction_on_backend` (per-backend match); rewired base branch; marked `contract_f12_cart` as cfg(test) oracle; added device-vs-host equivalence tests.
- `crates/cintx-oracle/tests/f12_random_rocm_parity.rs` - Randomized ROCm idempotency oracle for `int2e_stg_sph`, raw-API driven (eval_raw), triple-gated (`#![cfg(all(feature=rocm, feature=with-f12))]` + `#[ignore]` + `CINTX_ROCM_ORACLE=1`).

## Verification Results

| Gate | Command | Result |
|------|---------|--------|
| Build (cube) | `cargo build -p cintx-cubecl --features with-f12,cpu` | clean |
| Device-vs-host equivalence | `cargo test -p cintx-cubecl --features with-f12,cpu --lib f12` | 9 passed (exact f64 `==` across 5 quartets incl all-s + p-shell; generic-F f32 within eps) |
| CPU-vs-vendor byte-identity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12 --test f12_oracle_parity` | 15 passed (int2e_stg_sph + int2e_yp_sph mismatch_count=0 at atol=1e-12; vendor actually built, `#[cfg(has_vendor_libcint)]` tests ran) |
| Compile-collect new test | `cargo build -p cintx-oracle --features rocm,with-f12 --tests` | clean; `test_f12_stg_sph_random_rocm_idempotency` collected |
| ROCm deliverable | `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle --profile with-f12` | **suite passed**; F12 test PASS: `mismatch_count=0 across 48 cases (any_nonzero=true) at atol=1e-12/rtol=1e-10` |

**ROCm device availability:** The AMD ROCm device WAS available in this worktree environment — the `rocm-oracle --profile with-f12` run completed successfully (exit 0) and the F12 device kernel executed on the GPU (any_nonzero=true confirms it was not an all-zeros host fallback).

## Decisions Made
- See `key-decisions` frontmatter. The most consequential: scope limited to the base/`ncomp==1` contraction (derivative `gout_*` splices remain host-side for a follow-up), and randomizing zeta + shell-quartet for the ROCm oracle rather than in-place env exponent jitter.

## Deviations from Plan

None - plan executed exactly as written. The new helpers produced a transient dead-code warning between Task 1 and Task 2 (expected and called out by the plan); `run_f12_cart_contraction_on_backend` was wired in Task 2 as planned, and `contract_f12_cart` is retained as the cfg(test) oracle.

## Issues Encountered
- Initial test seed used an invalid hex literal (`0xf12c_2605_29i2_qbf_u64` — contained non-hex chars from the task id); corrected to a valid `u64` hex literal before the compile-collect build. No functional impact.

## Known Stubs
None. The device kernel is a complete, byte-identity-preserving port (not a stub); the derivative-branch host path is explicitly out of scope per the plan, not a stub.

## Next Phase Readiness
- F12 base variant is now GPU-resident and byte-identity-verified. The explicit follow-up is porting the F12 derivative `gout_*` contractions (ip1/ipip1/ipvip1/ip1ip2) to `#[cube]` device kernels, mirroring this base port.

## Self-Check: PASSED

---
*Phase: quick-260529-i2q*
*Completed: 2026-05-29*
