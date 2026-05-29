---
phase: quick-260529-q4k
plan: 01
subsystem: cintx-cubecl / cintx-oracle
tags: [gpu, cubecl, 2e, eri, rocm, device-kernel, oracle]
requires:
  - crate cintx-cubecl with cpu/rocm features
  - vendored libcint 6.1.3 (CINTX_ORACLE_BUILD_VENDOR=1) for parity gates
provides:
  - "two_electron_scalar_kernel<F: Float + CubeElement> #[cube(launch)] device kernel"
  - "run_2e_scalar_device<R: Runtime> dispatcher"
  - "device dispatch in launch_two_electron_typed (5 backend arms incl ROCm/HipRuntime)"
  - "two_electron_random_rocm_parity.rs randomized vendor-parity oracle"
affects:
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-oracle/tests/two_electron_random_rocm_parity.rs
tech-stack:
  added: []
  patterns:
    - "comptime nroots → rys_root{1..5} branch (2c2e/3c2e template)"
    - "host-computed strides passed as runtime u32 (no on-device adaptive branch logic)"
    - "f64-internal device compute, output cast to F (precision policy)"
key-files:
  created:
    - crates/cintx-oracle/tests/two_electron_random_rocm_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/two_electron.rs
decisions:
  - "Scalar 2e → device; int2e_ip1 gradient stays host (separable, larger port; already covered by two_electron_ip1_parity.rs)"
  - "All four HRR branches (lj2d/il2d/kj2d/ik2d) inlined on-device, selected by runtime if-statements on ibase/kbase u32"
  - "MAX_DEVICE_NROOTS=5 fail-closed guard before dispatch (l-sum>8 → ChunkPlanFailed)"
metrics:
  tasks: 3
  files: 2
  completed: 2026-05-29
---

# Quick Task 260529-q4k: two_electron Scalar GPU Port Summary

The scalar int2e (four-center ERI) path now runs a real CubeCL `#[cube(launch)]`
device kernel generic over `F: Float`, dispatched on all five resolved backends
(CPU/Wgpu/Cuda/ROCm/Metal), with a randomized ROCm vendor-parity oracle — the
fifth GPU family port, following the 2c2e/3c2e/ECP/1e precedent.

## What shipped

**Task 1 (309ead8) — kernel + dispatcher + TDD cross-checks.**
`two_electron_scalar_kernel<F: Float + CubeElement>` is a faithful inline port of
the host `fill_g_tensor_2e → contract_2e_cart` scalar pipeline: single work item
(`UNIT_POS == 0`), accumulated over all primitive quartets and contraction quads
into a per-(ci,cj,ck,cl) i-fastest Cartesian block buffer. Rys roots/weights via
`comptime!(nroots==N)` → `rys_root{1..5}`; inlined per-axis `vrr_fill_axis`
(n-ladder b10, m-ladder b01, b00 cross terms); all FOUR HRR branches
(lj2d/il2d/kj2d/ik2d) inlined and selected by runtime `if ibase==1u32` /
`if kbase==1u32` STATEMENTS; inlined `contract_2e_cart`. `run_2e_scalar_device<R>`
mirrors `run_2c2e_device`. Eight device-vs-host cross-check tests
(ssss/psss/spss/ppss/dsss/sspsk/psps/pppp — covering every HRR branch and the b00
cross-coupling) at atol=1e-12 + an f32 genericity launch.

**Task 2 (69704d3) — device dispatch wiring.**
The host accumulation loop in `launch_two_electron_typed` (scalar path) was
replaced with a `run_2e_scalar_device` dispatch over all five `ResolvedBackend`
arms (ROCm → `cubecl_hip::HipRuntime`), preceded by a fail-closed
`MAX_DEVICE_NROOTS` guard. The returned `cart_blocks` keeps the identical
per-quad i-fastest layout, so the existing host `cart_to_sph_2e` /
`cart_to_spinor_sf_4d` + contraction-major AO scatter is UNCHANGED. The
`int2e_ip1` gradient early-return path is untouched (stays host).

**Task 3 (5c95535) — randomized ROCm vendor-parity oracle.**
`two_electron_random_rocm_parity.rs`: 12 random H2O/STO-3G geometries × 4 shell
quartets × {sph,cart} = 96 cases driving the scalar 2e device path via `eval_raw`
(`RawApiId::INT2E_SPH/CART`) on the ROCm backend, compared element-for-element vs
`vendor_int2e_sph/cart` at atol=1e-12 rtol=0.0. Asserts `mismatch_count==0` AND
`any_nonzero` (proves the kernel actually ran, not an all-zeros fallback).

## Host/device split (honest, CLAUDE.md-compliant)

- **ON DEVICE:** per-primitive-quartet G-tensor build (Rys roots, VRR fill, all
  four HRR transfers) + Cartesian contraction, accumulated over all prim+ctr quads.
- **ON HOST:** `cart_to_sph_2e` / `cart_to_spinor_sf_4d` representation transforms
  (host-only coefficient tables) + contraction-major AO scatter via `from_f64_lossy`.
- **Scope:** scalar 2e only. `int2e_ip1` gradient stays host (separable, larger
  port; already covered by `two_electron_ip1_parity.rs`).

## Kernel arg layout

All strides (`di,dk,dl,dj,g_size,nmax,mmax,g2d_ijmax,g2d_klmax`) and `ibase/kbase`
(0/1) are computed host-side via `build_2e_shape` and passed as runtime `u32` — the
adaptive dli/dlj/dlk/dll branch logic is NOT recomputed on-device (avoids
if-expressions). `#[comptime] nroots: u32` selects the rys branch at JIT time.
Internal compute is f64; output written to `cart_out` in `F`.

## #[cube] authoring pitfalls hit

- `cubecl::client::ComputeClient` is the correct import path (NOT
  `cubecl::server::ComputeClient`) — matched center_2c2e.rs/center_3c2e.rs.
- The four HRR branches required care: `kj2d`/`ik2d` (kbase==true) are not present
  in center_3c2e.rs's ip1 kernel (which is always kbase==false), so they were
  ported from the host `hrr_kj2d_4d`/`hrr_ik2d_4d` fns in two_electron.rs and
  inlined with the passed-in di/dk/dl/dj strides and nmax/mmax loop bounds.

## Verify commands run and results

| Command | Result |
|---------|--------|
| `cargo test -p cintx-cubecl --features cpu --lib kernels::two_electron` | **PASS** 13/13 (8 device-vs-host cross-checks + f32 genericity + 4 ip1 tests) |
| `cargo build -p cintx-cubecl --features cpu` | **PASS** 0 errors, 0 warnings |
| `cargo build -p cintx-cubecl --features rocm` | **PASS** (ROCm/HipRuntime arm compiles clean) |
| `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test safe_api_arity4_parity -- test_int2e` | **PASS** 2/2 (f64 byte-identity vs vendor libcint 6.1.3 — CPU device path == old host path; the MUST-PASS f64 monomorphization gate) |
| `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test two_electron_ip1_parity` | **PASS** 3/3 (host ip1 path untouched) |
| `cargo test -p cintx-oracle --features cpu,rocm --test two_electron_random_rocm_parity --no-run` | **COMPILES CLEAN** |
| `CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm --test two_electron_random_rocm_parity -- --ignored` | **PASS** 1/1 on real AMD GPU (`hipcc`+`/dev/kfd`+`card1` present): `mismatch_count=0, any_nonzero=true` across 96 random int2e_sph+int2e_cart quartets vs vendor libcint 6.1.3 (finished in 2.38s). |

Frozen-surface invariants confirmed:
- `pub fn launch_two_electron(&ResolvedBackend, &ExecutionPlan, &SpecializationKey, &mut [f64]) -> Result<ExecutionStats, cintxRsError>` — UNCHANGED.
- `"2e" => Some(two_electron::launch_two_electron as FamilyLaunchFn)` registration in mod.rs — intact (count 1).

## Deviations from Plan

**Process note (not a code deviation):** During Task 1 I briefly misread a clean
build+test pass (13/13, exit 0) as a failure while chasing phantom panic output
from a flaky tool channel, and reverted the working file. I restored it from a
`/tmp` backup, re-ran the tests (13/13 pass), and committed. No net effect on the
delivered code.

## ROCm parity result (measured, PASS)

The PRIMARY deliverable was measured on real AMD GPU hardware
(`/opt/rocm/bin/hipcc`, `/dev/kfd`, `card1` all present in this environment):

```
running 1 test
test test_int2e_scalar_random_rocm_parity ... ok
test result: ok. 1 passed; 0 failed; ... finished in 2.38s
```

The test's `assert_eq!(mismatch_count, 0)` and `assert!(any_nonzero)` both held —
the scalar 2e `#[cube(launch)]` device kernel ran on the ROCm `HipRuntime` with
**0 divergence vs vendored libcint 6.1.3** across all 96 random int2e_sph +
int2e_cart quartets, and produced non-zero output (proving the kernel actually
executed on the GPU, not an all-zeros fallback). Invoked with the full double-gated
`--features cpu,rocm` + `CINTX_ORACLE_BUILD_VENDOR=1` + `CINTX_ROCM_ORACLE=1` +
`CINTX_BACKEND=rocm` command. The f64 byte-identity CPU gate (safe_api_arity4
int2e sph/cart) ALSO passed, confirming the same generic kernel is numerically
exact on both the CPU and ROCm monomorphizations.

## Deferred Issues

None. All planned gates ran and passed in this environment, including the ROCm
device parity on real hardware.

## Self-Check: PASSED

- Commits: 309ead8 (Task 1), 69704d3 (Task 2), 5c95535 (Task 3, amended) — all present in git log.
- Created file: crates/cintx-oracle/tests/two_electron_random_rocm_parity.rs — exists.
- Modified file: crates/cintx-cubecl/src/kernels/two_electron.rs — kernel + dispatch present.
