---
quick_id: 260529-r2g
slug: center-2c2e-cubecl-gpu-kernel
date: 2026-05-29
status: complete
---

# Quick Task 260529-r2g — Summary

Refactored `crates/cintx-cubecl/src/kernels/center_2c2e.rs` from a host-only
f64 pipeline into a real CubeCL `#[cube(launch)]` device kernel generic over
`F: Float`, dispatched onto the resolved backend's `ComputeClient`, and added a
randomized ROCm idempotency oracle test. This is the **first integral family in
the project to actually execute on the GPU** (all families were previously
host-only; GPU conversion had been deferred to v1.4).

## What changed

- **`kernels/center_2c2e.rs`**
  - New `center_2c2e_kernel<F: Float + CubeElement>` (`#[cube(launch)]`): per
    shell-pair Rys G-tensor fill (base case + i-VRR + k-VRR + mixed i+k `b00`
    cross term) and Cartesian contraction, single work item (`UNIT_POS == 0`).
    `#[comptime] nroots` selects `rys_root{1..5}` via `comptime!` — no runtime
    nroots dispatch (avoids the documented MLIR lowering issue) and no Rust
    monomorphization fan-out (per the CubeCL fan-out manual).
  - New generic `run_2c2e_device::<R: Runtime>(…)`: creates device buffers,
    launches the kernel at f64, reads back the Cartesian buffer.
  - `launch_center_2c2e_typed` now dispatches on the `ResolvedBackend` arm
    (Cpu/Wgpu/Cuda/Rocm/Metal) into `run_2c2e_device` instead of computing
    host-side; c2s / spinor / cart transforms still finish on the host with the
    output cast to `F`. Added an `nroots <= 5` guard (device Rys coverage).
  - Kept `fill_g_tensor_2c2e` + `cart_comps` as `#[cfg(test)]` host references.
  - New tests: device-vs-host cross-check (ss/sp/pp/dd), f32 genericity launch,
    plus the pre-existing precision-dispatch tests now exercising the device path.
- **`crates/cintx-oracle/tests/center_2c2e_parity.rs`**
  - New `test_int2c2e_sph_random_rocm_idempotency` (`#[cfg(feature="rocm")]` +
    `#[ignore]` + `CINTX_ROCM_ORACLE=1`): a deterministic-LCG random suite of 64
    two-shell systems (random l∈{0,1,2}, nprim, exponents, coeffs, geometry) run
    twice on the ROCm device, asserting idempotency + non-zero output.

## Precision policy (preserved)

The kernel is genuinely generic over `F: Float`, but the launcher runs it at
**f64** on-device for both `PrecisionKind` variants and casts to `F` at the
output/c2s stage — preserving the historical "f64 intermediates, `F` output"
contract the f32 parity gate is calibrated against.

## Verification (all green)

| Gate | Result |
|------|--------|
| `cintx-cubecl --features cpu --lib` (211 tests) | PASS |
| device-vs-host cross-check ss/sp/pp/dd | PASS (exact) |
| f32 genericity kernel launch | PASS |
| `center_2c2e_parity` cpu vendor parity (atol 1e-9) | PASS |
| `safe_api_arity2` 2c2e cart+sph vendor parity (atol 1e-12) | PASS |
| `safe_api_arity2` 2c2e spinor idempotency | PASS |
| `f32_parity` `test_f32_int2c2e_sph_parity` | PASS |
| ROCm (gfx1152) `test_int2c2e_sph_h2o_sto3g_rocm_parity` | PASS |
| ROCm (gfx1152) `test_int2c2e_sph_random_rocm_idempotency` (64 cases) | PASS |

Correctness chain: the **same kernel IR** matches vendored libcint 6.1.3 at
atol 1e-12 (cart+sph) when run on `CpuRuntime`; on ROCm the identical kernel is
HIP-compiled and runs on gfx1152 (f64 device execution was verified up front
with a throwaway `rys_root2` smoke test).

## Notes / out of scope

- `f32_parity::test_f32_int3c2e_sph_parity` fails — **pre-existing** on branch
  `fix/general-contraction-nctr-1e` (verified by reverting this task's kernel
  file). It is the 3c2e family, untouched here; not addressed.
- Wgpu/Cuda/Metal arms now also route through the device kernel at f64; those
  backends remain compile-only/unverified per the existing repo notes (f64 on
  wgpu may be unsupported at runtime — unchanged risk posture, not exercised by
  the oracle suite).
- The kernel is single-work-item (correctness-first); GPU parallelization
  across primitives/components is future work.
