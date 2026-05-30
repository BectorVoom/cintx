---
phase: 260529-fsa
plan: 01
subsystem: cintx-cubecl / cintx-oracle
tags: [cubecl, 4c1e, gpu-kernel, rocm, oracle]
requires:
  - cintx-cubecl::backend::ResolvedBackend
  - cintx-compat::raw (INT4C1E_SPH, eval_raw, env-layout constants)
  - cubecl 0.10.0 (#[cube(launch)], CpuRuntime, HipRuntime)
provides:
  - center_4c1e_kernel #[cube(launch)] generic-F device kernel
  - run_4c1e_device::<R> backend dispatch helper
  - test_int4c1e_sph_random_rocm_idempotency oracle (mismatch_count=0)
affects:
  - crates/cintx-cubecl/src/kernels/center_4c1e.rs
  - crates/cintx-oracle/tests/center_4c1e_parity.rs
tech-stack:
  added: []
  patterns:
    - "device kernel f64-internal / F-output (run_*_device::<R> + F::from_f64_lossy at c2s)"
    - "host f64 reference kept under #[cfg(test)] for device-vs-host cross-check"
    - "1D polynomial recurrence scratch buffer passed as &mut Array<F> (kernel never allocates)"
key-files:
  created:
    - crates/cintx-oracle/tests/center_4c1e_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/center_4c1e.rs
decisions:
  - "Reused 3c1e device-dispatch template verbatim, adapted to 4 shells + polynomial recurrence + 4-branch HRR"
  - "Scratch buf (1D polynomial) sized on host via shape_sizes_4c1e and passed as a device Array arg"
  - "No HIP-backend fix needed: kernel compiled and ran correctly on AMD GPU first try (mismatch_count=0)"
metrics:
  duration: ~25m
  completed: 2026-05-29
  tasks: 4
  files: 2
---

# Phase 260529-fsa Plan 01: center_4c1e CubeCL Device Kernel + ROCm Oracle Summary

Refactored `center_4c1e.rs` from a host-side f64 loop into a real CubeCL
`#[cube(launch)]` device kernel generic over `F`, dispatched onto the resolved
backend (CpuRuntime / HipRuntime / WgpuRuntime / CudaRuntime) via
`run_4c1e_device::<R>`, with a 64-case random ROCm idempotency oracle reporting
mismatch_count=0 on the AMD GPU.

## What Was Built

- **`center_4c1e_kernel` `#[cube(launch)]`** (generic over `F: Float + CubeElement`):
  single work-item (`UNIT_POS == 0`) port of the full host pipeline — inline
  Shape4c1e layout (nroots=1, ibase/kbase branch, di/dk/dl/dj strides), full
  primitive-quartet `while` loops with per-primitive coefficient folding, the
  polynomial 1D recurrence + 2D shift G-tensor fill per axis (z-axis gets the
  `fac/(aijkl*sqrt(aijkl))` prefactor; x/y start at 1.0), all four HRR branches
  (`hrr_ik2d`/`hrr_kj2d`/`hrr_il2d`/`hrr_lj2d`) inlined and branch-selected by
  kbase/ibase, and the Cartesian contraction (i fastest, l slowest). Follows all
  CubeCL authoring rules: statement-form `if`, `F::exp`/`F::sqrt`/`F::cast_from`/
  `F::new` free functions, u32 indices, `while` loops, no Vec/continue/break, no
  plain-fn calls.
- **`run_4c1e_device::<R: Runtime>`**: allocates the `g` (3*g_size) and `buf`
  (db*(bigger+1)) scratch Arrays + zeroed `cart_out`, launches at f64, reads back
  the Cartesian buffer. `shape_sizes_4c1e` computes the scratch sizes host-side so
  the kernel never allocates.
- **`launch_center_4c1e_typed`** rewired: kept `ensure_validated_4c1e` (spinor
  rejection first), 4-shell check, common_factor `SQRTPI*PI*sp_factor`, c2s/WR-06
  sentinel/ExecutionStats verbatim; replaced the host quartet loop with a
  per-(ci,cj,ck,cl) `match backend { Cpu/Wgpu/Cuda/Rocm/Metal }` dispatch.
- **Host f64 references** (`Shape4c1e`, `build_4c1e_shape`, `fill_4c1e_g_tensor`,
  the four `hrr_*_4d`, `contract_4c1e_cart`, `cart_comps`) moved under `#[cfg(test)]`.
- **Device-vs-host cross-check** (`device_cross_check` mod, cpu-gated): 6 tests
  (ssss, psss, sssp, spsp, ppss, pssp) at atol=1e-12+rtol=1e-10 across all four
  HRR branches + 1 generic-f32 launch evidence test.
- **`center_4c1e_parity.rs`** oracle: `test_int4c1e_sph_random_rocm_idempotency`
  (rocm+with-4c1e, ignored, env-gated, 64 random cases) + a cpu self-consistency
  smoke (cpu+with-4c1e).

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | center_4c1e #[cube(launch)] kernel + run_4c1e_device + dispatch | d288a08 | center_4c1e.rs |
| 2 | device-vs-host equivalence + generic-f32 unit tests | 548d5bf | center_4c1e.rs |
| 3 | random ROCm int4c1e_sph idempotency oracle | fbb3158 | center_4c1e_parity.rs |
| 4 | run ROCm oracle, confirm mismatch_count=0 | (no code commit) | — |

## Verification Results

- `cargo test -p cintx-cubecl --features cpu,with-4c1e --lib center_4c1e` —
  **12 passed, 0 failed** (5 pre-existing incl. `test_center_4c1e_parity_f64`
  byte-identity + `test_center_4c1e_f32_smoke`; 6 device-vs-host + 1 generic-f32).
- `cargo test -p cintx-oracle --features cpu,with-4c1e --test center_4c1e_parity` —
  **1 passed** (`test_int4c1e_sph_cpu_self_consistency`).
- ROCm oracle on the AMD GPU (verbatim PASS line):

  ```
    PASS: rocm random int4c1e_sph idempotency mismatch_count=0 across 64 cases at atol=1e-12/rtol=1e-10
  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.75s
  ```

  Invoked as `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle
  --features rocm,with-4c1e --test center_4c1e_parity -- --ignored` — the exact
  command `xtask rocm-oracle --profile with-4c1e` spawns. No HIP-backend fix was
  required; the kernel produced correct, deterministic results on the device on
  the first run.

## Deviations from Plan

### [Rule 3 - Blocking issue] xtask not a workspace member at the plan's base commit

- **Found during:** Task 4.
- **Issue:** The worktree was reset to base `062f01d` (per the spawn instruction),
  which predates the commit that added `xtask` to the workspace `members`. As a
  result `cargo run -p xtask -- rocm-oracle --profile with-4c1e` fails with
  "package `xtask` not found in workspace", and running the standalone `xtask`
  via `--manifest-path` fails with "cannot specify features for packages outside
  of workspace" (cintx-oracle is outside the xtask-only workspace).
- **Fix:** Ran the exact `cargo test` invocation `rocm_oracle.rs` would spawn,
  from the worktree root where `cintx-oracle` IS a workspace member:
  `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle
  --features rocm,with-4c1e --test center_4c1e_parity -- --ignored`. This is the
  authoritative device run; the xtask wrapper failure is purely a
  cwd/workspace-membership artifact of the base reset, not a kernel issue.
- **Files modified:** none (test-invocation workaround only).
- **Commit:** n/a.

## Authentication Gates

None.

## Known Stubs

None. The `launch_center_4c1e_typed` host quartet loop was fully replaced by the
device dispatch; no placeholder data paths remain.

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/center_4c1e.rs
- FOUND: crates/cintx-oracle/tests/center_4c1e_parity.rs
- FOUND commit d288a08 (Task 1)
- FOUND commit 548d5bf (Task 2)
- FOUND commit fbb3158 (Task 3)
