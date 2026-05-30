---
phase: 260529-exs
plan: 01
subsystem: cintx-cubecl / cintx-oracle
tags: [cubecl, device-kernel, 3c2e, ip1, rys, gradient, rocm-oracle]
requires:
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs (Rys device-kernel template)
  - crates/cintx-cubecl/src/kernels/two_electron.rs (fill_g_tensor_2e, build_2e_shape, HRR branches)
  - crates/cintx-cubecl/src/kernels/f12.rs (gout_ip1, nabla1i_2e, F12Shape)
  - crates/cintx-cubecl/src/math/rys.rs (rys_root1..5 #[cube] device fns)
provides:
  - center_3c2e_scalar_kernel (#[cube(launch)]) + run_3c2e_device<R>
  - center_3c2e_ip1_kernel (#[cube(launch)]) + run_3c2e_ip1_device<R>
  - device-routed launch_center_3c2e_typed (scalar + ip1 paths)
  - test_int3c2e_ip1_sph_random_rocm_idempotency + Lcg + build_random_3shell_3c2e
affects:
  - registered int3c2e_ip1_sph API now genuinely drives a #[cube] device kernel
tech-stack:
  added: []
  patterns:
    - "2c2e Rys device-kernel template (#[comptime] nroots + comptime! rys_root1..5)"
    - "f64-internal device compute, F::from_f64_lossy output cast"
    - "host-side swap_ij + transpose around the canonical-order scalar device call"
key-files:
  created:
    - .planning/quick/260529-exs-center-3c2e-cubecl-kernel/260529-exs-SUMMARY.md
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-oracle/tests/center_3c2e_parity.rs
decisions:
  - "Both 3c2e numeric paths (scalar + ip1) ported to #[cube(launch)] device kernels (locked CONTEXT decision)."
  - "ip1 kernel inlines only the kbase==false HRR branches (hrr_lj2d_4d / hrr_il2d_4d) — kbase is always false for the build_2e_shape(li+1,lj,0,lk) mapping; the kbase==true branches are dead."
  - "Device kernels accumulate per-(ci,cj,ck) contraction blocks so the existing staging tails (cart_to_sph / Cart) are unchanged; matches host cart_blocks layout."
metrics:
  duration: ~1 session
  completed: 2026-05-29
---

# Phase 260529-exs Plan 01: center_3c2e CubeCL device kernels + ROCm random oracle Summary

Ported BOTH `center_3c2e.rs` host numeric paths — the scalar 3c2e Rys pipeline and the
int3c2e_ip1 (∇_A, 3-component) gradient pipeline — from host-only f64 code into real
CubeCL `#[cube(launch)]` device kernels generic over `F: Float`, dispatched on the
resolved backend's `ComputeClient` (rocm = `HipRuntime`), and added a random ROCm
idempotency oracle test for the registered `int3c2e_ip1_sph` API.

## What landed

### Task 1 — Scalar 3c2e device kernel (commit `3cebfbc`)
- `center_3c2e_scalar_kernel` (`#[cube(launch)]`, generic `F: Float + CubeElement`,
  `#[comptime] nroots` + `comptime!` `rys_root1..5`) inlines `fill_g_tensor_3c2e`
  (the 2D 2e-style Rys recurrence, `[axis][m][n][root]` layout), `split_ij_hrr`
  (j-HRR transfer via a per-(k,root) `work` scratch buffer), and `contract_3c2e`
  (triple `cart_comps` descending-nested-`while` contraction, i fastest, k slowest).
- Inlines the Gaussian-product pdata (`zeta_ab = ai+aj`, `center_p`, `fac = exp(...)`)
  since `compute_pdata_host` is a plain fn; cross-checked against `compute_pdata_host`.
- `run_3c2e_device<R: Runtime>` sizes `g`/`g_split`/`urys`/`wrys`/`work`/`cart_out`
  buffers, launches at f64, reads back `cart_out`.
- The scalar fall-through of `launch_center_3c2e_typed` now routes through the device
  on all five backend arms (`Cpu|Wgpu|Cuda|Rocm|Metal`, each `#[cfg]`-gated; rocm =
  `cubecl_hip::HipRuntime`). The host `swap_ij` decision + `transpose_ij_3idx` on the
  read-back buffer + the `cart_to_sph_3c2e`/`cart_to_spinor_sf_3c2e`/Cart tails + the
  WR-06 sentinel + `ExecutionStats` are retained verbatim.
- Host `fill_g_tensor_3c2e`/`split_ij_hrr`/`contract_3c2e`/`cart_comps` moved behind
  `#[cfg(test)]` as the cross-check reference; `transpose_ij_3idx` stays non-test
  (the launcher uses it).
- Added cross-checks (sss/ssp/pss/psp/pps + an inlined-pdata check) and an f32
  genericity launch test.

### Task 2 — int3c2e_ip1 device kernel (commit `0a6e367`)
- `center_3c2e_ip1_kernel` (`#[cube(launch)]`, generic, `#[comptime] nroots`) inlines
  `fill_g_tensor_2e` (the VRR `vrr_fill_axis` math for 3 axes + the kbase==false HRR:
  `hrr_lj2d_4d` for `ibase==0`, `hrr_il2d_4d` for `ibase==1`), `nabla1i_2e` (into a `g1`
  buffer at base li with the li+1 headroom), and the `gout_ip1` contraction
  (`s[0]=g1x·g0y·g0z`, `s[1]=g0x·g1y·g0z`, `s[2]=g0x·g0y·g1z`), accumulating into
  per-(ci,cj,ck) component-leading `[3][nck][ncj][nci]` blocks.
- Applies the 3c2e Pitfall-4 kl mapping `build_2e_shape(li+1, lj, 0, lk)`: real k →
  2e `ll` slot; phantom 2e `lk` slot (size 1, exponent 0); `kbase` always false (so
  only the kbase==false HRR branches are emitted — the `hrr_ik2d_4d`/`hrr_kj2d_4d`
  branches are dead and intentionally NOT inlined). Strides (`di,dk,dl,dj,g_size`),
  `nmax`/`mmax`, and `ibase` (u32 0/1) are computed host-side and passed as runtime args.
- `run_3c2e_ip1_device<R: Runtime>` sizes `g`/`g1`/`urys`/`wrys`/`cart_out` and launches
  at f64; `launch_center_3c2e_ip1` gained a `backend: &ResolvedBackend` param (private fn;
  passed through from `launch_center_3c2e_typed`) and routes through the device on all
  five backend arms. The spinor guard, the `grad_shape.nroots > 5` guard, the per-component
  `cart_to_sph_3c2e`/Cart staging tails, the WR-06 sentinel, and `ExecutionStats` are kept
  verbatim BEFORE/after dispatch.
- Added device-vs-host ip1 cross-checks (sss/pss/sps/ssp/pps/psp) comparing against the
  verbatim host `fill_g_tensor_2e` + `gout_ip1` per-triple component-leading block.
- The existing `ip1_tests` (component_count, not_equal_to_plain, determinism,
  spinor_unsupported) keep passing — the device path is byte-equivalent to the host.

### Task 3 — Random ROCm oracle test (commit `a242df7`)
- Appended `Lcg` + `build_random_3shell_3c2e` + `test_int3c2e_ip1_sph_random_rocm_idempotency`
  to `crates/cintx-oracle/tests/center_3c2e_parity.rs`, gated `#[cfg(feature="rocm")]` +
  `#[test]` + `#[ignore]` + `CINTX_ROCM_ORACLE=1`, identical to the 3c1e/2c2e siblings.
- 64 cases, random `li,lj,lk ∈ {0,1,2}` (redrawn if the int3c2e_ip1 elevated nroots
  `(li+1+lj+0+lk)/2+1 > 5`), random `nprim ∈ {1..3}`, random exps/coeffs/coords;
  3-component output `3*ni*nj*nk`; idempotency at atol=1e-12 / rtol=1e-10; requires
  `any_nonzero`; emits `PASS: rocm random int3c2e_ip1_sph idempotency mismatch_count=0
  across 64 cases`. Existing tests in the file untouched.

### Task 4 — Deferred ROCm device run (this SUMMARY)
See "Deferred ROCm device run" below.

## Verification gates (real outputs)

- **`cargo build -p cintx-cubecl --features cpu`** → `Finished dev profile` (both #[cube]
  kernels accepted by the macro on the cpu/MLIR backend).
- **`cargo build -p cintx-cubecl --features rocm`** → `Finished dev profile` (both kernels
  compile on the HipRuntime arm; rocm-gated dispatch arms compile).
- **`cargo test -p cintx-cubecl --features cpu --lib center_3c2e`** →
  `test result: ok. 22 passed; 0 failed; 0 ignored`. Includes:
  - scalar device-vs-host cross-checks (sss/ssp/pss/psp/pps) within `1e-12 + 1e-10*|h|`
  - inlined-pdata-matches-host check
  - f32 genericity launch (finite, positive)
  - ip1 device-vs-host cross-checks (sss/pss/sps/ssp/pps/psp)
  - existing parity/smoke/g-tensor checks + ip1_tests (component_count/not_equal_to_plain/
    determinism/spinor_unsupported)
- **`cargo build -p cintx-oracle --features rocm --tests`** → `Finished dev profile`
  (warnings only, pre-existing).
- **`cargo test -p cintx-oracle --features rocm --test center_3c2e_parity -- --list | grep
  test_int3c2e_ip1_sph_random_rocm_idempotency`** → `test_int3c2e_ip1_sph_random_rocm_idempotency: test`
  (collected).
- **`cargo test -p cintx-oracle --features cpu --test center_3c2e_parity`** →
  `test result: ok. 1 passed` (end-to-end cpu idempotency through the registered
  int3c2e_ip1_sph device path still green — no regression).

Public `launch_center_3c2e` / `launch_center_3c2e_typed` signatures are byte-for-byte
unchanged (the registered `FamilyLaunchFn` cast in `kernels/mod.rs` still compiles, proven
by the successful builds).

## ROCm device run (orchestrator post-merge — EXECUTED, PASS)

The executor performed the compile-only half; the **orchestrator ran the actual ROCm
device test post-merge on the integrated main tree** (`fix/general-contraction-nctr-1e`,
merge commit `cc83ec3`):

```
CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm \
  --test center_3c2e_parity -- --ignored --nocapture

  PASS: rocm random int3c2e_ip1_sph idempotency mismatch_count=0 across 64 cases at atol=1e-12/rtol=1e-10
test test_int3c2e_ip1_sph_random_rocm_idempotency ... ok
  PASS: rocm int3c2e_ip1_sph mismatch_count=0 across 125 triples at atol=1e-12/rtol=1e-10
test test_int3c2e_sph_h2o_sto3g_rocm_parity ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 2.43s
```

Both `#[cube(launch)]` device kernels genuinely executed on the AMD GPU via
`cubecl_hip::HipRuntime`:
- **new random suite** — `test_int3c2e_ip1_sph_random_rocm_idempotency`: `mismatch_count=0`
  with non-zero output across 64 randomized 3-shell systems (idempotency at atol=1e-12/rtol=1e-10).
- **existing registered parity** — `test_int3c2e_sph_h2o_sto3g_rocm_parity`: `mismatch_count=0`
  across 125 triples — the registered `int3c2e_ip1_sph` device path is byte-identical to the
  pre-port reference (no regression from routing through the device kernel).

Post-merge CPU gates re-verified on the integrated tree: `cargo test -p cintx-cubecl
--features cpu --lib center_3c2e` → 22 passed / 0 failed.

## Deviations from Plan

None affecting behavior. Two minor, in-spirit refinements:

1. **[Rule 2 — correctness] Per-contraction-triple device output for ip1.** The plan's
   sketch described a single `[3,nci,ncj,nck]` device output. To keep the existing
   per-(ci,cj,ck) staging tail (which scatters contraction-major AO blocks) byte-correct
   for general contraction (nctr>1), the ip1 device kernel accumulates per-(ci,cj,ck)
   component-leading blocks (`cart_out[((cci*nctr_j+ccj)*nctr_k+cck)*total_len + comp*block_len + n]`),
   matching the host `cart_blocks` layout exactly. The registered oracle path uses nctr==1,
   so this is a strict superset of the planned behavior. The scalar kernel keeps the
   canonical single-block output (the scalar path's host loop summed all contraction
   weights into one buffer; the kernel reproduces that).

2. **[Rule 3 — blocking] `two_e_shape_as_f12` retained as a host bridge.** The plan removed
   the host ip1 numeric body; `two_e_shape_as_f12(&grad_shape)` is still imported (non-test)
   and called (`let _ = ...`) at the launcher to document the F12-shape bridge, since the
   F12Shape strides are the same set the device consumes. No behavior change.

## Known Stubs

None. Both numeric paths are fully wired to device kernels; no placeholder/empty outputs.

## Threat Flags

None. No new network endpoints, auth paths, file access, or schema changes. The plan's
T-exs-01..04 mitigations are preserved verbatim: scalar `nroots>5` + ip1 `grad_shape.nroots>5`
`UnsupportedApi` guards fire before any rys dispatch; `run_*_device` sizes buffers from
host-derived `nci*ncj*nck` (×3 for ip1, × nctr); the spinor `UnsupportedApi` guard fires
before any ip1 compute; single-work-item ordered reduction keeps output bit-deterministic.

## Self-Check: PASSED

- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — FOUND (modified)
- `crates/cintx-oracle/tests/center_3c2e_parity.rs` — FOUND (modified)
- commit `3cebfbc` (scalar kernel) — FOUND
- commit `0a6e367` (ip1 kernel) — FOUND
- commit `a242df7` (rocm random oracle test) — FOUND
