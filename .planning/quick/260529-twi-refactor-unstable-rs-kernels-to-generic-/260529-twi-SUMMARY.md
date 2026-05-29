---
quick_id: 260529-twi
status: complete
date: 2026-05-30
---

# Quick Task 260529-twi — Summary

**Goal:** Refactor all of `crates/cintx-cubecl/src/kernels/unstable.rs` (5 unstable-source-api
families) from host-only f64 into on-device CubeCL `#[cube(launch)]` kernels generic over `F`,
and run a randomized ROCm-backend oracle per family.

**Outcome:** All 5 families (origi, grids, breit, origk, ssc) now run a real `#[cube(launch)]`
device kernel for their scalar path, dispatched on all 5 `ResolvedBackend` arms
(Cpu/Wgpu/Cuda/ROCm-HIP/Metal). Verified on the AMD `gfx1152` GPU.

## Module split (Task 1, commit b9a4c6b)

`unstable.rs` (3511 lines) → behavior-preserving module directory
`unstable/{mod.rs (facade), shared.rs, origi.rs, grids.rs, breit.rs, origk.rs, ssc.rs}`.
Move-only; `kernels/mod.rs` registration unchanged (directory module resolves identically);
the `unstable_source_parity.rs` CPU baseline (8 passed / 15 failed) is unchanged by the split.

## Per-family device ports

| Family | Commit | Device kernel | ROCm oracle | Deferred-to-host (documented) |
|--------|--------|---------------|-------------|-------------------------------|
| origi  | 67f3778 | `origi_scalar_kernel<F>` (r2/r4, comptime `r_power`) | vendor parity, **96 cases, 0 mismatch** | ip2 variants |
| origk  | 93d3ab2 | `origk_scalar_kernel<F>` (r2/r4/r6, comptime `r_power`) | vendor parity, **144 cases, 0 mismatch** | ip1 gradient variants |
| ssc    | d9656a0 | `ssc_scalar_kernel<F>` (3c2e + ij-HRR split) | vendor parity, **48 cases, 0 mismatch** | host transpose+c2s (host split) |
| breit  | 1d7b927 | `breit_g_kernel<F>` (4-center G-tensor, 4-branch HRR) | vendor parity, spinor, non-square, **48 cases, 0 mismatch** | gout ladder + nabla/x1 + cart_to_spinor |
| grids  | 5a6d71f | `grids_scalar_kernel<F>` (nuclear-like, per grid point) | **device-vs-host on ROCm, 64 cases, 0 mismatch** (vendor blocked, see below) | ip/ipip/ipvip/spvsp |

Plus a build fix to the merged origi (6ec972a): un-gate the `compute_pdata_host` import that the
live host ip2 path needs (it was `#[cfg(test)]`-gated, so the merged tree did not build).

## Host/device split (all families)

- **ON DEVICE:** per-primitive G-tensor build (VRR/HRR) + Cartesian contraction, f64-internal /
  F-output, accumulated over primitive (and, for the scalar 1e/3c families, contraction) pairs.
- **ON HOST (unchanged):** `cart_to_sph_*` / `cart_to_spinor_*` coefficient-table transforms +
  AO scatter into `staging`, plus `compute_pdata_host` pair-data setup.
- Dispatch styles: origi/origk/ssc loop primitives+contractions inside one device launch
  (comptime `r_power`/`nroots`, host-computed strides as runtime u32); breit and grids dispatch
  once per quartet / per (grid, prim-pair) and accumulate on host (mirrors their host loops).

## grids pre-existing blocker (documented, not fixed)

`eval_raw(RawApiId::Symbol("int1e_grids_sph"))` rejects the grids 4-element shell tuple with
`InvalidShellTuple { expected: 2, got: 4 }` — a PRE-EXISTING upstream wiring failure that also
makes all `grids_parity` CPU-vs-vendor tests fail at baseline, independent of this GPU port
(out-of-scope per the user's decision). Consequently grids' ROCm validation is a DIRECT
device-vs-host parity test inside the cubecl crate
(`grids.rs::tests::test_grids_device_vs_host_rocm`, 64 cases, 0 mismatch on the GPU), and
`tests/grids_random_rocm_parity.rs` documents + LOCKS the blocker (asserts the rejection still
reproduces; it will fail loudly when the upstream wiring is fixed, signalling to wire a real
grids vendor oracle).

## Verification

- 16/16 in-crate device tests pass (`cargo test -p cintx-cubecl --features cpu,unstable-source-api
  --lib kernels::unstable`): device-vs-host f64 + f32 genericity for every family.
- `cargo build -p cintx-cubecl --features rocm,unstable-source-api` clean.
- `unstable_source_parity` CPU baseline 8 passed / 15 failed — UNCHANGED (no regression; the 15 are
  the pre-existing grids-all + origi-ip2 + origk-ip1 deferred/blocked sub-paths).
- FamilyLaunchFn signatures + `kernels/mod.rs` registration unchanged; no capi enum variants, no
  legacy `cint*` wrappers.

## `#[cube]` authoring pitfalls hit

- Comptime/runtime loop bound: `while nr < nroots` against a `#[comptime]` bound fails type
  inference (`NativeExpand` ambiguity) — materialize it first (`let nrys = nroots;`) and annotate
  the counter (`let mut nr: u32 = 0u32;`), as one_electron does.
- No plain-fn calls inside `#[cube]`: VRR/HRR recurrences re-expressed as `#[cube]` helpers
  (`grids_vrr_axis`/`grids_hrr_axis`, `origi_*`, `origk_*`) called directly.
- Cartesian components enumerated on-device via nested lx-descending `while` loops (no host
  `cart_comps` call); `as usize` only at index sites; `F::exp`/`F::sqrt`/`F::cast_from`; no
  if-expressions / continue / break.
- Spinor sizing in the breit oracle: kappa=0 spinor dim is `2*(2l+1)`, not `2*(l+1)` (they agree
  only at l=0) — a non-square p-shell quartet exposed it via `BufferTooSmall`.

## Orchestration notes (for resumption / future ports)

- Worktree isolation in this environment spawned agents from a 177-commit-stale base; each parallel
  executor must `git reset --hard <current-HEAD>` first (the workflow's base-correction step).
- One agent's edits leaked into the main tree via an editor-overlay vs worktree filesystem
  divergence; prefer Bash-backed edits in the worktree and verify `git status` before committing.
- breit + grids were finished inline in the main loop after the parallel executors hit the account
  session limit; their device kernels + tests + oracles were authored and verified directly.
