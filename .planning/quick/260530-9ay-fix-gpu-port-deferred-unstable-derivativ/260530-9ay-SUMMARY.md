---
quick_id: 260530-9ay
status: complete
date: 2026-05-30
---

# Quick Task 260530-9ay — Summary

**Goal:** Fix + GPU-port the deferred unstable derivative sub-paths from the prior task
(origi-ip2, origk-ip1, grids derivatives) — "fix the host math to vendor parity, then port
to device."

## Root cause: it was a manifest metadata bug, not a math bug

Investigation showed the deferred derivative ops emitted **zeros for components past the first**
— because their `component_rank` in `crates/cintx-ops/generated/compiled_manifest.lock.json` was
`"1"`, so the planner allocated a 1-component workspace and `launch_*` silently dropped the rest
(guard `if sph_off + sph_size <= staging.len()`). The contraction math was already correct. Fix
(commit `587c38d`): set the true component counts —

| op | was | now |
|----|-----|-----|
| int1e_r2/r4_origi_ip2_sph | 1 | **3** (∇-ket) |
| int3c1e_ip1_r2/r4/r6_origk_sph | 1 | **3** (∇-bra) |
| int1e_grids_ip_sph | 1 | **3** |
| int1e_grids_ipip_sph | 2 | **9** |
| int1e_grids_ipvip_sph | 2 | **9** |
| int1e_grids_spvsp_sph | (empty) | **4** |

This immediately fixed CPU vendor parity for **origi-ip2 (r2,r4)** and **origk-ip1 (r2,r4)**.
`build.rs` regenerates `api_manifest.rs`/`.csv` from the lock.

## Device ports (commits 35f1ece, 27eeca6, dd4f999 — merged at 51017b9)

| family | commit | device kernel | ROCm validation |
|--------|--------|---------------|-----------------|
| origi ip2 (r2/r4) | 35f1ece | `origi_ip2_kernel<F>` (overlap VRR/HRR + D_J + 3-comp r2/r4_ip2 contraction, comptime r_power) | vendor parity **96 cases, 0 mismatch** |
| origk ip1 (r2/r4/r6) | 27eeca6 | `origk_ip1_kernel<F>` (3c1e VRR/HRR-i/HRR-k + D_I + 3-comp r2/r4/r6, comptime r_power) | r2/r4 vendor **0**; r6 device-vs-host holds, vendor residual documented (41 flagged) |
| grids derivs (ip/ipip/ipvip/spvsp) | dd4f999 | one `grids_deriv_kernel<F>` (comptime op_kind 1-4: 3/9/9/4 comp) | device-vs-host **0 mismatch** (ip 64, ipip/ipvip/spvsp 48; (p,p)→host for nroots=3) |

All dispatched on all 5 backend arms (Rocm→`cubecl_hip::HipRuntime`); host c2s + AO scatter unchanged;
FROZEN launch_* signatures. grids vendor oracle stays blocked by the pre-existing
`InvalidShellTuple{2 vs 4}` (same as grids scalar) → device-vs-host is the validation.

## Residual: origk-ip1 r6 — ROOT CAUSE FOUND: it is a LIBCINT bug, cintx is correct

`int3c1e_ip1_r6_origk_sph` diverges ~6% from the vendor reference on the y-component (shls [3,4,0]).
A finite-difference investigation (perturbing the bra atom's coordinate in the env, central
difference, eps swept 1e-3/1e-4/1e-5) is conclusive:

| quantity | r4 | r6 |
|----------|----|----|
| cintx scalar == vendor scalar (incl. perturbed geom) | yes (~4e-19) | yes (~4e-19) |
| FD(−∂/∂Rᵢ of vendor scalar) [the true gradient] | 2.2369152237e-5 | **1.16045358e-5** (stable across eps) |
| vendor ip1 | 2.2369152253e-5 | **1.0944788e-5** |
| cintx ip1 | 2.2369152253e-5 | **1.16045358e-5** |
| vendor ip1 == FD(vendor scalar)? | **yes** (~1e-14) | **NO** (gap 6.6e-7, constant in eps) |

So for **r4**, libcint's ip1 equals the gradient of its own scalar and cintx matches. For **r6**,
**libcint's ip1 does NOT equal the gradient of its own scalar** — an internal inconsistency in the
vendored libcint. cintx's ip1_r6 equals the true gradient (FD of the scalar, which cintx reproduces
to 4e-19) and its gout matches libcint's `CINTgout1e_int3c1e_ip1_r6_origk` term-for-term. **cintx is
mathematically correct; the vendored libcint `int3c1e_ip1_r6_origk` autocode is buggy** (a rarely-used
high-order origin-derivative integral with 128+ g-buffers — classic autocode-error territory). The
constant-across-eps gap rules out finite-difference truncation.

Implication for the `unstable_source_parity` r6 vendor assertion: the vendor reference is provably
wrong for this one operator, so the assertion fails despite cintx being correct. Resolution is a
project-policy choice — re-base that single assertion on FD self-consistency (validate cintx against
the true gradient), accept the documented known-libcint-defect, or deliberately match-the-bug for
strict byte-compatibility. Device-vs-host parity for r6 holds in all cases.

## Verification (merged tree, AMD gfx1152)

- 26/26 in-crate unstable device tests pass (device-vs-host f64 + f32 for every family incl. the
  3 new derivative ports).
- `cargo build -p cintx-cubecl --features rocm,unstable-source-api` clean.
- `unstable_source_parity`: 12 passed / 11 failed — improved from the 8/15 baseline (fixed 4:
  origi-ip2 r2/r4, origk-ip1 r2/r4). Remaining 11 = grids (10, pre-existing eval_raw blocker) +
  origk-ip1 r6 (the documented residual). No regressions.
- ROCm oracles: origi-ip2 96/0, origk-ip1 r2/r4 0 (r6 documented), grids derivs device-vs-host 0.

## #[cube] pitfalls (consistent across all three ports)

- Comptime loop bounds must be materialized into a local before `while` comparisons (`let n =
  nroots; while i < n`) — comparing a counter directly against a `#[comptime]` param fails type
  inference (NativeExpand). Same for comptime op_kind/r_power: enumerate the full launch matrix in
  the host dispatcher (comptime params can't be threaded as runtime values).
- No device-local `Array`: all scratch (g/g1/g_di/urys/wrys) host-allocated, passed as `&mut Array<F>`.
- `#[cube]` helpers (origi_dj_axis, origk_di_axis, grids nabla helpers) called directly; no plain-fn
  calls; `F::exp`/`F::sqrt`/`F::cast_from`; u32 + `as usize` at index sites.
- Comp-slowest output layout `out[comp*nci*ncj(*nck) + ...]` matches the host per-component c2s loop.
