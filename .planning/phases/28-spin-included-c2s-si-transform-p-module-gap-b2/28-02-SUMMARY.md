---
phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
plan: 02
subsystem: cintx-cubecl / kernels
tags: [sigma-p, c2spinor, cube-kernel, gout-assembler, FND-05, gap-b2]
requires:
  - "one_electron.rs nabla1i / gout gradient machinery (template)"
  - "Phase 28-01 cart_to_spinor_si_2d + apply_bra_si_block (consumer, separate plan)"
provides:
  - "kernels/sigma_p.rs — generic rank-parameterized σ·p #[cube] G-tensor assembler"
  - "Four gc_x/gc_y/gc_z/gc_1 component-leading blocks (pre-blocked) for the si_2d transform"
  - "run_sigma_p_on_backend 5-arm dispatch (Cpu/Wgpu/Cuda/Rocm/Metal)"
affects:
  - "Phase 29 σ-group (int1e_sp/spsp/spnucsp/sprinvsp/sigma) reuse this assembler (D-03)"
tech-stack:
  added: []
  patterns:
    - "comptime tensor_rank parameter (D-03 reusability, comptime-nroots template analog)"
    - "host-wrapper + #[cube] pair for device-vs-host parity tests"
    - "component-LEADING / pre-blocked gc[comp*(nf*nc)+n] output (avoids host transpose)"
key-files:
  created:
    - "crates/cintx-cubecl/src/kernels/sigma_p.rs"
  modified:
    - "crates/cintx-cubecl/src/kernels/mod.rs"
decisions:
  - "Emit gc blocks PRE-BLOCKED (component-leading) on-device rather than interleaved gout[n*4+comp], so cart_to_spinor_si_2d reads gc_x=block0..gc_1=block3 with no host transpose (Spike Target C / RESEARCH Open Q #1)."
  - "Duplicate the VRR/HRR axis helpers locally (sigma_p_vrr_axis/sigma_p_hrr_axis) to keep sigma_p self-contained — one_electron's copies are module-private."
  - "Parameterize by #[comptime] tensor_rank so int1e_sigma (rank 3) reuses the same gc-block packing (D-03)."
metrics:
  duration: 9 min
  completed: 2026-05-31
---

# Phase 28 Plan 02: σ·p #[cube] G-tensor Assembler Summary

Generic rank-parameterized σ·p device `#[cube]` assembler emitting the four
component-leading `gc_x/gc_y/gc_z/gc_1` cartesian blocks (scalar slot zero for
`int1e_sp`), wired into `kernels/mod.rs`, with passing device-vs-host parity tests.

## What Was Built

A new device kernel module `crates/cintx-cubecl/src/kernels/sigma_p.rs`:

- **`sigma_p_kernel<F>`** (`#[cube(launch)]`) — builds the overlap base G-tensor
  `g0` (fixed-center VRR + HRR), applies the faithful `CINTnabla1i_1e` bra nabla
  `g1 = nabla_i(g0)` (`ai2 = -2*ai`; `g1[ix==0]=ai2*g0[ix+1]`;
  `g1[ix>=1]=ix*g0[ix-1]+ai2*g0[ix+1]`), then forms the 3 Pauli components per
  cart `n`: `s0=g1x*g0y*g0z` (gc_x), `s1=g0x*g1y*g0z` (gc_y),
  `s2=g0x*g0y*g1z` (gc_z). For `int1e_sp` (`tensor_rank==1`) the scalar slot
  `gc_1 == 0.0`. Output is **component-LEADING / pre-blocked**
  `gc_out[base + comp*block_len + (cj_idx*nci + ci_idx)]` with
  `base = (ci*nctr_j+cj) * (tensor_rank*N_GC * block_len)` — the layout libcint
  reaches via `CINTdmat_transpose`, emitted on-device so the host transform reads
  the four gc blocks in order with no extra transpose.
- **`sigma_p_vrr_axis` / `sigma_p_hrr_axis`** (`#[cube]`) — local copies of the
  overlap VRR/HRR axis recurrences.
- **`run_sigma_p_device<R>`** — buffer creation + `launch::<f64,R>` + readback;
  `tensor_rank` selects the comptime monomorphization.
- **`run_sigma_p_on_backend`** — 5-arm feature-gated dispatch
  (Cpu/Wgpu/Cuda/Rocm/Metal), `#[allow(dead_code)]` until the live `int1e_sp`
  Spinor dispatch arm is wired in a later plan.
- **`sigma_p_host`** (`#[cfg(test)]`) — pure-Rust reference replicating the kernel
  exactly for device-vs-host parity.

`kernels/mod.rs` declares `pub mod sigma_p;`.

## Tests (all passing, `cargo test -p cintx-cubecl --features cpu sigma_p`)

- `sigma_p_device_matches_host` — device-vs-host parity over 4 shell-pair cases
  (p×d non-square, p×p, s×s, d×p) at a relative band of `1e-9*(1+|h|)` (set just
  above the CubeCL CpuRuntime FP-env ~1e-11 perturbation, Pitfall 5; kernel is
  bit-faithful).
- `sigma_p_device_matches_host_ss_scalar_slot_zero` — s×s sanity: 4 gc blocks,
  scalar slot (block 3) == 0.0 for every cart n; Pauli blocks non-zero.
- `sigma_p_layout_is_component_blocked` — p×d (block_len=18): asserts 4 contiguous
  component-leading blocks, scalar block all-zero, cross-checked against host ref
  (confirms NOT `n*4+comp` interleaved).

## Verification

- `cargo build -p cintx-cubecl --locked --features cpu` — succeeds (no errors).
- `cargo test -p cintx-cubecl --features cpu sigma_p` — 3 passed, 0 failed.
- `cargo clippy -p cintx-cubecl --features cpu --lib` — clean for sigma_p.
- Grep gates: `#[cube]` count = 5 (>=1); `mod sigma_p` in mod.rs = 1; Pauli mix
  `g1.*g0.*g0` = 4 (>=1); zero scalar slot `0.0` present; output uses `block_len`
  block strides (component-blocked, no real `*4+comp` interleaved write — only
  doc-comment references to the avoided layout); `tensor_rank`/rank present (26).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] clippy `0u32 * block_len` always-zero (deny-level under `-D warnings`)**
- **Found during:** GREEN clippy gate.
- **Issue:** the gc_x write was `base + 0u32 * block_len + elem`, which clippy
  flags as `this operation will always return zero` (a hard error under the
  project's clippy config).
- **Fix:** simplified to `base + elem` (block 0); gc_y/gc_z/gc_1 keep
  `1/2/3 * block_len` strides. Semantically identical.
- **Files modified:** crates/cintx-cubecl/src/kernels/sigma_p.rs

**2. [Rule 3 - Blocking] backend dispatch arms needed per-variant `#[cfg(feature)]`**
- **Found during:** initial `--features cpu` build.
- **Issue:** the Wgpu/Cuda/Rocm/Metal `ResolvedBackend` variants and their
  runtime crates (`cubecl_wgpu`/`cubecl_cuda`/`cubecl_hip`) are feature-gated; an
  ungated `match` arm fails to compile under `--features cpu`.
- **Fix:** added `#[cfg(feature = "wgpu"/"cuda"/"rocm"/"metal")]` to each arm,
  mirroring `one_electron.rs::run_1e_grad_bra_on_backend`.
- **Files modified:** crates/cintx-cubecl/src/kernels/sigma_p.rs

**3. [Rule 1 - Style] doc-list overindentation clippy warning**
- **Found during:** GREEN clippy gate.
- **Issue:** 6-space-indented doc lines parsed as overindented list items.
- **Fix:** converted the algorithm pseudo-code into ```text fenced blocks.
- **Files modified:** crates/cintx-cubecl/src/kernels/sigma_p.rs

`cargo fmt` also normalized whitespace in two pre-existing files
(`one_electron.rs`, `center_2c2e.rs`) it touched while formatting the crate; those
are not part of this task and were not staged.

## Known Stubs

`run_sigma_p_on_backend` is `#[allow(dead_code)]` — it is intentionally not yet
called from a live launcher. The live `int1e_sp` Spinor dispatch arm (which feeds
these gc blocks into `cart_to_spinor_si_2d`) is wired in a later Phase-28 plan
task. This is the planned phase boundary (this plan delivers the assembler
foundation only), documented in 28-CONTEXT D-03/D-04, not a goal-blocking stub —
the assembler is fully exercised by the device-vs-host tests.

## Threat Surface

No new external surface. Pure on-device compute (STRIDE register T-28-02-01/02
both mitigated: gout component ordering transcribed verbatim with scalar slot 0
asserted == 0 for every cart n; component-blocked layout asserted by the layout
test). No network, auth, file, or user-input surface introduced.

## Self-Check: PASSED

- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — FOUND
- `pub mod sigma_p;` in `crates/cintx-cubecl/src/kernels/mod.rs` — FOUND
- 3/3 sigma_p tests pass under `--features cpu`
