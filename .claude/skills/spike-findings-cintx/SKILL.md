---
name: spike-findings-cintx
description: Implementation blueprint from spike experiments. Verified knowledge of the cintx device output layout — the per-component axis-fold offset formula, rank-tier coverage, 3-/4-index inner-block ordering, nctr>1 contraction composition, cart↔sph fold invariance, the interleaved-complex spinor divergence, and the dual hand-derived+vendor verification method. Load when registering a new integral family, writing/altering an oracle parity test, or touching output-buffer layout/stride, contraction blocking, the c2s transform, or spinor output.
---

<context>
## Project: cintx

cintx is a Rust reimplementation of libcint with result compatibility as the primary goal,
CubeCL as the compute backend. These findings concern the **device output buffer layout**:
how every integral family's per-component axis-fold is laid out, how it stays invariant
across cartesian and spherical paths, and how to prove a new family matches vendored
libcint byte-for-byte.

Spike sessions wrapped: 2026-05-31 (001–003 + frontier 004–006)
</context>

<requirements>
## Requirements (non-negotiable for the real build)

- Output buffer is **component-leading**: `comp` is the slowest axis, `comp_stride == ni*nj`.
- Per-component block is **column-major / bra-fastest**: `out[comp*(ni*nj) + (j*ni + i)]`,
  `i` (bra) stride 1, `j` (ket) stride `ni`. Verified across rank **3 / 9 / 27 / 81**.
- Cart and sph paths fold the component axis **identically**; only block dims differ
  (`ncart`→`nsph`). `sph_block[c] == cart_to_sph_1e(cart_block[c])` exactly.
- A `rank*ni*nj` buffer splits into exactly `rank` non-overlapping component slices — no
  trailing component truncated (D-08), no stuck-at-zero component (WR-04).
- Any layout/orientation verification MUST use a **non-square block with `ni>1` AND `nj>1`**
  (a `d` shell) — an s×p block is orientation-blind (D-07).
- The formula generalizes to **3-/4-index** families (i bra1 fastest) and composes with
  **nctr>1** (contraction-major `i_global=ci*di+ic`) — both vendor-identical across tiers.
- **Spinor is the one divergence**: interleaved-complex (`rank*ni_sp*nj_sp*2`, re/im fastest,
  `ni_sp=4l+2` @ kappa=0); component-leading + ket-major still hold around the interleave.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| Device block layout | references/device-block-layout.md | `out[comp*(ni*nj)+j*ni+i]` component-leading; pinned by hand-derived `R_c·S` + vendor across rank 3/9/27/81; generalizes to 3-/4-index (i-fastest) and nctr>1 (contraction-major) |
| Cart↔sph fold invariance | references/cart-sph-fold-invariance.md | sph device path == per-component `c2s(cart)` exactly; sph cannot drift independently |
| Spinor layout (divergence) | references/spinor-layout.md | interleaved-complex `rank*ni_sp*nj_sp*2` (re/im fastest, `ni_sp=4l+2`); component-leading + ket-major preserved around the interleave |

## Source Files

Original spike source files (runnable `#[ignore]`d harnesses + READMEs) are preserved in
`sources/` for complete reference. Run a harness with:
`cargo test -p cintx-oracle --features cpu --test spike_axis_fold_00N -- --ignored --nocapture`
(prefix `CINTX_ORACLE_BUILD_VENDOR=1` for the libcint ground-truth arm).
</findings_index>

<metadata>
## Processed Spikes

- 001-axis-fold-stride-probe
- 002-cart-vs-sph-fold-invariance
- 003-hand-checked-vendor-stride
- 004-multi-index-block-ordering
- 005-nctr-axisfold-composition
- 006-spinor-layout-divergence
</metadata>
