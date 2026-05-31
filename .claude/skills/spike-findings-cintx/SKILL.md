---
name: spike-findings-cintx
description: Implementation blueprint from spike experiments. Verified knowledge of the cintx device output layout — the per-component axis-fold offset formula, rank-tier coverage, cart↔sph fold invariance, and the dual hand-derived+vendor verification method. Load when registering a new integral family, writing/altering an oracle parity test, or touching output-buffer layout/stride or the c2s transform.
---

<context>
## Project: cintx

cintx is a Rust reimplementation of libcint with result compatibility as the primary goal,
CubeCL as the compute backend. These findings concern the **device output buffer layout**:
how every integral family's per-component axis-fold is laid out, how it stays invariant
across cartesian and spherical paths, and how to prove a new family matches vendored
libcint byte-for-byte.

Spike sessions wrapped: 2026-05-31
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
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| Device block layout | references/device-block-layout.md | `out[comp*(ni*nj)+j*ni+i]` component-leading, pinned by hand-derived `R_c·S` + vendor byte-identity across rank 3/9/27/81 |
| Cart↔sph fold invariance | references/cart-sph-fold-invariance.md | sph device path == per-component `c2s(cart)` exactly; sph cannot drift independently |

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
</metadata>
