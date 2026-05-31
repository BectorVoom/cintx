# Spike Manifest

## Idea

Exercise the cintx **per-component axis-fold** across all derivative/multipole **rank
tiers** (3, 9, 27, 81) and **both transform paths** (cartesian + spherical), and confirm
the **device block layout/stride** of the on-device kernel output against **hand-checked
vendor values** (hand-derived layout contract + vendored libcint FFI when available).

The cintx output-layout contract is component-leading column-major:

```
out[comp * (ni*nj) + (j*ni + i)]
     └ component (slowest, stride ni*nj)
                    └ ket j (stride ni)
                            └ bra i (stride 1, fastest)
```

Cart→sph changes **only** the per-component block dims (`ncart`→`nsph`); the component
axis stays outermost. The existing oracle suite asserts cintx==vendor element-wise but
never independently *recovers and names* this stride structure across rank tiers. These
spikes do.

## Requirements

Design decisions that emerged during spiking. Non-negotiable for the real build.

- Output buffer is **component-leading** (`comp` is the slowest axis): `comp_stride == ni*nj`.
- Per-component block is **column-major / bra-fastest**: `out[... + j*ni + i]`, `i` stride 1.
- Cart and sph paths fold the component axis **identically**; only `ni,nj` differ (ncart vs nsph).
- A `rank*ni*nj` buffer must split into exactly `rank` non-overlapping component slices with
  **no trailing component truncated** (D-08) and **no stuck-at-zero** component (WR-04).
- Layout forensics must use a **non-square** block (D-07) so a transpose/permute bug surfaces.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | axis-fold-stride-probe | standard | Per-component stride == `ni*nj`, clean `rank` slices, no truncation, across rank 3/9/27/81 (cart) | ✓ VALIDATED | layout, stride, rank-tiers |
| 003 | hand-checked-vendor-stride | standard | Non-square block: every `(comp,i,j)` lands at `comp*(ni*nj)+j*ni+i` vs hand-derived table + vendor FFI | ✓ VALIDATED | layout, vendor, orientation, transpose |
| 002 | cart-vs-sph-fold-invariance | standard | Component axis outermost in both paths; fold differs only by block dims + per-component c2s transform | ✓ VALIDATED | cart, sph, transform, invariance |
| 004 | multi-index-block-ordering | integration | 3-/4-index families: inner block is `ni`-fastest then nj/nk/[nl] per documented order — pinned by permutation-disagreement vs vendor, cart+sph | ✓ VALIDATED | layout, multi-index, 2e, 3c2e, vendor |
| 005 | nctr-axisfold-composition | frontier | nctr>1 at rank 3/9/27/81: `out[comp*(ni_full*nj_full)+(j_global*ni_full+i_global)]`, `i_global=ci*di+i` — vendor byte-identity across tiers | PENDING | layout, nctr, contraction, rank-tiers, vendor |
| 006 | spinor-layout-divergence | frontier | rank-3 spinor: interleaved-complex alpha/beta (NOT component-leading real); component axis folds around the complex interleave vs vendor | PENDING | layout, spinor, complex, divergence, vendor |

## Probe Target

`int1e_r / rr / rrr / rrrr` — the position-multipole ladder. A single uniform family that
spans exactly the four rank tiers (3/9/27/81) with cart+sph `RawApiId` consts and matching
`vendor_ffi::vendor_int1e_r{,r,rr,rrr}_{cart,sph}` references, on the
`build_h2o_sto3g_common_orig()` fixture. Spike 003 adds a custom d-shell fixture so a
non-square block has both `ni>1` and `nj>1` (orientation observable).

## How to Run

The runnable harnesses live as `#[ignore]`d integration tests in
`crates/cintx-oracle/tests/spike_axis_fold_*.rs` (gated so they never run in normal CI).
Recorded source copies live in each spike dir.

```bash
# structural (vendor-free) forensics + layout map:
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_001 -- --ignored --nocapture

# with vendored libcint ground truth:
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test spike_axis_fold_003 -- --ignored --nocapture
```
