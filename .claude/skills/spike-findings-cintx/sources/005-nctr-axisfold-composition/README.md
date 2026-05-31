---
spike: 005
name: nctr-axisfold-composition
type: frontier
validates: "Given a general-contraction (bra p nctr=2 × ket d) block at rank 3/9/27/81, when evaluated, then comp_stride == ni_full*nj_full and the contraction-MAJOR within-axis interleave (i_global=ci*di+ic) composes with component-leading folding — vendor byte-identity at every tier, contraction-minor reinterpretation rejected"
verdict: VALIDATED
related: [001, 003, 004]
tags: [layout, nctr, contraction, rank-tiers, vendor]
---

# Spike 005: nctr-axisfold-composition

## What This Validates

Spikes 001/003/004 used `nctr==1`. The component-leading fold is assumed to compose with
general-contraction blocking, where each axis index is **contraction-major**:

```
i_global = ci*di + ic     (ci = contraction 0..nctr_i, ic = angular 0..di)
out[comp * (ni_full*nj_full) + (j_global*ni_full + i_global)]      ni_full = nctr_i*di
```

This is precisely the row/column-major interleave whose transpose was a **latent
historical bug** (the nctr>1 coefficient-transpose family — latent because all prior
fixtures were nctr==1). This spike probes the composition across all four rank tiers.

## Research

- Confirmed nctr>1 1e block order (`moment_genctr_parity.rs:16-23`): per-component dense
  `[ni_full, nj_full]` in i-fastest column-major, contraction the MAJOR index within each
  axis (`i_global = ci*di + i_idx`), components outermost.
- Fixture mirrors `build_moment_genctr_fixture` (bra p **nctr=2**, ket d nctr=1, two
  centers, non-zero gauge origin, column-major env coeff block).
- `i_axis_contraction_minor()` is the negative control: reinterprets the i-axis as
  contraction-MINOR (`i_alt = ic*nctr_i + ci`) — only observable because nctr_i>1.

## How to Run

```bash
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_005 -- --ignored --nocapture
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test spike_axis_fold_005 -- --ignored --nocapture
```

## Investigation Trail

1. Reused the known-good nctr>1 moment fixture; extended the probe from rank-9 (the only
   tier `moment_genctr_parity` covers) to the full ladder r/rr/rrr/rrrr.
2. Vendor-free: `comp_stride == ni_full*nj_full` (36 cart / 30 sph) at every tier; the
   contraction-minor reinterpretation always changes the buffer → nctr structure is real.
3. Vendor: `mm(vendor,cintx)==0` at every tier and path; `mm(vendor, contraction-minor)`
   grows with rank (64→200→616→1880 cart) → the contraction-major composition is the one
   libcint uses, decisively, not a small-block coincidence.

## Results

**VALIDATED.** Contraction blocking composes with the per-component axis-fold at every rank
tier, byte-identical to vendored libcint:

| rep | block | r (3) | rr (9) | rrr (27) | rrrr (81) |
|-----|-------|-------|--------|----------|-----------|
| cart | 36 | mm=0 / minor 64 | mm=0 / 200 | mm=0 / 616 | mm=0 / 1880 |
| sph | 30 | mm=0 / 52 | mm=0 / 164 | mm=0 / 508 | mm=0 / 1556 |

(`mm` = mm(vendor,cintx); `minor` = mm(vendor, contraction-minor reinterpretation).)

**Signal:** the historical nctr>1 transpose-bug class is closed for the moment ladder across
ALL rank tiers, not just the single rank-9 case the existing `moment_genctr` test covers.
**Carry-forward (already in CONVENTIONS):** every new family needs an nctr>1 case; this
spike shows that case must also span rank tiers, and the contraction-minor negative control
is a cheap way to prove the test is actually order-sensitive.
