---
spike: 002
name: cart-vs-sph-fold-invariance
type: standard
validates: "Given int1e_r/rr/rrr/rrrr on a p×d block, when each is evaluated via the cart path and the sph path, then both keep the component axis outermost with the same component count, and per component sph_block[c] == cart_to_sph_1e(cart_block[c]) exactly — so the fold is path-invariant and the c2s transform never touches the component axis"
verdict: VALIDATED
related: [001, 003]
tags: [cart, sph, transform, invariance, layout]
---

# Spike 002: cart-vs-sph-fold-invariance

## What This Validates

That the per-component axis-fold is IDENTICAL on the cartesian and spherical device
paths. The only path difference is the per-component block dims (`ncart`→`nsph`); the
component axis is folded the same way and is never re-ordered by the c2s transform.

## Research

- `cintx_cubecl::transform::c2s::cart_to_sph_1e<F>(cart_buf, sph_buf, li, lj)` is the
  exact per-block transform cintx uses (`crates/cintx-cubecl/src/transform/c2s.rs:137`).
  Input layout `[ncj*nci]` j-outer/i-inner (`j*nci+i`) — the same per-component layout
  spike 003 confirmed the device emits.
- That `cart_to_sph_1e` is itself correct vs libcint is independently established by
  `crates/cintx-oracle/tests/cintc2s_bra_sph_parity.rs` (vendor parity over l=0..4).
  So proving `sph_device == c2s(cart_device)` per component closes the loop without
  needing vendor here.
- Probe block: `p × d` (l=1 × l=2) — the d ket gives the first non-trivial sph transform
  (ncart=6 → nsph=5), so the invariance is exercised on a real transform, not an identity.

## How to Run

```bash
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_002 -- --ignored --nocapture
```

## Investigation Trail

1. Evaluated each tier r/rr/rrr/rrrr twice (cart RawApiId, sph RawApiId) into
   component-leading buffers (cart block 3×6=18, sph block 3×5=15).
2. Asserted component-count invariance: `cart.len()/rank == 18`, `sph.len()/rank == 15`,
   same `rank` in both → the transform changes block dims only, not the component axis.
3. For every component `c`, reconstructed `cart_to_sph_1e(cart_block[c])` and compared to
   `sph_block[c]`.

## Results

**VALIDATED.** At every rank tier, `sph_block[c] == cart_to_sph_1e(cart_block[c])`
**exactly** (`worst Δ = 0.00e0`), for all `c` in `0..rank`:

| Tier | rank | cart_len | sph_len | per-comp c2s(cart)==sph | worst Δ |
|------|------|----------|---------|--------------------------|---------|
| int1e_r | 3 | 54 | 45 | ✓ | 0 |
| int1e_rr | 9 | 162 | 135 | ✓ | 0 |
| int1e_rrr | 27 | 486 | 405 | ✓ | 0 |
| int1e_rrrr | 81 | 1458 | 1215 | ✓ | 0 |

**Signal:** the sph device path is not a separate layout to verify — it is mechanically
`per-component c2s` of the cart path. So a layout regression need only be guarded on ONE
path (cart) plus the already-vendor-checked c2s; the sph fold cannot diverge independently.
Component `c` of cart maps to component `c` of sph with no reordering at any rank.
