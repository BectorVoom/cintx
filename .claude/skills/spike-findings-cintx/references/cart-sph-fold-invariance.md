# Cart ↔ Sph Fold Invariance

The spherical device path is mechanically `per-component c2s` of the cartesian device path
— not an independent layout that can drift.

## Requirements

- Cart and sph paths fold the component axis **identically**; only the per-component block
  dims differ (`ncart`→`nsph`). The component axis is never touched by the transform.
- Same component **count** (`rank`) in both paths; component `c` of cart → component `c` of sph.
- Per component: `sph_block[c] == cart_to_sph_1e(cart_block[c])` **exactly** (Δ=0).

## How to Build It

```rust
use cintx_cubecl::transform::c2s::cart_to_sph_1e;   // the exact routine cintx uses

// Evaluate the SAME shell pair via cart and sph RawApiIds into component-leading buffers.
let block_cart = ncart(li)*ncart(lj);   // e.g. p×d = 3×6 = 18
let block_sph  = nsph(li)*nsph(lj);     //          = 3×5 = 15
// ... eval_raw(t.cart, ...) -> cart ; eval_raw(t.sph, ...) -> sph ...

for c in 0..rank {
    let cart_block = &cart[c*block_cart..(c+1)*block_cart];
    let sph_block  = &sph[c*block_sph..(c+1)*block_sph];
    let mut recon = vec![0.0_f64; block_sph];
    cart_to_sph_1e::<f64>(cart_block, &mut recon, li, lj);   // input layout j*nci+i
    assert!(recon.iter().zip(sph_block).all(|(r,s)| (r-s).abs() <= 1e-12));
}
```

`cart_to_sph_1e` (`crates/cintx-cubecl/src/transform/c2s.rs:137`) expects the per-component
block in `j*nci + i` order — the same layout `device-block-layout.md` confirms the kernel
emits. It is independently vendor-checked by `cintc2s_bra_sph_parity.rs`, so proving
`sph_device == c2s(cart_device)` closes the loop without needing libcint here.

Use a `d` ket (`p × d`): the d shell's `ncart=6 → nsph=5` is the first non-trivial sph
transform — an s/p-only block would make the transform an identity and prove nothing.

## What to Avoid

- **Don't write a separate vendor-parity gate for the sph path of a layout-only change.**
  The sph fold cannot diverge independently from cart + the already-vendor-checked c2s.
  Guard layout on the cart path; the sph path follows mechanically.
- **Don't probe invariance on an s/p block** — the transform is (near-)identity there.

## Constraints

- Verified exactly (worst Δ = 0.00e0) at every rank tier 3/9/27/81 on a p×d block.
- `cart_to_sph_1e` is generic over `CintFloat`; the c2s coefficient table is FROZEN f64,
  cast per-accumulation. f64 monomorphization is byte-identical to the device path.

## Origin

Synthesized from spikes: 002
Source files available in: sources/002-cart-vs-sph-fold-invariance/
