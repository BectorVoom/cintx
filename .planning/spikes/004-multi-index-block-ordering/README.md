---
spike: 004
name: multi-index-block-ordering
type: integration
validates: "Given 3-/4-index families (int2e_ip1, int3c2e_ip2) on a non-square shell tuple with all axes >1, when evaluated into a component-leading buffer, then the inner block is i-fastest then j/k/[l] (the documented [nl][nk][nj][ni] order) — pinned by mm(vendor,cintx)=0 while every non-identity axis permutation diverges, cart+sph"
verdict: VALIDATED
related: [001, 003]
tags: [layout, multi-index, 2e, 3c2e, vendor, orientation]
---

# Spike 004: multi-index-block-ordering

## What This Validates

The integration gap left by spikes 001/003: they pinned `out[comp*(ni*nj)+j*ni+i]` for a
**2-index** inner block only, yet the skill states the contract as universal. This spike
generalizes the i/j transpose-disagreement method to N axes and confirms the inner block
ordering for **3-index** (`int3c2e_ip2`) and **4-index** (`int2e_ip1`) families:

```
2e   : out[comp*(ni*nj*nk*nl) + (((l*nk+k)*nj+j)*ni+i)]   i (bra1) fastest
3c2e : out[comp*(ni*nj*nk)    + ((k*nj+j)*ni+i)]          i fastest
```

## Research

- Documented 2e order (`two_electron_ip1_parity.rs:11-16`): component-leading F-order
  `out[comp*(ni*nj*nk*nl)+n]`, `n` walking `[nl][nk][nj][ni]` with **ni fastest**.
- 3c2e (`int3c2e_ip2_parity.rs:144`): `n_elem = 3*ni*nj*nk`, component-leading.
- Fixtures: copied `build_spd_fixture` (s/p/d on 2 atoms). Quartet `(0,p)(0,d)(1,p)(1,d)`
  → extents `[3,6,3,6]` cart (all axes >1, fully non-symmetric); triple `(0,p)(0,d)(1,p)`
  → `[3,6,3]`. Vendor `vendor_int2e_ip1_{cart,sph}` (`&[i32;4]`), `vendor_int3c2e_ip2_*`
  (`&[i32;3]`).
- Generalized `reindex(buf, rank, extents, perm)` (axis 0 = fastest): rebuilds the inner
  block under any axis permutation, so a non-identity perm is the N-axis analog of spike
  003's `to_j_fastest`.

## How to Run

```bash
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_004 -- --ignored --nocapture
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test spike_axis_fold_004 -- --ignored --nocapture
```

## Investigation Trail

1. Chose a quartet/triple with ALL axes >1 (`[3,6,3,6]` / `[3,6,3]` cart) so any wrong
   index order is observable — a tuple with a unit axis would hide an adjacent swap, the
   multi-index analog of spike 001's orientation-blind s×p block.
2. Vendor-free: confirmed component-leading split (`comp_stride == ni*nj*nk*[nl]`) and that
   three non-identity permutations each change the buffer (order-sensitive fixture).
3. Vendor: `mm(vendor,cintx) == 0` for both families, both paths. Every non-identity axis
   permutation diverges from vendor (min mismatch 350/20 cart, 238/20 sph) → the i-fastest
   order is the one libcint uses, not a symmetric-block coincidence.

## Results

**VALIDATED.** The component-leading offset formula generalizes cleanly to 3 and 4 indices;
the inner block is i (bra1) fastest then j/k/[l], byte-identical to libcint:

| Family | indices | rep | extents | mm(vendor,cintx) | min mm(vendor,perm) |
|--------|---------|-----|---------|------------------|----------------------|
| int2e_ip1 | 4 | cart | [3,6,3,6] | 0 | 350 |
| int2e_ip1 | 4 | sph | [3,5,3,5] | 0 | 238 |
| int3c2e_ip2 | 3 | cart | [3,6,3] | 0 | 20 |
| int3c2e_ip2 | 3 | sph | [3,5,3] | 0 | 20 |

**Signal:** the skill's "universal layout" claim is now backed for 2/3/4-index families, not
just the 1e moment ladder. The `reindex(extents, perm)` helper is the reusable N-axis
order-pinning primitive — fold it into the new-family parity template alongside the d-shell
orientation rule. **Carry-forward:** always pick a tuple with EVERY axis >1 for a
multi-index layout test (a unit axis hides an adjacent-axis swap).
