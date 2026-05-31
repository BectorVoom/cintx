# Device Block Layout (per-component axis-fold)

The proven memory layout of every cintx integral-family device kernel output, and the
proven way to verify a new family matches it.

## Requirements

- Output buffer is **component-leading**: `comp` is the slowest axis, `comp_stride == ni*nj`.
- Per-component block is **column-major / bra-fastest**: index `j*ni + i`, `i` (bra) stride 1.
- Full formula: `out[comp * (ni*nj) + (j*ni + i)]`, verified across rank **3/9/27/81**.
- A `rank*ni*nj` buffer splits into exactly `rank` non-overlapping component slices — **no
  trailing component truncated** (D-08), **no stuck-at-zero** component (WR-04).
- Layout/orientation verification MUST use a **non-square** block (D-07).

## How to Build It

A layout-regression test is a workspace-linked integration test driving the real
`eval_raw` path. Pattern (from `spike_axis_fold_001.rs` / `_003.rs`):

```rust
#![cfg(feature = "cpu")]
use cintx_compat::raw::{ANG_OF, BAS_SLOTS, RawApiId, eval_raw};
use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;

// Evaluate one shell pair into a component-leading rank*ni*nj buffer.
let (atm, bas, env) = build_h2o_sto3g_common_orig();
let (si, sj) = (0, 2);                       // O-1s × O-2p, non-square
let ni = ncart(bas[si*BAS_SLOTS+ANG_OF]);    // ncart(l) = (l+1)(l+2)/2
let nj = ncart(bas[sj*BAS_SLOTS+ANG_OF]);
let mut out = vec![0.0_f64; rank * ni * nj];
unsafe { eval_raw(RawApiId::INT1E_RR_CART, Some(&mut out), None,
                  &[si as i32, sj as i32], &atm, &bas, &env, None, None).unwrap(); }
// Contract: out.len()==rank*ni*nj ; out.len()/rank == ni*nj (comp_stride).
```

**Probe ladder** — `int1e_r / rr / rrr / rrrr` is the one uniform family spanning rank
3/9/27/81, with `RawApiId::INT1E_R{,R,RR,RRR}_{CART,SPH}` and matching
`cintx_oracle::vendor_ffi::vendor_int1e_r{,r,rr,rrr}_{cart,sph}`.

**Dual ground truth** (always assert the first; gate the second):

1. *Hand-derived, vendor-free.* For a single normalized s-primitive at hand-chosen `R`
   with gauge origin 0, `<g_R | r_c | g_R> = R_c * S`. Read `S` from cintx's own
   `INT1E_OVLP_CART`, then assert `r_block[c] == R[c]*S` — pins component identity x/y/z,
   component-leading layout, and origin handling, exactly (rel=0), no libcint. (Arm 1 of 003.)
2. *Vendor byte-identity.* Behind `#[cfg(has_vendor_libcint)]`, compare element-wise to
   `vendor_*` at `atol=1e-12`. This is what actually pins **component-outermost** (a
   component-interleaved layout has identical length and passes the structural checks).
   Build with `CINTX_ORACLE_BUILD_VENDOR=1` — vendored libcint 6.1.3 links cleanly here.

**Orientation (i-fastest) pin** — needs a block with `ni>1` AND `nj>1`, i.e. a `d` shell.
Build a `p × d` two-center fixture (see `spike_axis_fold_003.rs::build_p_times_d`), then:

```rust
let cintx_jf = to_j_fastest(&cintx, rank, ni, nj);     // reinterpret as i*nj+j
assert!(mismatches(&cintx, &cintx_jf) > 0);            // negative control: orientations differ
#[cfg(has_vendor_libcint)] {
    assert_eq!(mismatches(&vendor, &cintx), 0);        // claimed i-fastest is right
    assert!(mismatches(&vendor, &cintx_jf) > 0);       // j-fastest is decisively wrong
}
```

## What to Avoid

- **Don't verify layout with an s×p (STO-3G) non-square block** — it has a unit axis
  (`ni==1` or `nj==1`), so i-fastest vs j-fastest is unobservable. Orientation bugs hide.
  Use a `d` shell (`p×d` = 3×6 cart / 3×5 sph).
- **Don't trust buffer length as a layout check** — component-leading and
  component-interleaved have the *same* length. You need element-wise vendor identity or a
  hand-derived per-component value.
- **Don't re-add `if dst < staging.len()` guards** to per-chunk staging — family kernels
  are monolithic whole-block writers; staging must be FULL-block sized (Phase 25 lesson).
- **Don't assume a "legitimately-zero component" path is covered** — the moment fixtures
  fully populate every component (81/81 at rank 81), so they can't exercise a correctly-
  skipped-zero-component case. Add a parity-zero fixture if that path needs coverage.

## Constraints

- Rank tiers in the manifest lock (`compiled_manifest.lock.json`, 347 entries): 3 (×106),
  9 (×68), 27 (×15: `ipipipnuc/ipipiprinv/ipipnucip/ipiprinvip/rrr`), 81 (×15:
  `ipipipiprinv/ipipiprinvip/ipiprinvipip/rrrr/ipip1ipip2`), plus scalar/1, 4.
- Vendor parity is double-gated: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`
  (`has_vendor_libcint` cfg). Without both, the vendor arm compiles out silently.
- `int1e_r`-family reads the gauge origin (`drj = rj - env[PTR_COMMON_ORIG]`); set it to 0
  for the clean `R_c·S` hand-check.

## Multi-index families (3-/4-index) — spike 004

The component-leading formula generalizes; the inner block is **i (bra1) fastest**:

```
2e   : out[comp*(ni*nj*nk*nl) + (((l*nk+k)*nj+j)*ni+i)]
3c2e : out[comp*(ni*nj*nk)    + ((k*nj+j)*ni+i)]
```

Pin the ordering with the N-axis generalization of the i/j transpose test —
`reindex(buf, rank, extents, perm)` with `extents` in fastest→slowest order. Assert
`mm(vendor,cintx)==0` and that **every** non-identity axis permutation diverges. Verified
byte-identical for `int2e_ip1` (4-index) and `int3c2e_ip2` (3-index), cart+sph.

**Critical:** pick a tuple where EVERY axis >1 (e.g. `(0,p)(0,d)(1,p)(1,d)` → `[3,6,3,6]`).
A unit axis hides an adjacent-axis swap, exactly like the orientation-blind s×p block.

## General contraction (nctr>1) — spike 005

Contraction is the **MAJOR** within-axis index; it composes with component-leading folding:

```
i_global = ci*di + ic        (ci = contraction 0..nctr_i, ic = angular 0..di)
out[comp*(ni_full*nj_full) + (j_global*ni_full + i_global)]      ni_full = nctr_i*di
```

The env coefficient block is COLUMN-major (`env[ci*nprim+ip]`); cintx transposes to
row-major internally — the historical nctr-transpose bug class. Verified vendor-identical
across rank 3/9/27/81 (bra p nctr=2 × ket d). Negative control: reinterpreting the i-axis as
contraction-MINOR (`i_alt = ic*nctr_i + ci`) diverges from vendor. Probe nctr>1 **across all
rank tiers**, not just rank-9 (which is all the existing `moment_genctr` test covers).

## Origin

Synthesized from spikes: 001, 003, 004, 005
Source files available in: sources/001-axis-fold-stride-probe/, sources/003-hand-checked-vendor-stride/, sources/004-multi-index-block-ordering/, sources/005-nctr-axisfold-composition/
