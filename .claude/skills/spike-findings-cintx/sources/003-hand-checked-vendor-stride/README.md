---
spike: 003
name: hand-checked-vendor-stride
type: standard
validates: "Given a non-square block (single-s self-pair for Arm 1; p x d for Arm 2), when cintx device output is checked against a hand-derived analytic value AND vendored libcint, then every (comp,i,j) lands at comp*(ni*nj)+j*ni+i — component identity pinned by hand, i-fastest orientation pinned by vendor, across rank 3/9/27/81 and both cart+sph paths"
verdict: VALIDATED
related: [001, 002]
tags: [layout, vendor, orientation, transpose, hand-checked, cart, sph]
---

# Spike 003: hand-checked-vendor-stride

## What This Validates

The FULL device block offset formula `out[comp * (ni*nj) + (j*ni + i)]`, confirmed two
independent ways:

- **Arm 1 (vendor-free, hand-derived):** component-axis identity x/y/z and component-leading
  layout, via the algebraic invariant `<g_R | r_c | g_R> = R_c * S`.
- **Arm 2 (vendor-pinned):** the i-fastest (column-major) orientation *within* each
  component, on a non-square `p x d` block where `ni>1` AND `nj>1` (the orientation spike
  001 structurally could not see), across all rank tiers and both transform paths.

## Research

- Position operator origin semantics: `int1e_r` reads the gauge origin via libcint's
  `G1E_RCJ` (`drj = rj - env[PTR_COMMON_ORIG]`), confirmed in
  `crates/cintx-cubecl/src/kernels/one_electron.rs:2280-2520`. Setting `common_orig = 0`
  reduces the operator to bare `r`, so `<g_R|r_c|g_R> = R_c * S` exactly (S factors out,
  so the relation is normalization-independent).
- Custom-fixture `atm/bas/env` layout modeled on
  `crates/cintx-oracle/tests/moment_genctr_parity.rs:52-117`.
- `d`-shell (l=2) is supported by the cintx moment kernels (Arm 2 evaluates 3x6 cart /
  3x5 sph blocks at every tier) — STO-3G alone has only s,p, which is why spike 001 had a
  unit axis on every non-square block.

## How to Run

```bash
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_003 -- --ignored --nocapture
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test spike_axis_fold_003 -- --ignored --nocapture
```

## Investigation Trail

1. **Arm 1 built first** — single normalized s-primitive at hand-chosen
   `R = (0.30, -0.50, 0.70)`, gauge origin 0. Read `S` from cintx `int1e_ovlp`, then
   asserted `r_block[c] == R_c * S` per component.
   - Result: **exact** match, `rel = 0.00e0` for all three components. The relation holds
     to the bit. This simultaneously pins (a) component 0/1/2 = x/y/z, (b) component-leading
     layout, (c) that the gauge-origin=0 path is honored. A component swap, a
     component-interleave, or a wrong origin would each break it. No vendor used.
2. **Arm 2 negative control (vendor-free) added** — `p x d` two-center block (3x6 cart,
   3x5 sph). Built `to_j_fastest()` reinterpretation and asserted `cintx != cintx_jfastest`
   at every tier (self_diff = 34/100/298/892 cart, 28/89/268/809 sph). So the fixture is
   genuinely orientation-sensitive — the upcoming vendor comparison is not vacuous.
3. **Arm 2 vendor pin** — vendored libcint linked. For EVERY tier r/rr/rrr/rrrr and BOTH
   paths: `mm(vendor, cintx) == 0` (claimed i-fastest layout is right) AND
   `mm(vendor, j-fastest) > 0` (the OTHER orientation is decisively wrong). Orientation is
   therefore *pinned*, not a transpose-symmetric coincidence.

## Results

**VALIDATED.** The device block layout is exactly `out[comp*(ni*nj) + j*ni + i]`:

| Aspect | Method | Result |
|--------|--------|--------|
| Component identity (x/y/z) | hand-derived `R_c * S` | exact, rel=0 |
| Component-leading + origin=0 | hand-derived | exact |
| i-fastest orientation (within comp) | vendor, p×d, all tiers, cart+sph | `mm(vendor,cintx)=0`, j-fastest wrong |
| Negative control valid | vendor-free | i-fastest ≠ j-fastest everywhere |

**Surprise / signal:** the orientation gap is large and grows with rank
(34→100→298→892 cart), so a transposed-block regression at high rank would be loud, not
subtle — good news for catching it. Arm 1's exactness (not just within-tolerance) shows the
moment path carries no per-component reordering or scaling whatsoever for the rank-3 base.

**Carry-forward to the build:** every NEW family's parity test must use a block with
`ni>1` AND `nj>1` (a `d` shell) to exercise i/j orientation — an s×p block (the default
STO-3G non-square pair) has a unit axis and is orientation-blind.
