---
spike: 001
name: axis-fold-stride-probe
type: standard
validates: "Given int1e_r/rr/rrr/rrrr at rank 3/9/27/81, when evaluated into a component-leading cart buffer on a non-square block, then comp_stride == ni*nj, the buffer splits into exactly `rank` clean component slices with no truncation/stuck-at-zero, and (vendor) every element is byte-identical to libcint"
verdict: VALIDATED
related: [003, 002]
tags: [layout, stride, rank-tiers, cart, vendor]
---

# Spike 001: axis-fold-stride-probe

## What This Validates

Given the position-multipole ladder `int1e_r / rr / rrr / rrrr` (the one uniform family
spanning exactly rank **3 / 9 / 27 / 81**), when evaluated through `eval_raw` on the
CubeCL `CpuRuntime` into a `rank*ni*nj` cartesian buffer for a **non-square** shell pair,
then the per-component axis-fold obeys the component-leading column-major contract:

```
out[comp * (ni*nj) + (j*ni + i)]      comp_stride = ni*nj   (component is the SLOWEST axis)
```

## Research

- Layout contract source: `crates/cintx-runtime/src/planner.rs` `build_output_layout()`
  (`component_axis_leading: true`, `staging_elements = base * component_rank`).
- Manifest rank tiers (`compiled_manifest.lock.json`, 347 entries): rank-3 ×106, rank-9 ×68,
  rank-27 ×15, rank-81 ×15. The `r/rr/rrr/rrrr` ladder is the cleanest family hitting all four.
- Harness modeled on `crates/cintx-oracle/tests/moment_common.rs` (`collect_cintx_block`,
  `eval_raw(api, Some(&mut out), None, &shls, atm, bas, env, None, None)`), fixture
  `cintx_oracle::fixtures::build_h2o_sto3g_common_orig()`, vendor refs
  `cintx_oracle::vendor_ffi::vendor_int1e_r{,r,rr,rrr}_cart`.
- **Historical risk this targets:** a `component_rank` manifest *truncation* bug once
  dropped trailing components (latent because the buffer length still looked plausible).
  This probe asserts `len == rank*ni*nj` at every tier, exercising that directly.

## How to Run

```bash
# structural / size contract (vendor-free):
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_001 -- --ignored --nocapture

# + vendored libcint byte-identity ground truth:
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test spike_axis_fold_001 -- --ignored --nocapture
```

## What to Expect

A forensic table, one row per (tier × non-square shape), reporting recovered
`comp_stride`, `len`, and populated-component count, then `SPIKE 001 : PASS`.

## Investigation Trail

1. **Pass 1 (vendor-free).** All 4 tiers × 2 shapes: `comp_stride == ni*nj` exactly; buffer
   splits into `rank` clean slices; `len == rank*ni*nj` everywhere. No truncation.
2. **Surprise.** Every component is populated — **81/81 for `rrrr`**. On this fixture there
   are *no* legitimately-zero components, so the per-component support gate is maximally
   strict here (and conversely, this fixture cannot exercise a "legitimately-zero component
   correctly skipped" path — noted for the build / spike 003).
3. **Used two shape-transpose pairs** — `(0,2)` O-1s×O-2p = 1×3 and `(2,3)` O-2p×H1-1s = 3×1
   — to show `comp_stride` *tracks* `ni*nj` as `(ni,nj)` swap, not a coincidence of one shape.
4. **Pass 2 (vendor linked).** Vendored libcint 6.1.3 built and linked cleanly in this
   environment. Element-wise byte-identity (`atol=1e-12`) holds at **all 4 tiers × 2 shapes**.
   This is the assertion that actually *pins component-outermost*: a component-interleaved
   layout `out[(j*ni+i)*rank + comp]` has the **same length** and would pass steps 1-3 but
   fail here. It did not fail → layout is genuinely component-leading.

## Results

**VALIDATED.** The per-component axis-fold is component-leading with `comp_stride == ni*nj`
at every rank tier 3/9/27/81 on the cart path, byte-identical to vendored libcint.

- Size/count contract (no truncation, no over-alloc): ✓ all tiers.
- `comp_stride == ni*nj`, clean `rank`-way partition: ✓ both non-square shapes.
- Component-outermost (vendor-pinned, not just length): ✓ byte-identity at `atol=1e-12`.
- No stuck-at-zero: ✓ (in fact fully populated, 81/81 at rank 81).

**Carry-forward:** orientation (i-fastest vs j-fastest *within* a component) is NOT
observable here because every non-square STO-3G block has a unit axis (`ni==1` or `nj==1`).
That requires a block with `ni>1` AND `nj>1` (a `d` shell) → spike **003**.
