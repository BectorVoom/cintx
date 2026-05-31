---
spike: 006
name: spinor-layout-divergence
type: frontier
validates: "Given a rank-3 spinor family (int1e_ipovlp_spinor) on a non-square s×p block, when evaluated, then output is interleaved-complex (re,im fastest) with spinor block dims (ni_sp=4l+2), component axis still outermost — vendor byte-identity, with de-interleaved and i/j-transposed misreads both rejected"
verdict: VALIDATED
related: [001, 003, 002]
tags: [layout, spinor, complex, divergence, vendor]
---

# Spike 006: spinor-layout-divergence

## What This Validates

The one path where the layout contract genuinely **diverges** from the real
component-leading form proven in spikes 001–005. The spinor output is interleaved-complex:

```
out[comp * (ni_sp*nj_sp) * 2 + (j*ni_sp + i)*2 + {0:re, 1:im}]
    comp        : slowest (rank 3 gradient)           ← component-leading STILL holds
    (j*ni_sp+i) : ket-major, i (bra) fastest          ← same orientation as real families
    {re,im}     : FASTEST axis, per-element pair, ×2   ← THE divergence
    ni_sp = CINTcgto_spinor(shell) = 4l+2 (kappa==0)   ← spinor dims, NOT ncart/nsph
```

This spike *characterizes* the divergence (backing the skill's "spinor differs" note with a
concrete probe), rather than only asserting equality.

## Research

- Documented spinor layout (`one_electron_grad_spinor_parity.rs:9-11`):
  `out[comp*ni_sp*nj_sp*2 + (j*ni_sp+i)*2 + {0:re,1:im}]`, `ni_sp = CINTcgto_spinor`.
- `spinor_len_kappa0(l) = 4l+2` (kappa==0 → both GT j=l+1/2 and LT j=l−1/2 blocks).
- Per the spike-findings map, the spinor c2s path (`c2spinor.rs`) emits interleaved-complex,
  not component-leading real — this is the documented divergence point.
- Vendor `vendor_int1e_ipovlp_spinor(&mut[f64], &[i32;2], atm, natm, bas, nbas, env)`.

## How to Run

```bash
cargo test -p cintx-oracle --features cpu --test spike_axis_fold_006 -- --ignored --nocapture
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test spike_axis_fold_006 -- --ignored --nocapture
```

## Investigation Trail

1. Minimal 1-atom fixture, s + p shells, `KAPPA_OF=0` → non-square spinor block
   `ni_sp=2 × nj_sp=6`, `complex_block = 2*6*2 = 24`, `len = 3*24 = 72`.
2. Structural: `len == rank*ni_sp*nj_sp*2` (complex-interleaved); `comp_stride == 24`
   (component still outermost); imaginary lane non-trivially non-zero (`im_nnz=6`) → the
   buffer is *genuinely* complex, not real-padded.
3. Two misread negative controls: `deinterleave` (block-separated `[re|im]`) and
   `transpose_ij` (bra-major) both change the buffer (vendor-free).
4. Vendor: `mm(vendor,cintx)==0`; both misreads diverge (de-interleaved 23, transposed 18)
   → the per-element interleave AND ket-major orientation are pinned, not coincidence.

## Results

**VALIDATED.** The spinor layout is `out[comp*(ni_sp*nj_sp)*2 + (j*ni_sp+i)*2 + {re,im}]`,
byte-identical to vendor:

| Check | Result |
|-------|--------|
| Complex-interleaved (len, comp_stride) | ✓ len=72, comp_stride=24 |
| Genuinely complex (im lane nonzero) | ✓ im_nnz=6 |
| Per-element interleave pinned | ✓ de-interleaved misread mm(vendor)=23 |
| Ket-major i-fastest pinned | ✓ transposed misread mm(vendor)=18 |
| Vendor byte-identity | ✓ mm=0 |

**Divergence from the real-family contract (spikes 001–005):**
- Block dims are **spinor lengths** (`ni_sp = 4l+2` @ kappa=0), not `ncart`/`nsph`.
- Each matrix element is a **complex (re,im) pair** — the fastest axis, doubling the length.
- *What's preserved:* the component axis is still outermost, and within the complex block the
  ket-major / bra-fastest orientation matches the real families.

**Carry-forward:** spinor parity/layout tests must size buffers `rank*ni_sp*nj_sp*2`, treat
re/im as the fastest interleaved axis, and use a non-square spinor pair (s×p, 2×6) so the
ket-major orientation is observable. The component-leading + i-fastest invariants from
001/003 still apply *around* the complex interleave.
