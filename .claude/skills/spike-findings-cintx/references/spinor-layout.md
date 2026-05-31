# Spinor Layout (the one divergence)

Spinor families are the **single exception** to the real component-leading contract. Their
output is interleaved-complex with spinor block dims.

## Requirements

- Output is **interleaved-complex**: each matrix element is a `(re, im)` pair, re/im the
  FASTEST axis — the buffer is `×2` the real length.
- Block dims are **spinor lengths** `ni_sp = CINTcgto_spinor(shell)`, NOT `ncart`/`nsph`.
  For `kappa==0`, `ni_sp = 4l+2` (both GT j=l+1/2 and LT j=l−1/2 blocks).
- **Component-leading and ket-major i-fastest still hold** *around* the complex interleave.

## How to Build It

```
out[comp * (ni_sp*nj_sp) * 2 + (j*ni_sp + i)*2 + {0:re, 1:im}]
    comp        : slowest (e.g. rank 3 for a gradient)
    (j*ni_sp+i) : ket-major, i (bra) fastest
    {re,im}     : fastest axis, per-element pair
```

```rust
// size: rank * ni_sp * nj_sp * 2 ; set KAPPA_OF on spinor shells.
let ni = spinor_len_kappa0(li); // 4l+2 at kappa=0
let mut out = vec![0.0_f64; RANK * ni * nj * 2];
unsafe { eval_raw(RawApiId::INT1E_IPOVLP_SPINOR, Some(&mut out), None,
                  &[si as i32, sj as i32], &atm, &bas, &env, None, None).unwrap(); }
```

Verify against `vendor_int1e_ipovlp_spinor` (sig identical to the real vendor fns). Use a
**non-square** spinor pair (s×p → 2×6) so ket-major orientation is observable. Two misread
negative controls both diverge from vendor: block-separated complex (`[re|im]`) and i/j
transpose — confirming per-element interleave AND orientation are pinned.

## What to Avoid

- **Don't reuse the real-family sizing** (`rank*ni*nj`) — you'll under-allocate by 2× and
  mis-split re/im. Spinor is `rank*ni_sp*nj_sp*2`.
- **Don't use `ncart`/`nsph` for spinor block dims** — use `CINTcgto_spinor` / `4l+2`.
- **Don't test on a square spinor block** — orientation hides (same trap as D-07).

## Constraints

- Verified for `int1e_ipovlp_spinor` (rank-3 gradient), kappa=0, byte-identical to libcint.
- Other spinor families (kappa≠0 → `ni_sp = 2l+2` or `2l`) not yet spiked; the `4l+2`
  sizing here is kappa=0 specific.

## Origin

Synthesized from spikes: 006
Source files available in: sources/006-spinor-layout-divergence/
