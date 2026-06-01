---
plan: 30-01c
status: diagnosed-not-fixed
issue: int1e_cg_sa10sa01 / int1e_giao_sa10sa01 rank-9 spinor NOT byte-identical to vendor
created: 2026-06-01
---

# 30-01c Debug: rank-9 sa01 byte-identity blocker

## Symptom (ground truth, re-measured this session)

`CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity giao_sigma_1e_cg_sa10sa01 -- --include-ignored --nocapture`

Component 0 (σ-group 0), flat interleaved-complex indices 0–9:
- idx 0 (i0,re): cintx −1.68e-3, vendor −1.36e-2 (scaled-nonzero diff)
- idx 1 (i0,im): cintx **0.0**, vendor 5.09e-3
- idx 2 (i1,re): cintx **0.0**, vendor 6.10e-3
- idx 3 (i1,im): cintx 2.92e-3, vendor 4.93e-4
- idx 4 (i2,re): cintx −5.62e-3, vendor −4.55e-2
- idx 5 (i2,im): cintx **0.0**, vendor 1.77e-2
- idx 6 (i3,re): cintx **0.0**, vendor 1.71e-2
- idx 7 (i3,im): cintx 1.55e-2, vendor 8.87e-3
- idx 8 (j1,i0,re): cintx **0.0**, vendor 3.85e-3

Block dims: ni_sp=4 (p, kappa=+1 LT, nctr=2 → di=2), nj_sp=6 (d, kappa=−1 GT → dj=6).
Mix of exact-0 at both re and im slots + scaled-nonzero elsewhere.

## CORRECTION of the 30-01c-SUMMARY misdiagnosis

30-01c-SUMMARY blamed "the 36-component gout → 9×4 gc-block layout that
cart_to_spinor_si_2d consumes." **That is wrong.** Proven byte-faithful to vendor
this session (all with file:line evidence):

| Layer | cintx | vendor authority | verdict |
|-------|-------|------------------|---------|
| gout 36-entry body + 6 hard-zeros + 5 summed slots | sigma_p.rs:2309-2348 | intor3.c:1033-1068 | IDENTICAL |
| s[0..8] products | sigma_p.rs:2299-2307 | intor3.c:1021-1030 | IDENTICAL |
| gc-block order (x=0,y=1,z=2,1=3) | sigma_p.rs:2813-2816 | cart2sph.c:4971-4974 | IDENTICAL |
| transform variant: sa01 = REAL si | si_2d @ sigma_p.rs:2819 | c2s_si_1e (a_ket) @ intor3.c:1136 | CORRECT MATCH |
| int1e_type=1 (single rinv, charge +1) | sigma_p.rs:2432-2450 | g1e.c:226-228 | CORRECT |
| bra σ-mix signs | apply_bra_si_block_one c2spinor.rs:1093-1096 | a_bra_cart2spinor_si cart2sph.c:3958-3961 | IDENTICAL |
| ket transform (plain a_ket) | apply_ket_block c2spinor.rs:1205-1206 | a_ket_cart2spinor cart2sph.c:4461-4462 | IDENTICAL |
| si_2d vs si_2di = Stage-3 (re,im) vs (−im,re) | c2spinor.rs:725 / :843 | a_ket vs a_iket cart2sph.c:4473-4478 | CORRECT |
| nabla_i recurrence | sa01_nabla_i sigma_p.rs:2229-2234 | CINTnabla1i_2e g2e.c | IDENTICAL |
| nabla_j recurrence | sigma_p_nabla_j | CINTnabla1j_2e | IDENTICAL |
| x1i recurrence f[i]=g[i+1]+origin·g[i] | sigma_p_x1i / sa01_x1i_of_g1 | CINTx1i_2e g2e.c:4779 | IDENTICAL (algebraically) |

KEY FACT that overturns the SUMMARY: BOTH passing controls (cg_sa10sp, cg_sa10nucsp)
use the IMAGINARY transform `c2s_si_1ei` (intor3.c:1218 type0, :1311 type2). sa01 is the
ONLY family using the REAL `c2s_si_1e` (intor3.c:1136). So `cart_to_spinor_si_2d` (the
real variant) has ZERO passing-test coverage — but reading it against vendor proves it
correct (plain a_ket ket + plain (re,im) Stage-3 = exactly c2s_si_1e).

## Where the bug must be

Every layer above is byte-faithful by static read, yet output is wrong. The bug is in
the **only sa01-exclusive arithmetic with no passing-test coverage**: the on-the-fly
g-tensor assembly that feeds sa01_gout — `sa01_g1_bothside` (sigma_p.rs:2241),
`sa01_x1i_of_g1` (sigma_p.rs:2259), and their interaction with the HRR-built `g` array
headroom (sigma_p_hrr_axis sigma_p.rs, nmax=li+lj+2, lj_ext=lj+1).

Static line-by-line analysis of these helpers ALSO appears correct (nabla_i with iexp+1
stepping, x1i reading g[idx0+1], HRR headroom covering the +1 bra / +1 ket extensions).
The exact-zero (not just wrong) output positions mean a whole CONTRIBUTION is structurally
absent — a g-tensor term that should be nonzero is computed as 0, OR an HRR-uninitialized
slot (left at the 0-init from sigma_p.rs:2405-2409) is being read where vendor's
fully-materialized g1 array has a real value.

Vendor materializes g1 = D_J(g0)+D_I(g0) over bra range **i_l+1** as a STORED ARRAY
(intor3.c:1010-1012, note the `i_l+1`), THEN g3 = x1i(g1). cintx RECOMPUTES g1 per-element
inside the gout loop. Algebraically equal, but the suspect is an index/headroom edge where
cintx's recompute reads a g-array slot the HRR/VRR never populated (→ reads 0 → structural
zero in the output).

## DECISIVE NEXT EXPERIMENT (do this first next session)

Isolate g-tensor-vs-transform with the vendor CART path (bypasses c2s entirely):

1. Add `vendor_int1e_cg_sa10sa01_cart` FFI shim (vendor fn exists, intor3.c:1111) +
   append `int1e_cg_sa10sa01_cart` to the build.rs allowlist_function regex.
2. In a scratch test, compare cintx's pre-transform `gc` cart buffer (the
   `run_sa01_rys_on_backend` output, reshaped to vendor's 36-comp cart layout) against
   `vendor_int1e_cg_sa10sa01_cart` on a MINIMAL s×s single-primitive case (li=lj=0 →
   no exponent bookkeeping, 9 groups still exercised; then p×s to surface bra stepping).
3. If cart buffers DIFFER → bug is 100% in the g-tensor assembly (sa01_g1_bothside /
   sa01_x1i_of_g1 / HRR headroom). Hand-compute the 9 s[] for s×s (5 lines) to find which
   g-term is wrong. If cart buffers MATCH → bug is in the transform feed/call (re-open
   si_2d, but it reads correct by static analysis, so this is unlikely).

This is the "numerical reverse-engineering against vendor" RESEARCH Open Q1 deferred and
the spike-findings dual hand-derived+vendor method prescribes. The cart-path discriminator
is the fastest cut: one shim + one tiny test splits the problem in half.

## What NOT to do
- Do NOT touch the gout body, gc-block order, transform variant, or int1e_type — all proven correct.
- Do NOT weaken/delete the 3 #[ignore]d gates or the test_no_silent_skip sa01 assertions.
- Do NOT flip oracle_covered until byte-identity is real.
