---
id: wr03-3c1e-grad-nctr-gt1
created: {
  timestamp: 2026-05-30T04:02:28.028Z
}
source: 23-REVIEW.md (WR-03)
severity: warning
resolves_phase: null
resolved: {
  by: quick 260530-iiq
}
---

# int3c1e gradient launchers only contract the first nctr column — RESOLVED

## Original finding

`launch_center_3c1e_ip1` (center_3c1e.rs:1047-1049) and `launch_center_3c1e_iprinv`
(~1159-1161) weighted by `shell.coefficients[ip * n_ctr_*]` — contraction column 0 only —
unlike the scalar `center_3c1e_kernel` which dispatches one host launch per
`(ci,cj,ck)` column triple via `ci_sel/cj_sel/ck_sel`. Correct for nctr==1, but produced
incomplete gradients for genuinely general-contracted (nctr>1) shells.

## Resolution (quick 260530-iiq)

Fixed in `crates/cintx-cubecl/src/kernels/center_3c1e.rs`: both gradient launchers now
wrap the per-primitive accumulation in the same `(ck,cj,ci)` contraction-column loop as
the scalar path (each column using that column's coefficients), produce one
component-leading block per `(ci,cj,ck)` triple, and scatter it into a single dense
COMPONENT-LEADING interleaved output via `scatter_3c1e_grad_block` (contraction-MAJOR per
axis: `i_global = ci*nblk_i + i_idx`, matching libcint `c2s_{cart,sph}_3c2e1`).

**The scalar path shared — and ALSO had — the same nctr>1 limitation.** The original
review framed this as gradient-only, but the scalar `launch_center_3c1e_typed`
accumulated every `(ci,cj,ck)` triple into ONE block (`*dst += src`), dropping/merging
contraction columns exactly like the gradient launchers. The scalar path was fixed in the
same change (single interleaved nctr-blocked output).

**Deeper root cause (empirical, vs vendor):** the true defect was not just output block
placement. The libcint env coefficient block is COLUMN-MAJOR (`env[ci*nprim+ip]`, per
`CINTprim_to_ctr_0` in g1e.c), but cintx Shells are ROW-MAJOR
(`coefficients[ip*nctr+ci]`). The raw `eval_raw` env→Shell parse copied coefficients
verbatim, transposing nctr>1 coefficients for EVERY family (a latent, previously
untested bug — no raw nctr>1 parity test existed). Fixed by transposing column-major →
row-major at the raw boundary (`crates/cintx-compat/src/raw.rs`). nctr==1 is unaffected
(the two layouts coincide).

## Verification

`crates/cintx-oracle/tests/int3c1e_genctr_parity.rs` (new): scalar + ip1 + iprinv, cart +
sph, on a non-square general-contraction fixture (i=p nctr=2, j=d, k=s) — all
byte-identical to vendored libcint 6.1.3 at atol=1e-12 under the
`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` double gate. nctr==1 byte-identity
preserved (`int3c1e_ip_parity` 5/5, `center_3c2e_parity` green, `cintx-cubecl --lib`
280 green). The device `#[cube]` kernels were NOT touched — the fix is host-side block
placement plus the raw coeff transpose.
