---
id: wr03-3c1e-grad-nctr-gt1
created: {
  timestamp: 2026-05-30T04:02:28.028Z
}
source: 23-REVIEW.md (WR-03)
severity: warning
resolves_phase: null
---

# int3c1e gradient launchers only contract the first nctr column

`launch_center_3c1e_ip1` (center_3c1e.rs:1047-1049) and `launch_center_3c1e_iprinv`
(~1159-1161) weight by `shell.coefficients[ip * n_ctr_*]` — contraction column 0 only —
unlike the scalar `center_3c1e_kernel` which dispatches one host launch per
`(ci,cj,ck)` column triple via `ci_sel/cj_sel/ck_sel`. Correct for nctr==1 (all DRV1
vendor-parity fixtures, and this `fix/general-contraction-nctr-1e` branch's scope), but
produces incomplete gradients for genuinely general-contracted (nctr>1) shells.

Phase-23 DRV1-03 parity is verified byte-identical at nctr==1, so this is a latent
follow-up, not a phase-23 blocker. Resolve when general-contraction (nctr>1) gradient
support is in scope; mirror the scalar per-column host dispatch.
