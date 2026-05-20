# 19-05 partial gradient scaffolding (salvaged)

**Date:** 2026-05-20
**Status:** Salvaged from an interrupted executor run; NOT applied to `main`.

## Why this exists

The first execute-phase attempt at Plan 19-05 (ECP gradient, `int1e_ecp_ipnuc_{cart,sph}`)
was interrupted by a usage limit. The executor agent (worktree
`worktree-agent-a65dfbc93522f0339`) made **no commits** and left an uncommitted,
non-compiling edit to `crates/cintx-cubecl/src/kernels/ecp.rs`. That edit is
captured verbatim in `19-05-partial-gradient-scaffolding.patch`.

## Why it was not finished or merged

Plan 19-05's must-haves require **byte-identity at atol=1e-12** vs PySCF
`nr_ecp_deriv.c`. The gradient reuses `compute_type1_pair` / `compute_type2_pair`,
which Plan 04 (`19-04-SUMMARY.md`) shipped as a **direct-quadrature approximation**,
not PySCF's exact recurrences. Byte-identity is therefore blocked on a missing
PySCF K-Taylor port (the 400×24 `K_TAB` table + `ECPrad_part` / `ECPrad_block`),
which affects BOTH the scalar (Plan 04) and gradient (Plan 05) arms.

Decision (2026-05-20): **replan** — add an explicit K-Taylor port plan that makes
the scalar arm byte-identical, then re-sequence the gradient (19-05) on top of it.

## What is reusable in the patch

- `Y_ADDR` / `Z_ADDR` — Cartesian (l → l+1) address remap tables, verbatim from
  `nr_ecp_deriv.c::_y_addr` / `_z_addr` (120 entries = cumulative ncart for l=0..7).
  **Bug to fix on reuse:** declared `[usize; 135]` but hold 120 elements — must be `[usize; 120]`.
- `CART_POW_Y` / `CART_POW_Z` — ly / lz power tables, verbatim from
  `nr_ecp_deriv.c::_cart_pow_y` / `_cart_pow_z` (120 entries, correctly sized).
- `l_down_grad` / `l_up_grad` — structure for the `_l_down` / `_l_up` combiners
  (the `-2·alpha_i·(r-R_C)` factor and the angular-momentum-lowering term).
  **Unverified:** the p-shell normalization constants and the synthetic-shell
  evaluation approach were never tested against the oracle.
- `synthetic_single_prim_shell` — helper that builds a single-primitive shell at
  shifted angular momentum (li±1) to drive `compute_type{1,2}_pair`.

## What is NOT reusable

The inner evaluation calls `compute_type1_pair` / `compute_type2_pair` (the
approximate primitives). Once the K-Taylor port replaces those with PySCF's exact
`ECPrad_part` recurrences, the gradient inner loop must be rebuilt against the new
primitive signatures. Treat the patch as a reference for the table data and the
`_l_down`/`_l_up` shape, not as a drop-in.
