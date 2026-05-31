# Spike Wrap-Up Summary

**Date:** 2026-05-31
**Spikes processed:** 6 (across two sessions: 001–003 core, 004–006 frontier)
**Feature areas:** Device block layout; Cart↔sph fold invariance; Spinor layout (divergence)
**Skill output:** `./.claude/skills/spike-findings-cintx/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | axis-fold-stride-probe | standard | ✓ VALIDATED | Device block layout |
| 002 | cart-vs-sph-fold-invariance | standard | ✓ VALIDATED | Cart↔sph fold invariance |
| 003 | hand-checked-vendor-stride | standard | ✓ VALIDATED | Device block layout |
| 004 | multi-index-block-ordering | integration | ✓ VALIDATED | Device block layout |
| 005 | nctr-axisfold-composition | frontier | ✓ VALIDATED | Device block layout |
| 006 | spinor-layout-divergence | frontier | ✓ VALIDATED | Spinor layout (divergence) |

## Frontier session (004–006) findings

- **004:** the component-leading formula generalizes to 3-/4-index families (`int2e_ip1`,
  `int3c2e_ip2`) — inner block i (bra1) fastest, vendor-identical, every axis permutation
  rejected. Reusable `reindex(extents, perm)` N-axis order-pinning primitive.
- **005:** nctr>1 composes with the fold at every rank tier (contraction-major
  `i_global=ci*di+ic`), vendor-identical; closes the nctr-transpose bug class across tiers.
- **006:** spinor is the one genuine divergence — interleaved-complex (`rank*ni_sp*nj_sp*2`,
  re/im fastest, `ni_sp=4l+2`), but component-leading + ket-major hold around the interleave.

## Key Findings

- The cintx device output layout is **`out[comp*(ni*nj) + (j*ni + i)]`** — component-leading
  (comp slowest, stride `ni*nj`), per-component block column-major / bra-fastest — confirmed
  byte-identical to vendored libcint 6.1.3 across rank tiers **3/9/27/81** on both transform
  paths.
- Confirmed two independent ways: a **hand-derived** invariant `<g_R|r_c|g_R> = R_c·S`
  (exact, rel=0, no libcint — pins component identity x/y/z + layout + origin) and **vendor
  byte-identity** at `atol=1e-12` (pins component-outermost, which length alone cannot).
- The **spherical path is mechanically `per-component c2s(cart)`** (worst Δ = 0.00e0 at
  every tier) — it is not an independent layout that can drift.
- Method gotcha promoted to a build rule: **STO-3G s×p non-square blocks are
  orientation-blind** (unit axis), so every new family's layout/orientation test must use a
  `d` shell (`p×d` = 3×6 cart / 3×5 sph). This contradicts the existing oracle default
  (`non_square_shell_pair()` returns s×p) and is a real gap in the current parity template.
- Vendored libcint **builds and links cleanly in this environment** (`CINTX_ORACLE_BUILD_VENDOR=1`),
  so the full ground-truth path is reproducible.

## Carry-forward into phases 26–31 (family registration)

- Reuse the `int1e_r/rr/rrr/rrrr` ladder as the canonical rank-tier layout-regression family.
- Add a `ni>1 ∧ nj>1` (d-shell) orientation fixture to the new-family parity template.
- The moment fixtures fully populate every component → add a parity-zero fixture if a
  legitimately-zero-component-skipped path needs coverage.
