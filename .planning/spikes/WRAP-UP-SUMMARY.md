# Spike Wrap-Up Summary

**Date:** 2026-05-31
**Spikes processed:** 3
**Feature areas:** Device block layout; Cart↔sph fold invariance
**Skill output:** `./.claude/skills/spike-findings-cintx/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | axis-fold-stride-probe | standard | ✓ VALIDATED | Device block layout |
| 002 | cart-vs-sph-fold-invariance | standard | ✓ VALIDATED | Cart↔sph fold invariance |
| 003 | hand-checked-vendor-stride | standard | ✓ VALIDATED | Device block layout |

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
