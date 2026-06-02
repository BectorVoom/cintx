---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
plan: 02
subsystem: testing
tags: [spinor, sigma-p, sigma-r, si_2d, si_2di, sf_2d, rys-nuclear, cube-kernel, vendor-parity, libcint, relativistic]

# Dependency graph
requires:
  - phase: 29 (plan 01)
    provides: cart_to_spinor_si_2di transform, 7 int1e_*_spinor manifest rows (oracle_covered=false), 7 vendor_int1e_*_spinor FFI shims, rel_1e_sigma_parity.rs RED scaffold
  - phase: 28-gap-b2-c2s-si-sigma-p
    provides: cart_to_spinor_si_2d + cart_to_spinor_sf_2d transforms, σ·p assembler (sigma_p.rs), build_kappa_spinor_fixture, int1e_sp_spinor row
provides:
  - 7 1e Group-4 σ Spinor launcher arms (spsp/spnucsp/sprinvsp/srsr/srnucsr/sr/sigma) byte-identical to libcint 6.1.3 at atol=1e-12
  - sigma_1e.rs overlap-engine #[cube] σ kernel (sigma/sr/srsr/spsp) + unified launch_int1e_sigma_family_spinor_pair
  - sigma_1e_nuc.rs Rys nuclear-engine #[cube] σ kernel (spnucsp/srnucsr/sprinvsp), G2E_D/G2E_R selected by comptime use_r
  - int1e_sigma component_rank EMPIRICALLY resolved to 3 (3 stacked Pauli σ-matrices) — 29-01 rank-1 prior DISPROVEN
  - 8 manifest rows flipped oracle_covered=true (spinor-only): int1e_{sp,spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor
  - rel_1e_sigma_parity.rs GREEN (10/10): rank measurement + 7 byte-identity gates + no-silent-skip + kappa-sizing
affects: [29-group-4-wave-2, 29-group-4-wave-3, 30-giao-sigma]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Unified per-family σ launcher: comptime family selector drives gout assembly; per-family transform select (sf_2d/si_2d/si_2di) + per-arm fail-closed staging guard"
    - "Rys nuclear σ engine: comptime use_r switches G2E_D (nabla recurrence) vs G2E_R (index shift) on the SAME kernel body; origins+charges precomputed by caller (cubecl has no raw atm/bas/env constants)"
    - "component_rank empirical resolution: oversized NaN-filled vendor buffer pinpoints exact written extent (rposition of finite) without heap corruption"

key-files:
  created:
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs
    - crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs
  modified:
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/tests/rel_1e_sigma_parity.rs
    - crates/cintx-oracle/tests/si_transform_parity.rs

key-decisions:
  - "int1e_sigma is component_rank=3 (NOT 1): CINT1e_spinor_drv loops c2s_si_1ei ng[7]=3 times (cint1e.c:269) → 3 stacked Pauli σ-matrices; measured vendor output 144 = 3*ni_sp*nj_sp*2 on the kappa fixture (overrides the 29-01/RESEARCH rank-1 prior)"
  - "spsp routes through cart_to_spinor_sf_2d which does NOT own the KET→BRA transpose (unlike si_2d/si_2di) → transpose the scalar gc_1 block in fold_group before the SF fold"
  - "Overlap σ families (sigma/sr/srsr/spsp) use a single #[cube] kernel with +2 i / +1 j headroom; R_I/R_J = index shift (+1/+dj), D_I/D_J = nabla recurrence; per-cart gout transcribed verbatim from intor3.c"
  - "Nuclear σ families (spnucsp/srnucsr/sprinvsp) share one Rys kernel: spnucsp/srnucsr = atom-sum charge −Z, sprinvsp = single PTR_RINV_ORIG center +1; nroots=(li+lj+2)/2+1 clamped to 5"
  - "oracle_covered flipped spinor-only (sph=cart=false) per family ONLY after byte-identity green (SC#5); lock hand-edited (build.rs regenerates api_manifest.rs/.csv)"

patterns-established:
  - "Pattern 1: comptime family selector in a shared #[cube] σ kernel emits the per-family rank*N_GC component-leading gc blocks; the host launcher folds each σ-group through the family transform"
  - "Pattern 2: empirical component_rank gate — measure vendor output extent against ni_sp*nj_sp*2 with a NaN-headroom buffer BEFORE locking the rank, never assume"

requirements-completed: [REL-01, REL-02]

# Metrics
duration: 100min
completed: 2026-05-31
---

# Phase 29 Plan 02: Wave-1 1e Relativistic σ Families Summary

**Wired all 7 1e Group-4 σ Spinor launcher arms (spsp/spnucsp/sprinvsp/srsr/srnucsr/sr/sigma) onto their correct per-family transforms via two new #[cube] σ engines (overlap + Rys nuclear), empirically resolved int1e_sigma to component_rank=3 (disproving the 29-01 rank-1 prior), and flipped 8 manifest rows oracle_covered=true spinor-only — all byte-identical to vendored libcint 6.1.3 at atol=1e-12 on the non-square kappa fixture.**

## Performance

- **Duration:** ~100 min
- **Started:** 2026-05-31T13:00Z (approx, post-29-01)
- **Completed:** 2026-05-31T14:36Z
- **Tasks:** 3
- **Files modified:** 8 (2 created, 6 modified)

## Accomplishments

- **Task 1 — int1e_sigma rank empirically resolved (REL-02):** `test_sigma_rank_measured` calls `vendor_int1e_sigma_spinor` on the kappa fixture into an oversized NaN-headroom buffer and measures the written extent via `rposition(is_finite)`. Measured length = **144 = 3 × ni_sp·nj_sp·2 (=3×48)** → `component_rank = "3"`, NOT the 29-01-assumed rank 1. Root cause: `CINT1e_spinor_drv` loops `f_c2s` over `ncomp_tensor = ng[7] = 3` (cint1e.c:269), producing the three stacked Pauli σ-matrices σ_x/σ_y/σ_z (each `gout[n*12+..]` group of 4 is one σ-component's G-tensor blocks). Lock sigma row `component_rank` 1→3; api_manifest regenerated.
- **Task 2 — 7 launcher arms wired byte-identical (REL-01/02):**
  - `sigma_1e.rs`: overlap-engine `#[cube]` kernel (`sigma_ov_kernel`) covering `sigma` (g0 only, rank-3 σ-groups), `sr` (R_I bra, −s0/−s1/−s2/0), `srsr` (R_J·R_I composed, s5−s7/s6−s2/s1−s3/s0+s4+s8), `spsp` (D_J·D_I composed, scalar s0+s4+s8 → gc_1). Unified `launch_int1e_sigma_family_spinor_pair` selects the transform per family (`sf_2d`/`si_2d`/`si_2di`), applies its own fail-closed staging guard (`required = ni_sp·nj_sp·2·rank`), and loops the σ-groups for sigma.
  - `sigma_1e_nuc.rs`: Rys nuclear-engine `#[cube]` kernel (`sigma_nuc_kernel`) covering `spnucsp`/`srnucsr` (atom-sum −Z) and `sprinvsp` (single rinv +1); a comptime `use_r` flag switches `G2E_R` (index shift) vs `G2E_D` (nabla) on the same rank-4 σ·G gout. Origins/charges precomputed by the caller (the cubecl crate has no raw atm/bas/env slot constants).
  - **spsp transpose fix:** `cart_to_spinor_sf_2d` reads BRA-major `cart[i*ncj+j]` and does NOT own the KET→BRA transpose (unlike si_2d/si_2di), so `fold_group` transposes the KET-major scalar block before the SF fold — caught by the non-square fixture (the only family that failed before the fix).
- **Task 3 — 8 rows flipped + NO-SILENT-SKIP green:** `int1e_{sp,spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor` flipped `oracle_covered` false→true spinor-only (sph=cart=false). `rel_1e_sigma_parity::test_no_silent_skip` now re-runs BOTH the vendor AND cintx arms with a byte-identity re-check and asserts each row reads `oracle_covered=true` + spinor-only (SC#4/SC#5). `manifest-audit` green (`uncovered_count=0`); `cargo build --workspace --features cpu` clean (no OperatorId positional drift — rows were tail-appended in 29-01). `si_transform_parity::test_no_silent_skip` updated: the int1e_sp flip is 29-02's (Phase 28 deferred it).

## Task Commits

1. **Task 1: resolve int1e_sigma component_rank=3 empirically** — `6a299b6` (fix)
2. **Task 2: wire 7 1e Group-4 σ Spinor launchers (byte-identical vendor parity)** — `be8d2c0` (feat)
3. **Task 3: flip int1e_sp + 7 σ families oracle_covered=true (spinor-only)** — `ab5d4ac` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/kernels/sigma_1e.rs` (NEW) — overlap-engine σ kernel + unified launcher + per-family transform/fold
- `crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs` (NEW) — Rys nuclear-engine σ kernel (spnucsp/srnucsr/sprinvsp)
- `crates/cintx-cubecl/src/kernels/mod.rs` — register sigma_1e + sigma_1e_nuc modules
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — sigma component_rank 1→3; 8 rows oracle_covered false→true
- `crates/cintx-ops/src/generated/api_manifest.rs` / `.csv` — regenerated from the lock by build.rs
- `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` — real cintx collectors, rank measurement, gates LIVE, post-flip no-silent-skip
- `crates/cintx-oracle/tests/si_transform_parity.rs` — int1e_sp no-silent-skip expects oracle_covered=true (29-02 flip)

## Decisions Made

- **int1e_sigma is component_rank 3, empirically, not 1.** The 29-01 SUMMARY and 29-RESEARCH Open-Q1 resolved it as rank 1 ("σ-axis folds into the transform"). Task 1 measured the actual vendor output and found 3 stacked σ-matrices (the driver's `ncomp_tensor=3` loop). This is exactly the belt-and-suspenders the plan mandated; the strong prior was wrong and is overridden. The sigma launcher emits 3 σ-component groups and the test sizes vendor/cintx at 3×.
- **The σ·p/σ·r G-tensors are NOT expressible by the existing sigma_p.rs.** sigma_p only does single-side bra σ·p (rank-1, 4 blocks). srsr/spsp need both-side composed derivatives; the nuclear families need Rys. Two new engines were the right call (RESEARCH Open-Q2 anticipated a sibling for 2e; the 1e composed/nuclear gouts likewise needed dedicated kernels).
- **spsp's SF transform needs a launcher-side transpose.** si_2d/si_2di own the KET→BRA transpose; sf_2d does not. Rather than change the shared sf_2d (used elsewhere), the launcher transposes the scalar block for the one SF family.

## Deviations from Plan

None requiring a Rule-4 stop. Two in-scope refinements applied under Rules 1-3:

### Auto-fixed / Auto-resolved Issues

**1. [Rule 1 - Empirical correction] int1e_sigma component_rank is 3, not the planned/assumed 1**
- **Found during:** Task 1 (the plan's explicit "measure before locking" step)
- **Issue:** The plan's strong prior and 29-01 lock said rank 1; the vendor driver writes 3 stacked σ-matrices.
- **Fix:** Locked `component_rank="3"`, structured the sigma launcher with a 3-group loop, sized the test collectors at 3×. The plan explicitly authorized this branch ("if == 3*di*dj*2, set component_rank 3 and structure the launcher with a 3-component loop").
- **Files:** compiled_manifest.lock.json, sigma_1e.rs, rel_1e_sigma_parity.rs — **Commit:** 6a299b6 / be8d2c0

**2. [Rule 3 - Blocking] spsp SF transform KET→BRA transpose**
- **Found during:** Task 2 (spsp was the only family failing parity before the fix)
- **Issue:** cart_to_spinor_sf_2d expects BRA-major input and does not transpose; the device kernel emits KET-major.
- **Fix:** Transpose the scalar block in `fold_group` for the SF case (mirrors the transpose si_2d owns internally).
- **Files:** sigma_1e.rs — **Commit:** be8d2c0

**3. [Rule 3 - Blocking] si_transform_parity int1e_sp coverage assertion**
- **Found during:** Task 3
- **Issue:** Phase-28's `si_transform_parity::test_no_silent_skip` asserted int1e_sp stays oracle_covered=false; 29-02 flips it.
- **Fix:** Updated the assertion to expect oracle_covered=true (the 29-02 flip).
- **Files:** si_transform_parity.rs — **Commit:** ab5d4ac

## Issues Encountered

- The vendor `free(): invalid pointer` SIGABRT on the first no-silent-skip run was the symptom of the sigma rank-3 reality: the test buffer sized `di*dj*2` overflowed when sigma wrote 3×. Resolving Task 1 (rank=3 sizing) fixed it. This validated the plan's mandate to measure the vendor output shape before wiring sigma.

## Known Stubs

None. All 7 families are fully wired and byte-identical; no placeholder data, no TODO stubs remain in the launcher path. The 29-01 `collect_cintx_rel_1e` `unimplemented!` stub is replaced with the real per-family launch.

## Threat Flags

None. The two registered threats (T-29-03 per-arm staging guard, T-29-04 skipped-fixture flip refusal) are both mitigated: every launcher applies its own `BufferTooSmall` guard before any write, and no-silent-skip re-runs both arms with byte-identity before honoring the flip. No new network/auth/file-access surface.

## Next Phase Readiness

- **Wave 1 GREEN gate satisfied** — REL-01/02 closed: all 7 families + int1e_sp byte-identical at atol=1e-12, oracle_covered spinor-only, manifest-audit green, NO-SILENT-SKIP green. Wave 2 (2e foundation) may proceed.
- The two new engines (overlap + Rys nuclear σ) are reusable templates for the Wave-3 2e σ families (which need the analogous 2e G-tensor layout per RESEARCH Open-Q2).
- No blockers.

## Self-Check: PASSED

All created files exist on disk (sigma_1e.rs, sigma_1e_nuc.rs, 29-02-SUMMARY.md); all 3 task commits (6a299b6, be8d2c0, ab5d4ac) present in git history. rel_1e_sigma_parity 10/10 green; si_transform_parity green; manifest-audit uncovered_count=0; workspace builds clean.

---
*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Completed: 2026-05-31*
