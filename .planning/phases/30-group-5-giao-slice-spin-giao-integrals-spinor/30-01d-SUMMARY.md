---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01d
subsystem: cubecl
tags: [cubecl, spinor, giao, spgnucsp, spgsa01, spg-rys, london, rys, gauge, rank-9, vendor-parity, libcint, partial]

# Dependency graph
requires:
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 00
    provides: "gauge x1i-with-origin fold helpers + combined gauge∧kappa spinor fixture + giao_sigma_1e_parity scaffold + vendor shims (incl. spgnucsp/spgsa01) + bindgen allowlist (3b68ff1/30-00)"
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 01a
    provides: "spgsp 8-G London chain (sigma_p_nabla_i_combine + sigma_p_x1i + sigma_p_x1i_of_j) + GIAO×σ dispatch table (giao_family_id/rank/transform + launch_int1e_giao_sigma_family_spinor_pair)"
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 01b
    provides: "Rys+gauge nuclear engine pattern (sigma_nuc_gauge_kernel) + nucsp launcher precedent"
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 01c
    provides: "rank-9 sa01 Rys+gauge engine (sa01_rys_kernel, sa01_g1_bothside, sa01_x1i_of_g1, sa01_nuc_vrr_axis, REAL c2s_si_1e) + rinv-center threading lesson"
provides:
  - "spgnucsp_rys_kernel — net-new rank-3 spg-Rys/London device kernel (NUCLEAR Rys base + spgsp 8-G London chain g1=D_J + 12-comp London mix) — byte-identical to vendored libcint at atol=1e-12"
  - "launch_int1e_spgnucsp_spinor_pair (rank 3, c2s_si_1ei imaginary, common_factor 0.5, own fail-closed ×3 staging + nroots guards)"
  - "spgsa01_rys_kernel — net-new rank-9 spg-Rys/London device kernel (RINV Rys base + spgsp 8-G London chain with BOTH-SIDE g1=D_J+D_I + 36-comp London mix, REAL c2s_si_1e) — BUILDS GREEN, all-9 non-zero, NOT yet byte-identical (~0.5% uniform residual in the D_I-in-g1 chain)"
  - "spgnucsp/spgsa01 dispatch arms in launch_int1e_giao_sigma_family_spinor_pair (giao_family_id 7/8, rank 3/9, transform SiI/Si REAL)"
  - "spgnucsp byte-identity gate + full-9-family test_no_silent_skip (8 covered + spgnucsp; spgsa01 honest false)"
affects: [phase-30 Wave 2 (30-02, 6 × 2e GIAO×σ families)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "spg-Rys/London engine = the 30-01a spgsp 8-G London chain (G2E_R0I origin=ri + rirj=ri−rj post-multiply; neither reads PTR_COMMON_ORIG) grafted onto a Rys root loop (nuclear int1e_type 2 for spgnucsp / rinv int1e_type 1 for spgsa01) — the London fold lives INSIDE the root loop"
    - "spgnucsp g1 = D_J(g0) only (12-comp rank-3 mix, SiI imaginary); spgsa01 g1 = D_J(g0) + D_I(g0) both-side (36-comp rank-9 mix, REAL Si) — the both-side D_I raises the bra reach to li+3 (one deeper than spgnucsp's li+2) so spgsa01 headroom = nmax = li+lj+4"
    - "the 27-product s[] mix is byte-identical between spgnucsp and spgsa01 (intor3.c:1911 == :2071); they differ ONLY in g1 (D_J vs D_J+D_I), the gout mix, and the Rys base class"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/sigma_p.rs
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv

decisions:
  - "Implemented spgnucsp/spgsa01 as TWO net-new SELF-CONTAINED kernels (spgnucsp_rys_kernel + spgsa01_rys_kernel) combining the 30-01a spgsp 8-G London chain with the 30-01c Rys base, reusing the existing #[cube] helpers (sigma_p_nabla_j/x1i/x1i_of_j/nabla_i_combine, sa01_g1_bothside/x1i_of_g1/nuc_vrr_axis, sigma_p_hrr_axis); the int1e_sp / cg / giao / spgsp / sa01 paths are untouched."
  - "spgsa01 headroom set to nmax = li+lj+4 (Rule 1): the both-side g1's D_I term raises the deepest g0 bra read to li+3, one level beyond spgnucsp's li+2; the original li+lj+3 was boundary-correct-by-luck on the fixture but is the wrong minimum."
  - "spgnucsp flipped oracle_covered=true (byte-identical, verified); spgsa01 LEFT oracle_covered=false — it builds and runs all-9-non-zero but is NOT yet byte-identical (~0.5% uniform residual). T-30-01d-06: never over-claim coverage on a non-byte-identical family. Its gate is #[ignore]d and test_no_silent_skip enforces the honest false state."

# Metrics
duration: 110min
completed: 2026-06-01
status: PARTIAL (spgnucsp byte-identical + covered; spgsa01 BLOCKED on a ~0.5% residual)
---

# Phase 30 Plan 01d: GIAO×σ Sub-wave 1d — spg-Rys/London engine (spgnucsp green, spgsa01 BLOCKED)

**The NEW spg-Rys/London engine class (the 30-01a spgsp 8-G London chain — G2E_R0I origin=ri + rirj=ri−rj — grafted onto a Rys root loop) is implemented for both families. `int1e_spgnucsp` (rank 3, NUCLEAR Rys + g1=D_J, 12-comp, SiI imaginary) is byte-identical to vendored libcint 6.1.3 at atol=1e-12 on a NON-SQUARE p×d combined gauge∧kappa block and is oracle_covered=true. `int1e_spgsa01` (rank 9, RINV Rys + BOTH-SIDE g1=D_J+D_I, 36-comp, REAL Si) builds, runs, and produces all-9 non-zero output but is NOT yet byte-identical — a small (~0.5%) UNIFORM residual concentrated in the D_I-in-g1 chain. spgsa01 stays oracle_covered=false (gate #[ignore]d). EIGHT of the nine 1e GIAO×σ families are now gated green; the 1e half of GIAO-03 is NOT fully closed.**

## Status: PARTIAL — spgnucsp gated green; spgsa01 blocked

## Performance
- **Duration:** ~110 min
- **Completed:** 2026-06-01
- **Tasks:** 2 (Task 1 kernels — done/green-build; Task 2 gates+flip — spgnucsp flipped, spgsa01 honest-deferred)
- **Files modified:** 6

## Accomplishments
- **`spgnucsp_rys_kernel`** (sigma_p.rs, net-new): the NUCLEAR Rys base (atom-sum, charge −Z, int1e_type 2, `sa01_nuc_vrr_axis`, comptime nroots 1..=5) + the spgsp 8-G London chain (g0 base; g1=D_J(g0); g2=R0I(g0,ri); g3=R0I(g1); g4..g7=D_I(g0..g3) via `sigma_p_nabla_i_combine`) folding the 27-product `s[0..26]` into the **12-component** London `c[]·s[]` mix (`c = ri − rj`), transcribed VERBATIM from intor3.c:1953-1964. Headroom nmax = li+lj+3 (D_I(R0I(D_J)) reaches bra li+2). `launch_int1e_spgnucsp_spinor_pair`: rank 3, REAL→imaginary `cart_to_spinor_si_2di` (SiI, Pitfall 2), common_factor 0.5, OWN fail-closed full-block ×3 staging guard + fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` (no clamp).
- **`spgsa01_rys_kernel`** (sigma_p.rs, net-new): the RINV Rys base (single center charge +1, int1e_type 1) + the spgsp 8-G London chain with the **BOTH-SIDE g1 = D_J(g0) + D_I(g0)** (`sa01_g1_bothside`, `sa01_x1i_of_g1`) folding the same 27-product `s[]` into the **36-component** London mix (intor3.c:2100-2135 VERBATIM, all 9 σ-groups, no truncation). Headroom nmax = li+lj+4 (the both-side D_I raises the bra reach to li+3). `launch_int1e_spgsa01_spinor_pair`: rank 9, REAL `cart_to_spinor_si_2d` (Si, Pitfall 2), common_factor 0.5, OWN fail-closed full-block ×9 staging guard + fail-closed nroots guard (no clamp).
- **sigma_1e.rs dispatch**: spgnucsp/spgsa01 added to `giao_family_id` (7/8), `giao_family_rank` (3/9), `giao_family_transform` (SiI / Si REAL), + 2 dispatch arms threading the London origin=ri (neither reads common_orig); spgnucsp threads atom-summed nuclear centers, spgsa01 threads `rinv_orig` (env[PTR_RINV_ORIG]).
- **spgnucsp byte-identity gate** + the **full-9-family test_no_silent_skip**: spgnucsp byte-identical at atol=1e-12 on the NON-SQUARE p(LT,nctr=2)×d(GT) block, all-3 non-zero; oracle_covered=true. test_no_silent_skip asserts 8 of 9 covered (the 6 rank-3 + the 2 cg/giao sa01 rank-9) + spgnucsp byte-identity + spgsa01 RUNs/non-zero/false.
- **oracle_covered flip** for exactly `int1e_spgnucsp_spinor` (spinor-only, rank 3). api_manifest.rs/.csv regenerated from the lock by the cintx-ops build. manifest-audit green; no capi/legacy surface.

## Task Commits
1. **Task 1: spg-Rys/London kernels + dispatch arms** — `d711266` (feat) — builds green
2. **Task 2: spgnucsp byte-identity gate + flip; spgsa01 honest-deferred + headroom fix** — `142d023` (test)

## Files Modified
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — `spgnucsp_rys_kernel` + `run_spgnucsp_rys_device/on_backend` + `launch_int1e_spgnucsp_spinor_pair`; `spgsa01_rys_kernel` + `run_spgsa01_rys_device/on_backend` + `launch_int1e_spgsa01_spinor_pair`; spgsa01 headroom li+lj+4.
- `crates/cintx-cubecl/src/kernels/sigma_1e.rs` — spgnucsp/spgsa01 in `giao_family_id`/`giao_family_rank`/`giao_family_transform` + 2 dispatch arms.
- `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` — `collect_cintx_spg`/`collect_vendor_spg`, `giao_sigma_1e_spgnucsp` gate, `giao_sigma_1e_spgsa01` gate (#[ignore]d WIP), full-9-family `test_no_silent_skip` (sections C/D/E).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — flipped `oracle_covered` false→true for `int1e_spgnucsp_spinor` only (1-field diff; spgsa01 stays false).
- `crates/cintx-ops/src/generated/api_manifest.rs` / `.csv` — regenerated by the cintx-ops build from the lock.

## Deviations from Plan

### RULE 1 — spgsa01 headroom (li+lj+3 → li+lj+4)
- **Found during:** Task 2 spgsa01 vendor gate (small residual).
- **Issue:** The both-side g1 = D_J + D_I raises the deepest g0 bra read to li+3 (one deeper than spgnucsp's li+2). The initial li+lj+3 was boundary-correct-by-luck on the fixture.
- **Fix:** nmax = li+lj+4 in both the kernel and the device buffer sizing. (Stress-tested +6 — identical output, confirming headroom is now sufficient; the residual is NOT a headroom issue.)
- **Committed in:** `142d023`.

### RULE 4 / Blocker — spgsa01 NOT byte-identical (~0.5% uniform residual)
- **What:** `int1e_spgsa01` diverges from vendored libcint by a small (~0.5%), UNIFORM residual across ALL components (not structural zeros, not a wrong σ-group, not a transform/re-im swap). spgnucsp (identical engine minus the `+D_I` in g1) is byte-PERFECT.
- **Isolation performed (dual-verification, spike-findings method):**
  1. **Component spread:** all 9 σ-groups mismatch ≈ equally (396 of ~432 elements) → a globally-shared g-tensor term, NOT a per-group gout transcription error.
  2. **D_I-in-g1 isolation:** forcing g1 = D_J only (no D_I) → ~1.2e-2 error; the FULL both-side g1 → ~6e-5 residual; g5/g7-neighbors-only D_J → ~5.3e-2 (worse). ⇒ the both-side D_I term is NEEDED and is mostly correct; the residual is a small systematic offset on the otherwise-correct D_I contribution.
  3. **Headroom:** nmax +4 vs +6 → byte-identical output ⇒ NOT a headroom/uninitialized-slot issue.
  4. **Transcription:** the spgsa01 36-comp gout was diff'd against intor3.c:2100-2135 (match); the 27-product s[] is byte-identical to spgnucsp's (which passes); the per-index recurrences (`sa01_nabla_i`, `sigma_p_nabla_j`, `sigma_p_x1i`, `sa01_g1_bothside`, `sa01_x1i_of_g1`, `sigma_p_nabla_i_combine`) were verified line-by-line against `libcint-master/src/g2e.c` (`CINTnabla1i_2e`, `CINTnabla1j_2e`, `CINTx1i_2e`) — all match. cg/giao_sa10sa01 (rank 9, same REAL transform, same `sa01_g1_bothside` exact-index, same `sa01_x1i_of_g1` for g3) are byte-perfect, so the both-side g1 EXACT and g3 paths are proven.
- **Residual localization:** spgsa01 is the ONLY family that reads g0 at bra li+3 (via g7=D_I(R0I(both-side g1)) — the D_I-in-g1 adds one bra level over spgnucsp's li+2). The residual is concentrated in this deepest D_I-in-g1 → g3 → g7 chain. The exact term is not yet pinned by inspection; it needs the cart-discriminator dual-verification (the same instrument 30-01c used to root-cause its sa01 blocker — compare the PRE-transform cart `gc` against a vendor cart to isolate the g-tensor term).
- **Disposition:** did NOT flip `int1e_spgsa01_spinor` oracle_covered (no coverage claim on a non-byte-identical family, T-30-01d-06); marked the spgsa01 byte-identity gate `#[ignore]` with a precise WIP reason; `test_no_silent_skip` asserts spgsa01 RUNs + all-9-non-zero but stays oracle_covered=false. No fabricated green. Fix-attempt limit (≥3) reached after the headroom fix + multiple isolations.

## Known Stubs / Incomplete
- **`int1e_spgsa01` is NOT byte-identical** to vendor and remains `oracle_covered=false` (rank 9). The rank-9 spg-Rys/London engine produces all-9 non-zero output but with a ~0.5% uniform residual in the D_I-in-g1 chain. The byte-identity gate is `#[ignore]`d (`30-01d WIP: rank-9 spgsa01 both-side-g1 ~0.5% residual`).
- **The 1e half of GIAO-03 is NOT fully closed** — 8 of 9 1e GIAO×σ families are oracle_covered=true; spgsa01 is the lone remaining gap.

## Verification Performed
- `cargo build -p cintx-cubecl --features cpu` / `cargo build -p cintx-oracle --features cpu` — exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity` — **10 passed, 1 ignored (spgsa01 WIP), 0 failed**: `giao_sigma_1e_spgnucsp`, the 7 prior family gates, `giao_sigma_micro`, the collapse witnesses, and `test_no_silent_skip` all green at atol=1e-12 on the non-square block; only `giao_sigma_1e_spgsa01` is ignored.
- WITHOUT the env var: only `test_kappa_sizing_non_4l_plus_2` runs (1 passed) — double-gate confirmed (no silent pass, T-30-01d-06).
- `cargo run -p xtask -- manifest-audit` — exit 0.
- Acceptance greps: spgnucsp/spgsa01 present in sigma_p.rs (43) and sigma_1e.rs (13); `BufferTooSmall` per spg arm; no partial `if dst < staging.len()` guards (0); no nroots clamp (0); `UnsupportedApi` nroots guards present.
- `git diff --stat crates/cintx-capi/` empty; no new `cint1e_*` legacy wrappers; OperatorId resolved by symbol name (no positional shift).
- Python manifest check: `int1e_spgnucsp_spinor` oracle_covered=true (rank 3); `int1e_spgsa01_spinor` false (rank 9).

## Next Plan Readiness
- **Wave 1 is NOT fully gated green** — spgsa01 must be root-caused before the full 9-family 1e gate closes. Before/alongside 30-02:
  1. **Resolve the spgsa01 ~0.5% residual** (the blocker): use the cart-discriminator dual-verification (add a `spgsa01_cart_gc_for_test`-style PRE-transform cart dump, compare against a vendor cart for `int1e_spgsa01_cart`) to localize the exact D_I-in-g1 → g3 → g7 term. The residual is isolated to the deepest g0[bra li+3] read that only spgsa01 exercises. Un-ignore the gate, confirm byte-identity, flip `oracle_covered=true`, regenerate api_manifest.
  2. spgnucsp is gated green and may be relied upon by Wave 2.
- Re-verify command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity -- --include-ignored`.

## Self-Check: PASSED (with PARTIAL status)
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::spgnucsp_rys_kernel
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::spgsa01_rys_kernel
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::launch_int1e_spgnucsp_spinor_pair
- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs::giao_sigma_1e_spgnucsp
- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs::test_no_silent_skip
- FOUND commit: d711266 (Task 1)
- FOUND commit: 142d023 (Task 2)
- NOTE: spgnucsp byte-identical + oracle_covered=true; spgsa01 NOT byte-identical, oracle_covered=false (gate #[ignore]d). Wave-1 1e gate NOT fully closed.

---
*Phase: 30-group-5-giao-slice-spin-giao-integrals-spinor*
*Completed (partial): 2026-06-01 — spgnucsp green, spgsa01 BLOCKED on ~0.5% residual*
