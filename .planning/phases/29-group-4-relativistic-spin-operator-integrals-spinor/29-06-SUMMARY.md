---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
plan: 06
subsystem: oracle-parity
tags: [spinor, 2e, rel-03, rel-04, gaunt1, dkb, sigma, launcher, byte-identity, vendor-parity, wave-3, relativistic, phase-complete]

# Dependency graph
requires:
  - phase: 29 (plan 05)
    provides: gaunt1.c/dkb.c vendor build wiring + 15 int2e_*_spinor manifest rows + 15 vendor shims + rel_2e_sigma_parity RED scaffold
  - phase: 29 (plan 04)
    provides: int2e_spsp1 Spinor launcher (the reference arm) + int2e_common_factor + the D-03-proven 2e si/sf transform suite + build_kappa_spinor_2e_fixture
  - phase: 29 (plan 03)
    provides: the 6-fn 2e cart→spinor si/sf transform suite (cart_to_spinor_si_2e1/2e1i/2e2/2e2i + sf_2e1/sf_2e2)
provides:
  - all 16 2e Group-4 σ Spinor launcher arms (REL-03 spsp1/srsr1/spsp1spsp2/srsr1srsr2 + REL-04 ssp1ssp2/ssp1sps2/sps1ssp2/sps1sps2/spv1/vsp1/spv1spv2/vsp1spv2/spv1vsp2/vsp1vsp2/spv1spsp2/vsp1spsp2), byte-identical to vendored libcint 6.1.3 at atol=1e-12
  - gout_srsr1 (σ·r₁ R-shift), gout_spsp1spsp2/gout_srsr1srsr2 (rank-16 σ⊗σ), and the generic REL-04 gout engine (build_rel2e_cascade + rank-3/9/27 s[] + per-family folds), all transcribed verbatim
  - launch_rel2e_sigma_spinor_quartet + rel2e_family_dispatch: the generic family-parameterized launcher (gout × e1×e2 transform pair) with per-arm fail-closed staging guard
  - all 16 2e Group-4 rows flipped oracle_covered=true spinor-only → 24/24 Group-4 spinor families covered (8 1e + 16 2e); Phase 29 / Group 4 COMPLETE
affects: [30-giao-sigma, 31-breit-gaunt]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2-sided σ⊗σ family headroom is {1,1,1,1} (ALL four indices raised), NOT {1,1,1,0} — the σ·p₂/σ·r₂ operator nablas/shifts both k AND l. THE bug that produced 280/384 mismatches on the 2-sided families until corrected; the 1-sided (spsp1/srsr1) families stay {1,1,0,0}. Verify the headroom against each driver's optimizer ng[0..3] verbatim."
    - "2e σ⊗σ gout component order = e2*4 + e1 (e1 FAST), proven from libcint CINT2e_spinor_drv: it loops m∈ncomp_e2 calling f_e1_c2s(opij+n1*m, gctr) then advances gctr += nc*ncomp_e1 — so the e1 transform consumes the inner 4 gctr blocks per e2-group. opij[e2c] feeds ox/oy/oz/o1 (e2 σx,σy,σz,scalar) to c2s_si_2e2."
    - "G2E_R_I/J/K/L are pure G-tensor POINTER SHIFTS by g_stride_* (g2e.h:104-107: f = g + envs->g_stride_*), NOT position operators — in cintx terms g_stride_i == shape.di etc. srsr1/srsr1srsr2 read shifted indices on a same-headroom G-tensor (identical s[] + fold to spsp1/spsp1spsp2, only the derivative setup differs)."
    - "REL-04 gout engine = (build_rel2e_cascade applying the D_I/J/K/L nabla composition per family's optimizer ng + cascade) + (rank-3/9/27 s[] triple-product engine) + (a per-family fold closure s[] → ncomp). gaunt ssp/sps → si_2e1i+si_2e2i (BOTH imaginary); dkb vsp/spv → si_2e1[+sf/si]_2e2. Each family's headroom + cascade + fold transcribed verbatim from gaunt1.c/dkb.c."

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-oracle/tests/rel_2e_sigma_parity.rs
    - crates/cintx-oracle/tests/si_2e_transform_parity.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv

key-decisions:
  - "The 2-sided σ⊗σ headroom is {1,1,1,1}. Initially used {1,1,1,0} (raising i,j,k only), which gave 280/384 mismatches on spsp1spsp2 + srsr1srsr2 despite a byte-perfect gout, proven-correct e1 transform, and a libcint-faithful e2 transform. The all-s isolation test localized it to the G-tensor: libcint's ng={1,1,1,1,4,4,4,1} raises ALL FOUR angular indices because σ·p₂ nablas both k and l (the D_L step at l+0 still reads g0 at l+1). With l_inc=1 added, every value became byte-identical."
  - "Built a generic family-parameterized launcher (launch_rel2e_sigma_spinor_quartet) + gout engine (build_rel2e_cascade + rank-3/9/27 engines + per-family fold closures) rather than 16 hand-written launchers. The 16 gout folds + cascades were extracted verbatim from gaunt1.c/dkb.c/intor4.c (programmatically cross-checked s[]+fold byte-for-byte against the C source) and embedded as data; the launcher dispatches on (Rel2eGout, E1Transform, E2Transform) per the RESEARCH 2e map. Every one of the 16 families passed byte-identity on the first run after the engine compiled — strong evidence the transcription is correct."
  - "Flipped oracle_covered via a direct lock edit (the established Phase-27/28/29 pattern for σ families): the xtask oracle-covered-update runs the generic fixture-MATRIX parity, which does not exercise these spinor families (they have no generic-matrix fixture). Their byte-identity is proven by the dedicated rel_2e_sigma_parity.rs vendor double-gate test. Flipped spinor-only (forms stay [spinor], SC#5 — no cart/sph over-claim); only the oracle_covered bool changed (16 lines), so NO OperatorId positional drift."
  - "The 2e σ·p assembler (Open Q2) reused the existing gout_ipvip1 (∇_i∇_j) s[0..8] tensor for spsp1 (29-04) and the same s[] structure for the 2-sided rank-16 families; the genuinely-new code is the rank-16/rank-27 σ⊗σ folds + the R-shift srsr variants + the REL-04 cascade engine. The 2e si σ-mix transform (apply_2d_spinor_zi) from 29-03 was validated byte-identical FOR THE FIRST TIME here (29-03 only proved finiteness)."

patterns-established:
  - "Pattern: a Wave-3 family-batch LAUNCHER plan = (transcribe each family gout verbatim from the C source — cross-check s[]+fold programmatically) + (a generic gout engine parameterized by cascade+fold) + (a generic launcher dispatching gout × per-electron transform pair with a per-arm fail-closed staging guard) + (drop #[ignore], turn each byte-identity gate green) + (flip oracle_covered spinor-only via direct lock edit after parity green)."

requirements-completed: [REL-03, REL-04]

# Metrics
duration: 95min
completed: 2026-06-01
---

# Phase 29 Plan 06: Wave-3 2e σ Launcher Arms (REL-03/04) Summary

**Wired all 16 remaining 2e Group-4 relativistic σ Spinor launcher arms onto the proven 2e si/sf transform suite, turning `rel_2e_sigma_parity.rs` fully GREEN (18/18) at atol=1e-12 on the non-square nctr>1 kappa fixture under the vendor double-gate, then flipped every family `oracle_covered=true` spinor-only. The load-bearing finding: the 2-sided σ⊗σ families (`spsp1spsp2`, `srsr1srsr2`, and the dkb/gaunt 2-sided families) need `{1,1,1,1}` G-tensor headroom (ALL four angular indices raised) — using `{1,1,1,0}` produced a stubborn 280/384 mismatch despite a byte-perfect gout and verbatim transforms, because the σ·p₂ operator nablas both k AND l. With that corrected, all 16 families — spsp1/srsr1 (rank-4 σ·p/σ·r), spsp1spsp2/srsr1srsr2 (rank-16 σ⊗σ), the four gaunt ssp/sps (`c2s_si_2e1i+si_2e2i`, both imaginary), spv1/vsp1 (rank-4 σ·∇), and the six dkb 2-sided vsp/spv (rank-9 + rank-27) — are byte-identical to vendored libcint 6.1.3. This closes Group 4: 24/24 Group-4 spinor families (8 1e + 16 2e) are now `oracle_covered=true`.**

## Performance

- **Duration:** ~95 min (the 2-sided headroom diagnosis was ~half of it)
- **Completed:** 2026-06-01
- **Tasks:** 3
- **Files:** 7 modified (0 created)

## Accomplishments

- **Task 1 — REL-03 (srsr1, spsp1spsp2, srsr1srsr2):** Added `gout_srsr1` (σ·r₁, rank-4: the σ·r operator's `G2E_R_*` are pure G-tensor pointer shifts by `g_stride_*`, so the s[] + fold are identical to `gout_spsp1` — only the derivative setup is index-shifts instead of nablas), `fold_2sided_sigma16` (the shared 81-term s[] + 16-component σ⊗σ fold), `gout_spsp1spsp2` (nabla cascade) and `gout_srsr1srsr2` (R-shift cascade), all transcribed verbatim from intor4.c (s[] + fold cross-checked byte-for-byte against the C via a Python diff — 0 mismatches). Built the generic `launch_rel2e_sigma_spinor_quartet` + `rel2e_family_dispatch` (dispatches gout × e1×e2 transform pair) with a per-arm fail-closed staging guard. **Fixed the 2-sided `{1,1,1,1}` headroom** (the 280-mismatch bug). srsr1/spsp1spsp2/srsr1srsr2 byte-identical at atol=1e-12.
- **Task 2 — REL-04 (gaunt ssp/sps imaginary + dkb vsp/spv):** Added the generic REL-04 gout engine — `build_rel2e_cascade` (applies the D_I/J/K/L nabla composition per family) + `gout_rel2e_rank3`/`rank9`/`rank27` s[] engines + per-family fold closures — and 12 per-family gout functions (`gout_ssp1ssp2`/`ssp1sps2`/`sps1ssp2`/`sps1sps2` rank-9; `gout_spv1`/`vsp1` rank-4; `gout_spv1spv2`/`vsp1spv2`/`spv1vsp2`/`vsp1vsp2` rank-9; `gout_spv1spsp2`/`vsp1spsp2` rank-27), every cascade+fold+headroom transcribed verbatim from gaunt1.c/dkb.c. Extended `Rel2eGout`/`rel2e_family_dispatch` with the REL-04 families + their transform pairs (gaunt ssp/sps → `si_2e1i`+`si_2e2i`, both imaginary; dkb vsp/spv → `si_2e1`+`sf_2e2`/`si_2e2`). All 12 REL-04 families passed byte-identity on the FIRST run after the engine compiled.
- **Task 3 — oracle_covered flips + phase gate:** Flipped all 16 2e Group-4 rows `oracle_covered` false→true in `compiled_manifest.lock.json` **spinor-only** (forms stay `[spinor]`, component_rank stays `1`; only the bool changed → 16-line diff, no OperatorId drift). Regenerated `api_manifest.rs`/`.csv` from the lock (32-line diff). Updated the `test_all_rel_2e_rows_registered`, `test_no_silent_skip`, and `si_2e_transform_parity::test_no_silent_skip` assertions to expect the flipped state (each runs non-skipped under the double gate — the vendor arm is proven to execute). `manifest-audit` exits 0 (status ok). 24/24 Group-4 spinor families `oracle_covered=true`. **Group 4 complete.**

## Task Commits

1. **Task 1: REL-03 launcher arms (srsr1, spsp1spsp2, srsr1srsr2)** — `770ee4d` (feat)
2. **Task 2: REL-04 launcher arms (ssp/sps imaginary + vsp/spv)** — `f896e2d` (feat)
3. **Task 3: flip all 16 2e Group-4 oracle_covered=true (spinor-only)** — `6b60e06` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/kernels/f12.rs` — `gout_srsr1`, `fold_2sided_sigma16`, `gout_spsp1spsp2`, `gout_srsr1srsr2`; the REL-04 gout engine (`build_rel2e_cascade` + `s9_products` + `gout_rel2e_rank3/9/27`) + 12 per-family REL-04 gouts.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `Rel2eGout`/`E1Transform`/`E2Transform` enums, `rel2e_family_dispatch`, `launch_rel2e_sigma_spinor_quartet` + plan wrapper + dispatch arm.
- `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` — cintx collector wired to the generic launcher; all 15 gates LIVE; rows-registered + no-silent-skip assert post-flip oracle_covered=true.
- `crates/cintx-oracle/tests/si_2e_transform_parity.rs` — `test_no_silent_skip` updated for the spsp1 flip.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (+ regenerated `.rs`/`.csv`) — 16 oracle_covered flips.

## Decisions Made

- **`{1,1,1,1}` headroom for 2-sided σ⊗σ (the key fix).** Documented fully above; the all-s isolation test was decisive — every other component (gout s[]+fold byte-perfect vs C, e1 transform proven by spsp1, e2 transform libcint-faithful) checked out, leaving only the G-tensor headroom.
- **Generic gout engine + launcher over 16 hand-written arms.** Extracted the cascades + folds verbatim (programmatically) and embedded them as data; the launcher is parameterized by `(Rel2eGout, E1Transform, E2Transform)`. Every family passing byte-identity on first compile validated the transcription.
- **Direct lock edit for the flips** (established σ-family pattern); spinor-only, bool-only, no row insertion → no OperatorId drift.

## Deviations from Plan

None requiring a Rule-4 stop. The plan was executed as written. One in-plan diagnosis (not a deviation): the plan's Task-1 `<read_first>` pointed at `center_4c1e.rs::test_device_matches_host_spsp` for the 2e σ·p gout, but (as 29-04 already found) that test is an angular-momentum device/host check, not a σ·p assembler — the gouts were instead transcribed verbatim from intor4.c/gaunt1.c/dkb.c, which is strictly better provenance. The 280-mismatch 2-sided-headroom bug was found and fixed inline (an auto-fix Rule-1 within the task's own new code, not a pre-existing defect).

## Issues Encountered / Deferred

- **Pre-existing `cintx-oracle` lib-test failures (180 mismatches in `compare::tests::*` + `fixtures::tests::unstable_source_fixtures_require_opt_in`), logged to `deferred-items.md` §29-06.** **Verified independent of this plan:** reverted the Task-3 lock flip (stashed `compiled_manifest.lock.json` + regenerated `.rs`/`.csv`) and re-ran `compare::tests::evaluated_output_parity_and_optimizer_equivalence_hold` → **identical 180 mismatches**. The flip did NOT change the count: the 16 Group-4 σ families have no generic-matrix fixture (their byte-identity is proven by `rel_2e_sigma_parity.rs`, 18/18 green), so they are not in the `compare::tests` parity matrix. Pre-existing whole-fixture-matrix issue documented since 29-03/04/05 + project memory `oracle_vendor_lib_tests_uncovered`. Out of scope (SCOPE BOUNDARY).
- The 29-04 `int2e_common_factor` commit-hygiene gap (29-05 deferred item) was **already resolved** before 29-06 began (commit `dd5b772`); no action needed.

## Known Stubs

None. All 16 launcher arms + 16 gout functions + the generic engine are fully implemented and exercised; every one is byte-identical to vendored libcint 6.1.3. The cintx collector in `rel_2e_sigma_parity.rs` (formerly a RED panic-stub) now drives the real launchers. No family was left `oracle_covered=false` — every Group-4 driver has a real vendor byte-identity reference (no return-0/exit(1) stubs in this batch, unlike the Phase-27 spinor-deriv arms).

## Threat Flags

None new. The two registered threats are addressed:
- **T-29-11 (Tampering/DoS, per-arm output buffer, `mitigate`):** `launch_rel2e_sigma_spinor_quartet` has a fail-closed staging guard `required = ni_sp*nj_sp*nk_sp*nl_sp*2` that returns `BufferTooSmall` BEFORE any write (OOM-safe stop, no partial writes; this inline 2e arm bypasses any `launch_*_pair` guard).
- **T-29-12 (Spoofing, oracle_covered on a skipped/unlinked fixture, `mitigate`):** every flip is preceded by a GREEN byte-identity gate AND the always-on `test_no_silent_skip` asserts the vendor arm executed non-skipped under the double gate; flips are spinor-only (SC#5).

No new network/auth/file-access surface (host-side numerical launchers + gout transcription + manifest bool flips + oracle tests).

## Next Phase Readiness

- **Group 4 COMPLETE.** All REL-01..04 families are `oracle_covered=true` spinor-only (24/24); the 1e (`rel_1e_sigma_parity` 10/10) + 2e (`rel_2e_sigma_parity` 18/18) parity suites + the `si_2e_transform_parity` micro-test (4/4) are GREEN under the double gate; `manifest-audit` exits 0; `cintx-ops` resolver 13/13 (no OperatorId drift); `cintx-cubecl` lib 310/310.
- **Phase 30 (GIAO×σ) / Phase 31 (Breit–Gaunt) readiness:** the 2e si/sf transform suite, the generic `launch_rel2e_sigma_spinor_quartet` + `build_rel2e_cascade` gout engine, and the R-shift handling are reusable templates. The GIAO×σ 2e drivers (intor4.c L636/899/… `cg_sa10sp1`/`giao_sa10sp1`) reuse these transforms; the Breit/Gaunt gauge families extend the same cascade+fold engine.
- No blockers introduced by this plan.

## Self-Check: PASSED

- All 3 task commits present in git history: `770ee4d`, `f896e2d`, `6b60e06` (FOUND).
- All 16 gout functions + `launch_rel2e_sigma_spinor_quartet` + `rel2e_family_dispatch` grep-confirmed in source; the test references the committed symbols (`int2e_common_factor`, `launch_rel2e_sigma_spinor_quartet`, `rel2e_family_dispatch`) — all committed.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test rel_2e_sigma_parity` = 18 passed / 0 failed (every family byte-identical, vendor arm non-skipped); `--test si_2e_transform_parity` = 4/0; `--test rel_1e_sigma_parity` = 10/0.
- `cargo build --workspace --features cpu` exits 0; `cargo test -p cintx-cubecl --features cpu --lib` = 310/0; `cargo test -p cintx-ops --lib` = 13/0 (no OperatorId drift).
- `manifest-audit` exits 0 (status ok); 24/24 Group-4 spinor rows `oracle_covered=true` spinor-only (verified programmatically).
- Pre-existing `cintx-oracle` lib failures (180 mismatches) verified independent of the Task-3 flip (reverted-lock reproduced identical count).

---
*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Completed: 2026-06-01*
