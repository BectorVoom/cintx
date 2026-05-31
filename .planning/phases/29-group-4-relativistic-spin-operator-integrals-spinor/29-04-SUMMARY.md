---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
plan: 04
subsystem: oracle-parity
tags: [spinor, 2e, si_2e, spsp1, sigma-p, byte-identity, vendor-parity, d-03, blocking, wave-2, relativistic]

# Dependency graph
requires:
  - phase: 29 (plan 03)
    provides: the 2e cart→spinor si/sf transform suite (cart_to_spinor_si_2e1 + cart_to_spinor_sf_2e2) + build_kappa_spinor_2e_fixture (D-02)
  - phase: 29 (plan 02)
    provides: 7 1e Group-4 σ launcher arms; int1e_sp flip; the 1e σ·p launcher template (one_electron.rs is_sp arm)
  - phase: 25 (HESS-02)
    provides: gout_ipvip1 (∇_i∇_j) + fill_g_tensor_2e host Rys + the hess2e component-leading cart-block transpose pattern
provides:
  - int2e_spsp1_spinor manifest row (arity 4, component_rank 1, forms [spinor], oracle_covered false) + vendor_int2e_spsp1_spinor shim
  - gout_spsp1 — the σ·p₁ (σ·∇_i)(σ·∇_j) G-tensor assembler (= libcint CINTgout2e_int2e_spsp1, intor4.c:19-58 verbatim)
  - launch_int2e_spsp1_spinor_quartet + plan-based arm in launch_two_electron_typed (the first real 2e σ Spinor launcher)
  - int2e_common_factor helper (single-sourced 2e prefactor for external drivers)
  - si_2e_transform_parity.rs — the D-03 BLOCKING transform-level byte-identity micro-test (GREEN at atol=1e-12)
affects: [29-05, 29-06, 29-group-4-wave-3, 30-giao-sigma, 31-breit-gaunt]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2e σ family launcher = (host σ·p G-tensor assembler producing the 4 contiguous gc_x/gc_y/gc_z/gc_1 cart blocks per quad) → cart_to_spinor_si_2e1 (electron 1, owns KET→BRA transpose) → cart_to_spinor_sf_2e2 (electron 2, zcopy_iklj) → contraction-major scatter; fail-closed staging guard (required = ni*nj*nk*nl*2) BEFORE any write"
    - "the σ·p₁ gout (spsp1) REUSES the ipvip1 (∇_i∇_j) s[0..8] triple-product tensor verbatim — the only difference is the final fold: spsp1 emits 4 σ blocks (gc_x=s5-s7, gc_y=s6-s2, gc_z=s1-s3, gc_1=s0+s4+s8) instead of ipvip1's 9 raw components. Same g1=nabla1j(g,li+1), g2=nabla1i(g), g3=nabla1i(g1) derivative setup, same (i+1,j+1) headroom"

key-files:
  created:
    - crates/cintx-oracle/tests/si_2e_transform_parity.rs
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs

key-decisions:
  - "int2e_spsp1's σ·p₁ gout reuses the existing gout_ipvip1 s[0..8] tensor (verified against libcint CINTgout2e_int2e_spsp1, intor4.c:28-46: g1=G2E_D_J(g0,i_l+1), g2=G2E_D_I(g0), g3=G2E_D_I(g1) — identical to ipvip1's nabla1j(li+1)/nabla1i/nabla1i(g1)). gout_spsp1 differs ONLY in the final fold (4 σ blocks vs 9 raw comps) and headroom matches Hess2eKind::Ipvip1 = (i+1, j+1, k+0)"
  - "Drove the cintx side via a standalone pub launch_int2e_spsp1_spinor_quartet (the 2e analog of launch_int1e_sp_spinor_pair) called directly by the test, mirroring the 28-FND-05 test pattern; the plan-based launch_int2e_spsp1_spinor wrapper feeds the same quartet fn from eval_raw. int2e_common_factor exposes the prefactor so the test uses the IDENTICAL normalization without duplicating the constant"
  - "Manifest row inserted at index 348 (right after int1e_sp_spinor at 347) — past every hardcoded OperatorId const (the highest is int2e_stg_sph at 106 / int4c1e_cart at 24), so no positional OperatorId drift; cintx-ops resolver tests stay green and api_manifest.rs/csv regenerate from the lock via build.rs"
  - "spsp1 stays oracle_covered=false this plan (D-03 ordering invariant): the proof of the 2e transform suite is THIS dedicated transform micro-test, not a coverage flip. 29-06 flips it when the family is registered with its full eval_raw wiring"

patterns-established:
  - "Pattern: a new 2e σ family = (mine its libcint gout from intor4.c/gaunt1.c/dkb.c → if it shares an existing derivative tensor, reuse it and change only the fold) + (compose the per-electron transform pair from the 29-03 suite) + (fail-closed staging guard) + (zcopy_iklj contraction-major scatter). spsp1 is the reference."

requirements-completed: [REL-03]

# Metrics
duration: 55min
completed: 2026-06-01
---

# Phase 29 Plan 04: Wave-2 2e Transform Byte-Identity Gate (D-03 BLOCKING) Summary

**Discharged the D-03 [BLOCKING] structural mitigation: the brand-new 2e cart→spinor si/sf transform suite from 29-03 (`cart_to_spinor_si_2e1` + `cart_to_spinor_sf_2e2`) is now PROVEN byte-identical to vendored libcint at atol=1e-12, by driving the thinnest 2e σ family `int2e_spsp1_spinor` through it and comparing to `vendor_int2e_spsp1_spinor` on the non-square (2,6,2,4) nctr>1 GT/LT kappa fixture. The key finding: `int2e_spsp1`'s σ·p₁ gout reuses the existing `gout_ipvip1` (∇_i∇_j) s[0..8] triple-product tensor verbatim — only the final σ fold (gc_x=s5−s7, gc_y=s6−s2, gc_z=s1−s3, gc_1=s0+s4+s8) and a name differ. The gate (`si_2e_transform_parity.rs`) is GREEN with the vendor arm running NON-SKIPPED under the double gate, so Wave 3 (29-05/06) is unblocked.**

## Performance

- **Duration:** ~55 min
- **Completed:** 2026-06-01
- **Tasks:** 3
- **Files:** 7 (1 created, 6 modified; +834 lines, of which the regenerated manifest .rs/.csv)

## Accomplishments

- **Task 1 — manifest row + vendor shim + allowlist:** Added the `int2e_spsp1_spinor` ManifestEntry (arity 4, canonical_family/category "2e", `component_rank "1"`, `forms ["spinor"]`, `complex_output true`, `oracle_covered false`, stable) to `compiled_manifest.lock.json`; `api_manifest.rs`/`.csv` regenerate from the lock via build.rs. Cloned `vendor_int1e_sp_spinor` → `vendor_int2e_spsp1_spinor` (shls `&[i32;4]`, out sized `ni*nj*nk*nl*2` via `vendor_CINTcgto_spinor`) in `vendor_ffi.rs`. Appended `int2e_spsp1_spinor` to the build.rs bindgen allowlist regex (intor4.c was already compiled — no `.file()` change). Re-checked OperatorId drift: the row lands at index 348, past every hardcoded const (≤106), so the cintx-ops resolver tests (13/13) and the workspace build stay green.
- **Task 2 — σ·p₁ assembler + launcher arm (the genuinely-new 2e code):** Added `gout_spsp1` to f12.rs — a near-clone of `gout_ipvip1` confirmed (via `libcint-master/src/autocode/intor4.c:19-58`) to share the IDENTICAL `s[0..8]` triple-product tensor and derivative setup (`g1 = nabla1j(g, li+1)`, `g2 = nabla1i(g)`, `g3 = nabla1i(g1)`); it differs only by emitting the 4 σ-tensor cart blocks `gc_x = s[5]−s[7]`, `gc_y = s[6]−s[2]`, `gc_z = s[1]−s[3]`, `gc_1 = s[0]+s[4]+s[8]`. Added `launch_int2e_spsp1_spinor_quartet` (+ the plan-based `launch_int2e_spsp1_spinor` wrapper, dispatched in `launch_two_electron_typed` before the scalar path): σ·p assembler (headroom = ipvip1 `(i+1, j+1)`, weighted/summed over every primitive and contraction quad, transposed into the 4 contiguous component-leading gc blocks) → `cart_to_spinor_si_2e1` (electron 1, owns the KET→BRA transpose) → `cart_to_spinor_sf_2e2` (electron 2, spin-free) → `zcopy_iklj` contraction-major scatter. A fail-closed staging guard `required = ni_sp*nj_sp*nk_sp*nl_sp*2` returns `BufferTooSmall` BEFORE any write (T-29-07 / Phase-28 CR-01 — this inline 2e arm bypasses any `launch_*_pair` guard). Spinor-only (cart/sph rejected this phase); nctr>1 handled. Added `int2e_common_factor` to single-source the prefactor.
- **Task 3 — [BLOCKING] gate GREEN:** Created `crates/cintx-oracle/tests/si_2e_transform_parity.rs` (cloned from `si_transform_parity.rs`). Drives `int2e_spsp1_spinor` (cintx, via `launch_int2e_spsp1_spinor_quartet`) against `vendor_int2e_spsp1_spinor` on `build_kappa_spinor_2e_fixture`.
  - **PRIMARY GATE** `test_int2e_spsp1_kappa_spinor_byte_identity`: 0 mismatches at `ATOL=1e-12` over all 384 = (2·2)·(1·6)·(1·2)·(1·4)·2 elements.
  - **NO-SILENT-SKIP** `test_no_silent_skip`: under `has_vendor_libcint` the vendor arm MUST run and produce nonzero output (FAIL not skip); asserts `int2e_spsp1_spinor` stays `oracle_covered=false`.
  - **Kappa-sizing** `test_kappa_sizing_2e_non_4l_plus_2`: GT/LT (2l / 2l+2) on all four shells, never 4l+2.
  - Verified the vendor arm genuinely ran: 4 tests pass under `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu`; 3 pass on the determinism-only build (the byte-identity + no-silent-skip arms compile out without the env flag, exactly as the double gate intends).

## Task Commits

1. **Task 1: manifest row + vendor shim + allowlist** — `f91e351` (feat)
2. **Task 2: gout_spsp1 + launcher arm** — `fece759` (feat)
3. **Task 3: [BLOCKING] si_2e_transform_parity GREEN** — `e40adbc` (test)

## Files Created/Modified

- `crates/cintx-oracle/tests/si_2e_transform_parity.rs` — **created**; the D-03 BLOCKING gate.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (+ regenerated `src/generated/api_manifest.rs` / `.csv`) — `int2e_spsp1_spinor` row.
- `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int2e_spsp1_spinor` shim.
- `crates/cintx-oracle/build.rs` — `int2e_spsp1_spinor` appended to the allowlist regex.
- `crates/cintx-cubecl/src/kernels/f12.rs` — `gout_spsp1`.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `launch_int2e_spsp1_spinor_quartet` + plan wrapper + dispatch arm + `int2e_common_factor`.

## Decisions Made

- **`int2e_spsp1` reuses the `ipvip1` tensor.** Reading `intor4.c:19-58` against `gout_ipvip1` showed the s[0..8] triple products AND the g1/g2/g3 derivative setup are byte-identical (libcint `G2E_D_J(g0,i_l+1)` = cintx `nabla1j(g,li+1)`, etc.). So spsp1 needed no new derivative machinery — only the σ fold and a thin launcher. This is the lowest-risk way to satisfy "Open Q2" (the 2e σ·p assembler) and the cleanest evidence that the σ·p combination is correct: the same tensor that proves byte-identical as `int2e_ipvip1` (Phase-25 HESS-02) is re-folded here.
- **Standalone quartet fn drives the test.** Mirroring the 28-FND-05 `launch_int1e_sp_spinor_pair` pattern, the test calls `launch_int2e_spsp1_spinor_quartet` directly (clean, planner-free), and the eval_raw path feeds the same fn through the plan wrapper. `int2e_common_factor` keeps the (π³·2/√π·∏fac_sp) prefactor single-sourced so the test cannot drift from the eval_raw normalization.
- **Stays `oracle_covered=false`.** The D-03 proof is this transform micro-test; the coverage flip belongs to 29-06 (when the family gets full registration). The no-silent-skip test asserts the flag is still false to enforce the ordering invariant.

## Deviations from Plan

None requiring a Rule-4 stop. The plan was executed as written. One clarification resolved during execution (not a deviation): the plan's Task-2 `<read_first>` pointed at `center_4c1e.rs::test_device_matches_host_spsp` as the σ·p gout source, but that test is the s-**p**-s-**p** *angular-momentum* device/host check, NOT a σ·p assembler — there is no existing 2e σ·p assembler (Open Q2 correctly flagged it as "expected NEW code"). The genuinely-new `gout_spsp1` was instead mined from the live `gout_ipvip1` + the verbatim libcint `CINTgout2e_int2e_spsp1` source, which is a strictly better provenance (a tensor already vendor-validated as `int2e_ipvip1`). Net effect: the new assembler matches the plan's intent exactly.

## Issues Encountered / Deferred

- **Pre-existing out-of-scope `cintx-oracle` lib-test failures (NOT regressions), logged to `deferred-items.md`:**
  - `compare::tests::{evaluated_output_parity_and_optimizer_equivalence_hold, parity_artifacts_are_written, parity_mismatch_report_is_written_before_failure}` — 158 mismatches in the full fixture-matrix oracle parity. **Verified pre-existing** by checking out the baseline manifest at commit `ac1d313` (parent of all 29-04 commits) and re-running `compare::tests::` — fails identically with 158 mismatches and NONE of the 29-04 changes present. The new `int2e_spsp1_spinor` row is `oracle_covered=false`, so it is not in the parity matrix at all.
  - `fixtures::tests::unstable_source_fixtures_require_opt_in` — needs the `unstable-source-api` feature (already documented in 29-03's deferred items).
  - All four are out of scope per the SCOPE BOUNDARY rule (pre-existing, not caused by, nor fixable within, this plan).

## Known Stubs

None. `gout_spsp1`, the launcher, the vendor shim, the manifest row, and the gating test are all fully implemented and exercised. `int2e_spsp1_spinor` intentionally stays `oracle_covered=false` until 29-06 (the planned Wave-2 foundation state, not an incomplete deliverable — the byte-identity is proven by the dedicated micro-test, exactly as D-03 specifies).

## Threat Flags

None new. The two registered threats are addressed: T-29-07 (int2e_spsp1 launcher output buffer Tampering/DoS, `mitigate`) — the fail-closed `BufferTooSmall` staging guard (`required = ni*nj*nk*nl*2`) fires before any write (OOM-safe stop, no partial writes); T-29-08 (oracle_covered spoofing, `mitigate`) — `test_no_silent_skip` asserts the vendor arm executed under the double gate AND that the family stays `oracle_covered=false`. No new network/auth/file-access surface (host-side numerical launcher + transform + an oracle test).

## Next Phase Readiness

- **D-03 discharged — Wave 3 unblocked.** The 2e si/sf transform suite is pinned byte-identical to vendored libcint at spike-level rigor on the thinnest family. A wrong 2e layout/stride/sign would have surfaced HERE; it did not (0 mismatches). 29-05/06 may now wire the remaining 15 Wave-3 2e σ families onto the proven transform.
- **Reference launcher for Wave 3:** `launch_int2e_spsp1_spinor_quartet` is the template — Wave-3 families differ only in (a) the gout fold mined from intor4.c/gaunt1.c/dkb.c, and (b) the electron-1×electron-2 transform pair selected from the 29-03 suite (e.g. spsp1spsp2 → si_2e1+si_2e2; ssp1ssp2 → si_2e1i+si_2e2i). Note REL-04 still requires adding `gaunt1.c` + `dkb.c` to the oracle build.rs (per 29-RESEARCH §REL-04 landmine), out of scope here.
- No blockers introduced by this plan.

## Self-Check: PASSED

- Created file exists on disk: `crates/cintx-oracle/tests/si_2e_transform_parity.rs` (FOUND).
- All 3 task commits present in git history: `f91e351`, `fece759`, `e40adbc` (FOUND).
- `gout_spsp1`, `launch_int2e_spsp1_spinor_quartet`, `vendor_int2e_spsp1_spinor`, and the `int2e_spsp1_spinor` manifest row are grep-confirmed present.
- `cargo build --workspace --features cpu` exits 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test si_2e_transform_parity` = 4 passed / 0 failed (vendor arm ran NON-SKIPPED); 3 passed / 0 failed on the determinism-only build.
- `cargo test -p cintx-cubecl --features cpu --lib c2spinor` = 45/0; `--lib two_electron` = 18/0.

---
*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Completed: 2026-06-01*
