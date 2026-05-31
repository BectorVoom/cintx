---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
verified: 2026-06-01T00:00:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
human_verification_resolved: "2026-06-01 — user approved; the sole human item (vendor-gated parity build) was run live by the orchestrator this session: rel_1e 10/10, rel_2e 18/18, si_2e 4/4, byte-identical at atol=1e-12. Reclassified human_needed → passed."
human_verification:
  - test: "Run vendor-gated parity suites under double gate: CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test rel_1e_sigma_parity && cargo test -p cintx-oracle --features cpu --test rel_2e_sigma_parity && cargo test -p cintx-oracle --features cpu --test si_2e_transform_parity"
    expected: "rel_1e_sigma_parity 10/10, rel_2e_sigma_parity 18/18, si_2e_transform_parity 4/4 — all green, all byte-identical at atol=1e-12; test_no_silent_skip passes under the double gate confirming vendor arms ran non-skipped"
    why_human: "Requires CINTX_ORACLE_BUILD_VENDOR=1 environment which builds libcint 6.1.3 from source; not available in the static verification environment. The orchestrator independently confirmed these results (rel_1e 10/10, rel_2e 18/18, si_2e 4/4) but verifier cannot re-run the vendor build."
---

# Phase 29: Group 4 — Relativistic Spin-Operator Integrals (Spinor) Verification Report

**Phase Goal:** Group 4 — Relativistic Spin-Operator Integrals (spinor). The relativistic σ-operator families (1e: spsp, spnucsp, sprinvsp, srsr, sr, sigma, sp; 2e: spsp1/srsr1/spsp1spsp2/srsr1srsr2/ssp*/sps*/vsp*/spv*) at spinor byte-identity via the Gap B2 c2s_si path.
**Verified:** 2026-06-01T00:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|---------|
| 1  | REL-01: `int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp` match vendored libcint at atol=1e-12 (spinor); spsp uses `cart_to_spinor_sf_2d` (scalar ∇²), spnucsp/sprinvsp use `cart_to_spinor_si_2d` | ✓ VERIFIED | All three present in manifest with `oracle_covered: true`, spinor-only. `sigma_1e.rs` dispatches spsp→TransformKind::Sf, spnucsp/sprinvsp→TransformKind::Si (sigma_1e.rs:622-624). Orchestrator confirms rel_1e 10/10 parity. |
| 2  | REL-02: `int1e_srsr`, `int1e_srnucsr`, `int1e_sr`, `int1e_sigma`, `int1e_sp` match at atol=1e-12 (spinor); sr/sigma use `cart_to_spinor_si_2di` | ✓ VERIFIED | All five present with `oracle_covered: true`, spinor-only. `sigma_1e.rs:623`: `"sr" \| "sigma" => TransformKind::SiI`. `cart_to_spinor_si_2di` exists at c2spinor.rs:754. Orchestrator confirms rel_1e 10/10 green. |
| 3  | `int1e_sigma` component_rank is empirically confirmed as 3 (not assumed 1) and locked in the manifest | ✓ VERIFIED | `test_sigma_rank_measured` (rel_1e_sigma_parity.rs:342) calls vendor_int1e_sigma_spinor on an oversized buffer and asserts `written == 3*rank1_len`. Manifest lock JSON line 11480: `"component_rank": "3"`. api_manifest.rs confirms `component_rank: "3", oracle_covered: true`. The 29-01 rank-1 prior was disproven and corrected. |
| 4  | REL-03: `int2e_spsp1`, `int2e_srsr1`, `int2e_spsp1spsp2`, `int2e_srsr1srsr2` match at atol=1e-12 (spinor) via c2s_si_2e1+c2s_sf_2e2 / c2s_si_2e1+c2s_si_2e2 | ✓ VERIFIED | All four present with `oracle_covered: true`, spinor-only. two_electron.rs imports all transform variants (lines 16-17). Headroom: Spsp1/Srsr1 = (1,1,0,0), Spsp1spsp2/Srsr1srsr2 = (1,1,1,1) (the key {1,1,1,1} finding, two_electron.rs:2979). Orchestrator confirms rel_2e 18/18 green. |
| 5  | REL-04: `int2e_ssp1ssp2`, `int2e_ssp1sps2`, `int2e_sps1ssp2`, `int2e_sps1sps2` (imaginary, via c2s_si_2e1i+c2s_si_2e2i) and all vsp*/spv* families match at atol=1e-12 (spinor) | ✓ VERIFIED | All 12 REL-04 families present with `oracle_covered: true`, spinor-only. two_electron.rs:2836-2848 dispatches ssp/sps→SiI, vsp1/spv1→Si+Sf, 2-sided spv/vsp→Si+Si. gaunt1.c and dkb.c wired in oracle build.rs:237-238. Orchestrator confirms rel_2e 18/18. |
| 6  | The 2e si/sf transform suite (6 functions + apply_2d_spinor_zi) is structurally sound and byte-identical to vendored libcint via the D-03 BLOCKING micro-test | ✓ VERIFIED | All 6 fns present: `cart_to_spinor_{si_2e1,si_2e1i,si_2e2,si_2e2i,sf_2e1,sf_2e2}` (c2spinor.rs). `apply_2d_spinor_zi` at c2spinor.rs:1772 with v11R/v12R/v21R/v22R assignments transcribed verbatim from libcint cart2sph.c:4118-4186. si_2e_transform_parity.rs has ATOL, test_no_silent_skip, build_kappa_spinor_2e_fixture. Orchestrator confirms si_2e 4/4. |
| 7  | All 24 Group-4 spinor families (8 1e + 16 2e) are `oracle_covered=true` spinor-only in the manifest; manifest-audit is green | ✓ VERIFIED | Every family checked: all 8 1e and all 16 2e families have `"oracle_covered": true` and `"forms": ["spinor"]` in compiled_manifest.lock.json. `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit` exits 0 with `status: ok`, `uncovered_stable_entries: []`. |
| 8  | CR-01 (REVIEW BLOCKER) fixed: `sigma_1e_nuc.rs` nroots `.clamp(1,5)` replaced with a fail-closed `UnsupportedApi` guard | ✓ VERIFIED | `grep "clamp" sigma_1e_nuc.rs` returns empty. sigma_1e_nuc.rs:493-498 shows: `const MAX_DEVICE_NROOTS: u32 = 5; let nroots = (order / 2 + 1) as u32; if nroots > MAX_DEVICE_NROOTS { return Err(cintxRsError::UnsupportedApi { requested: ... }); }` — fail-closed, no partial write, typed error. rel_1e stays 10/10 green post-fix. |

**Score:** 8/8 truths verified

---

### Deferred Items

No items deferred to later milestone phases.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` | `cart_to_spinor_si_2di` (imaginary-ket 1e) + 2e transform suite (6 fns) + `apply_2d_spinor_zi` | ✓ VERIFIED | fn at L754. All 6 2e fns at L1410-L1579. apply_2d_spinor_zi at L1772. No hardcoded `4*l+2` in new fns; all use `spinor_len`. KET→BRA transpose owned inside si_2di (c2spinor.rs:40-56). |
| `crates/cintx-cubecl/src/kernels/sigma_1e.rs` | 7 1e σ family launcher arms via unified `launch_int1e_sigma_family_spinor_pair` | ✓ VERIFIED | File exists. `family_dispatch` maps sigma/sr/srsr/spsp/spnucsp/srnucsr/sprinvsp (L66-72). Staging guard with rank: `required = ni_sp * nj_sp * 2 * rank` (L712-718). |
| `crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs` | Rys nuclear σ kernel (spnucsp/srnucsr/sprinvsp); fail-closed nroots guard | ✓ VERIFIED | File exists. `MAX_DEVICE_NROOTS = 5` guard at L493-498. No `.clamp()`. |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | 16 2e σ Spinor launcher arms via `launch_rel2e_sigma_spinor_quartet`; per-arm fail-closed staging guard | ✓ VERIFIED | Generic launcher at L3005. `rel2e_family_dispatch` at L2825 covers all 16 families. Staging guard at L3037-3044 (`required = ni_sp * nj_sp * nk_sp * nl_sp * 2`). All 5 transform variants imported (L16-17). |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | 24 Group-4 spinor rows (8 1e + 16 2e), oracle_covered=true, spinor-only, component_rank 1 (except sigma=3) | ✓ VERIFIED | All 24 rows present. int1e_sigma component_rank="3"; all others "1". All `"oracle_covered": true`. All `"forms": ["spinor"]`. |
| `crates/cintx-oracle/build.rs` | gaunt1.c + dkb.c .file() entries; full allowlist with all 24 new symbols | ✓ VERIFIED | Lines 237-238 have `.file(libcint_root.join("src/autocode/gaunt1.c"))` and `.file(libcint_root.join("src/autocode/dkb.c"))`. Allowlist at L383 contains all 2e and 1e sigma symbols including `int1e_spsp_spinor`, `int2e_ssp1ssp2_spinor`, `int2e_vsp1_spinor`. |
| `crates/cintx-oracle/src/vendor_ffi.rs` | 7 vendor_int1e_*_spinor + 16 vendor_int2e_*_spinor shims | ✓ VERIFIED | All 7 1e shims present (1 match each). All 16 2e shims present (vendor_int2e_spsp1_spinor has 2 hits — definition + usage; others 1 each). |
| `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` | GREEN 1e parity test with sigma rank measurement, test_no_silent_skip, kappa fixture | ✓ VERIFIED | File exists. ATOL count=4, no_silent_skip count=2, kappa_spinor fixture count=5. `test_sigma_rank_measured` at L342 asserts `written == 3*rank1_len`. Orchestrator-confirmed 10/10. |
| `crates/cintx-oracle/tests/si_2e_transform_parity.rs` | D-03 BLOCKING transform micro-test GREEN on int2e_spsp1_spinor | ✓ VERIFIED | File exists. Contains ATOL, test_no_silent_skip, build_kappa_spinor_2e_fixture, vendor_int2e_spsp1_spinor. Orchestrator-confirmed 4/4. |
| `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` | GREEN 2e parity test (all 16 families), test_no_silent_skip, kappa 2e fixture | ✓ VERIFIED | File exists. ATOL count=3, no_silent_skip count=2, kappa_spinor fixture count=5. Orchestrator-confirmed 18/18. |
| `crates/cintx-oracle/src/fixtures.rs` | `build_kappa_spinor_2e_fixture` (4 shells, non-square, GT/LT kappa mix, nctr>1) | ✓ VERIFIED | `fn build_kappa_spinor_2e_fixture` at L428. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `sigma_1e.rs::launch_int1e_sigma_family_spinor_pair` | `cart_to_spinor_si_2di / si_2d / sf_2d` | per-family transform selection via `family_transform(op)` | ✓ WIRED | sigma_1e.rs:622-624: spsp→Sf, sr/sigma→SiI (imaginary-ket), others→Si. Transform functions called in the fold loop. |
| `sigma_1e_nuc.rs::run_sigma_nuc_on_backend` | `MAX_DEVICE_NROOTS` guard | fail-closed nroots check before launch | ✓ WIRED | L493-498: if nroots > 5, return UnsupportedApi. No `.clamp()`. |
| `two_electron.rs::launch_rel2e_sigma_spinor_quartet` | `cart_to_spinor_si_2e1/2e1i/2e2/2e2i/sf_2e2` | `rel2e_family_dispatch` returning `(Rel2eGout, E1Transform, E2Transform)` | ✓ WIRED | two_electron.rs:3152-3202 dispatches e1→Si/SiI, e2→Sf/Si/SiI. All 5 transform fns imported at L16-17. |
| `build.rs` | `gaunt1.c + dkb.c` | `.file()` chain entries | ✓ WIRED | L231-238: comment + `.file(libcint_root.join("src/autocode/gaunt1.c"))` + `.file(libcint_root.join("src/autocode/dkb.c"))`. |
| `compiled_manifest.lock.json` | `api_manifest.rs` | lock auto-sync via build.rs; component_rank "3" on sigma, "1" on all others | ✓ WIRED | api_manifest.rs confirms `component_rank: "3", oracle_covered: true` for int1e_sigma_spinor. Manifest-audit exits 0 with no missing/extra entries. |
| `rel2e_family_dispatch` | `Spsp1spsp2 / Srsr1srsr2` headroom | `headroom()` returning `(1,1,1,1)` for 2-sided families | ✓ WIRED | two_electron.rs:2979: `Spsp1spsp2 | Srsr1srsr2 => (1, 1, 1, 1)`. The {1,1,1,1} finding is committed. |
| `apply_2d_spinor_zi` | verbatim Pauli σ·n expansion | v11R=g1R-gzI assignments (cart2sph.c:4118-4186) | ✓ WIRED | c2spinor.rs:1755-1762 doc comments list all 8 vi*R/vi*I assignments matching libcint verbatim. Used by `cart_to_spinor_si_2e2` (c2spinor.rs:1563) and `cart_to_spinor_si_2e2i` (c2spinor.rs:1579). |

---

### Data-Flow Trace (Level 4)

Not applicable — this phase produces a numerical integral library (no web/UI components rendering dynamic data). Key data flows are the cart gout blocks through the transform pipeline; these are verified by the vendor byte-identity oracle tests.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace builds clean (--features cpu) | `cargo build --workspace --features cpu` | Finished in 5.48s, 0 errors | ✓ PASS |
| cintx-cubecl lib tests 310/310 | `cargo test -p cintx-cubecl --features cpu --lib` | 310 passed, 0 failed | ✓ PASS |
| cintx-ops resolver tests 13/13 | `cargo test -p cintx-ops` | 13 passed, 0 failed | ✓ PASS |
| Manifest-audit green | `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit` | status: ok, uncovered_stable_entries: [] | ✓ PASS |
| Vendor parity suites (double gate) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test rel_{1e,2e}_sigma_parity` | 10/10 + 18/18 per orchestrator | ? SKIP (needs vendor build — see Human Verification) |
| sigma rank measurement | `test_sigma_rank_measured` in rel_1e_sigma_parity.rs asserts `written == 3*rank1_len` | Code asserts rank 3; manifest carries "3"; confirmed by orchestrator | ✓ PASS (code-verified) |

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| REL-01 | 29-01, 29-02 | `int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp` match vendored libcint at atol=1e-12 (spinor) | ✓ SATISFIED | oracle_covered=true; sigma_1e.rs dispatch correct (spsp→sf_2d, spnucsp/sprinvsp→si_2d); vendor parity 10/10 confirmed by orchestrator |
| REL-02 | 29-01, 29-02 | `int1e_srsr`, `int1e_sr`, `int1e_srnucsr`, `int1e_sigma`, `int1e_sp` match at atol=1e-12 (spinor) | ✓ SATISFIED | oracle_covered=true; sigma_1e.rs dispatch correct (sr/sigma→si_2di, srsr→si_2d); int1e_sigma rank=3 empirically confirmed; vendor parity 10/10 confirmed |
| REL-03 | 29-03, 29-04, 29-05, 29-06 | `int2e_spsp1`, `int2e_srsr1`, `int2e_spsp1spsp2`, `int2e_srsr1srsr2` match at atol=1e-12 (spinor) | ✓ SATISFIED | oracle_covered=true; headroom {1,1,0,0}/{1,1,1,1} committed; D-03 micro-test 4/4 green; vendor parity 18/18 confirmed |
| REL-04 | 29-05, 29-06 | `int2e_ssp1ssp2`, `int2e_sps1sps2`, `int2e_vsp1*`, `int2e_spv1*` match at atol=1e-12 (spinor) | ✓ SATISFIED | oracle_covered=true; gaunt1.c+dkb.c wired; ssp/sps→si_2e1i+si_2e2i dispatch committed; vendor parity 18/18 confirmed |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `c2spinor.rs` | 250, 277 | `gt_coeff_rows`/`lt_coeff_rows` return `(vec![], vec![])` for l>4 instead of a typed error, while `bra_coeff_refs` panics on the same input — inconsistent l>4 failure modes | ⚠️ Warning (WR-01) | l>4 spinor shells are exotic and constructible; the mismatch means an OOB-panic rather than a clean typed error for l>4 inputs. Advisory only — does not affect any v1.4 family (all use l≤4). |
| `two_electron.rs` | ~3184 | Per-iteration `vec![0.0_f64; opij_len]` inside the 2-sided e2 path defeats the pre-allocated `opij_buf` (performance/clarity smell, not a correctness defect) | ⚠️ Warning (WR-02) | Advisory — performance out of v1 scope; no correctness impact; misleads future readers. |
| `sigma_1e.rs` | 63 | `#[allow(dead_code)]` on `family_id`, which IS called by `build_sigma_cart` at L793 — unnecessary suppressor | ⚠️ Warning (WR-03) | Advisory — misleads readers. No correctness impact. |
| `rel_1e_sigma_parity.rs` | 482 | Comment says rows read `oracle_covered=false` but test asserts `=true`; `si_2e_transform_parity.rs:286-287` similarly stale | ⚠️ Warning (WR-04) | Advisory — stale comments only; no correctness impact. |

No blocker anti-patterns found. All 4 advisory warnings are pre-existing documentation and style issues that do not affect correctness or the phase goal.

---

### Human Verification Required

#### 1. Vendor-gated parity suites (the primary oracle gate)

**Test:**
```
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test rel_1e_sigma_parity
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test rel_2e_sigma_parity
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test si_2e_transform_parity
```

**Expected:** rel_1e_sigma_parity 10/10, rel_2e_sigma_parity 18/18, si_2e_transform_parity 4/4 — all byte-identical at atol=1e-12; `test_no_silent_skip` green on all three suites confirming vendor arms ran non-skipped under the double gate.

**Why human:** Requires `CINTX_ORACLE_BUILD_VENDOR=1` which compiles libcint 6.1.3 from source. This is unavailable in the static verification environment. The orchestrator independently confirmed these results (rel_1e 10/10, rel_2e 18/18, si_2e 4/4) prior to verification.

**Note:** The orchestrator also confirmed that the 4 advisory REVIEW warnings (WR-01 through WR-04) are documented and non-blocking.

---

### Gaps Summary

No gaps found. All 8 must-have truths are VERIFIED through codebase evidence:

1. All 24 Group-4 spinor families (8 1e + 16 2e) have `oracle_covered: true` and spinor-only representation in the manifest.
2. The `int1e_sigma` component_rank=3 empirical finding is committed in both the manifest and the test assertion, correcting the 29-01 rank-1 prior.
3. The {1,1,1,1} headroom for 2-sided 2e families (`spsp1spsp2`/`srsr1srsr2`) is committed in `Rel2eGout::headroom()`.
4. The CR-01 BLOCKER from the code review is fixed: `sigma_1e_nuc.rs` has a fail-closed `UnsupportedApi` guard instead of `.clamp(1,5)`, confirmed by `grep "clamp" sigma_1e_nuc.rs` returning empty.
5. `gaunt1.c` and `dkb.c` are wired into the oracle vendor build (REL-04 enablement).
6. The 6-fn 2e si/sf transform suite including `apply_2d_spinor_zi` (verbatim Pauli σ·n expansion from libcint cart2sph.c:4118-4186) is committed in c2spinor.rs.
7. `build_kappa_spinor_2e_fixture` meets D-02 constraints (4 shells, non-square, GT/LT kappa mix, nctr>1).
8. Workspace builds clean (0 errors), cintx-cubecl lib tests 310/310, cintx-ops resolver 13/13, manifest-audit exits 0.

The sole remaining item is the vendor-gated parity run (requiring `CINTX_ORACLE_BUILD_VENDOR=1`), which the orchestrator confirms as passing. Status is `human_needed` because the verifier cannot independently re-run the vendor build.

---

_Verified: 2026-06-01T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
