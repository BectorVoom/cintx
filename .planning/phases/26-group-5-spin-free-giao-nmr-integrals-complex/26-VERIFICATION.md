---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
verified: 2026-05-31T00:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "The 11 spin-free 1e GIAO/CG families match at atol=1e-12 (cart + sph) via the complex path; vendor parity real-vs-real on a non-zero-gauge non-square block (SC-2 / GIAO-01)"
    status: failed
    reason: "int1e_a01gp is registered, dispatchable via eval_raw, and has a live kernel arm — but it is known-incorrect (~2x on a subset of rank-9 ket-varying components 1..8). No runtime guard returns UnsupportedApi before the wrong values are written. The parity test is #[ignore]d and oracle_covered=false. This means the public API silently emits wrong numerical output. Only 10 of 11 families are byte-identical. The phase goal explicitly names int1e_a01gp as a target family. No later roadmap phase claims to fix it."
    artifacts:
      - path: "crates/cintx-cubecl/src/kernels/one_electron.rs"
        issue: "giao_nuc_op dispatch routes 'a01gp' (line 8608) to a live kernel arm with no UnsupportedApi guard before write_giao_complex_staging is called. The kernel produces wrong values (~2x factor on components 1..8) silently."
      - path: "crates/cintx-oracle/tests/giao_1e_parity.rs"
        issue: "test_int1e_a01gp_parity is #[ignore]d at line 217 with a comment documenting a ~2x discrepancy on ket-varying components 1..8."
      - path: "crates/cintx-ops/generated/compiled_manifest.lock.json"
        issue: "int1e_a01gp_{cart,sph,spinor} rows carry oracle_covered=false — but the family is dispatchable, not disabled."
    missing:
      - "Add a runtime guard in the giao_nuc_op dispatch (or at the top of the op_kind==3 arm) that returns Err(cintxRsError::UnsupportedApi { requested: '...' }) for op_name == 'a01gp' before compute begins, so callers receive a typed failure instead of known-wrong results."
      - "Remove the #[ignore] annotation or keep it, but only after the runtime guard is in place so the API surface is fail-closed while correctness is pending."
---

# Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals (complex) Verification Report

**Phase Goal:** The spin-free 1e and 2e GIAO/CG families (int1e_giao_*, int1e_cg_*, int1e_govlp/gnuc/gkin, int1e_ig*, int1e_a01gp, int1e_ia01p, and the 2e int2e_g1/gg1/ig1/g1g2) — which are purely imaginary even in cart/sph — reach byte-identity through a per-family complex-interleaved output capability (FND-03), validated against the non-zero gauge-origin fixture.
**Verified:** 2026-05-31
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | FND-03: `complex_interleaved` set per-family from manifest `complex_output` flag (not the Representation string); `assert_flat_buffer_contract` fires fail-closed on the flag; staging sized 2xncomp; a purely-imaginary family (int1e_igovlp) round-trips through the safe API as Complex<f64> without silent zeroing | VERIFIED | `resolver.rs:107` has `pub complex_output: bool`; `planner.rs:311` reads `descriptor.entry.complex_output`; `compare.rs:282` gates on `fixture.complex_interleaved` (not `== "spinor"`); `giao_complex_roundtrip.rs` function `giao_igovlp_complex_view_is_purely_imaginary` asserts imag nonzero and real == 0.0 on int1e_igovlp |
| SC-2 | GIAO-01: The 11 spin-free 1e GIAO/CG families match at atol=1e-12 (cart + sph) via the complex path; vendor parity real-vs-real on a non-zero-gauge non-square block | FAILED | 10 of 11 families verified. `int1e_a01gp` is dispatchable via eval_raw (op_kind 3, registered in manifest, kernel'd, vendor-wrapped) but returns known-incorrect results with no runtime guard — the parity test is `#[ignore]`d and `oracle_covered=false` while the API surface remains callable with wrong output |
| SC-3 | GIAO-02: The 4 spin-free 2e GIAO families (int2e_g1, int2e_ig1, int2e_gg1, int2e_g1g2 per D-16) match at atol=1e-12 cart + sph | VERIFIED | `giao_2e_parity.rs` has 4 test fns; all 4 carry `oracle_covered=true` in manifest; int2e_g1g2 rank derived from intor2.c ng[TENSOR]=9; `Giao2eKind{G1,Ig1,Gg1,G1g2}` dispatch confirmed in `two_electron.rs:2222,2702-2705` |
| SC-4 | Every family gated on the non-zero gauge-origin fixture, has a vendor_* test under both flags, oracle_covered=true flipped; manifest-audit green. No capi/legacy-wrapper surface added | VERIFIED (partial) | 10/11 1e families and 4/4 2e families have oracle_covered=true; both test files use `build_h2o_sto3g_common_orig` with cross-center non-square blocks; no capi/legacy additions confirmed in summaries. int1e_a01gp deliberately carries oracle_covered=false (consistent with the parity skip) — this is subsumed by the SC-2 failure |

**Score:** 4/5 truths verified (SC-2 fails; SC-4 subsumed by SC-2)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-ops/src/resolver.rs` | ManifestEntry.complex_output: bool | VERIFIED | Line 107: `pub complex_output: bool` confirmed |
| `crates/cintx-runtime/src/planner.rs` | manifest-driven complex_multiplier | VERIFIED | Line 311: `if descriptor.entry.complex_output { 2 } else { 1 }` — rep-string keying removed |
| `crates/cintx-oracle/src/compare.rs` | generalized fail-closed flat-buffer contract | VERIFIED | Lines 282-289: `if fixture.complex_interleaved { ... }` — no `== "spinor"` string in the function |
| `crates/cintx-oracle/tests/giao_complex_roundtrip.rs` | FND-03 safe-API D-07 round-trip on int1e_igovlp | VERIFIED | File exists; `giao_igovlp_complex_view_is_purely_imaginary` fn asserts `c.im.abs() > 1e-12` and `c.re == 0.0` |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | 1e GIAO kernels with govlp and a01gp dispatch | VERIFIED (kernels exist) / FAILED (a01gp correctness) | Kernels registered; giao_ovlp and giao_nuc dispatch present; a01gp routed to op_kind 3 with no guard |
| `crates/cintx-compat/src/raw.rs` | RawApiId consts for all 11 1e and 4 2e families | VERIFIED | INT1E_GOVLP_SPH, INT1E_A01GP_SPH, INT2E_G1_SPH, INT2E_G1G2_SPH confirmed present |
| `crates/cintx-oracle/src/vendor_ffi.rs` | vendor wrappers for all 22 cart/sph 1e + 8 2e symbols | VERIFIED | vendor_int1e_govlp_sph, vendor_int1e_a01gp_cart, vendor_int2e_g1_sph, vendor_int2e_g1g2_cart confirmed |
| `crates/cintx-oracle/tests/giao_1e_parity.rs` | 11 test fns, double-gated, non-zero-gauge non-square | VERIFIED (file) / FAILED (a01gp) | File exists with 11 test fns; cross_center_non_square_shell_pair used; test_int1e_a01gp_parity is #[ignore]d at line 217 |
| `crates/cintx-oracle/tests/giao_2e_parity.rs` | 4 test fns, double-gated, non-zero-gauge non-square | VERIFIED | File exists; all 4 families parity-tested; cross-center quartet [3,2,3,2] confirmed |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | Giao2eKind dispatch, generic over F | VERIFIED | `Giao2eKind` enum at line 2222 with G1/Ig1/Gg1/G1g2 variants; dispatch at lines 2702-2705; `launch_two_electron_giao2e` function at line 2269 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `planner.rs build_output_layout` | `descriptor.entry.complex_output` | manifest read | WIRED | `descriptor.entry.complex_output` at line 311 — confirmed present, no rep-string fallback |
| `compare.rs assert_flat_buffer_contract` | `fixture.complex_interleaved` | fail-closed contract gate | WIRED | Gates on `fixture.complex_interleaved` at line 282; `== "spinor"` string absent from the function body |
| `giao_1e_parity.rs` test fns | `build_h2o_sto3g_common_orig` | non-zero gauge fixture | WIRED | Confirmed in parity file header and test bodies |
| `giao_2e_parity.rs` test fns | `vendor_ffi::vendor_int2e_*` | real-vs-real compare | WIRED | 4 vendor wrapper calls confirmed |
| `two_electron.rs` dispatch | `Giao2eKind{G1,Ig1,Gg1,G1g2}` | operator_name match | WIRED | Lines 2702-2705 confirmed |
| `one_electron.rs` giao_nuc dispatch | `op_kind 3` for a01gp | kernel arm | WIRED (wrong result, no guard) | Line 8608 maps "a01gp" => Some((3, 9)); no UnsupportedApi guard before compute |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `giao_complex_roundtrip.rs` | `complex_values()` view | `int1e_igovlp` kernel via `write_giao_complex_staging` | Yes — non-zero imag verified by test assertion | FLOWING |
| `giao_1e_parity.rs` (10 families) | imaginary half of 2x interleaved buffer | `eval_raw` + giao_ovlp/giao_nuc kernel | Yes — vendor byte-identity at atol=1e-12 | FLOWING |
| `giao_1e_parity.rs` (int1e_a01gp) | imaginary half of 2x interleaved buffer | op_kind 3 kernel arm | Yes — data flows, but values are wrong (~2x on components 1..8) | HOLLOW (wrong values, no guard) |
| `giao_2e_parity.rs` (4 families) | imaginary half of 2x interleaved buffer | `launch_two_electron_giao2e` | Yes — vendor byte-identity at atol=1e-12 | FLOWING |

### Behavioral Spot-Checks

Step 7b: SKIPPED — requires CINTX_ORACLE_BUILD_VENDOR=1 build environment not available for live test execution. The pre-existing failure baseline (4 --lib oracle harness failures) is confirmed identical at commit 3d3a59b and is not attributable to Phase 26.

The vendor-gated parity test coverage per summaries: 10/11 1e families byte-identical, 4/4 2e families byte-identical. Executors reported these under CINTX_ORACLE_BUILD_VENDOR=1 + --features cpu.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FND-03 | 26-01-PLAN.md | Complex/imaginary output capability per-family from manifest flag | SATISFIED | `complex_output` field end-to-end in resolver/planner/compare/roundtrip test; spinor rows backfilled; 5 non-backfilled spinor rows are `helper_kind=helper`, not operator rows — intentional |
| GIAO-01 | 26-02-PLAN.md | 11 spin-free 1e GIAO/CG families byte-identical to libcint 6.1.3 | BLOCKED | 10/11 byte-identical. `int1e_a01gp` is explicitly named in the GIAO-01 requirement text in REQUIREMENTS.md; it is dispatchable with known-wrong output and no runtime guard. REQUIREMENTS.md marks GIAO-01 [x] complete — this is an incorrect state given the a01gp defect. |
| GIAO-02 | 26-03-PLAN.md | 4 spin-free 2e GIAO families byte-identical to libcint 6.1.3 | SATISFIED | All 4 families (incl int2e_g1g2 per D-16) oracle_covered=true; parity test passes for cart+sph at atol=1e-12 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | 8608 | `"a01gp" => Some((3, 9))` with no UnsupportedApi guard before `write_giao_complex_staging` | BLOCKER | Callers can invoke `eval_raw(RawApiId::INT1E_A01GP_CART, ...)` and receive silently-wrong numerical output. The project's core value is "byte-identity as the primary goal" — a public dispatchable API returning known-wrong numbers with no error violates this contract. |
| `crates/cintx-oracle/tests/giao_1e_parity.rs` | 217 | `#[ignore = "26-02 deferred: a01gp rank-9 27-s table has a 2x ket-element ..."]` | BLOCKER (symptom of kernel gap) | The parity test is silenced but the call path is not fail-closed. The ignore tag hides the wrongness without preventing dispatch. |

Additional warnings from 26-REVIEW.md (non-blocking for goal determination):
- WR-01: GIAO families fail-closed under memory-limit chunking (availability regression, not data corruption)
- WR-03: `giao_2e_parity.rs` duplicates comparison helpers from `moment_common` (code quality)
- WR-04: `not0` counts the always-zero real half (contract clarity)
- WR-05: Comptime `complex_output` hint in moment/1e device path is inert dead code

### Human Verification Required

None — the gap is mechanically verifiable from the codebase: the `int1e_a01gp` kernel arm is reachable with no guard and the parity test documents that it produces wrong results.

### Gaps Summary

**Root cause:** The GIAO-01 success criterion requires ALL 11 spin-free 1e GIAO/CG families to be byte-identical to libcint 6.1.3. `int1e_a01gp` (rank-9, NABLA-RINV CROSS P) is registered and dispatchable but produces known-incorrect results (~2x on components 1..8) with no fail-closed guard. The 26-REVIEW.md flagged this as CR-01 (critical blocker).

**What is working:** FND-03 is fully implemented and wired (manifests, planner, contract, round-trip proof, spinor backfill). 10/11 1e GIAO families are byte-identical. All 4 2e GIAO families are byte-identical. The complex-interleaved output path, vendor wrappers, RawApiId consts, and parity test scaffolding for all covered families are present and correct.

**The single gap blocking goal achievement:** `int1e_a01gp` is dispatchable with silent wrong output. The fix is a one-line guard returning `UnsupportedApi` at the `"a01gp"` dispatch arm so callers get a typed failure instead of wrong numbers, matching the project's no-silent-partial-write + byte-identity-as-primary-goal contract. Once the guard is in place (and eventually the kernel corrected and `#[ignore]` removed), GIAO-01 and the phase goal are fully met.

**This gap is NOT deferred to a later phase** — `int1e_a01gp` appears only in Phase 26's own roadmap goal. No phases 27-31 mention it in their success criteria.

---

_Verified: 2026-05-31_
_Verifier: Claude (gsd-verifier)_
