---
phase: 27-spinor-derivative-transform-gap-b1
plan: 02a
type: execute
wave: 2
depends_on: [01]
files_modified:
  - crates/cintx-oracle/tests/spinor_deriv_parity.rs
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
autonomous: true
requirements: [FND-04]
must_haves:
  truths:
    - "The arity-3 spinor-derivative parity collectors size the auxiliary-k axis SPHERICALLY as nsph(lk) = (2lk+1)*nctr_k, NOT spinor (4lk+2)*nctr_k — so the p×d×s kappa=0 nctr_k=1 buffer is 360 elements (3·6·10·1·2), not 720."
    - "fixtures.rs ao_count_for_rep sizes the auxiliary-k shell of an arity-3 spinor representation as the spherical count nsph(lk), while bra i and ket j stay CINTcgto_spinor (4l+2) — eliminating the compat-dims over-sizing that produced the spike's spurious 720/BufferTooSmall{required:720}."
    - "vendor_ffi.rs doc comments and the spinor_deriv_parity.rs header no longer claim the aux-k is spinor-sized; they state the source-verified spherical aux-k rule (cint3c2e.c:631-636 is_ssc=0 branch)."
    - "No other family's dims (arity-2 1e/2c2e spinor, arity-3/4 cart/sph, ECP) changes — only the arity-3 SPINOR aux-k axis is corrected."
  artifacts:
    - path: "crates/cintx-oracle/tests/spinor_deriv_parity.rs"
      provides: "arity-3 collectors with spherical aux-k sizing + corrected header/comments + corrected SK sizing assertion"
      contains: "nsph"
    - path: "crates/cintx-oracle/src/fixtures.rs"
      provides: "ao_count_for_rep arity-3 spinor aux-k spherical fix"
      contains: "ao_count_for_rep"
    - path: "crates/cintx-oracle/src/vendor_ffi.rs"
      provides: "corrected aux-k doc comments on the 3c2e/3c1e spinor wrappers"
      contains: "vendor_int3c2e_ip1_spinor"
  key_links:
    - from: "crates/cintx-oracle/tests/spinor_deriv_parity.rs"
      to: "crates/cintx-oracle/src/fixtures.rs"
      via: "both must agree the arity-3 spinor aux-k axis is spherical nsph(lk)"
      pattern: "nsph"
    - from: "crates/cintx-oracle/src/fixtures.rs ao_count_for_rep"
      to: "libcint CINT3c2e_spinor_drv (cint3c2e.c:631-636)"
      via: "is_ssc=0 branch counts[2] = (k_l*2+1)*x_ctr[2]"
      pattern: "CINTcgto_spheric"
---

<objective>
Reconcile the committed Plan-01 scaffolding to the SOURCE-VERIFIED aux-k contract: the auxiliary-k axis of arity-3 spinor derivative families (int3c2e_ip1/ip2_spinor, int3c1e_ip1/iprinv_spinor) is SPHERICAL `nsph(lk) = (2lk+1)*nctr_k`, NOT spinor `(4lk+2)*nctr_k`. Only bra i and ket j are spinor-sized (`CINTcgto_spinor = 4l+2`). This corrects three committed files that encoded the disproven D2/D3 assumption: the parity test collectors (`spinor_deriv_parity.rs`), `fixtures.rs::ao_count_for_rep`, and the doc comments on the 3c2e/3c1e spinor wrappers in `vendor_ffi.rs`.

Purpose: Plan 01 was committed with the WRONG aux-k contract (the over-sized 720-element buffer was an artifact of the `ao_count_for_rep` over-sizing bug, not a real vendor requirement). The corrected `27-SPIKE-FINDINGS.md` (⚠ CORRECTION NOTICE) is authoritative: `CINT3c2e_spinor_drv` (cint3c2e.c:631-636) uses the `is_ssc=0` branch `counts[2] = (k_l*2+1)*x_ctr[2]` for aux-k. The arity-3 parity tests in Plan 04 compare cintx against a vendor buffer sized by these collectors — if the collectors stay spinor-sized (720) but cintx emits the correct spherical aux-k (360), the lengths mismatch and Plan 04 can never go green. This plan MUST land BEFORE Plans 03/04 run their parity tests (it is wave 2, parallel with Plan 02's wrapper work — disjoint files).

Output: corrected arity-3 aux-k sizing in the test collectors + `ao_count_for_rep` + corrected doc comments; a runnable assertion that the corrected p×d×s spinor-derivative buffer is 360 elements, not 720.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/27-spinor-derivative-transform-gap-b1/27-CONTEXT.md
@.planning/phases/27-spinor-derivative-transform-gap-b1/27-RESEARCH.md
@.planning/phases/27-spinor-derivative-transform-gap-b1/27-PATTERNS.md
@.planning/phases/27-spinor-derivative-transform-gap-b1/27-SPIKE-FINDINGS.md

<interfaces>
SOURCE-VERIFIED aux-k rule (27-SPIKE-FINDINGS.md ⚠ CORRECTION NOTICE — AUTHORITATIVE):
  CINT3c2e_spinor_drv (cint3c2e.c:631-636):
    counts[0] = CINTcgto_spinor(shls[0]);          // bra i: SPINOR (4l+2)
    counts[1] = CINTcgto_spinor(shls[1]);          // ket j: SPINOR (4l+2)
    if (is_ssc) counts[2] = nfk * x_ctr[2];        // (not used by ip1/ip2 — is_ssc=0)
    else        counts[2] = (k_l*2+1) * x_ctr[2];  // aux k: SPHERICAL nsph(lk) = (2lk+1)*nctr_k
  int3c2e_ip1_spinor / int3c2e_ip2_spinor (autocode/int3c2e.c:94/175) call with is_ssc=0.
  int3c1e_spinor sizes aux-k spherically the same way.
  THE ONLY AXIS THIS PLAN CHANGES is the auxiliary-k (SK) axis: spinor (4lk+2)*nctr_k → spherical
  (2lk+1)*nctr_k. For the aux-k s-shell (lk=0, nctr_k=1): spinor would be (4*0+2)*1 = 2; spherical
  is (2*0+1)*1 = 1. So the SK axis halves from 2 to 1; bra i and ket j are untouched (still spinor).

  Canonical "360 vs 720" figure (27-SPIKE-FINDINGS CORRECTION NOTICE) — the SINGLE-contraction
  (nctr=1) per-shell-tuple product rank·ni_sp·nj_sp·nk·2 with i,j single-contraction spinor lengths
  (p→6, d→10) and the aux-k correction (1 vs 2):
       CORRECT spherical:  3 · 6 · 10 · 1 · 2 = 360
       over-sized spinor:  3 · 6 · 10 · 2 · 2 = 720   (the disproven artifact)
  The committed D-08 fixture has nctr_i=2 on the bra, so its collectors size i with shell_nsp_full
  (2*6 = 12); the buffer for THAT fixture is 3·12·10·{2→1}·2 = {1440 → 720} — i.e. it HALVES when
  the aux-k goes spinor(2)→spherical(1). The acceptance criterion pins the canonical 360 on the
  single-contraction shape so the correction is unambiguous, and separately asserts the fixture's
  SK count is 1 (spherical), not 2 (spinor).

spinor_deriv_parity.rs (committed Plan 01) sites to FIX:
  L33-34   header doc claiming aux-k = CINTcgto_spinor(k)=4l+2 — REWRITE to spherical nsph(lk).
  L54      const SK: usize = 2;  // shell index (NOT a sizing constant) — verify it is the
           SHELL INDEX (aux-k is shell 2), not a count. (It is the shell index; leave the
           value, but ensure no code treats SK as a length.)
  L68-71   fn shell_nsp_full — KEEP for bra i/ket j; ADD a spherical helper for aux-k.
  L136     doc on collect_cintx_3c claiming aux-k SPINOR-sized — REWRITE to spherical.
  L138-150 collect_cintx_3c: nk = shell_nsp_full(bas, SK) — CHANGE to spherical nsph helper.
  L172-187 collect_vendor_3c: nk = shell_nsp_full(bas, SK) — CHANGE to spherical.
  L443     assert_eq!(shell_nsp_full(&bas, SK), 2) — CHANGE to assert the SPHERICAL aux-k = 1.

fixtures.rs (committed Plan 01) site to FIX:
  L864-874 ao_count_for_rep: Representation::Spinor arm applies CINTcgto_spinor to ALL shells.
           For an arity-3 spinor representation the aux-k shell (the LAST/k shell of a triple)
           must use the SPHERICAL count CINTcgto_spheric. (See dims_for_arity L876-887 — it maps
           each shell of the arity through ao_count_for_rep; the k shell is the arity-3 tail.)
  Existing spherical helper available: CINTcgto_spheric(shell, bas) (used by the Spheric arm L869).

vendor_ffi.rs (committed Plan 01) doc-comment sites to FIX (NO code/signature change — the
wrappers take a caller-sized `out`; only the doc comments lie):
  L4448-4455  block comment "the auxiliary-k axis ... is SPINOR-sized (CINTcgto_spinor) ... for
              ALL THREE shell axes" — REWRITE: aux-k is SPHERICAL nsph(lk); only i,j spinor.
  L4543-4547  vendor_int3c2e_ip1_spinor doc "aux-k axis is SPINOR-sized" — REWRITE spherical.
  L4573-4576  vendor_int3c1e_ip1_spinor doc "aux-k axis SPINOR-sized" — REWRITE spherical.
  L4602-4607  vendor_int3c1e_iprinv_spinor doc "aux-k axis SPINOR-sized" — REWRITE spherical.

Helper to add in spinor_deriv_parity.rs (spherical aux-k axis length):
  fn shell_nsph_full(bas: &[i32], s: usize) -> usize { (2*shell_ang(bas,s)+1) as usize * shell_nctr(bas,s) }
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Correct fixtures.rs ao_count_for_rep aux-k sizing + vendor_ffi.rs doc comments</name>
  <files>crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/vendor_ffi.rs</files>
  <read_first>
    - crates/cintx-oracle/src/fixtures.rs (ao_count_for_rep L864-874; dims_for_arity L876-887; CINTcgto_spheric / CINTcgto_spinor call sites)
    - crates/cintx-oracle/src/vendor_ffi.rs (the 3c2e/3c1e spinor wrapper block L4448-4631; vendor_cgto_spheric L99-101)
    - .planning/phases/27-spinor-derivative-transform-gap-b1/27-SPIKE-FINDINGS.md (⚠ CORRECTION NOTICE — the source-verified rule)
  </read_first>
  <action>
    fixtures.rs — In `ao_count_for_rep` (L864-874), the `Representation::Spinor` arm applies `CINTcgto_spinor` to every shell. For arity-3 spinor representations the aux-k shell must size SPHERICALLY. Implement the correction at the call boundary so it is unambiguous which axis is the aux-k:

    Preferred shape: change `ao_count_for_rep` to take the axis ROLE (or change `dims_for_arity` L876-887 to size the arity-3 tail shell spherically when `representation == Representation::Spinor`). Concretely, in `dims_for_arity`, when `representation == Representation::Spinor && arity == 3`, size shells `[i, j]` with `CINTcgto_spinor` (the existing `ao_count_for_rep` Spinor arm) and the LAST shell `[k]` with `CINTcgto_spheric` (the spherical count), matching libcint `CINT3c2e_spinor_drv` is_ssc=0 (`counts[2] = (k_l*2+1)*x_ctr[2]`). Do NOT change arity-2 (1e/2c2e) spinor sizing, and do NOT change Cart/Spheric arms. Add a code comment citing cint3c2e.c:631-636 and the 27-SPIKE-FINDINGS CORRECTION NOTICE.

    If `dims_for_arity` is the only consumer that maps shells positionally, prefer fixing it there (it knows the arity and the tail index = arity-1). Keep `ao_count_for_rep` itself as the per-(shell,representation) primitive; the arity-3 spinor aux-k correction is a positional decision that belongs in `dims_for_arity`.

    vendor_ffi.rs — DOC-COMMENT ONLY (no signature/code change; the wrappers take a caller-sized `out`):
    - Rewrite the block comment at L4448-4455 to state: "the auxiliary-k axis of the arity-3 families is SPHERICAL nsph(lk) = (2lk+1)*nctr_k (libcint CINT3c2e_spinor_drv is_ssc=0, cint3c2e.c:631-636); only bra i and ket j are spinor-sized (CINTcgto_spinor=4l+2). Size the out buffer with vendor_cgto_spheric for the aux-k axis."
    - Rewrite the per-wrapper docs at L4543-4547 (vendor_int3c2e_ip1_spinor), L4573-4576 (vendor_int3c1e_ip1_spinor), L4602-4607 (vendor_int3c1e_iprinv_spinor): replace "aux-k axis is SPINOR-sized (CINTcgto_spinor)" with "aux-k axis is SPHERICAL (nsph(lk)=(2lk+1)*nctr_k)". The buffer formula in the doc becomes `3 * ni_sp * nj_sp * nk_sph * 2` where `nk_sph = (2lk+1)*nctr_k`.
  </action>
  <verify>
    <automated>cargo build -p cintx-oracle --features cpu 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `dims_for_arity` (or `ao_count_for_rep`) sizes the arity-3 spinor aux-k (tail) shell with the spherical count: `grep -c 'CINTcgto_spheric' crates/cintx-oracle/src/fixtures.rs` increases by >= 1 inside the spinor arity-3 path, AND a comment citing `cint3c2e.c:631-636` is present: `grep -c '631-636\|is_ssc' crates/cintx-oracle/src/fixtures.rs` >= 1.
    - No change to arity-2 spinor or Cart/Spheric sizing: `grep -c 'CINTcgto_spinor' crates/cintx-oracle/src/fixtures.rs` still returns >= 1 (bra i / ket j path intact).
    - vendor_ffi.rs doc comments no longer claim spinor aux-k: `grep -c 'aux-k axis is SPINOR-sized\|aux-k axis SPINOR-sized\|spinor-sized via CINTcgto_spinor for ALL THREE' crates/cintx-oracle/src/vendor_ffi.rs` returns 0.
    - vendor_ffi.rs now documents spherical aux-k: `grep -c 'aux-k.*SPHERICAL\|nsph(lk)' crates/cintx-oracle/src/vendor_ffi.rs` >= 3 (one per 3c2e/3c1e_ip1/3c1e_iprinv wrapper + the block comment).
    - `cargo build -p cintx-oracle --features cpu` exits 0.
  </acceptance_criteria>
  <done>ao_count_for_rep / dims_for_arity sizes arity-3 spinor aux-k spherically (only i,j spinor); vendor_ffi.rs doc comments corrected to spherical aux-k with the cint3c2e.c:631-636 citation; crate builds.</done>
</task>

<task type="auto">
  <name>Task 2: Correct spinor_deriv_parity.rs aux-k collectors + header + SK sizing assertion (buffer = 360, not 720)</name>
  <files>crates/cintx-oracle/tests/spinor_deriv_parity.rs</files>
  <read_first>
    - crates/cintx-oracle/tests/spinor_deriv_parity.rs (header L33-34; const SK L54; shell_nsp_full L68-71; collect_cintx_3c L135-150; collect_vendor_3c L172-187; test_fixture_builds_without_vendor SK assert L443)
    - .planning/phases/27-spinor-derivative-transform-gap-b1/27-SPIKE-FINDINGS.md (⚠ CORRECTION NOTICE)
    - crates/cintx-compat/src/raw.rs (ANG_OF, NCTR_OF, BAS_SLOTS — already imported in the test)
  </read_first>
  <action>
    Add a spherical aux-k helper near `shell_nsp_full` (L68-71):
      `/// Spherical axis length for shell `s`: `(2l+1) * nctr` (libcint CINT3c2e_spinor_drv is_ssc=0).`
      `fn shell_nsph_full(bas: &[i32], s: usize) -> usize { (2 * shell_ang(bas, s) + 1) as usize * shell_nctr(bas, s) }`

    In `collect_cintx_3c` (L138-150): change `let nk = shell_nsp_full(bas, SK);` to `let nk = shell_nsph_full(bas, SK);` (the aux-k axis is spherical). Leave `ni`/`nj` on `shell_nsp_full` (bra i / ket j stay spinor). The out buffer becomes `ncomp * ni * nj * nk * 2` with the corrected (smaller) nk.

    In `collect_vendor_3c` (L172-187): identical change — `nk` uses `shell_nsph_full(bas, SK)`. The vendor and cintx buffers MUST be sized identically so `count_mismatches` length-asserts pass in Plan 04.

    Header doc (L33-34): rewrite the "Aux-k spinor sizing (27-SPIKE-FINDINGS D2/D3)" paragraph to: "Aux-k SPHERICAL sizing (27-SPIKE-FINDINGS ⚠ CORRECTION NOTICE): the arity-3 families size the auxiliary-k axis SPHERICALLY as nsph(lk) = (2lk+1)*nctr_k (libcint CINT3c2e_spinor_drv is_ssc=0, cint3c2e.c:631-636), NOT CINTcgto_spinor. Only bra i and ket j are spinor-sized (4l+2)." Update the doc on `collect_cintx_3c` (L136) the same way.

    SK sizing assertion in `test_fixture_builds_without_vendor` (L443): change `assert_eq!(shell_nsp_full(&bas, SK), 2)` to assert the SPHERICAL aux-k count: `assert_eq!(shell_nsph_full(&bas, SK), 1, "aux-k (s, nctr=1) spherical count = (2*0+1)*1 = 1");`. Keep the i/j asserts (12, 10) unchanged. Update the inline comment "p(nctr=2)=12, d=10, s=2" to "...s=1 (SPHERICAL aux-k)".

    Add a runnable buffer-size assertion that pins the corrected 360 figure (NOT 720). In `test_fixture_builds_without_vendor` (runs under `--features cpu` without vendor), append:
      `// Canonical 27-SPIKE-FINDINGS figure: single-contraction p×d×s kappa=0 rank-3 buffer.`
      `// nctr_i=1 → ni_sp=6, nj_sp=10, nk_sph=1, ncomp=3, complex → 3*6*10*1*2 = 360 (NOT 720).`
      `let nctr1_buf = 3 * spinor_len_kappa0(1) * spinor_len_kappa0(2) * 1 /*nsph s*/ * 2;`
      `assert_eq!(nctr1_buf, 360, "corrected single-contraction spinor-deriv buffer is 360, not the over-sized 720");`
      Also assert the committed nctr=2 fixture's actual arity-3 buffer halves vs the old over-size:
      `let fixture_buf = 3 * shell_nsp_full(&bas, SI) * shell_nsp_full(&bas, SJ) * shell_nsph_full(&bas, SK) * 2;`
      `assert_eq!(fixture_buf, 3 * 12 * 10 * 1 * 2, "fixture arity-3 buffer uses spherical aux-k (k=1), not spinor (k=2)");`
  </action>
  <verify>
    <automated>cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_fixture_builds_without_vendor</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'fn shell_nsph_full' crates/cintx-oracle/tests/spinor_deriv_parity.rs` returns 1.
    - The arity-3 collectors size aux-k spherically: `grep -c 'shell_nsph_full(bas, SK)' crates/cintx-oracle/tests/spinor_deriv_parity.rs` returns 2 (collect_cintx_3c + collect_vendor_3c).
    - No collector still sizes aux-k as spinor: `grep -c 'shell_nsp_full(bas, SK)' crates/cintx-oracle/tests/spinor_deriv_parity.rs` returns 0 (the SK axis no longer uses the spinor helper; SI/SJ still do).
    - Header/comments corrected: `grep -c 'SPINOR-sized.*aux\|aux-k is SPINOR\|aux-k axis with .CINTcgto_spinor' crates/cintx-oracle/tests/spinor_deriv_parity.rs` returns 0; `grep -c 'SPHERICAL\|nsph(lk)' crates/cintx-oracle/tests/spinor_deriv_parity.rs` >= 2.
    - The 360 assertion exists: `grep -c '360' crates/cintx-oracle/tests/spinor_deriv_parity.rs` >= 1, and the SK spherical assertion reads 1: `grep -c 'shell_nsph_full(&bas, SK), 1' crates/cintx-oracle/tests/spinor_deriv_parity.rs` returns 1.
    - `cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_fixture_builds_without_vendor` exits 0 (the non-vendor smoke test, with the new 360/aux-k assertions, passes).
    - The vendor parity bodies still COMPILE (the corrected collectors typecheck): `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity --no-run` exits 0.
  </acceptance_criteria>
  <done>spinor_deriv_parity.rs arity-3 collectors size aux-k spherically (shell_nsph_full); header/comments corrected; SK assertion = 1; the 360-not-720 buffer assertion passes; vendor bodies still compile.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| committed Plan-01 scaffolding → Plan 04 parity comparison | The collectors size BOTH the vendor and cintx arity-3 buffers; if they disagree with what cintx actually emits (spherical aux-k), the length assertion in count_mismatches fails and Plan 04 can never reach byte-identity. This plan makes the scaffolding match the source-verified contract before Plan 04 runs. |
| libcint source (cint3c2e.c:631-636) → fixture dims | The aux-k sizing is dictated by libcint's is_ssc=0 branch, not a free choice; the fix cites the exact source lines so it is auditable. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-27-18 | Tampering | arity-3 spinor aux-k over-sizing (the disproven 720) | mitigate | dims_for_arity sizes the arity-3 spinor tail shell with CINTcgto_spheric; the collectors use shell_nsph_full for the SK axis; a runnable assertion pins the corrected 360 figure and the fixture's k=1 spherical count. |
| T-27-19 | Tampering | accidental change to a DIFFERENT axis/family while fixing aux-k | mitigate | Acceptance asserts arity-2 spinor + Cart/Spheric sizing unchanged (CINTcgto_spinor still present; SI/SJ still use shell_nsp_full); only the SK (aux-k) axis flips to spherical. |
| T-27-20 | Repudiation | stale doc comments still asserting the disproven spinor aux-k | mitigate | Acceptance greps that no "aux-k ... SPINOR-sized" doc remains in vendor_ffi.rs or the test header, and that the spherical rule + cint3c2e.c:631-636 citation are present. |
</threat_model>

<verification>
- `cargo build -p cintx-oracle --features cpu` green.
- `cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_fixture_builds_without_vendor` green (360 + spherical aux-k assertions pass).
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity --no-run` exits 0 (corrected collectors compile).
- No spinor-aux-k claim remains in vendor_ffi.rs or the test header (grep-confirmed); arity-2/Cart/Spheric sizing untouched.
</verification>

<success_criteria>
- The committed Plan-01 scaffolding (spinor_deriv_parity.rs collectors, fixtures.rs ao_count_for_rep/dims_for_arity, vendor_ffi.rs docs) sizes the arity-3 spinor aux-k SPHERICALLY (nsph(lk)), matching libcint CINT3c2e_spinor_drv is_ssc=0.
- The corrected single-contraction p×d×s buffer is 360 (3·6·10·1·2), not 720; the committed nctr=2 fixture's aux-k count is 1 (spherical), not 2 (spinor).
- Bra i / ket j stay spinor-sized (4l+2); no other family/axis changes.
</success_criteria>

<output>
After completion, create `.planning/phases/27-spinor-derivative-transform-gap-b1/27-02a-SUMMARY.md`
</output>
</content>
</invoke>
