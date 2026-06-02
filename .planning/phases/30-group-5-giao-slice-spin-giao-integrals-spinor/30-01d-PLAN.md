---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01d
type: execute
wave: 1
depends_on: [01c]
files_modified:
  - crates/cintx-cubecl/src/kernels/sigma_1e.rs
  - crates/cintx-cubecl/src/kernels/sigma_p.rs
  - crates/cintx-ops/generated/compiled_manifest.lock.json
  - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
autonomous: true
requirements: [GIAO-03]

must_haves:
  truths:
    - "int1e_spgnucsp and int1e_spgsa01 match vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture, on a NON-SQUARE block"
    - "int1e_spgnucsp carries component_rank 3, routes through c2s_si_1ei (imaginary), 12-comp gout; int1e_spgsa01 carries component_rank 9, routes through c2s_si_1e (REAL), 36-comp gout"
    - "Both spg families combine the London (G2E_R0I origin=ri + rirj=ri-rj post-multiply) with the Rys root loop (int1e_type 2 nuclear for spgnucsp, type 1 rinv for spgsa01); neither reads PTR_COMMON_ORIG"
    - "Each of the two spg inline Spinor launcher arms has its own fail-closed BufferTooSmall staging guard sized full-block, plus a fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi guard (no Rys-nroots clamp)"
    - "All 9 1e GIAO×σ families are oracle_covered=true spinor-only with a non-skipped vendor test; the full 9-family suite + manifest-audit are green (1e half of GIAO-03 closed)"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/sigma_p.rs"
      provides: "spg-Rys/London gout variants for spgnucsp (12-comp rank 3) and spgsa01 (36-comp rank 9), G2E_R0I+rirj inside the Rys loop"
      contains: "spgnucsp"
    - path: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      provides: "family_id/family_rank/family_transform + build_sigma_cart Rys arms for spgnucsp/spgsa01 + per-arm staging + nroots guards"
      contains: "spgnucsp"
    - path: "crates/cintx-ops/generated/compiled_manifest.lock.json"
      provides: "int1e_spgnucsp_spinor (rank 3) / int1e_spgsa01_spinor (rank 9) flipped oracle_covered=true; all 9 1e rows now true spinor-only"
      contains: "int1e_spgnucsp_spinor"
    - path: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      provides: "per-family byte-identity gate for spgnucsp/spgsa01 + the FULL 9-family no-silent-skip assertion"
      contains: "int1e_spgsa01_spinor"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      to: "crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs run_sigma_nuc_on_backend"
      via: "build_sigma_cart routes the spg Rys arms to the Rys backend with the London (ri / ri-rj) gout build (rank 3 nucsp / rank 9 sa01)"
      pattern: "run_sigma_nuc_on_backend"
    - from: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      to: "cintx_ops::generated::MANIFEST_ENTRIES"
      via: "the full 9-family no-silent-skip asserts entry.oracle_covered for ALL nine 1e GIAO×σ families"
      pattern: "oracle_covered"
---

<objective>
Sub-wave 1d (final Wave-1 sub-wave): implement the NEW **spg-Rys/London** engine class — the London overlap engine from 30-01a now combined with a Rys root loop — and prove `int1e_spgnucsp` and `int1e_spgsa01` at spinor byte-identity (atol=1e-12) on the combined gauge∧kappa fixture. Then run the FULL 9-family Wave-1 gate, closing the 1e half of GIAO-03.

These two families share 30-01a's London structure (`G2E_R0I` origin=`ri` + `rirj = ri − rj` post-multiply in the gout; neither reads PTR_COMMON_ORIG) but add the Rys atom-sum:
- `int1e_spgnucsp`: 12-component gout, rank 3, `c2s_si_1ei` (imaginary), int1e_type 2 (nuclear).
- `int1e_spgsa01`: 36-component gout, rank 9, `c2s_si_1e` (REAL), int1e_type 1 (rinv) — the rank-9 path from 30-01c, now with the London phase instead of the dri/natural gauge fold.

Purpose: Closes Wave 1. With all 9 1e GIAO×σ families gated green, the full 1e parity suite + manifest-audit lock the 1e half of GIAO-03 before Wave 2 (30-02) begins. Reuses the 3b68ff1 scaffolding (manifest rows already at the right ranks, vendor shims, bindgen allowlist), the 30-01a London gout structure, and the 30-01b/c Rys+gauge engines.
Output: the spg-Rys/London gout variants in sigma_p.rs, 2 sigma_1e.rs Rys dispatch arms with per-arm staging + fail-closed nroots guards, the per-family byte-identity gates, the FULL 9-family no-silent-skip gate, and oracle_covered=true flipped for the final 2 rows (all 9 now true).
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md

@.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-RESEARCH.md
@.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-PATTERNS.md
@.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-00-SUMMARY.md
@.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01a-PLAN.md
@.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01c-PLAN.md
@.claude/skills/spike-findings-cintx/SKILL.md

<interfaces>
<!-- Already landed by 30-00 + 3b68ff1 + 30-01a/b/c. REUSE — do NOT re-register. -->
- compiled_manifest.lock.json: int1e_spgnucsp_spinor (rank 3) + int1e_spgsa01_spinor (rank 9) rows ALREADY present (oracle_covered=false). This sub-wave FLIPS these last two to true → all 9 1e rows true.
- vendor_ffi.rs: vendor_int1e_spgnucsp_spinor (L4475), vendor_int1e_spgsa01_spinor (L4501) ALREADY present. Bindgen allowlist ALREADY extended. Do NOT re-add.
- sigma_p.rs: 30-01a spgsp London gout variant (G2E_R0I origin=ri + rirj=ri-rj) + 30-01b nucsp Rys+gauge + 30-01c rank-9 sa01 Rys+gauge variants present. The spg-Rys variants COMBINE the 30-01a London structure with the 30-01b/c Rys loop.
- sigma_1e.rs: family_id L64, family_rank L79 (return 9 for spgsa01, 3 for spgnucsp), TransformKind L611, family_transform L620 (spgnucsp → SiI imaginary; spgsa01 → Si REAL), build_sigma_cart L775 + Rys branch ~L817.

<!-- The spg-Rys/London engine (RESEARCH §Per-Family Map). Transcribe from intor3.c, do NOT pick by analogy. -->
int1e_spgnucsp | gout L1878 | c2s: cart_to_spinor_si_2di (SiI, imaginary) | builder: G2E_R0I (origin=ri) + rirj=ri-rj | reads COMMON_ORIG: NO | ng[]={2,1,0,0,3,4,0,3} | rank 3 | Rys: YES
int1e_spgsa01  | gout L2036 | c2s: cart_to_spinor_si_2d  (Si, REAL)      | builder: G2E_R0I (origin=ri) + rirj=ri-rj | reads COMMON_ORIG: NO | ng[]={3,1,0,0,3,4,0,9} | rank 9 | Rys: YES

Both spg families use G2E_R0I (origin=ri, the bra center — NOT dri/common_orig) PLUS the London rirj=ri-rj post-multiply (same structure as 30-01a's spgsp, intor3.c:1736-1741), now inside the Rys loop. spgnucsp 12-comp gout (rank 3) via cart_to_spinor_si_2di; spgsa01 36-comp gout (rank 9) via the REAL cart_to_spinor_si_2d. Transcribe the gout bodies verbatim (spgnucsp L1878, spgsa01 L2036 — the full 36-line mix for sa01, RESEARCH Open Q1). Neither reads PTR_COMMON_ORIG.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: spg-Rys/London gout variants (spgnucsp 12-comp / spgsa01 36-comp) in sigma_p.rs + sigma_1e.rs dispatch arms</name>
  <files>crates/cintx-cubecl/src/kernels/sigma_p.rs, crates/cintx-cubecl/src/kernels/sigma_1e.rs</files>
  <read_first>
    - crates/cintx-cubecl/src/kernels/sigma_p.rs (30-01a's spgsp London gout variant — G2E_R0I origin=ri + rirj=ri-rj post-multiply — is the London structure to REUSE; 30-01b nucsp Rys variant + 30-01c rank-9 sa01 Rys variant are the Rys structures to REUSE; add the two spg-Rys/London variants ALONGSIDE; do NOT touch int1e_sp / cg / giao / spgsp-overlap paths)
    - crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs (run_sigma_nuc_on_backend — the Rys backend; the spg Rys arms reuse it with the London gout build)
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs (family_id L64, family_rank L79 — 3 for spgnucsp / 9 for spgsa01, family_transform L620 — spgnucsp SiI / spgsa01 Si REAL, build_sigma_cart L775 + Rys branch ~L817; launch staging guard L712-719)
    - crates/cintx-cubecl/src/kernels/one_electron.rs:9340 (the fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi guard to mirror)
    - libcint-master/src/autocode/intor3.c gout bodies VERBATIM: spgnucsp CINTgout1e_int1e_spgnucsp L1878 (12-line gout + the rirj=ri-rj London block) and spgsa01 CINTgout1e_int1e_spgsa01 L2036 (the FULL 36-line gout + rirj London block — RESEARCH Open Q1, copy carefully, no truncation). Read the WHOLE gout bodies.
    - libcint-master/src/g2e.h:93-104 (G2E_R0I origin=ri — CINTx1i_2e) + g1e.h:48-62 (G2E_D_I/D_J nabla); intor3.c:1736-1741 (the rirj=ri-rj London block shape, shared with spgsp)
    - .planning/phases/30-.../30-RESEARCH.md §"Family-class structure" (spg* row: G1E/G2E_R0I + London), §"Per-Family Map" (spgnucsp rank 3 SiI / spgsa01 rank 9 Si), §"Code Examples" (London factor), Open Q1 (36-comp), Open Q2 (Rys headroom + fail-closed nroots)
    - .claude/skills/spike-findings-cintx/SKILL.md references/spinor-layout.md (rank-3 12-gout and rank-9 36-gout layouts; gc 4-block, k = tensor*4 + e1; no truncation)
    - docs/manual/Cubecl/*.md (#[cube] constraints: no plain-fn calls, no if-expr, F::exp/F::sqrt, u32/i32 only, no continue/break, fma()=fused = host mul_add; Rys roots device-side, no clamp)
  </read_first>
  <behavior>
    - spgnucsp on the combined gauge∧kappa fixture (non-square p×d block) is byte-identical to vendor_int1e_spgnucsp_spinor at atol=1e-12 (12 components, rank 3).
    - spgsa01 is byte-identical to vendor_int1e_spgsa01_spinor at atol=1e-12 across ALL 9 rank components (36-component gout, no truncation).
    - (RED gates landed in Task 2; kernels authored GREEN-first against them.)
  </behavior>
  <action>
    (a) sigma_p.rs — add the two spg-Rys/London gout variants ALONGSIDE the existing variants (do NOT touch int1e_sp tensor_rank==1, cg, giao, or the 30-01a spgsp-overlap path). Both build the G2E_* G-tensor with `G2E_R0I` (origin = `ri`, the bra center — NOT dri/common_orig; neither family reads PTR_COMMON_ORIG) via the 30-00 sigma_p_x1i recurrence (f[i]=g[i+1]+ri*g[i]), the London `rirj = ri − rj` post-multiply in the gout (same `c[0..2] = rirj` block as 30-01a's spgsp), INSIDE the Rys root loop. spgnucsp: transcribe the 12-line `gout[n*12+k]` mix VERBATIM from intor3.c:1878 (int1e_type 2 nuclear, rank 3). spgsa01: transcribe the FULL 36-line `gout[n*36+k]` mix VERBATIM from intor3.c:2036 (int1e_type 1 rinv, rank 9 — NO truncation to 12). Author #[cube]-legally; Rys roots device-side, never clamped.
    (b) sigma_1e.rs — extend the dispatch for `spgnucsp` and `spgsa01`: add to `family_id`; set `family_rank` → 3 for spgnucsp, **9** for spgsa01; set `family_transform` → `SiI` (c2s_si_1ei imaginary) for spgnucsp, **Si** (c2s_si_1e REAL) for spgsa01. In `build_sigma_cart`, route both through the Rys branch to `super::sigma_1e_nuc::run_sigma_nuc_on_backend` with the London gout build; pass `ri`/`rj` (origin=ri, London=ri-rj) — NOT common_orig. EACH arm MUST have (1) its own fail-closed full-block staging guard `staging_required = ni_sp*nj_sp*2*rank; if staging.len() < staging_required { return Err(cintxRsError::BufferTooSmall { required: staging_required, provided: staging.len() }) }` (rank=3 spgnucsp / rank=9 spgsa01 — no partial guards) and (2) a fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard — NEVER clamp. spinor sizing via spinor_len (never 4l+2).
    (c) OperatorId shift: re-grep `OperatorId::new(` / `_OPERATOR_ID: u32 =`; confirm `int4c1e_cart` still resolves to `OperatorId::new(24)` (resolver.rs:556) or resolve by symbol name.
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu 2>&1 | tail -3 && cargo build -p cintx-oracle --features cpu 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'spgnucsp\|spgsa01' crates/cintx-cubecl/src/kernels/sigma_p.rs` is ≥ 1 (both spg-Rys variants present)
    - `grep -c 'spgnucsp\|spgsa01' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows both operators in the dispatch
    - `grep -n 'BufferTooSmall' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows a guard for each spg arm (full-block `ni_sp*nj_sp*2*rank`, rank 3 / 9); `grep -n 'if dst < staging.len()' crates/cintx-cubecl/src/kernels/sigma_1e.rs` returns nothing
    - `grep -niE 'roots?[^a-z]*\.?clamp|clamp.*nroots' crates/cintx-cubecl/src/kernels/sigma_1e.rs crates/cintx-cubecl/src/kernels/sigma_p.rs` returns 0 (no clamp); `grep -c 'UnsupportedApi' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows the nroots guard present
    - `grep -n 'OperatorId::new(24)' crates/cintx-compat/src/resolver.rs` still resolves int4c1e_cart (or by symbol name)
    - `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0
  </acceptance_criteria>
  <done>The spg-Rys/London variants exist in sigma_p.rs (G2E_R0I origin=ri + rirj=ri-rj London inside the Rys loop; spgnucsp 12-comp rank 3, spgsa01 full 36-comp rank 9, no truncation); sigma_1e.rs dispatches spgnucsp (SiI, rank 3) and spgsa01 (Si REAL, rank 9) through the Rys-nuc backend with per-arm fail-closed full-block staging guards and a fail-closed nroots guard (no clamp); OperatorId re-verified; both crates build.</done>
</task>

<task type="auto">
  <name>Task 2: spg byte-identity gates + flip final 2 rows + FULL 9-family suite + manifest-audit (Wave-1 close)</name>
  <files>crates/cintx-oracle/tests/giao_sigma_1e_parity.rs, crates/cintx-ops/generated/compiled_manifest.lock.json</files>
  <read_first>
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs (the file after 30-01c — extend with the spg gates; the per-family component_rank helper returns 9 for spgsa01, 3 for spgnucsp; GIAO_1E_FAMILIES must now list ALL 9 symbols for the full no-silent-skip)
    - crates/cintx-oracle/src/vendor_ffi.rs:4475 (vendor_int1e_spgnucsp_spinor), :4501 (vendor_int1e_spgsa01_spinor) — already present (3b68ff1)
    - crates/cintx-oracle/src/fixtures.rs (build_gauge_kappa_spinor_fixture — non-square p×d, GT/LT, nctr>1)
    - xtask/src/oracle_covered_update.rs (the flip mechanism + the SC#4 skipped-fixture-flip-refusal guard)
    - .planning/phases/30-.../30-RESEARCH.md §"Validation Architecture" (double-gate + no-silent-skip), §"Registration Mechanics" step 5
    - .planning/phases/30-.../30-02-PLAN.md (Wave 2 — confirm the 1e parity test name/shape it expects to remain green; do NOT modify 30-02)
    - memory reference_oracle_vendor_parity_invocation (double-gate)
  </read_first>
  <action>
    (a) Extend giao_sigma_1e_parity.rs with per-family byte-identity gates for `int1e_spgnucsp_spinor` (rank 3) and `int1e_spgsa01_spinor` (rank 9) on a NON-SQUARE block (p×d); for spgsa01 assert all 9 components non-zero. Ensure `GIAO_1E_FAMILIES` now lists ALL NINE 1e symbols and `test_no_silent_skip` iterates the full set (each: both arms RUN + non-zero + byte-identical AND entry.oracle_covered true; for the three sa01 families also entry.component_rank == "9"). Double-gated `#[cfg(has_vendor_libcint)] #[cfg(feature="cpu")]`, `count_mismatches(..., 1e-12, 0.0) == 0`. Remove any remaining `#[ignore]` RED stub from 30-00 (giao_sigma_1e_full_parity_red) now that the full set is live.
    (b) With the gates green, flip `oracle_covered` false→true for the final TWO rows `int1e_spgnucsp_spinor` and `int1e_spgsa01_spinor` in compiled_manifest.lock.json — SPINOR-ONLY. After this, ALL NINE 1e GIAO×σ rows are oracle_covered=true. Do NOT change component_rank. Run `cargo run -p xtask -- manifest-audit` (green). Then run the FULL 9-family vendor suite green. Confirm NO capi enum variants / NO legacy cint* wrappers added.
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e -- --nocapture 2>&1 | tail -25 && cargo run -p xtask -- manifest-audit 2>&1 | tail -10</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'int1e_spgnucsp_spinor\|int1e_spgsa01_spinor' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` is ≥ 2; `grep -c 'int1e_spgsp_spinor\|int1e_spgnucsp_spinor\|int1e_spgsa01_spinor\|int1e_cg_sa10sp_spinor\|int1e_cg_sa10nucsp_spinor\|int1e_cg_sa10sa01_spinor\|int1e_giao_sa10sp_spinor\|int1e_giao_sa10nucsp_spinor\|int1e_giao_sa10sa01_spinor' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` confirms all 9 families referenced
    - `grep -c '#\[ignore\]' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` returns 0 (the Wave-1 RED stub removed)
    - `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 — all 9 1e family byte-identity gates pass at atol=1e-12 on a non-square block AND test_no_silent_skip passes for the full 9-family set (vendor arms executed, not skipped; sa01 component_rank "9" asserted)
    - WITHOUT the env var, test_no_silent_skip FAILS or is gated out (not a silent pass)
    - `python3 -c "import json; d=json.load(open('crates/cintx-ops/generated/compiled_manifest.lock.json')); e={x['id']['symbol']:x for x in (d.get('entries') or d.get('api') or []) if 'id' in x}; fams=['int1e_spgsp_spinor','int1e_spgnucsp_spinor','int1e_spgsa01_spinor','int1e_cg_sa10sp_spinor','int1e_cg_sa10nucsp_spinor','int1e_cg_sa10sa01_spinor','int1e_giao_sa10sp_spinor','int1e_giao_sa10nucsp_spinor','int1e_giao_sa10sa01_spinor']; assert all(e[f]['oracle_covered'] and e[f]['forms']==['spinor'] for f in fams); assert all(e[f]['component_rank']=='9' for f in ['int1e_spgsa01_spinor','int1e_cg_sa10sa01_spinor','int1e_giao_sa10sa01_spinor']); print('all 9 covered')"` prints "all 9 covered"
    - `cargo run -p xtask -- manifest-audit` exits 0 (green)
    - `git diff --stat crates/cintx-capi/` is empty and no new `cint1e_*` legacy wrapper symbols
  </acceptance_criteria>
  <done>int1e_spgnucsp and int1e_spgsa01 are byte-identical to vendored libcint at atol=1e-12 (spinor) on a non-square combined gauge∧kappa block; ALL NINE 1e GIAO×σ families are oracle_covered=true spinor-only (sa01 ×3 at rank 9); the full 9-family no-silent-skip suite is green under both flags; manifest-audit green; no capi/legacy surface. Wave 1 is fully gated green — the 1e half of GIAO-03 is closed; Wave 2 (30-02) may begin.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host dispatch → device Rys kernel | the London origin (ri) + phase (ri-rj) + per-family rank cross into the Rys loop; a wrong origin-class (using dri/common_orig) or rank-3 truncation of spgsa01 corrupts/drops output |
| device staging → interleaved spinor output | the two spg inline launcher arms scatter into the complex buffer (12-comp / 36-comp); an unguarded/under-sized arm overruns or drops components |
| Rys nroots envelope → device kernel | an out-of-envelope corpus shell could silently truncate if nroots is clamped instead of fail-closed |
| vendor test gate → coverage flip | flipping oracle_covered=true on a silently-skipped family over-claims the full Wave-1 coverage |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-30-01d-01 | Tampering | spg families implemented reading common_orig/dri instead of G2E_R0I origin=ri + rirj=ri-rj London | mitigate | Task 1 transcribes the G2E_R0I origin=ri builder + rirj=ri-rj London block verbatim; spg families do NOT read PTR_COMMON_ORIG. The atol=1e-12 gate fails closed on a wrong origin-class. |
| T-30-01d-02 | Tampering (data integrity) | spgsa01 registered/built at rank 3 → drops 6 of 9 tensor components (Pitfall 3) | mitigate | family_rank returns 9 for spgsa01; staging sized ×9; the gate's component_rank=="9" assertion + all-9-non-zero assertion fail closed on truncation. |
| T-30-01d-03 | Tampering | wrong c2s transform (spgnucsp SiI imaginary vs spgsa01 Si REAL) corrupts the re/im lane split (Pitfall 2) | mitigate | family_transform set per the verified map (spgnucsp SiI, spgsa01 Si REAL); the atol=1e-12 gate catches a re/im swap. |
| T-30-01d-04 | Tampering / DoS | missing per-arm fail-closed staging guard → silent partial write / mid-scatter panic | mitigate | Both spg arms assert `staging.len() >= ni_sp*nj_sp*2*rank` (full-block, rank 3/9) before any write; no partial guards. Acceptance criterion greps BufferTooSmall per arm and forbids partial guards. |
| T-30-01d-05 | Tampering | Rys nroots clamped on an out-of-envelope shell → silent truncation | mitigate | Fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard; acceptance criterion greps to forbid any nroots clamp. |
| T-30-01d-06 | Spoofing/Repudiation | silent vendor-test skip masks a parity failure across the full 9-family suite, then oracle_covered flipped | mitigate | The FULL 9-family test_no_silent_skip requires every arm to RUN under the double gate; SC#4 skipped-fixture-flip-refusal refuses to flip a skipped family. Acceptance criterion confirms the test FAILS (not skips) without CINTX_ORACLE_BUILD_VENDOR=1. |
</threat_model>

<verification>
- `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 (ALL 9 1e families byte-identical at atol=1e-12 on a non-square block + the full-set test_no_silent_skip green; no `#[ignore]`).
- `cargo run -p xtask -- manifest-audit` exits 0.
- All 9 1e GIAO×σ rows are oracle_covered=true spinor-only (the three sa01 at component_rank "9").
- No rank truncation; no Rys-nroots clamp; no capi/legacy surface.
</verification>

<success_criteria>
int1e_spgnucsp (NEW spg-Rys/London, 12-comp rank 3, c2s_si_1ei) and int1e_spgsa01 (NEW spg-Rys/London, 36-comp rank 9, REAL c2s_si_1e) — both G2E_R0I origin=ri + rirj=ri-rj London inside the Rys loop — match vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture on a non-square block, with non-skipped vendor tests, per-arm fail-closed full-block staging guards, fail-closed nroots guards (no clamp), and oracle_covered=true spinor-only. With these two flipped, ALL NINE 1e GIAO×σ families are oracle_covered=true; the full 9-family suite + manifest-audit are green. Wave 1 is fully gated green — the 1e half of GIAO-03 is closed.
</success_criteria>

<output>
After completion, create `.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01d-SUMMARY.md`
</output>
