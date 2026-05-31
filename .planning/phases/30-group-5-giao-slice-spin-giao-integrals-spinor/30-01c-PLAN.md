---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01c
type: execute
wave: 1
depends_on: [01b]
files_modified:
  - crates/cintx-cubecl/src/kernels/sigma_1e.rs
  - crates/cintx-cubecl/src/kernels/sigma_p.rs
  - crates/cintx-ops/generated/compiled_manifest.lock.json
  - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
autonomous: true
requirements: [GIAO-03]

must_haves:
  truths:
    - "int1e_cg_sa10sa01 and int1e_giao_sa10sa01 match vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture, on a NON-SQUARE block"
    - "Both sa01 families carry component_rank 9 (NOT 3 — the GIAO g-factor raises rank; truncation drops 6 of 9 tensor components) and route through c2s_si_1e (cart_to_spinor_si_2d, REAL — not the imaginary si_1ei)"
    - "The rank-9 36-component gout is produced via g1 = ∇_j(g0) + ∇_i(g0) plus the x1i-with-origin gauge fold (dri for cg, natural for giao), inside the Rys root loop (int1e_type 1, rinv)"
    - "Each of the two sa01 inline Spinor launcher arms has its own fail-closed BufferTooSmall staging guard sized full-block (ni_sp*nj_sp*2*9), plus a fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi guard (no Rys-nroots clamp)"
    - "Both sa01 vendor byte-identity tests are non-skipped under BOTH --features cpu AND CINTX_ORACLE_BUILD_VENDOR=1; manifest-audit is green"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/sigma_p.rs"
      provides: "Rys+gauge rank-9 gout variant for cg_sa10sa01/giao_sa10sa01 (g1 = ∇_j(g0)+∇_i(g0), x1i, 36-comp gout, rinv)"
      contains: "sa10sa01"
    - path: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      provides: "family_id/family_rank(→9)/family_transform(→Si real) + build_sigma_cart Rys arms for cg_sa10sa01/giao_sa10sa01 + per-arm staging + nroots guards"
      contains: "cg_sa10sa01"
    - path: "crates/cintx-ops/generated/compiled_manifest.lock.json"
      provides: "int1e_cg_sa10sa01_spinor / int1e_giao_sa10sa01_spinor flipped oracle_covered=true (rank 9, spinor-only)"
      contains: "int1e_cg_sa10sa01_spinor"
    - path: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      provides: "per-family byte-identity gate for cg_sa10sa01/giao_sa10sa01 (rank-9, all 9 components non-zero) + extended no-silent-skip"
      contains: "int1e_cg_sa10sa01_spinor"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      to: "crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs run_sigma_nuc_on_backend"
      via: "build_sigma_cart routes the rank-9 sa01 Rys arms to the Rys backend with rank 9 + gauge origin + Si (real) transform"
      pattern: "family_rank"
    - from: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      to: "cintx_ops::generated::MANIFEST_ENTRIES"
      via: "no-silent-skip asserts entry.oracle_covered AND component_rank 9 for cg_sa10sa01/giao_sa10sa01"
      pattern: "component_rank"
---

<objective>
Sub-wave 1c: implement the NEW **Rys+gauge rank-9** engine class and prove `int1e_cg_sa10sa01` and `int1e_giao_sa10sa01` at spinor byte-identity (atol=1e-12) on the combined gauge∧kappa fixture.

This is the highest-rank engine in Wave 1: the `sa01` ("other-side" spin-angular) arms carry **component_rank 9** (the GIAO g-factor raises the rank; registering at rank 3 truncates 6 of 9 tensor components — RESEARCH Pitfall 3), produce a **36-component gout**, route through `c2s_si_1e` (the **REAL** si transform `cart_to_spinor_si_2d`, NOT the imaginary `si_1ei` the sp/nucsp arms use — RESEARCH Pitfall 2), and build via `g1 = ∇_j(g0) + ∇_i(g0)` (both-side nabla) plus the `x1i`-with-origin gauge fold inside the Rys loop (int1e_type 1, rinv). cg vs giao differ ONLY in the builder: `G2E_RCI` (dri = ri − common_orig) vs `G2E_R_I` (natural bra center).

Purpose: Lands the rank-9 Rys-gauge engine, gated green after 1b, before the spg-Rys sub-wave (1d). Reuses the 3b68ff1 scaffolding (manifest rows already at rank 9, vendor shims, bindgen allowlist) and the Phase-29 σ·p-nuc Rys path.
Output: the rank-9 Rys+gauge sa01 gout variant in sigma_p.rs, 2 sigma_1e.rs Rys dispatch arms (rank 9, Si real transform) with per-arm staging + fail-closed nroots guards, the per-family byte-identity gate for these 2 families, and oracle_covered=true flipped for exactly these 2 rows.
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
@.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01b-PLAN.md
@.claude/skills/spike-findings-cintx/SKILL.md

<interfaces>
<!-- Already landed by 30-00 + 3b68ff1 + 30-01a/b. REUSE — do NOT re-register. -->
- compiled_manifest.lock.json: int1e_cg_sa10sa01_spinor + int1e_giao_sa10sa01_spinor rows ALREADY present at component_rank "9" (oracle_covered=false). This sub-wave only FLIPS these two to true. Do NOT change the rank — it is already 9.
- vendor_ffi.rs: vendor_int1e_cg_sa10sa01_spinor (L4553), vendor_int1e_giao_sa10sa01_spinor (L4605) ALREADY present. Bindgen allowlist ALREADY extended. Do NOT re-add.
- sigma_p.rs: 30-00 gauge helpers (sigma_p_x1i recurrence, sigma_p_nabla_j) + 30-01a spgsp + 30-01b nucsp Rys+gauge variants present. Add the rank-9 sa01 variant alongside.
- sigma_1e.rs: family_id L64, family_rank L79 (MUST return 9 for *_sa01), TransformKind {Sf,Si,SiI} L611, family_transform L620 (sa01 → Si = c2s_si_1e REAL), build_sigma_cart L775 + Rys branch ~L817.

<!-- The Rys+gauge rank-9 sa01 engine (RESEARCH §Per-Family Map + Open Q1). Transcribe from intor3.c, do NOT pick by analogy. -->
int1e_cg_sa10sa01   | gout L998  | c2s: cart_to_spinor_si_2d (Si, REAL) | builder: G2E_RCI (dri = ri - common_orig) | reads COMMON_ORIG: YES | ng[]={2,1,0,0,2,4,0,9} | rank 9 | Rys: YES
int1e_giao_sa10sa01 | gout L1323 | c2s: cart_to_spinor_si_2d (Si, REAL) | builder: G2E_R_I (natural bra center)     | reads COMMON_ORIG: NO  | ng[]={2,1,0,0,2,4,0,9} | rank 9 | Rys: YES

ng[7] = 9 → spinor component_rank 9 (verified). The sa01 arms use the REAL si transform (c2s_si_1e / cart_to_spinor_si_2d), NOT the imaginary si_1ei: the spin operator sits on the ket-side angular operator rather than as an imaginary ∇ phase (RESEARCH §Transform rule + Pitfall 2). The full 36-line gout mix is at intor3.c:998 (cg) / :1323 (giao); transcribe verbatim (RESEARCH Open Q1) — the cg/giao gout bodies are identical; only the G2E_RCI/G2E_R_I builder differs. g1 = ∇_j(g0)+∇_i(g0) (both-side nabla) per the seed.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Rys+gauge rank-9 sa01 gout variant in sigma_p.rs + sigma_1e.rs rank-9 dispatch arms</name>
  <files>crates/cintx-cubecl/src/kernels/sigma_p.rs, crates/cintx-cubecl/src/kernels/sigma_1e.rs</files>
  <read_first>
    - crates/cintx-cubecl/src/kernels/sigma_p.rs (the gauge helpers sigma_p_x1i / sigma_p_nabla_j + the 30-00/01a/01b variants — add the rank-9 sa01 Rys+gauge variant ALONGSIDE; do NOT touch int1e_sp / the cg-overlap / spgsp / nucsp paths)
    - crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs (run_sigma_nuc_on_backend — the Rys backend; the rank-9 sa01 arms reuse it with rank=9 and the both-side-nabla gauge gout)
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs (family_id L64, family_rank L79 — RETURN 9 for *_sa01, family_transform L620 — sa01 → Si REAL, build_sigma_cart L775 + Rys branch ~L817; launch staging guard L712-719)
    - crates/cintx-cubecl/src/kernels/one_electron.rs:9340 (the fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi guard to mirror)
    - libcint-master/src/autocode/intor3.c gout bodies VERBATIM: cg_sa10sa01 CINTgout1e_int1e_cg_sa10sa01 L998 (the FULL 36-line s[0..*]→gout[n*36+k] mix, incl. the dri read) and giao_sa10sa01 L1323 (natural center, SAME gout body). Read the WHOLE 36-line gout — RESEARCH Open Q1 explicitly defers full transcription to plan authoring; copy carefully, no truncation.
    - libcint-master/src/g2e.h:93-104 (G2E_RCI dri / G2E_R_I natural — CINTx1i_2e) + g1e.h:48-62 (G1E_D_I / G1E_D_J — the ∇_i / ∇_j nabla for g1 = ∇_j(g0)+∇_i(g0))
    - .planning/phases/30-.../30-RESEARCH.md §"Per-Family Map" (sa01 rows: rank 9, Si real), §"Common Pitfalls" Pitfall 2 (sa01 = real si_1e), Pitfall 3 (rank-9, no truncation), Open Q1 (36-comp gout), Open Q2 (Rys headroom + fail-closed nroots)
    - .claude/skills/spike-findings-cintx/SKILL.md references/spinor-layout.md (rank-9 output layout — 9 component slices, no truncation; 36-gout → 9 groups × gc 4-block, k = tensor*4 + e1)
    - docs/manual/Cubecl/*.md (#[cube] constraints: no plain-fn calls, no if-expr, F::exp/F::sqrt, u32/i32 only, no continue/break, fma()=fused = host mul_add; Rys roots device-side, no clamp)
  </read_first>
  <behavior>
    - cg_sa10sa01 on the combined gauge∧kappa fixture (non-square p×d block) is byte-identical to vendor_int1e_cg_sa10sa01_spinor at atol=1e-12 across ALL 9 components (the RED gate landed in Task 2; kernel authored GREEN-first).
    - giao_sa10sa01 (same gout body, G2E_R_I natural builder) is byte-identical to vendor_int1e_giao_sa10sa01_spinor at atol=1e-12 across all 9 components.
    - All 9 rank components are non-zero in the output (guards against rank-3 truncation — Pitfall 3).
    - common_orig=[0,0,0] with the bra shell at origin collapses cg_sa10sa01 → giao_sa10sa01 (gauge term live).
  </behavior>
  <action>
    (a) sigma_p.rs — add the rank-9 Rys+gauge sa01 gout variant ALONGSIDE the existing variants (do NOT touch int1e_sp tensor_rank==1, the 30-00 cg-overlap, 30-01a spgsp, or 30-01b nucsp paths). Build the rinv G2E_* G-tensor (int1e_type 1) with `g1 = ∇_j(g0) + ∇_i(g0)` (both-side nabla via G1E_D_J + G1E_D_I) and the `x1i`-with-origin gauge fold (reuse sigma_p_x1i: f[i]=g[i+1]+origin*g[i]) with origin = dri (cg) / natural-center origin-free branch (giao), INSIDE the Rys root loop. Transcribe the FULL 36-line `gout[n*36+k]` mix VERBATIM from intor3.c:998 (cg) — the giao arm reuses the SAME gout body. The cart mix folds into a 36-component (9 tensor × 4 gc) gout — NO truncation to 12. Author #[cube]-legally; Rys roots device-side, never clamped.
    (b) sigma_1e.rs — extend the dispatch for `cg_sa10sa01` and `giao_sa10sa01`: add to `family_id`; set `family_rank` → **9** for both (NOT 3 — Pitfall 3); set `family_transform` → **Si** (c2s_si_1e, the REAL si transform via cart_to_spinor_si_2d — NOT SiI; Pitfall 2). In `build_sigma_cart`, route both through the Rys branch to `super::sigma_1e_nuc::run_sigma_nuc_on_backend` with rank 9 and the new gauge gout variant; thread `origin = plan.operator_env_params.common_orig.unwrap_or([0.0;3])` → dri for cg, origin-free [0,0,0] for giao. EACH arm MUST have (1) its own fail-closed full-block staging guard `staging_required = ni_sp*nj_sp*2*9; if staging.len() < staging_required { return Err(cintxRsError::BufferTooSmall { required: staging_required, provided: staging.len() }) }` (rank=9 — no partial guards) and (2) a fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard — NEVER clamp. spinor sizing via spinor_len (never 4l+2).
    (c) OperatorId shift: re-grep `OperatorId::new(` / `_OPERATOR_ID: u32 =`; confirm `int4c1e_cart` still resolves to `OperatorId::new(24)` (resolver.rs:556) or resolve by symbol name.
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu 2>&1 | tail -3 && cargo build -p cintx-oracle --features cpu 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'sa10sa01\|sa01' crates/cintx-cubecl/src/kernels/sigma_p.rs` is ≥ 1 (rank-9 sa01 variant present)
    - `grep -c 'cg_sa10sa01\|giao_sa10sa01' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows both operators in the dispatch; `family_rank` returns 9 for them (verified at Task-2 runtime via the manifest component_rank assertion)
    - `grep -n 'BufferTooSmall' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows a guard for each sa01 arm sized with rank 9 (`ni_sp*nj_sp*2*9` or `*rank` with rank=9); `grep -n 'if dst < staging.len()' crates/cintx-cubecl/src/kernels/sigma_1e.rs` returns nothing
    - `grep -niE 'roots?[^a-z]*\.?clamp|clamp.*nroots' crates/cintx-cubecl/src/kernels/sigma_1e.rs crates/cintx-cubecl/src/kernels/sigma_p.rs` returns 0 (no clamp); `grep -c 'UnsupportedApi' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows the nroots guard present
    - `grep -n 'OperatorId::new(24)' crates/cintx-compat/src/resolver.rs` still resolves int4c1e_cart (or by symbol name)
    - `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0
  </acceptance_criteria>
  <done>The rank-9 Rys+gauge sa01 variant exists in sigma_p.rs (g1 = ∇_j(g0)+∇_i(g0) + x1i-with-origin inside the Rys loop, full 36-comp gout, no truncation); sigma_1e.rs dispatches cg_sa10sa01/giao_sa10sa01 at rank 9 through the REAL c2s_si_1e transform with per-arm fail-closed full-block (×9) staging guards and a fail-closed nroots guard (no clamp); OperatorId re-verified; both crates build.</done>
</task>

<task type="auto">
  <name>Task 2: sa01 rank-9 byte-identity gate + flip oracle_covered=true + manifest-audit</name>
  <files>crates/cintx-oracle/tests/giao_sigma_1e_parity.rs, crates/cintx-ops/generated/compiled_manifest.lock.json</files>
  <read_first>
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs (the file after 30-01b — extend with the sa01 gates; the per-family component_rank helper MUST return 9 for *_sa01)
    - crates/cintx-oracle/src/vendor_ffi.rs:4553 (vendor_int1e_cg_sa10sa01_spinor), :4605 (vendor_int1e_giao_sa10sa01_spinor) — already present (3b68ff1)
    - crates/cintx-oracle/src/fixtures.rs (build_gauge_kappa_spinor_fixture — non-square p×d, GT/LT, nctr>1)
    - xtask/src/oracle_covered_update.rs (the flip mechanism + the SC#4 skipped-fixture-flip-refusal guard)
    - .planning/phases/30-.../30-RESEARCH.md §"Validation Architecture" (double-gate + no-silent-skip), §"Common Pitfalls" Pitfall 3 (rank-9, all 9 components non-zero)
    - memory reference_oracle_vendor_parity_invocation (double-gate)
  </read_first>
  <action>
    (a) Extend giao_sigma_1e_parity.rs with per-family byte-identity gates for `int1e_cg_sa10sa01_spinor` and `int1e_giao_sa10sa01_spinor`, on a NON-SQUARE block (p×d). The per-family component_rank helper MUST return 9 for these (so the collectors size the buffer for all 9 components — a rank-3 collector would silently miss the truncation). Double-gated `#[cfg(has_vendor_libcint)] #[cfg(feature="cpu")]`, asserting `count_mismatches(&vendor, &cintx, 1e-12, 0.0) == 0`. Add an assertion that ALL 9 rank components are non-zero in both arms (guards against rank-3 truncation). Extend `test_no_silent_skip` to cover these 2 families (both arms RUN + non-zero + byte-identical AND entry.oracle_covered true AND entry.component_rank == "9" in MANIFEST_ENTRIES). Add the cg→giao differential witness at common_orig=[0,0,0].
    (b) With the gate green, flip `oracle_covered` false→true for exactly the TWO rows `int1e_cg_sa10sa01_spinor` and `int1e_giao_sa10sa01_spinor` in compiled_manifest.lock.json — SPINOR-ONLY. Do NOT change their component_rank (already 9). Leave the remaining 2 rows (spgnucsp, spgsa01) oracle_covered=false. Run `cargo run -p xtask -- manifest-audit` (green). Confirm NO capi enum variants / NO legacy cint* wrappers added.
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e -- --nocapture 2>&1 | tail -20 && cargo run -p xtask -- manifest-audit 2>&1 | tail -8</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'int1e_cg_sa10sa01_spinor\|int1e_giao_sa10sa01_spinor' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` is ≥ 2
    - `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 — the cg_sa10sa01/giao_sa10sa01 byte-identity gates pass at atol=1e-12 on a non-square block across all 9 components AND test_no_silent_skip passes (vendor arm executed, component_rank "9" asserted); the 30-01a/b gates still pass
    - WITHOUT the env var, test_no_silent_skip FAILS or is gated out (not a silent pass)
    - `python3 -c "import json; d=json.load(open('crates/cintx-ops/generated/compiled_manifest.lock.json')); e={x['id']['symbol']:x for x in (d.get('entries') or d.get('api') or []) if 'id' in x}; assert e['int1e_cg_sa10sa01_spinor']['oracle_covered'] and e['int1e_giao_sa10sa01_spinor']['oracle_covered']; assert e['int1e_cg_sa10sa01_spinor']['component_rank']=='9'; assert not e['int1e_spgsa01_spinor']['oracle_covered']; print('ok')"` prints ok (these 2 flipped at rank 9, spg* still false)
    - `cargo run -p xtask -- manifest-audit` exits 0 (green)
    - `git diff --stat crates/cintx-capi/` is empty and no new `cint1e_*` legacy wrapper symbols
  </acceptance_criteria>
  <done>int1e_cg_sa10sa01 and int1e_giao_sa10sa01 are byte-identical to vendored libcint at atol=1e-12 (spinor) on a non-square combined gauge∧kappa block across all 9 components, with the cg→giao collapse witness; exactly these 2 rank-9 rows are oracle_covered=true spinor-only; test_no_silent_skip green under both flags; manifest-audit green; no capi/legacy surface. Sub-wave 1c gated green — 30-01d may begin.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host dispatch → device Rys kernel | the gauge origin + rank-9 selection cross into the Rys loop; a wrong origin-class or a rank-3 truncation drops tensor components |
| device staging → interleaved rank-9 spinor output | the two sa01 inline launcher arms scatter 9 components into the complex buffer; an unguarded/under-sized arm overruns or drops components |
| Rys nroots envelope → device kernel | an out-of-envelope corpus shell could silently truncate if nroots is clamped instead of fail-closed |
| vendor test gate → coverage flip | flipping oracle_covered=true on a silently-skipped family over-claims coverage |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-30-01c-01 | Tampering (data integrity) | sa01 registered/built at rank 3 → drops 6 of 9 tensor components (Pitfall 3) | mitigate | family_rank returns 9; the staging guard sizes ×9; the gate's component_rank=="9" assertion + the all-9-components-non-zero assertion (Task 2) fail closed on truncation. |
| T-30-01c-02 | Tampering | sa01 routed through the imaginary si_1ei instead of the REAL si_1e → re/im lane corruption (Pitfall 2) | mitigate | family_transform set to Si (c2s_si_1e REAL) per the verified RESEARCH map; the atol=1e-12 gate catches a re/im swap. |
| T-30-01c-03 | Tampering / DoS | missing per-arm fail-closed staging guard → silent partial write / mid-scatter panic on the larger rank-9 buffer | mitigate | Both sa01 arms assert `staging.len() >= ni_sp*nj_sp*2*9` (full-block) before any write; no partial guards. Acceptance criterion greps BufferTooSmall per arm and forbids partial guards. |
| T-30-01c-04 | Tampering | Rys nroots clamped on an out-of-envelope shell → silent truncation | mitigate | Fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard; acceptance criterion greps to forbid any nroots clamp. |
| T-30-01c-05 | Spoofing/Repudiation | silent vendor-test skip masks a parity failure, then oracle_covered flipped | mitigate | test_no_silent_skip requires both arms to RUN under the double gate; SC#4 skipped-fixture-flip-refusal refuses to flip a skipped family. Acceptance criterion confirms the test FAILS (not skips) without CINTX_ORACLE_BUILD_VENDOR=1. |
| T-30-01c-06 | Tampering | square fixture block hides a KET→BRA transpose bug | mitigate | The gate drives a NON-SQUARE (p×d) block; a non-square assertion is baked into the test. |
</threat_model>

<verification>
- `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 (cg_sa10sa01/giao_sa10sa01 byte-identical at atol=1e-12 across all 9 components on a non-square block + the 30-01a/b gates + test_no_silent_skip green).
- `cargo run -p xtask -- manifest-audit` exits 0.
- Exactly int1e_cg_sa10sa01_spinor / int1e_giao_sa10sa01_spinor are newly oracle_covered=true spinor-only at component_rank "9"; the 2 remaining rows (spgnucsp, spgsa01) stay false.
- No rank truncation; no Rys-nroots clamp; no capi/legacy surface.
</verification>

<success_criteria>
int1e_cg_sa10sa01 and int1e_giao_sa10sa01 (the NEW Rys+gauge rank-9 rinv engine, g1 = ∇_j(g0)+∇_i(g0) + x1i-with-origin inside the Rys loop, 36-comp gout, component_rank 9, REAL c2s_si_1e) match vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture on a non-square block across all 9 components, with non-skipped vendor tests, per-arm fail-closed full-block (×9) staging guards, fail-closed nroots guards (no clamp), and oracle_covered=true spinor-only; manifest-audit is green. Sub-wave 1c is gated green.
</success_criteria>

<output>
After completion, create `.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01c-SUMMARY.md`
</output>
