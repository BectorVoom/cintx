---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01b
type: execute
wave: 1
depends_on: [01a]
files_modified:
  - crates/cintx-cubecl/src/kernels/sigma_1e.rs
  - crates/cintx-cubecl/src/kernels/sigma_p.rs
  - crates/cintx-ops/generated/compiled_manifest.lock.json
  - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
autonomous: true
requirements: [GIAO-03]

must_haves:
  truths:
    - "int1e_cg_sa10nucsp and int1e_giao_sa10nucsp match vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture, on a NON-SQUARE block"
    - "Both nucsp families carry component_rank 3 and route through c2s_si_1ei (cart_to_spinor_si_2di, imaginary)"
    - "The gauge x1i-with-origin fold (dri = ri - common_orig for cg; natural center for giao) is applied INSIDE the Rys root loop, 12-component gout (int1e_type 2, nuclear)"
    - "Each of the two nucsp inline Spinor launcher arms has its own fail-closed BufferTooSmall staging guard sized full-block, plus a fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi guard (no Rys-nroots clamp)"
    - "Both nucsp vendor byte-identity tests are non-skipped under BOTH --features cpu AND CINTX_ORACLE_BUILD_VENDOR=1; manifest-audit is green"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/sigma_p.rs"
      provides: "Rys+gauge nuclear gout variant for cg_sa10nucsp/giao_sa10nucsp (x1i-with-origin inside the Rys loop, 12-comp gout)"
      contains: "nucsp"
    - path: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      provides: "family_id/family_rank/family_transform + build_sigma_cart Rys-nuc arms for cg_sa10nucsp/giao_sa10nucsp + per-arm staging + nroots guards"
      contains: "cg_sa10nucsp"
    - path: "crates/cintx-ops/generated/compiled_manifest.lock.json"
      provides: "int1e_cg_sa10nucsp_spinor / int1e_giao_sa10nucsp_spinor flipped oracle_covered=true (rank 3, spinor-only)"
      contains: "int1e_cg_sa10nucsp_spinor"
    - path: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      provides: "per-family byte-identity gate for cg_sa10nucsp/giao_sa10nucsp + extended no-silent-skip"
      contains: "int1e_cg_sa10nucsp_spinor"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      to: "crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs run_sigma_nuc_on_backend"
      via: "build_sigma_cart routes *nucsp* Rys arms to the Rys-nuc backend with the gauge origin threaded"
      pattern: "run_sigma_nuc_on_backend"
    - from: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      to: "cintx_ops::generated::MANIFEST_ENTRIES"
      via: "no-silent-skip asserts entry.oracle_covered for cg_sa10nucsp/giao_sa10nucsp"
      pattern: "oracle_covered"
---

<objective>
Sub-wave 1b: implement the NEW **Rys+gauge nuclear** engine class and prove `int1e_cg_sa10nucsp` and `int1e_giao_sa10nucsp` at spinor byte-identity (atol=1e-12) on the combined gauge∧kappa fixture.

This engine differs from the 30-01a overlap engine in one structural way: the gauge `x1i`-with-origin fold (the same `f[i]=g[i+1]+origin*g[i]` recurrence proven in 30-00) is now applied INSIDE the Rys root loop, against the nuclear-attraction G2E_* G-tensor (int1e_type 2). The cg and giao gout BODIES are byte-identical per arm — only the builder differs: `G2E_RCI` (origin = `dri = ri − common_orig`) for cg vs `G2E_R_I` (natural bra center) for giao.

Purpose: Lands the first Rys-bearing gauge engine, gated green after 1a, before the rank-9 Rys sub-wave (1c) and the spg-Rys sub-wave (1d). Reuses the 3b68ff1 scaffolding (manifest rows, vendor shims, bindgen allowlist already in place) and the Phase-29 σ·p-nuc Rys path.
Output: the Rys+gauge nucsp gout variant in sigma_p.rs, 2 sigma_1e.rs Rys-nuc dispatch arms with per-arm staging + fail-closed nroots guards, the per-family byte-identity gate for these 2 families, and oracle_covered=true flipped for exactly these 2 rows.
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
@.claude/skills/spike-findings-cintx/SKILL.md

<interfaces>
<!-- Already landed by 30-00 + 3b68ff1 + 30-01a. REUSE — do NOT re-register. -->
- compiled_manifest.lock.json: int1e_cg_sa10nucsp_spinor + int1e_giao_sa10nucsp_spinor rows ALREADY present (oracle_covered=false; rank 3). This sub-wave only FLIPS these two to true.
- vendor_ffi.rs: vendor_int1e_cg_sa10nucsp_spinor (L4527), vendor_int1e_giao_sa10nucsp_spinor (L4579) ALREADY present. Bindgen allowlist ALREADY extended. Do NOT re-add.
- sigma_p.rs: 30-00 gauge helpers (sigma_p_x1i recurrence f[i]=g[i+1]+origin*g[i], sigma_p_nabla_j, sigma_p_x1i_of_j) reusable. 30-01a added the spgsp London variant alongside. Add the nucsp Rys+gauge variant alongside both.
- sigma_1e.rs: family_id L64, family_rank L79, TransformKind L611, family_transform L620, build_sigma_cart L775 — its Rys branch already routes *nucsp*/*sa01* style arms to super::sigma_1e_nuc::run_sigma_nuc_on_backend (~L817). The 30-00 fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi pattern mirrors one_electron.rs:9340.
- The Phase-29 σ·p-nuc Rys path (sigma_1e_nuc::run_sigma_nuc_on_backend) is reusable; add a headroom bump only if the corpus shell exceeds the device nroots envelope (fail-closed, never clamp — RESEARCH Open Q2).

<!-- The Rys+gauge nucsp engine (RESEARCH §Per-Family Map). Transcribe from intor3.c, do NOT pick by analogy. -->
int1e_cg_sa10nucsp   | gout L1230 | c2s: cart_to_spinor_si_2di (SiI, imaginary) | builder: G2E_RCI (dri = ri - common_orig) | reads COMMON_ORIG: YES | ng[]={1,1,0,0,2,4,0,3} | rank 3 | Rys: YES
int1e_giao_sa10nucsp | gout L1547 | c2s: cart_to_spinor_si_2di (SiI, imaginary) | builder: G2E_R_I (natural bra center)     | reads COMMON_ORIG: NO  | ng[]={1,1,0,0,2,4,0,3} | rank 3 | Rys: YES

The cg/giao nucsp gout BODIES are byte-identical (verified line-for-line in RESEARCH §Family-class structure); ONLY the builder differs (G2E_RCI dri vs G2E_R_I natural). dri = ri - env[PTR_COMMON_ORIG] (intor3.c:1239), read host-side: `let origin = plan.operator_env_params.common_orig.unwrap_or([0.0;3]); let dri = [ri[k]-origin[k]];`.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Rys+gauge nucsp gout variant in sigma_p.rs + sigma_1e.rs Rys-nuc dispatch arms</name>
  <files>crates/cintx-cubecl/src/kernels/sigma_p.rs, crates/cintx-cubecl/src/kernels/sigma_1e.rs</files>
  <read_first>
    - crates/cintx-cubecl/src/kernels/sigma_p.rs (the 30-00 gauge helpers: sigma_p_x1i recurrence f[i]=g[i+1]+origin*g[i], sigma_p_nabla_j, sigma_p_x1i_of_j, sigma_p_cg_sa10sp_kernel L600; 30-01a's spgsp variant — add the nucsp Rys+gauge variant ALONGSIDE, do NOT touch int1e_sp / cg-overlap / spgsp paths)
    - crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs (run_sigma_nuc_on_backend — the Phase-29 σ·p-nuc Rys path; the nucsp arms reuse it, threading the gauge origin into the gout build inside the Rys root loop)
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs (build_sigma_cart L775 + its Rys branch ~L817 routing *nucsp*/*sa01* to run_sigma_nuc_on_backend; family_id L64, family_rank L79, family_transform L620; launch staging guard L712-719)
    - crates/cintx-cubecl/src/kernels/one_electron.rs:9340 (the fail-closed nroots > MAX_DEVICE_NROOTS → UnsupportedApi guard to mirror)
    - libcint-master/src/autocode/intor3.c gout bodies VERBATIM: cg_sa10nucsp L1230 (incl. the dri read L1239) and giao_sa10nucsp L1547 (natural center). Read the WHOLE gout body. Confirm the cg/giao nucsp gout lines are byte-for-byte identical; only the G2E_RCI/G2E_R_I builder differs.
    - libcint-master/src/g2e.h:93-104 (G2E_RCI dri / G2E_R_I natural — CINTx1i_2e, the 2e-form x1i-with-origin used by the nuclear engine) and g1e.c:429-451 (the CINTx1i recurrence shape)
    - .planning/phases/30-.../30-RESEARCH.md §"Family-class structure" (cg/giao nucsp row), §"Per-Family Map" (rank 3, ng[]), Open Q2 (Rys headroom + fail-closed nroots guard)
    - .claude/skills/spike-findings-cintx/SKILL.md (4-gc-block component-leading KET-major packing; rank-3 12-gout → 3 groups × gc 4-block, k = tensor*4 + e1)
    - docs/manual/Cubecl/*.md (#[cube] constraints: no plain-fn calls, no if-expr, F::exp/F::sqrt, u32/i32 only, no continue/break, fma()=fused = host mul_add; Rys roots stay device-side, no clamp)
  </read_first>
  <behavior>
    - cg_sa10nucsp on the combined gauge∧kappa fixture (non-square p×d block) is byte-identical to vendor_int1e_cg_sa10nucsp_spinor at atol=1e-12 (the RED gate landed in Task 2; kernel authored GREEN-first against it).
    - giao_sa10nucsp (same gout body, G2E_R_I natural builder) is byte-identical to vendor_int1e_giao_sa10nucsp_spinor at atol=1e-12.
    - Setting common_orig=[0,0,0] with a bra shell at origin must collapse cg_sa10nucsp → giao_sa10nucsp (the gauge term is live, not zeroed) — the same differential witness as 30-00.
  </behavior>
  <action>
    (a) sigma_p.rs — add the Rys+gauge nucsp gout variant ALONGSIDE the existing variants (do NOT touch int1e_sp tensor_rank==1, the 30-00 cg-overlap kernel, or 30-01a's spgsp variant). The nucsp engine builds the nuclear-attraction G2E_* G-tensor with the gauge `x1i`-with-origin fold applied INSIDE the Rys root loop: reuse the 30-00 sigma_p_x1i recurrence (f[i]=g[i+1]+origin*g[i]) with `origin = dri` (cg) or the natural-center origin-free branch (giao). Transcribe the 12-line `gout[n*12+k]` mix VERBATIM from intor3.c:1230 (cg) — the giao arm reuses the SAME gout body (only the builder/origin differs). The cart mix folds into a 12-component (3 tensor × 4 gc) gout. Author #[cube]-legally; Rys roots stay device-side, never clamped.
    (b) sigma_1e.rs — extend the dispatch for `cg_sa10nucsp` and `giao_sa10nucsp`: add to `family_id`, set `family_transform` → `SiI` (c2s_si_1ei, imaginary) for both, `family_rank` → 3 for both. In `build_sigma_cart`, route both through the Rys branch to `super::sigma_1e_nuc::run_sigma_nuc_on_backend` with the new gauge gout variant; thread `origin = plan.operator_env_params.common_orig.unwrap_or([0.0;3])` → `dri` for cg, origin-free [0,0,0] for giao. EACH arm MUST have (1) its own fail-closed full-block staging guard `staging_required = ni_sp*nj_sp*2*rank; if staging.len() < staging_required { return Err(cintxRsError::BufferTooSmall { required: staging_required, provided: staging.len() }) }` (no partial `if dst < staging.len()` guards) and (2) a fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard mirroring one_electron.rs:9340 — NEVER clamp nroots. spinor sizing via spinor_len (never 4l+2).
    (c) OperatorId shift: re-grep `OperatorId::new(` / `_OPERATOR_ID: u32 =`; confirm `int4c1e_cart` still resolves to `OperatorId::new(24)` (resolver.rs:556) or resolve by symbol name (no NEW rows added — already registered in 3b68ff1).
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu 2>&1 | tail -3 && cargo build -p cintx-oracle --features cpu 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'nucsp' crates/cintx-cubecl/src/kernels/sigma_p.rs` is ≥ 1 (Rys+gauge nucsp variant present)
    - `grep -c 'cg_sa10nucsp\|giao_sa10nucsp' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows both operators in the dispatch
    - `grep -n 'BufferTooSmall' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows a guard for each new nucsp arm (full-block `ni_sp*nj_sp*2*rank`); `grep -n 'if dst < staging.len()' crates/cintx-cubecl/src/kernels/sigma_1e.rs` returns nothing
    - `grep -niE 'roots?[^a-z]*\.?clamp|clamp.*nroots' crates/cintx-cubecl/src/kernels/sigma_1e.rs crates/cintx-cubecl/src/kernels/sigma_p.rs` returns 0 (no Rys-nroots clamp — fail-closed instead); `grep -c 'UnsupportedApi' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows the nroots fail-closed guard present
    - `grep -n 'OperatorId::new(24)' crates/cintx-compat/src/resolver.rs` still resolves int4c1e_cart (or by symbol name)
    - `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0
  </acceptance_criteria>
  <done>The Rys+gauge nucsp variant exists in sigma_p.rs (x1i-with-origin inside the Rys loop, 12-comp gout, cg/giao share the gout body); sigma_1e.rs routes cg_sa10nucsp/giao_sa10nucsp through the Rys-nuc backend at rank 3 with c2s_si_1ei, per-arm fail-closed full-block staging guards, and a fail-closed nroots guard (no clamp); OperatorId re-verified; both crates build.</done>
</task>

<task type="auto">
  <name>Task 2: nucsp byte-identity gate + flip oracle_covered=true + manifest-audit</name>
  <files>crates/cintx-oracle/tests/giao_sigma_1e_parity.rs, crates/cintx-ops/generated/compiled_manifest.lock.json</files>
  <read_first>
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs (the file after 30-01a — extend with the nucsp gates; reuse collect_vendor_giao_1e/collect_cintx_giao_1e and the giao_1e_byte_identity_gate! macro)
    - crates/cintx-oracle/src/vendor_ffi.rs:4527 (vendor_int1e_cg_sa10nucsp_spinor), :4579 (vendor_int1e_giao_sa10nucsp_spinor) — already present (3b68ff1)
    - crates/cintx-oracle/src/fixtures.rs (build_gauge_kappa_spinor_fixture — combined gauge∧kappa 1e fixture; non-square p×d, GT/LT, nctr>1)
    - xtask/src/oracle_covered_update.rs (the flip mechanism + the SC#4 skipped-fixture-flip-refusal guard)
    - .planning/phases/30-.../30-RESEARCH.md §"Validation Architecture" (double-gate + no-silent-skip), §"Registration Mechanics" step 5
    - memory reference_oracle_vendor_parity_invocation (double-gate)
  </read_first>
  <action>
    (a) Extend giao_sigma_1e_parity.rs with per-family byte-identity gates for `int1e_cg_sa10nucsp_spinor` and `int1e_giao_sa10nucsp_spinor`, reusing the 30-01a collectors and macro on a NON-SQUARE block (p×d). Double-gated `#[cfg(has_vendor_libcint)] #[cfg(feature="cpu")]`, asserting `count_mismatches(&vendor, &cintx, 1e-12, 0.0) == 0` AND both arms non-zero. Extend `test_no_silent_skip` to cover these 2 families (both arms RUN + non-zero + byte-identical AND entry.oracle_covered true in MANIFEST_ENTRIES). Add the cg→giao differential witness: with common_orig=[0,0,0] and the bra shell at origin, cg_sa10nucsp output equals giao_sa10nucsp (proves the gauge term is live, not zeroed).
    (b) With the gate green, flip `oracle_covered` false→true for exactly the TWO rows `int1e_cg_sa10nucsp_spinor` and `int1e_giao_sa10nucsp_spinor` in compiled_manifest.lock.json — SPINOR-ONLY. Use xtask/src/oracle_covered_update.rs or edit the lock rows directly. Leave the remaining 4 rows oracle_covered=false. Run `cargo run -p xtask -- manifest-audit` (green). Confirm NO capi enum variants / NO legacy cint* wrappers added.
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e -- --nocapture 2>&1 | tail -20 && cargo run -p xtask -- manifest-audit 2>&1 | tail -8</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'int1e_cg_sa10nucsp_spinor\|int1e_giao_sa10nucsp_spinor' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` is ≥ 2
    - `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 — the cg_sa10nucsp/giao_sa10nucsp byte-identity gates pass at atol=1e-12 on a non-square block AND test_no_silent_skip passes (vendor arm executed, not skipped); the 30-01a gates (spgsp/cg_sa10sp/giao_sa10sp) still pass
    - WITHOUT the env var, test_no_silent_skip FAILS or is gated out (not a silent pass)
    - `python3 -c "import json; d=json.load(open('crates/cintx-ops/generated/compiled_manifest.lock.json')); e={x['id']['symbol']:x for x in (d.get('entries') or d.get('api') or []) if 'id' in x}; assert e['int1e_cg_sa10nucsp_spinor']['oracle_covered'] and e['int1e_giao_sa10nucsp_spinor']['oracle_covered']; assert not e['int1e_cg_sa10sa01_spinor']['oracle_covered']; print('ok')"` prints ok (these 2 flipped, sa01/spg* still false)
    - `cargo run -p xtask -- manifest-audit` exits 0 (green)
    - `git diff --stat crates/cintx-capi/` is empty and no new `cint1e_*` legacy wrapper symbols
  </acceptance_criteria>
  <done>int1e_cg_sa10nucsp and int1e_giao_sa10nucsp are byte-identical to vendored libcint at atol=1e-12 (spinor) on a non-square combined gauge∧kappa block, with the cg→giao collapse witness; exactly these 2 rows are oracle_covered=true spinor-only; test_no_silent_skip green under both flags; manifest-audit green; no capi/legacy surface. Sub-wave 1b gated green — 30-01c may begin.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host dispatch → device Rys kernel | the gauge origin (dri for cg, natural for giao) crosses into the Rys root loop; a wrong origin-class corrupts the nuclear gout |
| device staging → interleaved spinor output | the two nucsp inline launcher arms scatter into the complex buffer; an unguarded arm overruns/partial-writes |
| Rys nroots envelope → device kernel | an out-of-envelope corpus shell could silently truncate if nroots is clamped instead of fail-closed |
| vendor test gate → coverage flip | flipping oracle_covered=true on a silently-skipped family over-claims coverage |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-30-01b-01 | Tampering | gauge x1i fold applied OUTSIDE the Rys root loop (or wrong origin-class) → wrong nuclear gout | mitigate | Task 1 applies the x1i-with-origin fold INSIDE the Rys loop on the G2E_* tensor, origin=dri (cg)/natural (giao) per the verified RESEARCH map; the atol=1e-12 gate + cg→giao collapse witness (Task 2) fail closed on a mislocated/miswired fold. |
| T-30-01b-02 | Tampering / DoS | missing per-arm fail-closed staging guard → silent partial write / mid-scatter panic | mitigate | Both nucsp arms assert `staging.len() >= ni_sp*nj_sp*2*rank` (full-block) before any write; no partial guards. Acceptance criterion greps BufferTooSmall per arm and forbids partial guards. |
| T-30-01b-03 | Tampering | Rys nroots clamped on an out-of-envelope shell → silent truncation | mitigate | Fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard (mirroring one_electron.rs:9340); acceptance criterion greps to forbid any nroots clamp. |
| T-30-01b-04 | Spoofing/Repudiation | silent vendor-test skip masks a parity failure, then oracle_covered flipped | mitigate | test_no_silent_skip (Task 2) requires both arms to RUN under the double gate; SC#4 skipped-fixture-flip-refusal refuses to flip a skipped family. Acceptance criterion confirms the test FAILS (not skips) without CINTX_ORACLE_BUILD_VENDOR=1. |
| T-30-01b-05 | Tampering | square fixture block hides a KET→BRA transpose bug | mitigate | The gate drives a NON-SQUARE (p×d) block (memory project_1e_gpu_port_scalar_only); a non-square assertion is baked into the test. |
| T-30-01b-06 | Tampering | CubeCL CpuRuntime FP-env side effect (~1e-11) trips the flat atol=1e-12 gate even with a bit-identical kernel | mitigate | If a nucsp family is off by ~1e-11, suspect the launch FP-env side effect FIRST (RESEARCH Pitfall 7 / memory) before chasing kernel numerics; mitigate by keeping the band host or batching launches. |
</threat_model>

<verification>
- `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 (cg_sa10nucsp/giao_sa10nucsp byte-identical at atol=1e-12 on a non-square block + the 30-01a gates + test_no_silent_skip green).
- `cargo run -p xtask -- manifest-audit` exits 0.
- Exactly int1e_cg_sa10nucsp_spinor / int1e_giao_sa10nucsp_spinor are newly oracle_covered=true spinor-only; the 4 remaining GIAO×σ 1e rows (sa01 ×3, spgnucsp) stay false.
- No Rys-nroots clamp; no capi/legacy surface.
</verification>

<success_criteria>
int1e_cg_sa10nucsp and int1e_giao_sa10nucsp (the NEW Rys+gauge nuclear engine, x1i-with-origin inside the Rys loop, 12-comp gout, rank 3, c2s_si_1ei) match vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture on a non-square block, with non-skipped vendor tests, per-arm fail-closed full-block staging guards, fail-closed nroots guards (no clamp), and oracle_covered=true spinor-only; manifest-audit is green. Sub-wave 1b is gated green.
</success_criteria>

<output>
After completion, create `.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01b-SUMMARY.md`
</output>
