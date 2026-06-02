---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01a
type: execute
wave: 1
depends_on: [00]
files_modified:
  - crates/cintx-cubecl/src/kernels/sigma_p.rs
  - crates/cintx-cubecl/src/kernels/sigma_1e.rs
  - crates/cintx-ops/generated/compiled_manifest.lock.json
  - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
autonomous: true
requirements: [GIAO-03]

must_haves:
  truths:
    - "int1e_spgsp matches vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture, on a NON-SQUARE block"
    - "int1e_cg_sa10sp and int1e_giao_sa10sp (proven in 30-00) are dispatched through sigma_1e.rs and flipped oracle_covered=true in this sub-wave's gate"
    - "int1e_spgsp carries component_rank 3 and routes through c2s_si_1ei (cart_to_spinor_si_2di, imaginary)"
    - "The int1e_spgsp inline Spinor launcher arm has its own fail-closed BufferTooSmall staging guard sized full-block (ni_sp*nj_sp*2*rank)"
    - "The spgsp vendor byte-identity test is non-skipped under BOTH --features cpu AND CINTX_ORACLE_BUILD_VENDOR=1; manifest-audit is green"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/sigma_p.rs"
      provides: "8-G-tensor London overlap gout variant for int1e_spgsp (G1E_R0I origin=ri + rirj=ri-rj post-multiply)"
      contains: "spgsp"
    - path: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      provides: "family_id/family_rank/family_transform + build_sigma_cart arms for spgsp, cg_sa10sp, giao_sa10sp + per-arm fail-closed staging guards"
      contains: "spgsp"
    - path: "crates/cintx-ops/generated/compiled_manifest.lock.json"
      provides: "int1e_spgsp_spinor / int1e_cg_sa10sp_spinor / int1e_giao_sa10sp_spinor flipped oracle_covered=true (rank 3, spinor-only)"
      contains: "int1e_spgsp_spinor"
    - path: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      provides: "per-family byte-identity gate for spgsp/cg_sa10sp/giao_sa10sp + sub-wave-a no-silent-skip assertion"
      contains: "int1e_spgsp_spinor"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/sigma_1e.rs"
      to: "crates/cintx-cubecl/src/kernels/sigma_p.rs spgsp gout variant"
      via: "family_id #[comptime] selector -> build_sigma_cart -> run_sigma_p_*_on_backend London variant"
      pattern: "family_id"
    - from: "crates/cintx-oracle/tests/giao_sigma_1e_parity.rs"
      to: "cintx_ops::generated::MANIFEST_ENTRIES"
      via: "no-silent-skip asserts entry.oracle_covered for spgsp/cg_sa10sp/giao_sa10sp"
      pattern: "oracle_covered"
---

<objective>
Sub-wave 1a: implement the NEW **8-G-tensor London overlap** engine class and prove `int1e_spgsp` at spinor byte-identity (atol=1e-12) on the combined gauge∧kappa fixture, while also wiring and gating the two families already proven in 30-00 (`int1e_cg_sa10sp`, `int1e_giao_sa10sp`) through the `sigma_1e.rs` dispatch table.

This is the first of four engine-class sub-waves that replace the original monolithic 30-01. The seed re-plan disproved the "transcribe onto a proven fold" framing: `spgsp` is NOT the cg/giao gauge fold — it is a distinct overlap engine using `G1E_R0I` (origin = `ri`, NOT `dri`) PLUS a London `rirj = ri − rj` post-multiply in the gout, an 8-G-tensor build that collapses a 27-component cart mix into a 12-component gout.

Purpose: Lands the lowest-risk net-new overlap engine (no Rys atom-sum) first, gated green before the Rys-gauge sub-waves (b/c) and the spg-Rys sub-wave (d). Reuses the 3b68ff1 scaffolding (manifest rows, vendor shims, bindgen allowlist already in place).
Output: the spgsp London gout variant in sigma_p.rs, 3 sigma_1e.rs dispatch arms (spgsp + cg_sa10sp + giao_sa10sp) with per-arm staging guards, the per-family byte-identity gate for these 3 families, and oracle_covered=true flipped for exactly these 3 rows.
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
@.claude/skills/spike-findings-cintx/SKILL.md

<interfaces>
<!-- Already landed by 30-00 + 3b68ff1 (commit). REUSE — do NOT re-register. -->
- compiled_manifest.lock.json: all 9 spinor rows ALREADY present (oracle_covered=false; spgsp/cg_sa10sp/giao_sa10sp rank 3). This sub-wave only FLIPS spgsp/cg_sa10sp/giao_sa10sp to true.
- vendor_ffi.rs: vendor_int1e_spgsp_spinor (L4449), vendor_int1e_cg_sa10sp_spinor (L4394), vendor_int1e_giao_sa10sp_spinor (L4423) ALREADY present. Bindgen allowlist ALREADY extended. Do NOT re-add.
- sigma_p.rs: sigma_p_cg_sa10sp_kernel (L600, rank 3, runtime origin, #[comptime] variant), run_sigma_p_cg_on_backend (L811+), launch_int1e_cg_sa10sp_spinor_pair (L953, c2s_si_1ei, common_factor 0.5), launch_int1e_giao_sa10sp_spinor_pair (L1070, = cg launcher @ origin=[0,0,0]). The cg/giao sp families are DONE — this sub-wave only routes them through sigma_1e.rs dispatch + adds the spgsp variant alongside.
- sigma_1e.rs: family_id(op)->Option<u32> (L64), family_rank(op)->usize (L79), TransformKind {Sf,Si,SiI} (L611), family_transform(op) (L620), fold_group (L631), launch_int1e_sigma_family_spinor_pair + staging guard (L680, guard at L712-719), build_sigma_cart (L775).
- tests/giao_sigma_1e_parity.rs: 30-00's giao_sigma_micro gate is live here; giao_sigma_1e_full_parity_red is an #[ignore]d RED stub to extend.

<!-- The spgsp engine (RESEARCH §Per-Family Map + §Family-class structure). Transcribe from intor3.c, do NOT pick by analogy. -->
int1e_spgsp | gout L1724 | c2s transform: cart_to_spinor_si_2di (SiI, imaginary) | builder: G1E_R0I (origin=ri) + London rirj=ri-rj post-multiply in gout | reads COMMON_ORIG: NO | ng[]={2,1,0,0,3,4,1,3} | ncomp_tensor(cart)=3 | spinor component_rank=3 | Rys: NO

London block (intor3.c:1736-1741):
  rirj[0] = ri[0]-rj[0]; rirj[1]=ri[1]-rj[1]; rirj[2]=ri[2]-rj[2];   // ri-rj, NOT common_orig
  c[0]=1*rirj[0]; c[1]=1*rirj[1]; c[2]=1*rirj[2];
  // gout[n*12+0] = + c[1]*s[17] - c[2]*s[14] - c[1]*s[25] + c[2]*s[22];  ... (c[] post-multiplied per component)
The G-tensor build is the 8-G structure: D_J (ket nabla → i_l+2 headroom), R0I (x1i with origin=ri), D_I (bra nabla back-compose), folding the 27-component cart mix s[0..26] into the 12-component gout (3 tensor × 4 gc blocks).
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: spgsp 8-G-tensor London overlap gout variant in sigma_p.rs + sigma_1e.rs dispatch arms</name>
  <files>crates/cintx-cubecl/src/kernels/sigma_p.rs, crates/cintx-cubecl/src/kernels/sigma_1e.rs</files>
  <read_first>
    - crates/cintx-cubecl/src/kernels/sigma_p.rs (the 30-00 gauge variant: sigma_p_cg_sa10sp_kernel L600, sigma_p_nabla_j/sigma_p_x1i/sigma_p_x1i_of_j helpers, run_sigma_p_cg_on_backend L811, launch_int1e_cg_sa10sp_spinor_pair L953 — extend ALONGSIDE with the spgsp London variant; do NOT touch the cg path)
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs (family_id L64, family_rank L79, TransformKind L611, family_transform L620, fold_group L631, launch_int1e_sigma_family_spinor_pair + staging guard L680-719, build_sigma_cart L775)
    - libcint-master/src/autocode/intor3.c gout body VERBATIM: int1e_spgsp CINTgout1e_int1e_spgsp at L1724 — INCLUDING the London rirj block L1736-1741 and the full 12-line gout[n*12+k] mix (s[0..26] → c[] post-multiply). Read the WHOLE gout body, not a sample.
    - libcint-master/src/g1e.h:48-62 (G1E_R0I origin=ri; G1E_D_I bra nabla; G1E_D_J ket nabla) and g1e.c:429-451 (CINTx1i_1e recurrence f[i]=g[i+1]+origin*g[i])
    - .planning/phases/30-.../30-RESEARCH.md §"The Gauge-Origin Fold — Exact Structure", §"Family-class structure" (spg* row: G1E_R0I + London), §"Code Examples" (London factor for spg*)
    - .claude/skills/spike-findings-cintx/SKILL.md (per-component axis-fold offset formula; 4-gc-block component-leading KET-major packing; rank-3 12-gout → 3 groups × gc 4-block mapping k = tensor*4 + e1)
    - docs/manual/Cubecl/*.md (#[cube] constraints: no plain-fn calls, no if-expr, F::exp/F::sqrt, u32/i32 only, no continue/break, fma()=fused single-rounding = host mul_add)
  </read_first>
  <behavior>
    - spgsp on the combined gauge∧kappa fixture (non-square p×d block) is byte-identical to vendor_int1e_spgsp_spinor at atol=1e-12 (this is the RED gate landed in Task 2; the kernel here is authored GREEN-first against it, mirroring 30-00's cross-task TDD).
    - cg_sa10sp / giao_sa10sp continue to pass their 30-00 micro-test after being routed through sigma_1e.rs dispatch (no regression).
    - A square block would hide the KET-major/BRA-major transpose bug — the gate MUST use a non-square block.
  </behavior>
  <action>
    (a) sigma_p.rs — add the spgsp London gout variant ALONGSIDE the 30-00 cg_sa10sp kernel (do NOT edit the int1e_sp tensor_rank==1 path or the cg path). The spgsp engine differs from cg in THREE ways: (1) builder is `G1E_R0I` (origin = `ri`, the bra center itself — NOT `dri = ri - common_orig`; spgsp does NOT read PTR_COMMON_ORIG); (2) an 8-G-tensor build — ket nabla `G1E_D_J` (raising j → needs i_l+2 ket headroom), then `x1i`-with-origin=`ri` via the existing sigma_p_x1i recurrence (f[i]=g[i+1]+ri*g[i]), then bra nabla `G1E_D_I` back-compose; (3) a London `rirj = ri - rj` (ri MINUS rj, NOT common_orig) post-multiply in the gout: compute `c[0..2] = rirj[0..2]` and apply the per-component signed mix `gout[n*12+k]` transcribed VERBATIM from intor3.c:1736-1758 — the 27-component cart mix s[0..26] folds into the 12-component (3 tensor × 4 gc) gout. Pass `ri` and `rj` (NOT origin/common_orig) into the kernel; spgsp's origin is `ri` and its London phase is `ri-rj`, neither of which reads PTR_COMMON_ORIG. Author #[cube]-legally.
    (b) sigma_1e.rs — extend the dispatch for THREE families this sub-wave: `spgsp`, `cg_sa10sp`, `giao_sa10sp`. Add them to `family_id` (#[comptime] selector), set `family_transform` → `SiI` (c2s_si_1ei, imaginary) for all three, set `family_rank` → 3 for all three. In `build_sigma_cart`, route spgsp to the new sigma_p.rs London variant, and route cg_sa10sp/giao_sa10sp to the existing launch_int1e_cg_sa10sp_spinor_pair / launch_int1e_giao_sa10sp_spinor_pair (thread `origin = plan.operator_env_params.common_orig.unwrap_or([0.0;3])` for cg; origin-free [0,0,0] for giao; ri/rj for spgsp). EACH of the three launcher arms MUST have its own fail-closed staging guard `staging_required = ni_sp*nj_sp*2*rank; if staging.len() < staging_required { return Err(cintxRsError::BufferTooSmall { required: staging_required, provided: staging.len() }) }` — full-block, before any write, NO partial `if dst < staging.len()` guards. spinor sizing via spinor_len (never 4l+2).
    (c) OperatorId shift (RESEARCH Pitfall 6): the manifest rows are already registered (3b68ff1) so no NEW rows are added here, but re-grep `OperatorId::new(` and `_OPERATOR_ID: u32 =` and confirm `int4c1e_cart` still resolves to `OperatorId::new(24)` (resolver.rs:556) — if a prior scaffolding shift moved it, resolve by symbol name. The `OperatorId::new(0)` consts in planner.rs/builder.rs are dummy ops (safe).
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu 2>&1 | tail -3 && cargo build -p cintx-oracle --features cpu 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'spgsp' crates/cintx-cubecl/src/kernels/sigma_p.rs` is ≥ 1 (spgsp London variant present)
    - `grep -vn '^[[:space:]]*//' crates/cintx-cubecl/src/kernels/sigma_p.rs | grep -ci 'cross.product\|cross_product'` returns 0 (the London/gauge fold is the x1i recurrence, NOT a cross-product — RESEARCH Pitfall 1)
    - `grep -c 'spgsp\|cg_sa10sp\|giao_sa10sp' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows all three operators present in the dispatch
    - `grep -n 'BufferTooSmall' crates/cintx-cubecl/src/kernels/sigma_1e.rs` shows a guard for each of the three new arms (full-block sized `ni_sp*nj_sp*2*rank`); `grep -n 'if dst < staging.len()' crates/cintx-cubecl/src/kernels/sigma_1e.rs` returns nothing (no partial guards)
    - `grep -n 'OperatorId::new(24)' crates/cintx-compat/src/resolver.rs` still resolves int4c1e_cart (or the assert was changed to resolve by symbol name)
    - `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0
  </acceptance_criteria>
  <done>The spgsp 8-G London overlap variant exists in sigma_p.rs (G1E_R0I origin=ri + rirj=ri-rj post-multiply, 27→12 gout, no cross-product); sigma_1e.rs dispatches spgsp/cg_sa10sp/giao_sa10sp through the c2s_si_1ei transform at rank 3 with per-arm fail-closed full-block staging guards; OperatorId re-verified; both crates build.</done>
</task>

<task type="auto">
  <name>Task 2: spgsp/cg_sa10sp/giao_sa10sp byte-identity gate + flip oracle_covered=true + manifest-audit</name>
  <files>crates/cintx-oracle/tests/giao_sigma_1e_parity.rs, crates/cintx-ops/generated/compiled_manifest.lock.json</files>
  <read_first>
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs (the 30-00 file: giao_sigma_micro gate + giao_sigma_1e_full_parity_red #[ignore] stub — extend, do NOT recreate)
    - crates/cintx-oracle/tests/rel_1e_sigma_parity.rs:32-66,313,387-472 (the Phase-29 scaffold: FAMILIES list, per-family component_rank helper, giao_1e_byte_identity_gate! macro, test_no_silent_skip reading MANIFEST_ENTRIES, test_kappa_sizing_non_4l_plus_2)
    - crates/cintx-oracle/src/vendor_ffi.rs:4449 (vendor_int1e_spgsp_spinor — already present, 3b68ff1), :4394/:4423 (cg/giao sp shims)
    - crates/cintx-oracle/src/fixtures.rs (build_gauge_kappa_spinor_fixture — the 30-00 combined gauge∧kappa 1e fixture; non-square p×d, GT/LT, nctr>1)
    - xtask/src/oracle_covered_update.rs (the flip mechanism + the SC#4 skipped-fixture-flip-refusal guard)
    - .planning/phases/30-.../30-RESEARCH.md §"Validation Architecture" (double-gate invocation + no-silent-skip semantics), §"Registration Mechanics" step 5
    - memory reference_oracle_vendor_parity_invocation (double-gate: --features cpu AND CINTX_ORACLE_BUILD_VENDOR=1, else silent skip)
  </read_first>
  <action>
    (a) Extend giao_sigma_1e_parity.rs with per-family byte-identity gates for `int1e_spgsp_spinor`, `int1e_cg_sa10sp_spinor`, `int1e_giao_sa10sp_spinor`. Define `collect_vendor_giao_1e`/`collect_cintx_giao_1e` collectors driving build_gauge_kappa_spinor_fixture() on a NON-SQUARE block (p×d — square blocks are transpose-symmetric and hide the KET-major/BRA-major orientation bug; bake a non-square assertion into the test). Instantiate `giao_1e_byte_identity_gate!` once per family, double-gated `#[cfg(has_vendor_libcint)] #[cfg(feature="cpu")]`, asserting `count_mismatches(&vendor, &cintx, ATOL=1e-12, RTOL=0.0) == 0` AND both arms non-zero. Add a sub-wave-a `test_no_silent_skip` that, for exactly these 3 families, asserts both arms RUN + non-zero + byte-identical AND `entry.oracle_covered` is true in MANIFEST_ENTRIES (this asserts the flip in (b)). Keep the 30-00 giao_sigma_micro gate live.
    (b) With the gate green, flip `oracle_covered` false→true for exactly the THREE rows `int1e_spgsp_spinor`, `int1e_cg_sa10sp_spinor`, `int1e_giao_sa10sp_spinor` in compiled_manifest.lock.json — SPINOR-ONLY (forms:["spinor"]). Use xtask/src/oracle_covered_update.rs if it targets these symbols, else edit the lock rows directly. Leave the other 6 rows oracle_covered=false (their sub-waves b/c/d flip them). Run `cargo run -p xtask -- manifest-audit` (green; both audit sides derive from the lock, auto-sync). Confirm NO capi enum variants and NO legacy cint* wrappers were added (new-family surface policy).
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e -- --nocapture 2>&1 | tail -20 && cargo run -p xtask -- manifest-audit 2>&1 | tail -8</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'int1e_spgsp_spinor\|int1e_cg_sa10sp_spinor\|int1e_giao_sa10sp_spinor' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` is ≥ 3 (gate per family); `grep -n 'fn test_no_silent_skip' crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` returns a match
    - `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 — the spgsp/cg_sa10sp/giao_sa10sp byte-identity gates pass at atol=1e-12 on a non-square block AND test_no_silent_skip passes (no `#[ignore]`, vendor arm executed, not skipped)
    - WITHOUT the env var, `cargo test -p cintx-oracle --features cpu giao_sigma_1e` shows test_no_silent_skip FAILS or is gated out (not a silent pass) — confirms the double-gate
    - `python3 -c "import json; d=json.load(open('crates/cintx-ops/generated/compiled_manifest.lock.json')); e={x['id']['symbol']:x for x in (d.get('entries') or d.get('api') or []) if 'id' in x}; assert e['int1e_spgsp_spinor']['oracle_covered'] and e['int1e_cg_sa10sp_spinor']['oracle_covered'] and e['int1e_giao_sa10sp_spinor']['oracle_covered']; assert not e['int1e_spgnucsp_spinor']['oracle_covered']; print('ok')"` prints ok (these 3 flipped, the rest still false)
    - `cargo run -p xtask -- manifest-audit` exits 0 (green)
    - `git diff --stat crates/cintx-capi/` is empty and `git diff` shows no new `cint1e_*` legacy wrapper symbols
  </acceptance_criteria>
  <done>int1e_spgsp is byte-identical to vendored libcint at atol=1e-12 (spinor) on a non-square combined gauge∧kappa block; cg_sa10sp/giao_sa10sp are dispatched through sigma_1e.rs and gated; exactly these 3 rows are oracle_covered=true spinor-only; test_no_silent_skip green under both flags; manifest-audit green; no capi/legacy surface. Sub-wave 1a gated green — 30-01b may begin.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host dispatch → device kernel | the spgsp London phase (ri-rj) + origin=ri cross into the device; a wrong origin-class (using dri/common_orig instead of ri, or rj instead of ri-rj) corrupts output |
| device staging → interleaved spinor output | the spgsp inline launcher arm scatters into the complex buffer; an unguarded arm overruns/partial-writes |
| vendor test gate → coverage flip | flipping oracle_covered=true on a silently-skipped family over-claims coverage |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-30-01a-01 | Tampering | spgsp implemented as the cg gauge fold (reading common_orig / dri) instead of the G1E_R0I origin=ri + rirj=ri-rj London engine | mitigate | Task 1 transcribes the spgsp builder (G1E_R0I origin=ri) and London block (rirj=ri-rj, NOT common_orig) verbatim from intor3.c:1724-1758; spgsp does NOT read PTR_COMMON_ORIG. The atol=1e-12 byte-identity gate (Task 2) fails closed on a wrong origin-class. |
| T-30-01a-02 | Tampering / DoS | missing per-inline-arm fail-closed staging guard → silent partial write / mid-scatter panic (Phase-28 CR-01 repeat) | mitigate | The spgsp arm (and the cg/giao arms) assert `staging.len() >= ni_sp*nj_sp*2*rank` (full-block) before any write; no partial `if dst < staging.len()` guards. Acceptance criterion greps BufferTooSmall per arm and forbids partial guards. |
| T-30-01a-03 | Spoofing/Repudiation | silent vendor-test skip masks a parity failure, then oracle_covered flipped | mitigate | test_no_silent_skip (Task 2) requires both arms to RUN and produce nonzero output under the double gate; the SC#4 skipped-fixture-flip-refusal guard refuses to flip a skipped family. Acceptance criterion confirms the test FAILS (not skips) without CINTX_ORACLE_BUILD_VENDOR=1. |
| T-30-01a-04 | Tampering | square fixture block hides a KET→BRA transpose bug | mitigate | The gate drives a NON-SQUARE (p×d) block (per memory project_1e_gpu_port_scalar_only); a non-square assertion is baked into the test. |
| T-30-01a-05 | Tampering | wrong c2s transform (si_1e real vs si_1ei imaginary) corrupts the re/im lane split | mitigate | family_transform is set to SiI (c2s_si_1ei, imaginary) per the verified RESEARCH per-family map; the atol=1e-12 gate catches a re/im swap. |
</threat_model>

<verification>
- `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` exits 0 (spgsp/cg_sa10sp/giao_sa10sp byte-identical at atol=1e-12 on a non-square block + test_no_silent_skip green).
- `cargo run -p xtask -- manifest-audit` exits 0.
- Exactly int1e_spgsp_spinor / int1e_cg_sa10sp_spinor / int1e_giao_sa10sp_spinor are oracle_covered=true spinor-only; the other 6 GIAO×σ 1e rows remain false.
- No capi/legacy surface added.
</verification>

<success_criteria>
int1e_spgsp (the NEW 8-G-tensor London overlap engine, G1E_R0I origin=ri + rirj=ri-rj post-multiply, 27→12 gout, rank 3, c2s_si_1ei) matches vendored libcint at atol=1e-12 spinor on the combined gauge∧kappa fixture on a non-square block, with a non-skipped vendor test, a per-arm fail-closed full-block staging guard, and oracle_covered=true spinor-only; the two 30-00-proven families (cg_sa10sp, giao_sa10sp) are dispatched through sigma_1e.rs and flipped in the same gate; manifest-audit is green. Sub-wave 1a is gated green.
</success_criteria>

<output>
After completion, create `.planning/phases/30-group-5-giao-slice-spin-giao-integrals-spinor/30-01a-SUMMARY.md`
</output>
