# Phase 29: Group 4 — Relativistic Spin-Operator Integrals (spinor) - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Register and prove the **full relativistic σ-operator family set** at **spinor byte-identity** (atol=1e-12) against vendored libcint 6.1.3:

- **1e (REL-01/02):** `int1e_sp`, `int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp`, `int1e_srsr`, `int1e_sr`/`srnucsr`, `int1e_sigma` — routed through the Gap-B2 `cart_to_spinor_si_2d` (bra-Pauli-mix / ket-ordinary) + the Phase-28 `kernels/sigma_p.rs` σ·p assembler. These reuse the Phase-28 foundation directly. `int1e_sp` (the Phase-28 proof vehicle, left `UnsupportedApi` by D-01) is **flipped** here.
- **2e (REL-03/04):** `int2e_spsp1`, `int2e_srsr1` (+ `spsp1spsp2`/`srsr1srsr2`), `int2e_ssp1ssp2`, `int2e_sps1sps2`, `int2e_vsp1*`, `int2e_spv1*`. These require a **brand-new 2e si-transform foundation** (`c2s_si_2e1/2e2` + imaginary `2e1i/2e2i` + the `c2s_sf_2e1/2e2` partner for the non-σ electron) that cintx does **not** have yet.

Every family is exercised on a kappa-bearing relativistic spinor fixture (N>0, non-skipped), gets a dedicated `vendor_*` test under both gate flags, and is flipped `oracle_covered=true` **spinor-only** (cart/sph σ intermediates are not over-claimed). `manifest-audit` green. **No capi enum variants, no legacy `cint*` wrappers** (new-family surface policy).

**Out of scope (deferred to later phases):** GIAO×σ slice (Phase 30), gauge/Breit–Gaunt 2e (Phase 31), the full-parity PARITY-01 gate (Phase 31).
</domain>

<decisions>
## Implementation Decisions

### 2e si-transform scope
- **D-01 (user choice — FULL 2e si suite):** Phase 29 builds the **complete 2e si-transform foundation**: `c2s_si_2e1` + `c2s_si_2e2` (real), the imaginary `c2s_si_2e1i` + `c2s_si_2e2i`, **and** the `c2s_sf_2e1`/`c2s_sf_2e2` partner transforms for the electron that does **not** carry σ. This is the honest scope required to actually close BOTH REL-03 and REL-04.
  - **Correction to a Phase-28 record:** Phase 28's deferred-ideas note (`28-CONTEXT.md` §deferred) wrongly attributed the 2e si transforms (`c2s_si_2e1/2e1i/2e2/2e2i`) to Phases 30/31. Verified against vendored `autocode/intor4.c`: REL-03/04's 2e σ families call these here. They are a **Phase-29 deliverable**. Phases 30/31 *reuse* them; they do not introduce them.
  - **Verified driver wiring** (`libcint-master/src/autocode/intor4.c`): `int2e_spsp1_spinor` → `c2s_si_2e1 + c2s_sf_2e2` (intor4.c:85); `int2e_spsp1spsp2_spinor`-style → `c2s_si_2e1 + c2s_si_2e2` (:277/:541); `int2e_srsr1_spinor` → `c2s_si_2e1 + c2s_sf_2e2` (:349); the `ssp/sps/vsp/spv` (REL-04) families use the **imaginary** `c2s_si_2e1i`/`c2s_si_2e2i` arms (:636/:899/:990/:1249/…). The `c2s_si_2e*` symbols are defined at `cart2sph.c:5592/5639/5687/5752`.

### 2e kappa-bearing fixture
- **D-02 (user choice — NEW 4-shell kappa 2e fixture):** Add a dedicated `build_kappa_spinor_2e_fixture` to `fixtures.rs`: **4 spinor shells** (2-electron config), **non-square**, **genuine kappa≠0** GT/LT mix (so the `2l`/`2l+2` sizing path is stressed, not just `4l+2`), with **≥1 shell carrying nctr>1**. This extends the Phase-28 adversarial rigor (`build_kappa_spinor_fixture`, the 1e bra/ket gate) to a 2-electron configuration. The small real heavy-atom case stays a **secondary** realism cross-check, not the primary gate.

### Spike gate (transcribe + vendor gate — NO separate spike)
- **D-03 (user choice — NO spike; transcribe-from-libcint + vendor byte-identity gate is the backstop):** Phase 29 does **not** run a separate hard-gate design spike (departing from the Phase-28 D-06 precedent). All transforms and σ gouts (`c2s_si_2e1/2e2(+i)`, σ·p-on-both-sides for `spsp`, `srsr`, the `vsp/spv` i-variant gouts) are **transcribed directly from libcint** and proven by the atol=1e-12 vendor byte-identity gate.
  - **Accepted risk (capture for the planner/executor):** the `c2s_si_2e1/2e2` layout is genuinely new and unproven in cintx; a wrong layout/stride/sign assumption surfaces only at the vendor gate, with rework risk. The core single-block si coupling math was already proven in Phase 28 — risk is concentrated in the **2e transform layout** and the **per-family gout patterns**.
  - **Structural mitigation (Claude's discretion — fold into the plan, NOT a separate spike phase):** make a **transform-level byte-identity micro-test** the **first task of the 2e wave** — verify `c2s_si_2e1/2e2(+i)` against vendored `c2s_si_2e*` (or against a thin family that exercises only the transform) **before** any 2e family is wired onto it. This buys spike-level rigor on the novel piece without a separate spike, honoring D-03.

### Plan / wave decomposition
- **D-04 (user choice — sequential de-risk, 3 waves):**
  - **Wave 1 — 1e σ:** register + prove REL-01/02 (`int1e_spsp/spnucsp/sprinvsp/srsr/sr/srnucsr/sigma`) and **flip `int1e_sp`** on the existing Phase-28 `si_2d` + σ·p foundation. Adds nothing structurally new beyond Phase 28 → lowest risk, lands first.
  - **Wave 2 — 2e foundation:** build the full 2e si-transform suite (D-01) + the `build_kappa_spinor_2e_fixture` (D-02) + the 2e transform-level byte-identity micro-test (D-03 mitigation). Gated before Wave 3.
  - **Wave 3 — 2e families:** register + prove REL-03/04 (`int2e_spsp1/srsr1/ssp1ssp2/sps1sps2/vsp1*/spv1*`) on the Wave-2 foundation.
  - Each wave is gated (vendor parity green) before the next begins.

### Claude's Discretion
- Internal module naming/factoring for the new 2e si transforms (`c2s_si_2e*`) and where they live in `c2spinor.rs`.
- Exact per-family gout component ordering for `spsp/spnucsp/sprinvsp/srsr/sigma` and the 2e families — resolve from `autocode/intor3.c`/`intor4.c` during research/planning.
- Exact molecule/element + kappa assignments for `build_kappa_spinor_2e_fixture` (subject to D-02 hard constraints: 4 shells, non-square, kappa≠0 GT/LT mix, ≥1 nctr>1) and the heavy-atom cross-check.
- Precise plan boundaries inside each wave (e.g. Wave 1 may split 1e-σ-with-p vs `int1e_sigma` pure-σ).
- Whether the 2e transform micro-test compares to vendored `c2s_si_2e*` directly or via a thin driving family.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **REL-01** (line 109), **REL-02** (110), **REL-03** (111), **REL-04** (112): the four Group-4 requirement definitions. PARITY-01 (line 128) is the Phase-31 consumer.
- `.planning/ROADMAP.md` §"Phase 29" (lines 666-677) — Goal, the 5 success criteria, dependency on Phase 28 (Gap B2) and Phase 27 (Gap B1).

### Prior phase context (decided conventions to inherit — do NOT re-decide)
- `.planning/phases/28-spin-included-c2s-si-transform-p-module-gap-b2/28-CONTEXT.md` — the Gap-B2 foundation this phase consumes: the `si_2d` host transform, the reusable σ·p `#[cube]` assembler (D-03/D-04 there), the kappa-fixture design (D-05), bra-Pauli-mix/ket-ordinary split, no-silent-skip + skipped-fixture-flip-refusal. **NOTE its deferred-ideas note on the 2e si transforms is corrected by D-01 above.**
- `.planning/phases/27-spinor-derivative-transform-gap-b1/27-CONTEXT.md` — Gap B1: KET→BRA transpose ownership inside the transform, adversarial-fixture rationale, no-silent-skip coverage assertion, new-family surface policy.
- `.planning/phases/12-real-spinor-transform-c2spinor-replacement/12-CONTEXT.md` — CG coefficient source, the four `sf/iket_sf/si/iket_si` code paths, interleaved `[re,im,…]` staging, kappa→block dispatch (kappa<0=GT, kappa>0=LT, kappa==0=both).

### Spinor transform & σ·p machinery (the files this phase extends)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — `cart_to_spinor_si_2d` (Phase-28 1e si transform, the **structural template** for the new 2e si transforms), `cart_to_spinor_sf_2d`, single-block `cart_to_spinor_si`/`iket_si`, `spinor_len(l, kappa)` (GT/LT/both sizing — drives the kappa≠0 path). The new `c2s_si_2e1/2e2(+i)` + `c2s_sf_2e1/2e2` land here.
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — the Phase-28 reusable generic `#[cube]` σ·p G-tensor assembler (emits pre-blocked `gc_x/gc_y/gc_z/gc_1`). The 1e σ families (Wave 1) reuse it directly; the 2e σ families build on the same pattern.
- `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — CG coupling coefficient tables.
- `crates/cintx-cubecl/src/transform/mod.rs` — `apply_representation_transform()`; Spinor is dispatched explicitly in kernel launchers, NOT through the generic transform arm.

### Kernel launchers (call sites + reuse)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — 1e launch path; the `int1e_sp` Spinor arm (Phase-28 28-04) is the template for wiring the rest of the 1e σ families. **Each inline Spinor arm needs its own fail-closed staging guard** (Phase-28 CR-01 / `spinor_dispatch_arm_needs_own_staging_guard`).
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — has `test_device_matches_host_spsp` (L1878), an existing spsp device/host harness to mine for the σ·p-on-both-sides pattern.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — the 2e launch path the REL-03/04 families wire into.

### Manifest & coverage
- `crates/cintx-ops/src/generated/api_manifest.rs` + `compiled_manifest.lock.json` — currently only `int1e_sp_spinor` exists (Phase 28). Add ManifestEntry rows for every Group-4 σ family. The lock is the source of truth; edits auto-sync both audit sides. **Verify `component_rank` against the rank-tier table for every flipped row** (component_rank truncation landmine).
- `xtask/src/oracle_covered_update.rs` — the flip mechanism + the SC#4 skipped-fixture-flip-refusal guard (inherited from Phase 28).
- **Landmine:** adding manifest rows shifts positional `OperatorId`s → breaks any hardcoded `OperatorId::new(N)` / `_OPERATOR_ID: u32 = N` test consts (`operator_id_shift_breaks_hardcoded_test_consts`). Resolve by symbol name; re-grep after adding rows.

### Oracle / vendor parity infrastructure
- `crates/cintx-oracle/build.rs` — **`autocode/intor4.c` is ALREADY wired** (lines 62, 229). The 2e σ spinor drivers compile; only `vendor_*` FFI shims (and suppl-header `extern` decls for symbols absent from `cint_funcs.h`) are needed — NOT a build change.
- `crates/cintx-oracle/src/vendor_ffi.rs` — vendored libcint FFI; add `vendor_int1e_*`/`vendor_int2e_*` shims for each Group-4 family.
- `crates/cintx-oracle/src/fixtures.rs` — `build_kappa_spinor_fixture` (Phase-28 1e gate, the template); add `build_kappa_spinor_2e_fixture` (D-02).
- `crates/cintx-oracle/src/compare.rs` — oracle comparison, atol=1e-12.

### Upstream reference (byte-authoritative — transcribe from here, D-03)
- `libcint-master/src/cart2sph.c` — `c2s_si_2e1` (L5592), `c2s_si_2e1i` (L5639), `c2s_si_2e2` (L5687), `c2s_si_2e2i` (L5752); plus `c2s_si_1e` (L4947), `a_bra_cart2spinor_si` (L3920) already used by Phase 28.
- `libcint-master/src/autocode/intor4.c` — the 2e σ family drivers + their `c2s_si_2e*`/`c2s_sf_2e*` transform pairings (REL-03/04). `int2e_spsp1_spinor` L79-86; the i-variant arms at L636/899/990/1249/1539/1700/2197.
- `libcint-master/src/autocode/intor3.c` — the 1e σ family drivers (`int1e_sp/spsp/spnucsp/sprinvsp/srsr/sigma_spinor`) + their gout (REL-01/02).

### Skill
- `.claude/skills/spike-findings-cintx/SKILL.md` + `references/spinor-layout.md` — interleaved-complex layout `out[comp*(ni_sp*nj_sp)*2 + (j*ni_sp+i)*2 + {re,im}]`, component-leading + ket-major around the interleave, `ni_sp=4l+2` @ kappa=0 / `2l`|`2l+2` @ kappa≠0. **Load before touching c2s/output layout** — directly relevant to the new 2e si transforms.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The entire Phase-28 Gap-B2 foundation is in place:** `cart_to_spinor_si_2d` (host) + `kernels/sigma_p.rs` σ·p `#[cube]` assembler + `build_kappa_spinor_fixture`. Wave 1 (1e σ) wires families onto this with no new structural pieces.
- `cart_to_spinor_si_2d` is the structural template for the new 2e si transforms (`c2s_si_2e1/2e2`).
- `spinor_len(l, kappa)` already handles GT/LT/both sizing — drives the kappa≠0 buffer sizing the D-02 2e fixture stresses.
- `center_4c1e.rs::test_device_matches_host_spsp` (L1878) — an existing spsp device/host harness to mine for the σ·p-on-both-sides pattern.
- `intor4.c` is already in the oracle `build.rs` — no oracle build change for the 2e block.

### Established Patterns
- transforms (`c2spinor.rs`) are HOST fns on the contracted cart staging post-kernel; σ·p gout/nabla is DEVICE `#[cube]`.
- New-family surface = manifest + RawApiId + kernel + vendor-FFI + oracle ONLY; no capi/legacy.
- Vendor parity double-gated: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`; add the no-silent-skip assertion (Phase 27 D-10) so families never silently skip.
- Interleaved `[re0,im0,re1,im1,…]` complex staging, column-major (j outer, i inner); oracle compares the flat buffer directly.
- Flip `oracle_covered=true` **spinor-only** — do not over-claim cart/sph σ intermediates (SC#5).

### Integration Points / Landmines
- **bra-Pauli-mix / ket-ordinary** (Phase 28) — for `int1e_sp`. For `spsp` (σ·p on BOTH sides) the second σ·p is folded into the gout per libcint; the 1e si transform stays bra-only. Confirm the 2e analog (`c2s_si_2e1` on electron 1, `c2s_sf_2e2`/`c2s_si_2e2` on electron 2) from `intor4.c`.
- **kappa≠0 changes spinor sizing** (`di = 2l` or `2l+2`, NOT `4l+2`) — sizing MUST come from `spinor_len`, never hardcoded.
- **KET→BRA transpose** — own it inside the transform (B1 D-06); device cart blocks are KET-major.
- **nctr>1 column/row-major coeff transpose** — the 2e fixture keeps an nctr>1 shell.
- **component_rank truncation** — verify rank values in the lock for any flipped row.
- **OperatorId shift** — adding manifest rows re-points hardcoded test consts; resolve by symbol name, re-grep after adding rows.
- **Each inline Spinor dispatch arm needs its own fail-closed staging guard** (Phase-28 CR-01).
- **CubeCL CpuRuntime FP-environment side effect** — a `#[cube]` launch can perturb subsequent host f64 accumulation ~1e-11 and trip flat-atol gates; suspect this before chasing kernel numerics.

</code_context>

<specifics>
## Specific Ideas

The user wants the **honest full scope** on the 2e foundation — build the complete `c2s_si_2e1/2e2(+i)` + `c2s_sf_2e` suite so REL-03 AND REL-04 actually close, explicitly correcting the Phase-28 record that punted these to 30/31. The user accepts **no separate design spike** (a deliberate departure from Phase-28's hard-gate precedent), relying on the atol=1e-12 vendor byte-identity gate as the correctness backstop — with the understanding that the novel 2e transform layout carries rework risk, mitigated structurally by making a transform-level byte-identity micro-test the first task of the 2e wave. Sequential 3-wave de-risk: 1e σ (free-rides Phase 28) → 2e foundation → 2e families.
</specifics>

<deferred>
## Deferred Ideas

- **GIAO×σ slice (Phase 30)** — `int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, 2e `int2e_cg_sa10*`/`giao_sa10*`. Reuses Phase 29's σ·p pattern + the 2e si transforms built here, combined with the gauge origin (Phase 22) and complex output (Phase 26).
- **Gauge / Breit–Gaunt 2e (Phase 31)** — `int2e_gauge_r1/r2_*`, Gaunt `ssp/sps`. Reuses the Group-4 σ·p + 2e si machinery via per-block `launch_breit` decomposition.
- **PARITY-01 full-parity gate (Phase 31)** — every libcint 6.1.3 family `oracle_covered=true`, empty unsupported list.

### Reviewed Todos (not folded)
- `oracle-cart-offset-vendor-zero` (score 0.6) — the lurking `CINTshells_cart_offset[4]` (cintx=8 vendor=0) failure in `compare::tests::helper_coverage_matches_manifest`. Belongs to the **PARITY-01 / Phase-31** full-parity gate (must be green there), not Group-4 σ family work. Deferred.
- `rys-nroots-ge6-wheeler-fallback` (score 0.6) — Rys nroots≥6 Wheeler fallback; general math-infra carry-over, no Group-4 σ relevance. Deferred.

</deferred>

---

*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Context gathered: 2026-05-31*
