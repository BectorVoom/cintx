# Phase 30: Group 5 (GIAO×σ slice) — Spin-GIAO Integrals (spinor) - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Register and prove the **non-Gaunt GIAO×σ family set** at **spinor byte-identity** (atol=1e-12) against vendored libcint 6.1.3, closing **GIAO-03** and completing the magnetic-property suite (relativistic-NMR corrections). The integrals combine the **purely-imaginary GIAO gauge factor** (Phase 22 gauge origin) with the **σ·p `c2s_si` path** (Phase 28/29) and the **complex-interleaved output** (Phase 26).

**In scope — the family set:**

- **1e (intor3.c):** `int1e_spgsp`, `int1e_spgnucsp`, `int1e_spgsa01`; `int1e_cg_sa10sp`, `int1e_cg_sa10nucsp`, `int1e_cg_sa10sa01`; `int1e_giao_sa10sp`, `int1e_giao_sa10nucsp`, `int1e_giao_sa10sa01`. All `sa01` ("other-side" spin-angular) arms are **included** (D-01b). These route mostly through `c2s_si_1ei` (the imaginary 1e si transform — GIAO is purely imaginary), with a few `c2s_si_1e`/`c2s_sf_1e` arms.
- **2e (intor4.c):** `int2e_spgsp1`, `int2e_spgsp1spsp2`; `int2e_cg_sa10sp1`, `int2e_cg_sa10sp1spsp2`; `int2e_giao_sa10sp1`, `int2e_giao_sa10sp1spsp2`. These reuse the Phase-29 2e si/sf transform suite (`c2s_si_2e1/2e2(+i)`, `c2s_sf_2e1/2e2`).

Every family is exercised on a **combined gauge≠0 ∧ kappa≠0** spinor fixture (D-02), gets a dedicated `vendor_*` test under BOTH gate flags (non-skipped), and is flipped `oracle_covered=true` **spinor-only**. `manifest-audit` green. **No capi enum variants, no legacy `cint*` wrappers** (new-family surface policy).

**Out of scope (deferred to Phase 31):**
- The **Gaunt GIAO families** `int2e_cg_ssa10ssp2` / `int2e_giao_ssa10ssp2` (in `autocode/gaunt1.c`, driven by `launch_breit`) — D-01a. The GIAO-03 glob `int2e_cg_sa10*` literally excludes `ssa10`; these belong to Phase 31's Gaunt/`launch_breit` scope (BREIT-03) and its full-parity gate.
- Gauge / Breit–Gaunt 2e (`int2e_gauge_r1/r2_*`), the PARITY-01 full-parity gate.

**GIAO-03 closure note:** because the GIAO-03 requirement glob excludes `ssa10`, GIAO-03 is **fully satisfied** by this phase's non-Gaunt set — no Phase-31 spillover of GIAO-03 itself.
</domain>

<decisions>
## Implementation Decisions

### Family scope boundary
- **D-01a (user choice — defer Gaunt to Phase 31):** The Gaunt GIAO families `int2e_{cg,giao}_ssa10ssp2` are **NOT** in Phase 30. They live in `autocode/gaunt1.c`, decompose via `launch_breit` (Phase 14 / `BreitShape` dependency), and the GIAO-03 glob `int2e_cg_sa10*` excludes the `ssa10` prefix. Phase 31's BREIT-03 + full-parity gate (SC#4) captures them. Keeps Phase 30 cohesive with **no new `launch_breit` dependency**.
- **D-01b (user choice — include all `sa01` arms):** `int1e_spgsa01`, `int1e_cg_sa10sa01`, `int1e_giao_sa10sa01` ARE in scope. They are 1e GIAO×σ families in `intor3.c` covered by the GIAO-03 globs `int1e_spg*` / `*_sa10*`, reuse the same si/sf transforms + gauge gout, and closing them here makes GIAO-03 fully complete with no 1e-magnetic spillover into Phase 31.

### Gauge+kappa fixture
- **D-02 (user choice — ONE combined gauge∧kappa fixture):** Build a **single** fixture that is simultaneously **gauge-origin ≠ 0** (`with_common_origin` / `PTR_COMMON_ORIG`) AND **kappa ≠ 0** GT/LT spinor mix, **non-square**, with **≥1 shell carrying nctr>1**. Provide a 1e form (Wave 1 gate) and extend it to a **4-shell 2e** form (Wave 2 gate). This extends Phase-29's `build_kappa_spinor_2e_fixture` with a non-zero common origin. Rationale: the integrand **couples** gauge and kappa, so one fixture exercises the real gauge×kappa cross-term and is the honest single SC#2 gate. (A separate-fixtures composition was rejected — neither alone stresses the cross-term.)

### De-risk approach
- **D-03 (user choice — inherit Phase-29 transcribe + vendor gate; gout-level micro-test first):** **No separate design spike.** Transcribe the gouts/transforms directly from libcint `intor3.c`/`intor4.c` and prove by the atol=1e-12 vendor byte-identity gate. Apply the Phase-29 D-03 mitigation but **at the GOUT level** (since the transforms already exist): make a **gauge-gout byte-identity micro-test** — verifying the gauge-origin-dependent GIAO `g`-factor folded into the σ·p gout, against a thin family that exercises only that gout — the **FIRST task**, before wiring the full family set onto it.
  - **Accepted risk (capture for planner/executor):** the novel piece is folding the gauge-origin GIAO factor (Phase-22 `common_orig`) into the σ·p device assembler (`sigma_p.rs`). A wrong gauge-factor sign/stride/origin-read surfaces only at the gate; the gout micro-test buys spike-level rigor on exactly that piece without a separate spike phase.

### Plan / wave decomposition
- **D-04 (user choice — gout micro-test → 1e-all → 2e-all):**
  - **Wave 0 — gauge-gout micro-test + combined fixture (1e form):** land the D-02 combined fixture (1e) and the D-03 gauge-gout byte-identity micro-test before any family is wired.
  - **Wave 1 — all 1e families:** register + prove `spg{sp,nucsp,sa01}` + `{cg,giao}_sa10{sp,nucsp,sa01}` on the existing `c2s_si_1ei`/`c2s_si_1e`/`c2s_sf_1e` transforms + the gauge gout. Lowest risk (no 2e launcher), lands first.
  - **Wave 2 — all 2e families:** extend the fixture to the 4-shell 2e form, then register + prove `spgsp1(spsp2)` + `{cg,giao}_sa10sp1(spsp2)` on the Phase-29 2e si/sf suite.
  - Each wave is gated (vendor parity green) before the next begins. Mirrors Phase-29's 1e→foundation→2e shape (here foundation collapses to the Wave-0 micro-test, since the transforms already exist).

### Claude's Discretion
- Internal module/factoring for the gauge-origin σ·p gout — extend `kernels/sigma_p.rs` with a gauge-origin-dependent variant vs a dedicated `giao_sigma` assembler. Resolve during research/planning from `intor3.c`/`intor4.c` gout bodies.
- Exact per-family gout component ordering and the precise `c2s_si_1ei` vs `c2s_si_1e` vs `c2s_sf_1e` arm per family — transcribe from the verified driver pairings (see canonical refs).
- Exact molecule/element + kappa + gauge-origin coordinates for the combined fixture (subject to D-02 hard constraints: gauge≠0, kappa≠0 GT/LT, non-square, ≥1 nctr>1).
- Whether the Wave-0 gout micro-test compares to a vendored thin family or to a hand-derived reference.
- Precise plan boundaries inside each wave (e.g. Wave 1 may split the `spg*` group from the `*_sa10*` groups).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **GIAO-03** (line 118): the requirement this phase closes (glob `int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`, via FND-05). **FND-01** (line 78): the `PTR_COMMON_ORIG` gauge-origin plumbing this phase depends on. GIAO-01/02 (lines 116-117) are the spin-free GIAO precedents.
- `.planning/ROADMAP.md` §"Phase 30" — Goal, the 2 success criteria (SC#2 = both-fixture gating + non-skipped `vendor_*` + spinor-only flip + manifest-audit green + no capi/legacy). §"Phase 31" — the deferred Gaunt `ssa10ssp2` + BREIT-03 + PARITY-01 full-parity consumer.

### Prior phase context (decided conventions to inherit — do NOT re-decide)
- `.planning/phases/29-group-4-relativistic-spin-operator-integrals-spinor/29-CONTEXT.md` — the Group-4 σ foundation this phase reuses: the 2e si/sf transform suite (`c2s_si_2e1/2e2(+i)`, `c2s_sf_2e1/2e2`), the kappa-fixture design (`build_kappa_spinor_2e_fixture`), transcribe+vendor-gate de-risk (D-03), the spinor-only flip, the full landmine list (component_rank, OperatorId shift, per-arm staging guard, KET→BRA transpose, nctr coeff transpose, CubeCL FP side effect).
- `.planning/phases/28-spin-included-c2s-si-transform-p-module-gap-b2/28-CONTEXT.md` — the Gap-B2 foundation: `si_2d` host transform, the reusable σ·p `#[cube]` assembler (`sigma_p.rs`), bra-Pauli-mix/ket-ordinary split, no-silent-skip + skipped-fixture-flip-refusal.
- `.planning/phases/22-gauge-origin-env-slot-gap-a-ptr-common-orig/` (FND-01) — the gauge-origin env slot precedent and its non-zero gauge fixture, the basis for D-02's combined fixture.
- `.planning/phases/12-real-spinor-transform-c2spinor-replacement/12-CONTEXT.md` — CG coefficient source, the `sf/iket_sf/si/iket_si` code paths, interleaved `[re,im,…]` staging, kappa→block dispatch.

### Spinor transform & σ·p machinery (already built — this phase REUSES)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — **`cart_to_spinor_si_2di` (L754, = libcint `c2s_si_1ei`, the imaginary 1e si transform — the PRIMARY transform for the GIAO×σ 1e families)**, `cart_to_spinor_si_2d` (`c2s_si_1e`), `cart_to_spinor_sf_2d` (`c2s_sf_1e`), the 2e suite `cart_to_spinor_si_2e1/2e2(+i)` (L1515-1579) + `cart_to_spinor_sf_2e1/2e2`, `spinor_len(l, kappa)` (GT/LT/both sizing). **No new transform needs building.**
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — the Phase-28 reusable generic `#[cube]` σ·p G-tensor assembler. The gauge-origin GIAO `g`-factor folds in HERE (the genuinely-new work, D-03).
- `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — CG coupling coefficient tables.

### Gauge-origin plumbing (already built — FND-01)
- `crates/cintx-rs/src/builder.rs` (L109-113) — `with_common_origin([f64;3])` safe-API setter.
- `crates/cintx-runtime/src/options.rs` (L124-127) — `common_orig: Option<[f64;3]>` on `ExecutionPlan` options.
- `crates/cintx-runtime/src/validator.rs` (L210-217) — `validate_common_orig_env_params` → `PTR_COMMON_ORIG` finiteness gate.
- `crates/cintx-cubecl/src/.../raw.rs` (`eval_raw`) — the env-read for `PTR_COMMON_ORIG`; the device gout must consume this origin.

### Kernel launchers (call sites + reuse)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — 1e launch path; the existing σ-family Spinor arms (Phase 28/29) are the template. **Each inline Spinor arm needs its own fail-closed staging guard** (Phase-28 CR-01).
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — the 2e launch path the `spgsp1`/`sa10sp1` families wire into (reusing the Phase-29 2e σ launcher pattern).

### Manifest & coverage
- `crates/cintx-ops/src/generated/api_manifest.rs` + `compiled_manifest.lock.json` — add ManifestEntry rows for every GIAO×σ family. The lock is the source of truth; edits auto-sync both audit sides. **Verify `component_rank` against the rank-tier table for every flipped row** (component_rank truncation landmine — GIAO `g`-factor raises rank).
- `xtask/src/oracle_covered_update.rs` — the flip mechanism + the SC#4 skipped-fixture-flip-refusal guard.
- **Landmine:** adding manifest rows shifts positional `OperatorId`s → breaks hardcoded `OperatorId::new(N)` / `_OPERATOR_ID: u32 = N` test consts. Resolve by symbol name; re-grep after adding rows.

### Oracle / vendor parity infrastructure
- `crates/cintx-oracle/build.rs` — `autocode/intor3.c` + `intor4.c` are ALREADY wired (Phase 29). The GIAO×σ spinor drivers compile; only `vendor_*` FFI shims (+ suppl-header `extern` decls for symbols absent from `cint_funcs.h`) are needed — NOT a build change. **`gaunt1.c` is NOT needed this phase** (deferred Gaunt families, D-01a).
- `crates/cintx-oracle/src/vendor_ffi.rs` — add `vendor_int1e_*`/`vendor_int2e_*` shims for each GIAO×σ family.
- `crates/cintx-oracle/src/fixtures.rs` — add the combined gauge∧kappa fixture(s) (D-02), extending `build_kappa_spinor_fixture` / `build_kappa_spinor_2e_fixture` with `with_common_origin`.
- `crates/cintx-oracle/src/compare.rs` — oracle comparison, atol=1e-12.

### Upstream reference (byte-authoritative — transcribe from here, D-03)
- `libcint-master/src/autocode/intor3.c` — the 1e GIAO×σ family drivers + gouts: `CINTgout1e_int1e_cg_sa10sa01` (L998), `int1e_cg_sa10sp` (gout L1141), `CINTgout1e_int1e_giao_sa10sp` (L1462), `CINTgout1e_int1e_spgsp` (L1724), `CINTgout1e_int1e_spgnucsp` (L1878), `CINTgout1e_int1e_spgsa01` (L2036). The driver `c2s_*` pairings: GIAO×σ 1e families predominantly use **`c2s_si_1ei`** (L81/285/351/1225/1542/1873/etc.), with `c2s_si_1e`/`c2s_sf_1e` for specific arms.
- `libcint-master/src/autocode/intor4.c` — the 2e GIAO×σ family drivers (`int2e_cg_sa10sp1` gout L547, `int2e_giao_sa10sp1` gout L905, + the `spsp2` variants) and their `c2s_si_2e*`/`c2s_sf_2e*` transform pairings.
- `libcint-master/src/cart2sph.c` — `c2s_si_1ei` (L~), `c2s_si_2e1/2e1i/2e2/2e2i` (L5592/5639/5687/5752) — already transcribed in Phase 28/29; reference for verifying the existing cintx transforms match the GIAO×σ usage.

### Skill
- `.claude/skills/spike-findings-cintx/SKILL.md` + `references/spinor-layout.md` — interleaved-complex layout `out[comp*(ni_sp*nj_sp)*2 + (j*ni_sp+i)*2 + {re,im}]`, `ni_sp=4l+2` @ kappa=0 / `2l`|`2l+2` @ kappa≠0. **Load before touching output layout / the gauge gout.**
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The entire transform foundation already exists.** `cart_to_spinor_si_2di` (`c2s_si_1ei`, the imaginary 1e si — primary for GIAO×σ 1e), `cart_to_spinor_si_2d`/`sf_2d`, and the full Phase-29 2e si/sf suite. **No new transform is built this phase.**
- **Gauge origin is plumbed end-to-end** (FND-01): `with_common_origin` → `common_orig` → validator → `PTR_COMMON_ORIG` env slot. The device gout reads the origin; no new plumbing, only consumption.
- `kernels/sigma_p.rs` σ·p `#[cube]` assembler is the host for the new gauge `g`-factor fold.
- `build_kappa_spinor_fixture` / `build_kappa_spinor_2e_fixture` (Phase 29) — the templates for the D-02 combined fixture; add `with_common_origin`.
- `intor3.c` + `intor4.c` already in the oracle `build.rs` — no oracle build change.

### Established Patterns
- transforms (`c2spinor.rs`) are HOST fns on the contracted cart staging post-kernel; σ·p gout/nabla + the gauge `g`-factor are DEVICE `#[cube]`.
- New-family surface = manifest + RawApiId + kernel + vendor-FFI + oracle ONLY; no capi/legacy.
- Vendor parity double-gated: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`; add the no-silent-skip assertion so families never silently skip.
- GIAO families are **purely imaginary** — staged into the imaginary lane of the interleaved `[re,im,…]` buffer; oracle compares the flat buffer directly.
- Flip `oracle_covered=true` **spinor-only** — do not over-claim cart/sph GIAO×σ intermediates.

### Integration Points / Landmines
- **Gauge-origin gout fold** — the GIAO `g = (r - common_orig) × ...` factor must read `PTR_COMMON_ORIG` inside the device gout; a wrong origin-read/sign surfaces only at the vendor gate (the D-03 micro-test targets exactly this).
- **`c2s_si_1ei` (imaginary) vs `c2s_si_1e` (real)** — GIAO×σ 1e families mostly use the imaginary arm; pick the correct transform per family from the verified `intor3.c` driver pairing, not by analogy.
- **kappa≠0 changes spinor sizing** (`di = 2l` or `2l+2`, NOT `4l+2`) — from `spinor_len`, never hardcoded; the D-02 fixture stresses this.
- **KET→BRA transpose** — owned inside the transform; device cart blocks are KET-major.
- **nctr>1 column/row-major coeff transpose** — the D-02 fixture keeps an nctr>1 shell.
- **component_rank truncation** — the GIAO `g`-factor raises component rank; verify rank values in the lock for any flipped row.
- **OperatorId shift** — adding manifest rows re-points hardcoded test consts; resolve by symbol name, re-grep after adding rows.
- **Each inline Spinor dispatch arm needs its own fail-closed staging guard** (Phase-28 CR-01).
- **CubeCL CpuRuntime FP-environment side effect** — a `#[cube]` launch can perturb subsequent host f64 accumulation ~1e-11 and trip flat-atol gates; suspect this before chasing kernel numerics.
</code_context>

<specifics>
## Specific Ideas

The user wants GIAO-03 closed **fully and cohesively in Phase 30** — all 1e GIAO×σ families including the `sa01` "other-side" arms — while keeping the phase free of any new `launch_breit` dependency by deferring the Gaunt `ssa10ssp2` families to Phase 31 (where the full-parity gate captures them). Because most of the machinery already exists (imaginary si transform, 2e si/sf suite, gauge-origin plumbing), the user accepts the Phase-29 no-spike convention, relocating the de-risk to a **gauge-gout byte-identity micro-test** as the first task — the gauge `g`-factor fold into the σ·p assembler is the only genuinely-new piece and carries the rework risk. A **single combined gauge≠0 ∧ kappa≠0 fixture** is the honest SC#2 gate because the integrand couples both. Sequential de-risk: gout micro-test → all 1e → all 2e.
</specifics>

<deferred>
## Deferred Ideas

- **Gaunt GIAO families (Phase 31)** — `int2e_cg_ssa10ssp2`, `int2e_giao_ssa10ssp2` (`autocode/gaunt1.c`, `launch_breit` decomposition). Reuse this phase's gauge-gout + the Group-4 σ·p / 2e si machinery; captured by BREIT-03 + the PARITY-01 full-parity gate.
- **Gauge / Breit–Gaunt 2e (Phase 31)** — `int2e_gauge_r1/r2_*`, Gaunt `ssp/sps` (BREIT-01/02/03).
- **PARITY-01 full-parity gate (Phase 31)** — every libcint 6.1.3 family `oracle_covered=true`, empty unsupported list.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 30-group-5-giao-slice-spin-giao-integrals-spinor*
*Context gathered: 2026-06-01*
