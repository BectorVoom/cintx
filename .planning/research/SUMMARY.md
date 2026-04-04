# Project Research Summary

**Project:** cintx v1.2 — Full API Parity & Unified Oracle Gate
**Domain:** Rust reimplementation of libcint 6.1.3 — quantum chemistry integral library
**Researched:** 2026-04-04
**Confidence:** HIGH

## Executive Summary

cintx v1.2 is a targeted completion milestone for an already-functional Rust reimplementation of libcint. The five base integral families (1e, 2e, 2c2e, 3c1e, 3c2e) are live with oracle CI parity at v1.1. v1.2 closes the remaining gaps to claim full API parity: F12/STG/YP short-range correlation kernels, real spinor Clebsch-Gordan transforms (replacing a provably-wrong placeholder), extended 4c1e oracle coverage, helper/transform API oracle wiring, unstable-source-api families, and manifest lock regeneration across all four feature profiles. No new external crates are required — all new math (STG roots, Clebsch-Gordan coefficient tables) follows the same embedded-static-table strategy already used for Rys roots and Condon-Shortley transforms.

The recommended build order is strictly dependency-ordered: complete helper/transform API oracle wiring first (pure host-side, no kernel risk), then 4c1e real kernel (same Rys infrastructure as 2e), then the spinor Clebsch-Gordan transform replacement (prerequisite for spinor oracle coverage), then F12/STG/YP kernels (reuse Rys/pdata infrastructure verified in the 4c1e step), then unstable-source-api families, and finally a unified oracle tolerance audit across all profiles. This sequencing ensures each step builds on a verified foundation rather than compounding undetected errors across the dispatch chain.

The primary risks are: (a) the unified atol=1e-12 goal conflicting with numerically-achievable precision floors for 3c1e/4c1e/F12 — the design doc explicitly caps these at 1e-6 to 1e-7, so the correct interpretation is "no family exempt from oracle comparison" rather than "every family at 1e-12"; (b) the spinor transform stub shipping as a real transform because existing tests only check buffer length, not correctness; and (c) silent wrong-result paths for F12 when `env[PTR_F12_ZETA]` is zero. All three risks are mitigated by gating oracle comparison against the real upstream reference before advancing each phase.

## Key Findings

### Recommended Stack

The v1.2 stack requires no new external crates. All additions are internal: a new `math/stg.rs` sub-module in `cintx-cubecl` for STG root evaluation via Clenshaw/DCT recurrence (embedding `roots_xw.dat` table as static arrays), a new `transform/c2spinor_coeffs.rs` module with real/imaginary Clebsch-Gordan tables extracted from `libcint-master/src/cart2sph.c` lines 809-3535, and an `operator_env_params` field extension on `ExecutionPlan` in `cintx-runtime` to carry `PTR_F12_ZETA` (env[9]) to kernel launchers. The one dependency change is promoting `num-complex 0.4.6` (already a transitive lockfile entry via cubecl) to a direct dependency in `cintx-core` and `cintx-cubecl`.

**Core technologies (unchanged from v1.1):**
- `cubecl 0.9.0` (locked): GPU+CPU compute backend — backend-agnostic public API; CPU path provides oracle parity CI without GPU hardware
- `thiserror 2.0.18`: Public typed error surface — library-facing error enums without leaking implementation details
- `anyhow 1.0.102`: App-boundary errors in xtask, oracle harness, benchmarks
- `bindgen 0.71.1`: Oracle binding generation — upgrade deliberately, not automatically
- `num-complex 0.4.6` (new direct dep): Typed `Complex<f64>` for spinor output path — already in lockfile, zero cost to promote

**New internal modules (no new crates):**
- `cintx-cubecl/src/math/stg.rs`: Clenshaw/DCT port of `CINTstg_roots`; embeds `roots_xw.dat` table as static arrays indexed by `nroots * 196 * (iu + it * 10)`
- `cintx-cubecl/src/transform/c2spinor_coeffs.rs`: Real and imaginary Clebsch-Gordan tables for l=0..8, extracted from `cart2sph.c`
- `cintx-runtime`: `ExecutionPlan` extended with `operator_env_params: Option<OperatorEnvParams>` for F12 zeta routing

### Expected Features

**Must have (P1 — v1.2 release gate requires these):**
- F12/STG/YP real kernels: 10 sph-only symbols (`int2e_stg_sph`, `int2e_yp_sph`, and four derivative variants each) plus matching optimizer symbols; oracle gate at atol=1e-6 under `with-f12` profile
- F12 sph-only enforcement CI gate: "cart/spinor symbol count is zero" verified in `with-f12` profile; this is itself a pass condition, not merely a restriction
- 4c1e full oracle coverage: `int4c1e_cart` and `int4c1e_sph` passing oracle within `Validated4C1E` envelope (cart/sph, scalar, max(l)<=4); bug-envelope rejection tests verify `UnsupportedApi` for out-of-envelope inputs including all spinor requests
- 4c1e workaround path: `compat::workaround::int4c1e_via_2e_trace` for callers needing 4c1e outside the validated envelope
- Full cart-to-spinor transforms: all four `CINTc2s_*spinor*` variants with correct Clebsch-Gordan coupling (current stub is provably wrong); spinor oracle gate for 1e family
- Helper API oracle CI gate: count/offset/norm functions verified against upstream libcint 6.1.3 using exact integer equality (not float tolerance)
- Legacy wrapper oracle CI gate: `cNAME_sph` legacy symbols verified against upstream
- Manifest lock regenerated for full support matrix: `compiled_manifest.lock.json` covering all four profiles `{base, with-f12, with-4c1e, with-f12+with-4c1e}`
- Unified oracle tolerance: base scalar families (1e, 2e, 2c2e, 3c1e, 3c2e) verified at atol=1e-12 where achievable; per-design-doc exceptions for 4c1e/F12/spinor (atol=1e-6)

**Should have (P2 — unstable-source-api coverage):**
- origi/origk families: `int1e_r2_origi`, `int1e_r4_origi`, `int3c1e_r*_origk` variants behind `unstable-source-api`; needed for magnetic property calculations (GIAO, London orbitals)
- `int1e_grids` family: DFT numerical grid integration behind `unstable-source-api`; requires `NGRIDS`/`PTR_GRIDS` env slot parsing and GPU coordinate upload
- Spinor oracle coverage for 2e/2c2e families: after 1e spinor is proven correct

**Defer to v1.2.x or v1.3 (P3):**
- Breit integral kernels (`int2e_breit_r1p2`, `int2e_breit_r2p2`): high-accuracy relativistic only; new 2e Breit-Pauli operator variant; ship as documented `UnsupportedApi` stubs initially
- `int3c2e_ssc`: spin-orbit coupling; new 3c2e variant
- Feature-matrix CI evidence report: tooling quality improvement, not a correctness gate
- `int1e_r4_origi_ip2` / `int1e_r2_origi_ip2`: higher-derivative origi variants

**Anti-features (do not implement):**
- GTG integrals: upstream explicitly documents "bugs in gtg type integrals" in CMakeLists.txt; no valid oracle exists; classify as `planned-excluded`
- F12 cart/spinor variants: do not exist in compiled upstream library; return `UnsupportedRepresentation`
- 4c1e beyond `Validated4C1E` envelope: upstream has known bugs; return `UnsupportedApi` with explicit reason
- Async `evaluate()` API: explicitly rejected by design doc section 1.3
- Bitwise-identical libcint internals: makes GPU backend impossible; oracle tolerance contract is the correct compatibility claim

### Architecture Approach

The v1.2 architecture is additive: the core dispatch chain (manifest resolver -> runtime planner -> cubecl dispatcher -> kernel) is unchanged. F12/STG/YP slots in as a new `canonical_family: "f12"` entry resolving to a new `kernels/f12.rs` file with the same `FamilyLaunchFn` signature. Spinor support is a pure math change inside `c2spinor.rs` with no dispatch modifications. 4c1e replaces the zero-fill stub inside the existing `center_4c1e.rs` with real Rys/Obara-Saika computation. Helper/transform completions are pure host-side Rust in `cintx-compat` with no kernel involvement. The manifest `oracle_covered` field flips from `false` to `true` as each symbol gains CI comparison — this is the primary v1.2 tracking mechanism.

**Major components and their v1.2 changes:**
1. `cintx-cubecl/src/kernels/f12.rs` (new): STG and YP 2e kernel variants sharing pdata/Rys infrastructure from `two_electron.rs`; `kernels/mod.rs` gains `"f12"` dispatch arm
2. `cintx-cubecl/src/transform/c2spinor.rs` (rewrite): replace amplitude-averaging stub with real Clebsch-Gordan coupling matrix application per (l, kappa); add `c2spinor_coeffs.rs` with coefficient tables
3. `cintx-cubecl/src/kernels/center_4c1e.rs` (complete stub): replace zero-fill with real Rys quadrature plus Obara-Saika recurrence within `Validated4C1E` envelope
4. `cintx-compat/src/helpers.rs`, `transform.rs`, `optimizer.rs` (complete): add remaining helper symbols, remaining transform symbols (spinor-dependent ones unblocked after component 2), F12 optimizer init
5. `cintx-oracle/src/compare.rs`, `fixtures.rs` (extend): add F12/4c1e/spinor tolerance entries, new fixtures with proper `env[PTR_F12_ZETA]` population, integer-equality path for helper APIs
6. `cintx-ops/src/generated/api_manifest.rs` (update): flip `oracle_covered: false` to `true` as each symbol gains CI coverage
7. `cintx-runtime/ExecutionPlan` (extend): add `operator_env_params` field for F12 zeta routing

**Unchanged (do not touch):**
- `cintx-core` domain types, `cintxRsError` variants, `Representation` enum
- `cintx-runtime` planner, validator, scheduler, `BackendExecutor` trait
- `cintx-rs` safe facade public surface
- `cintx-capi` C ABI shim surface
- CI workflow YAML structure, artifact paths, job topology

### Critical Pitfalls

1. **Flat atol=1e-12 applied to 3c1e/4c1e/F12 before empirical calibration**: Rys quadrature for high-order 3c1e accumulates 1e-8 to 1e-9 rounding; upstream testsuite uses 1e-7 for high-order 3c1e. Apply per-family tolerances per design doc section 13.8; "unified 1e-12" applies to base scalar families only. Run the full oracle sweep and measure the empirical error distribution before writing any CI gates with 1e-12.

2. **The spinor transform stub ships as a real implementation**: `c2spinor.rs` computes `(|re|+|im|)*0.5`, which is not a Clebsch-Gordan transform. Current tests only check buffer length. Spinor oracle comparisons will fail completely with this stub in place. Treat it as equivalent to `todo!()` and rewrite entirely before authoring any spinor oracle fixtures.

3. **F12 `env[PTR_F12_ZETA]=0` silently falls back to plain Coulomb**: `CINTstg_roots` is only called when `zeta > 0`. When zero, the code calls `CINTrys_roots` (plain 2e path), producing integrals that look correct but are not testing STG physics at all. The validator must reject F12/STG/YP calls where `env[9] == 0.0` with a typed error.

4. **4c1e spinor path is unimplemented upstream and must return `UnsupportedApi`**: `int4c1e_spinor` in upstream contains only `fprintf(stderr, "int4c1e_spinor not implemented\n"); return 0;`. Oracle comparison trivially passes (both sides zero). The `Validated4C1E` classifier must check representation before angular momentum and must explicitly reject spinor.

5. **YP and STG have distinct 4D recurrence routing for ibase/kbase combinations**: The `f_g0_2d4d` function pointer differs for YP vs STG when `ibase=true` or `kbase=true`. If both are implemented as a shared CubeCL kernel with a runtime zeta flag, the ibase/kbase routing diverges and YP produces wrong results for the majority of shell combinations. Implement as separate kernel entry points.

6. **Manifest lock regeneration must follow oracle parity, not precede it**: If the lock is regenerated without passing oracle first, the audit gate silently accepts incorrect coverage. Run `cargo xtask manifest-audit` immediately after adding each new symbol, treat failures as blocking, and do not regenerate the lock as a CI workaround.

7. **Helper API oracle comparison must use exact integer equality**: Count/offset helpers return integers. Float atol comparison masks off-by-one errors for shell sizes. Add a separate integer comparison path in the oracle harness before authoring helper fixtures.

## Implications for Roadmap

Based on combined research, the dependency chain enforces a natural 7-phase structure. Each phase gates the next via oracle comparison.

### Phase 1: Helper and Transform API Completion
**Rationale:** Pure host-side Rust in `cintx-compat` with no kernel dependency. Delivers expanded oracle coverage without touching the GPU dispatch chain, providing a clean CI baseline before any kernel work begins. Unblocks `helper-legacy-parity` CI gate items blocking release gate item 7. Non-spinor transform symbols can be completed now; spinor-dependent transform symbols are deferred to Phase 3.
**Delivers:** All non-spinor helper/transform/optimizer symbols wired to oracle CI; `helper-legacy-parity` gate passing; integer-equality comparison path added to oracle harness
**Addresses features:** Helper API oracle CI gate, legacy wrapper oracle CI gate (P1 table stakes)
**Avoids pitfalls:** Pitfall 7 (integer vs float comparison — add integer path first, then author fixtures)

### Phase 2: 4c1e Real Kernel Within Validated4C1E Envelope
**Rationale:** 4c1e uses the same Rys quadrature infrastructure as 2e. Completing it before F12 stress-tests the four-center Rys path at maximum complexity before adding the operator-kernel change that F12 introduces. This phase also adds the 4c1e identity test and the `int4c1e_via_2e_trace` workaround path.
**Delivers:** `int4c1e_cart` and `int4c1e_sph` passing oracle within Validated4C1E; bug-envelope rejection tests; `compat::workaround::int4c1e_via_2e_trace`; 4c1e spinor explicitly returning `UnsupportedApi`
**Uses:** Existing `rys_roots_host`, `compute_pdata_host`, Obara-Saika recurrences — no new dependencies
**Avoids pitfalls:** Pitfall 4 (4c1e spinor must return `UnsupportedApi`), Pitfall 8 (4c1e identity relation index permutation — copy from upstream testsuite verbatim, do not simplify)

### Phase 3: Real Spinor Transform (c2spinor Replacement)
**Rationale:** Spinor support is a prerequisite for oracle coverage of any spinor-form integral and for the spinor-dependent transform symbols deferred from Phase 1. The `c2spinor.rs` rewrite is a pure math change with no dispatch-chain risk — isolated to one file plus the new coefficients module. Completing it here unblocks Phase 5 spinor oracle coverage.
**Delivers:** Correct Clebsch-Gordan coupling matrix application in `c2spinor.rs`; `c2spinor_coeffs.rs` with real/imaginary tables for l=0..8; spinor oracle gate for 1e family; spinor-dependent transform symbols wired to oracle CI; `num-complex` promoted to direct dependency in `cintx-core` and `cintx-cubecl`
**Avoids pitfalls:** Pitfall 2 (stub cannot be tested with length checks alone), Pitfall 7 (spinor staging buffer must be sized `spinor_component_count * 2` for interleaved re/im doubles)

### Phase 4: F12/STG/YP Kernel Implementation
**Rationale:** F12 reuses the Rys/pdata infrastructure validated in Phase 2. The operator change (Coulomb kernel swap to STG/YP geminal) is the only new physics. Sph-only restriction simplifies testing since c2s is already validated. All 10 F12 sph symbols plus matching optimizer symbols must land in this phase.
**Delivers:** `kernels/f12.rs` with STG and YP variants as separate kernel entry points; `math/stg.rs` with embedded `roots_xw.dat` tables; `ExecutionPlan.operator_env_params` carrying `PTR_F12_ZETA`; F12 oracle gate under `with-f12` profile at atol=1e-6; "cart/spinor count is zero" CI enforcement; F12 optimizer coverage in `cintx-compat/src/optimizer.rs`
**Avoids pitfalls:** Pitfall 3 (validator must reject `env[PTR_F12_ZETA]==0.0`), Pitfall 5 (STG roots must replicate `t = min(t, 19682.99)` clamp), Pitfall 6 (YP and STG must be separate kernel paths for ibase/kbase routing divergence)

### Phase 5: Unstable-Source-API Families
**Rationale:** Source-only families have the weakest upstream specification and benefit from all previous infrastructure being stable. Oracle strategy differs: requires `dlsym`-based dynamic lookup rather than bindgen-generated bindings for source-only symbols not in headers. origi/origk and `int1e_grids` are P2; Breit/ssc are P3 and ship as `UnsupportedApi` stubs.
**Delivers:** `int1e_r2_origi`, `int1e_r4_origi`, `int3c1e_r*_origk` kernels behind `#[cfg(feature = "unstable-source-api")]`; `int1e_grids` with `NGRIDS`/`PTR_GRIDS` env parsing; oracle gate in nightly extended CI with `--include-unstable-source true`; Breit/ssc as documented `UnsupportedApi` stubs
**Avoids pitfalls:** Pitfall 11 (unstable-source oracle requires dlsym, not bindgen — verify harness can call upstream reference before writing any kernels)

### Phase 6: Manifest Lock Regeneration — Full Support Matrix
**Rationale:** After all feature-gated families have oracle coverage, regenerate `compiled_manifest.lock.json` for all four profiles in one coordinated operation. This must follow oracle parity, not precede it. The four-profile libcint build driven by `xtask manifest-audit` closes release gate item 1.
**Delivers:** `compiled_manifest.lock.json` covering `{base, with-f12, with-4c1e, with-f12+with-4c1e}`; `manifest-audit` CI gate passing with zero diff; all `oracle_covered` fields set to `true` for their respective profiles
**Avoids pitfalls:** Pitfall 10 (lock regeneration must follow oracle parity; F12 `with-f12` profile must have zero cart/spinor symbols verified by `nm -D`)

### Phase 7: Unified Oracle Tolerance Audit
**Rationale:** After all families have real kernels and oracle coverage, run a full tolerance audit across all profiles to confirm no family is exempt from comparison and per-family tolerances match design doc section 13.8. This is a verification and documentation phase, not an implementation phase.
**Delivers:** Per-family atol/rtol constants in `compare.rs` verified against empirical oracle sweep data; base scalar families (1e, 2e, 2c2e, 3c1e, 3c2e) at atol=1e-12 where achievable; 4c1e/F12/spinor exceptions documented with measurement evidence; every `stability: Stable` manifest entry has `oracle_covered: true` with a passing CI record
**Avoids pitfalls:** Pitfall 1 (3c1e has an empirical floor of 1e-7 for high-order shells; do not set 1e-12 without measurement; run sweep first, then set constants)

### Phase Ordering Rationale

- Phases 1-2 first because they have no unsolved math dependencies and deliver measurable CI progress (passing oracle gates) immediately, without touching the GPU kernel dispatch chain.
- Phase 3 (spinor) before Phase 4 (F12) because spinor Clebsch-Gordan completes the deferred transform symbols from Phase 1 and is lower-risk in isolation. F12 is sph-only and does not depend on spinor, but the spinor rewrite is fully bounded to one file.
- Phase 5 (unstable-source) last among kernel phases because source-only families have weaker upstream specification and benefit from the full oracle infrastructure being proven on well-documented families first.
- Phase 6 (manifest lock) must follow all kernel phases because the lock covers all four profiles; regenerating it early risks accepting incorrect coverage.
- Phase 7 (tolerance audit) last because it requires complete oracle coverage across all families to measure empirical error distributions.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (F12/STG/YP):** The STG roots Clenshaw/DCT port requires careful embedding of `roots_xw.dat` as Rust static arrays and exact replication of the t-clamp behavior. YP vs STG routing for ibase/kbase is a subtle correctness risk. Recommend `/gsd:research-phase` to lock the kernel design before coding.
- **Phase 5 (unstable-source-api):** Requires verifying at the start of phase planning that the oracle harness can call source-only upstream symbols via `dlsym`. If it cannot, the harness must be extended before any kernel is written.
- **Phase 7 (tolerance audit):** The 3c1e empirical precision floor for high angular momentum shells is not yet measured empirically against the cintx GPU path. Phase planning must include a measurement run before setting any new tolerance constants.

Phases with standard patterns (skip research-phase):
- **Phase 1 (helper/transform completion):** Pure host-side Rust; patterns are established in `helpers.rs` and `transform.rs`; oracle harness extension follows the existing `IMPLEMENTED_HELPER_SYMBOLS` pattern.
- **Phase 2 (4c1e kernel):** Same Rys/pdata/Obara-Saika infrastructure as 2e; `center_4c1e.rs` already has the correct envelope classifier; implementation follows `two_electron.rs` as template.
- **Phase 3 (spinor transform):** Coefficient tables are directly extractable from `cart2sph.c` lines 809-3535; the transform call sites already accept `(l, kappa)` parameters; the rewrite is isolated to one file.
- **Phase 6 (manifest lock):** `xtask manifest-audit` command already exists; the four-profile build procedure is documented in design doc section 3.3.1.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technology choices verified by direct codebase inspection; no new external crates required; `num-complex` already in lockfile as transitive dep |
| Features | HIGH | Feature scope derived from direct inspection of `libcint-master/src/`, design doc section 14.1, and `api_manifest.csv`; what ships vs defers is unambiguous |
| Architecture | HIGH | v1.1 architecture is verified working; v1.2 changes are additive with clear component boundaries; dispatch chain unchanged |
| Pitfalls | HIGH (code-derived) / MEDIUM (precision) | Critical pitfalls for spinor stub, F12 zeta, 4c1e spinor, and YP routing are from direct code inspection; precision floor claims for 3c1e/STG quadrature are from algorithm analysis and require empirical measurement during Phase 7 |

**Overall confidence:** HIGH

### Gaps to Address

- **3c1e empirical precision floor:** Current oracle tolerance for 3c1e is 1e-7 based on upstream empirical evidence. Whether existing `cintx-cubecl` 3c1e kernels can achieve 1e-12 for low-order shells has not been measured against the GPU path. Phase 7 planning must include a measurement sweep before setting any new constants.
- **Unstable-source oracle harness capability:** It is not yet verified that `cintx-oracle` can call source-only symbols via `dlsym`. This must be confirmed as the first action of Phase 5 planning before any unstable-source kernel work begins.
- **STG t-clamp GPU behavior:** The `t = min(t, 19682.99)` clamp in `CINTstg_roots` must be replicated exactly in the CubeCL implementation. Whether GPU floating-point handling of the clamp produces bit-identical results to the CPU path requires explicit fixture testing at the boundary (Phase 4).
- **"Unified atol=1e-12" interpretation:** PROJECT.md states atol=1e-12 as a v1.2 goal; ARCHITECTURE.md and PITFALLS.md both indicate this means "no family exempt from oracle comparison" rather than "every family achieves 1e-12." This interpretation should be locked in phase planning before the Phase 7 tolerance audit is written, to avoid writing gates that permanently fail for 3c1e/4c1e/F12.

## Sources

### Primary (HIGH confidence — direct code inspection)
- `libcint-master/src/stg_roots.c` — STG root algorithm: Clenshaw/DCT over `roots_xw.dat`, t-clamp at 19682.99, no external dep
- `libcint-master/src/g2e_f12.c` lines 113-127 — F12 kernel structure and YP vs STG ibase/kbase routing divergence; `PTR_F12_ZETA = env[9]`
- `libcint-master/src/cart2sph.c` lines 809-3535 — real/imaginary spinor Clebsch-Gordan tables for sf (j=l+1/2) and si (j=l-1/2)
- `libcint-master/src/cint4c1e.c:349-353` — `int4c1e_spinor` is explicitly unimplemented upstream (fprintf + return 0)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — confirmed amplitude-averaging stub, not a valid Clebsch-Gordan transform
- `crates/cintx-oracle/src/compare.rs` lines 21-31 — per-family tolerance constants; no `"f12"` arm in `tolerance_for_family`
- `crates/cintx-ops/src/generated/api_manifest.rs` — `ManifestEntry` schema with `oracle_covered`, `canonical_family`, `feature_flag`, `stability`
- `crates/cintx-compat/src/helpers.rs` — `len_spinor`, `len_spheric`, `len_cartesian` confirmed implementations
- `.planning/PROJECT.md` — v1.1 oracle gate closure confirmed; v1.2 requirements list
- `.planning/REQUIREMENTS.md` — v1.1 deferred items confirmed; spinor, F12, unstable-source deferred explicitly
- `libcint-master/include/cint.h.in` line 40 — `PTR_F12_ZETA = 9` confirmed

### Secondary (MEDIUM confidence — algorithm analysis)
- Design doc section 13.8 — per-family tolerance table; atol=1e-6 for F12/spinor/4c1e is the upstream empirical floor
- crates.io survey as of 2026-04-04 — no external Rust crate exists for STG roots, YP correlation factor, or 2j-spinor CG coefficients; in-house port confirmed as the only viable strategy

### Tertiary (LOW confidence — inference)
- CUDA/Metal backend spinor behavior under floating-point reordering — GPU implementations typically show 2-5 ULP divergence relative to sequential CPU for Rys quadrature; not yet measured for spinor path on any backend

---
*Research completed: 2026-04-04*
*Ready for roadmap: yes*
