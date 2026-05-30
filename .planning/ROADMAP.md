# Roadmap

## Phases

- [x] **Phase 1: Manifest & Planner Foundation** - Lock down typed domain models, manifest registry, and planner scaffolding so everything else has a deterministic catalog to build against. (v1.0, completed 2026-03-21)
- [x] **Phase 2: Execution & Compatibility Stabilization** - Wire the CubeCL-backed planner to the raw compat layer, including helper/legacy transforms, workspace queries, typed errors, and shape/optimizer guarantees. (v1.0, completed 2026-03-26)
- [x] **Phase 3: Safe Surface, C ABI Shim & Optional Families** - Layer the safe Rust facade, optional C shim, and feature-gated optional families on the stabilized runtime. (v1.0, completed 2026-03-28)
- [x] **Phase 4: Verification & Release Automation** - Close the manifest/oracle loop with CI, benchmarks, and diagnostics that block regressions before release. (v1.0, completed 2026-03-31)
- [x] **Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend)** - Replace synthetic execution with a real wgpu-backed CubeCL path and capability-aware fail-closed verification. (v1.0, completed 2026-04-02)
- [x] **Phase 6: Fix raw eval staging retrieval and capability fingerprint propagation** - Close audit gaps: wire eval_raw() staging output retrieval, propagate wgpu fingerprint into capability token, add regression tests. (v1.0, completed 2026-04-05)
- [x] **Phase 7: Executor Infrastructure Rewrite** - Replace executor internals with direct CubeCL client API, introduce ResolvedBackend dispatch, CPU backend feature, and f64 strategy decision. (v1.1, completed 2026-04-05)
- [x] **Phase 8: Gaussian Primitive Infrastructure and Boys Function** - Build shared math foundation as `#[cube]` functions: Boys function, Rys roots/weights, primitive pair evaluation, and Obara-Saika recurrence. (v1.1, completed 2026-04-05)
- [x] **Phase 9: 1e Real Kernel and Cart-to-Sph Transform** - Implement real overlap, kinetic, and nuclear attraction kernels with correct Condon-Shortley c2s transform, validating the end-to-end compute pipeline. (v1.1, completed 2026-04-05)
- [x] **Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure** - Implement all remaining integral family kernels and close the oracle parity gate for all five base families. (v1.1, completed 2026-04-05)
- [x] **Phase 11: Helper/Transform Completion & 4c1e Real Kernel** - Wire all helper, transform, and wrapper symbols to oracle CI; replace the 4c1e stub with real Rys quadrature within the Validated4C1E envelope. (completed 2026-04-05)
- [x] **Phase 12: Real Spinor Transform (c2spinor Replacement)** - Rewrite c2spinor.rs with correct Clebsch-Gordan coupling; unblock spinor oracle coverage for all families that depend on it. (completed 2026-04-05)
- [x] **Phase 13: F12/STG/YP Kernels** - Implement STG and YP geminal 2e kernels with separate dispatch paths, PTR_F12_ZETA env plumbing, and sph-only oracle gate under the with-f12 profile. (completed 2026-04-05)
- [x] **Phase 14: Unstable-Source-API Families** - Implement origi, grids, Breit, origk, and ssc families behind the unstable-source-api gate with oracle parity in nightly CI. (completed 2026-04-05)
- [x] **Phase 15: Oracle Tolerance Unification & Manifest Lock Closure** - Audit every family's empirical precision floor, set per-family atol/rtol constants, regenerate the four-profile manifest lock, and close the unified oracle CI gate. (completed 2026-04-06)
- [x] **Phase 16: Multi-Backend Support (cuda / rocm / metal) with Feature + Env-Var Selection** - Add additive Cargo feature flags for cuda, rocm (cubecl-hip), and metal alongside the existing wgpu and unconditional cpu backends; wire `CINTX_BACKEND` env-var runtime selection across compiled-in backends with hard-error on missing-feature mismatch. (completed 2026-05-09)
- [x] **Phase 17: Real-Integral Evaluation in Safe API (issue #11 Task 3)** - Replace the synthetic `(idx + 1)` / `((idx + 1) * 0.5)` pattern in `SessionRequest::fill_staging_values` with real `cintx-compat::raw::eval_raw` dispatch so the safe API delivers byte-identity values against libcint for every arity-2 intor it already accepts. No public API change. (completed 2026-05-11)
- [ ] **Phase 18: SessionRequest Arity ≥3 Dispatch (issue #11 Task 2)** - Extend `SessionRequest::evaluate` to dispatch arity-3 and arity-4 shell tuples (covering `int2e_*`, `int3c1e*`, `int3c2e_*`, `int4c1e_*`) through the existing operator catalog with F-order AO layout matching libcint memory layout.
- [x] **Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator (issue #11 Task 1)** (completed 2026-05-20) - Implement Type-1 (Coulomb-like) and Type-2 (spin-orbit-like) ECP projectors and expose them through `SessionRequest` alongside ordinary one-electron operators. Cu/LANL2DZ in the oracle corpus provides a byte-identity gate against libcint.
- [x] **Phase 20: Generic Float Precision (f64/f32 Switch)** - Parameterize the cintx compute path (CubeCL kernels, shared `#[cube]` math, staging buffers, safe-API outputs) over a generic `F: Float` so callers pick f64 (default, byte-identity) or f32 (loose-tolerance, unlocks non-`SHADER_F64` GPUs) via `evaluate::<F>()`. Raw compat `env`/`atm`/`bas` and the C ABI shim stay f64. Milestone-sized cross-cutting refactor (~3,396 f64 sites, 8 crates) planned as a single phase per operator decision 2026-05-20. (8/8 plans executed; verification gaps_found 2026-05-21; gap-closure plans 20-09..20-11 added — PREC-02 Complex<F> + PREC-05 f32 multi-component/f12, see 20-VERIFICATION.md) (completed 2026-05-21)
- [x] **Phase 21: Plain-Coulomb Gradient Integral Families (`ip1`/`iprinv`)** - Implement the 6 plain-Coulomb first-derivative integral families every HF/DFT/MP2/CCSD analytical gradient needs (`int2e_ip1`, `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`, `int1e_iprinv`, `ECPscalar_iprinv`), byte-identical to libcint 6.1.3 under the oracle gate, and repair the registered-but-stubbed `int3c2e_ip1`. Adds the missing `PTR_RINV_ORIG` env slot. Un-gates pyscf_rs Phase 7's analytical-gradient numeric arms with zero pyscf_rs rework. (completed 2026-05-26)
- [x] **Phase 22: Gauge-Origin Env Slot (Gap A — `PTR_COMMON_ORIG`)** - Plumb the `PTR_COMMON_ORIG` gauge-origin env slot (env[1..3]) end-to-end on the `PTR_RINV_ORIG` precedent and add the non-zero gauge-origin oracle fixture that gates all moment + GIAO parity. (v1.4) (completed 2026-05-29)
- [x] **Phase 23: Group 1 — Remaining 1st-Derivative Families (cart/sph)** - The 8 remaining first-derivative families (`int2e_ip2`, `int1e_ip*ip`, `int3c1e_ip1/iprinv`, `int2c2e_ip1/ip2`, `int3c2e_ip2`) at byte-identity, reusing the Phase-21 nabla/`gout_ip1` engine. (v1.4) (completed 2026-05-30)
- [ ] **Phase 24: Group 3 — Position / Multipole-Moment Integrals** - Dipole through hexadecapole moments (`int1e_r/rr/rrr/rrrr`, `r2/r4`, `z/zz`, `p4`, `rinv/drinv`, `irp`) plus `_origj` variants, gated on the non-zero gauge-origin fixture. (v1.4)
- [ ] **Phase 25: Group 2 — Hessian & Higher-Order Derivatives** - 2nd/3rd/4th-order derivative families (`int1e_ipip*`, `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, the promoted 2e Hessian set, 4th-order families) at component_rank 9/27/81, with the Rys `nroots>=6` Wheeler fallback and fail-closed high-rank staging landing first. (v1.4)
- [ ] **Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals (complex)** - Spin-free 1e+2e GIAO/CG families (purely imaginary, even in cart/sph) at byte-identity, introducing the complex-interleaved output capability. (v1.4)
- [ ] **Phase 27: Spinor-Derivative Transform (Gap B1)** - `cart_to_spinor_sf_derivative_*` so `ip`-decorated spinor families move from `UnsupportedApi` to byte-identity, closing the Phase-21 R5/D-03 deferral. (v1.4)
- [ ] **Phase 28: Spin-Included `c2s_si` Transform + σ·p Module (Gap B2)** - The 4-block (`gc_x/y/z/1`) spin-included spinor transform plus the σ·p G-tensor assembler, validated against a kappa-bearing relativistic fixture — the prerequisite for all σ-operator families. (v1.4)
- [ ] **Phase 29: Group 4 — Relativistic Spin-Operator Integrals (spinor)** - The relativistic σ-operator families (`spsp`, `spnucsp`, `sprinvsp`, `srsr`, `sigma`, `sp`, 2e `spsp1/srsr1/ssp*/sps*/vsp*`) at spinor byte-identity via the Gap B2 `c2s_si` path. (v1.4)
- [ ] **Phase 30: Group 5 (GIAO×σ slice) — Spin-GIAO Integrals (spinor)** - The relativistic-NMR GIAO×σ slice (`int1e_spg*`, `spgnucsp`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`) at spinor byte-identity, completing the magnetic-property suite. (v1.4)
- [ ] **Phase 31: Group 6 — Gauge / Breit–Gaunt 2e + Full-Parity Verification (apex)** - The Dirac–Coulomb–Breit 2e set (`int2e_gauge_r1/r2_*`, Gaunt `ssp/sps`) at spinor byte-identity AND the milestone-closing full-parity gate: every libcint 6.1.3 family `oracle_covered=true` with an empty unsupported-families list. (v1.4)

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| Phase 1: Manifest & Planner Foundation | v1.0 | 2/2 | Complete | 2026-03-21 |
| Phase 2: Execution & Compatibility Stabilization | v1.0 | 7/7 | Complete | 2026-03-26 |
| Phase 3: Safe Surface, C ABI Shim & Optional Families | v1.0 | 6/6 | Complete | 2026-03-28 |
| Phase 4: Verification & Release Automation | v1.0 | 7/7 | Complete | 2026-03-31 |
| Phase 5: Re-implement detailed-design GPU path | v1.0 | 5/5 | Complete | 2026-04-02 |
| Phase 6: Fix raw eval staging & fingerprint | v1.0 | 2/2 | Complete | 2026-04-05 |
| Phase 7: Executor Infrastructure Rewrite | v1.1 | 3/3 | Complete | 2026-04-05 |
| Phase 8: Gaussian Primitive Infrastructure and Boys Function | v1.1 | 4/4 | Complete | 2026-04-05 |
| Phase 9: 1e Real Kernel and Cart-to-Sph Transform | v1.1 | 5/5 | Complete | 2026-04-05 |
| Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure | v1.1 | 6/6 | Complete | 2026-04-05 |
| Phase 11: Helper/Transform Completion & 4c1e Real Kernel | v1.2 | 4/4 | Complete | 2026-04-05 |
| Phase 12: Real Spinor Transform (c2spinor Replacement) | v1.2 | 5/5 | Complete | 2026-04-05 |
| Phase 13: F12/STG/YP Kernels | v1.2 | 4/4 | Complete | 2026-04-05 |
| Phase 14: Unstable-Source-API Families | v1.2 | 0/5 | Planned | - |
| Phase 15: Oracle Tolerance Unification & Manifest Lock Closure | v1.2 | 0/3 | Planned | - |
| Phase 16: Multi-Backend Support (cuda / rocm / metal) | v1.2 | 4/4 | Complete | 2026-05-09 |
| Phase 17: Real-Integral Evaluation in Safe API | v1.3 | 0/3 | Planned | - |
| Phase 18: SessionRequest Arity ≥3 Dispatch | v1.3 | 0/4 | Planned | - |
| Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator | v1.3 | 8/8 | Complete | 2026-05-20 |
| Phase 20: Generic Float Precision (f64/f32 Switch) | v1.3 | 11/11 | Complete | 2026-05-21 |
| Phase 21: Plain-Coulomb Gradient Integral Families (`ip1`/`iprinv`) | v1.3 | 0/8 | Planned | - |
| Phase 22: Gauge-Origin Env Slot (Gap A — PTR_COMMON_ORIG) | v1.4 | 0/2 | Planned | - |
| Phase 23: Group 1 — Remaining 1st-Derivative Families | v1.4 | 5/5 | Complete | 2026-05-30 |
| Phase 24: Group 3 — Position / Multipole-Moment Integrals | v1.4 | 0/0 | Not started | - |
| Phase 25: Group 2 — Hessian & Higher-Order Derivatives | v1.4 | 0/0 | Not started | - |
| Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals | v1.4 | 0/0 | Not started | - |
| Phase 27: Spinor-Derivative Transform (Gap B1) | v1.4 | 0/0 | Not started | - |
| Phase 28: Spin-Included c2s_si Transform + σ·p Module (Gap B2) | v1.4 | 0/0 | Not started | - |
| Phase 29: Group 4 — Relativistic Spin-Operator Integrals | v1.4 | 0/0 | Not started | - |
| Phase 30: Group 5 (GIAO×σ slice) — Spin-GIAO Integrals | v1.4 | 0/0 | Not started | - |
| Phase 31: Group 6 — Gauge/Breit–Gaunt 2e + Full-Parity Verification | v1.4 | 0/0 | Not started | - |

## v1.2 Milestone: Full API Parity & Unified Oracle Gate

### Phase 11: Helper/Transform Completion & 4c1e Real Kernel

**Goal**: Every helper, transform, and wrapper symbol in the manifest is oracle-wired and returns libcint-compatible values; the 4c1e stub is replaced with a real Rys quadrature kernel within the Validated4C1E envelope.
**Depends on**: Phase 10
**Requirements**: HELP-01, HELP-02, HELP-03, HELP-04, 4C1E-01, 4C1E-02, 4C1E-03, 4C1E-04
**Plans**: 4/4 plans executed

Plans:

- [x] 11-01-PLAN.md — Unify tolerance constants to atol=1e-12, fix CINTgto_norm formula, add numeric helper/transform oracle comparisons.
- [x] 11-02-PLAN.md — Replace 4c1e stub with real polynomial-recurrence G-tensor kernel and fix spinor-first validation ordering.
- [x] 11-03-PLAN.md — Add workaround module, legacy wrapper numeric oracle, vendor 4c1e FFI, and close all oracle gates.
- [x] 11-04-PLAN.md — Gap closure: add cart legacy symbol vendor FFI and numeric oracle comparison for full HELP-03 coverage.

### Phase 12: Real Spinor Transform (c2spinor Replacement)

**Goal**: The cart-to-spinor transform applies correct Clebsch-Gordan coupling coefficients for all angular momenta up to l=4, enabling oracle-verifiable spinor outputs for every base family that supports spinor representation.
**Depends on**: Phase 11
**Requirements**: SPIN-01, SPIN-02, SPIN-03, SPIN-04
**Plans**: 5/5 plans executed

Plans:

- [x] 12-01-PLAN.md — Extract CG coefficient tables from libcint cart2sph.c, implement four spinor transform variants, rewire compat entry points.
- [x] 12-02-PLAN.md — Add vendor FFI wrappers for 1e spinor integrals and oracle parity gate test.
- [x] 12-03-PLAN.md — Add vendor FFI wrappers for multi-center spinor integrals and oracle parity gate tests.
- [x] 12-04-PLAN.md — Gap closure: implement multi-center spinor transforms and wire Spinor arms in 2e, 2c2e, 3c2e kernel launchers.
- [x] 12-05-PLAN.md — Gap closure: un-ignore multi-center spinor oracle parity tests and verify end-to-end.

### Phase 13: F12/STG/YP Kernels

**Goal**: STG and YP geminal two-electron kernels are implemented as separate dispatch paths with PTR_F12_ZETA env plumbing, covering all 10 with-f12 sph symbols at oracle parity.
**Depends on**: Phase 12
**Requirements**: F12-01, F12-02, F12-03, F12-04, F12-05
**Plans**: 4/4 plans executed

Plans:

- [x] 13-01-PLAN.md — Port CINTstg_roots math, add InvalidEnvParam error, update manifest canonical_family, extend ExecutionPlan, wire f12 dispatch.
- [x] 13-02-PLAN.md — Implement 10 F12 kernel entry points (5 STG + 5 YP) with distinct weight post-processing and raw compat zeta plumbing.
- [x] 13-03-PLAN.md — Add vendor FFI, oracle parity tests for all 10 symbols at atol=1e-12, zeta=0 rejection test, mark oracle_covered.
- [x] 13-04-PLAN.md — Gap closure: implement multi-component sph transform for F12 derivative operators and replace idempotency tests with oracle parity.

### Phase 14: Unstable-Source-API Families

**Goal**: All unstable-source families — origi, grids, Breit, origk, and ssc — are fully implemented behind the unstable-source-api gate with oracle parity at atol=1e-12 in nightly CI.
**Depends on**: Phase 13
**Requirements**: USRC-01, USRC-02, USRC-03, USRC-04, USRC-05, USRC-06
**Success Criteria** (what must be TRUE):

  1. `int1e_r2_origi` and `int1e_r4_origi` (origi family, 4 symbols total) are implemented behind `#[cfg(feature = "unstable-source-api")]` and pass oracle parity at atol=1e-12 (USRC-01).
  2. `int1e_grids` family is implemented with correct `NGRIDS`/`PTR_GRIDS` env slot parsing and coordinate upload; oracle parity passes at atol=1e-12 (USRC-02).
  3. Breit family (`int2e_breit_r1p2`, `int2e_breit_r2p2`) is fully implemented behind the unstable-source-api gate and passes oracle parity at atol=1e-12 (USRC-03).
  4. `int3c1e_r*_origk` variants (origk family, 6 symbols) are implemented behind the unstable-source-api gate and pass oracle parity at atol=1e-12 (USRC-04).
  5. ssc family (`int3c2e_ssc`) is fully implemented behind the unstable-source-api gate and passes oracle parity at atol=1e-12 (USRC-05).
  6. Nightly CI runs the oracle with `--include-unstable-source=true` and reports 0 mismatches for all unstable-source symbols (USRC-06).

**Plans**: 5 plans

Plans:

- [x] 14-01-PLAN.md — Infrastructure: feature gates, manifest entries, GridsEnvParams, oracle build/FFI/scaffold, kernel dispatch stubs.
- [x] 14-02-PLAN.md — Implement origi, origk, and ssc kernels with oracle parity tests (11 symbols).
- [x] 14-03-PLAN.md — Implement grids kernel with NGRIDS handling and oracle parity tests (5 symbols).
- [x] 14-04-PLAN.md — Implement Breit composite kernel (Gaunt+gauge) with spinor oracle parity tests (2 symbols).
- [ ] 14-05-PLAN.md — CI nightly job, xtask profile validation fix, manifest lock regeneration.

### Phase 15: Oracle Tolerance Unification & Manifest Lock Closure

**Goal**: Every family passes oracle at the unified atol=1e-12 threshold; the four-profile manifest lock is regenerated after oracle parity is confirmed; and every `stability: Stable` manifest entry has `oracle_covered: true` with a passing CI record.
**Depends on**: Phase 14
**Requirements**: ORAC-01, ORAC-02, ORAC-03, ORAC-04
**Success Criteria** (what must be TRUE):

  1. The single oracle tolerance constant in `compare.rs` is atol=1e-12 for every family — no per-family exceptions, no design-doc overrides. Any family that fails at 1e-12 is treated as a kernel bug to be fixed, not a tolerance to be loosened (ORAC-01).
  2. All families — 1e, 2e, 2c2e, 3c1e, 3c2e, 4c1e, F12/STG/YP, and all unstable-source families — pass oracle at atol=1e-12. No existing base family regresses from the tolerance tightening (ORAC-04).
  3. `compiled_manifest.lock.json` is regenerated for all four profiles (base, with-f12, with-4c1e, with-f12+with-4c1e) after oracle parity is confirmed — not before; `manifest-audit` CI gate passes with zero diff (ORAC-02).
  4. CI oracle-parity gate passes all four profiles at atol=1e-12 under `--features cpu` with `mismatch_count == 0`; every `stability: Stable` manifest entry has `oracle_covered: true` (ORAC-03).

**Plans**: 3 plans

Plans:

- [x] 15-01-PLAN.md — Refactor tolerance_for_family to catch-all and replace PHASE4_ORACLE_FAMILIES with manifest-driven derivation.
- [x] 15-02-PLAN.md — Create oracle-covered-update xtask, add oracle_covered check to manifest-audit, stamp and regenerate lock.
- [x] 15-03-PLAN.md — Switch CI oracle_parity_gate to matrix strategy over four profiles.

### Phase 16: Multi-Backend Support (cuda / rocm / metal) with Feature + Env-Var Selection

**Goal**: `cintx-cubecl` exposes additive Cargo features `cuda`, `rocm`, `metal` alongside the existing `wgpu`, with `cpu` becoming unconditionally compiled; the `CINTX_BACKEND` env var selects among compiled-in backends at runtime, defaulting to `cpu`, and requesting an un-compiled backend produces a typed hard error rather than a silent fallback.
**Depends on**: Phase 15
**Requirements**: BACK-01, BACK-02, BACK-03, BACK-04, BACK-05, BACK-06, BACK-07 (derived from success criteria 1-7 during `/gsd:plan-phase`).
**Success Criteria** (what must be TRUE):

  1. `cintx-cubecl/Cargo.toml` exposes additive features `cuda`, `rocm`, `metal`, `wgpu`; `cpu` is unconditional (no longer a feature flag) and the `cubecl-cpu` runtime is always linked.
  2. `BackendKind` (in `cintx-runtime`) and `ResolvedBackend` (in `cintx-cubecl`) are extended with `Cuda`, `Rocm`, and `Metal` variants, each gated behind its respective `#[cfg(feature = "...")]`.
  3. `cargo check` with every non-empty subset of `{cuda, rocm, metal, wgpu}` builds cleanly on this dev host (Linux, AMD ROCm); `cargo test --features cpu` (the default) passes the existing oracle suite with zero mismatches.
  4. `cargo test --features rocm` runs at least one oracle smoke test under `CINTX_BACKEND=rocm` on AMD/ROCm-capable hosts and matches existing parity tolerances.
  5. `CINTX_BACKEND=<name>` selects the named backend at runtime when its feature is enabled; an unset env var resolves to `cpu`; a value naming a backend whose feature is **not** compiled in returns a typed `cintxRsError` (`UnsupportedApi` or new `BackendNotCompiled` variant) — never a silent fallback.
  6. `cuda` and `metal` are documented in module-level docs as **"compile-only on this host; runtime behavior delegated to upstream cubecl"** with a link to `notes/cuda-metal-verification-gap.md`. No oracle parity gate is added for cuda or metal in this phase.
  7. The feature matrix is exercised in CI (at minimum `cargo check` per feature combo on the existing CI runners — no new GPU runners required for this phase).

**Plans**: 4 plans
Plans:

- [ ] 16-01-PLAN.md — Wave 0: migration audit (D-12, 30 callsites) + BackendNotCompiled error variant + CINTX_STATUS_BACKEND_NOT_COMPILED C-ABI code + CHANGELOG pre-announce.
- [ ] 16-02-PLAN.md — Wave 1: Cargo feature wiring (cuda/rocm/metal/wgpu per D-07 with M1 metal-as-wgpu-alias amendment) + per-backend modules + per-variant cfg gating on BackendKind/ResolvedBackend (D-10) + fallible resolve_backend_kind (D-03) + BackendIntent::default flip to Cpu (D-11) + compiled_backends() introspection helper.
- [ ] 16-03-PLAN.md — Wave 2: feature_matrix_gate CI job (3-cell: cpu-only / cpu+wgpu / all-features) added inside compat-governance-pr.yml; ROCm install on the all-features cell; manual branch-protection registration of three new required-status-check entries.
- [ ] 16-04-PLAN.md — Wave 3: ROCm full base-family oracle suite (7 #[ignore]'d tests across 5 files at atol=1e-12, env-gated by CINTX_ROCM_ORACLE=1) + xtask rocm-oracle wrapper command + cuda/metal verification-gap doc citations (BACK-06 closure). No CI gate added (D-15).

**Notes**:

  - See `.planning/notes/cuda-metal-verification-gap.md` for the explicit risk-accept on cuda/metal runtime verification.
  - See `.planning/seeds/gpu-ci-runners.md` for the follow-up that would close the gap when hardware/CI is available.
  - Open research question to resolve before planning: do `cubecl-cuda 0.10.0`, `cubecl-hip 0.10.0`, `cubecl-metal 0.10.0`, and `cubecl-wgpu 0.10.0` resolve cleanly together? See `.planning/research/questions.md`.

## v1.3 Milestone: Safe API Closure for pyscf_rs Consumer

The three v1.3 phases close the remaining gaps between cintx's safe API and what
pyscf_rs's `pyscf-gto` crate needs to drive real workflows. Source: issue #11
(https://github.com/BectorVoom/cintx/issues/11). Downstream-consumer context:
`.planning/notes/pyscf-rs-as-cintx-consumer.md`.

Ordering rationale: phases are listed in dependency order (17 → 18 → 19) but
the issue author flagged the tasks as independent. The ordering reflects
verification convenience — Phase 17 lands real values so Phase 18's arity-3/4
dispatch can be oracle-checked byte-for-byte; Phase 19's ECP dispatch flows
through the same safe-API path that Phase 17 made real. During `/gsd:plan-phase`,
the order may be relaxed to parallelize if the planner judges the risk acceptable.

### Phase 17: Real-Integral Evaluation in Safe API

**Goal**: `SessionRequest::evaluate` returns real libcint-compatible values for every arity-2 intor it accepts today — the synthetic `(idx + 1)` / `((idx + 1) * 0.5)` placeholder in `fill_staging_values` is replaced with a real `cintx-compat::raw::eval_raw` dispatch under the hood. No public API change; internal evaluator swap only.
**Depends on**: Phase 16
**Closes**: issue #11 Task 3
**Requirements**: RVAL-01, RVAL-02, RVAL-03 (derived from success criteria 1-3 during `/gsd:spec-phase` or `/gsd:plan-phase`).
**Success Criteria** (what must be TRUE):

  1. `crates/cintx-rs/src/api.rs::fill_staging_values` invokes `cintx-compat::raw::eval_raw` (or an equivalent compat dispatch) for every operator the safe API currently routes; no synthetic-pattern fallback remains in the arity-2 path. (RVAL-01)
  2. `cintx-oracle/tests/one_electron_parity.rs` is extended (or a sibling test added) that drives every supported arity-2 intor through `SessionRequest::evaluate` and asserts byte-identity against libcint at the unified `atol=1e-12` Phase 15 tolerance. (RVAL-02)
  3. No public API change in `cintx-rs`: `SessionRequest` constructors, accessors, and error types stay source- and SemVer-compatible with v1.2. (RVAL-03)

**Plans**: 3 plans

Plans:
**Wave 1**

- [x] 17-01-PLAN.md — Wave 0: add `cintx-rs` path-dep to `crates/cintx-oracle/Cargo.toml` so the new parity test can call `SessionRequest::evaluate`.
- [x] 17-02-PLAN.md — Wave 1: delete the synthetic stub `CubeClExecutor` + `fill_staging_values` in `crates/cintx-rs/src/api.rs`; add `use cintx_cubecl::CubeClExecutor;`; rewrite the brittle `owned_values[0] == 1.0` unit test to deterministic + nonzero + invariants.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 17-03-PLAN.md — Wave 1: add `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (12 per-symbol tests at atol=1e-12: 8 cart/sph vendor-parity + 4 spinor idempotency).

**Notes**:

  - Smallest of the three issue #11 tasks; explicitly callable as an isolated PR.
  - Downstream impact: unblocks every arity-2 intor in pyscf_rs `pyscf-gto/src/intor.rs` immediately on land.

### Phase 18: SessionRequest Arity ≥3 Dispatch ✓ Complete (2026-05-12)

**Goal**: `SessionRequest::evaluate` dispatches arity-3 and arity-4 shell tuples through the existing operator catalog, returning tensors with F-order AO axes that match libcint memory layout. Covers `int2e_*` (the SCF J/K hot path), `int3c1e`, `int3c1e_p2`, `int3c2e_ip1`, `int3c2e_sph`, `int3c2e_cart`, `int4c1e_sph`, and `int4c1e_cart`.
**Depends on**: Phase 17
**Closes**: issue #11 Task 2
**Requirements**: ARITY-01, ARITY-02, ARITY-03, ARITY-04, ARITY-05 (derived from success criteria 1-5 during `/gsd:spec-phase` or `/gsd:plan-phase`).
**Success Criteria** (what must be TRUE):

  1. `SessionRequest::evaluate` accepts arity-3 shell tuples `(i, j, k)` and arity-4 tuples `(i, j, k, l)` and routes them to the existing `cintx-ops` resolver entries (`cintx-ops/src/resolver.rs:316`, `cintx-ops/src/generated/api_manifest.csv:21-25`) — no parallel evaluator API is introduced. (ARITY-01)
  2. Arity-3 operators `int3c1e`, `int3c1e_p2`, `int3c2e_ip1`, `int3c2e_sph`, `int3c2e_cart` and arity-4 operators `int2e_sph`, `int2e_cart`, `int4c1e_sph`, `int4c1e_cart` round-trip through the safe API with byte-identity values against libcint at `atol=1e-12`. (ARITY-02)
  3. Output tensors expose F-order AO axes consistent with libcint memory layout so downstream consumers (notably pyscf_rs `pyscf-gto/src/intor.rs`) can treat the safe API as a drop-in alternative to the raw `eval_raw` path. (ARITY-03)
  4. Two-electron symmetry packing follows pyscf's `aosym` convention (`s1`, `s2ij`, `s2kl`, `s4`, `s8`) where supported, or returns a typed error documenting which packings are not yet implemented. (ARITY-04)
  5. Oracle parity tests for arity-3 and arity-4 dispatch are added to `cintx-oracle` and gate CI alongside the existing arity-2 parity tests. (ARITY-05)

**Plans**: 4 plans

Plans:
**Wave 1**

- [x] 18-01-PLAN.md — Wave 0: manifest expansion (R1: add plain int3c2e_{cart,sph}) + AoSymmetry enum + ExecutionOptions::aosym + re-exports + 3 new vendor FFI wrappers + resolver misc_wrapper_macro arm.
- [x] 18-02-PLAN.md — Wave 1: safe-API surface (aosym preflight in query_workspace, FacadeError::UnsupportedAoSymmetry variant + kind() arm, F-order rustdoc on IntegralTensor, INT4C1E_CART_OPERATOR_ID shift 22 -> 24, two aosym unit tests).

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 18-03-PLAN.md — Wave 2: 8 arity-3 oracle parity tests against vendored libcint at atol=1e-12 (cart/sph for int3c1e, int3c1e_p2, int3c2e_ip1, plain int3c2e).
- [x] 18-04-PLAN.md — Wave 2: 4 arity-4 oracle parity tests against vendored libcint at atol=1e-12 (int2e_{cart,sph} base; int4c1e_{cart,sph} per-test with-4c1e gated).

**Outcome (2026-05-12):** 10/12 oracle parity tests pass at atol=1e-12. SC#1, SC#4, SC#5 fully verified. SC#2/SC#3 verified for 10 of 12 operator/representation pairs. Deferred (out of phase): int3c1e_p2_{cart,sph} kernel divergence from vendored libcint (~1e-2 to 1e-4) — pre-dates Phase 18, tracked in 18-HUMAN-UAT.md Gap 2 for a /gsd:debug session.

**Cross-cutting constraints:**

- Per-symbol nonzero sentinel (`any_nonzero` flag asserted after the sweep) guards against zero-fill regressions per PATTERNS.md §Shared Patterns.

**Notes**:

  - `int2e_sph` is the single hottest call in any closed-shell SCF workflow — closing this gap is the biggest practical unblocker for pyscf_rs's `pyscf-scf` crate.
  - `int3c2e_*` enables density fitting (pyscf_rs `pyscf-df` crate).

### Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator

**Goal**: cintx implements Type-1 (Coulomb-like) and Type-2 (spin-orbit-like) ECP projectors and exposes them through `SessionRequest` alongside ordinary one-electron operators. Symbols delivered: `int1e_ecp_sph`, `int1e_ecp_cart`, and gradient variants `int1e_ecp_ipnuc_sph`/`int1e_ecp_ipnuc_cart` (the gradient variants land in this phase only if Phase 19 does not need a separate gradient-layer prerequisite phase; otherwise they defer to a follow-up).
**Depends on**: Phase 18
**Closes**: issue #11 Task 1
**Requirements**: ECP-01, ECP-02, ECP-03, ECP-04, ECP-05 (derived from success criteria 1-5 during `/gsd:spec-phase` or `/gsd:plan-phase`).
**Success Criteria** (what must be TRUE):

  1. Type-1 (local, Coulomb-like) ECP projector evaluation is implemented in cintx (as a CubeCL `#[cube]` kernel or compat-side path consistent with the Phase 8-10 evaluator pattern) and registered in the operator catalog (`cintx-ops`). (ECP-01)
  2. Type-2 (semi-local, spin-orbit-like) ECP projector evaluation with correct spherical-harmonic angular projectors and Bessel-modulated radial integrals is implemented and registered in the operator catalog. (ECP-02)
  3. `SessionRequest::evaluate` dispatches `int1e_ecp_sph` and `int1e_ecp_cart` through the same safe-API surface as ordinary one-electron operators — no parallel ECP API is introduced. (ECP-03)
  4. Cu/LANL2DZ (already present in the oracle test corpus) passes byte-identity parity against libcint at `atol=1e-12` through both `cintx-compat::raw::eval_raw` and `SessionRequest::evaluate`. Secondary cross-check against `libECP` (chrr, JCC 2017) is added as a non-blocking oracle. (ECP-04)
  5. Decision on whether `int1e_ecp_ipnuc_*` gradient variants land in this phase or defer to a follow-up phase is recorded in the phase's SPEC.md before plan-phase starts. (ECP-05)

**Plans**: 8 plans (19-01..04 executed; 19-05/06 stale plans superseded after the 2026-05-20 K-Taylor byte-identity replan; new 19-05..08 added per D-13..D-17)

Plans:
**Wave 0**

- [x] 19-01-PLAN.md — Wave 0: vendor PySCF nr_ecp subtree (Apache-2.0); extend cintx-oracle/build.rs with parallel cc::Build + has_vendor_pyscf_nr_ecp cfg; expand api_manifest.csv with 4 ECP rows (cart/sph × {ecp, ecp_ipnuc}) + regenerate lock; build Cu/LANL2DZ fixture; land empty stubs for bessel.rs, radial_quadrature.rs, EcpShell. (Completed 2026-05-12; SUMMARY: `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-01-SUMMARY.md`; new OperatorIds 26..=29; INT4C1E_CART_OPERATOR_ID=24 preserved.)

**Wave 1** *(blocked on Wave 0 completion; 02 and 03 ran in parallel)*

- [x] 19-02-PLAN.md — Wave 1: implement modified spherical Bessel i_l(x) in math/bessel.rs (paired #[cube] + *_host(), three numerical branches: small-x Taylor, moderate-x direct Taylor sum, large-x asymptotic — direct evaluation, no recurrence) + Gauss-Chebyshev + Gauss-Hermite nodes/weights in math/radial_quadrature.rs. (Completed 2026-05-12; SUMMARY: `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-02-SUMMARY.md`.)
- [x] 19-03-PLAN.md — Wave 1: EcpShell + BasisSet::ecp_shells extension + OperatorId::INT1E_ECP_* constants + is_ecp() helper + cintx-compat::raw ECP slot constants + EcpBasArray typed view + eval_raw ECP dispatch arm + FacadeError::MissingEcpBasis variant + SessionRequest::query_workspace preflight. (Completed 2026-05-12; SUMMARY: `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-03-SUMMARY.md`.)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 19-04-PLAN.md — Wave 2: launch_ecp kernel scaffolding (canonical_family "ecp" registered; vendor_ECPscalar_{cart,sph} FFI; safe_api_ecp_parity.rs harness). Shipped a DIRECT-QUADRATURE approximation that does NOT reach atol=1e-12; parity tests #[ignore]'d, oracle_covered stayed false. (Completed 2026-05-12; SUMMARY: `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-04-SUMMARY.md` — the authoritative blocker write-up.)

> **REPLAN 2026-05-20 (K-Taylor byte-identity port).** 19-04's direct-quadrature kernel cannot reach byte-identity. The original 19-05 (gradient) was halted and the original 19-06 (libecpint) was unstarted; both PLAN.md files are preserved as `.superseded.md`. The four plans below (19-05..08) replace them per D-13..D-17. 19-01..04 are frozen.

**Wave 1 (replan) — K-Taylor port foundation**

- [x] 19-05-PLAN.md — K-Taylor port foundation: embed _sph_ine_tab (400×24) + _sph_ine_tab_order7 (400×8×8) as LE-f64 binary blobs via include_bytes! + bytemuck (D-14, roots_xw_data.rs precedent); xtask gen-ecp-tables subcommand extracting the tables from nr_ecp.c + a --check drift-gate (D-15); host-first port of ECPsph_ine_opt + ECPrad_part + type1_rad_part + type2_facs_rad (D-16). [ECP-01, ECP-02, ECP-04]

**Wave 2 (replan) — scalar close**

- [x] 19-06-PLAN.md — Scalar close: replace the direct-quadrature compute_type1_pair/compute_type2_pair bodies with the K-Taylor recurrence; remove #[ignore] from the two scalar parity tests; iterate to atol=1e-12 byte-identity over the full Cu/LANL2DZ Cartesian product; flip oracle_covered=true on int1e_ecp_{cart,sph}. Closes ECP-01/02/03 + scalar half of ECP-04. [ECP-01, ECP-02, ECP-03, ECP-04] (depends 19-05)

**Wave 3 (replan) — gradient (supersedes halted 19-05)**

- [x] 19-07-PLAN.md — Gradient: port nr_ecp_deriv.c (_deriv1_cart, comp=3) for int1e_ecp_ipnuc_{cart,sph} on the K-Taylor foundation; F-order [axis, ao_j, ao_i] axis-slowest (D-11); vendor_ECPscalar_ipnuc_{cart,sph} wrappers; two ipnuc parity tests at atol=1e-12; flip oracle_covered=true on the two ipnuc rows. Closes ECP-05 + gradient half of ECP-04. [ECP-01, ECP-02, ECP-04, ECP-05] (depends 19-06)

**Wave 4 (replan) — optional secondary oracle (supersedes 19-06)**

- [x] 19-08-PLAN.md — Optional non-blocking libecpint (Shaw & Hill, JCP 147 074108, 2017, MIT) secondary cross-check behind has_libecpint_oracle cfg (emitted only when CINTX_LIBECPINT_ORACLE=1); env-gated #[ignore] cross-check tests at informational atol≈1e-9 per D-02 REVISED. [ECP-04] (depends 19-06)

**Notes**:

  - Largest of the three issue #11 tasks — requires a new evaluator implementation, not just dispatch wiring.
  - libcint's upstream ECP code was reportedly copied from PySCF; cross-reference against pyscf's `pyscf/gto/ecp.py` may be useful during planning.
  - Reference implementation outside libcint: https://github.com/chrr/libECP (libECP, JCC 2017).
  - Downstream marker: pyscf_rs `crates/pyscf-gto/src/ecp_engine_stub.rs` is the placeholder that this phase replaces.

### Phase 20: Generic Float Precision (f64/f32 Switch)

**Goal**: cintx parameterizes its compute path over a generic float type `F: Float` so callers evaluate integrals in f64 (default, byte-identity) or f32 (loose-tolerance, unlocks adapters lacking `SHADER_F64`). Precision is chosen at the call site via a method-level generic `evaluate::<F>()`; `evaluate()` continues to mean f64 and every existing call site compiles unchanged. The full compute path — CubeCL kernels, shared `#[cube]` math (Boys / Rys / Obara-Saika), staging buffers, and safe-API outputs — threads `F`. Raw compat `env`/`atm`/`bas` arrays and the C ABI shim stay f64.
**Depends on**: Phase 7 (the "both backends produce f64-precision results" strategy this relaxes — `07-CONTEXT.md` D-09), Phases 8-10 (the shared `#[cube]` math + per-family kernels being genericized), Phase 15 (per-family tolerance model the f32 oracle gate mirrors), Phase 16 (`SHADER_F64` capability-gating context for D-10).
**Requirements**: PREC-01, PREC-02, PREC-03, PREC-04, PREC-05, PREC-06, PREC-07 (derived from the milestone-level decisions in `20-CONTEXT.md` D-01..D-12).
**Scope note**: Milestone-sized (~3,396 `f64` sites across 8 crates). Discuss-phase recommended a separate v1.4 milestone; operator decision (2026-05-20) is to plan and execute the full scope as a single Phase 20 in v1.3. Decisions in `20-CONTEXT.md` are milestone-level and authoritative.
**Success Criteria** (what must be TRUE):

  1. A generic float type `F: Float` is threaded through the full compute path — CubeCL kernels become `#[cube] fn ...<F: Float>(...)`, shared math (Boys/Rys/Obara-Saika), staging buffers, and safe-API outputs all parameterize on `F`; the concrete `f64` monomorphization is preserved and const f64 tables cast to `F`. (PREC-01)
  2. `SessionRequest` setup stays monomorphic; `evaluate::<F>()` is a method-level generic returning `TypedEvaluationOutput<F>` (with `owned_values: Vec<F>`, default `F = f64`); `evaluate()` delegates to `evaluate::<f64>()`; spinor/complex outputs propagate as `Complex<F>`; every existing call site compiles unchanged. (PREC-02)
  3. Raw compat `env`/`atm`/`bas` arrays and the C ABI shim (`cintx-capi`) remain f64 — the libcint ABI is untouched; precision conversion happens only at the kernel/staging boundary (host f64 `env` → device `F` buffers). (PREC-03)
  4. The f64 path keeps strict byte-identity against libcint at the existing per-family atol (~1e-12); all existing oracle gates, manifest locks, and tests pass unchanged. (PREC-04)
  5. The f32 path has a separate oracle gate at a realistic single-precision tolerance (~1e-4 rtol; per-family floors empirical, mirroring Phase 15) and is verified against libcint — just not byte-identical. (PREC-05)
  6. The f32 path unlocks the wgpu backend on adapters lacking `SHADER_F64` — it does NOT gate on the `SHADER_F64` capability that the f64 path requires (`check_shader_f64_in_features`). (PREC-06)
  7. The refactor is performed using the serena MCP server's symbol-aware tools (`find_symbol`, `find_referencing_symbols`, `rename_symbol`, `replace_symbol_body`, `insert_before/after_symbol`), not blind text replacement, so const tables and deliberately-f64 sites (env ABI per PREC-03, C ABI) are not corrupted. (PREC-07)

**Notes**:

  - Largest single phase in the project — a cross-cutting type-parameter refactor, not a new evaluator.
  - **RESEARCH FLAGS** from `20-CONTEXT.md`: (a) confirm the CubeCL `Float` trait surface covers the transcendentals the kernels need (`exp`, `sqrt`, `erf`) and that const-table casting is sound under monomorphization; (b) confirm f32 shader capability is universally available on wgpu adapters.
  - Claude's discretion: exact per-family f32 tolerance floors, helper genericization order, and whether to introduce a sealed `Scalar`/`CintFloat` super-trait bridging device-side CubeCL `Float` and host-side `num_traits::Float`.

**Plans**: 11 plans (8 executed; 3 gap-closure plans 20-09..20-11 added 2026-05-21 to close the PREC-02 Complex<F> and PREC-05 f32 multi-component/f12 gaps)

Plans:
**Wave 1**

- [x] 20-01-PLAN.md — Wave 0 scaffolding: serena onboarding gate (D-11), `CintFloat` sealed trait + `PrecisionKind` enum in cintx-core, `ExecutionPlan.precision` field, num-traits direct dep, and the A5 bytemuck staging-cast validation spike.

**Wave 2** *(parallel; blocked on Wave 1)*

- [x] 20-02-PLAN.md — Shared math leaves group A: genericize boys.rs (reference refactor), obara_saika.rs, pdata.rs, stg.rs over `F` (device `F: Float` / host `F: CintFloat`); const tables FROZEN f64.
- [x] 20-03-PLAN.md — Shared math leaves group B: genericize rys.rs (1,140 f64-lines, isolated for budget) and the c2s cart-to-sph transform over `F`; coefficient tables/blobs FROZEN f64.

**Wave 3** *(blocked on Wave 2)*

- [x] 20-04-PLAN.md — Kernel launchers group A: precision-dispatch 1e/2e/2c2e launchers (keep FamilyLaunchFn signature, `match plan.precision` -> generic `_typed::<F>` inner + bytemuck staging cast) + thread `F` through the spinor transform interleaved accumulation.

**Wave 4** *(blocked on Wave 3)*

- [x] 20-05-PLAN.md — Kernel launchers group B: precision-dispatch 3c1e/3c2e/4c1e/f12 launchers (copy the Wave-3 dispatcher pattern); `f12_zeta` stays `Option<f64>`, cast to `F` at the kernel boundary.

**Wave 5** *(blocked on Wave 4)*

- [x] 20-06-PLAN.md — Executor capability branch (f32 bypasses `SHADER_F64` via `check_capability`; f64 retains it) + thread `ExecutionOptions.precision` -> `ExecutionPlan.precision`; staging over-allocation soundness; raw env/atm/bas + C ABI stay f64.

**Wave 6** *(blocked on Wave 5)*

- [x] 20-07-PLAN.md — Safe API: `evaluate::<F: CintFloat>()` method-level generic returning `TypedEvaluationOutput<F = f64>` / `IntegralTensor<F = f64>`; `evaluate()` stays byte-identical f64; `CintFloat::PRECISION` maps `F` -> runtime tag.

**Wave 7** *(blocked on Wave 6)*

- [x] 20-08-PLAN.md — Separate f32 oracle gate: `f32_tolerance_for_family` + F32 constants (parallel to the FROZEN f64 model) + `tests/f32_parity.rs` driving `evaluate::<f32>()` with empirically derived per-family rtol floors; the f64 byte-identity gate stays untouched.


**Wave 8** *(gap closure — VERIFICATION.md gaps_found 2026-05-21; existing 20-01..20-08 complete and untouched)*

- [x] 20-09-PLAN.md — Gap 1 (PREC-02 / D-04 / SC-2): expose spinor/complex safe-API outputs as `num_complex::Complex<F>` via a `complex_values()` typed view reinterpreting the contiguous interleaved `Vec<F>`; real path unchanged; f64 oracle byte-identical. (IMPLEMENT path — no override.) **COMPLETE 2026-05-21** — num-complex 0.4 direct dep; complex_values() on IntegralTensor<F> + TypedEvaluationOutput<F>; spinor_evaluate_exposes_complex_values_some_prec02 smoke test; 31/31 cintx-rs tests green; 11/11 f64 oracle integration tests green.

**Wave 8 (parallel with 20-09; disjoint files)**

- [x] 20-10-PLAN.md — Gap 2a (PREC-05): kernel + math hardening — CR-01 (bound F32 copy/not0 to true `out_elems`, `BufferTooSmall` guard, 7 kernels), CR-02 + WR-01 (f12 `staging_f64` sized to `out_elems`, true-byte stats), WR-03 (f64-intermediate pdata), WR-04 (fail-loud `to_f64`), WR-05 (precision-appropriate Boys tol, host==device), WR-06 (f32-safe not0 sentinel); f64 integration oracle byte-identical.

**Wave 9** *(blocked on 20-10)*

- [x] 20-11-PLAN.md — Gap 2b (PREC-05): vendor-gated f32 oracle tests for a multi-component / f12-derivative operator (`int2e_stg_ip1_sph`, ncomp=3 — the CR-01/CR-02 corruption regime) driving `evaluate::<f32>()` at the empirical f32 floor; load-bearing (FAILS pre-20-10); FROZEN f64 gate untouched.

### Phase 21: Plain-Coulomb Gradient Integral Families (`ip1`/`iprinv`)

**Goal**: cintx implements the 6 plain-Coulomb first-derivative (∂/∂nuclear-coordinate) integral families that every HF/DFT/MP2/CCSD analytical gradient needs — `int2e_ip1`, `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`, `int1e_iprinv`, `ECPscalar_iprinv` — byte-identical to libcint 6.1.3 under the oracle gate, and repairs the registered-but-stubbed `int3c2e_ip1` (currently an operator-blind scalar kernel that silently returns the non-derivative integral). Adds the missing `PTR_RINV_ORIG` env slot. Landing these un-gates pyscf_rs Phase 7's analytical-gradient numeric arms (RHF/UHF/RKS/UKS/MP2/CCSD + CPHF + geomopt) with zero pyscf_rs rework — only a `workflow_dispatch` gate flips, because the dispatch shape + component-leading layout are already wired in pyscf-gto.
**Depends on**: Phase 18 (arity-≥3 `SessionRequest` dispatch — required only for the `int2e_ip1` *safe-API* path; the raw/compat `eval_raw` path already dispatches arity-4, so the raw arm can land independently). Phases 8-10 (the shared `#[cube]` Boys/Rys/Obara-Saika math + per-family kernels the gradient reuses), Phase 19 (the scalar-ECP K-Taylor byte-identity foundation `ECPscalar_iprinv` builds on — Risk R4).
**Requirements**: GRAD-01, GRAD-02, GRAD-03, GRAD-04, GRAD-05, GRAD-06, GRAD-07, GRAD-08, GRAD-09, GRAD-10
**Success Criteria** (what must be TRUE):

  1. The `PTR_RINV_ORIG` env slot (`env[4..6]`) is plumbed end-to-end following the `f12_zeta` precedent: `OperatorEnvParams.rinv_orig: Option<[f64;3]>` field, `raw.rs::eval_raw` env-read, `validator.rs` gate (an `iprinv` operator without an origin is rejected), origin threaded into the `one_electron`/`ecp` kernels, and a `with_rinv_origin`-style setter on the safe-API options; env round-trip + validator-rejection unit tests pass. (GRAD-01)
  2. All 6 gradient families plus the `int3c2e_ip1` correction are registered in `compiled_manifest.lock.json` with `"component_rank":"3"` per representation, with matching RawApiId consts, legacy wrappers, and CAPI enum variants; `cargo build` regenerates `api_manifest.rs`; the manifest-audit xtask is green and every symbol resolves through `eval_raw` (kernels may return `UnsupportedApi` until they land). (GRAD-02)
  3. `int1e_ipovlp` (cart + sph, 3 components) matches vendored libcint 6.1.3 at atol=1e-12 on the H2O/STO-3G corpus. (GRAD-03)
  4. `int1e_ipkin` (cart + sph, 3 components) matches vendored libcint 6.1.3 at atol=1e-12. (GRAD-04)
  5. `int1e_ipnuc` (cart + sph, 3 components; ∇ on the bra center, summed over all nuclei) matches vendored libcint 6.1.3 at atol=1e-12. (GRAD-05)
  6. `int1e_iprinv` (cart + sph, 3 components; single rinv origin via the GRAD-01 env slot, no `-Z_C` factor) matches vendored libcint 6.1.3 at atol=1e-12. (GRAD-06)
  7. `int2e_ip1` (arity-4, 3 components; component-leading `[3, nl, nk, nj, ni]` F-order matching pyscf-gto `layout_table.rs`) matches vendored `int2e_ip1` at atol=1e-12 for s/p/d quartets. (GRAD-07)
  8. `int3c2e_ip1` ships a real derivative kernel replacing the operator-blind scalar stub in `center_3c2e.rs`, and its oracle reference is flipped from the plain `vendor_int3c2e` to `vendor_int3c2e_ip1`; matches at atol=1e-12. (GRAD-08)
  9. `ECPscalar_iprinv` (per-nucleus ECP force; single rinv origin, no all-slot `-Z_C` accumulation) matches vendored libcint at atol=1e-12 on Cu/LANL2DZ, after confirming Phase 19's scalar-ECP K-Taylor byte-identity path (Risk R4). (GRAD-09)
  10. Phase verification + pyscf_rs hand-off: the component-leading F-order layout is validated against the vendor layout (Risk R3); cintx ROADMAP/STATE/REQUIREMENTS are updated; a hand-off note records which pyscf_rs Phase 7 `workflow_dispatch` gradient arms now un-gate. (GRAD-10)

**Plans**: 8 plans

Plans:
**Wave 1** — rinv-origin env infrastructure + manifest registration (foundation)

- [x] 21-01-PLAN.md — Wave 1: `PTR_RINV_ORIG` env-slot plumbing (the `f12_zeta` 4-step pattern): `OperatorEnvParams.rinv_orig`, `raw.rs` env-read, `validator.rs` gate, thread into `one_electron`/`ecp` kernels; `with_rinv_origin` safe-API setter; env round-trip + validator-rejects-missing-origin unit tests. [GRAD-01]
- [x] 21-02-PLAN.md — Wave 1: register all 6 families (+ `int3c2e_ip1` correction) in `compiled_manifest.lock.json` with `component_rank:"3"`; add RawApiId consts, legacy wrappers, CAPI enum variants; `cargo build` regenerates the manifest; manifest-audit xtask green; symbols resolve through `eval_raw` (UnsupportedApi from kernels until Wave 2/3). [GRAD-02]

**Wave 2** *(blocked on Wave 1)* — 1e gradient kernels (no Rys risk except ipnuc; 21-04 runs after 21-03 — both edit `one_electron.rs`, executor serializes on the `files_modified` overlap)

- [x] 21-03-PLAN.md — Wave 2: `int1e_ipovlp` + `int1e_ipkin` — `nabla1i` on the overlap/kinetic G-tensors (Obara-Saika; the `contract_kinetic` `CINTnabla1j_1e` code at `one_electron.rs:208` is the pattern). Oracle vs `vendor_int1e_ipovlp`/`ipkin` at atol=1e-12. [GRAD-03, GRAD-04]
- [x] 21-04-PLAN.md — Wave 2: `int1e_ipnuc` (∇ on bra, sum over all atoms) + `int1e_iprinv` (single origin via the Wave-1 env slot, no `-Z_C` factor). Both reuse the `gout_ip1` nabla on the nuclear Rys tensor; differ only in atom-loop vs single-origin and prefactor. Oracle vs vendor at atol=1e-12. [GRAD-05, GRAD-06]

**Wave 3** *(blocked on Wave 1)* — int2e_ip1 (Rys; also exposes `gout_ip1`/`F12Shape` as `pub(crate)` for Wave 4)

- [x] 21-05-PLAN.md — Wave 3: `int2e_ip1` — new gradient path in `two_electron.rs`: `build_2e_shape(li+1, lj, lk, ll)`, `fill_g_tensor_2e` + `rys_roots_host`, then `gout_ip1` (reused from `f12.rs`). Component-leading `[3, nl, nk, nj, ni]` F-order matching pyscf-gto `layout_table.rs`. Oracle vs `vendor_int2e_ip1` at atol=1e-12 for s/p/d. Confirm pyscf-gto's call path (raw vs safe / Phase 18) before committing the surface (Risk R6). [GRAD-07]

**Wave 4** *(blocked on Waves 1-3)* — int3c2e_ip1 repair + ECP gradient

- [x] 21-06-PLAN.md — Wave 4: `int3c2e_ip1` real derivative kernel (repair family 0) — same `gout_ip1` reuse in `center_3c2e.rs` (**depends on 21-05's `pub(crate)` exposure**). Flip oracle from plain `vendor_int3c2e` to `vendor_int3c2e_ip1`. [GRAD-08]
- [x] 21-07-PLAN.md — Wave 4: `ECPscalar_iprinv` — per-nucleus selector in `launch_ecp` (the `ipnuc` driver `deriv1_cart_pair` at `ecp.rs:1181` sums all ECP slots; iprinv selects one via the Wave-1 rinv origin) + drop the `-Z_C`/all-slot accumulation; reuse the salvaged `19-05` tables. **Pre-req: confirm scalar-ECP K-Taylor byte-identity (Risk R4).** Oracle: Cu/LANL2DZ iprinv vs vendor. [GRAD-09]

**Wave 5** *(blocked on Waves 1-4)* — verification + close-out

- [x] 21-08-PLAN.md — Wave 5: phase verification + the pyscf_rs hand-off note (which Phase 7 `workflow_dispatch` arms un-gate, `int3c2e_ip1` re-gating history); validate component-leading F-order vs vendor layout (Risk R3); update cintx ROADMAP/STATE/REQUIREMENTS. [GRAD-10]

**Risks**:

  - **R1** — `int3c2e_ip1` is a latent silent-wrong RUNTIME path (verified): `center_3c2e.rs::launch_center_3c2e_typed` is operator-blind, scalar-output, no derivative. The oracle "passes" only because it references plain `vendor_int3c2e`; pyscf_rs's DF-grad runtime consumes it as a derivative. Fixed in 21-06.
  - **R2** — Rys roots >5 for high-l: the gradient's `li+1` pushes f/g quartets past nroots=5 (unsupported, same ceiling as base int2e). Document the l-limit; gate high-l grads behind the deferred Wheeler-fallback work, not this phase.
  - **R3** — F-order component-layout mismatch: pyscf-gto declares component-leading `[3, …]` F-order in `layout_table.rs`; the kernel staging must match exactly or pyscf-rs repacks wrong. Validate against vendor layout in the oracle.
  - **R4** — ECP scalar K-Taylor: `ECPscalar_iprinv` byte-identity is only reachable if the scalar ECP primitives are PySCF-exact (K_TAB/ECPrad_part), not the old direct-quadrature approximation. Confirm Phase 19's Cu/LANL2DZ gate exercises the exact path before starting 21-07; otherwise insert a K-Taylor-port plan first.
  - **R5** — spinor variants: the manifest carries `spinor` representations, but pyscf_rs needs only `sph`/`cart`. Scope spinor gradient kernels OUT (register-but-`UnsupportedApi`) unless a consumer needs them.
  - **R6** — Phase 18 coupling: `int2e_ip1` safe-API needs arity-4 dispatch (Phase 18). De-risk by confirming pyscf-gto's call path (raw vs safe) up front; the raw/compat arm can land independently of Phase 18.

**Notes**:

  - Source proposal: `.planning/notes/phase-21-coulomb-gradient-intors-PLAN.md` (drafted 2026-05-26, verified against the tree).
  - The first-derivative machinery is generic and already exists: `gout_ip1` + `nabla1i_2e`/`nabla1j_2e`/`nabla1k_2e` in `crates/cintx-cubecl/src/kernels/f12.rs:590-785` contain zero F12-specific logic and implement the standard libcint `∂/∂A χ_l = -2α·χ_{l+1} + l·χ_{l-1}` identity — reused verbatim for the plain-Coulomb families.
  - Consumer / driver: pyscf_rs Phase 7 (Gradients + Geomopt); see pyscf_rs `.planning/phases/07-gradients-geomopt/07-RESEARCH.md` §"Gradient-Integral Availability Matrix".
  - Wave 2 and the raw-path of Wave 3 are independent of Phase 18; only the `int2e_ip1` safe-API arm is Phase-18-coupled.

## v1.4 Milestone: Full libcint 6.1.3 Family Parity

The ten v1.4 phases (22–31) add the ~140 remaining libcint 6.1.3 integral families
to byte-identity at atol=1e-12 under the vendor-gated oracle (`--features cpu` +
`CINTX_ORACLE_BUILD_VENDOR=1`), reaching complete libcint API parity. They decompose
into 6 family groups plus the foundational env-slot, complex-output, and spinor-transform
prerequisites those groups depend on. Source: `.planning/research/SUMMARY-v1.4.md`,
`ARCHITECTURE-v1.4.md`, `PITFALLS-v1.4.md`.

**Per-family surface scope (v1.4 decision):** each family is
`manifest row (component_rank) → RawApiId const → kernel → vendor FFI (vendor_int*)
+ byte-identity oracle parity test → flip oracle_covered`. The C ABI shim
(`cintx-capi`) enum variants and the legacy `cint*` wrappers (`cintx-compat/legacy.rs`)
are **NOT** added for v1.4 families — the oracle byte-identity gate exercises the raw
`eval_raw` + vendor-FFI path only. The inbound vendor FFI is kept (it is the libcint
reference the byte-identity test compares against). No success criterion below requires
capi or legacy-wrapper surfaces.

**Ordering rationale (three dependency chains):**
1. **Gap A (FND-01) first** — cheap, isolated, unblocks two groups (moments + GIAO);
   the non-zero gauge-origin fixture it creates is the correctness gate for both.
2. **Real cart/sph work before spinor foundations** — Groups 1, 2, 3, and spin-free
   Group 5 deliver the bulk of the non-relativistic derivative + property surface
   before the expensive spinor foundations (Gap B1/B2) are needed.
3. **σ foundations (FND-04, FND-05) before σ families** — the `c2s_si` 4-block
   transform must pass a kappa-bearing fixture before any Group 4/6/GIAO×σ family is
   registered `oracle_covered`; eager registration on the scalar `cart_to_spinor_sf`
   transform produces silently-wrong spinor output (Pitfall 2).

Hard ordering constraints honored: σ families (Group 4, GIAO×σ, Group 6) come after
FND-05; moments + GIAO come after FND-01; high-l families (Group 2/4/6 high-l quartets)
depend on FND-02 (Wheeler nroots≥6).

### Phase 22: Gauge-Origin Env Slot (Gap A — `PTR_COMMON_ORIG`)

**Goal**: The `PTR_COMMON_ORIG` gauge-origin env slot (`env[1..3]`) is read end-to-end through the `eval_raw` → planner → kernel path on the Phase-21 `PTR_RINV_ORIG` precedent, and a non-zero gauge-origin oracle fixture exists so that every downstream moment and GIAO parity test is gated on a value that is multiplied by a *non-zero* origin (not the trivially-passing zero origin of H2O/STO-3G).
**Depends on**: Phase 21 (the `PTR_RINV_ORIG` env-slot block this is modeled on, `raw.rs:599-616`)
**Requirements**: FND-01
**Success Criteria** (what must be TRUE):

  1. `OperatorEnvParams` carries a new `common_orig: Option<[f64;3]>` field and `raw.rs::eval_raw` reads `env[PTR_COMMON_ORIG=1..3]` into it via a read block that mirrors the `PTR_RINV_ORIG` block; an env round-trip unit test passes (FND-01).
  2. A `with_common_origin`-style setter is exposed on the safe-API options, and a `validate_common_orig_env_params` validator enforces **finiteness, not presence** (per 22-CONTEXT D-01 — `None` defaults to `[0,0,0]` and is valid; only a `Some(NaN/inf)` is rejected with a typed `InvalidEnvParam`). The env-read is **operator-agnostic** (D-02 — no operator-name predicate). D-01 validator unit tests pass (FND-01).
  3. A non-zero gauge-origin oracle fixture (H2O/STO-3G with `env[PTR_COMMON_ORIG] != 0`) is added to `fixtures.rs` and is the declared parity gate for Phases 24 (moments) and 26/30 (GIAO) — a zero-origin-only test is documented as a vacuous gate for this slot (FND-01).

**Plans**: 2 plans

Plans:

**Wave 1**
- [x] 22-01-PLAN.md — Core PTR_COMMON_ORIG slot plumbing: const + OperatorEnvParams.common_orig field + operator-agnostic eval_raw env[1..3] read + finiteness validator (D-01) + ExecutionOptions.common_orig + with_common_origin setter + api.rs propagation.

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 22-02-PLAN.md — Non-zero gauge-origin H2O/STO-3G oracle fixture (data infra for Phases 24/26) + raw<->plan slot round-trip test (D-03 slot verification, no consuming kernel).

### Phase 23: Group 1 — Remaining 1st-Derivative Families (cart/sph)

**Goal**: The 8 remaining plain first-derivative families (`int2e_ip2`, `int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip`, `int3c1e_ip1`, `int3c1e_iprinv`, `int2c2e_ip1`, `int2c2e_ip2`, `int3c2e_ip2`) reach byte-identity (cart + sph, component_rank 3) by extending the Phase-21 nabla/`gout_ip1` engine to the ket-side, both-side, and remaining-center derivatives — zero new foundations.
**Depends on**: Phase 21 (the `gout_ip1` + `nabla1i/j/k` engine in `f12.rs:590-785`, reused verbatim)
**Requirements**: DRV1-01, DRV1-02, DRV1-03, DRV1-04, DRV1-05
**Success Criteria** (what must be TRUE):

  1. `int2e_ip2` (arity-4, ∇ on the ket bra-center) matches vendored libcint 6.1.3 at atol=1e-12 (cart + sph) under the vendor gate, with element-for-element byte-identity confirming the component-leading F-order layout (DRV1-01).
  2. `int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip` (∇ on both bra and ket, rank 9) each match at atol=1e-12 (cart + sph) (DRV1-02).
  3. `int3c1e_ip1` and `int3c1e_iprinv` (the `iprinv` variant reusing the already-plumbed `PTR_RINV_ORIG`) match at atol=1e-12 (cart + sph) (DRV1-03).
  4. `int2c2e_ip1` and `int2c2e_ip2` match at atol=1e-12 (cart + sph) (DRV1-04).
  5. `int3c2e_ip2` matches at atol=1e-12 (cart + sph) (DRV1-05).
  6. Each family is registered with its `component_rank`, dispatches through `eval_raw`, has a dedicated `vendor_*` parity test executing under both `--features cpu` and `CINTX_ORACLE_BUILD_VENDOR=1` (`running N>0 tests`), and is flipped `oracle_covered=true`; `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: 5 plans

> **Scope note:** cluster C (DRV1-02, the rank-9 both-side `int1e_ipovlpip/ipkinip/ipnucip`) is ALREADY COMPLETE and vendor-verified (commit `319d055`). The plans below cover only the remaining clusters A & B (6 families); plan 05 is a DRV1-02 regression guard that re-runs the existing cluster-C parity test (no re-implementation).

Plans:

**Wave 1** — engine plumbing + DRV1-02 regression guard (parallel; disjoint files)
- [x] 23-01-PLAN.md — Promote `nabla1j_2e`/`nabla1k_2e` to `pub(crate)`, add `nabla1l_2e` (mirror `nabla1l_breit` for the 3c2e `ll`-slot), and a nabla-parameterized single-side contraction in `f12.rs` (unblocks clusters A). []
- [x] 23-05-PLAN.md — DRV1-02 regression guard: re-run the existing cluster-C `one_electron_grad_both_parity` vendor test under the double gate (no source change). [DRV1-02]

**Wave 2** *(blocked on 23-01)* — cluster A part 1 (pure Phase-21 reuse)
- [x] 23-02-PLAN.md — `int2e_ip2` (nabla1k on ket bra-center k) + `int2c2e_ip1/ip2` (nabla1i/k, lj=ll=0); manifest rank-3 ×3 reps, RawApiId, dispatch (center_2c2e dispatch ADDED), vendor FFI + 2 parity tests at atol=1e-12. [DRV1-01, DRV1-04]

**Wave 3** *(blocked on 23-01, 23-02 — shares registration files with 23-02)* — cluster A part 2 (Pitfall 2)
- [x] 23-03-PLAN.md — `int3c2e_ip2` (∇ on auxiliary k → cintx `ll` slot → `nabla1l_2e`, NOT nabla1k); manifest rank-3 ×3 reps, RawApiId, dispatch, vendor FFI + parity test; assert ip2≠ip1; atol=1e-12. [DRV1-05]

**Wave 4** *(blocked on 23-03 — shares registration files)* — cluster B (the 3c1e pair; the only new base kernel)
- [x] 23-04-PLAN.md — `int3c1e_ip1` (overlap deriv, no Rys) + `int3c1e_iprinv` (NEW Rys-driven `fill_g_tensor_3c1e_nuc` base reusing `rys_roots_host` + the plumbed `PTR_RINV_ORIG`; fail-closed at nroots>5/fff); manifest rank-3 ×3 reps ×2, RawApiId, dispatch (center_3c1e dispatch ADDED), vendor FFI + parity test; atol=1e-12; capi/legacy untouched. [DRV1-03]

### Phase 24: Group 3 — Position / Multipole-Moment Integrals

**Goal**: The full position/multipole-moment family set (`int1e_r/rr/rrr/rrrr`, `int1e_r2/r4`, `int1e_z/zz`, `int1e_p4`, plain `int1e_rinv`, `int1e_drinv`, `int1e_irp`, and the `_origj` variants) reaches byte-identity (cart + sph) on the position-operator G-tensor, with every `r`-operator family reading the gauge origin and validated against the non-zero gauge-origin fixture — not trivially-passing at zero origin.
**Depends on**: Phase 22 (the `PTR_COMMON_ORIG` slot + gauge-origin fixture; libcint's `int1e_r` computes `drj = rj - env[PTR_COMMON_ORIG]`, so even plain moments read it). Can run in parallel with Phase 23.
**Requirements**: MOM-01, MOM-02, MOM-03, MOM-04
**Success Criteria** (what must be TRUE):

  1. Dipole `int1e_r` (and `int1e_r_origj`) match at atol=1e-12 against the **non-zero** gauge-origin fixture (cart + sph), with the angular-momentum headroom raised on the **ket** (`ng[1]`), not the bra — a regression confirms the result is not transposed (MOM-01).
  2. `int1e_rr`, `int1e_r2`, `int1e_z`, `int1e_zz` and their `_origj` variants match at atol=1e-12 (cart + sph), with the `rr` Cartesian component order copied from the libcint gout index map (MOM-02).
  3. `int1e_rrr`, `int1e_rrrr`, `int1e_r4` (octupole / hexadecapole, ket headroom up to `ng[1]=4`, rank up to 81) match at atol=1e-12 (cart + sph) (MOM-03).
  4. `int1e_p4`, `int1e_drinv`, plain `int1e_rinv`, `int1e_irp` match at atol=1e-12 (cart + sph) (MOM-04).
  5. Each family is registered with its `component_rank`, dispatches through `eval_raw`, has a dedicated `vendor_*` parity test executing under both flags, and is flipped `oracle_covered=true`; `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: 5 plans

Plans:
**Wave 1**
- [x] 24-01-PLAN.md — Wave 0: vendor-FFI wrappers + bindgen allowlist (38 symbols) + 4 moment parity scaffolds + non-zero rinv_orig helper + OQ-2 triage
- [ ] 24-02-PLAN.md — Wave 1 (Cluster A): r/rr/rrr/rrrr/r2/r4/z/zz + 6 _origj via one parameterized moment kernel (origin-source branch, verbatim gout, ket headroom)

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 24-03-PLAN.md — Wave 2 (Cluster B): rinv/drinv single-center Rys, PTR_RINV_ORIG, charge=+1, no atom-sum

**Wave 3** *(blocked on Wave 2 completion)*
- [ ] 24-04-PLAN.md — Wave 2 (Cluster C): p4 (∇⁴) overlap-derivative, both-side headroom ng={2,2,...}

**Wave 4** *(blocked on Wave 3 completion)*
- [ ] 24-05-PLAN.md — Wave 2 (Cluster D): irp (i·r×∇) rank-9 3×3 r⊗∇ tensor, reads PTR_COMMON_ORIG

### Phase 25: Group 2 — Hessian & Higher-Order Derivatives

**Goal**: The 2nd/3rd/4th-order derivative families (`int1e_ipip*`, the 2e Hessian set promoted from `unstable`, `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, the 4th-order `ipipip*` families) reach byte-identity (cart + sph) at component_rank 9/27/81, after the fail-closed high-rank staging cleanup lands and the Rys `nroots>=6` Wheeler fallback removes the high-angular-momentum ceiling so no family returns `UnsupportedApi` purely due to `nroots>5`.
**Depends on**: Phase 23 (the Group-1 first-order engine this composes to 2nd+ order). Phase 24 (the ket-headroom plumbing for multi-center elevation).
**Requirements**: HESS-01, HESS-02, HESS-03, HESS-04, FND-02, FND-06
**Success Criteria** (what must be TRUE):

  1. The Rys `nroots>=6` Wheeler-fallback is implemented and byte-identical vs vendored libcint for nroots 6..~13; the `executor.rs` `ang_momentum>4` gate is extended to admit g/h once the roots support them; no family returns `UnsupportedApi` purely because `nroots>5` (FND-02; closes `.planning/todos/pending/rys-nroots-ge6-wheeler-fallback.md`).
  2. High-rank (component_rank 9/27/81) staging is **fail-closed**: an upfront `BufferTooSmall`-style size assertion replaces the `if dst < staging.len()` per-element scatter guards (no silent partial writes), and the chunk planner's OOM-safe-stop is re-validated with an OOM test at rank 81 (FND-06).
  3. `int1e_ipipovlp`, `int1e_ipipnuc`, `int1e_ipipkin`, `int1e_ipiprinv` (rank 9) match at atol=1e-12 (cart + sph), with the per-family `ng[]` headroom tuple driving G-tensor sizing (bra +2) and element-for-element byte-identity confirming the ×9 component order (HESS-01).
  4. The 2e Hessian set (`int2e_ipip1`, `int2e_ipvip1`, `int2e_ip1ip2`, `int2e_ipip1ipip2`) — promoted from `unstable::source::2e` and re-routed through the stable raw-api map — matches at atol=1e-12 (cart + sph) (HESS-02).
  5. `int2c2e_ipip1`, `int3c2e_ipip1`, `int3c2e_ipip2` match at atol=1e-12 (cart + sph) (HESS-03).
  6. 3rd/4th-order families (`int1e_ipipipnuc`, `int1e_ipipipiprinv`, and siblings) match at atol=1e-12 (cart + sph) with `ng[]`-driven bra+ket headroom (deriv4 raises bra +2 AND ket +2) (HESS-04).
  7. `deriv3.c` and `deriv4.c` are added to the oracle `cc::Build` with suppl-header `extern` decls + allowlist entries; each family has a dedicated `vendor_*` test executing under both flags and is flipped `oracle_covered=true`; `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: TBD
**Research flag**: The Wheeler `nroots>=6` fallback scope (FND-02) is a milestone-level decision that must be resolved before this phase's plans are finalized.

### Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals (complex)

**Goal**: The spin-free 1e and 2e GIAO/CG families (`int1e_giao_*`, `int1e_cg_*`, `int1e_govlp/gnuc/gkin`, `int1e_ig*`, `int1e_a01gp`, `int1e_ia01p`, and the 2e `int2e_g1/gg1/ig1/giao_*`) — which are **purely imaginary even in cart/sph** — reach byte-identity through a per-family complex-interleaved output capability, validated against the non-zero gauge-origin fixture so the imaginary content actually lands (not silently zeroed).
**Depends on**: Phase 22 (gauge origin — GIAO = gauge-including atomic orbital). Phases 23 + 24 (the nabla step + position-operator tensor the `r_gauge × ∇` factor combines).
**Requirements**: GIAO-01, GIAO-02, FND-03
**Success Criteria** (what must be TRUE):

  1. Complex/imaginary output capability is real: `complex_interleaved` is set per-family from driver routing (not the representation string), `assert_flat_buffer_contract` fires on the flag (a complex cart/sph family staged as real-only FAILS the contract), staging is sized `2×ncomp×…`, and a purely-imaginary family (e.g. `int1e_igovlp`) round-trips through the safe API without silent zeroing (FND-03).
  2. The spin-free 1e GIAO/CG families (`int1e_giao_*`, `int1e_cg_*`, `int1e_govlp/gnuc/gkin`, `int1e_ig*`, `int1e_a01gp`, `int1e_ia01p`) match at atol=1e-12 (cart + sph) via the complex path, with the vendor wrapper passing the same `2×`-interleaved buffer to the `double complex *out` libcint symbol (GIAO-01).
  3. The 2e GIAO families (`int2e_g1`, `int2e_gg1`, `int2e_ig1`, `int2e_giao_*`) match at atol=1e-12, with `autocode/intor4.c` added to the oracle `cc::Build` (GIAO-02).
  4. Every family is gated on the non-zero gauge-origin fixture (a zero-origin GIAO test is doubly-trivial), has a dedicated `vendor_*` test executing under both flags, and is flipped `oracle_covered=true`; `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: TBD

### Phase 27: Spinor-Derivative Transform (Gap B1)

**Goal**: The spinor-derivative transform `cart_to_spinor_sf_derivative_*` is implemented in `c2spinor.rs` so that `int1e_ipovlp_spinor` and the sibling `ip`-decorated spinor families move from `UnsupportedApi` to byte-identity at atol=1e-12 — closing the Phase-21 R5/D-03 deferral and unblocking the spinor variants of the Group 1/2/5 derivative families.
**Depends on**: Phase 23 (the scalar first-derivative kernels whose 3-component Cartesian blocks the spinor transform folds per-component). Independent of Gap B2; can run in parallel with Phase 26.
**Requirements**: FND-04
**Success Criteria** (what must be TRUE):

  1. `cart_to_spinor_sf_derivative_*` is added to `transform/c2spinor.rs`, applying the cart→spinor coupling per derivative component and folding the `[3, …]` component axis correctly (FND-04).
  2. `int1e_ipovlp_spinor` moves from `UnsupportedApi` to byte-identity at atol=1e-12 against a spinor fixture, and the sibling `ip`-decorated spinor families that depend only on B1 are flipped `oracle_covered=true` (FND-04).
  3. A dedicated `vendor_*` spinor parity test executes under both flags (`running N>0 tests`) and is not a `skipped` fixture; `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: TBD
**Research flag**: The spinor-derivative per-component axis-fold design is not yet exercised — a one-day design spike against `int1e_ipovlp_spinor` is recommended before this phase's plans are finalized.

### Phase 28: Spin-Included `c2s_si` Transform + σ·p Module (Gap B2)

**Goal**: The spin-included spinor transform `cart_to_spinor_si_*` (the 4-block `gc_x/gc_y/gc_z/gc_1` input of libcint's `c2s_si_1e`, `cart2sph.c:4947`) and the companion σ·p G-tensor assembler module are implemented and validated against a kappa-bearing relativistic oracle fixture — the single largest architectural addition in v1.4 and the hard prerequisite for every σ-operator family (Groups 4, 6, and the GIAO×σ slice of 5).
**Depends on**: Phase 12 (the scalar Clebsch-Gordan spinor transform this generalizes). Phase 27 is a sibling foundation (B1), not a hard dependency.
**Requirements**: FND-05
**Success Criteria** (what must be TRUE):

  1. `cart_to_spinor_si_*` is added to `c2spinor.rs`, consuming the 4-block `gc_x/gc_y/gc_z/gc_1` G-tensor (the three Pauli-σ component blocks plus the scalar), with the block count asserted at the transform boundary; the σ-coupling matches libcint `c2s_si_1e` (FND-05).
  2. A σ·p G-tensor assembler module (with the 12-component Pauli `gout` emitter) produces the four `gc_*` blocks the `si` transform reads in order (FND-05).
  3. A kappa-bearing relativistic oracle fixture (a molecule with spinor shells — H2O/STO-3G has none) is added to `fixtures.rs`, and the `si` transform + σ·p assembler pass an end-to-end byte-identity check at atol=1e-12 against it (FND-05).
  4. `oracle-covered-update` mechanically refuses to flip `oracle_covered=true` for a σ/spinor family whose only fixture was `skipped`; all σ families stay `UnsupportedApi` until this phase passes on the kappa fixture. No capi/legacy-wrapper surface is added.

**Plans**: TBD
**Research flag**: Confirm the `a_bra_cart2spinor_si` 4-block stride/ordering from `cart2sph.c:4947-4992` (a design spike) before this phase's plans are finalized.

### Phase 29: Group 4 — Relativistic Spin-Operator Integrals (spinor)

**Goal**: The relativistic spin-operator families (`int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp`, `int1e_srsr`, `int1e_sr/srnucsr`, `int1e_sigma`, `int1e_sp`, and the 2e `int2e_spsp1/srsr1`, `int2e_ssp1ssp2/sps1sps2`, `int2e_vsp1*/spv1*`) reach byte-identity (spinor) through the Gap B2 `c2s_si` path and the new σ·p module — the Dirac/X2C/DKH and spin-orbit-coupling integrals no other Rust library currently provides.
**Depends on**: Phase 28 (Gap B2 — the `c2s_si` 4-block transform + σ·p module; all Group 4 families gate on it). Phase 27 (Gap B1) for the `ip`-decorated spin gradients.
**Requirements**: REL-01, REL-02, REL-03, REL-04
**Success Criteria** (what must be TRUE):

  1. `int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp` match vendored libcint at atol=1e-12 (spinor) via the Gap B2 `c2s_si` path — routing through the scalar `cart_to_spinor_sf` is explicitly rejected (REL-01).
  2. `int1e_srsr`, `int1e_sr`/`srnucsr`, `int1e_sigma`, `int1e_sp` match at atol=1e-12 (spinor), with the σ 12-component Pauli pattern copied verbatim from the libcint gout (REL-02).
  3. `int2e_spsp1`, `int2e_srsr1` (and `spsp1spsp2`/`srsr1srsr2`) match at atol=1e-12 (spinor), with `autocode/intor4.c` wired into the oracle build for the spin 2e block (REL-03).
  4. `int2e_ssp1ssp2`, `int2e_sps1sps2`, `int2e_vsp1*`, `int2e_spv1*` match at atol=1e-12 (spinor) (REL-04).
  5. Every family is exercised on the kappa-bearing relativistic fixture (N>0 evaluated, non-skipped), has a dedicated `vendor_*` test under both flags, and is flipped `oracle_covered=true` only on the spinor representation (cart/sph σ intermediates are not over-claimed); `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: TBD

### Phase 30: Group 5 (GIAO×σ slice) — Spin-GIAO Integrals (spinor)

**Goal**: The relativistic-NMR GIAO×σ slice (`int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, the 2e `int2e_cg_sa10*`/`giao_sa10*`) reaches byte-identity (spinor) by combining the complex-interleaved output capability (Phase 26) with the σ·p `c2s_si` path (Phase 28) and the gauge origin (Phase 22) — completing the magnetic-property suite including relativistic corrections.
**Depends on**: Phase 22 (gauge origin). Phase 28 (Gap B2 σ path). Phase 29 (the σ·p pattern reused directly). Phase 26 (complex output capability).
**Requirements**: GIAO-03
**Success Criteria** (what must be TRUE):

  1. The GIAO×σ family set (`int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`) matches vendored libcint at atol=1e-12 (spinor) via the Gap B2 σ path and the complex-interleaved output (GIAO-03).
  2. Every family is gated on BOTH the non-zero gauge-origin fixture AND the kappa-bearing relativistic fixture, has a dedicated `vendor_*` test executing under both flags (non-skipped), and is flipped `oracle_covered=true`; `manifest-audit` is green. No capi/legacy-wrapper surface is added.

**Plans**: TBD

### Phase 31: Group 6 — Gauge / Breit–Gaunt 2e + Full-Parity Verification (apex)

**Goal**: The Dirac–Coulomb–Breit 2e family set (`int2e_gauge_r1_{ssp,sps}{ssp,sps}`, `int2e_gauge_r2_{ssp,sps}{ssp,sps}`, and the Gaunt `ssp/sps` families) reaches byte-identity (spinor) by per-block decomposition of the existing `launch_breit` driver on the Group-4 σ·p machinery — AND the milestone-closing full-parity gate is met: `manifest-audit` shows every libcint 6.1.3 family `oracle_covered=true` for its physical representations, the full vendor-gated oracle suite is green, and the unsupported-families list (vs `cint_funcs.h` + supplemental headers) is empty.
**Depends on**: Phase 29 (Group 4 — the σ·p + Gaunt-style σ machinery). Phases 27/28 (Gaps B1/B2). Phase 14 (the existing `launch_breit`/`BreitShape` the gauge symbols decompose).
**Requirements**: BREIT-01, BREIT-02, BREIT-03, PARITY-01
**Success Criteria** (what must be TRUE):

  1. `int2e_gauge_r1_{ssp,sps}{ssp,sps}` (4 symbols) match vendored libcint at atol=1e-12 (spinor), routing through `c2s_si_2e1i`/`c2s_si_2e2i` (verified `breit1.c:211`) with the complex `double complex *out` buffer sized `2×` (BREIT-01).
  2. `int2e_gauge_r2_{ssp,sps}{ssp,sps}` (4 symbols) match at atol=1e-12 (spinor) (BREIT-02).
  3. The Gaunt `ssp/sps` families match at atol=1e-12 (spinor), reusing the existing `launch_breit` decomposition, with `autocode/gaunt1.c` + `breit1.c` added to the oracle `cc::Build` and suppl-header `extern` decls (these symbols are absent from `cint_funcs.h`) (BREIT-03).
  4. `manifest-audit` is green with EVERY libcint 6.1.3 family `oracle_covered=true` for its physical representations (cart/sph; spinor where physical, σ families spinor-only); the full vendor-gated oracle suite is green under both flags; and the "unsupported libcint families" list (vs `cint_funcs.h` + supplemental headers) is empty — full API parity is mechanically verifiable (PARITY-01).
  5. No capi/legacy-wrapper surface is added for any Group 6 family; the byte-identity gate exercises the raw `eval_raw` + vendor-FFI path only.

**Plans**: TBD
