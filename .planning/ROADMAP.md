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
- [ ] **Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator (issue #11 Task 1)** - Implement Type-1 (Coulomb-like) and Type-2 (spin-orbit-like) ECP projectors and expose them through `SessionRequest` alongside ordinary one-electron operators. Cu/LANL2DZ in the oracle corpus provides a byte-identity gate against libcint.

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
| Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator | v1.3 | 0/6 | Planned | - |

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
**Plans**: 6 plans (planned 2026-05-12; oracle pivot to PySCF nr_ecp per D-01 REVISED; gradients in scope per D-10)

Plans:
**Wave 0**
- [ ] 19-01-PLAN.md — Wave 0: vendor PySCF nr_ecp subtree (Apache-2.0); extend cintx-oracle/build.rs with parallel cc::Build + has_vendor_pyscf_nr_ecp cfg; expand api_manifest.csv with 4 ECP rows (cart/sph × {ecp, ecp_ipnuc}) + regenerate lock; build Cu/LANL2DZ fixture; land empty stubs for bessel.rs, radial_quadrature.rs, EcpShell.

**Wave 1** *(blocked on Wave 0 completion; 02 and 03 run in parallel)*
- [ ] 19-02-PLAN.md — Wave 1: implement modified spherical Bessel i_l(x) in math/bessel.rs (paired #[cube] + *_host(), three numerical branches per PySCF nr_ecp.h K_TAB tables) + Gauss-Chebyshev (Type-2 radial) and Gauss-Hermite (Type-1 radial) nodes/weights in math/radial_quadrature.rs. Host-side unit tests at atol=1e-12 cover all branches.
- [ ] 19-03-PLAN.md — Wave 1: EcpShell + BasisSet::ecp_shells extension + OperatorId::INT1E_ECP_* constants + is_ecp() helper + cintx-compat::raw ECP slot constants (AS_ECPBAS_OFFSET=18, AS_NECPBAS=19, RADI_POWER=3, SO_TYPE_OF=4, ECP_LMAX=5 per PySCF nr_ecp.h verbatim) + EcpBasArray typed view + eval_raw ECP dispatch arm + FacadeError::MissingEcpBasis variant + SessionRequest::query_workspace preflight.

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 19-04-PLAN.md — Wave 2: launch_ecp kernel (Type-1 + Type-2 scalar) in crates/cintx-cubecl/src/kernels/ecp.rs; canonical_family "ecp" registered unconditionally in kernels/mod.rs; vendor_ECPscalar_{cart,sph} FFI wrappers; two named per-symbol parity tests at atol=1e-12 vs PySCF nr_ecp over Cu/LANL2DZ Cartesian product; flip oracle_covered=true on the two scalar manifest rows.

**Wave 3** *(blocked on Wave 2 completion)*
- [ ] 19-05-PLAN.md — Wave 3: gradient branch in launch_ecp (Type-1 + Type-2 derivatives) per PySCF nr_ecp_deriv.c; F-order [axis, ao_j, ao_i] component layout (component_rank=3, axis slowest-varying, int3c2e_ip1_* precedent); vendor_ECPscalar_ipnuc_{cart,sph} FFI wrappers; two more parity tests at atol=1e-12; flip oracle_covered=true on the two ipnuc manifest rows.

**Wave 4** *(independent of Wave 2/3; optional)*
- [ ] 19-06-PLAN.md — Wave 4 (optional, non-blocking): libecpint (Shaw & Hill, JCP 147 074108, 2017, MIT) secondary cross-check oracle behind cargo cfg has_libecpint_oracle (emitted only when CINTX_LIBECPINT_ORACLE=1); four #[ignore]+env-gated cross-check tests at informational atol=1e-9 per D-02 REVISED.

**Notes**:
  - Largest of the three issue #11 tasks — requires a new evaluator implementation, not just dispatch wiring.
  - libcint's upstream ECP code was reportedly copied from PySCF; cross-reference against pyscf's `pyscf/gto/ecp.py` may be useful during planning.
  - Reference implementation outside libcint: https://github.com/chrr/libECP (libECP, JCC 2017).
  - Downstream marker: pyscf_rs `crates/pyscf-gto/src/ecp_engine_stub.rs` is the placeholder that this phase replaces.
