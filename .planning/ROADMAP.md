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
