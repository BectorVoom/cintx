# Roadmap

## Phases
- [x] **Phase 1: Manifest & Planner Foundation** - Lock down typed domain models, manifest registry, and planner scaffolding so everything else has a deterministic catalog to build against.
- [x] **Phase 2: Execution & Compatibility Stabilization** - Wire the CubeCL-backed planner to the raw compat layer, including helper/legacy transforms, workspace queries, typed errors, and shape/optimizer guarantees.
- [x] **Phase 3: Safe Surface, C ABI Shim & Optional Families** - Layer the safe Rust facade, optional C shim, and feature-gated optional families on the stabilized runtime.
- [x] **Phase 4: Verification & Release Automation** - Close the manifest/oracle loop with CI, benchmarks, and diagnostics that block regressions before release.
- [x] **Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend)** - Replace synthetic execution with a real wgpu-backed CubeCL path and capability-aware fail-closed verification.
- [x] **Phase 6: Fix raw eval staging retrieval and capability fingerprint propagation** - Close audit gaps: wire eval_raw() staging output retrieval, propagate wgpu fingerprint into capability token, add regression tests.
- [x] **Phase 7: Executor Infrastructure Rewrite** - Replace executor internals with direct CubeCL client API, introduce ResolvedBackend dispatch, CPU backend feature, and f64 strategy decision — prerequisite gate for all real kernel work.
- [x] **Phase 8: Gaussian Primitive Infrastructure and Boys Function** - Build shared math foundation as `#[cube]` functions: Boys function, Rys roots/weights, primitive pair evaluation, and Obara-Saika recurrence. (completed 2026-04-03)
- [x] **Phase 9: 1e Real Kernel and Cart-to-Sph Transform** - Implement real overlap, kinetic, and nuclear attraction kernels with correct Condon-Shortley c2s transform, validating the end-to-end compute pipeline.
- [x] **Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure** - Implement all remaining integral family kernels and close the oracle parity gate for all five base families, completing v1.1. (completed 2026-04-03)
- [x] **Phase 11: Helper/Transform Completion & 4c1e Real Kernel** - Wire all helper, transform, and wrapper symbols to oracle CI; replace the 4c1e stub with real Rys quadrature within the Validated4C1E envelope. (completed 2026-04-04)
- [x] **Phase 12: Real Spinor Transform (c2spinor Replacement)** - Rewrite c2spinor.rs with correct Clebsch-Gordan coupling; unblock spinor oracle coverage for all families that depend on it. (completed 2026-04-05)
- [x] **Phase 13: F12/STG/YP Kernels** - Implement STG and YP geminal 2e kernels with separate dispatch paths, PTR_F12_ZETA env plumbing, and sph-only oracle gate under the with-f12 profile. (completed 2026-04-05)
- [ ] **Phase 14: Unstable-Source-API Families** - Implement origi, grids, Breit (stub), origk, and ssc (stub) families behind the unstable-source-api gate with oracle parity in nightly CI.
- [ ] **Phase 15: Oracle Tolerance Unification & Manifest Lock Closure** - Audit every family's empirical precision floor, set per-family atol/rtol constants, regenerate the four-profile manifest lock, and close the unified oracle CI gate.

## Phase Details

### Phase 1: Manifest & Planner Foundation
**Goal**: Establish the typed domain structures, manifest lock, registry, and planner foundations that every later layer consumes.
**Depends on**: Nothing
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Maintainers can instantiate atoms, shells, basis sets, environment parameters, operator IDs, and tensor layouts through the typed Rust structures defined in the manifest (BASE-01).
  2. The manifest generation pipeline emits a lock that classifies stable, optional, and unstable-source families across the support matrix and becomes the canonical input for downstream gating (BASE-02).
  3. The manifest-aware registry resolves which integral families and representations are available without depending on raw symbol names, so consumers can pick kernels declaratively (BASE-03).
**Plans**: 4 plans
Plans:
- [x] 01-PLAN.md - Upgrade oracle fixtures and parity reporting to required profile coverage with non-fail-fast mismatch artifacts.
- [x] 02-PLAN.md - Implement xtask gate commands for manifest drift, oracle parity, helper/legacy parity, and OOM-contract enforcement.
- [x] 03-PLAN.md - Wire merge-blocking PR CI gates and required profile matrix verification through the new xtask command surface.
- [x] 04-PLAN.md - Add benchmark/diagnostics automation with threshold gating and advisory-vs-required GPU workflow policy.

### Phase 2: Execution & Compatibility Stabilization
**Goal**: Prove the CubeCL-backed planner and compat/API surface can consume the manifest, honor workspace queries, handle memory limits, and deliver upstream-compatible outputs.
**Depends on**: Phase 1
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-05, EXEC-02, EXEC-03, EXEC-04, EXEC-05
**Success Criteria** (what must be TRUE):
  1. Compat callers can invoke the raw APIs with the documented `atm/bas/env/shls/dims/opt/cache` inputs and reach the helper/legacy/transform symbols preserved in the manifest (COMP-01, COMP-03).
  2. Workspace and output query helpers return buffer sizes and workspace estimates before evaluation, letting callers plan allocations safely (COMP-02).
  3. The CubeCL-backed planner evaluates the 1e, 2e, 2c2e, 3c1e, and 3c2e families through the shared backend, selecting kernels based on the manifest registry (EXEC-02).
  4. Memory-limited runs chunk deterministically, surface typed `MemoryLimitExceeded` or `UnsupportedApi`, and never write partial results, keeping validation failures explicit (EXEC-03, COMP-05).
  5. Outputs appear with the expected cart/sph/spinor shapes and ordering and stay numerically equivalent whether optimizer support is toggled (EXEC-04, EXEC-05).
**Plans**: 7 plans
Plans:
- [x] 02-PLAN.md - Activate the Phase 2 workspace members and wire compat-to-CubeCL plus oracle-to-compat crate dependencies.
- [x] 03-PLAN.md - Expand the canonical manifest and public error surface for helper/legacy coverage plus typed raw failures.
- [x] 04-PLAN.md - Add the backend-neutral runtime execution contract, deterministic scheduling, and runtime-owned execution metrics.
- [x] 05-PLAN.md - Implement the CubeCL executor core and the `1e`/`2e`/`2c2e` kernel slice.
- [x] 06-PLAN.md - Build the raw compat query/evaluate pipeline and enforce no-partial-write layout rules.
- [x] 07-PLAN.md - Add helper/transform/optimizer/legacy compat APIs and oracle parity coverage.
- [x] 08-PLAN.md - Finish the `3c1e`/`3c2e` CubeCL kernels and cart/sph/spinor transform routing.

### Phase 3: Safe Surface, C ABI Shim & Optional Families
**Goal**: Expose the safe Rust facade, optional C shim, and gated optional families once the runtime is stable.
**Depends on**: Phase 2
**Requirements**: EXEC-01, COMP-04, OPT-01, OPT-02, OPT-03
**Success Criteria** (what must be TRUE):
  1. The safe Rust API splits `query_workspace()` from `evaluate()`, letting callers observe workspace needs before committing to execution (EXEC-01).
  2. The optional C ABI shim accepts compat-style inputs, returns integer status codes, and exposes thread-local last-error details for C integrators (COMP-04).
  3. `with-f12`, `with-4c1e`, and other optional-family gates only enable validated envelopes and emit `UnsupportedApi` for out-of-envelope arguments (OPT-01, OPT-02).
  4. Source-only APIs stay behind `unstable-source-api` so the GA surface remains stable until the maintainer intentionally enables those symbols (OPT-03).
**Plans**: 6 plans
Plans:
- [x] 01-PLAN.md - Activate Phase 3 workspace/feature topology and stable-vs-unstable namespace scaffolding for `cintx-rs`/`cintx-capi`.
- [x] 02-PLAN.md - Add manifest-driven optional-family and unstable-source gates with strict runtime envelope enforcement.
- [x] 03-PLAN.md - Implement the safe Rust session facade with split `query_workspace()`/`evaluate()` and owned typed outputs.
- [x] 04-PLAN.md - Implement the optional C ABI shim with integer status taxonomy and thread-local last-error copy-out APIs.
- [x] 05-PLAN.md - Raise manifest artifact depth and expand safe builder/prelude ergonomics to satisfy Phase 3 must-have substance gates.
- [x] 06-PLAN.md - Wire safe evaluate policy enforcement through shared compat raw gates for optional/source UnsupportedApi parity.

### Phase 4: Verification & Release Automation
**Goal**: Close the manifest/oracle verification loop, run multi-profile CI/benchmarks, and surface diagnostics that block regressions before release.
**Depends on**: Phase 3
**Requirements**: VERI-01, VERI-02, VERI-03, VERI-04
**Success Criteria** (what must be TRUE):
  1. The oracle suite compares the stable and optional APIs against upstream libcint per manifest family with documented tolerances and flags regressions (VERI-01).
  2. CI workflows block manifest drift, helper/legacy parity regressions, CubeCL consistency failures, and OOM contract violations before merges land (VERI-02).
  3. Benchmarks capture throughput, memory usage, and CPU-GPU crossover regressions for trend tracking (VERI-03).
  4. Tracing and diagnostics expose planner chunking, fallback, transfer, and OOM behavior for manual inspection (VERI-04).
**Plans**: 7 plans
Plans:
- [x] 01-PLAN.md - Upgrade oracle fixtures and parity reporting to required profile coverage with non-fail-fast mismatch artifacts.
- [x] 02-PLAN.md - Implement xtask gate commands for manifest drift, oracle parity, helper/legacy parity, and OOM-contract enforcement.
- [x] 03-PLAN.md - Wire merge-blocking PR CI gates and required profile matrix verification through the new xtask command surface.
- [x] 04-PLAN.md - Add benchmark/diagnostics automation with threshold gating and advisory-vs-required GPU workflow policy.
- [x] 05-PLAN.md - Close the remaining oracle crate export-surface substance gap in `crates/cintx-oracle/src/lib.rs`.
- [x] 06-PLAN.md - Close the `gpu_bench_required` runner contract and required/fallback artifact validation gap for release/scheduled GPU verification.
- [x] 07-PLAN.md - Close the release governance workflow min-lines gap with policy-preserving, substantive workflow hardening.

## Progress
| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| Phase 1: Manifest & Planner Foundation | 2/2 | Complete | 2026-03-21 |
| Phase 2: Execution & Compatibility Stabilization | 7/7 | Complete | 2026-03-26 |
| Phase 3: Safe Surface, C ABI Shim & Optional Families | 6/6 | Complete | 2026-03-28 |
| Phase 4: Verification & Release Automation | 7/7 | Complete | 2026-03-31 |
| Phase 5: Re-implement detailed-design GPU path | 5/5 | Complete | 2026-04-02 |
| Phase 6: Fix raw eval staging & fingerprint | 2/2 | Complete | 2026-04-05 |
| Phase 7: Executor Infrastructure Rewrite | 3/3 | Complete | 2026-04-05 |
| Phase 8: Gaussian Primitive Infrastructure and Boys Function | 4/4 | Complete | 2026-04-05 |
| Phase 9: 1e Real Kernel and Cart-to-Sph Transform | 5/5 | Complete | 2026-04-05 |
| Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure | 6/6 | Complete | 2026-04-05 |
| Phase 11: Helper/Transform Completion & 4c1e Real Kernel | 4/4 | Complete | 2026-04-05 |
| Phase 12: Real Spinor Transform (c2spinor Replacement) | 5/5 | Complete | 2026-04-05 |
| Phase 13: F12/STG/YP Kernels | 3/4 | Gap closure | - |
| Phase 14: Unstable-Source-API Families | 0/TBD | Not started | - |
| Phase 15: Oracle Tolerance Unification & Manifest Lock Closure | 0/TBD | Not started | - |

### Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend)

**Goal:** Re-implement the compute path so supported evaluations execute through a real CubeCL + wgpu backend with explicit capability gating, no synthetic fallback, and layered verification evidence.
**Requirements**: EXEC-02, EXEC-03, COMP-05, VERI-02, VERI-04
**Depends on:** Phase 4
**Plans:** 5/5 plans executed

Plans:
- [x] 01-PLAN.md - Add runtime backend intent/capability query-evaluate contract and fail-closed drift checks.
- [x] 02-PLAN.md - Implement CubeCL wgpu bootstrap + capability snapshot preflight contracts.
- [x] 03-PLAN.md - Replace synthetic CubeCL executor staging path with real chunked wgpu execution and unsupported taxonomy.
- [x] 04-PLAN.md - Align compat/raw and safe facade with shared CubeCL executor plus anti-pseudo layered tests.
- [x] 05-PLAN.md - Add capability-aware xtask artifacts and PR/release CI gates for wgpu regression enforcement.

### Phase 6: Fix raw eval staging retrieval and capability fingerprint propagation

**Goal:** Close milestone audit gaps: fix eval_raw() to retrieve executor staging output instead of writing zeros, propagate wgpu bootstrap fingerprint into BackendCapabilityToken for drift detection, and add regression coverage.
**Requirements**: COMP-01, COMP-04, COMP-05, EXEC-02, EXEC-04, EXEC-05, VERI-01
**Depends on:** Phase 5
**Gap Closure:** Closes gaps from v1.0 milestone audit
**Plans:** 2/2 plans executed

Plans:
- [x] 06-01-PLAN.md — Fix eval_raw() staging retrieval with RecordingExecutor and propagate wgpu fingerprint in compat raw and safe facade paths.
- [x] 06-02-PLAN.md — Add regression tests for staging retrieval, fingerprint propagation, base family coverage, and deterministic output.

---

## v1.1 Milestone: CubeCL Direct Client API & Real Kernel Compute

### Phase 7: Executor Infrastructure Rewrite
**Goal**: Executor internals use CubeCL client API directly, with ResolvedBackend dispatch between wgpu and CPU arms, and the f64 precision strategy resolved before any real kernel is written.
**Depends on**: Phase 6
**Requirements**: EXEC-06, EXEC-07, EXEC-08, EXEC-09, VERI-06
**Success Criteria** (what must be TRUE):
  1. Executor allocates and reads device buffers via `client.create()`, `client.empty()`, and `client.read()` — the stub host-side probe path is gone from `cintx-cubecl`.
  2. `ResolvedBackend` enum dispatches kernel launches to either a wgpu or CPU CubeCL runtime arm; backend selection is determined by `BackendIntent` at executor init time.
  3. `RecordingExecutor` is deleted from `cintx-compat` and `cintx-rs`; staging output flows directly from the executor's `client.read()` result into `io.staging_output()`.
  4. CPU backend is enabled via `cpu = ["cubecl/cpu"]` feature in `cintx-cubecl` and oracle parity tests execute without GPU hardware under `--features cpu`.
  5. f64 strategy is documented and enforced: oracle parity tests run against the CPU backend; wgpu path gates on `SHADER_F64` availability and returns `UnsupportedApi` when absent.
**Plans**: 3/3 plans executed

Plans:
- [x] 07-01-PLAN.md — Add ResolvedBackend enum with wgpu/cpu arms, cpu feature flag, bytemuck dep, and updated FamilyLaunchFn signatures.
- [x] 07-02-PLAN.md — Rewrite CubeClExecutor to use ResolvedBackend dispatch, direct staging pass, and f64 SHADER_F64 capability gate.
- [x] 07-03-PLAN.md — Delete RecordingExecutor from cintx-compat and cintx-rs; wire eval_raw and safe facade to direct executor staging.

### Phase 8: Gaussian Primitive Infrastructure and Boys Function
**Goal**: All shared math required by integral kernels exists as validated `#[cube]` functions, confirmed against libcint reference values before any kernel consumes them.
**Depends on**: Phase 7
**Requirements**: MATH-01, MATH-02, MATH-03, MATH-04
**Success Criteria** (what must be TRUE):
  1. Boys function `Fm(x, m)` produces values matching `libcint-master/src/fmt.c` reference to within 1e-12 atol across the full domain used by 1e and 2e families, using upward recurrence for small `x` and asymptotic expansion for large `x`.
  2. Gaussian product center and pair data (`pdata`) computation produces correct overlap distribution exponents, centers, and pair weights for two-center and four-center shell pairs.
  3. Rys quadrature roots and weights match libcint `polyfits.c` reference coefficients for all quadrature degrees needed by 2e/2c2e/3c2e, with explicit coverage bounds documented.
  4. Obara-Saika horizontal and vertical recurrence `#[cube]` functions compile and link from inside kernel functions without E0433 errors, and produce correct auxiliary integrals for d-function test cases.
**Plans**: 4 plans

Plans:
- [x] 08-01-PLAN.md — Create math module with Boys function and PairData #[cube] implementations plus validation tests.
- [x] 08-02-PLAN.md — Implement Rys quadrature polynomial fit evaluation as #[cube] functions with validation tests.
- [x] 08-03-PLAN.md — Implement Obara-Saika vrr_step/hrr_step #[cube] functions and math integration test.
- [x] 08-04-PLAN.md — Close verification gaps: wire Rys-Boys crosscheck in integration test and fix MATH-03 tracking.

### Phase 9: 1e Real Kernel and Cart-to-Sph Transform
**Goal**: Users can execute real overlap, kinetic, and nuclear attraction evaluations that produce libcint-compatible spherical outputs, validating the entire compute pipeline end-to-end.
**Depends on**: Phase 8
**Requirements**: KERN-01, KERN-06, VERI-05
**Success Criteria** (what must be TRUE):
  1. `int1e_ovlp_sph` evaluation produces non-zero output values matching upstream libcint 6.1.3 to within atol 1e-11 / rtol 1e-9 for a standard H2O STO-3G test case under the CPU backend.
  2. `int1e_kin_sph` and `int1e_nuc_sph` evaluations pass oracle parity for the same test case at the same tolerances.
  3. Cart-to-sph transform uses real Condon-Shortley coefficients for all angular momenta up to g-function (l=4); staging buffer sizing is updated to reflect the correct spherical component count (5 for d-shell, not 6).
  4. Oracle parity CI gate for the 1e family passes under `--features cpu` with `mismatch_count == 0`.
**Plans**: 5 plans

Plans:
- [x] 09-01-PLAN.md — Implement Condon-Shortley cart-to-sph coefficients and transform function for l=0..4.
- [x] 09-02-PLAN.md — Replace 1e kernel stub with real overlap, kinetic, and nuclear attraction host-side pipeline.
- [x] 09-03-PLAN.md — Wire H2O STO-3G oracle parity tests for int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph.
- [x] 09-04-PLAN.md — Wire vendored libcint 6.1.3 compilation and true oracle parity comparison for 1e operators.
- [x] 09-05-PLAN.md — Update KERN-06 tracking and commit oracle parity artifact.

### Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure
**Goal**: All five base integral families produce real libcint-compatible values and the oracle parity gate closes across the full v1.1 compatibility matrix, completing the milestone.
**Depends on**: Phase 9
**Requirements**: KERN-02, KERN-03, KERN-04, KERN-05, VERI-05, VERI-07
**Success Criteria** (what must be TRUE):
  1. `int2e_sph` (four-center ERI) evaluation produces non-zero Rys quadrature results matching upstream libcint 6.1.3 to within atol 1e-12 / rtol 1e-10 for an H2O cc-pVDZ test case under the CPU backend.
  2. `int2c2e_sph`, `int3c1e_sph`, and `int3c2e_sph` evaluations pass oracle parity at their respective family tolerances (2c2e atol 1e-9, 3c1e atol 1e-7, 3c2e atol 1e-9).
  3. Oracle parity CI gate reports `mismatch_count == 0` for all five base families across all required profiles (base, with-f12, with-4c1e) under `--features cpu`.
  4. The two v1.0 human UAT items are resolved: `eval_raw()` returns non-zero values for a real basis set call, and the C ABI shim returns non-error status on a real GPU evaluation.
**Plans**: 6 plans

Plans:
- [x] 10-01-PLAN.md — Shared infrastructure: rys_root3-5 host wrappers, multi-index c2s transforms, oracle vendor build extension, vendor FFI.
- [x] 10-02-PLAN.md — 2c2e kernel implementation and oracle parity test.
- [x] 10-03-PLAN.md — 3c1e kernel implementation and oracle parity test.
- [x] 10-04-PLAN.md — 3c2e kernel implementation and oracle parity test.
- [x] 10-05-PLAN.md — 2e ERI kernel implementation and oracle parity test.
- [x] 10-06-PLAN.md — Oracle gate closure across all five families and v1.0 UAT item resolution.

---

## v1.2 Milestone: Full API Parity & Unified Oracle Gate

### Phase 11: Helper/Transform Completion & 4c1e Real Kernel
**Goal**: Every helper, transform, and wrapper symbol in the manifest is oracle-wired and returns libcint-compatible values; the 4c1e stub is replaced with a real Rys quadrature kernel within the Validated4C1E envelope.
**Depends on**: Phase 10
**Requirements**: HELP-01, HELP-02, HELP-03, HELP-04, 4C1E-01, 4C1E-02, 4C1E-03, 4C1E-04
**Success Criteria** (what must be TRUE):
  1. Oracle harness runs every helper symbol (count, offset, norm) against vendored libcint 6.1.3 using exact integer equality — not float tolerance — and reports 0 mismatches across all four feature profiles (HELP-01, HELP-04).
  2. Oracle harness runs every transform symbol and every legacy wrapper symbol against vendored libcint 6.1.3 at atol=1e-12 and reports 0 mismatches; the helper-legacy-parity CI gate passes (HELP-02, HELP-03, HELP-04).
  3. `int4c1e_sph` evaluation produces real Rys quadrature results matching libcint 6.1.3 to atol=1e-12 for cart/sph inputs within the Validated4C1E envelope (max(l)<=4, scalar representation) (4C1E-01).
  4. `compat::workaround::int4c1e_via_2e_trace` produces results matching direct 4c1e evaluation; oracle parity CI gate for the with-4c1e profile passes with 0 mismatches at atol=1e-12 (4C1E-02, 4C1E-04).
  5. Out-of-envelope 4c1e inputs and all spinor 4c1e requests return `UnsupportedApi` with explicit reason; the `Validated4C1E` classifier rejects spinor unconditionally before checking angular momentum (4C1E-03).
**Plans**: 4 plans

Plans:
- [x] 11-01-PLAN.md — Unify tolerance constants to atol=1e-12, fix CINTgto_norm formula, add numeric helper/transform oracle comparisons.
- [x] 11-02-PLAN.md — Replace 4c1e stub with real polynomial-recurrence G-tensor kernel and fix spinor-first validation ordering.
- [x] 11-03-PLAN.md — Add workaround module, legacy wrapper numeric oracle, vendor 4c1e FFI, and close all oracle gates.
- [x] 11-04-PLAN.md — Gap closure: add cart legacy symbol vendor FFI and numeric oracle comparison for full HELP-03 coverage.

### Phase 12: Real Spinor Transform (c2spinor Replacement)
**Goal**: The cart-to-spinor transform applies correct Clebsch-Gordan coupling coefficients for all angular momenta up to l=4, enabling oracle-verifiable spinor outputs for every base family that supports spinor representation.
**Depends on**: Phase 11
**Requirements**: SPIN-01, SPIN-02, SPIN-03, SPIN-04
**Success Criteria** (what must be TRUE):
  1. `c2spinor.rs` applies the correct Clebsch-Gordan coupling matrix from `c2spinor_coeffs.rs` for all (l, kappa) combinations up to l=4; the amplitude-averaging stub is fully removed and the old tests that only checked buffer length are replaced with value-correctness tests (SPIN-01).
  2. All four `CINTc2s_*spinor*` variants (ket_spinor, iket_spinor, ket_spinor_sf, ket_spinor_si) are implemented and reachable through the manifest dispatch; `kappa` parameter is correctly interpreted in transform selection (SPIN-02, SPIN-04).
  3. Spinor-form evaluations for the 1e family pass oracle parity against libcint 6.1.3 with 0 mismatches; spinor staging buffers are sized `spinor_component_count * 2` to accommodate interleaved real/imaginary doubles (SPIN-03).
  4. Spinor-form evaluations for 2e, 2c2e, 3c1e, and 3c2e families pass oracle parity against libcint 6.1.3 with 0 mismatches at family-appropriate tolerances (SPIN-03).
**Plans**: 5 plans

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
**Success Criteria** (what must be TRUE):
  1. `kernels/f12.rs` implements STG and YP as separate kernel entry points; the ibase/kbase routing divergence between STG and YP is handled without a shared code path; STG roots replicate the `t = min(t, 19682.99)` clamp from `CINTstg_roots` exactly (F12-01, F12-02).
  2. `ExecutionPlan` carries `operator_env_params` with `PTR_F12_ZETA` (env[9]); the validator rejects F12/STG/YP calls where `env[9] == 0.0` with a typed `InvalidEnvParam` error rather than silently falling back to plain Coulomb (F12-04, F12-05).
  3. All 10 with-f12 sph symbols pass oracle parity against libcint 6.1.3 at the family-appropriate tolerance; the oracle harness confirms that cart and spinor symbol counts for the with-f12 profile are zero (F12-03).
  4. Oracle fixtures validate that a call with `zeta=0` is either rejected by the validator or produces an explicit Coulomb-equivalent result with a documented contract — not a silent wrong result (F12-05).
**Plans**: 4 plans

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
**Plans**: TBD

### Phase 15: Oracle Tolerance Unification & Manifest Lock Closure
**Goal**: Every family passes oracle at the unified atol=1e-12 threshold; the four-profile manifest lock is regenerated after oracle parity is confirmed; and every `stability: Stable` manifest entry has `oracle_covered: true` with a passing CI record.
**Depends on**: Phase 14
**Requirements**: ORAC-01, ORAC-02, ORAC-03, ORAC-04
**Success Criteria** (what must be TRUE):
  1. The single oracle tolerance constant in `compare.rs` is atol=1e-12 for every family — no per-family exceptions, no design-doc overrides. Any family that fails at 1e-12 is treated as a kernel bug to be fixed, not a tolerance to be loosened (ORAC-01).
  2. All families — 1e, 2e, 2c2e, 3c1e, 3c2e, 4c1e, F12/STG/YP, and all unstable-source families — pass oracle at atol=1e-12. No existing base family regresses from the tolerance tightening (ORAC-04).
  3. `compiled_manifest.lock.json` is regenerated for all four profiles (base, with-f12, with-4c1e, with-f12+with-4c1e) after oracle parity is confirmed — not before; `manifest-audit` CI gate passes with zero diff (ORAC-02).
  4. CI oracle-parity gate passes all four profiles at atol=1e-12 under `--features cpu` with `mismatch_count == 0`; every `stability: Stable` manifest entry has `oracle_covered: true` (ORAC-03).
**Plans**: TBD
