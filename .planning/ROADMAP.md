# Roadmap

## Phases
- [x] **Phase 1: Manifest & Planner Foundation** - Lock down typed domain models, manifest registry, and planner scaffolding so everything else has a deterministic catalog to build against.
- [x] **Phase 2: Execution & Compatibility Stabilization** - Wire the CubeCL-backed planner to the raw compat layer, including helper/legacy transforms, workspace queries, typed errors, and shape/optimizer guarantees.
- [x] **Phase 3: Safe Surface, C ABI Shim & Optional Families** - Layer the safe Rust facade, optional C shim, and feature-gated optional families on the stabilized runtime.
- [x] **Phase 4: Verification & Release Automation** - Close the manifest/oracle loop with CI, benchmarks, and diagnostics that block regressions before release.
- [x] **Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend)** - Replace synthetic execution with a real wgpu-backed CubeCL path and capability-aware fail-closed verification.
- [ ] **Phase 6: Fix raw eval staging retrieval and capability fingerprint propagation** - Close audit gaps: wire eval_raw() staging output retrieval, propagate wgpu fingerprint into capability token, add regression tests.
- [ ] **Phase 7: Executor Infrastructure Rewrite** - Replace executor internals with direct CubeCL client API, introduce ResolvedBackend dispatch, CPU backend feature, and f64 strategy decision — prerequisite gate for all real kernel work.
- [x] **Phase 8: Gaussian Primitive Infrastructure and Boys Function** - Build shared math foundation as `#[cube]` functions: Boys function, Rys roots/weights, primitive pair evaluation, and Obara-Saika recurrence. (completed 2026-04-03)
- [ ] **Phase 9: 1e Real Kernel and Cart-to-Sph Transform** - Implement real overlap, kinetic, and nuclear attraction kernels with correct Condon-Shortley c2s transform, validating the end-to-end compute pipeline.
- [ ] **Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure** - Implement all remaining integral family kernels and close the oracle parity gate for all five base families, completing v1.1.

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
| Phase 6: Fix raw eval staging & fingerprint | 0/2 | Not started | - |
| Phase 7: Executor Infrastructure Rewrite | 1/3 | In Progress | 2026-04-02 |
| Phase 8: Gaussian Primitive Infrastructure and Boys Function | 3/4 | In Progress | - |
| Phase 9: 1e Real Kernel and Cart-to-Sph Transform | 3/5 | In Progress | 2026-04-03 |
| Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure | 0/6 | Not started | - |

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
**Plans:** 1/2 plans executed

Plans:
- [x] 06-01-PLAN.md — Fix eval_raw() staging retrieval with RecordingExecutor and propagate wgpu fingerprint in compat raw and safe facade paths.
- [ ] 06-02-PLAN.md — Add regression tests for staging retrieval, fingerprint propagation, base family coverage, and deterministic output.

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
**Plans**: 3 plans

Plans:
- [ ] 07-01-PLAN.md — Add ResolvedBackend enum with wgpu/cpu arms, cpu feature flag, bytemuck dep, and updated FamilyLaunchFn signatures.
- [ ] 07-02-PLAN.md — Rewrite CubeClExecutor to use ResolvedBackend dispatch, direct staging pass, and f64 SHADER_F64 capability gate.
- [ ] 07-03-PLAN.md — Delete RecordingExecutor from cintx-compat and cintx-rs; wire eval_raw and safe facade to direct executor staging.

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
- [ ] 10-04-PLAN.md — 3c2e kernel implementation and oracle parity test.
- [ ] 10-05-PLAN.md — 2e ERI kernel implementation and oracle parity test.
- [ ] 10-06-PLAN.md — Oracle gate closure across all five families and v1.0 UAT item resolution.
