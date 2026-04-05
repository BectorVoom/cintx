# Requirements: cintx

**Defined:** 2026-03-21
**Core Value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

## v1 Requirements

### Foundations

- [x] **BASE-01**: Rust caller can model atoms, shells, basis sets, environment parameters, operators, and tensor layouts through explicit typed domain structures.
- [x] **BASE-02**: Maintainer can generate and lock a manifest-backed API inventory that classifies stable, optional, and unstable-source families across the supported feature matrix.
- [x] **BASE-03**: Rust caller can resolve supported integral families and representations through a manifest-aware registry without relying on raw symbol names.

### Compatibility

- [x] **COMP-01**: Compat caller can invoke raw APIs using `atm`, `bas`, `env`, `shls`, `dims`, `opt`, and `cache` inputs that match documented layout contracts.
- [x] **COMP-02**: Compat caller can query required output sizes and workspace requirements without performing a full evaluation or writing output buffers.
- [x] **COMP-03**: Compat caller can use helper, transform, optimizer-lifecycle, and legacy wrapper APIs that are included in the upstream compatibility scope.
- [x] **COMP-04**: C integrator can enable an optional C ABI shim that returns integer status codes and exposes thread-local last-error details.
- [x] **COMP-05**: Compat caller receives typed validation failures or explicit `UnsupportedApi` errors instead of silent truncation, partial writes, or undefined behavior.

### Execution

- [x] **EXEC-01**: Rust caller can query workspace needs separately from evaluation through the safe API.
- [x] **EXEC-02**: Rust or compat caller can evaluate supported 1e, 2e, 2c2e, 3c1e, and 3c2e families through the shared planner and CubeCL backend.
- [x] **EXEC-03**: Caller can enforce memory limits so large evaluations chunk safely or fail with typed memory-limit or allocation errors and no partial writes.
- [x] **EXEC-04**: Caller receives outputs with upstream-compatible cart, sph, and spinor shapes, ordering, and complex-layout semantics.
- [x] **EXEC-05**: Caller gets numerically equivalent results within accepted tolerance regardless of whether optimizer support is enabled.

### Optional Families

- [x] **OPT-01**: Caller can enable sph-only F12, STG, and YP families behind `with-f12`, and unsupported representations fail explicitly.
- [x] **OPT-02**: Caller can enable 4c1e behind `with-4c1e` only within the validated bug envelope, and out-of-envelope cases fail explicitly.
- [x] **OPT-03**: Maintainer can expose approved source-only families behind `unstable-source-api` without changing the stable GA surface.

### Verification

- [x] **VERI-01**: Maintainer can compare stable and enabled optional APIs against vendored upstream libcint through oracle tests with family-appropriate tolerances.
- [x] **VERI-02**: CI can block manifest drift, helper/legacy parity regressions, CubeCL consistency failures, and OOM contract violations across the support matrix.
- [x] **VERI-03**: Maintainer can benchmark representative workloads and track throughput, memory, and CPU-GPU crossover regressions over time.
- [x] **VERI-04**: Maintainer can inspect planner, chunking, transfer, fallback, and OOM behavior through structured tracing and diagnostics.

## v1.1 Requirements

### Executor Infrastructure

- [x] **EXEC-06**: Executor internals use CubeCL client API directly (`WgpuRuntime::client()`, `client.create()`/`client.read()`/`client.empty()`, `ArrayArg::from_raw_parts`)
- [x] **EXEC-07**: RecordingExecutor removed from cintx-compat and cintx-rs — real kernel values flow through `io.staging_output()` directly
- [x] **EXEC-08**: ResolvedBackend enum dispatches between Wgpu and Cpu runtime arms with per-arm kernel launch
- [x] **EXEC-09**: CPU backend enabled via `cpu = ["cubecl/cpu"]` feature in cintx-cubecl for CI oracle testing without GPU

### Kernel Compute

- [x] **KERN-01**: 1e family kernels (overlap, kinetic, nuclear attraction) produce real values via `#[cube(launch)]`
- [x] **KERN-02**: 2e ERI kernel implements Rys quadrature with real Gaussian integral evaluation
- [x] **KERN-03**: 2c2e two-center two-electron kernel produces real values
- [x] **KERN-04**: 3c1e three-center one-electron kernel produces real values
- [x] **KERN-05**: 3c2e three-center two-electron kernel produces real values
- [x] **KERN-06**: Cart-to-sph transform implements real Condon-Shortley coefficients replacing stub blend

### Math Infrastructure

- [x] **MATH-01**: Boys function implemented as `#[cube]` functions with gridded Taylor expansion uploaded to device
- [x] **MATH-02**: Gaussian primitive pair evaluation (overlap distribution, screening) implemented as `#[cube]` functions
- [x] **MATH-03**: Rys quadrature roots and weights computed on-device via polynomial fit tables
- [x] **MATH-04**: Obara-Saika horizontal and vertical recurrence relations implemented as `#[cube]` functions

### Verification (v1.1)

- [x] **VERI-05**: Oracle parity verified per family as each kernel lands (not deferred to end)
- [x] **VERI-06**: f64 precision strategy resolved — CPU backend as primary oracle path; wgpu SHADER_F64 tested opportunistically
- [x] **VERI-07**: v1.0 human UAT items (non-zero eval_raw output, C ABI shim output on real GPU) resolved

## v1.2 Requirements

### Helper & Transform Completion

- [x] **HELP-01**: Oracle harness compares every helper symbol in the manifest against vendored libcint with atol=1e-12
- [x] **HELP-02**: Oracle harness compares every transform symbol in the manifest against vendored libcint with atol=1e-12
- [x] **HELP-03**: Oracle harness compares every legacy wrapper symbol in the manifest against vendored libcint with atol=1e-12
- [x] **HELP-04**: CI helper-legacy-parity gate passes with 0 mismatches across all four feature profiles

### 4c1e Kernel & Oracle

- [x] **4C1E-01**: int4c1e_sph evaluation produces real Rys quadrature results matching libcint 6.1.3 to atol=1e-12 within Validated4C1E envelope
- [x] **4C1E-02**: int4c1e_via_2e_trace workaround path produces results matching direct 4c1e evaluation
- [x] **4C1E-03**: Out-of-envelope 4c1e inputs return UnsupportedApi; spinor 4c1e returns UnsupportedApi unconditionally
- [x] **4C1E-04**: Oracle parity CI gate for with-4c1e profile passes with 0 mismatches at atol=1e-12

### Spinor Representation

- [ ] **SPIN-01**: Cart-to-spinor transform implements real Clebsch-Gordan coupling coefficients for all angular momenta up to g-function (l=4)
- [ ] **SPIN-02**: All CINTc2s_*spinor* transform variants (ket_spinor, iket_spinor, ket_spinor_sf, ket_spinor_si) are implemented
- [x] **SPIN-03**: Spinor-form base family evaluations (1e, 2e, 2c2e, 3c1e, 3c2e spinor) match libcint to atol=1e-12
- [ ] **SPIN-04**: kappa parameter is correctly interpreted and applied in spinor transform dispatch

### F12/STG/YP Kernels

- [x] **F12-01**: STG (Slater-type geminal) kernel implements modified Rys quadrature with tabulated polynomial roots matching libcint
- [x] **F12-02**: YP (Yukawa potential) kernel implements correct routing distinct from STG path
- [x] **F12-03**: All 10 with-f12 sph symbols pass oracle parity against libcint at atol=1e-12
- [x] **F12-04**: PTR_F12_ZETA (env[9]) is correctly plumbed through ExecutionPlan to kernel launchers
- [x] **F12-05**: Oracle fixtures validate that zeta=0 is rejected or produces Coulomb-equivalent results explicitly

### Unstable-Source API

- [ ] **USRC-01**: origi family (4 symbols, 1e) implemented behind unstable-source-api gate with oracle parity at atol=1e-12
- [ ] **USRC-02**: grids family (1e grid-based integrals) implemented with NGRIDS/PTR_GRIDS env parsing and oracle parity at atol=1e-12
- [ ] **USRC-03**: Breit family (2 symbols, 2e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [ ] **USRC-04**: origk family (6 symbols, 3c1e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [ ] **USRC-05**: ssc family (1 symbol, 3c2e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [ ] **USRC-06**: Nightly CI job runs oracle with --include-unstable-source=true and 0 mismatches

### Oracle & Tolerance Unification

- [ ] **ORAC-01**: Oracle tolerance unified to atol=1e-12 for every family with no per-family exceptions
- [ ] **ORAC-02**: Four-profile manifest lock regenerated covering all implemented APIs
- [ ] **ORAC-03**: CI oracle-parity gate passes all four profiles (base, with-f12, with-4c1e, with-f12+with-4c1e) at atol=1e-12
- [ ] **ORAC-04**: Existing base families (1e, 2e, 2c2e, 3c1e, 3c2e) pass oracle at tightened atol=1e-12

## v2 Requirements

### Expanded Coverage

- **NEXT-04**: Rust caller can use richer builder ergonomics and convenience APIs once the core compatibility surface is stable.
- **NEXT-05**: Maintainer can add deeper benchmark reporting and public performance dashboards once correctness and release gating are stable.
- **NEXT-06**: Project can consider additional compute backends or fallback strategies only if CubeCL becomes a sustained correctness or maintainability blocker.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Public GTG support | Explicitly excluded from initial GA because upstream marks GTG deprecated and incorrect |
| Bitwise-identical libcint internals | The project targets result compatibility, not implementation identity |
| Public Fortran wrapper reproduction | Not part of the Rust library's migration or compatibility goals |
| Public asynchronous API | Excluded from the initial design to keep execution, allocation, and compatibility behavior predictable |
| Best-effort partial writes on failure | Violates the OOM-safe stop and explicit-layout contract |
| CUDA/ROCm/Metal backend implementation | Architecture supports them via ResolvedBackend, but only wgpu+cpu in scope |
| h-function (l>=5) angular momentum | Register pressure risk, defer until g-function validated across all families |
| Screening/batching optimizations | Performance work after correctness is proven |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Complete |
| BASE-02 | Phase 1 | Complete |
| BASE-03 | Phase 1 | Complete |
| COMP-01 | Phase 6 | Complete |
| COMP-02 | Phase 2 | Complete |
| COMP-03 | Phase 2 | Complete |
| COMP-04 | Phase 6 | Complete |
| COMP-05 | Phase 6 | Complete |
| EXEC-01 | Phase 3 | Complete |
| EXEC-02 | Phase 6 | Complete |
| EXEC-03 | Phase 2 | Complete |
| EXEC-04 | Phase 6 | Complete |
| EXEC-05 | Phase 6 | Complete |
| OPT-01 | Phase 3 | Complete |
| OPT-02 | Phase 3 | Complete |
| OPT-03 | Phase 3 | Complete |
| VERI-01 | Phase 6 | Complete |
| VERI-02 | Phase 4 | Complete |
| VERI-03 | Phase 4 | Complete |
| VERI-04 | Phase 4 | Complete |
| EXEC-06 | Phase 7 | Complete |
| EXEC-07 | Phase 7 | Complete |
| EXEC-08 | Phase 7 | Complete |
| EXEC-09 | Phase 7 | Complete |
| MATH-01 | Phase 8 | Complete |
| MATH-02 | Phase 8 | Complete |
| MATH-03 | Phase 8 | Complete |
| MATH-04 | Phase 8 | Complete |
| KERN-01 | Phase 9 | Complete |
| KERN-02 | Phase 10 | Complete |
| KERN-03 | Phase 10 | Complete |
| KERN-04 | Phase 10 | Complete |
| KERN-05 | Phase 10 | Complete |
| KERN-06 | Phase 9 | Complete |
| VERI-05 | Phase 9 | Complete |
| VERI-06 | Phase 7 | Complete |
| VERI-07 | Phase 10 | Complete |
| HELP-01 | Phase 11 | Complete |
| HELP-02 | Phase 11 | Complete |
| HELP-03 | Phase 11 | Complete |
| HELP-04 | Phase 11 | Complete |
| 4C1E-01 | Phase 11 | Complete |
| 4C1E-02 | Phase 11 | Complete |
| 4C1E-03 | Phase 11 | Complete |
| 4C1E-04 | Phase 11 | Complete |
| SPIN-01 | Phase 12 | Pending |
| SPIN-02 | Phase 12 | Pending |
| SPIN-03 | Phase 12 | Complete |
| SPIN-04 | Phase 12 | Pending |
| F12-01 | Phase 13 | Complete |
| F12-02 | Phase 13 | Complete |
| F12-03 | Phase 13 | Complete |
| F12-04 | Phase 13 | Complete |
| F12-05 | Phase 13 | Complete |
| USRC-01 | Phase 14 | Pending |
| USRC-02 | Phase 14 | Pending |
| USRC-03 | Phase 14 | Pending |
| USRC-04 | Phase 14 | Pending |
| USRC-05 | Phase 14 | Pending |
| USRC-06 | Phase 14 | Pending |
| ORAC-01 | Phase 15 | Pending |
| ORAC-02 | Phase 15 | Pending |
| ORAC-03 | Phase 15 | Pending |
| ORAC-04 | Phase 15 | Pending |

**Coverage:**
- v1.0 requirements: 20 total (all complete)
- v1.1 requirements: 17 total (all complete)
- v1.2 requirements: 27 total (note: coverage line previously stated 30; actual count from requirement IDs is 27)
- Mapped to phases: 27/27 (Phases 11-15)
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-04-04 after v1.2 roadmap creation — all 27 v1.2 requirements mapped to Phases 11-15*
