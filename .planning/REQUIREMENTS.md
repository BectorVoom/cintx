# Requirements: cintx

**Defined:** 2026-03-21
**Core Value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

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

- [x] **USRC-01**: origi family (4 symbols, 1e) implemented behind unstable-source-api gate with oracle parity at atol=1e-12
- [x] **USRC-02**: grids family (1e grid-based integrals) implemented with NGRIDS/PTR_GRIDS env parsing and oracle parity at atol=1e-12
- [x] **USRC-03**: Breit family (2 symbols, 2e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [x] **USRC-04**: origk family (6 symbols, 3c1e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [x] **USRC-05**: ssc family (1 symbol, 3c2e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [x] **USRC-06**: Nightly CI job runs oracle with --include-unstable-source=true and 0 mismatches

### Oracle & Tolerance Unification

- [x] **ORAC-01**: Oracle tolerance unified to atol=1e-12 for every family with no per-family exceptions
- [ ] **ORAC-02**: Four-profile manifest lock regenerated covering all implemented APIs
- [ ] **ORAC-03**: CI oracle-parity gate passes all four profiles (base, with-f12, with-4c1e, with-f12+with-4c1e) at atol=1e-12
- [x] **ORAC-04**: Existing base families (1e, 2e, 2c2e, 3c1e, 3c2e) pass oracle at tightened atol=1e-12

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
| USRC-01 | Phase 14 | Complete |
| USRC-02 | Phase 14 | Complete |
| USRC-03 | Phase 14 | Complete |
| USRC-04 | Phase 14 | Complete |
| USRC-05 | Phase 14 | Complete |
| USRC-06 | Phase 14 | Complete |
| ORAC-01 | Phase 15 | Complete |
| ORAC-02 | Phase 15 | Pending |
| ORAC-03 | Phase 15 | Pending |
| ORAC-04 | Phase 15 | Complete |

**Coverage:**
- v1.2 requirements: 27 total (Phases 11-15)
- Complete: 17/27
- Pending: 10/27

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-04-05 after v1.1 milestone complete — v1.0/v1.1 requirements archived to milestones/*
