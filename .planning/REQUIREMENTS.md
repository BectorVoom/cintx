# Requirements: cintx

**Defined:** 2026-03-21
**Core Value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

## v1 Requirements

### Foundations

- [x] **BASE-01**: Rust caller can model atoms, shells, basis sets, environment parameters, operators, and tensor layouts through explicit typed domain structures.
- [x] **BASE-02**: Maintainer can generate and lock a manifest-backed API inventory that classifies stable, optional, and unstable-source families across the supported feature matrix.
- [x] **BASE-03**: Rust caller can resolve supported integral families and representations through a manifest-aware registry without relying on raw symbol names.

### Compatibility

- [ ] **COMP-01**: Compat caller can invoke raw APIs using `atm`, `bas`, `env`, `shls`, `dims`, `opt`, and `cache` inputs that match documented layout contracts.
- [x] **COMP-02**: Compat caller can query required output sizes and workspace requirements without performing a full evaluation or writing output buffers.
- [x] **COMP-03**: Compat caller can use helper, transform, optimizer-lifecycle, and legacy wrapper APIs that are included in the upstream compatibility scope.
- [ ] **COMP-04**: C integrator can enable an optional C ABI shim that returns integer status codes and exposes thread-local last-error details.
- [ ] **COMP-05**: Compat caller receives typed validation failures or explicit `UnsupportedApi` errors instead of silent truncation, partial writes, or undefined behavior.

### Execution

- [x] **EXEC-01**: Rust caller can query workspace needs separately from evaluation through the safe API.
- [ ] **EXEC-02**: Rust or compat caller can evaluate supported 1e, 2e, 2c2e, 3c1e, and 3c2e families through the shared planner and CubeCL backend.
- [x] **EXEC-03**: Caller can enforce memory limits so large evaluations chunk safely or fail with typed memory-limit or allocation errors and no partial writes.
- [ ] **EXEC-04**: Caller receives outputs with upstream-compatible cart, sph, and spinor shapes, ordering, and complex-layout semantics.
- [ ] **EXEC-05**: Caller gets numerically equivalent results within accepted tolerance regardless of whether optimizer support is enabled.

### Optional Families

- [x] **OPT-01**: Caller can enable sph-only F12, STG, and YP families behind `with-f12`, and unsupported representations fail explicitly.
- [x] **OPT-02**: Caller can enable 4c1e behind `with-4c1e` only within the validated bug envelope, and out-of-envelope cases fail explicitly.
- [x] **OPT-03**: Maintainer can expose approved source-only families behind `unstable-source-api` without changing the stable GA surface.

### Verification

- [ ] **VERI-01**: Maintainer can compare stable and enabled optional APIs against vendored upstream libcint through oracle tests with family-appropriate tolerances.
- [x] **VERI-02**: CI can block manifest drift, helper/legacy parity regressions, CubeCL consistency failures, and OOM contract violations across the support matrix.
- [x] **VERI-03**: Maintainer can benchmark representative workloads and track throughput, memory, and CPU-GPU crossover regressions over time.
- [x] **VERI-04**: Maintainer can inspect planner, chunking, transfer, fallback, and OOM behavior through structured tracing and diagnostics.

## v2 Requirements

### Expanded Coverage

- **NEXT-01**: Caller can use cart and spinor representations for F12, STG, and YP families once the manifest, oracle, and feature-matrix coverage prove them stable.
- **NEXT-02**: Caller can use 4c1e beyond the initial validated bug envelope after dedicated oracle, identity, fuzz, and multi-device consistency gates pass.
- **NEXT-03**: Maintainer can promote selected source-only APIs from unstable to stable after repeated release-cycle verification.

### Deferred Product Surface

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

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Complete |
| BASE-02 | Phase 1 | Complete |
| BASE-03 | Phase 1 | Complete |
| COMP-01 | Phase 6 | Pending |
| COMP-02 | Phase 2 | Complete |
| COMP-03 | Phase 2 | Complete |
| COMP-04 | Phase 6 | Pending |
| COMP-05 | Phase 6 | Pending |
| EXEC-01 | Phase 3 | Complete |
| EXEC-02 | Phase 6 | Pending |
| EXEC-03 | Phase 2 | Complete |
| EXEC-04 | Phase 6 | Pending |
| EXEC-05 | Phase 6 | Pending |
| OPT-01 | Phase 3 | Complete |
| OPT-02 | Phase 3 | Complete |
| OPT-03 | Phase 3 | Complete |
| VERI-01 | Phase 6 | Pending |
| VERI-02 | Phase 4 | Complete |
| VERI-03 | Phase 4 | Complete |
| VERI-04 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 20 total
- Satisfied: 13
- Pending (Phase 6 gap closure): 7
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-04-02 after v1.0 milestone audit gap closure planning*
