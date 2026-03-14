# Requirements: libcint-rs

**Defined:** 2026-03-14
**Core Value:** Users can compute libcint-equivalent integrals through a Rust-native library with explicit safety/error guarantees and verifiable compatibility gates.

## v1 Requirements

### Compatibility Core

- [ ] **COMP-01**: User can compute stable-family integrals (`1e/2e/2c2e/3c1e/3c2e`) with results matching oracle tolerances for cart/sph/spinor representations
- [ ] **COMP-02**: User gets parity for helper/transform essentials (AO counts, offsets, normalization, cart/sph/spinor transforms) required by migration workflows
- [ ] **COMP-03**: User can rely on manifest-backed API coverage where each supported symbol maps to an explicit profile and stability level
- [ ] **COMP-04**: User can trust compatibility claims because each stable-family requirement is validated by automated oracle regression gates

### Raw Compatibility API

- [ ] **RAW-01**: User can call a raw API surface with `atm/bas/env`, `shls`, `dims`, `cache`, and `opt` contracts compatible with libcint usage patterns
- [ ] **RAW-02**: User can query required workspace when output/cache pointers are null-equivalent and then execute successfully with provided buffers
- [ ] **RAW-03**: User gets explicit error when provided `dims`/buffer shape is incompatible; partial writes and silent truncation do not occur
- [ ] **RAW-04**: User receives numerically equivalent results with and without optimizer usage for supported operators

### Safe Rust API

- [ ] **SAFE-01**: User can construct typed input models (`Atom`, `Shell`, basis/environment context) without raw pointer arithmetic
- [ ] **SAFE-02**: User can call `query_workspace` to obtain deterministic workspace requirements before evaluation
- [ ] **SAFE-03**: User can call `evaluate`/`evaluate_into` with typed tensor views and receive representation-correct output layout
- [ ] **SAFE-04**: User receives typed errors that distinguish unsupported API, input-layout failure, memory failure, and backend execution failure

### Memory and Error Guarantees

- [ ] **MEM-01**: User can set `memory_limit_bytes` and get either chunked execution or explicit `MemoryLimitExceeded` without process abort
- [ ] **MEM-02**: User never experiences unhandled OOM abort in supported execution paths because large allocations use fallible allocation policy
- [ ] **MEM-03**: User can diagnose failure causes through structured error messages and trace metadata

### Execution Backends

- [ ] **EXEC-01**: User can run all v1-supported requirements on CPU reference backend as the correctness baseline
- [ ] **EXEC-02**: User can enable CubeCL acceleration via feature flag and receive deterministic CPU fallback when workloads are unsupported/unfavorable
- [ ] **EXEC-03**: User can inspect backend dispatch reason (CPU/GPU/fallback) through tracing output

### Optional and Migration Surfaces

- [ ] **ABIC-01**: User can opt into a C ABI shim that exposes status-code based calls and last-error retrieval for phased migration
- [ ] **OPTF-01**: User can opt into `with-f12` support only within documented supported envelopes; unsupported representations return explicit errors
- [ ] **OPTF-02**: User can opt into `with-4c1e` only within validated envelope; out-of-envelope requests return `UnsupportedApi`

### Verification and Release Gates

- [ ] **VERI-01**: User can trust releases because CI fails on unapproved compiled-manifest lock drift across support profiles
- [ ] **VERI-02**: User gets regression protection from oracle CI matrix covering base and optional supported profiles
- [ ] **VERI-03**: User gets validated layout and failure semantics through dedicated tests for spinor/complex layout, helper parity, and OOM/error paths

## v2 Requirements

### Future Extensions

- **ASYNC-01**: User can optionally use async execution APIs once core parity and observability are stable
- **UNST-01**: User can consume promoted source-only unstable families after multi-release validation evidence
- **GTG-01**: User can evaluate GTG families only if independently implemented and fully validated against explicit acceptance criteria
- **AUTO-01**: User gets hardware-adaptive GPU threshold auto-tuning after stable benchmark corpus is established

## Out of Scope

| Feature | Reason |
|---------|--------|
| Bitwise identity with libcint internals | Project targets numerical-result compatibility, not implementation equivalence |
| Reproducing upstream internal scratch/layout implementation details | Internal strategy is allowed to differ if external contracts and results hold |
| Public async API in initial release | Adds complexity without improving compatibility baseline; deferred to v2 |
| GTG in initial GA surface | Upstream marks GTG path as deprecated/incorrect; excluded from initial scope |
| Fortran wrapper parity in initial release | Migration priority is Rust safe API + raw compatibility + optional C ABI |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| COMP-01 | Phase 2 | Pending |
| COMP-02 | Phase 3 | Pending |
| COMP-03 | Phase 3 | Pending |
| COMP-04 | Phase 3 | Pending |
| RAW-01 | Phase 2 | Pending |
| RAW-02 | Phase 2 | Pending |
| RAW-03 | Phase 2 | Pending |
| RAW-04 | Phase 3 | Pending |
| SAFE-01 | Phase 1 | Pending |
| SAFE-02 | Phase 1 | Pending |
| SAFE-03 | Phase 2 | Pending |
| SAFE-04 | Phase 1 | Pending |
| MEM-01 | Phase 2 | Pending |
| MEM-02 | Phase 2 | Pending |
| MEM-03 | Phase 1 | Pending |
| EXEC-01 | Phase 2 | Pending |
| EXEC-02 | Phase 4 | Pending |
| EXEC-03 | Phase 4 | Pending |
| ABIC-01 | Phase 4 | Pending |
| OPTF-01 | Phase 4 | Pending |
| OPTF-02 | Phase 4 | Pending |
| VERI-01 | Phase 3 | Pending |
| VERI-02 | Phase 3 | Pending |
| VERI-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after roadmap mapping*
