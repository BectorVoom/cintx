# Roadmap: libcint-rs

## Overview

This roadmap delivers libcint-compatible results in a Rust-first library by sequencing typed contract foundations, CPU correctness parity, compatibility proof gates, and finally optional acceleration and migration surfaces.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Contracts and Typed Foundations** - Establish safe typed models, workspace introspection, and typed error diagnostics.
- [ ] **Phase 2: CPU Compatibility Execution** - Deliver stable-family CPU execution through safe and raw APIs with memory guarantees.
- [ ] **Phase 3: Verification and Compatibility Governance** - Prove compatibility claims with helper parity, manifest governance, and CI release gates.
- [ ] **Phase 4: Optional Backends and Migration Surfaces** - Add opt-in GPU acceleration, C ABI migration shim, and optional family support envelopes.

## Phase Details

### Phase 1: Contracts and Typed Foundations
**Goal**: Users can prepare and validate integral requests through typed Rust contracts with deterministic workspace requirements and diagnosable failures.
**Depends on**: Nothing (first phase)
**Requirements**: SAFE-01, SAFE-02, SAFE-04, MEM-03
**Success Criteria** (what must be TRUE):
  1. User can construct `Atom`, `Shell`, and basis/environment inputs through typed APIs without raw pointer arithmetic.
  2. User can call `query_workspace` and receive deterministic workspace sizing before evaluation.
  3. User receives typed errors that distinguish unsupported API, input-layout failure, memory failure, and backend execution failure.
  4. User can diagnose failures via structured error messages and trace metadata.
**Plans**: 2
- [ ] 01-01: Typed domain contracts and error taxonomy
- [ ] 01-02: Workspace query and diagnostics contract

### Phase 2: CPU Compatibility Execution
**Goal**: Users can run supported stable-family integrals on CPU via both safe and raw interfaces while preserving explicit memory-limit behavior.
**Depends on**: Phase 1
**Requirements**: COMP-01, RAW-01, RAW-02, RAW-03, SAFE-03, MEM-01, MEM-02, EXEC-01
**Success Criteria** (what must be TRUE):
  1. User can evaluate stable-family integrals (`1e/2e/2c2e/3c1e/3c2e`) through safe APIs and get oracle-tolerance results across cart/sph/spinor representations.
  2. User can call raw APIs with libcint-compatible `atm/bas/env`, `shls`, `dims`, `cache`, and `opt` contracts, including workspace query then execution flow.
  3. User gets explicit failures for incompatible `dims` or buffer shapes, and the runtime never performs silent truncation or partial writes.
  4. User can set `memory_limit_bytes` and observe chunked execution or explicit `MemoryLimitExceeded`, with no unhandled OOM abort in supported paths.
**Plans**: TBD

### Phase 3: Verification and Compatibility Governance
**Goal**: Users can trust compatibility claims because helper parity, API coverage claims, and regression protection are automated and enforceable.
**Depends on**: Phase 2
**Requirements**: COMP-02, COMP-03, COMP-04, RAW-04, VERI-01, VERI-02, VERI-03
**Success Criteria** (what must be TRUE):
  1. User gets helper/transform parity (AO counts, offsets, normalization, cart/sph/spinor transforms) required by migration workflows.
  2. User can inspect manifest-backed API coverage where each supported symbol is tied to an explicit support profile and stability level.
  3. User can trust releases because CI blocks unapproved compiled-manifest lock drift and oracle-regression failures across supported profiles.
  4. User receives regression protection for optimizer on/off numerical equivalence plus spinor/layout and OOM/error-path semantics.
**Plans**: TBD

### Phase 4: Optional Backends and Migration Surfaces
**Goal**: Users can opt into acceleration and migration extensions without weakening baseline safety, compatibility, or explicit support boundaries.
**Depends on**: Phase 3
**Requirements**: EXEC-02, EXEC-03, ABIC-01, OPTF-01, OPTF-02
**Success Criteria** (what must be TRUE):
  1. User can enable CubeCL acceleration through feature flags and gets deterministic CPU fallback when workloads are unsupported or unfavorable.
  2. User can inspect backend dispatch and fallback reasons through tracing output for each execution path.
  3. User can opt into a C ABI shim with status-code calls and last-error retrieval for phased migration.
  4. User can opt into `with-f12` and `with-4c1e` only within validated support envelopes, and out-of-envelope usage returns explicit unsupported errors.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Contracts and Typed Foundations | 0/2 | Not started | - |
| 2. CPU Compatibility Execution | 0/TBD | Not started | - |
| 3. Verification and Compatibility Governance | 0/TBD | Not started | - |
| 4. Optional Backends and Migration Surfaces | 0/TBD | Not started | - |
