# Roadmap

## Phases
- [x] **Phase 1: Manifest & Planner Foundation** - Lock down typed domain models, manifest registry, and planner scaffolding so everything else has a deterministic catalog to build against.
- [ ] **Phase 2: Execution & Compatibility Stabilization** - Wire the CubeCL-backed planner to the raw compat layer, including helper/legacy transforms, workspace queries, typed errors, and shape/optimizer guarantees.
- [ ] **Phase 3: Safe Surface, C ABI Shim & Optional Families** - Layer the safe Rust façade, optional C shim, and feature-gated optional families on the stabilized runtime.
- [ ] **Phase 4: Verification & Release Automation** - Close the manifest/oracle loop with CI, benchmarks, and diagnostics that block regressions before release.

## Phase Details

### Phase 1: Manifest & Planner Foundation
**Goal**: Establish the typed domain structures, manifest lock, registry, and planner foundations that every later layer consumes.
**Depends on**: Nothing
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Maintainers can instantiate atoms, shells, basis sets, environment parameters, operator IDs, and tensor layouts through the typed Rust structures defined in the manifest (BASE-01).
  2. The manifest generation pipeline emits a lock that classifies stable, optional, and unstable-source families across the support matrix and becomes the canonical input for downstream gating (BASE-02).
  3. The manifest-aware registry resolves which integral families and representations are available without depending on raw symbol names, so consumers can pick kernels declaratively (BASE-03).
**Plans**: TBD

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
- [ ] 07-PLAN.md - Add helper/transform/optimizer/legacy compat APIs and oracle parity coverage.
- [ ] 08-PLAN.md - Finish the `3c1e`/`3c2e` CubeCL kernels and cart/sph/spinor transform routing.

### Phase 3: Safe Surface, C ABI Shim & Optional Families
**Goal**: Expose the safe Rust façade, optional C shim, and gated optional families once the runtime is stable.
**Depends on**: Phase 2
**Requirements**: EXEC-01, COMP-04, OPT-01, OPT-02, OPT-03
**Success Criteria** (what must be TRUE):
  1. The safe Rust API splits `query_workspace()` from `evaluate()`, letting callers observe workspace needs before committing to execution (EXEC-01).
  2. The optional C ABI shim accepts compat-style inputs, returns integer status codes, and exposes thread-local last-error details for C integrators (COMP-04).
  3. `with-f12`, `with-4c1e`, and other optional-family gates only enable validated envelopes and emit `UnsupportedApi` for out-of-envelope arguments (OPT-01, OPT-02).
  4. Source-only APIs stay behind `unstable-source-api` so the GA surface remains stable until the maintainer intentionally enables those symbols (OPT-03).
**Plans**: TBD

### Phase 4: Verification & Release Automation
**Goal**: Close the manifest/oracle verification loop, run multi-profile CI/benchmarks, and surface diagnostics that block regressions before release.
**Depends on**: Phase 3
**Requirements**: VERI-01, VERI-02, VERI-03, VERI-04
**Success Criteria** (what must be TRUE):
  1. The oracle suite compares the stable and optional APIs against upstream libcint per manifest family with documented tolerances and flags regressions (VERI-01).
  2. CI workflows block manifest drift, helper/legacy parity regressions, CubeCL consistency failures, and OOM contract violations before merges land (VERI-02).
  3. Benchmarks capture throughput, memory usage, and CPU-GPU crossover regressions for trend tracking (VERI-03).
  4. Tracing and diagnostics expose planner chunking, fallback, transfer, and OOM behavior for manual inspection (VERI-04).
**Plans**: TBD

## Progress
| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| Phase 1: Manifest & Planner Foundation | 2/2 | Complete | 2026-03-21 |
| Phase 2: Execution & Compatibility Stabilization | 5/7 | In Progress | - |
| Phase 3: Safe Surface, C ABI Shim & Optional Families | 0/0 | Not started | - |
| Phase 4: Verification & Release Automation | 0/0 | Not started | - |
