# Phase 4: Verification & Release Automation - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 4 closes the verification and release loop for cintx: oracle comparison, feature-matrix CI gates, benchmark regression tracking, and runtime diagnostics visibility that block compatibility regressions before release.

This phase clarifies how strict the verification and release automation should be. It does not add new integral families or expand runtime capability scope.

</domain>

<decisions>
## Implementation Decisions

### Oracle Comparison Policy
- **D-01:** Merge-blocking oracle coverage includes stable APIs and optional-profile APIs when those profiles are enabled in matrix jobs.
- **D-02:** Family tolerances stay as an explicit per-family table in code and must be changed only through deliberate reviewed updates.
- **D-03:** Optional-family oracle checks are required; unstable-source oracle checks remain extended/nightly coverage rather than default merge blockers.
- **D-04:** Oracle jobs should emit complete mismatch reports across the full fixture set before failing (no first-mismatch fail-fast mode).

### CI Gate and Matrix Policy
- **D-05:** Required merge-blocking PR gates include manifest drift checks, oracle parity checks, helper/legacy parity checks, and OOM-contract checks.
- **D-06:** Required feature-matrix verification covers all approved profiles: `base`, `with-f12`, `with-4c1e`, and `with-f12+with-4c1e`.
- **D-07:** GPU consistency/benchmark jobs are advisory on PRs but required in scheduled/merge-queue verification flows.
- **D-08:** Required verification gate failures block merges (normal infra reruns allowed, but no policy-level bypass).

### Benchmark and Diagnostics Policy
- **D-09:** Benchmark automation runs on nightly and release-oriented workflows, not on every merge-blocking PR.
- **D-10:** Phase 4 baseline suites include micro family benchmarks, macro molecule benchmarks, and CPU-GPU crossover tracking.
- **D-11:** After baselines are established, benchmark gates fail only when regressions exceed defined thresholds (not report-only and not any-slowdown-fails).
- **D-12:** Verification workflows persist structured trace+metrics diagnostics (planner/chunk/fallback/transfer/OOM) with artifactized outputs honoring required `/mnt/data` paths.

### Carried Forward from Prior Phases
- **D-13:** Verification remains fail-closed: unsupported envelopes return explicit `UnsupportedApi` and no partial-write behavior is allowed.
- **D-14:** The compiled manifest lock remains the API source of truth; release automation must gate against lock/coverage drift.

### the agent's Discretion
- Exact numeric benchmark threshold values and warmup/stabilization policy before hard enforcement.
- Concrete CI workflow decomposition (job fan-out, retry budget, and naming), as long as D-05 through D-08 hold.
- Final artifact JSON schema field names and xtask command UX for report generation.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope and Requirements
- `.planning/ROADMAP.md` - Phase 4 goal, requirement IDs, and success criteria.
- `.planning/REQUIREMENTS.md` - `VERI-01`, `VERI-02`, `VERI-03`, and `VERI-04` requirement contracts.
- `.planning/PROJECT.md` - compatibility-first and verification/release constraints.
- `.planning/STATE.md` - current milestone state and phase transition context.

### Design Authority
- `docs/design/cintx_detailed_design.md` Section 3.2 and Section 3.3 - manifest inventory and lock governance.
- `docs/design/cintx_detailed_design.md` Section 10.1 - feature matrix and optional-family profile envelope.
- `docs/design/cintx_detailed_design.md` Section 12.5 - validated 4c1e bug-envelope verification policy.
- `docs/design/cintx_detailed_design.md` Section 13.4 - helper/transform parity and oracle comparison gate expectations.
- `docs/design/cintx_detailed_design.md` Section 14.1 - release gate and regression-blocking policy.
- `docs/design/cintx_detailed_design.md` Section 16.2, Section 16.4, and Section 16.5 - release checklist and promotion criteria tied to verification evidence.

### Existing Oracle and Runtime Contracts
- `crates/cintx-oracle/src/compare.rs` - current parity harness structure, tolerance table, and artifact reporting behavior.
- `crates/cintx-oracle/src/fixtures.rs` - manifest-derived fixture generation and required `/mnt/data` artifact writing contract.
- `crates/cintx-oracle/build.rs` - vendored upstream binding/build gate for oracle harness work.
- `crates/cintx-runtime/src/planner.rs` - tracing spans and execution-policy diagnostics (`query_workspace`/`evaluate`).
- `crates/cintx-runtime/src/workspace.rs` - chunking/fallback reasons and memory-limit contract behavior.
- `crates/cintx-runtime/src/metrics.rs` - run metrics fields used for diagnostic artifacts.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - canonical profile and symbol coverage source of truth.

### Phase 4 Delivery Landing Zones
- `ci/oracle-compare.yml` - oracle CI workflow landing zone.
- `ci/feature-matrix.yml` - feature-profile CI workflow landing zone.
- `ci/gpu-bench.yml` - GPU consistency and benchmark workflow landing zone.
- `xtask/src/main.rs` - xtask command entrypoint for release automation tooling.
- `xtask/src/manifest_audit.rs` - manifest drift gate tooling landing zone.
- `xtask/src/oracle_update.rs` - oracle refresh/report tooling landing zone.
- `xtask/src/bench_report.rs` - benchmark report aggregation landing zone.
- `benches/micro_families.rs` - family microbench suite landing zone.
- `benches/macro_molecules.rs` - macro benchmark suite landing zone.
- `benches/crossover_cpu_gpu.rs` - CPU-GPU crossover benchmark suite landing zone.
- `docs/rust_crate_test_guideline.md` - mandatory test design constraints before adding verification tests.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cintx-oracle/src/compare.rs`: Existing parity harness already checks helper/transform/optimizer surfaces, family tolerances, and artifact output pathways.
- `crates/cintx-oracle/src/fixtures.rs`: Existing manifest-driven fixture matrix builder and `/mnt/data` required-path artifact helper can be generalized from Phase 2 to Phase 4 scope.
- `crates/cintx-runtime/src/planner.rs`, `workspace.rs`, and `metrics.rs`: Existing tracing spans and metrics (`chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`) provide direct inputs for VERI-04 diagnostics.
- `crates/cintx-ops/generated/compiled_manifest.lock.json`: Existing lock artifact can be used directly for manifest drift and coverage gate assertions.

### Established Patterns
- Verification is manifest-driven rather than hardcoded symbol lists.
- Failure behavior is fail-closed (`UnsupportedApi`, typed memory errors, no partial writes).
- Artifacts target required `/mnt/data` paths with explicit fallback metadata when required path write fails.
- Runtime observability uses structured tracing spans and explicit fields, not freeform logging.

### Integration Points
- CI stubs in `ci/*.yml` are the insertion points for merge-blocking and scheduled verification policy.
- xtask stubs in `xtask/src/*.rs` are the insertion points for manifest audit, oracle update, and benchmark report commands.
- Bench stubs in `benches/*.rs` are the insertion points for micro/macro/crossover regression suites.
- Oracle harness and runtime diagnostics modules provide the data producers that CI and xtask automation should consume.

</code_context>

<specifics>
## Specific Ideas

No external product references were requested. Decisions prioritize strict regression prevention, full diff visibility for oracle failures, and deterministic merge-blocking policy.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within Phase 4 scope.

</deferred>

---

*Phase: 04-verification-release-automation*
*Context gathered: 2026-03-28*
