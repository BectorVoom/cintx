---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 03
type: execute
wave: 3
depends_on:
  - 02
files_modified:
  - crates/cintx-cubecl/src/executor.rs
  - crates/cintx-cubecl/src/transfer.rs
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
  - COMP-05
must_haves:
  truths:
    - "Each scheduled chunk executes through CubeCL wgpu path; synthetic staging fill is removed."
    - "Unsupported family/representation/capability cases return explicit reason taxonomy."
    - "Backend staging-only ownership and compat final-write ownership remain unchanged."
  artifacts:
    - path: crates/cintx-cubecl/src/executor.rs
      provides: "WGPU-backed executor path with fail-closed capability preflight and no CPU substitute compute."
      min_lines: 420
    - path: crates/cintx-cubecl/src/transfer.rs
      provides: "Chunk-level transfer planning with adapter-aware OOM/error mapping."
      min_lines: 180
    - path: crates/cintx-cubecl/src/kernels/mod.rs
      provides: "Reason-taxonomy-aware family resolution and launch routing."
      min_lines: 200
  key_links:
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/runtime_bootstrap.rs
      via: "Executor uses bootstrap capability report before query/execute."
      pattern: "bootstrap_wgpu_runtime|wgpu-capability|BackendCapabilityToken"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-runtime/src/dispatch.rs
      via: "Ownership contract checks remain BackendStagingOnly -> CompatFinalWrite."
      pattern: "ensure_output_contract|BackendStagingOnly|CompatFinalWrite"
---

<objective>
Replace pseudo backend execution paths with real CubeCL-driven chunk execution and explicit unsupported taxonomy.
Purpose: Implement D-05 through D-12 without regressing runtime ownership and memory contracts.
Output: Updated CubeCL executor/transfer/kernel flow and regression tests proving no synthetic fallback path.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md
@.planning/phases/02-execution-compatibility-stabilization/05-PLAN-SUMMARY.md
@.planning/phases/02-execution-compatibility-stabilization/08-PLAN-SUMMARY.md
@AGENTS.md
@docs/design/cintx_detailed_design.md
@docs/manual/Cubecl/Cubecl_vector.md
@docs/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md
@crates/cintx-cubecl/src/executor.rs
@crates/cintx-cubecl/src/transfer.rs
@crates/cintx-cubecl/src/kernels/mod.rs
@crates/cintx-cubecl/src/runtime_bootstrap.rs
@crates/cintx-cubecl/src/capability.rs
<interfaces>
From `crates/cintx-runtime/src/dispatch.rs`:
```rust
pub trait BackendExecutor {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool;
    fn query_workspace(&self, plan: &ExecutionPlan<'_>) -> Result<WorkspaceBytes, cintxRsError>;
    fn execute(&self, plan: &ExecutionPlan<'_>, io: &mut ExecutionIo<'_>) -> Result<ExecutionStats, cintxRsError>;
}
```

From `crates/cintx-cubecl/src/capability.rs` (Plan 02 output):
```rust
pub enum CapabilityReason { MissingAdapter, MissingFeature(&'static str), LimitTooLow { .. }, FamilyUnsupported(&'static str), RepresentationUnsupported(&'static str) }
pub struct WgpuPreflightReport { pub snapshot: WgpuCapabilitySnapshot, pub capability_token: BackendCapabilityToken, ... }
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Rewire executor and kernel launch flow to real CubeCL chunk execution</name>
  <files>crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/transfer.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-cubecl/src/kernels/two_electron.rs, crates/cintx-cubecl/src/kernels/center_2c2e.rs, crates/cintx-cubecl/src/kernels/center_3c1e.rs, crates/cintx-cubecl/src/kernels/center_3c2e.rs</files>
  <read_first>crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/transfer.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/runtime_bootstrap.rs, crates/cintx-runtime/src/planner.rs, docs/manual/Cubecl/Cubecl_vector.md, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md</read_first>
  <behavior>
    - Test 1: Executor no longer contains `fill_cartesian_staging` or equivalent synthetic staging write helper (D-05).
    - Test 2: `execute()` records per-chunk transfer/not0 metrics after real kernel launch path (D-07).
    - Test 3: Ownership contract checks still fail when backend attempts final-write ownership takeover (D-06).
  </behavior>
  <action>
In `executor.rs`, remove `CUBECL_RUNTIME_PROFILE` constant usage and synthetic staging helpers (`fill_cartesian_staging`). At query/execute entry, call `bootstrap_wgpu_runtime(&plan.workspace.backend_intent)` and map capability failures to typed `UnsupportedApi` text prefixed with `wgpu-capability:` (D-01, D-02). Update kernel launch signatures to accept mutable staging output slices so each chunk writes through CubeCL launch/readback path instead of locally fabricated values (D-05, D-07). Keep explicit `BackendStagingOnly -> CompatFinalWrite` checks before and after backend execution (D-06). In `transfer.rs`, include adapter identifier from preflight report when mapping `DeviceOutOfMemory` and transfer diagnostics.
  </action>
  <acceptance_criteria>
    - `rg -n "bootstrap_wgpu_runtime|wgpu-capability|BackendStagingOnly|CompatFinalWrite" crates/cintx-cubecl/src/executor.rs`
    - `rg -n "fill_cartesian_staging|CUBECL_RUNTIME_PROFILE" crates/cintx-cubecl/src/executor.rs` returns no matches
    - `rg -n "staging_output\\(|record_transfer_bytes|record_not0|chunk\\(\\)" crates/cintx-cubecl/src/executor.rs crates/cintx-cubecl/src/kernels/*.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl executor::tests::execute_uses_wgpu_bootstrap_and_preserves_output_contract -- --exact && cargo test -p cintx-cubecl executor::tests::executor_no_longer_uses_monotonic_stub_sequence -- --exact</automated>
  </verify>
  <done>CubeCL chunk execution no longer relies on synthetic CPU-side staging fill and keeps ownership contracts intact.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement explicit unsupported reason taxonomy and Validated4C1E capability gates</name>
  <files>crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/kernels/mod.rs</files>
  <read_first>crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/capability.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md, docs/design/cintx_detailed_design.md</read_first>
  <behavior>
    - Test 1: 4c1e out-of-envelope failures reference `outside Validated4C1E` with capability-specific suffix (D-11).
    - Test 2: Unsupported family failures include `unsupported_family:<family>` (D-12).
    - Test 3: Unsupported representation failures include `unsupported_representation:<rep>` (D-12).
  </behavior>
  <action>
Replace generic unsupported strings in executor/kernel resolution with explicit reason taxonomy: `unsupported_family:<canonical_family>`, `unsupported_representation:<representation>`, and `outside Validated4C1E (<reason>)`. Update `ensure_validated_4c1e` to require wgpu capability preflight success instead of cpu-profile checks (D-11). Ensure out-of-envelope and capability-missing cases fail before kernel dispatch (D-09). Add unit tests in executor/kernels modules asserting exact reason prefixes and that unsupported paths never trigger hidden fallback.
  </action>
  <acceptance_criteria>
    - `rg -n "unsupported_family:|unsupported_representation:|outside Validated4C1E" crates/cintx-cubecl/src/executor.rs crates/cintx-cubecl/src/kernels/mod.rs`
    - `rg -n "cpu\\\"|CubeCL backend must be cpu" crates/cintx-cubecl/src/executor.rs` returns no matches
    - `rg -n "missing_wgpu_capability|wgpu-capability" crates/cintx-cubecl/src/executor.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl executor::tests::outside_validated_4c1e_requires_wgpu_capability_reason -- --exact && cargo test -p cintx-cubecl kernels::tests::unsupported_family_reports_taxonomy_reason -- --exact && cargo test -p cintx-cubecl kernels::tests::unsupported_representation_reports_taxonomy_reason -- --exact</automated>
  </verify>
  <done>Unsupported and envelope-gated paths now fail with specific taxonomy reasons and no fallback masking behavior.</done>
</task>

</tasks>

<verification>
Run CubeCL executor and kernel regression tests verifying wgpu preflight, no synthetic staging fallback, and explicit unsupported taxonomy.
</verification>

<success_criteria>
CubeCL backend execution path is real and fail-closed: chunk execution goes through CubeCL launch path, unsupported requests are explicit, and ownership contracts remain enforced.
</success_criteria>

<output>
After completion, create `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/03-PLAN-SUMMARY.md`
</output>
