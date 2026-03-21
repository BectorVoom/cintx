---
phase: 02-execution-compatibility-stabilization
plan: 04
type: execute
wave: 2
depends_on:
  - 02
  - 03
files_modified:
  - crates/cintx-runtime/src/lib.rs
  - crates/cintx-runtime/src/dispatch.rs
  - crates/cintx-runtime/src/planner.rs
  - crates/cintx-runtime/src/metrics.rs
  - crates/cintx-runtime/src/scheduler.rs
  - crates/cintx-cubecl/src/lib.rs
  - crates/cintx-cubecl/src/executor.rs
  - crates/cintx-cubecl/src/resident_cache.rs
  - crates/cintx-cubecl/src/specialization.rs
  - crates/cintx-cubecl/src/transfer.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
must_haves:
  truths:
    - "The shared runtime no longer stops at allocation rehearsal; it can hand validated chunk plans to a backend-neutral execution contract."
    - "A CubeCL executor exists for the Phase 2 base families and is selected through runtime dispatch metadata instead of compat-only symbol switches."
    - "Memory-limit chunking remains authoritative: no backend launch or output write begins before runtime validation, workspace sizing, and allocation all succeed."
  artifacts:
    - path: crates/cintx-runtime/src/dispatch.rs
      provides: "Backend-neutral execution trait plus dispatch metadata for runtime-to-backend handoff."
      min_lines: 80
    - path: crates/cintx-runtime/src/planner.rs
      provides: "Execution-plan handoff that invokes a backend executor instead of just allocating/releasing chunks."
      min_lines: 140
    - path: crates/cintx-cubecl/src/executor.rs
      provides: "CubeCL backend implementation for the runtime execution trait."
      min_lines: 120
    - path: crates/cintx-cubecl/src/specialization.rs
      provides: "Family/representation specialization keys for CubeCL dispatch."
      min_lines: 60
  key_links:
    - from: crates/cintx-runtime/src/planner.rs
      to: crates/cintx-runtime/src/dispatch.rs
      via: "`evaluate()` resolves a `DispatchDecision` and hands chunk execution to `BackendExecutor`."
      pattern: "BackendExecutor"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/specialization.rs
      via: "The CubeCL executor selects specialization keys from manifest family and representation metadata."
      pattern: "SpecializationKey"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/transfer.rs
      via: "Transfers stage metadata/workspace/output without bypassing the fallible allocation contract."
      pattern: "TransferPlan"
---

<objective>
Introduce the backend-neutral runtime execution contract and the CubeCL executor core that Phase 2 raw compat calls will run through.
Purpose: Turn the Phase 1 planner into a real execution path while preserving manifest-driven dispatch, deterministic chunking, and explicit memory-limit failures.
Output: `BackendExecutor` runtime interfaces plus the CubeCL executor/resident-cache/transfer core.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-manifest-planner-foundation/02-PLAN-SUMMARY.md
@.planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md
@crates/cintx-runtime/src/planner.rs
@crates/cintx-runtime/src/workspace.rs
@crates/cintx-ops/src/resolver.rs
<interfaces>
From `crates/cintx-runtime/src/planner.rs`:
```rust
pub struct ExecutionPlan<'a> {
    pub basis: &'a BasisSet,
    pub descriptor: &'a OperatorDescriptor,
    pub representation: Representation,
    pub shells: ValidatedShellTuple,
    pub workspace: &'a WorkspaceQuery,
}

pub fn query_workspace(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    opts: &ExecutionOptions,
) -> Result<WorkspaceQuery, cintxRsError>;
```

From `docs/design/cintx_detailed_design.md` §5.2:
```rust
pub trait BackendExecutor {
    fn supports(&self, plan: &ExecutionPlan) -> bool;
    fn query_workspace(&self, plan: &ExecutionPlan) -> Result<WorkspaceBytes, cintxRsError>;
    fn execute(&self, plan: &ExecutionPlan, io: &mut ExecutionIo<'_>) -> Result<ExecutionStats, cintxRsError>;
}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace the planner's allocation-only loop with a backend-neutral execution contract</name>
  <files>crates/cintx-runtime/src/lib.rs, crates/cintx-runtime/src/dispatch.rs, crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/metrics.rs, crates/cintx-runtime/src/scheduler.rs</files>
  <read_first>crates/cintx-runtime/src/lib.rs, crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/workspace.rs, crates/cintx-runtime/src/dispatch.rs, docs/design/cintx_detailed_design.md §5.2 and §7.1-7.8, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Replace the current `evaluate(plan, opts, allocator)` allocate/release loop with a backend-neutral execution contract. In `dispatch.rs`, define `BackendExecutor` exactly around `supports`, `query_workspace`, and `execute`, plus `ExecutionIo<'a>`, `WorkspaceBytes`, `DispatchDecision`, and a `DispatchFamily` enum that covers exactly `OneElectron`, `TwoElectron`, `Center2c2e`, `Center3c1e`, and `Center3c2e`. In `planner.rs`, extend `ExecutionPlan` with the resolved dispatch family, component count, and output-layout metadata needed by the backend, and change `evaluate()` so it validates the query contract, allocates workspace, constructs `ExecutionIo`, and delegates chunk execution to a `&dyn BackendExecutor` instead of stopping after buffer allocation. In `scheduler.rs`, centralize iteration over `WorkspaceQuery.chunks` so chunk ordering remains deterministic under `memory_limit_bytes`. In `metrics.rs`, add per-run metrics for chunk count, peak workspace bytes, transfer bytes, and `not0` so later oracle/debugging work can inspect execution behavior. Update `lib.rs` exports accordingly.
  </action>
  <acceptance_criteria>
    - `rg -n "trait BackendExecutor" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "enum DispatchFamily" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "struct ExecutionIo" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "executor: &dyn BackendExecutor|&dyn BackendExecutor" crates/cintx-runtime/src/planner.rs`
    - `rg -n "peak_workspace_bytes|transfer_bytes|not0" crates/cintx-runtime/src/metrics.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-runtime --lib</automated>
  </verify>
  <done>The runtime can now delegate validated chunk execution to a backend trait while preserving deterministic scheduling and explicit execution metrics, satisfying the shared-planner half of EXEC-02 and EXEC-03.</done>
</task>

<task type="auto">
  <name>Task 2: Implement the CubeCL executor, specialization keys, resident cache, and transfer planner</name>
  <files>crates/cintx-cubecl/src/lib.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/resident_cache.rs, crates/cintx-cubecl/src/specialization.rs, crates/cintx-cubecl/src/transfer.rs</files>
  <read_first>crates/cintx-cubecl/src/lib.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/resident_cache.rs, crates/cintx-cubecl/src/specialization.rs, crates/cintx-cubecl/src/transfer.rs, crates/cintx-runtime/src/dispatch.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, AGENTS.md</read_first>
  <action>
Implement the CubeCL backend core in the `cubecl 0.9.0` line only. In `executor.rs`, add `CubeClExecutor` that implements `BackendExecutor` and refuses unsupported families through `cintxRsError::UnsupportedApi`. In `specialization.rs`, define a `SpecializationKey` that includes canonical family, representation, component rank, and shell angular-momentum tuple so family kernels can specialize without leaking raw symbol names. In `resident_cache.rs`, add a basis/device-scoped resident metadata cache keyed by basis hash plus representation. In `transfer.rs`, add a `TransferPlan` that stages shell metadata, workspace buffers, and output slices while keeping host CPU work limited to validation, marshaling, and copy orchestration. Keep `center_4c1e` unsupported in this phase even if a stub file exists, and ensure allocation/transfer failures map to `HostAllocationFailed` or `DeviceOutOfMemory` rather than ad hoc errors. Update `lib.rs` exports and add focused unit tests around specialization keys and executor family support.
  </action>
  <acceptance_criteria>
    - `rg -n "struct CubeClExecutor" crates/cintx-cubecl/src/executor.rs`
    - `rg -n "impl BackendExecutor for CubeClExecutor" crates/cintx-cubecl/src/executor.rs`
    - `rg -n "struct SpecializationKey" crates/cintx-cubecl/src/specialization.rs`
    - `rg -n "struct TransferPlan" crates/cintx-cubecl/src/transfer.rs`
    - `rg -n "DeviceResidentCache|ResidentCache" crates/cintx-cubecl/src/resident_cache.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>The CubeCL backend core exists, plugs into runtime through `BackendExecutor`, and stages the data needed for the base-family kernels without broadening support beyond the locked Phase 2 scope.</done>
</task>

</tasks>

<verification>
Run the runtime and CubeCL library test suites together after wiring the executor trait to prove the planner and backend agree on chunk scheduling, metrics, and supported-family selection.
</verification>

<success_criteria>
`cintx-runtime` delegates execution through a backend trait, `cintx-cubecl` implements that trait for the base Phase 2 families, and chunked execution still fails cleanly under memory-pressure conditions.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/04-PLAN-SUMMARY.md`
</output>
