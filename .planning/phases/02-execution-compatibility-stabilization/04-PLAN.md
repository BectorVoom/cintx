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
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
must_haves:
  truths:
    - "The shared runtime no longer stops at allocation rehearsal; it can hand validated chunk plans to a backend-neutral execution contract."
    - "Execution dispatch metadata, chunk ordering, and run metrics stay runtime-owned so compat and later safe APIs share one planner path."
    - "The runtime contract explicitly declares backend output ownership as staging-only (`BackendStagingOnly`) and reserves final caller-visible flat writes for `cintx-compat::layout` (`CompatFinalWrite`) before CubeCL family work starts."
    - "Memory-limit chunking remains authoritative: no backend launch or output write begins before runtime validation, workspace sizing, and allocation all succeed."
  artifacts:
    - path: crates/cintx-runtime/src/dispatch.rs
      provides: "Backend-neutral execution trait plus dispatch metadata for runtime-to-backend handoff."
      min_lines: 80
    - path: crates/cintx-runtime/src/planner.rs
      provides: "Execution-plan handoff that invokes a backend executor instead of just allocating/releasing chunks."
      min_lines: 140
    - path: crates/cintx-runtime/src/scheduler.rs
      provides: "Deterministic chunk iteration helpers that keep memory-limit behavior runtime-owned."
      min_lines: 40
    - path: crates/cintx-runtime/src/metrics.rs
      provides: "Per-run transfer/workspace/not0 metrics collected independently of any specific backend."
      min_lines: 40
  key_links:
    - from: crates/cintx-runtime/src/planner.rs
      to: crates/cintx-runtime/src/dispatch.rs
      via: "`evaluate()` resolves a `DispatchDecision` and hands chunk execution to `BackendExecutor`."
      pattern: "BackendExecutor"
    - from: crates/cintx-runtime/src/planner.rs
      to: crates/cintx-runtime/src/scheduler.rs
      via: "Planner-driven evaluation walks the runtime-owned chunk schedule instead of embedding ordering in compat or backend code."
      pattern: "schedule|chunks"
    - from: crates/cintx-runtime/src/planner.rs
      to: crates/cintx-runtime/src/metrics.rs
      via: "Runtime evaluation reports chunk count, workspace bytes, transfer bytes, and `not0` through shared metrics."
      pattern: "transfer_bytes|not0|chunk_count"
    - from: crates/cintx-runtime/src/dispatch.rs
      to: crates/cintx-compat/src/layout.rs
      via: "Dispatch/output contract states backend returns staging buffers only and compat owns final flat writes, so later plans cannot split caller-visible write ownership."
      pattern: "OutputOwnership|BackendStagingOnly|CompatFinalWrite"
---

<objective>
Introduce the backend-neutral runtime execution contract that every Phase 2 caller and backend will share.
Purpose: Turn the Phase 1 planner into a real execution path while preserving manifest-driven dispatch, deterministic chunking, and explicit memory-limit failures before any CubeCL-specific work lands.
Output: `BackendExecutor` runtime interfaces plus runtime-owned scheduling and execution metrics.
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
  <name>Task 1: Define the backend-neutral execution contract and dispatch metadata</name>
  <files>crates/cintx-runtime/src/lib.rs, crates/cintx-runtime/src/dispatch.rs</files>
  <read_first>crates/cintx-runtime/src/lib.rs, crates/cintx-runtime/src/dispatch.rs, crates/cintx-runtime/src/planner.rs, docs/design/cintx_detailed_design.md §5.2 and §7.1-7.8, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
In `dispatch.rs`, define `BackendExecutor` exactly around `supports`, `query_workspace`, and `execute`, plus `ExecutionIo<'a>`, `WorkspaceBytes`, `DispatchDecision`, and a `DispatchFamily` enum that covers exactly `OneElectron`, `TwoElectron`, `Center2c2e`, `Center3c1e`, and `Center3c2e`. Also define an explicit output-ownership contract in runtime with concrete values `OutputOwnership::BackendStagingOnly` and `OutputOwnership::CompatFinalWrite`, and wire it into `DispatchDecision` and `ExecutionIo` so backend execution can only fill staging buffers/metadata and cannot own caller-visible flat writes. Keep the trait backend-neutral: no CubeCL types, client handles, or device-specific caches in `cintx-runtime`. Update `lib.rs` exports so later plans can import the execution contract directly from `cintx_runtime`.
  </action>
  <acceptance_criteria>
    - `rg -n "trait BackendExecutor" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "enum DispatchFamily" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "struct ExecutionIo" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "enum OutputOwnership|BackendStagingOnly|CompatFinalWrite" crates/cintx-runtime/src/dispatch.rs`
    - `rg -n "pub use .*BackendExecutor|pub use .*DispatchDecision" crates/cintx-runtime/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-runtime --lib</automated>
  </verify>
  <done>The runtime exposes a backend-neutral execution contract and dispatch metadata that later CubeCL and compat plans can consume without adding backend-specific types to the shared planner.</done>
</task>

<task type="auto">
  <name>Task 2: Rework planner evaluation around deterministic scheduling and runtime-owned metrics</name>
  <files>crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/metrics.rs, crates/cintx-runtime/src/scheduler.rs</files>
  <read_first>crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/metrics.rs, crates/cintx-runtime/src/scheduler.rs, crates/cintx-runtime/src/workspace.rs, crates/cintx-runtime/src/dispatch.rs, docs/design/cintx_detailed_design.md §5.2 and §7.1-7.8, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
In `planner.rs`, extend `ExecutionPlan` with the resolved dispatch family, component count, and output-layout metadata that any backend will need, then change `evaluate()` so it validates the query contract, allocates workspace, constructs `ExecutionIo`, and delegates chunk execution to a `&dyn BackendExecutor` instead of stopping after buffer allocation. Enforce `OutputOwnership::CompatFinalWrite` in planner/dispatch handoff and reject any backend path that attempts to bypass staging-only semantics. In `scheduler.rs`, centralize iteration over `WorkspaceQuery.chunks` so chunk ordering remains deterministic under `memory_limit_bytes`. In `metrics.rs`, add per-run metrics for chunk count, peak workspace bytes, transfer bytes, and `not0` so later compat/oracle work can inspect execution behavior without embedding metric logic in a backend crate.
  </action>
  <acceptance_criteria>
    - `rg -n "executor: &dyn BackendExecutor|&dyn BackendExecutor" crates/cintx-runtime/src/planner.rs`
    - `rg -n "OutputOwnership|CompatFinalWrite|staging" crates/cintx-runtime/src/planner.rs`
    - `rg -n "peak_workspace_bytes|transfer_bytes|not0" crates/cintx-runtime/src/metrics.rs`
    - `rg -n "schedule|chunks" crates/cintx-runtime/src/scheduler.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-runtime --lib</automated>
  </verify>
  <done>The runtime can now delegate validated chunk execution to a backend trait while preserving deterministic scheduling and explicit execution metrics, satisfying the shared-planner half of EXEC-02 and EXEC-03.</done>
</task>

</tasks>

<verification>
Run the runtime library test suite after wiring the executor trait to prove the planner, scheduler, and metrics all compile around the new backend-neutral contract before CubeCL work starts.
</verification>

<success_criteria>
`cintx-runtime` delegates execution through a backend trait, owns deterministic chunk scheduling and run metrics, and keeps memory-pressure failures explicit without taking on backend-specific code.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/04-PLAN-SUMMARY.md`
</output>
