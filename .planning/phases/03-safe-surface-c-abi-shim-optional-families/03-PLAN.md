---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 03
type: execute
wave: 3
depends_on:
  - 01
  - 02
files_modified:
  - crates/cintx-rs/src/lib.rs
  - crates/cintx-rs/src/api.rs
  - crates/cintx-rs/src/builder.rs
  - crates/cintx-rs/src/error.rs
  - crates/cintx-rs/src/prelude.rs
autonomous: true
requirements:
  - EXEC-01
must_haves:
  truths:
    - "Safe Rust callers can perform workspace planning through `query_workspace()` before evaluation and inspect structured planning metadata."
    - "Safe evaluation consumes a typed session/request contract and returns owned typed outputs without caller-managed raw output buffers."
    - "Facade-level errors are exposed as a stable typed enum that preserves unsupported/layout/memory/validation categories."
    - "Query/evaluate contract drift is rejected explicitly and still preserves fail-closed no-partial-write behavior from runtime/compat."
  artifacts:
    - path: crates/cintx-rs/src/error.rs
      provides: "Stable facade error enum and mapping from `cintxRsError` categories."
      min_lines: 80
    - path: crates/cintx-rs/src/api.rs
      provides: "Typed session/query/evaluate facade with structured workspace plan metadata and owned output return type."
      min_lines: 220
    - path: crates/cintx-rs/src/builder.rs
      provides: "Typed request/session builder APIs used to create safe execution sessions."
      min_lines: 120
    - path: crates/cintx-rs/src/prelude.rs
      provides: "Curated stable safe-API re-exports for users."
      min_lines: 30
  key_links:
    - from: crates/cintx-rs/src/api.rs
      to: crates/cintx-runtime/src/planner.rs
      via: "Safe query/evaluate methods call runtime `query_workspace` and `evaluate` rather than duplicating planner logic."
      pattern: "query_workspace|ExecutionPlan::new|evaluate"
    - from: crates/cintx-rs/src/error.rs
      to: crates/cintx-core/src/error.rs
      via: "Facade error mapping preserves core categories into stable safe-surface variants."
      pattern: "UnsupportedApi|InvalidDims|MemoryLimitExceeded|BufferTooSmall"
    - from: crates/cintx-rs/src/api.rs
      to: crates/cintx-compat/src/raw.rs
      via: "Optional-family and unstable-source `UnsupportedApi` decisions propagate through facade error mapping."
      pattern: "UnsupportedApi"
---

<objective>
Implement the safe Rust facade with typed session contracts, structured workspace planning metadata, and owned-output evaluation.
Purpose: Deliver the Phase 3 safe-surface UX while preserving runtime contract guarantees and typed failure modes.
Output: Production-safe `cintx-rs` API (`query_workspace` + `evaluate`) and facade error/output types with focused tests.
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
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md §5.4, §6.4, §11.1
@crates/cintx-runtime/src/planner.rs
@crates/cintx-runtime/src/options.rs
@crates/cintx-runtime/src/lib.rs
@crates/cintx-core/src/error.rs
@crates/cintx-rs/src/lib.rs
@crates/cintx-rs/src/api.rs
@crates/cintx-rs/src/builder.rs
@crates/cintx-rs/src/prelude.rs
<interfaces>
From `crates/cintx-runtime/src/planner.rs`:
```rust
pub fn query_workspace(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    opts: &ExecutionOptions,
) -> Result<WorkspaceQuery, cintxRsError>;

pub fn evaluate(
    plan: ExecutionPlan<'_>,
    opts: &ExecutionOptions,
    allocator: &mut dyn WorkspaceAllocator,
    executor: &dyn BackendExecutor,
) -> Result<ExecutionStats, cintxRsError>;
```

From `crates/cintx-runtime/src/workspace.rs`:
```rust
pub struct WorkspaceQuery {
    pub bytes: usize,
    pub chunk_count: usize,
    pub chunks: Vec<ChunkInfo>,
    pub memory_limit_bytes: Option<usize>,
    pub chunk_size_override: Option<usize>,
}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement typed safe-session contracts and stable facade error/output types</name>
  <files>crates/cintx-rs/src/lib.rs, crates/cintx-rs/src/api.rs, crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/error.rs, crates/cintx-rs/src/prelude.rs</files>
  <read_first>crates/cintx-rs/src/lib.rs, crates/cintx-rs/src/api.rs, crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/prelude.rs, crates/cintx-core/src/error.rs, crates/cintx-runtime/src/workspace.rs, docs/design/cintx_detailed_design.md §5.4, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md</read_first>
  <action>
Replace the facade stubs with concrete typed API contracts. Introduce a typed session/request object that binds operator, representation, basis, shells, and execution options into one validated safe context (per D-01). Define a structured workspace-plan metadata type carrying bytes/chunks plus an execution token contract, not just scalar bytes (per D-03). Add stable facade error types in `error.rs` that map `cintxRsError` into explicit categories (`UnsupportedApi`, layout/dims, memory, validation) without exposing backend internals (per D-04). Define an owned safe output type (tensor/layout metadata + owned data buffer) and keep it on the stable side of the facade contract for use in `evaluate` (per D-02). Preserve unstable namespace boundaries from Plan 01 so stable exports remain unchanged when `unstable-source-api` is disabled (per D-13 and D-16).
  </action>
  <acceptance_criteria>
    - `rg -n "struct .*Session|struct .*Request|struct .*Workspace" crates/cintx-rs/src/api.rs crates/cintx-rs/src/builder.rs`
    - `rg -n "enum .*Error|UnsupportedApi|Memory|InvalidDims|Validation" crates/cintx-rs/src/error.rs`
    - `rg -n "owned|Vec<f64>|Integral|Tensor" crates/cintx-rs/src/api.rs`
    - `rg -n "pub use .*Session|pub use .*Error" crates/cintx-rs/src/prelude.rs crates/cintx-rs/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-rs --lib</automated>
  </verify>
  <done>The safe crate exposes typed session/query/output/error contracts that are stable, explicit, and ready for query/evaluate execution wiring.</done>
</task>

<task type="auto">
  <name>Task 2: Wire safe `query_workspace()` and `evaluate()` to runtime with fail-closed owned-output behavior</name>
  <files>crates/cintx-rs/src/api.rs, crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/error.rs, crates/cintx-rs/src/prelude.rs</files>
  <read_first>crates/cintx-rs/src/api.rs, crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/error.rs, crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/workspace.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-compat/src/raw.rs, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement runtime wiring so safe API methods call existing runtime contracts rather than duplicating planning logic. `query_workspace()` must call runtime `query_workspace` and return the structured safe workspace-plan metadata object bound to the session (per D-03 and D-17). `evaluate()` must consume/validate that query contract, build `ExecutionPlan`, run backend evaluation through `CubeClExecutor`, and return owned output plus stats (per D-01 and D-02). Reject query/evaluate drift explicitly and preserve fail-closed ownership behavior from Phase 2 (`BackendStagingOnly -> CompatFinalWrite` semantics and no partial writes) when mapping to facade results (per D-18). Ensure unsupported optional/unstable requests propagate as stable facade `UnsupportedApi` errors without silently falling back.
  </action>
  <acceptance_criteria>
    - `rg -n "query_workspace\\(|ExecutionPlan::new|evaluate\\(" crates/cintx-rs/src/api.rs`
    - `rg -n "CubeClExecutor|HostWorkspaceAllocator" crates/cintx-rs/src/api.rs`
    - `rg -n "contract drift|planning_matches|UnsupportedApi" crates/cintx-rs/src/api.rs crates/cintx-rs/src/error.rs`
    - `rg -n "owned|bytes_written|workspace_bytes|chunk_count" crates/cintx-rs/src/api.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-rs --lib</automated>
  </verify>
  <done>Safe callers can query workspace and then evaluate through one typed session path with owned outputs and explicit typed failures, meeting EXEC-01.</done>
</task>

</tasks>

<verification>
Run `cargo test -p cintx-rs --lib` to confirm facade contracts, runtime wiring, and safe error mappings across success/failure paths.
</verification>

<success_criteria>
`cintx-rs` exposes a usable safe API where `query_workspace()` and `evaluate()` are separate but connected through typed contracts, with owned outputs and stable typed errors.
</success_criteria>

<output>
After completion, create `.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-PLAN-SUMMARY.md`
</output>
