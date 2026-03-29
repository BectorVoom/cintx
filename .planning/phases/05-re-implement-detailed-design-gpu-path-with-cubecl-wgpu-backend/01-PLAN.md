---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-runtime/src/options.rs
  - crates/cintx-runtime/src/workspace.rs
  - crates/cintx-runtime/src/planner.rs
  - crates/cintx-runtime/src/lib.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
  - COMP-05
must_haves:
  truths:
    - "Query captures backend intent metadata as part of the execution contract."
    - "Evaluate fails closed when backend intent/capability differs from query."
    - "Runtime chunk planning contract remains deterministic while backend policy drift becomes explicit."
  artifacts:
    - path: crates/cintx-runtime/src/options.rs
      provides: "Typed backend intent + capability token fields on ExecutionOptions."
      min_lines: 70
    - path: crates/cintx-runtime/src/workspace.rs
      provides: "WorkspaceQuery backend contract storage and planning_matches checks."
      min_lines: 220
    - path: crates/cintx-runtime/src/planner.rs
      provides: "Query/evaluate enforcement for backend contract drift."
      min_lines: 420
  key_links:
    - from: crates/cintx-runtime/src/options.rs
      to: crates/cintx-runtime/src/workspace.rs
      via: "WorkspaceQuery carries and validates options backend contract fields."
      pattern: "backend_intent|backend_capability_token|planning_matches"
    - from: crates/cintx-runtime/src/workspace.rs
      to: crates/cintx-runtime/src/planner.rs
      via: "evaluate rejects mismatched query/evaluate backend contract."
      pattern: "planning_matches\\(opts\\)|backend.*drift|ChunkPlanFailed"
---

<objective>
Define a typed backend-intent contract in runtime query/evaluate so Phase 5 can enforce wgpu fail-closed behavior.
Purpose: Implement D-03 and D-08 before backend execution rewiring so later plans can consume a stable contract.
Output: Runtime metadata types, planner drift enforcement, and tests that fail on backend-policy mismatch.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/STATE.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md
@docs/design/cintx_detailed_design.md
@AGENTS.md
@crates/cintx-runtime/src/options.rs
@crates/cintx-runtime/src/workspace.rs
@crates/cintx-runtime/src/planner.rs
@crates/cintx-runtime/src/lib.rs
<interfaces>
From `crates/cintx-runtime/src/options.rs`:
```rust
#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub memory_limit_bytes: Option<usize>,
    pub trace_span: Option<Span>,
    pub chunk_size_override: Option<usize>,
    pub profile_label: Option<&'static str>,
}
```

From `crates/cintx-runtime/src/workspace.rs`:
```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceQuery {
    pub bytes: usize,
    pub chunk_count: usize,
    pub memory_limit_bytes: Option<usize>,
    pub chunk_size_override: Option<usize>,
}
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add runtime backend intent/capability contract types</name>
  <files>crates/cintx-runtime/src/options.rs, crates/cintx-runtime/src/workspace.rs, crates/cintx-runtime/src/lib.rs</files>
  <read_first>crates/cintx-runtime/src/options.rs, crates/cintx-runtime/src/workspace.rs, crates/cintx-runtime/src/lib.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md, docs/design/cintx_detailed_design.md</read_first>
  <behavior>
    - Test 1: `ExecutionOptions::default()` yields `backend_intent.backend == Wgpu` and `backend_intent.selector == "auto"`.
    - Test 2: `WorkspaceQuery::planning_matches()` returns `false` when backend intent or capability token changes even if memory/chunk settings stay equal.
  </behavior>
  <action>
Define concrete runtime metadata in `options.rs`: `BackendKind` (with `Wgpu`), `BackendIntent { backend: BackendKind, selector: String }`, and `BackendCapabilityToken { adapter_name: String, backend_api: String, capability_fingerprint: u64 }`. Add `backend_intent` and `backend_capability_token` fields to `ExecutionOptions` with default values `Wgpu` and selector `"auto"` per D-03. Extend `WorkspaceQuery` in `workspace.rs` with matching fields and update `planning_matches()` to compare all four contract fields (`memory_limit_bytes`, `chunk_size_override`, `backend_intent`, `backend_capability_token`). Re-export new types in `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "enum BackendKind|struct BackendIntent|struct BackendCapabilityToken|backend_intent|backend_capability_token" crates/cintx-runtime/src/options.rs crates/cintx-runtime/src/workspace.rs crates/cintx-runtime/src/lib.rs`
    - `rg -n "selector:\\s*\"auto\"|BackendKind::Wgpu|planning_matches" crates/cintx-runtime/src/options.rs crates/cintx-runtime/src/workspace.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-runtime workspace::tests::planning_matches_checks_backend_contract -- --exact</automated>
  </verify>
  <done>Runtime exposes explicit backend-intent contract fields and workspace matching fails on backend metadata drift.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Enforce backend contract drift checks in planner query/evaluate</name>
  <files>crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/workspace.rs</files>
  <read_first>crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/workspace.rs, crates/cintx-runtime/src/options.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md</read_first>
  <behavior>
    - Test 1: `evaluate()` returns `ChunkPlanFailed { from: "evaluate" }` when `backend_intent` differs from query result.
    - Test 2: `evaluate()` returns `ChunkPlanFailed { from: "evaluate" }` when `backend_capability_token` differs from query result.
    - Test 3: `query_workspace()` persists backend contract fields into `WorkspaceQuery`.
  </behavior>
  <action>
Update `query_workspace()` in `planner.rs` to copy `opts.backend_intent` and `opts.backend_capability_token` into `WorkspaceQuery` and emit these values in query tracing fields per D-08. Keep existing memory/chunk contract behavior unchanged. Update `evaluate()` mismatch error detail to explicitly include backend-contract drift language. Add targeted unit tests in `planner.rs` and `workspace.rs` covering intent drift, capability-token drift, and persisted query metadata.
  </action>
  <acceptance_criteria>
    - `rg -n "backend_intent|backend_capability_token|profile|query_workspace" crates/cintx-runtime/src/planner.rs`
    - `rg -n "backend.*drift|ChunkPlanFailed" crates/cintx-runtime/src/planner.rs`
    - `rg -n "evaluate_rejects_query_workspace_backend_intent_drift|evaluate_rejects_query_workspace_backend_capability_token_drift|query_workspace_records_backend_contract_metadata" crates/cintx-runtime/src/planner.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-runtime planner::tests::query_workspace_records_backend_contract_metadata -- --exact && cargo test -p cintx-runtime planner::tests::evaluate_rejects_query_workspace_backend_intent_drift -- --exact && cargo test -p cintx-runtime planner::tests::evaluate_rejects_query_workspace_backend_capability_token_drift -- --exact</automated>
  </verify>
  <done>Runtime query/evaluate contract now fails closed on backend metadata drift and has explicit regression coverage.</done>
</task>

</tasks>

<verification>
Run new runtime unit tests proving backend intent/capability metadata is carried from query to evaluate and rejected on drift.
</verification>

<success_criteria>
`ExecutionOptions` and `WorkspaceQuery` encode backend contract fields, planner stores them during query, and evaluate rejects mismatches with typed errors before backend execution.
</success_criteria>

<output>
After completion, create `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/01-PLAN-SUMMARY.md`
</output>
