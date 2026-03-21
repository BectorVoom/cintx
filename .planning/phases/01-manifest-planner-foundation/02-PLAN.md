---
phase: 01-manifest-planner-foundation
plan: 02
type: execute
wave: 2
depends_on:
  - 01
files_modified:
  - crates/cintx-core/src/error.rs
  - crates/cintx-runtime/src/options.rs
  - crates/cintx-runtime/src/planner.rs
  - crates/cintx-runtime/src/validator.rs
  - crates/cintx-runtime/src/workspace.rs
autonomous: true
requirements:
  - BASE-01
  - BASE-03
must_haves:
  truths:
    - "`query_workspace()` and `evaluate()` live in `cintx-runtime` and return `WorkspaceQuery`/`ExecutionStats` while honoring `memory_limit_bytes` chunking per docs/design §7.2-7.5, satisfying D-09 and D-10."
    - "Typed errors such as `UnsupportedApi`, `InvalidShellTuple`, `InvalidDims`, and `MemoryLimitExceeded` are defined in `cintx-core` and used throughout validation/planning to keep raw symbol errors explicit, satisfying D-11."
    - "Workspace/Chunk planning emits tracing spans for planner decisions, batch splits, chunk counts, and fallback reasons, satisfying D-12."
  artifacts:
    - path: crates/cintx-core/src/error.rs
      provides: "Typed `thiserror::Error` variants that map validator/planner failures to the public contract."
      min_lines: 40
    - path: crates/cintx-runtime/src/planner.rs
      provides: "`query_workspace` and `evaluate` implementations wired to the resolver and the workspace estimator."
      min_lines: 80
    - path: crates/cintx-runtime/src/workspace.rs
      provides: "`WorkspaceAllocator`, `WorkspaceQuery`, and `ChunkPlanner` that enforce `memory_limit_bytes`."
      min_lines: 80
  key_links:
    - from: crates/cintx-runtime/src/validator.rs
      to: crates/cintx-core/src/error.rs
      via: "Returns `cintxRsError` variants on invalid shell tuples/dims."
      pattern: "cintxRsError"
    - from: crates/cintx-runtime/src/planner.rs
      to: crates/cintx-runtime/src/workspace.rs
      via: "`query_workspace` → `WorkspaceEstimator` → `ChunkPlanner`"
      pattern: "ChunkPlanner"
---

<objective>
Wire the runtime planner/validator/workspace stack to the canonical manifest/resolver so query/evaluate obeys memory limits and surfaces typed errors.
Purpose: Fulfill D-09..D-12 by giving downstream safe/compat callers deterministic chunking, typed failure modes, and tracing before CubeCL kernels land.
Output: `cintxRsError`, `ExecutionOptions`, workspace estimators, `ChunkPlanner`, and planner entry points that reference resolver metadata.
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
@docs/design/cintx_detailed_design.md
@crates/cintx-ops/src/resolver.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Define typed runtime errors and execution options</name>
  <files>crates/cintx-core/src/error.rs, crates/cintx-runtime/src/options.rs</files>
  <read_first>crates/cintx-core/src/error.rs, crates/cintx-runtime/src/options.rs, docs/design/cintx_detailed_design.md §§7.2-7.6, AGENTS.md (error stack guidance)</read_first>
  <action>
In `error.rs`, declare `#[derive(thiserror::Error, Debug)] pub enum cintxRsError` with variants for `UnsupportedApi { requested: String }`, `InvalidShellTuple { expected: usize, got: usize }`, `InvalidDims { expected: usize, provided: usize }`, `MemoryLimitExceeded { requested: usize, limit: usize }`, `HostAllocationFailed`, `DeviceOutOfMemory`, and `ChunkPlanFailed { from: &'static str, detail: String }`. Annotate each variant with `#[error(...)]` messages that mention the raw fields so callers can surface the same semantics as docs/design §7.3-7.5. In `options.rs`, define `ExecutionOptions` (`memory_limit_bytes: Option<usize>`, `trace_span: Option<tracing::Span>`, `chunk_size_override: Option<usize>`, `profile_label: Option<&'static str>`) with helpers for default limits; keep this in `cintx-runtime` but re-export relevant trait bounds so the planner can accept options without pulling `anyhow`.
  </action>
  <acceptance_criteria>
    - `rg -n "pub enum cintxRsError" crates/cintx-core/src/error.rs`
    - `rg -n "MemoryLimitExceeded" crates/cintx-core/src/error.rs`
    - `rg -n "ExecutionOptions" crates/cintx-runtime/src/options.rs`
    - `rg -n "trace_span" crates/cintx-runtime/src/options.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-core --lib</automated>
  </verify>
  <done>Typed errors and execution options exist so planner + validator workloads can signal invalid layouts, unsupported APIs, and memory-limit failures without relying on `anyhow` (per D-11) while exposing deterministic options for chunk control.</done>
</task>

<task type="auto">
  <name>Task 2: Implement validator/planner/workspace contract</name>
  <files>crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/validator.rs, crates/cintx-runtime/src/workspace.rs</files>
  <read_first>crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/validator.rs, crates/cintx-runtime/src/workspace.rs, docs/design/cintx_detailed_design.md §§4.6, 4.9, 7.2-7.6, crates/cintx-ops/src/resolver.rs</read_first>
  <action>
Build the validator/planner/workspace pipeline that D-09 through D-12 describe. In `validator.rs`, validate shell tuples/dims/env/representation and raise `cintxRsError` variants when the layout is invalid, and expose `ValidatedShellTuple`. In `workspace.rs`, define `WorkspaceQuery { bytes: usize, alignment: usize }`, `WorkspaceAllocator` (trait that only uses `FallibleBuffer`), `ChunkInfo`, `ChunkPlanner`, and `WorkspaceRequest` so every workspace allocation passes through a central fallible allocator; `ChunkPlanner` must look at `ExecutionOptions::memory_limit_bytes` and split the required bytes into at least one chunk, trace the chunk count/fallback reason, and emit `MemoryLimitExceeded` when no chunk can fit. In `planner.rs`, add `pub fn query_workspace(op: OperatorId, rep: Representation, basis: &BasisSet, shells: ShellTuple, opts: &ExecutionOptions) -> Result<WorkspaceQuery, cintxRsError>` and `pub fn evaluate(plan: ExecutionPlan<'_>, opts: &ExecutionOptions, allocator: &mut dyn WorkspaceAllocator) -> Result<ExecutionStats, cintxRsError>`; both functions must resolve descriptors via `Resolver::descriptor(op)` and feed the metadata into the workspace estimator/chunk planner before handing work to CubeCL. Emit tracing spans for planner decisions, chunk splits, and fallback reasons before returning. Make `ExecutionPlan` borrow `&'a BasisSet`, the `OperatorDescriptor`, and the `WorkspaceQuery` so `execute` uses deterministic chunk layout.
  </action>
  <acceptance_criteria>
    - `rg -n "query_workspace(" crates/cintx-runtime/src/planner.rs`
    - `rg -n "evaluate(" crates/cintx-runtime/src/planner.rs`
    - `rg -n "ChunkPlanner" crates/cintx-runtime/src/workspace.rs`
    - `rg -n "WorkspaceAllocator" crates/cintx-runtime/src/workspace.rs`
    - `rg -n "cintxRsError" crates/cintx-runtime/src/validator.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-runtime --lib</automated>
  </verify>
  <done>BASE-01/BASE-03 rely on this path: validators emit typed errors, query/evaluate honor memory limits with `ChunkPlanner`, and the planner traces decisions while resolving manifest metadata.</done>
</task>

</tasks>

<verification>
Wave 2 builders can run `cargo test -p cintx-core --lib` and `cargo test -p cintx-runtime --lib` to prove the runtime contract compiles and links to the errors/options.
</verification>

<success_criteria>
`query_workspace`/`evaluate` exist, the workspace layer enforces memory limits, and every failure is surfaced through `cintxRsError` variants.
</success_criteria>

<output>
After completion, create `.planning/phases/01-manifest-planner-foundation/02-PLAN-SUMMARY.md`
</output>
