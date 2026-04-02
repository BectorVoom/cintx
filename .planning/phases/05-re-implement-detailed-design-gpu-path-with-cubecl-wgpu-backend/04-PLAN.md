---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 04
type: execute
wave: 4
depends_on:
  - 03
files_modified:
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-rs/src/api.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
  - COMP-05
  - VERI-04
must_haves:
  truths:
    - "Compat and safe facade both consume the same CubeCL executor path."
    - "Query/evaluate tokens include backend intent/capability drift checks in safe facade."
    - "Regression tests fail if pseudo/synthetic output behavior silently returns."
  artifacts:
    - path: crates/cintx-compat/src/raw.rs
      provides: "Raw policy gate alignment with wgpu capability and unsupported taxonomy."
      min_lines: 1400
    - path: crates/cintx-rs/src/api.rs
      provides: "Safe facade runtime bridge using shared CubeCL executor without local synthetic stub."
      min_lines: 760
  key_links:
    - from: crates/cintx-rs/src/api.rs
      to: crates/cintx-cubecl/src/executor.rs
      via: "Safe evaluate uses shared CubeCL executor instead of local stub executor."
      pattern: "use cintx_cubecl::CubeClExecutor|RecordingExecutor"
    - from: crates/cintx-compat/src/raw.rs
      to: crates/cintx-runtime/src/planner.rs
      via: "Raw execution options carry backend intent and rely on runtime query/evaluate drift checks."
      pattern: "ExecutionOptions|backend_intent|query_workspace|evaluate"
---

<objective>
Align compat and safe facade execution with the new wgpu-backed CubeCL path and add anti-pseudo regression coverage.
Purpose: Implement D-08, D-10, D-13, D-15, and D-16 across raw and safe public surfaces.
Output: Shared executor usage in safe facade, updated raw policy gates, and layered regression tests.
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
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-PLAN-SUMMARY.md
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/06-PLAN-SUMMARY.md
@AGENTS.md
@docs/rust_crate_test_guideline.md
@crates/cintx-compat/src/raw.rs
@crates/cintx-rs/src/api.rs
@crates/cintx-cubecl/src/executor.rs
@crates/cintx-runtime/src/options.rs
<interfaces>
From `crates/cintx-rs/src/api.rs`:
```rust
pub struct WorkspaceExecutionToken {
    operator: OperatorId,
    representation: Representation,
    shell_count: usize,
    required_workspace_bytes: usize,
    memory_limit_bytes: Option<usize>,
    chunk_size_override: Option<usize>,
}
```

From `crates/cintx-compat/src/raw.rs`:
```rust
fn execution_options_from_opt(opt: Option<&RawOptimizerHandle>) -> ExecutionOptions
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Remove safe local stub executor and propagate backend contract metadata</name>
  <files>crates/cintx-rs/src/api.rs, crates/cintx-compat/src/raw.rs</files>
  <read_first>crates/cintx-rs/src/api.rs, crates/cintx-compat/src/raw.rs, crates/cintx-runtime/src/options.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md</read_first>
  <behavior>
    - Test 1: Safe facade imports and uses `cintx_cubecl::CubeClExecutor` (no local stub executor implementation remains) (D-05).
    - Test 2: `WorkspaceExecutionToken::ensure_matches` fails when backend selector/capability token drift is introduced (D-08).
    - Test 3: Raw `validate_4c1e_envelope` no longer references CPU runtime profile constants (D-11).
  </behavior>
  <action>
In `api.rs`, delete the local stub `CubeClExecutor` implementation and synthetic `fill_staging_values` helper; use `cintx_cubecl::CubeClExecutor` directly inside `RecordingExecutor` path (D-05, D-13). Extend `WorkspaceExecutionToken` with backend contract fields from runtime options (`backend_intent`, `backend_capability_token`) and include them in `from_request` and `ensure_matches` comparisons (D-08). In `raw.rs`, remove `CUBECL_RUNTIME_PROFILE` import and CPU-profile gate from `validate_4c1e_envelope`; keep explicit envelope/capability checks routed through unsupported taxonomy text (D-11, D-12). Set `execution_options_from_opt` to preserve wgpu backend intent metadata for both query and evaluate.
  </action>
  <acceptance_criteria>
    - `rg -n "use cintx_cubecl::CubeClExecutor|RecordingExecutor" crates/cintx-rs/src/api.rs`
    - `rg -n "struct CubeClExecutor|fill_staging_values" crates/cintx-rs/src/api.rs` returns no matches
    - `rg -n "backend_intent|backend_capability_token|ensure_matches" crates/cintx-rs/src/api.rs`
    - `rg -n "CUBECL_RUNTIME_PROFILE|backend must be cpu" crates/cintx-compat/src/raw.rs` returns no matches
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-rs api::tests::query_evaluate_backend_selector_drift_is_detected_before_execution -- --exact && cargo test -p cintx-compat raw::tests::validate_4c1e_envelope_no_longer_references_cpu_profile_gate -- --exact</automated>
  </verify>
  <done>Safe and compat surfaces use shared CubeCL execution path with backend contract drift enforcement and no CPU-profile fallback logic.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add layered anti-pseudo and unsupported taxonomy regression tests</name>
  <files>crates/cintx-rs/src/api.rs, crates/cintx-compat/src/raw.rs</files>
  <read_first>docs/rust_crate_test_guideline.md, crates/cintx-rs/src/api.rs, crates/cintx-compat/src/raw.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md</read_first>
  <behavior>
    - Test 1: Safe evaluate output for a fixed fixture is not the historical monotonic synthetic sequence (D-15).
    - Test 2: Raw/safe unsupported paths assert taxonomy prefixes (`unsupported_family`, `unsupported_representation`, `outside Validated4C1E`) (D-16).
    - Test 3: Layered regression set covers runtime + cubecl + compat interaction paths (D-13).
  </behavior>
  <action>
Add explicit anti-pseudo regression tests in `api.rs` and `raw.rs`: for fixed fixtures, assert returned values do not match prior synthetic patterns (`1.0, 2.0, 3.0...` for cart and paired `v,-v` for spinor) and that unsupported errors include taxonomy prefixes from plan 03 (D-15, D-16). Add tests that mutate backend selector between query/evaluate in both safe and raw paths to ensure drift is rejected before execution (D-08). Keep tests narrow and deterministic per `docs/rust_crate_test_guideline.md`.
  </action>
  <acceptance_criteria>
    - `rg -n "monotonic|synthetic|stub sequence|outside Validated4C1E|unsupported_family|unsupported_representation" crates/cintx-rs/src/api.rs crates/cintx-compat/src/raw.rs`
    - `rg -n "backend selector|contract drift|planning_matches" crates/cintx-rs/src/api.rs crates/cintx-compat/src/raw.rs`
    - `rg -n "#\\[test\\]" crates/cintx-rs/src/api.rs crates/cintx-compat/src/raw.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-rs api::tests::evaluate_output_is_not_monotonic_stub_sequence -- --exact && cargo test -p cintx-compat raw::tests::unsupported_behavior_reports_reason_taxonomy -- --exact && cargo test -p cintx-rs --lib && cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>Layered regressions now fail if pseudo execution reappears or unsupported behavior loses explicit taxonomy/artifact evidence.</done>
</task>

</tasks>

<verification>
Run targeted and crate-level tests for `cintx-rs` and `cintx-compat`; verify that anti-pseudo and taxonomy assertions cover runtime-cubecl-compat boundaries.
</verification>

<success_criteria>
Public raw and safe surfaces are aligned with the real wgpu CubeCL path, and regressions that reintroduce synthetic outputs or generic unsupported errors are blocked by tests.
</success_criteria>

<output>
After completion, create `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/04-PLAN-SUMMARY.md`
</output>
