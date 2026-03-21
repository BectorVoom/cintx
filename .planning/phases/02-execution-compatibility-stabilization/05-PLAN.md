---
phase: 02-execution-compatibility-stabilization
plan: 05
type: execute
wave: 3
depends_on:
  - 04
files_modified:
  - crates/cintx-cubecl/src/lib.rs
  - crates/cintx-cubecl/src/executor.rs
  - crates/cintx-cubecl/src/resident_cache.rs
  - crates/cintx-cubecl/src/specialization.rs
  - crates/cintx-cubecl/src/transfer.rs
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
must_haves:
  truths:
    - "The CubeCL backend exposes a concrete `CubeClExecutor` type that later compat work can instantiate through the explicit Wave 1 crate dependency."
    - "The CubeCL backend can execute the first base-family slice `1e`, `2e`, and `2c2e` instead of stopping at executor scaffolding."
    - "Resident-cache and transfer planning stay inside `cintx-cubecl`, keep host work limited to marshaling/copy orchestration, and continue to fail with typed allocation errors."
  artifacts:
    - path: crates/cintx-cubecl/src/executor.rs
      provides: "Concrete `CubeClExecutor` implementation of the runtime `BackendExecutor` contract."
      min_lines: 120
    - path: crates/cintx-cubecl/src/transfer.rs
      provides: "Transfer planning for shell metadata, workspace buffers, and output slices."
      min_lines: 60
    - path: crates/cintx-cubecl/src/kernels/one_electron.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `1e` families."
      min_lines: 60
    - path: crates/cintx-cubecl/src/kernels/two_electron.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `2e` families."
      min_lines: 60
    - path: crates/cintx-cubecl/src/kernels/center_2c2e.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `2c2e` families."
      min_lines: 60
  key_links:
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/transfer.rs
      via: "The concrete backend stages metadata, workspace, and output buffers through a transfer plan without bypassing the fallible allocation contract."
      pattern: "TransferPlan"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/resident_cache.rs
      via: "Executor launches reuse device-resident metadata through the basis/device cache instead of rebuilding state for every chunk."
      pattern: "ResidentCache|DeviceResidentCache"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/kernels/mod.rs
      via: "Executor family dispatch selects the correct kernel module for the initial `1e`/`2e`/`2c2e` launch slice."
      pattern: "one_electron|two_electron|center_2c2e"
---

<objective>
Build the concrete CubeCL backend core and the first executable base-family slice after the runtime contract lands.
Purpose: Convert the backend seam into a real `CubeClExecutor`, resident-cache/transfer pipeline, and `1e`/`2e`/`2c2e` execution coverage before the remaining `3c*` and transform work.
Output: `CubeClExecutor` core plus the `1e`/`2e`/`2c2e` family kernel registry.
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
@.planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md
@crates/cintx-runtime/src/dispatch.rs
@crates/cintx-cubecl/src/executor.rs
@crates/cintx-cubecl/src/kernels/mod.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement the CubeCL executor core, specialization keys, resident cache, and transfer planner</name>
  <files>crates/cintx-cubecl/src/lib.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/resident_cache.rs, crates/cintx-cubecl/src/specialization.rs, crates/cintx-cubecl/src/transfer.rs</files>
  <read_first>crates/cintx-cubecl/src/lib.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/resident_cache.rs, crates/cintx-cubecl/src/specialization.rs, crates/cintx-cubecl/src/transfer.rs, crates/cintx-runtime/src/dispatch.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, AGENTS.md</read_first>
  <action>
Implement the CubeCL backend core in the `cubecl 0.9.0` line only. In `executor.rs`, add `CubeClExecutor` that implements `BackendExecutor`, exports a concrete constructor the compat crate can call later, and refuses unsupported families through `cintxRsError::UnsupportedApi`. In `specialization.rs`, define a `SpecializationKey` that includes canonical family, representation, component rank, and shell angular-momentum tuple so family kernels can specialize without leaking raw symbol names. In `resident_cache.rs`, add a basis/device-scoped resident metadata cache keyed by basis hash plus representation. In `transfer.rs`, add a `TransferPlan` that stages shell metadata, workspace buffers, and output slices while keeping host CPU work limited to validation, marshaling, and copy orchestration. Keep `center_4c1e` unsupported in this phase even if a stub file exists, and ensure allocation/transfer failures map to `HostAllocationFailed` or `DeviceOutOfMemory` rather than ad hoc errors. Update `lib.rs` exports and add focused unit tests around specialization keys and executor family support.
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
  <done>The CubeCL backend core exists, plugs into runtime through `BackendExecutor`, and exposes the concrete executor surface that later compat work will call without broadening support beyond the locked Phase 2 scope.</done>
</task>

<task type="auto">
  <name>Task 2: Implement the `1e`, `2e`, and `2c2e` CubeCL family registry and launch path</name>
  <files>crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-cubecl/src/kernels/two_electron.rs, crates/cintx-cubecl/src/kernels/center_2c2e.rs, crates/cintx-cubecl/src/executor.rs</files>
  <read_first>crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-cubecl/src/kernels/two_electron.rs, crates/cintx-cubecl/src/kernels/center_2c2e.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, docs/design/cintx_detailed_design.md §7.1 and §7.6</read_first>
  <action>
Replace the kernel stubs for `1e`, `2e`, and `2c2e` with concrete family launch modules. In each file, expose a family-specific launch entry that accepts the runtime `ExecutionPlan`, specialization key, and transfer plan, and returns typed `ExecutionStats`. In `kernels/mod.rs`, register exactly these three families with the executor-facing lookup table and keep `center_4c1e`, `center_3c1e`, and `center_3c2e` out of the registry until the follow-on plan lands. Use manifest `canonical_family` and representation metadata to choose the launch path; do not branch on raw symbol names in the executor. Add unit tests that prove the family registry resolves `1e`, `2e`, and `2c2e` and rejects unsupported families.
  </action>
  <acceptance_criteria>
    - `rg -n "one[_ ]electron|family_name.*1e|canonical_family.*1e" crates/cintx-cubecl/src/kernels/one_electron.rs`
    - `rg -n "two[_ ]electron|canonical_family.*2e" crates/cintx-cubecl/src/kernels/two_electron.rs`
    - `rg -n "2c2e|center_2c2e" crates/cintx-cubecl/src/kernels/center_2c2e.rs`
    - `rg -n "center_4c1e|center_3c1e|center_3c2e" crates/cintx-cubecl/src/kernels/mod.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>The CubeCL backend has concrete family launch paths for `1e`, `2e`, and `2c2e`, and the family registry continues to fail closed for unsupported families until the follow-on plan lands.</done>
</task>

</tasks>

<verification>
Run the CubeCL library tests after both tasks to prove the executor core, cache/transfer pipeline, and the first family registry slice all compile and agree on supported-family selection.
</verification>

<success_criteria>
The CubeCL backend exposes a concrete executor, stages device/cache/transfer state explicitly, and executes the `1e`/`2e`/`2c2e` slice without expanding Phase 2 to unsupported families.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/05-PLAN-SUMMARY.md`
</output>
