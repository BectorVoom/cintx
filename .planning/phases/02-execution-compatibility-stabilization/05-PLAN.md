---
phase: 02-execution-compatibility-stabilization
plan: 05
type: execute
wave: 3
depends_on:
  - 04
files_modified:
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-cubecl/src/transform/mod.rs
  - crates/cintx-cubecl/src/transform/c2s.rs
  - crates/cintx-cubecl/src/transform/c2spinor.rs
  - crates/cintx-cubecl/src/executor.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-04
must_haves:
  truths:
    - "The CubeCL backend can execute the Phase 2 base families `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e` instead of stopping at executor scaffolding."
    - "Cart, spherical, and spinor outputs flow through explicit transform/writer code so family results honor upstream-compatible shape, ordering, and complex-layout semantics."
    - "Family support stays bounded: `4c1e` remains unsupported in this phase even though a stub exists in the tree."
  artifacts:
    - path: crates/cintx-cubecl/src/kernels/one_electron.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `1e` families."
      min_lines: 60
    - path: crates/cintx-cubecl/src/kernels/two_electron.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `2e` families."
      min_lines: 60
    - path: crates/cintx-cubecl/src/transform/c2s.rs
      provides: "Cartesian-to-spherical output transforms used by compat/base execution."
      min_lines: 60
    - path: crates/cintx-cubecl/src/transform/c2spinor.rs
      provides: "Cartesian-to-spinor output transforms with interleaved-double layout handling."
      min_lines: 60
  key_links:
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/kernels/mod.rs
      via: "Executor family dispatch selects the correct kernel module for each manifest family."
      pattern: "one_electron|two_electron|center_2c2e|center_3c1e|center_3c2e"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/transform/mod.rs
      via: "Kernel outputs route through transform helpers before host-visible writes."
      pattern: "c2s|c2spinor"
---

<objective>
Fill in the CubeCL family kernels and output transforms so the backend can produce the Phase 2 base-family results with the right shapes and ordering.
Purpose: Convert the executor core into actual base-family evaluation coverage and lock in the cart/sph/spinor layout semantics required by EXEC-04.
Output: Family kernel modules plus transform helpers wired into the CubeCL executor.
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
@crates/cintx-cubecl/src/executor.rs
@crates/cintx-cubecl/src/kernels/mod.rs
@crates/cintx-cubecl/src/transform/mod.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement the `1e`, `2e`, and `2c2e` CubeCL family kernels</name>
  <files>crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-cubecl/src/kernels/two_electron.rs, crates/cintx-cubecl/src/kernels/center_2c2e.rs</files>
  <read_first>crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-cubecl/src/kernels/two_electron.rs, crates/cintx-cubecl/src/kernels/center_2c2e.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, docs/design/cintx_detailed_design.md §7.1 and §7.6</read_first>
  <action>
Replace the kernel stubs for `1e`, `2e`, and `2c2e` with concrete family launch modules. In each file, expose a family-specific launch entry that accepts the runtime `ExecutionPlan`, specialization key, and transfer plan, and returns typed `ExecutionStats`. In `kernels/mod.rs`, register exactly these three families with the executor-facing lookup table and keep `center_4c1e` out of the registry. Use manifest `canonical_family` and representation metadata to choose the launch path; do not branch on raw symbol names in the executor. Add unit tests that prove the family registry resolves `1e`, `2e`, and `2c2e` and rejects `4c1e`.
  </action>
  <acceptance_criteria>
    - `rg -n "one[_ ]electron|family_name.*1e|canonical_family.*1e" crates/cintx-cubecl/src/kernels/one_electron.rs`
    - `rg -n "two[_ ]electron|canonical_family.*2e" crates/cintx-cubecl/src/kernels/two_electron.rs`
    - `rg -n "2c2e|center_2c2e" crates/cintx-cubecl/src/kernels/center_2c2e.rs`
    - `rg -n "center_4c1e" crates/cintx-cubecl/src/kernels/mod.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>The CubeCL backend has concrete family launch paths for `1e`, `2e`, and `2c2e`, and the family registry continues to fail closed for `4c1e`.</done>
</task>

<task type="auto">
  <name>Task 2: Implement the `3c1e` and `3c2e` CubeCL family kernels</name>
  <files>crates/cintx-cubecl/src/kernels/center_3c1e.rs, crates/cintx-cubecl/src/kernels/center_3c2e.rs, crates/cintx-cubecl/src/kernels/mod.rs</files>
  <read_first>crates/cintx-cubecl/src/kernels/center_3c1e.rs, crates/cintx-cubecl/src/kernels/center_3c2e.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Replace the `3c1e` and `3c2e` kernel stubs with concrete family launch modules that plug into the same registry used by Task 1. Each module must accept the shared `ExecutionPlan`, derive chunk extents from runtime shell/output metadata, and return typed `ExecutionStats` without introducing family-specific buffer math in the executor. Update the kernel registry so `3c1e` and `3c2e` sit alongside the earlier families and keep any unsupported family path mapped to `UnsupportedApi`.
  </action>
  <acceptance_criteria>
    - `rg -n "3c1e" crates/cintx-cubecl/src/kernels/center_3c1e.rs`
    - `rg -n "3c2e" crates/cintx-cubecl/src/kernels/center_3c2e.rs`
    - `rg -n "center_3c1e|center_3c2e" crates/cintx-cubecl/src/kernels/mod.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>The base-family registry now covers `3c1e` and `3c2e`, completing the Phase 2 family set required by EXEC-02.</done>
</task>

<task type="auto">
  <name>Task 3: Implement cart-to-spherical and cart-to-spinor output transforms and wire them into execution</name>
  <files>crates/cintx-cubecl/src/transform/mod.rs, crates/cintx-cubecl/src/transform/c2s.rs, crates/cintx-cubecl/src/transform/c2spinor.rs, crates/cintx-cubecl/src/executor.rs</files>
  <read_first>crates/cintx-cubecl/src/transform/mod.rs, crates/cintx-cubecl/src/transform/c2s.rs, crates/cintx-cubecl/src/transform/c2spinor.rs, crates/cintx-cubecl/src/executor.rs, docs/design/cintx_detailed_design.md §3.6.1, §7.2, and §11.1, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Implement explicit output-shape transforms for cart/sph/spinor writes. In `transform/c2s.rs`, add the cartesian-to-spherical path used for bra/ket style output shaping. In `transform/c2spinor.rs`, add the cartesian-to-spinor path that writes interleaved doubles in the upstream-compatible order instead of exposing a separate complex layout family. In `transform/mod.rs`, export only the transform helpers needed by the executor. Update `executor.rs` so family launches always route through the correct transform/writer path based on runtime representation metadata before any caller-visible write is finalized. Add tests that assert the representation switch picks the correct transform and that spinor outputs report interleaved-double semantics.
  </action>
  <acceptance_criteria>
    - `rg -n "spheric|spherical" crates/cintx-cubecl/src/transform/c2s.rs`
    - `rg -n "spinor|interleaved" crates/cintx-cubecl/src/transform/c2spinor.rs`
    - `rg -n "c2s|c2spinor" crates/cintx-cubecl/src/executor.rs`
    - `rg -n "pub mod c2s|pub mod c2spinor" crates/cintx-cubecl/src/transform/mod.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>CubeCL execution now produces cart/sph/spinor outputs through explicit transform code, satisfying the shape and complex-layout part of EXEC-04.</done>
</task>

</tasks>

<verification>
Run the CubeCL library tests after all three tasks to prove the family registry covers the base Phase 2 set and the representation-specific transform paths are active.
</verification>

<success_criteria>
The CubeCL backend can execute the base families and route results through the correct cart/sph/spinor transform paths without expanding Phase 2 to unsupported families.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/05-PLAN-SUMMARY.md`
</output>
