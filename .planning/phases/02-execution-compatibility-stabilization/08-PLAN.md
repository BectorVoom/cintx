---
phase: 02-execution-compatibility-stabilization
plan: 08
type: execute
wave: 4
depends_on:
  - 05
files_modified:
  - crates/cintx-cubecl/src/kernels/mod.rs
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
    - "The CubeCL backend completes the remaining base-family coverage for `3c1e` and `3c2e` instead of stopping at the first executor slice."
    - "Cart, spherical, and spinor outputs route through explicit CubeCL transform staging before results are handed back to compat, preserving upstream-compatible shapes and interleaved spinor layout."
    - "`cintx-cubecl` owns kernel execution plus representation-specific staging transforms, while `cintx-compat::layout` remains the sole owner of final caller-visible flat writes."
    - "Family support stays bounded: `4c1e` remains unsupported in this phase even though scaffolding exists in the tree."
  artifacts:
    - path: crates/cintx-cubecl/src/kernels/center_3c1e.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `3c1e` families."
      min_lines: 60
    - path: crates/cintx-cubecl/src/kernels/center_3c2e.rs
      provides: "CubeCL kernel entry points and launch wiring for manifest `3c2e` families."
      min_lines: 60
    - path: crates/cintx-cubecl/src/transform/c2s.rs
      provides: "Cartesian-to-spherical staging transforms used before compat finalizes caller-visible output."
      min_lines: 60
    - path: crates/cintx-cubecl/src/transform/c2spinor.rs
      provides: "Cartesian-to-spinor staging transforms with interleaved-double layout handling before compat final write."
      min_lines: 60
  key_links:
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/kernels/mod.rs
      via: "Executor family dispatch adds the remaining `3c1e` and `3c2e` launch paths without changing how the earlier family slice is selected."
      pattern: "center_3c1e|center_3c2e"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-cubecl/src/transform/mod.rs
      via: "Kernel outputs route through the representation-specific transform helpers into staging buffers before compat commits host-visible writes."
      pattern: "c2s|c2spinor|staging"
    - from: crates/cintx-cubecl/src/executor.rs
      to: crates/cintx-compat/src/layout.rs
      via: "The backend returns transformed staging results and metadata to compat; it does not own the final flat write into caller memory."
      pattern: "staging|layout"
---

<objective>
Finish the remaining CubeCL family and transform work that was intentionally split out of the executor-core plan.
Purpose: Complete the `3c1e`/`3c2e` family coverage and the cart/sph/spinor output-shaping path without overloading the earlier backend-core plan.
Output: `3c1e`/`3c2e` kernel modules plus transform helpers wired into the CubeCL executor.
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
@crates/cintx-compat/src/layout.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement the `3c1e` and `3c2e` CubeCL family kernels</name>
  <files>crates/cintx-cubecl/src/kernels/center_3c1e.rs, crates/cintx-cubecl/src/kernels/center_3c2e.rs, crates/cintx-cubecl/src/kernels/mod.rs</files>
  <read_first>crates/cintx-cubecl/src/kernels/center_3c1e.rs, crates/cintx-cubecl/src/kernels/center_3c2e.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Replace the `3c1e` and `3c2e` kernel stubs with concrete family launch modules that plug into the same registry introduced in Plan 05. Each module must accept the shared `ExecutionPlan`, derive chunk extents from runtime shell/output metadata, and return typed `ExecutionStats` without introducing family-specific buffer math in the executor or taking ownership of caller-visible flat writes. Update the kernel registry so `3c1e` and `3c2e` sit alongside the earlier families and keep any unsupported family path, including `4c1e`, mapped to `UnsupportedApi`.
  </action>
  <acceptance_criteria>
    - `rg -n "3c1e" crates/cintx-cubecl/src/kernels/center_3c1e.rs`
    - `rg -n "3c2e" crates/cintx-cubecl/src/kernels/center_3c2e.rs`
    - `rg -n "center_3c1e|center_3c2e" crates/cintx-cubecl/src/kernels/mod.rs`
    - `rg -n "4c1e|UnsupportedApi" crates/cintx-cubecl/src/kernels/mod.rs crates/cintx-cubecl/src/executor.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>The base-family registry now covers `3c1e` and `3c2e`, completing the Phase 2 family set required by EXEC-02.</done>
</task>

<task type="auto">
  <name>Task 2: Implement cart-to-spherical and cart-to-spinor output transforms and wire them into execution</name>
  <files>crates/cintx-cubecl/src/transform/mod.rs, crates/cintx-cubecl/src/transform/c2s.rs, crates/cintx-cubecl/src/transform/c2spinor.rs, crates/cintx-cubecl/src/executor.rs</files>
  <read_first>crates/cintx-cubecl/src/transform/mod.rs, crates/cintx-cubecl/src/transform/c2s.rs, crates/cintx-cubecl/src/transform/c2spinor.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-compat/src/layout.rs, docs/design/cintx_detailed_design.md §3.6.1, §7.2, and §11.1, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Implement explicit output-shape transforms for cart/sph/spinor staging. In `transform/c2s.rs`, add the cartesian-to-spherical path used for bra/ket style output shaping. In `transform/c2spinor.rs`, add the cartesian-to-spinor path that preserves interleaved doubles in the upstream-compatible order instead of exposing a separate complex layout family. In `transform/mod.rs`, export only the transform helpers needed by the executor. Update `executor.rs` so family launches always route through the correct transform path based on runtime representation metadata, emitting representation-shaped staging buffers and metadata that `cintx-compat::layout` will later commit into the caller-owned flat slice. Do not let `cintx-cubecl` transform modules or the executor write directly into caller-owned flat buffers. Add tests that assert the representation switch picks the correct transform, that spinor outputs report interleaved-double semantics, and that the executor-to-compat handoff remains staging-only.
  </action>
  <acceptance_criteria>
    - `rg -n "spheric|spherical" crates/cintx-cubecl/src/transform/c2s.rs`
    - `rg -n "spinor|interleaved" crates/cintx-cubecl/src/transform/c2spinor.rs`
    - `rg -n "c2s|c2spinor|staging|stage" crates/cintx-cubecl/src/executor.rs`
    - `rg -n "pub mod c2s|pub mod c2spinor" crates/cintx-cubecl/src/transform/mod.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib</automated>
  </verify>
  <done>CubeCL execution now produces cart/sph/spinor staging outputs through explicit transform code and hands them back to compat for the final flat write, satisfying the shape and complex-layout part of EXEC-04 without splitting write ownership.</done>
</task>

</tasks>

<verification>
Run the CubeCL library tests after both tasks to prove the family registry now covers the full Phase 2 base set and the representation-specific transform paths are active without taking over compat's caller-visible write contract.
</verification>

<success_criteria>
The CubeCL backend can execute the remaining `3c1e` and `3c2e` families and route results through the correct cart/sph/spinor staging transform paths while leaving the final caller-visible flat write to compat, without expanding Phase 2 to unsupported families.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/08-PLAN-SUMMARY.md`
</output>
