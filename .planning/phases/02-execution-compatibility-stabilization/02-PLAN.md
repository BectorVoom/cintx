---
phase: 02-execution-compatibility-stabilization
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - Cargo.toml
  - Cargo.lock
  - crates/cintx-compat/Cargo.toml
  - crates/cintx-cubecl/Cargo.toml
  - crates/cintx-oracle/Cargo.toml
  - crates/cintx-compat/src/lib.rs
  - crates/cintx-cubecl/src/lib.rs
  - crates/cintx-oracle/src/lib.rs
autonomous: true
requirements:
  - EXEC-02
  - EXEC-03
must_haves:
  truths:
    - "Maintainers can build and test `cintx-compat`, `cintx-cubecl`, and `cintx-oracle` through the active workspace instead of leaving Phase 2 crates outside CI and local verification."
    - "Phase 2 execution work stays scoped to compat/runtime/backend/oracle crates; `cintx-rs` and `cintx-capi` remain out of the workspace until Phase 3."
    - "The workspace lock captures the Phase 2 dependency graph deterministically so later CubeCL, compat, and oracle work can verify under `cargo --locked`."
  artifacts:
    - path: Cargo.toml
      provides: "Workspace membership and default-member activation for the Phase 2 crates."
      min_lines: 20
    - path: crates/cintx-compat/Cargo.toml
      provides: "Compat crate dependency contract on `cintx-core`, `cintx-ops`, and `cintx-runtime`."
      min_lines: 10
    - path: crates/cintx-cubecl/Cargo.toml
      provides: "CubeCL backend dependency contract pinned to the approved `cubecl 0.9.x` line."
      min_lines: 10
    - path: crates/cintx-oracle/Cargo.toml
      provides: "Oracle crate dependency contract for vendored upstream comparison work."
      min_lines: 10
  key_links:
    - from: Cargo.toml
      to: crates/cintx-compat/Cargo.toml
      via: "Adds `cintx-compat` to workspace/default-members so compat code is built by normal verification commands."
      pattern: "cintx-compat"
    - from: Cargo.toml
      to: crates/cintx-cubecl/Cargo.toml
      via: "Adds `cintx-cubecl` to workspace/default-members so backend code participates in Phase 2 checks."
      pattern: "cintx-cubecl"
    - from: Cargo.toml
      to: crates/cintx-oracle/Cargo.toml
      via: "Adds `cintx-oracle` to workspace/default-members so parity tests can land before Phase 4."
      pattern: "cintx-oracle"
---

<objective>
Activate the Phase 2 crates in the live workspace and make them buildable verification targets before implementation work lands.
Purpose: Prevent the compat, CubeCL, and oracle work from drifting outside the main Cargo graph while keeping the safe facade and C ABI shim deferred to Phase 3.
Output: Updated workspace membership, locked crate dependencies, and crate roots that compile under `cargo test`.
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
@.planning/phases/01-manifest-planner-foundation/01-SUMMARY.md
@.planning/phases/01-manifest-planner-foundation/02-PLAN-SUMMARY.md
@.planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md
@AGENTS.md
@Cargo.toml
@crates/cintx-compat/Cargo.toml
@crates/cintx-cubecl/Cargo.toml
@crates/cintx-oracle/Cargo.toml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add compat, CubeCL, and oracle crates to the active workspace</name>
  <files>Cargo.toml, Cargo.lock, crates/cintx-compat/Cargo.toml, crates/cintx-cubecl/Cargo.toml, crates/cintx-oracle/Cargo.toml</files>
  <read_first>Cargo.toml, Cargo.lock, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, AGENTS.md, crates/cintx-runtime/Cargo.toml, crates/cintx-compat/Cargo.toml, crates/cintx-cubecl/Cargo.toml, crates/cintx-oracle/Cargo.toml</read_first>
  <action>
Update the root workspace so `[workspace].members` and `[workspace].default-members` include exactly `crates/cintx-core`, `crates/cintx-ops`, `crates/cintx-runtime`, `crates/cintx-compat`, `crates/cintx-cubecl`, and `crates/cintx-oracle`. Keep `crates/cintx-rs` and `crates/cintx-capi` out of the workspace for this phase. In `crates/cintx-compat/Cargo.toml`, add path dependencies on `cintx-core`, `cintx-ops`, and `cintx-runtime`, plus `smallvec = "1"` and `tracing = "0.1"`. In `crates/cintx-cubecl/Cargo.toml`, add `cubecl = "0.9.0"`, path dependencies on `cintx-core`, `cintx-ops`, and `cintx-runtime`, plus `smallvec = "1"` and `tracing = "0.1"`. In `crates/cintx-oracle/Cargo.toml`, add `anyhow = "1.0.102"`, `serde_json = "1.0.145"`, and path dependencies on `cintx-core` and `cintx-ops`, and add `bindgen = "0.71.1"` plus `cc = "1.2.15"` under build-dependencies. Regenerate `Cargo.lock` so the new members resolve under the pinned toolchain.
  </action>
  <acceptance_criteria>
    - `rg -n "crates/cintx-compat" Cargo.toml`
    - `rg -n "crates/cintx-cubecl" Cargo.toml`
    - `rg -n "crates/cintx-oracle" Cargo.toml`
    - `rg -n "cubecl = \\\"0\\.9\\.0\\\"" crates/cintx-cubecl/Cargo.toml`
    - `rg -n "bindgen = \\\"0\\.71\\.1\\\"" crates/cintx-oracle/Cargo.toml`
    - `rg -n "cc = \\\"1\\.2\\.15\\\"" crates/cintx-oracle/Cargo.toml`
    - `rg -n "smallvec = \\\"1\\\"" crates/cintx-compat/Cargo.toml crates/cintx-cubecl/Cargo.toml`
  </acceptance_criteria>
  <verify>
    <automated>cargo metadata --no-deps >/tmp/cintx-phase2-metadata.json && rg -n "cintx-(compat|cubecl|oracle)" /tmp/cintx-phase2-metadata.json</automated>
  </verify>
  <done>The phase-2 crates are first-class workspace members with the exact dependency floor needed for compat, CubeCL, and oracle work, while Phase 3 crates stay out of scope.</done>
</task>

<task type="auto">
  <name>Task 2: Make the newly activated crate roots pass library smoke tests</name>
  <files>crates/cintx-compat/src/lib.rs, crates/cintx-cubecl/src/lib.rs, crates/cintx-oracle/src/lib.rs</files>
  <read_first>crates/cintx-compat/src/lib.rs, crates/cintx-cubecl/src/lib.rs, crates/cintx-oracle/src/lib.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, AGENTS.md</read_first>
  <action>
Replace the current one-line stub crate roots with explicit module exports and minimal `#[cfg(test)]` smoke tests that prove each crate now builds inside the workspace. `crates/cintx-compat/src/lib.rs` must continue exporting `helpers`, `layout`, `legacy`, `optimizer`, `raw`, and `transform`. `crates/cintx-cubecl/src/lib.rs` must continue exporting `executor`, `kernels`, `resident_cache`, `specialization`, `transfer`, and `transform`. `crates/cintx-oracle/src/lib.rs` must continue exporting `compare` and `fixtures`. Keep these smoke tests narrow: assert the exported modules compile and the crate roots remain Phase-2-only surfaces, without adding safe facade or C ABI exports here.
  </action>
  <acceptance_criteria>
    - `rg -n "#\\[cfg\\(test\\)\\]" crates/cintx-compat/src/lib.rs`
    - `rg -n "#\\[cfg\\(test\\)\\]" crates/cintx-cubecl/src/lib.rs`
    - `rg -n "#\\[cfg\\(test\\)\\]" crates/cintx-oracle/src/lib.rs`
    - `rg -n "pub mod transform;" crates/cintx-cubecl/src/lib.rs`
    - `rg -n "pub mod compare;" crates/cintx-oracle/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib && cargo test -p cintx-cubecl --lib && cargo test -p cintx-oracle --lib</automated>
  </verify>
  <done>The activated crates are now visible to normal `cargo test` workflows and have minimal smoke coverage that will catch future workspace-regression mistakes early.</done>
</task>

</tasks>

<verification>
Run `cargo metadata --no-deps` to confirm workspace membership, then run each new crate's library tests to prove Phase 2 code is no longer outside the build graph.
</verification>

<success_criteria>
The workspace now includes compat/CubeCL/oracle crates, their manifests use the approved dependency lines, and all three crates pass basic library tests as active members.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/02-PLAN-SUMMARY.md`
</output>
