---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - Cargo.toml
  - Cargo.lock
  - crates/cintx-rs/Cargo.toml
  - crates/cintx-capi/Cargo.toml
  - crates/cintx-rs/src/lib.rs
  - crates/cintx-rs/src/api.rs
  - crates/cintx-rs/src/builder.rs
  - crates/cintx-rs/src/prelude.rs
  - crates/cintx-capi/src/lib.rs
autonomous: true
requirements:
  - OPT-03
must_haves:
  truths:
    - "`cintx-rs` and `cintx-capi` are active workspace members so Phase 3 surfaces build under normal workspace verification."
    - "Cargo feature gates exist for `with-f12`, `with-4c1e`, and `unstable-source-api`, enabling compile-time control of optional and unstable surfaces."
    - "Stable `cintx-rs` exports remain unchanged unless `unstable-source-api` is enabled, and unstable APIs live in an explicit unstable namespace."
    - "`cintx-capi` remains a stable-only boundary in this phase and does not expose unstable source-only symbols."
  artifacts:
    - path: Cargo.toml
      provides: "Workspace/member activation and top-level Phase 3 feature declarations."
      min_lines: 40
    - path: crates/cintx-rs/Cargo.toml
      provides: "Facade crate dependency and feature wiring for stable/optional/unstable gating."
      min_lines: 20
    - path: crates/cintx-capi/Cargo.toml
      provides: "C ABI crate dependency and feature wiring constrained to stable exports."
      min_lines: 20
    - path: crates/cintx-rs/src/api.rs
      provides: "Safe facade contract scaffold with explicit stable vs unstable namespace split."
      min_lines: 40
  key_links:
    - from: Cargo.toml
      to: crates/cintx-rs/Cargo.toml
      via: "Workspace and feature wiring makes `cintx-rs` compile under Phase 3 feature profiles."
      pattern: "cintx-rs|with-f12|with-4c1e|unstable-source-api"
    - from: Cargo.toml
      to: crates/cintx-capi/Cargo.toml
      via: "Workspace and feature wiring makes the optional C ABI crate buildable without exposing unstable symbols."
      pattern: "cintx-capi|capi"
    - from: crates/cintx-rs/src/lib.rs
      to: crates/cintx-rs/src/api.rs
      via: "Stable exports stay default while unstable exports are compile-gated behind `unstable-source-api`."
      pattern: "cfg\\(feature = \"unstable-source-api\"\\)"
---

<objective>
Establish Phase 3 workspace, feature, and namespace scaffolding for safe and C ABI surfaces before behavioral implementation.
Purpose: Lock compile-time boundaries for optional/unstable families and prevent stable-surface drift.
Output: Active workspace members for `cintx-rs`/`cintx-capi`, feature-gated manifests, and explicit stable-vs-unstable export scaffolds.
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
@Cargo.toml
@crates/cintx-rs/Cargo.toml
@crates/cintx-rs/src/lib.rs
@crates/cintx-rs/src/api.rs
@crates/cintx-capi/Cargo.toml
@crates/cintx-capi/src/lib.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Activate Phase 3 crates and feature topology in workspace manifests</name>
  <files>Cargo.toml, Cargo.lock, crates/cintx-rs/Cargo.toml, crates/cintx-capi/Cargo.toml</files>
  <read_first>Cargo.toml, Cargo.lock, crates/cintx-rs/Cargo.toml, crates/cintx-capi/Cargo.toml, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-RESEARCH.md, AGENTS.md</read_first>
  <action>
Update workspace manifests so `crates/cintx-rs` and `crates/cintx-capi` are included in both `[workspace].members` and `[workspace].default-members`. Add explicit Phase 3 feature gates (`with-f12`, `with-4c1e`, `unstable-source-api`, and `capi`) and wire dependencies so optional-family and unstable-family gates are compile-time visible across relevant crates (per D-09). Keep feature naming and mapping explicit (`with-f12`/`with-4c1e` in workspace, with upstream underscore translation where needed) to avoid profile drift. Regenerate `Cargo.lock` after dependency/feature wiring changes. Keep C ABI as stable-surface only and do not introduce unstable C exports in manifest wiring (per D-14). Record comments/docs in manifest files that unstable-to-stable promotion is release-gated by manifest/oracle evidence plus explicit maintainer decision (per D-15).
  </action>
  <acceptance_criteria>
    - `rg -n "crates/cintx-rs|crates/cintx-capi" Cargo.toml`
    - `rg -n "with-f12|with-4c1e|unstable-source-api|capi" Cargo.toml`
    - `rg -n "with-f12|with-4c1e|unstable-source-api" crates/cintx-rs/Cargo.toml`
    - `rg -n "with-f12|with-4c1e|capi" crates/cintx-capi/Cargo.toml`
  </acceptance_criteria>
  <verify>
    <automated>cargo metadata --no-deps >/tmp/cintx-phase3-metadata.json && rg -n "cintx-rs|cintx-capi" /tmp/cintx-phase3-metadata.json && cargo check -p cintx-rs && cargo check -p cintx-capi</automated>
  </verify>
  <done>Phase 3 crates and feature gates are buildable in the workspace with explicit compile-time controls for optional/unstable families and stable-only C ABI boundaries.</done>
</task>

<task type="auto">
  <name>Task 2: Define stable and unstable namespace scaffolds for facade and C ABI boundaries</name>
  <files>crates/cintx-rs/src/lib.rs, crates/cintx-rs/src/api.rs, crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/prelude.rs, crates/cintx-capi/src/lib.rs</files>
  <read_first>crates/cintx-rs/src/lib.rs, crates/cintx-rs/src/api.rs, crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/prelude.rs, crates/cintx-capi/src/lib.rs, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md, docs/design/cintx_detailed_design.md §5.4, §10.1</read_first>
  <action>
Replace stub module text with explicit API-surface scaffolding. In `cintx-rs`, declare stable facade modules/exports and a separate unstable namespace compiled only behind `#[cfg(feature = "unstable-source-api")]` (per D-13). Ensure unstable symbols are absent from stable builds and that disabled unstable requests will map to explicit `UnsupportedApi` once behavior is implemented (per D-16). Keep safe-surface signatures centered on typed session/query/evaluate contracts to support later D-01/D-03 implementation without exposing raw compat pointers. In `cintx-capi/src/lib.rs`, keep exports limited to stable shim/error modules and explicitly avoid unstable-source C exports in Phase 3 (per D-14).
  </action>
  <acceptance_criteria>
    - `rg -n "cfg\\(feature = \"unstable-source-api\"\\)" crates/cintx-rs/src/lib.rs crates/cintx-rs/src/api.rs`
    - `rg -n "pub mod api|pub mod builder|pub mod prelude" crates/cintx-rs/src/lib.rs`
    - `rg -n "pub mod errors|pub mod shim" crates/cintx-capi/src/lib.rs`
    - `rg -n "unsupported|unstable" crates/cintx-rs/src/api.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo check -p cintx-rs && cargo check -p cintx-rs --features unstable-source-api && cargo check -p cintx-capi</automated>
  </verify>
  <done>Stable and unstable namespace boundaries are explicit in source and compile-gated correctly, with C ABI remaining stable-only for Phase 3.</done>
</task>

</tasks>

<verification>
Run workspace metadata and crate checks to confirm Phase 3 crate activation plus stable/unstable compile gating behavior.
</verification>

<success_criteria>
`cintx-rs` and `cintx-capi` are active build targets, optional/unstable feature gates are wired, and stable-vs-unstable namespace boundaries are encoded in source before behavioral implementation begins.
</success_criteria>

<output>
After completion, create `.planning/phases/03-safe-surface-c-abi-shim-optional-families/01-PLAN-SUMMARY.md`
</output>
