---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 05
type: execute
wave: 4
depends_on:
  - 01
  - 03
files_modified:
  - crates/cintx-rs/Cargo.toml
  - crates/cintx-capi/Cargo.toml
  - crates/cintx-rs/src/builder.rs
  - crates/cintx-rs/src/prelude.rs
autonomous: true
requirements:
  - EXEC-01
  - OPT-03
gap_closure: true
must_haves:
  truths:
    - "Safe-surface crate manifests expose explicit feature/dependency wiring so optional and unstable gates are auditable at the crate boundary."
    - "Rust callers can construct typed sessions through the stable `SessionBuilder` and import the needed safe-surface entry points via `prelude`."
    - "The C ABI crate remains stable-only while forwarding optional-family feature gates without exposing unstable-source C exports."
  artifacts:
    - path: crates/cintx-rs/Cargo.toml
      provides: "Safe crate feature/dependency wiring, including explicit compat bridge dependency and threshold-compliant manifest depth."
      min_lines: 20
    - path: crates/cintx-capi/Cargo.toml
      provides: "Stable-only C ABI crate wiring with explicit optional-family forwarding and threshold-compliant manifest depth."
      min_lines: 20
    - path: crates/cintx-rs/src/builder.rs
      provides: "Expanded typed builder API and tests for session/options composition."
      min_lines: 120
    - path: crates/cintx-rs/src/prelude.rs
      provides: "Expanded curated stable re-exports and threshold-compliant prelude contract."
      min_lines: 30
  key_links:
    - from: crates/cintx-rs/Cargo.toml
      to: crates/cintx-rs/src/builder.rs
      via: "Manifest feature/dependency wiring supports the concrete safe-builder API surface used by stable callers."
      pattern: "cintx-compat|with-f12|with-4c1e|unstable-source-api"
    - from: crates/cintx-rs/src/prelude.rs
      to: crates/cintx-rs/src/builder.rs
      via: "Stable prelude re-exports include the expanded builder entry points used for safe session creation."
      pattern: "SessionBuilder|SessionRequest|ExecutionOptions"
    - from: crates/cintx-capi/Cargo.toml
      to: crates/cintx-capi/src/lib.rs
      via: "Stable-only capi feature wiring is preserved while optional-family forwarding remains explicit."
      pattern: "capi|with-f12|with-4c1e"
---

<objective>
Close the Phase 03 artifact-depth verification gaps for crate manifests and safe facade ergonomic scaffolding.
Purpose: Resolve verifier-reported stub-depth failures without changing already-completed Plan 01-04 intent.
Output: Threshold-compliant Cargo manifests plus expanded builder/prelude artifacts that satisfy declared must-have substance checks.
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
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-VERIFICATION.md
@AGENTS.md
@crates/cintx-rs/Cargo.toml
@crates/cintx-capi/Cargo.toml
@crates/cintx-rs/src/builder.rs
@crates/cintx-rs/src/prelude.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Raise crate-manifest depth and explicit feature/dependency wiring to close threshold gaps</name>
  <files>crates/cintx-rs/Cargo.toml, crates/cintx-capi/Cargo.toml</files>
  <read_first>crates/cintx-rs/Cargo.toml, crates/cintx-capi/Cargo.toml, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-VERIFICATION.md, .planning/REQUIREMENTS.md</read_first>
  <action>
Update both crate manifests so they are not threshold stubs and have concrete, auditable wiring. In `crates/cintx-rs/Cargo.toml`, add an explicit `cintx-compat` dependency and feature forwarding entries so `with-f12`, `with-4c1e`, and `unstable-source-api` are visibly propagated from the safe facade layer (per D-09 and D-16). In `crates/cintx-capi/Cargo.toml`, keep `capi` stable-only and add explicit comments/feature forwarding entries that preserve no-unstable C exports in this phase (per D-14). Keep both files at or above 20 lines with concrete key/value entries (not blank padding).
  </action>
  <acceptance_criteria>
    - `wc -l crates/cintx-rs/Cargo.toml | awk '{print $1}'` returns `>= 20`
    - `wc -l crates/cintx-capi/Cargo.toml | awk '{print $1}'` returns `>= 20`
    - `rg -n "cintx-compat" crates/cintx-rs/Cargo.toml`
    - `rg -n "with-f12|with-4c1e|unstable-source-api" crates/cintx-rs/Cargo.toml`
    - `rg -n "capi|with-f12|with-4c1e" crates/cintx-capi/Cargo.toml`
  </acceptance_criteria>
  <verify>
    <automated>cargo check -p cintx-rs && cargo check -p cintx-rs --features "with-f12 with-4c1e unstable-source-api" && cargo check -p cintx-capi</automated>
  </verify>
  <done>Both crate manifests satisfy the declared min_lines thresholds and expose concrete feature/dependency wiring aligned with Phase 03 contracts.</done>
</task>

<task type="auto">
  <name>Task 2: Expand safe builder and prelude into threshold-compliant ergonomic artifacts</name>
  <files>crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/prelude.rs</files>
  <read_first>crates/cintx-rs/src/builder.rs, crates/cintx-rs/src/prelude.rs, crates/cintx-rs/src/api.rs, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-VERIFICATION.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Expand `SessionBuilder` with concrete typed convenience methods and coverage tests so the file reaches at least 120 lines through real API surface, not filler. Add methods for explicit option composition (`profile_label`, memory-limit setters, chunk-size setters, and immutable rebuild helpers) that produce the same `SessionRequest` contract consumed by `query_workspace`/`evaluate` (per D-01 and D-03). Expand `prelude.rs` to at least 30 lines by grouping and documenting stable exports for builder/request/query/output/error entry points while preserving the existing unstable cfg-gated re-export boundary (per D-13). Add/adjust unit tests in `builder.rs` validating option propagation and built-request invariants.
  </action>
  <acceptance_criteria>
    - `wc -l crates/cintx-rs/src/builder.rs | awk '{print $1}'` returns `>= 120`
    - `wc -l crates/cintx-rs/src/prelude.rs | awk '{print $1}'` returns `>= 30`
    - `rg -n "fn profile_label|fn memory_limit|fn chunk_size|fn build" crates/cintx-rs/src/builder.rs`
    - `rg -n "pub use crate::api::|pub use crate::builder::SessionBuilder|cfg\\(feature = \"unstable-source-api\"\\)" crates/cintx-rs/src/prelude.rs`
    - `rg -n "#\\[test\\]" crates/cintx-rs/src/builder.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-rs --lib</automated>
  </verify>
  <done>The safe builder/prelude files satisfy declared depth thresholds with concrete typed APIs and tests, closing the verifier’s artifact-substance gaps.</done>
</task>

</tasks>

<verification>
Run crate checks and `cintx-rs` library tests to confirm manifest/build stability and threshold-compliant safe builder/prelude artifacts.
</verification>

<success_criteria>
All four threshold gaps from 03-VERIFICATION are reduced to zero for `crates/cintx-rs/Cargo.toml`, `crates/cintx-capi/Cargo.toml`, `crates/cintx-rs/src/builder.rs`, and `crates/cintx-rs/src/prelude.rs`.
</success_criteria>

<output>
After completion, create `.planning/phases/03-safe-surface-c-abi-shim-optional-families/05-PLAN-SUMMARY.md`
</output>
