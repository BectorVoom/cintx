---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 06
type: execute
wave: 5
depends_on:
  - 02
  - 05
files_modified:
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-rs/src/api.rs
autonomous: true
requirements:
  - EXEC-01
  - OPT-01
  - OPT-02
  - OPT-03
gap_closure: true
must_haves:
  truths:
    - "Optional/unstable UnsupportedApi decisions used by the safe facade are sourced from shared compat-raw gate logic instead of duplicated facade-only checks."
    - "Safe `query_workspace()`/`evaluate()` preserve their typed EXEC-01 contract while enforcing with-f12/with-4c1e/source-only gates through the compat raw policy path."
    - "Out-of-envelope optional calls (F12 and Validated4C1E) fail with explicit reasons consistent with compat raw behavior."
  artifacts:
    - path: crates/cintx-compat/src/raw.rs
      provides: "Public compat-raw gate function for safe-facade optional/source policy enforcement plus regression tests."
      min_lines: 1360
    - path: crates/cintx-rs/src/api.rs
      provides: "Safe-facade query/evaluate path wired to compat raw policy checks before backend execution."
      min_lines: 560
  key_links:
    - from: crates/cintx-rs/src/api.rs
      to: crates/cintx-compat/src/raw.rs
      via: "Safe facade invokes compat raw policy helper to propagate UnsupportedApi decisions from shared gating logic."
      pattern: "cintx_compat::raw::|UnsupportedApi"
    - from: crates/cintx-compat/src/raw.rs
      to: crates/cintx-ops/src/resolver.rs
      via: "Compat raw policy helper remains resolver/profile-driven for optional and source-only checks."
      pattern: "descriptor_by_symbol|compiled_in_profiles|SourceOnly"
    - from: crates/cintx-rs/src/api.rs
      to: crates/cintx-runtime/src/planner.rs
      via: "Safe facade keeps runtime query/evaluate wiring while adding compat-policy preflight."
      pattern: "query_workspace|ExecutionPlan::new|evaluate"
---

<objective>
Close the unresolved safe-facade/compat key-link gap by wiring UnsupportedApi propagation through shared compat-raw gating logic.
Purpose: Satisfy the verifier’s missing `api.rs -> compat/raw.rs` contract without regressing the existing typed runtime-safe facade path.
Output: Compat raw policy helper + safe API integration + regression tests that prove optional/source gate parity.
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
@crates/cintx-compat/src/raw.rs
@crates/cintx-rs/src/api.rs
@crates/cintx-runtime/src/planner.rs
@crates/cintx-ops/src/resolver.rs
<interfaces>
From `crates/cintx-compat/src/raw.rs` (existing internal gates to reuse):
```rust
fn active_manifest_profile() -> &'static str;
fn validate_f12_envelope(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    dims: Option<&[i32]>,
    natural_extents: &[usize],
) -> Result<(), cintxRsError>;
fn validate_4c1e_envelope(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shells: &ShellTuple,
    dims: Option<&[i32]>,
    natural_extents: &[usize],
) -> Result<(), cintxRsError>;
```

From `crates/cintx-rs/src/api.rs` (must remain intact):
```rust
pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError>;
pub fn evaluate(self) -> Result<TypedEvaluationOutput, FacadeError>;
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Expose compat-raw safe-facade policy gate as a reusable public helper</name>
  <files>crates/cintx-compat/src/raw.rs</files>
  <read_first>crates/cintx-compat/src/raw.rs, crates/cintx-ops/src/resolver.rs, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-VERIFICATION.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Add a public helper in `cintx_compat::raw` that accepts the safe-facade execution metadata (`descriptor`, `representation`, `shell tuple`, and natural extents) and applies the same profile/source/optional envelope checks currently used by raw dispatch. The helper must call the existing `active_manifest_profile`, `descriptor.is_compiled_in_profile`, source-only gate, `validate_f12_envelope`, and `validate_4c1e_envelope` logic rather than duplicating alternate rules. Return the exact `cintxRsError::UnsupportedApi` text paths already produced by raw gates so reason strings stay consistent across safe and raw surfaces.
  </action>
  <acceptance_criteria>
    - `rg -n "pub fn .*safe.*gate|pub fn .*facade.*gate" crates/cintx-compat/src/raw.rs`
    - `rg -n "active_manifest_profile|is_compiled_in_profile|is_source_only|validate_f12_envelope|validate_4c1e_envelope" crates/cintx-compat/src/raw.rs`
    - `rg -n "#\\[test\\].*safe|#\\[test\\].*facade" crates/cintx-compat/src/raw.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib && cargo test -p cintx-compat --lib --features with-f12 && cargo test -p cintx-compat --lib --features with-4c1e</automated>
  </verify>
  <done>The compat raw module exports one shared policy gate callable by `cintx-rs`, with tests proving optional/source gate behavior stays deterministic.</done>
</task>

<task type="auto">
  <name>Task 2: Wire safe query/evaluate flow to compat-raw policy gate and remove orphaned link gap</name>
  <files>crates/cintx-rs/src/api.rs</files>
  <read_first>crates/cintx-rs/src/api.rs, crates/cintx-rs/Cargo.toml, crates/cintx-compat/src/raw.rs, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-VERIFICATION.md, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Import and invoke the new compat-raw policy helper inside the safe facade flow so optional/unstable UnsupportedApi decisions come from shared compat rules. Call the helper after `ExecutionPlan::new` yields descriptor/output metadata and before runtime backend execution, passing natural extents and typed shells from the safe session. Keep `query_workspace()`/`evaluate()` typed contracts unchanged (EXEC-01), and keep fail-closed behavior unchanged (no partial write, explicit typed errors). Add regression tests in `api.rs` asserting that out-of-envelope optional requests surface `FacadeError::UnsupportedApi` with the compat-origin reason phrases (`with-f12 sph envelope`, `outside Validated4C1E`, source-only feature gate text).
  </action>
  <acceptance_criteria>
    - `rg -n "use cintx_compat::raw|cintx_compat::raw::" crates/cintx-rs/src/api.rs`
    - `rg -n "ExecutionPlan::new|query_workspace\\(|evaluate\\(" crates/cintx-rs/src/api.rs`
    - `rg -n "with-f12 sph envelope|outside Validated4C1E|unstable-source-api" crates/cintx-rs/src/api.rs`
    - `rg -n "#\\[test\\].*unsupported|#\\[test\\].*validated4c1e|#\\[test\\].*source" crates/cintx-rs/src/api.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-rs --lib && cargo test -p cintx-rs --lib --features with-f12 && cargo test -p cintx-rs --lib --features with-4c1e</automated>
  </verify>
  <done>The missing `api.rs -> compat/raw.rs` key link is now real code, and safe-facade optional/source UnsupportedApi outcomes are compat-policy-derived and test-covered.</done>
</task>

</tasks>

<verification>
Run compat and safe-crate feature-matrix library tests to confirm shared policy-gate wiring and optional/source rejection parity across both surfaces.
</verification>

<success_criteria>
Phase 03 verification no longer reports the unresolved safe-facade/compat key link gap, and optional/source UnsupportedApi behavior is demonstrated as compat-policy-sourced.
</success_criteria>

<output>
After completion, create `.planning/phases/03-safe-surface-c-abi-shim-optional-families/06-PLAN-SUMMARY.md`
</output>
