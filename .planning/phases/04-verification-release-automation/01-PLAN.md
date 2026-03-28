---
phase: 04-verification-release-automation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/src/compare.rs
  - crates/cintx-oracle/src/lib.rs
autonomous: true
requirements:
  - VERI-01
must_haves:
  truths:
    - "Oracle comparison covers the required merge-blocking profiles (`base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`) and includes optional-family fixtures when enabled."
    - "Family tolerances remain explicit code-level constants and are not inferred at runtime."
    - "Oracle execution writes a complete mismatch report over the full fixture set before returning failure."
  artifacts:
    - path: crates/cintx-oracle/src/fixtures.rs
      provides: "Manifest-profile-aware fixture matrix generation using compiled manifest lock metadata."
      min_lines: 380
    - path: crates/cintx-oracle/src/compare.rs
      provides: "Full-matrix parity evaluator with non-fail-fast mismatch aggregation and explicit tolerance tables."
      min_lines: 760
    - path: crates/cintx-oracle/src/lib.rs
      provides: "Public exports for profile-aware oracle report entry points consumed by Phase 4 gate tooling."
      min_lines: 20
  key_links:
    - from: crates/cintx-oracle/src/fixtures.rs
      to: crates/cintx-ops/generated/compiled_manifest.lock.json
      via: "Fixture enumeration filters by compiled profile membership and stability class from lock metadata."
      pattern: "compiled_manifest.lock.json|with-f12\\+with-4c1e|stability"
    - from: crates/cintx-oracle/src/compare.rs
      to: crates/cintx-compat/src/raw.rs
      via: "Oracle parity compares compat raw outputs against legacy wrapper proxy outputs for each fixture."
      pattern: "raw::eval_raw|eval_legacy_symbol"
    - from: crates/cintx-oracle/src/compare.rs
      to: crates/cintx-oracle/src/fixtures.rs
      via: "Mismatch reports are persisted through shared artifact writer using required `/mnt/data` path metadata."
      pattern: "write_pretty_json_artifact|required_path|mismatch_count"
---

<objective>
Implement the Phase 4 oracle engine baseline with profile-aware fixture scope and non-fail-fast mismatch reporting.
Purpose: Satisfy VERI-01 and decisions D-01/D-02/D-03/D-04/D-14 by making oracle coverage profile-complete, tolerance-explicit, and fully auditable.
Output: Updated oracle fixture and compare modules that emit full mismatch artifacts for required profiles before failing.
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
@.planning/phases/04-verification-release-automation/04-CONTEXT.md
@.planning/phases/04-verification-release-automation/04-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md
@crates/cintx-oracle/src/fixtures.rs
@crates/cintx-oracle/src/compare.rs
@crates/cintx-ops/generated/compiled_manifest.lock.json
<interfaces>
From `crates/cintx-oracle/src/fixtures.rs`:
```rust
pub const PHASE2_FAMILIES: &[&str] = &["1e", "2e", "2c2e", "3c1e", "3c2e"];
pub fn build_phase2_representation_matrix(inputs: &OracleRawInputs) -> Result<Vec<OracleFixture>>;
pub fn write_representation_matrix_artifact(matrix: &[OracleFixture]) -> Result<ArtifactWriteResult>;
pub fn write_pretty_json_artifact(required_path: &'static str, fallback_name: &str, value: &Value) -> Result<ArtifactWriteResult>;
```

From `crates/cintx-oracle/src/compare.rs`:
```rust
pub fn tolerance_for_family(family: &str) -> Result<FamilyTolerance>;
pub fn verify_helper_surface_coverage(inputs: &OracleRawInputs) -> Result<()>;
pub fn generate_phase2_parity_report(inputs: &OracleRawInputs) -> Result<Phase2ParityReport>;
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Expand oracle fixture generation to required Phase 4 profile matrix</name>
  <files>crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/lib.rs</files>
  <read_first>crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/lib.rs, crates/cintx-ops/generated/compiled_manifest.lock.json, .planning/phases/04-verification-release-automation/04-CONTEXT.md, docs/design/cintx_detailed_design.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement profile-aware fixture APIs per D-01, D-03, and D-14 with concrete profile constants `base`, `with-f12`, `with-4c1e`, and `with-f12+with-4c1e`. Replace phase-specific naming by introducing `PHASE4_APPROVED_PROFILES`, `PHASE4_ORACLE_FAMILIES`, and profile-scoped builders `build_profile_representation_matrix(inputs, profile, include_unstable_source)` and `build_required_profile_matrices(inputs)`. Ensure stability filtering is concrete: include `stable` and `optional` entries by default; include `unstable_source` entries only when `include_unstable_source == true`. Keep required artifact metadata rooted at `/mnt/data` with fallback through `CINTX_ARTIFACT_DIR`.
  </action>
  <acceptance_criteria>
    - `rg -n "PHASE4_APPROVED_PROFILES|base|with-f12|with-4c1e|with-f12\\+with-4c1e" crates/cintx-oracle/src/fixtures.rs`
    - `rg -n "build_profile_representation_matrix|build_required_profile_matrices|include_unstable_source" crates/cintx-oracle/src/fixtures.rs`
    - `rg -n "stable|optional|unstable_source" crates/cintx-oracle/src/fixtures.rs`
    - `rg -n "pub use .*build_profile_representation_matrix|pub use .*build_required_profile_matrices" crates/cintx-oracle/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-oracle --lib fixtures::tests::representation_matrix_matches_manifest_fixtures -- --exact</automated>
  </verify>
  <done>Oracle fixtures can be generated for all four approved profiles plus optional unstable-source nightly mode without hardcoded symbol lists.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor parity comparison to full mismatch aggregation with explicit tolerance governance</name>
  <files>crates/cintx-oracle/src/compare.rs</files>
  <read_first>crates/cintx-oracle/src/compare.rs, crates/cintx-oracle/src/fixtures.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md, .planning/phases/04-verification-release-automation/04-RESEARCH.md, docs/design/cintx_detailed_design.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement a profile-aware parity entrypoint `generate_profile_parity_report(inputs, profile, include_unstable_source)` and keep family tolerances as explicit constants/functions in code (D-02). Replace fail-fast `bail!` behavior with mismatch accumulation per fixture (D-04): collect raw-vs-upstream, raw-vs-optimizer, and layout failures into a `mismatches` array, always write report JSON containing `profile`, `fixture_count`, `mismatch_count`, and `mismatches`, then return failure only after report persistence if `mismatch_count > 0`. Enforce required-profile inclusion semantics (D-01/D-03) by using the profile fixture builders from Task 1 and defaulting merge-blocking mode to `include_unstable_source = false`.
  </action>
  <acceptance_criteria>
    - `rg -n "generate_profile_parity_report|profile.*include_unstable_source" crates/cintx-oracle/src/compare.rs`
    - `rg -n "mismatch_count|mismatches|fixture_count" crates/cintx-oracle/src/compare.rs`
    - `rg -n "TOL_1E_ATOL|TOL_2E_ATOL|TOL_2C2E_3C2E_ATOL|TOL_3C1E_ATOL|ZERO_THRESHOLD|tolerance_for_family" crates/cintx-oracle/src/compare.rs`
    - `rg -n "write_pretty_json_artifact\\(|oracle parity failed with .* mismatches" crates/cintx-oracle/src/compare.rs`
    - `rg -n "#\\[test\\].*parity|#\\[test\\].*mismatch|#\\[test\\].*artifacts" crates/cintx-oracle/src/compare.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-oracle --lib</automated>
  </verify>
  <done>Oracle parity always emits complete mismatch artifacts and fails only after full-fixture analysis with explicit per-family tolerances.</done>
</task>

</tasks>

<verification>
Run the oracle crate test suite and confirm profile-aware fixture generation plus full-matrix mismatch reporting are both active and artifact-backed.
</verification>

<success_criteria>
VERI-01 is implementable through a manifest-driven oracle comparator that covers required profiles, keeps tolerance policy explicit in code, and produces full mismatch evidence before failing.
</success_criteria>

<output>
After completion, create `.planning/phases/04-verification-release-automation/01-PLAN-SUMMARY.md`
</output>
