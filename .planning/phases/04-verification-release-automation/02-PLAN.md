---
phase: 04-verification-release-automation
plan: 02
type: execute
wave: 2
depends_on:
  - 01
files_modified:
  - xtask/Cargo.toml
  - xtask/src/main.rs
  - xtask/src/manifest_audit.rs
  - xtask/src/oracle_update.rs
autonomous: true
requirements:
  - VERI-02
must_haves:
  truths:
    - "Maintainers can run merge-blocking verification gates from one xtask CLI entrypoint without ad-hoc command chains."
    - "Manifest drift and profile coverage checks are evaluated against the compiled manifest lock, not hand-maintained lists."
    - "Helper/legacy parity and OOM no-partial-write checks are executable as explicit gate commands for CI."
  artifacts:
    - path: xtask/src/main.rs
      provides: "Executable command router for manifest audit, oracle compare, helper/legacy parity, and OOM contract checks."
      min_lines: 130
    - path: xtask/src/manifest_audit.rs
      provides: "Manifest lock drift checker enforcing required profile set and lock-derived symbol coverage."
      min_lines: 180
    - path: xtask/src/oracle_update.rs
      provides: "Oracle gate runner with full-report mode plus helper/legacy and OOM contract gate helpers."
      min_lines: 220
  key_links:
    - from: xtask/src/manifest_audit.rs
      to: crates/cintx-ops/generated/compiled_manifest.lock.json
      via: "Audit compares runtime-generated coverage against lock profile/symbol data."
      pattern: "compiled_manifest.lock.json|with-f12\\+with-4c1e|profiles"
    - from: xtask/src/oracle_update.rs
      to: crates/cintx-oracle/src/compare.rs
      via: "xtask gate calls profile-aware oracle comparison and helper coverage checks."
      pattern: "generate_profile_parity_report|verify_helper_surface_coverage"
    - from: xtask/src/oracle_update.rs
      to: crates/cintx-compat/src/raw.rs
      via: "OOM gate command executes no-partial-write failure-path regression tests."
      pattern: "memory_limit_failure_keeps_output_slice_unchanged"
---

<objective>
Create executable Phase 4 gate commands in xtask for manifest drift, oracle parity, helper/legacy parity, and OOM-contract enforcement.
Purpose: Satisfy the VERI-02 command surface needed by CI while honoring D-05, D-06, D-08, D-13, and D-14.
Output: Non-stub xtask CLI and gate modules with deterministic profile and failure semantics.
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
@xtask/Cargo.toml
@xtask/src/main.rs
@xtask/src/manifest_audit.rs
@xtask/src/oracle_update.rs
@crates/cintx-oracle/src/fixtures.rs
@crates/cintx-oracle/src/compare.rs
@crates/cintx-compat/src/raw.rs
@crates/cintx-runtime/src/workspace.rs
<interfaces>
From `crates/cintx-oracle/src/compare.rs` (after Plan 01):
```rust
pub fn verify_helper_surface_coverage(inputs: &OracleRawInputs) -> Result<()>;
pub fn generate_profile_parity_report(
    inputs: &OracleRawInputs,
    profile: &str,
    include_unstable_source: bool,
) -> Result<ProfileParityReport>;
```

From `crates/cintx-oracle/src/fixtures.rs` (after Plan 01):
```rust
pub fn build_required_profile_matrices(inputs: &OracleRawInputs) -> Result<Vec<ProfileFixtureMatrix>>;
pub fn build_profile_representation_matrix(
    inputs: &OracleRawInputs,
    profile: &str,
    include_unstable_source: bool,
) -> Result<Vec<OracleFixture>>;
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace xtask stubs with concrete gate CLI contracts</name>
  <files>xtask/Cargo.toml, xtask/src/main.rs</files>
  <read_first>xtask/Cargo.toml, xtask/src/main.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md, .planning/phases/04-verification-release-automation/04-RESEARCH.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Add real xtask dependencies (`anyhow = "1.0.102"`, `serde_json = "1.0.145"`, `cintx-oracle = { path = "../crates/cintx-oracle" }`, `cintx-ops = { path = "../crates/cintx-ops" }`) and replace the stub `main()` with a subcommand router exposing exactly these commands: `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, `oom-contract-check`. Parse the profile list as the concrete comma-separated set `base,with-f12,with-4c1e,with-f12+with-4c1e` (D-06), default `oracle-compare` to `include_unstable_source=false` (D-03), and return non-zero exit status on any gate failure (D-08).
  </action>
  <acceptance_criteria>
    - `rg -n "name = \"xtask\"|anyhow|serde_json|cintx-oracle|cintx-ops" xtask/Cargo.toml`
    - `rg -n "manifest-audit|oracle-compare|helper-legacy-parity|oom-contract-check" xtask/src/main.rs`
    - `rg -n "base,with-f12,with-4c1e,with-f12\\+with-4c1e|include_unstable_source" xtask/src/main.rs`
    - `rg -n "std::process::exit\\(1\\)|return Err\\(" xtask/src/main.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo run --manifest-path xtask/Cargo.toml -- --help</automated>
  </verify>
  <done>`xtask` is executable and exposes the required Phase 4 gate command contracts with deterministic profile defaults.</done>
</task>

<task type="auto">
  <name>Task 2: Implement lock-driven manifest drift and profile coverage audit command</name>
  <files>xtask/src/manifest_audit.rs</files>
  <read_first>xtask/src/manifest_audit.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-ops/generated/compiled_manifest.lock.json, .planning/phases/04-verification-release-automation/04-CONTEXT.md, docs/design/cintx_detailed_design.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement `run_manifest_audit(profiles: &[String], check_lock: bool)` that enforces D-14 lock authority and D-06 profile coverage. Require the approved profile set to exactly match `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`; fail if any profile is missing or extra. Read `crates/cintx-ops/generated/compiled_manifest.lock.json`, regenerate profile fixture symbol coverage via oracle fixture builders, and emit a JSON diff report with `missing_in_lock`, `missing_in_generated`, and `profile_scope_mismatch` fields at `/mnt/data/cintx_phase_04_manifest_audit.json` (with fallback metadata via `CINTX_ARTIFACT_DIR`).
  </action>
  <acceptance_criteria>
    - `rg -n "run_manifest_audit|profile_scope_mismatch|missing_in_lock|missing_in_generated" xtask/src/manifest_audit.rs`
    - `rg -n "base|with-f12|with-4c1e|with-f12\\+with-4c1e" xtask/src/manifest_audit.rs`
    - `rg -n "compiled_manifest.lock.json|cintx_phase_04_manifest_audit.json|/mnt/data" xtask/src/manifest_audit.rs`
    - `rg -n "build_required_profile_matrices|build_profile_representation_matrix" xtask/src/manifest_audit.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo run --manifest-path xtask/Cargo.toml -- manifest-audit --profiles base,with-f12,with-4c1e,with-f12+with-4c1e --check-lock</automated>
  </verify>
  <done>Manifest audit command fails on lock/profile drift and emits machine-readable drift evidence for CI consumption.</done>
</task>

<task type="auto">
  <name>Task 3: Implement oracle parity, helper/legacy parity, and OOM contract gate runners</name>
  <files>xtask/src/oracle_update.rs</files>
  <read_first>xtask/src/oracle_update.rs, crates/cintx-oracle/src/compare.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-compat/src/raw.rs, crates/cintx-runtime/src/workspace.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement three gate functions and wire them to subcommands from Task 1: `run_oracle_compare`, `run_helper_legacy_parity`, and `run_oom_contract_check`. `run_oracle_compare` must iterate required profiles (D-01/D-05/D-06), call `generate_profile_parity_report` with `include_unstable_source=false` by default (D-03), and persist per-profile reports. `run_helper_legacy_parity` must call oracle helper coverage checks so helper/legacy parity is a first-class gate (D-05). `run_oom_contract_check` must execute concrete no-partial-write/OOM regressions through cargo test invocations including `cargo test -p cintx-compat raw::tests::memory_limit_failure_keeps_output_slice_unchanged -- --exact` and `cargo test -p cintx-runtime workspace::tests::chunk_planner_reports_limit_exceeded_when_no_chunk_can_fit -- --exact` (D-13).
  </action>
  <acceptance_criteria>
    - `rg -n "run_oracle_compare|run_helper_legacy_parity|run_oom_contract_check" xtask/src/oracle_update.rs`
    - `rg -n "generate_profile_parity_report|verify_helper_surface_coverage|include_unstable_source" xtask/src/oracle_update.rs`
    - `rg -n "memory_limit_failure_keeps_output_slice_unchanged|chunk_planner_reports_limit_exceeded_when_no_chunk_can_fit" xtask/src/oracle_update.rs`
    - `rg -n "base|with-f12|with-4c1e|with-f12\\+with-4c1e" xtask/src/oracle_update.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles base,with-f12,with-4c1e,with-f12+with-4c1e --include-unstable-source false && cargo run --manifest-path xtask/Cargo.toml -- helper-legacy-parity --profile base && cargo run --manifest-path xtask/Cargo.toml -- oom-contract-check</automated>
  </verify>
  <done>All required VERI-02 gate commands are executable from xtask and fail-closed on parity, lock drift, helper/legacy, or OOM contract regressions.</done>
</task>

</tasks>

<verification>
Run xtask command help and execute each gate command at least once to prove the CLI contracts are real and CI-callable.
</verification>

<success_criteria>
Phase 4 has an executable gate command surface that enforces manifest lock authority, required profile matrix checks, helper/legacy parity, and OOM no-partial-write regressions.
</success_criteria>

<output>
After completion, create `.planning/phases/04-verification-release-automation/02-PLAN-SUMMARY.md`
</output>
