---
phase: 04-verification-release-automation
plan: 03
type: execute
wave: 3
depends_on:
  - 02
files_modified:
  - ci/oracle-compare.yml
  - ci/feature-matrix.yml
  - .github/workflows/compat-governance-pr.yml
autonomous: true
requirements:
  - VERI-02
must_haves:
  truths:
    - "PR verification runs merge-blocking manifest drift, oracle parity, helper/legacy parity, and OOM-contract gates."
    - "Required profile matrix coverage includes `base`, `with-f12`, `with-4c1e`, and `with-f12+with-4c1e`."
    - "Workflow jobs collect full matrix evidence instead of canceling on first failing profile."
  artifacts:
    - path: ci/oracle-compare.yml
      provides: "Canonical oracle/profile gate command template with non-fail-fast profile matrix and artifact upload steps."
      min_lines: 70
    - path: ci/feature-matrix.yml
      provides: "Canonical manifest-drift and OOM gate command template for required profile matrix."
      min_lines: 70
    - path: .github/workflows/compat-governance-pr.yml
      provides: "Executable PR workflow wiring the four required merge-blocking gates."
      min_lines: 110
  key_links:
    - from: .github/workflows/compat-governance-pr.yml
      to: xtask/src/main.rs
      via: "PR jobs execute xtask gate commands directly."
      pattern: "cargo run --manifest-path xtask/Cargo.toml -- (manifest-audit|oracle-compare|helper-legacy-parity|oom-contract-check)"
    - from: ci/oracle-compare.yml
      to: crates/cintx-oracle/src/compare.rs
      via: "Template command references profile-aware oracle compare gate."
      pattern: "oracle-compare|include-unstable-source false"
    - from: ci/feature-matrix.yml
      to: xtask/src/manifest_audit.rs
      via: "Template command enforces compiled manifest lock drift and profile scope checks."
      pattern: "manifest-audit|with-f12\\+with-4c1e"
---

<objective>
Wire merge-blocking PR CI around the new xtask gate surface and required feature-profile matrix.
Purpose: Complete the CI-policy half of VERI-02 under D-05, D-06, and D-08 without policy bypass paths.
Output: Non-stub CI templates and a PR workflow that executes the four required verification gates.
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
@ci/oracle-compare.yml
@ci/feature-matrix.yml
@.github/workflows/compat-governance-pr.yml
@xtask/src/main.rs
@xtask/src/manifest_audit.rs
@xtask/src/oracle_update.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace CI template stubs with concrete Phase 4 gate templates</name>
  <files>ci/oracle-compare.yml, ci/feature-matrix.yml</files>
  <read_first>ci/oracle-compare.yml, ci/feature-matrix.yml, xtask/src/main.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md, .planning/phases/04-verification-release-automation/04-RESEARCH.md</read_first>
  <action>
Replace both stub files with canonical command templates used by governance workflows. `ci/oracle-compare.yml` must define matrix profiles exactly as `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`, set `fail-fast: false`, run `xtask oracle-compare --include-unstable-source false` and `xtask helper-legacy-parity`, and upload mismatch artifacts. `ci/feature-matrix.yml` must run `xtask manifest-audit --check-lock` and `xtask oom-contract-check` with the same profile matrix and `fail-fast: false`. Keep command names and profile strings identical to avoid drift between templates and executable workflows.
  </action>
  <acceptance_criteria>
    - `rg -n "fail-fast: false|base|with-f12|with-4c1e|with-f12\\+with-4c1e" ci/oracle-compare.yml ci/feature-matrix.yml`
    - `rg -n "oracle-compare --profiles|include-unstable-source false|helper-legacy-parity" ci/oracle-compare.yml`
    - `rg -n "manifest-audit --profiles|--check-lock|oom-contract-check" ci/feature-matrix.yml`
    - `rg -n "upload-artifact|/mnt/data|CINTX_ARTIFACT_DIR" ci/oracle-compare.yml`
  </acceptance_criteria>
  <verify>
    <automated>rg -n "fail-fast: false|with-f12\\+with-4c1e|manifest-audit|oracle-compare|helper-legacy-parity|oom-contract-check" ci/oracle-compare.yml ci/feature-matrix.yml</automated>
  </verify>
  <done>CI templates are no longer placeholders and contain concrete profile-matrix gate commands aligned with Phase 4 policy.</done>
</task>

<task type="auto">
  <name>Task 2: Rewire PR governance workflow to enforce required merge-blocking gates</name>
  <files>.github/workflows/compat-governance-pr.yml</files>
  <read_first>.github/workflows/compat-governance-pr.yml, ci/oracle-compare.yml, ci/feature-matrix.yml, xtask/src/main.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md</read_first>
  <action>
Replace obsolete Phase 3 test/bin invocations with Phase 4 xtask gates per D-05 and D-08. Define four explicit required jobs in the PR workflow: `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, and `oom_contract_gate`. Each job must run on `ubuntu-latest`, install pinned Rust toolchain from `rust-toolchain.toml`, and invoke `cargo run --manifest-path xtask/Cargo.toml -- ...` with the required profile set `base,with-f12,with-4c1e,with-f12+with-4c1e` (D-06). Remove any references to missing `phase3_*` tests or `cargo run --bin manifest_audit`. Do not use `continue-on-error` on required jobs.
  </action>
  <acceptance_criteria>
    - `rg -n "manifest_drift_gate|oracle_parity_gate|helper_legacy_parity_gate|oom_contract_gate" .github/workflows/compat-governance-pr.yml`
    - `rg -n "cargo run --manifest-path xtask/Cargo.toml -- manifest-audit|oracle-compare|helper-legacy-parity|oom-contract-check" .github/workflows/compat-governance-pr.yml`
    - `rg -n "base,with-f12,with-4c1e,with-f12\\+with-4c1e" .github/workflows/compat-governance-pr.yml`
    - `rg -n "phase3_|cargo run --bin manifest_audit" .github/workflows/compat-governance-pr.yml` returns no matches
    - `rg -n "continue-on-error:\\s*true" .github/workflows/compat-governance-pr.yml` returns no matches
  </acceptance_criteria>
  <verify>
    <automated>rg -n "manifest_drift_gate|oracle_parity_gate|helper_legacy_parity_gate|oom_contract_gate|with-f12\\+with-4c1e" .github/workflows/compat-governance-pr.yml</automated>
  </verify>
  <done>PR governance workflow runs only real Phase 4 gate commands and fails hard when required checks fail.</done>
</task>

</tasks>

<verification>
Inspect workflow files with regex checks to confirm profile matrix coverage, non-fail-fast evidence collection, and required gate command wiring.
</verification>

<success_criteria>
VERI-02 merge-blocking policy is encoded in PR automation: manifest drift, oracle parity, helper/legacy parity, and OOM checks are all runnable required gates.
</success_criteria>

<output>
After completion, create `.planning/phases/04-verification-release-automation/03-PLAN-SUMMARY.md`
</output>
