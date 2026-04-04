---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 05
type: execute
wave: 4
depends_on:
  - 03
files_modified:
  - xtask/src/main.rs
  - xtask/src/wgpu_capability_gate.rs
  - ci/gpu-bench.yml
  - .github/workflows/compat-governance-pr.yml
  - .github/workflows/compat-governance-release.yml
autonomous: true
requirements:
  - VERI-02
  - VERI-04
must_haves:
  truths:
    - "CI includes capability-aware wgpu gates with required vs advisory behavior."
    - "Gate artifacts record backend/adapter capability context and explicit skip reasons."
    - "Capability absence is explicit and auditable, never silent fallback."
  artifacts:
    - path: xtask/src/wgpu_capability_gate.rs
      provides: "Capability-aware gate command with typed artifact output and fallback path handling."
      min_lines: 220
    - path: .github/workflows/compat-governance-pr.yml
      provides: "Advisory PR wgpu capability gate wiring and artifact uploads."
      min_lines: 280
    - path: .github/workflows/compat-governance-release.yml
      provides: "Required release wgpu capability gate wiring with fail-closed policy."
      min_lines: 320
  key_links:
    - from: xtask/src/main.rs
      to: xtask/src/wgpu_capability_gate.rs
      via: "New `wgpu-capability-gate` command wired into xtask parser/dispatcher."
      pattern: "WgpuCapabilityGate|wgpu-capability-gate|run_wgpu_capability_gate"
    - from: .github/workflows/compat-governance-release.yml
      to: xtask/src/wgpu_capability_gate.rs
      via: "Release gate runs xtask command with `--require-adapter true` and uploads artifact."
      pattern: "wgpu_capability_required|--require-adapter true|cintx_phase_05_wgpu_capability_gate.json"
---

<objective>
Add capability-aware verification gates and artifacts for wgpu execution in xtask and CI workflows.
Purpose: Implement D-04, D-10, D-14, and D-16 at verification/release automation boundaries.
Output: New xtask gate command plus PR/release workflow wiring with required skip/evidence metadata.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md
@.planning/phases/04-verification-release-automation/02-PLAN-SUMMARY.md
@.planning/phases/04-verification-release-automation/03-PLAN-SUMMARY.md
@.planning/phases/04-verification-release-automation/06-PLAN-SUMMARY.md
@AGENTS.md
@xtask/src/main.rs
@xtask/src/oracle_update.rs
@ci/gpu-bench.yml
@.github/workflows/compat-governance-pr.yml
@.github/workflows/compat-governance-release.yml
<interfaces>
From `xtask/src/main.rs`:
```rust
enum Command {
    ManifestAudit { .. },
    BenchReport { .. },
    OracleCompare { .. },
    HelperLegacyParity { .. },
    OomContractCheck,
}
```

From `xtask/src/oracle_update.rs`:
```rust
fn write_json_with_fallback(required_path: &str, fallback_name: &str, value: &Value) -> Result<ArtifactWrite>
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add xtask `wgpu-capability-gate` command with artifactized skip metadata</name>
  <files>xtask/src/main.rs, xtask/src/wgpu_capability_gate.rs</files>
  <read_first>xtask/src/main.rs, xtask/src/oracle_update.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md, docs/rust_crate_test_guideline.md</read_first>
  <behavior>
    - Test 1: Command parser accepts `wgpu-capability-gate --profiles ... --require-adapter true|false`.
    - Test 2: Artifact JSON always includes `adapter_found`, `adapter_name`, `capability_fingerprint`, `status`, and `skip_reason`.
    - Test 3: Command exits non-zero only when `--require-adapter true` and capability is unavailable.
  </behavior>
  <action>
Create `xtask/src/wgpu_capability_gate.rs` with `run_wgpu_capability_gate(profiles: &[String], require_adapter: bool) -> anyhow::Result<()>`. Reuse phase-4 artifact conventions: required path `/mnt/data/cintx_phase_05_wgpu_capability_gate.json`, fallback path via `CINTX_ARTIFACT_DIR` to `/tmp/cintx_artifacts`. Populate artifact fields with backend/adapter metadata and explicit `status` values (`ok`, `capability-unavailable`, `failed`) per D-04/D-10. Wire parser/dispatcher in `xtask/src/main.rs` with command name `wgpu-capability-gate` and explicit boolean parsing. Include strict command-line validation so unknown profiles/flags fail closed.
  </action>
  <acceptance_criteria>
    - `rg -n "wgpu-capability-gate|WgpuCapabilityGate|run_wgpu_capability_gate" xtask/src/main.rs xtask/src/wgpu_capability_gate.rs`
    - `rg -n "cintx_phase_05_wgpu_capability_gate.json|CINTX_ARTIFACT_DIR|adapter_found|capability_fingerprint|skip_reason" xtask/src/wgpu_capability_gate.rs`
    - `rg -n "require_adapter|profiles|unsupported profile|unknown" xtask/src/main.rs xtask/src/wgpu_capability_gate.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo run --manifest-path xtask/Cargo.toml -- wgpu-capability-gate --profiles base --require-adapter false</automated>
  </verify>
  <done>xtask can produce capability-aware artifacts and explicit skip/fail behavior for wgpu availability checks.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Wire capability-aware PR/release gates and artifact uploads</name>
  <files>ci/gpu-bench.yml, .github/workflows/compat-governance-pr.yml, .github/workflows/compat-governance-release.yml</files>
  <read_first>ci/gpu-bench.yml, .github/workflows/compat-governance-pr.yml, .github/workflows/compat-governance-release.yml, xtask/src/main.rs, xtask/src/wgpu_capability_gate.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md</read_first>
  <behavior>
    - Test 1: PR workflow contains advisory `wgpu_capability_advisory` gate with `--require-adapter false` and `continue-on-error: true` (D-14).
    - Test 2: Release workflow contains required `wgpu_capability_required` gate with `--require-adapter true` and `continue-on-error: false` (D-14).
    - Test 3: Both workflows upload phase-5 capability artifact path and preserve skip metadata (D-10, D-16).
  </behavior>
  <action>
Update `ci/gpu-bench.yml` to run `cargo run --locked --manifest-path xtask/Cargo.toml -- wgpu-capability-gate --profiles "${CINTX_REQUIRED_PROFILES}" --require-adapter ${{ inputs.continue_on_error && 'false' || 'true' }}` before benchmark suites. In PR workflow add `wgpu_capability_advisory` job (advisory, `continue-on-error: true`) and artifact upload for `/mnt/data/cintx_phase_05_wgpu_capability_gate.json` + fallback path. In release workflow add `wgpu_capability_required` job (required, GPU runner labels, `continue-on-error: false`) and enforce artifact presence checks for phase-5 capability report. Keep existing phase-4 gates unchanged; append capability gate wiring only.
  </action>
  <acceptance_criteria>
    - `rg -n "wgpu_capability_advisory|--require-adapter false|continue-on-error:\\s*true|cintx_phase_05_wgpu_capability_gate.json" .github/workflows/compat-governance-pr.yml`
    - `rg -n "wgpu_capability_required|--require-adapter true|continue-on-error:\\s*false|runs-on:\\s*\\[self-hosted, linux, x64, gpu\\]" .github/workflows/compat-governance-release.yml`
    - `rg -n "wgpu-capability-gate|cintx_phase_05_wgpu_capability_gate.json|upload-artifact" ci/gpu-bench.yml .github/workflows/compat-governance-pr.yml .github/workflows/compat-governance-release.yml`
  </acceptance_criteria>
  <verify>
    <automated>rg -n "wgpu_capability_advisory|wgpu_capability_required|wgpu-capability-gate|cintx_phase_05_wgpu_capability_gate.json|continue-on-error" ci/gpu-bench.yml .github/workflows/compat-governance-pr.yml .github/workflows/compat-governance-release.yml</automated>
  </verify>
  <done>Capability-aware wgpu verification gates are wired in CI with explicit advisory/required semantics and auditable artifacts.</done>
</task>

</tasks>

<verification>
Run xtask capability gate command locally, then inspect workflow/template files for required/advisory gate wiring and artifact contract markers.
</verification>

<success_criteria>
Phase 5 verification artifacts and CI gates clearly encode capability availability, prevent silent fallback behavior, and preserve reproducible backend context in uploaded evidence.
</success_criteria>

<output>
After completion, create `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-PLAN-SUMMARY.md`
</output>
