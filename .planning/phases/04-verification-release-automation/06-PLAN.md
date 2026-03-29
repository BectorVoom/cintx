---
phase: 04-verification-release-automation
plan: 06
type: execute
wave: 6
depends_on:
  - 05
files_modified:
  - .github/workflows/compat-governance-release.yml
  - ci/gpu-bench.yml
autonomous: true
requirements:
  - VERI-02
  - VERI-03
  - VERI-04
gap_closure: true
must_haves:
  truths:
    - "`gpu_bench_required` runs on an explicit GPU-capable runner contract and remains blocking."
    - "Release GPU bench execution fails when neither required-path nor fallback benchmark/diagnostics artifacts are present."
    - "Release artifact uploads include bench rows, bench report, and runtime diagnostics from required and fallback locations."
  artifacts:
    - path: .github/workflows/compat-governance-release.yml
      provides: "Required release/schedule GPU gate bound to GPU-capable runner labels with artifact contract checks."
      min_lines: 180
    - path: ci/gpu-bench.yml
      provides: "Template-level GPU bench runner and artifact validation contract aligned with the required release gate."
      min_lines: 90
  key_links:
    - from: .github/workflows/compat-governance-release.yml
      to: ci/gpu-bench.yml
      via: "Runner labels and required bench-report command remain aligned between required workflow and template policy."
      pattern: "runs-on:\\s*\\[self-hosted,\\s*linux,\\s*x64,\\s*gpu\\]|bench-report --thresholds ci/benchmark-thresholds.json --mode enforce"
    - from: .github/workflows/compat-governance-release.yml
      to: /mnt/data/cintx_phase_04_runtime_diagnostics.json
      via: "Artifact validation step asserts required/fallback report files exist before artifact upload."
      pattern: "Validate bench artifact contract|/mnt/data/cintx_phase_04_bench_report.json|/tmp/cintx_artifacts/cintx_phase_04_runtime_diagnostics.json"
---

<objective>
Close the diagnosed Phase 04 UAT Test 2 blocker around `gpu_bench_required` runner semantics and artifact contract validation.
Purpose: Enforce the Phase 4 release/scheduled GPU policy (D-07, D-08, D-12) in executable workflow configuration.
Output: Updated required release GPU workflow and GPU bench template with explicit GPU runner contract and artifact validation checks.
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
@.planning/phases/04-verification-release-automation/04-HUMAN-UAT.md
@.planning/phases/04-verification-release-automation/04-VERIFICATION.md
@.planning/debug/phase04-test2-gpu-bench-block.md
@.planning/phases/04-verification-release-automation/04-PLAN-SUMMARY.md
@AGENTS.md
@.github/workflows/compat-governance-release.yml
@ci/gpu-bench.yml
<interfaces>
From `.github/workflows/compat-governance-release.yml`:
```yaml
gpu_bench_required:
  continue-on-error: false
  env:
    CINTX_ARTIFACT_DIR: /mnt/data/cintx/release-gpu-bench/${{ github.run_id }}
  steps:
    - cargo bench -p cintx --bench micro_families --bench macro_molecules --bench crossover_cpu_gpu
    - cargo run --manifest-path xtask/Cargo.toml -- bench-report --thresholds ci/benchmark-thresholds.json --mode enforce
```

From `ci/gpu-bench.yml`:
```yaml
gpu_bench_template:
  runs-on: ubuntu-latest
  continue-on-error: ${{ inputs.continue_on_error }}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Bind required GPU bench jobs to a GPU-capable runner contract</name>
  <files>.github/workflows/compat-governance-release.yml, ci/gpu-bench.yml</files>
  <read_first>.github/workflows/compat-governance-release.yml, ci/gpu-bench.yml, .planning/phases/04-verification-release-automation/04-HUMAN-UAT.md, .planning/debug/phase04-test2-gpu-bench-block.md</read_first>
  <action>
Update `gpu_bench_required` and `gpu_bench_template` to use an explicit GPU runner label set (`[self-hosted, linux, x64, gpu]`) instead of `ubuntu-latest`, per D-07 and the diagnosed Test 2 root cause. Keep `continue-on-error: false` on `gpu_bench_required` (D-08). Do not alter non-GPU release jobs in this gap closure.
  </action>
  <verify>
    <automated>rg -n "gpu_bench_required|gpu_bench_template|runs-on:\\s*\\[self-hosted,\\s*linux,\\s*x64,\\s*gpu\\]|continue-on-error:\\s*false" .github/workflows/compat-governance-release.yml ci/gpu-bench.yml</automated>
  </verify>
  <done>`gpu_bench_required` and template GPU jobs no longer target `ubuntu-latest`; required release GPU gate remains blocking.</done>
</task>

<task type="auto">
  <name>Task 2: Enforce required/fallback artifact contract before upload</name>
  <files>.github/workflows/compat-governance-release.yml, ci/gpu-bench.yml</files>
  <read_first>.github/workflows/compat-governance-release.yml, ci/gpu-bench.yml, xtask/src/bench_report.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md</read_first>
  <action>
Add a `Validate bench artifact contract` shell step after `bench-report` in both files. The step must fail if neither required nor fallback files exist for: `cintx_phase_04_bench_report.json` and `cintx_phase_04_runtime_diagnostics.json` (per D-12). Keep upload-artifact paths covering required (`/mnt/data/...`) and fallback (`/tmp/cintx_artifacts/...`) outputs so failed runs still preserve diagnostics evidence.
  </action>
  <verify>
    <automated>rg -n "Validate bench artifact contract|cintx_phase_04_bench_report.json|cintx_phase_04_runtime_diagnostics.json|/tmp/cintx_artifacts|upload-artifact@v4" .github/workflows/compat-governance-release.yml ci/gpu-bench.yml</automated>
  </verify>
  <done>Release GPU workflow and template fail fast on missing bench/diagnostics artifacts while preserving required and fallback artifact upload coverage.</done>
</task>

</tasks>

<verification>
Verify runner labels no longer use `ubuntu-latest` for required/template GPU jobs, then verify artifact contract checks and upload paths are present for required and fallback report outputs.
</verification>

<success_criteria>
Phase 04 UAT Test 2 gap is closed at plan level: the required release GPU gate is explicitly GPU-bound and enforces benchmark/diagnostics artifact presence before upload.
</success_criteria>

<output>
After completion, create `.planning/phases/04-verification-release-automation/06-PLAN-SUMMARY.md`
</output>
