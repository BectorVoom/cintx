---
phase: 04-verification-release-automation
plan: 04
type: execute
wave: 4
depends_on:
  - 02
  - 03
files_modified:
  - Cargo.toml
  - benches/micro_families.rs
  - benches/macro_molecules.rs
  - benches/crossover_cpu_gpu.rs
  - xtask/src/bench_report.rs
  - ci/benchmark-thresholds.json
  - ci/gpu-bench.yml
  - .github/workflows/compat-governance-pr.yml
  - .github/workflows/compat-governance-release.yml
autonomous: true
requirements:
  - VERI-03
  - VERI-04
must_haves:
  truths:
    - "Benchmark automation includes micro-family, macro-molecule, and CPU-GPU crossover suites and stores reproducible baseline evidence."
    - "Benchmark gates fail only when configured thresholds are exceeded, not on any slowdown."
    - "Runtime diagnostics artifacts include planner/chunk/fallback/transfer/OOM fields and persist required `/mnt/data` metadata in CI outputs."
    - "GPU consistency and benchmark jobs are advisory on PRs and required on release/scheduled verification flows."
  artifacts:
    - path: benches/micro_families.rs
      provides: "Criterion microbench suite for family/representation throughput and memory trend points."
      min_lines: 120
    - path: benches/macro_molecules.rs
      provides: "Criterion macro workload suite for representative molecule scenarios."
      min_lines: 120
    - path: benches/crossover_cpu_gpu.rs
      provides: "CPU-GPU crossover benchmark suite capturing transfer break-even and dispatch trend points."
      min_lines: 120
    - path: xtask/src/bench_report.rs
      provides: "Threshold-aware benchmark and diagnostics report generator."
      min_lines: 240
    - path: ci/benchmark-thresholds.json
      provides: "Version-controlled numeric regression thresholds for micro/macro/crossover suites."
      min_lines: 20
    - path: ci/gpu-bench.yml
      provides: "GPU consistency and benchmark workflow policy template with artifact upload steps."
      min_lines: 80
  key_links:
    - from: xtask/src/bench_report.rs
      to: crates/cintx-runtime/src/metrics.rs
      via: "Diagnostics artifacts include `chunk_count`, `fallback_reason`, `transfer_bytes`, and `not0` fields from runtime metrics contracts."
      pattern: "chunk_count|fallback_reason|transfer_bytes|not0"
    - from: .github/workflows/compat-governance-pr.yml
      to: ci/gpu-bench.yml
      via: "PR workflow runs GPU/bench job in advisory mode using continue-on-error."
      pattern: "gpu_bench_advisory|continue-on-error:\\s*true"
    - from: .github/workflows/compat-governance-release.yml
      to: ci/gpu-bench.yml
      via: "Release/scheduled workflow runs GPU/bench verification as required gates."
      pattern: "gpu_bench_required|continue-on-error:\\s*false"
---

<objective>
Add benchmark regression automation and runtime diagnostics artifact pipelines, then wire GPU policy across PR and release flows.
Purpose: Complete VERI-03 and VERI-04 while implementing D-07, D-09, D-10, D-11, and D-12 with concrete gate behavior.
Output: Bench suites, threshold configuration, bench/diagnostics reporting command, and GPU/benchmark workflow wiring for advisory vs required contexts.
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
@Cargo.toml
@benches/micro_families.rs
@benches/macro_molecules.rs
@benches/crossover_cpu_gpu.rs
@xtask/src/bench_report.rs
@ci/gpu-bench.yml
@.github/workflows/compat-governance-pr.yml
@.github/workflows/compat-governance-release.yml
@crates/cintx-runtime/src/metrics.rs
@crates/cintx-runtime/src/planner.rs
<interfaces>
From `crates/cintx-runtime/src/metrics.rs`:
```rust
pub struct ExecutionStats {
    pub chunk_count: usize,
    pub transfer_bytes: usize,
    pub not0: i32,
    pub fallback_reason: Option<&'static str>,
    // ...
}
```

From `xtask/src/main.rs` (Plan 02 command contract):
```rust
bench-report
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement Phase 4 benchmark suites and Cargo bench wiring</name>
  <files>Cargo.toml, benches/micro_families.rs, benches/macro_molecules.rs, benches/crossover_cpu_gpu.rs</files>
  <read_first>Cargo.toml, benches/micro_families.rs, benches/macro_molecules.rs, benches/crossover_cpu_gpu.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md, docs/design/cintx_detailed_design.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Wire Criterion benchmarks with concrete suite coverage per D-10: micro (`micro_families`), macro (`macro_molecules`), and crossover (`crossover_cpu_gpu`). Add `criterion = "0.8.2"` under `[dev-dependencies]` and configure explicit bench targets with `harness = false`. Ensure each bench writes structured summary rows with suite IDs, profile labels, throughput metrics, and memory/workspace counters that `xtask bench-report` can parse. Keep benchmark execution out of merge-blocking PR gates (D-09); these suites are for nightly/release and advisory PR runs only.
  </action>
  <acceptance_criteria>
    - `rg -n "\\[dev-dependencies\\]|criterion\\s*=\\s*\"0.8.2\"|\\[\\[bench\\]\\]|micro_families|macro_molecules|crossover_cpu_gpu|harness\\s*=\\s*false" Cargo.toml`
    - `rg -n "criterion_group!|criterion_main!|micro_families|macro_molecules|crossover_cpu_gpu" benches/micro_families.rs benches/macro_molecules.rs benches/crossover_cpu_gpu.rs`
    - `rg -n "profile|throughput|workspace|transfer|not0" benches/micro_families.rs benches/macro_molecules.rs benches/crossover_cpu_gpu.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo bench --no-run --bench micro_families --bench macro_molecules --bench crossover_cpu_gpu</automated>
  </verify>
  <done>All three Phase 4 benchmark suites compile under Criterion and are ready for nightly/release and advisory PR execution.</done>
</task>

<task type="auto">
  <name>Task 2: Implement threshold-based bench and diagnostics report generation</name>
  <files>xtask/src/bench_report.rs, ci/benchmark-thresholds.json</files>
  <read_first>xtask/src/bench_report.rs, xtask/src/main.rs, ci/benchmark-thresholds.json, crates/cintx-runtime/src/metrics.rs, crates/cintx-runtime/src/planner.rs, .planning/phases/04-verification-release-automation/04-CONTEXT.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Replace the bench-report stub with a concrete report engine that reads Criterion results and emits benchmark + diagnostics JSON artifacts. Add version-controlled thresholds in `ci/benchmark-thresholds.json` with explicit numeric keys: `micro_families.throughput_regression_pct`, `macro_molecules.throughput_regression_pct`, `crossover_cpu_gpu.crossover_shift_pct`, and memory thresholds. Implement fail policy per D-11: exit non-zero only when measured regressions exceed threshold values, not on any slowdown. Persist artifacts to `/mnt/data/cintx_phase_04_bench_report.json` and `/mnt/data/cintx_phase_04_runtime_diagnostics.json` with fallback metadata mirroring oracle artifact behavior (D-12), and include runtime diagnostics fields `chunk_count`, `fallback_reason`, `transfer_bytes`, and `not0`.
  </action>
  <acceptance_criteria>
    - `rg -n "throughput_regression_pct|crossover_shift_pct|memory_regression_pct" ci/benchmark-thresholds.json`
    - `rg -n "cintx_phase_04_bench_report.json|cintx_phase_04_runtime_diagnostics.json|/mnt/data|CINTX_ARTIFACT_DIR" xtask/src/bench_report.rs`
    - `rg -n "chunk_count|fallback_reason|transfer_bytes|not0" xtask/src/bench_report.rs`
    - `rg -n "threshold|exceed|regression|exit\\(1\\)|anyhow::bail!" xtask/src/bench_report.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo run --manifest-path xtask/Cargo.toml -- bench-report --thresholds ci/benchmark-thresholds.json --mode calibration</automated>
  </verify>
  <done>Bench and diagnostics reporting is threshold-aware, artifactized, and fail-closed only when configured regression limits are exceeded.</done>
</task>

<task type="auto">
  <name>Task 3: Wire GPU benchmark policy across PR-advisory and release-required workflows</name>
  <files>ci/gpu-bench.yml, .github/workflows/compat-governance-pr.yml, .github/workflows/compat-governance-release.yml</files>
  <read_first>ci/gpu-bench.yml, .github/workflows/compat-governance-pr.yml, .github/workflows/compat-governance-release.yml, ci/oracle-compare.yml, ci/feature-matrix.yml, .planning/phases/04-verification-release-automation/04-CONTEXT.md, .planning/phases/04-verification-release-automation/04-RESEARCH.md</read_first>
  <action>
Replace `ci/gpu-bench.yml` stub with concrete GPU consistency + benchmark commands and artifact upload instructions. Update PR workflow to add `gpu_bench_advisory` job with `continue-on-error: true` (D-07) and ensure it runs benchmark + diagnostics commands without blocking required merge gates (D-09). Update release workflow so GPU/benchmark jobs are required (`continue-on-error: false`) on release/tag/scheduled/merge-queue paths (D-07). Ensure both workflows call `xtask bench-report --thresholds ci/benchmark-thresholds.json` and upload benchmark/diagnostics artifacts from required `/mnt/data` paths (with fallback metadata) per D-12.
  </action>
  <acceptance_criteria>
    - `rg -n "gpu_bench_advisory|continue-on-error:\\s*true|bench-report --thresholds ci/benchmark-thresholds.json" .github/workflows/compat-governance-pr.yml`
    - `rg -n "gpu_bench_required|continue-on-error:\\s*false|bench-report --thresholds ci/benchmark-thresholds.json" .github/workflows/compat-governance-release.yml`
    - `rg -n "schedule:|workflow_dispatch:|release:|tags:" .github/workflows/compat-governance-release.yml`
    - `rg -n "upload-artifact|cintx_phase_04_bench_report.json|cintx_phase_04_runtime_diagnostics.json|/mnt/data" ci/gpu-bench.yml .github/workflows/compat-governance-pr.yml .github/workflows/compat-governance-release.yml`
  </acceptance_criteria>
  <verify>
    <automated>rg -n "gpu_bench_advisory|gpu_bench_required|continue-on-error|bench-report --thresholds ci/benchmark-thresholds.json|cintx_phase_04_runtime_diagnostics.json" ci/gpu-bench.yml .github/workflows/compat-governance-pr.yml .github/workflows/compat-governance-release.yml</automated>
  </verify>
  <done>GPU and benchmark policy is correctly split into PR-advisory and release/scheduled-required flows with persisted diagnostics artifacts.</done>
</task>

</tasks>

<verification>
Compile benchmark targets, run bench-report calibration mode, and assert workflow files encode advisory-vs-required GPU semantics with artifactized diagnostics.
</verification>

<success_criteria>
Phase 4 benchmark and diagnostics automation exists end-to-end: suites run, thresholds gate regressions, runtime telemetry is artifactized, and workflow policy matches advisory PR / required release behavior.
</success_criteria>

<output>
After completion, create `.planning/phases/04-verification-release-automation/04-PLAN-SUMMARY.md`
</output>
