# Phase 4: Verification & Release Automation - Research

**Researched:** 2026-03-28
**Domain:** Oracle verification closure, CI release gates, benchmark regression tracking, and runtime diagnostics artifacts
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Oracle Comparison Policy
- **D-01:** Merge-blocking oracle coverage includes stable APIs and optional-profile APIs when those profiles are enabled in matrix jobs.
- **D-02:** Family tolerances stay as an explicit per-family table in code and must be changed only through deliberate reviewed updates.
- **D-03:** Optional-family oracle checks are required; unstable-source oracle checks remain extended/nightly coverage rather than default merge blockers.
- **D-04:** Oracle jobs should emit complete mismatch reports across the full fixture set before failing (no first-mismatch fail-fast mode).

### CI Gate and Matrix Policy
- **D-05:** Required merge-blocking PR gates include manifest drift checks, oracle parity checks, helper/legacy parity checks, and OOM-contract checks.
- **D-06:** Required feature-matrix verification covers all approved profiles: `base`, `with-f12`, `with-4c1e`, and `with-f12+with-4c1e`.
- **D-07:** GPU consistency/benchmark jobs are advisory on PRs but required in scheduled/merge-queue verification flows.
- **D-08:** Required verification gate failures block merges (normal infra reruns allowed, but no policy-level bypass).

### Benchmark and Diagnostics Policy
- **D-09:** Benchmark automation runs on nightly and release-oriented workflows, not on every merge-blocking PR.
- **D-10:** Phase 4 baseline suites include micro family benchmarks, macro molecule benchmarks, and CPU-GPU crossover tracking.
- **D-11:** After baselines are established, benchmark gates fail only when regressions exceed defined thresholds (not report-only and not any-slowdown-fails).
- **D-12:** Verification workflows persist structured trace+metrics diagnostics (planner/chunk/fallback/transfer/OOM) with artifactized outputs honoring required `/mnt/data` paths.

### Carried Forward from Prior Phases
- **D-13:** Verification remains fail-closed: unsupported envelopes return explicit `UnsupportedApi` and no partial-write behavior is allowed.
- **D-14:** The compiled manifest lock remains the API source of truth; release automation must gate against lock/coverage drift.

### Claude's Discretion
- Exact numeric benchmark threshold values and warmup/stabilization policy before hard enforcement.
- Concrete CI workflow decomposition (job fan-out, retry budget, and naming), as long as D-05 through D-08 hold.
- Final artifact JSON schema field names and xtask command UX for report generation.

### Deferred Ideas (OUT OF SCOPE)
None - discussion stayed within Phase 4 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VERI-01 | Maintainer can compare stable and enabled optional APIs against vendored upstream libcint through oracle tests with family-appropriate tolerances. | Manifest-derived profile matrix, explicit tolerance table governance, and non-fail-fast mismatch reporting pattern for oracle runs. |
| VERI-02 | CI can block manifest drift, helper/legacy parity regressions, CubeCL consistency failures, and OOM contract violations across the support matrix. | Required PR gates + scheduled/release gates split, matrix profile coverage, branch-protection required checks mapping, and xtask gate commands. |
| VERI-03 | Maintainer can benchmark representative workloads and track throughput, memory, and CPU-GPU crossover regressions over time. | Criterion suites (`micro`, `macro`, `crossover`) with baseline retention, threshold policy, artifactized trend reports, and nightly/release execution policy. |
| VERI-04 | Maintainer can inspect planner, chunking, transfer, fallback, and OOM behavior through structured tracing and diagnostics. | `tracing` + `tracing-subscriber` JSON diagnostics pipeline wired to runtime metrics fields (`chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`) and `/mnt/data` artifacts. |
</phase_requirements>

## Summary

Phase 4 should be planned as a migration from placeholder and stale governance plumbing to a single manifest-driven verification system. Current landing zones (`ci/*.yml`, `xtask/src/*`, `benches/*.rs`) are stubs, and existing GitHub governance workflows still call missing test/bin targets (`phase3_*`, `manifest_audit`) that do not exist in this branch state. This means the first work items must establish real gate executables and wire them into required checks before expanding coverage.

The core technical foundation already exists in local crates: `cintx-oracle` has per-family tolerance logic, manifest-derived fixture generation, helper/transform/optimizer parity checks, and required-path artifact writers; `cintx-runtime` already emits structured planner/chunk/fallback/transfer metrics fields and enforces fail-closed memory/OOM contracts. Phase 4 should reuse these primitives rather than introducing parallel verification code paths.

External standards confirm the intended execution model: GitHub Actions matrix/fail-fast behavior and branch-protection required checks, nextest machine-readable reports, cargo-hack feature-matrix checks, and criterion baseline comparison APIs. The planner should therefore generate tasks in this order: gate command surface (xtask), PR-required CI matrix, scheduled/release advisory+required jobs, then benchmark/diagnostics thresholding.

**Primary recommendation:** Implement Phase 4 as a manifest-first verification pipeline with explicit gate commands and required-check wiring, then layer benchmark thresholds and diagnostics artifacts on top.

## Standard Stack

### Core
| Library/Tool | Version | Purpose | Why Standard |
|--------------|---------|---------|--------------|
| `cintx-oracle` (workspace crate) | `0.1.0` | Oracle parity engine, tolerance table, fixture matrix, artifact emission | Existing code already enforces manifest-derived parity and helper/legacy checks; extend it rather than replacing it. |
| `cintx-ops` manifest lock + resolver | schema `1`, workspace `0.1.0` | Source of truth for profile/family/symbol coverage and drift checks | Design sections 3.2/3.3/14.1 require lock-based gating across `{base, with-f12, with-4c1e, with-f12+with-4c1e}`. |
| `tracing` | `0.1.44` | Runtime span/field instrumentation for planner/chunk/transfer/OOM | Already used in runtime; required for VERI-04 evidence outputs. |
| `cargo-nextest` | `0.9.132` (cargo index) | Fast, profile-aware test execution and report output in CI | Better CI ergonomics and machine-readable outputs for gate jobs. |
| `cargo-hack` | `0.6.44` (cargo index) | Exhaustive feature/profile matrix checking | Canonical tool for matrix validation without hand-maintaining command permutations. |
| `criterion` | `0.8.2` | Baseline-aware benchmark harness for micro/macro/crossover suites | Supports save/retain baseline and statistical regression comparison needed for VERI-03. |

### Supporting
| Library/Tool | Version | Purpose | When to Use |
|--------------|---------|---------|-------------|
| `tracing-subscriber` | `0.3.23` (cargo index) | JSON/log layer for diagnostics artifact streams | Use in verification binaries/xtask runners that serialize structured trace evidence. |
| `serde_json` | `1.0.149` (cargo index) | Structured gate and benchmark report schemas | Use for deterministic report artifacts consumed by CI/release gates. |
| GitHub Actions matrix + required checks | Workflow syntax (current docs) | Merge-blocking gate orchestration and profile fan-out | Required for D-05..D-08 policy enforcement. |
| `actions/upload-artifact` | v4 line | Persist mismatch reports, traces, benchmark snapshots | Use for PR/scheduled/release artifact retention and manual diagnostics review. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cargo-nextest` for CI gates | plain `cargo test` only | Simpler but weaker machine-readable reporting and slower large matrix runs. |
| `cargo-hack` profile fan-out | hand-written per-profile cargo commands | More drift-prone and duplicates feature matrix logic in YAML. |
| Criterion baselines | ad-hoc timer scripts | Misses statistical significance and baseline retention APIs. |
| Structured trace artifacts | freeform text logs only | Hard to diff and weak for manual regression triage. |

**Installation:**
```bash
cargo install --locked cargo-nextest cargo-hack
cargo add --dev criterion
cargo add tracing-subscriber serde_json
```

**Version verification (2026-03-28):**
- Local cargo index checks:
  - `for c in anyhow thiserror tracing criterion cargo-nextest cargo-hack cc bindgen tracing-subscriber serde_json; do cargo search "$c" --limit 1; done`
  - observed latest: `anyhow 1.0.102`, `thiserror 2.0.18`, `tracing 0.1.44`, `criterion 0.8.2`, `cargo-nextest 0.9.132`, `cargo-hack 0.6.44`, `cc 1.2.58`, `bindgen 0.72.1`, `tracing-subscriber 0.3.23`, `serde_json 1.0.149`.
- Publish-date verification from docs.rs:
  - `anyhow 1.0.102` built/published 2026-02-20 (docs.rs builds page).
  - `thiserror 2.0.18` built/published 2026-01-18 (docs.rs builds page).
  - `tracing 0.1.44` released 2025-12-18 (docs.rs versions list).
  - `criterion 0.8.2` released 2026-02-04 (docs.rs versions list).
  - `bindgen 0.72.1` released 2025-08-31 (docs.rs versions list).
- Note: Cargo index and docs mirrors are slightly out of sync for some tools (`cargo-nextest`, `cargo-hack`, `tracing-subscriber`), so planner should pin exact versions at implementation time with a final `cargo search` check.

## Architecture Patterns

### Recommended Project Structure
```text
ci/
├── oracle-compare.yml        # Required PR gate + scheduled/release oracle matrix
├── feature-matrix.yml        # Required profile matrix + drift/parity/OOM gates
└── gpu-bench.yml             # Advisory on PR, required on schedule/merge queue

xtask/src/
├── main.rs                   # command router for all gate/bench/report commands
├── manifest_audit.rs         # lock drift + coverage checks
├── oracle_update.rs          # oracle parity execution/report generation
└── bench_report.rs           # benchmark baseline compare + threshold verdicts

benches/
├── micro_families.rs         # family/rep micro throughput + memory metrics
├── macro_molecules.rs        # end-to-end representative workloads
└── crossover_cpu_gpu.rs      # CPU-GPU crossover and transfer-break-even tracking
```

### Pattern 1: Manifest-Driven Oracle Matrix (All Required Profiles)
**What:** Build oracle targets from the compiled manifest lock and run them per approved profile.
**When to use:** Every merge-blocking oracle gate and scheduled full verification.
**Example:**
```rust
// Source: crates/cintx-oracle/src/fixtures.rs + crates/cintx-ops/generated/compiled_manifest.lock.json
let profiles = ["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"];
for profile in profiles {
    let fixtures = build_manifest_profile_matrix(profile)?;
    run_oracle_comparison(fixtures, tolerance_table_for_profile(profile))?;
}
```

### Pattern 2: Full Mismatch Collection, Then Fail
**What:** Collect all fixture mismatches and emit one complete report artifact before returning failure.
**When to use:** Oracle compare commands implementing D-04 (no first-mismatch fail-fast).
**Example:**
```rust
// Source intent: 04-CONTEXT.md D-04; current compare.rs is fail-fast and needs this refactor
let mut mismatches = Vec::new();
for fixture in fixtures {
    let diff = compare_fixture(&fixture)?;
    if !diff.within_tolerance {
        mismatches.push(diff);
    }
}
write_mismatch_report(&mismatches)?;
if !mismatches.is_empty() {
    anyhow::bail!("oracle parity failed with {} mismatches", mismatches.len());
}
```

### Pattern 3: CI Gate Split (Required vs Advisory)
**What:** Required checks run in PR jobs; expensive GPU/bench suites are advisory on PR but required on scheduled/release flows.
**When to use:** D-05..D-08 enforcement.
**Example:**
```yaml
# Source: https://docs.github.com/actions/automating-your-workflow-with-github-actions/workflow-syntax-for-github-actions
strategy:
  fail-fast: false
  matrix:
    profile: [base, with-f12, with-4c1e, with-f12+with-4c1e]
steps:
  - run: cargo hack test --workspace --feature-powerset --depth 1 --locked
```

### Pattern 4: Structured Diagnostics Artifacts
**What:** Emit JSON-formatted tracing plus run metrics for planner/chunk/fallback/transfer/OOM inspection.
**When to use:** All verification workflows that need VERI-04 forensic evidence.
**Example:**
```rust
// Source: crates/cintx-runtime/src/planner.rs metrics fields + https://docs.rs/tracing-subscriber
let subscriber = tracing_subscriber::fmt().json().flatten_event(true).finish();
tracing::subscriber::with_default(subscriber, || run_verification_pass())?;
```

### Anti-Patterns to Avoid
- **Keeping legacy Phase 3 workflow targets:** Current workflows call missing targets (`phase3_*`, `manifest_audit` bin) and cannot serve as merge gates.
- **Fail-fast matrix defaults for evidence-heavy jobs:** GitHub matrix `fail-fast: true` cancels remaining jobs and hides full regression scope.
- **Profile coverage drift from hardcoded lists:** Always derive profile set from approved matrix in manifest policy (`base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`).
- **Benchmarking every PR as required:** Violates D-09 and increases false-noise from runtime variance.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature-profile matrix execution | Custom shell loops in YAML | `cargo-hack` matrix checks | Handles feature combinations and lock discipline with less drift risk. |
| Oracle target enumeration | Hand-maintained symbol list | Manifest lock + resolver metadata | Prevents stale target lists and enforces source-of-truth policy. |
| Benchmark statistics engine | Manual timers + CSV math | `criterion` baseline APIs | Gives noise handling and stable comparison semantics. |
| Diagnostics logging format | Ad-hoc println!/string logs | `tracing` + JSON subscriber + artifact upload | Produces machine-readable, reviewable evidence for regressions. |

**Key insight:** Phase 4 succeeds by wiring existing verification primitives into enforceable gates, not by inventing new verification logic.

## Common Pitfalls

### Pitfall 1: Existing CI Looks Real but Is Non-Functional
**What goes wrong:** Workflows appear present but invoke non-existent tests/binaries, so policy gates are illusory.
**Why it happens:** Legacy workflow files were not updated alongside crate/test surface changes.
**How to avoid:** Add Wave 0 task that validates every workflow command with `--no-run`/dry-run equivalents.
**Warning signs:** `cargo test --test phase3_*` and `cargo run --bin manifest_audit` fail with "no target named".

### Pitfall 2: Oracle Comparison Stops at First Failure
**What goes wrong:** First mismatch hides total regression surface; reviewers cannot assess blast radius.
**Why it happens:** Current `generate_phase2_parity_report` bails immediately on mismatch.
**How to avoid:** Accumulate per-fixture failures and fail only after writing complete report artifacts.
**Warning signs:** Single-symbol failure messages with no summary counts or matrix-wide report.

### Pitfall 3: Matrix Fail-Fast Cancels Coverage
**What goes wrong:** One profile failure cancels other profile jobs, reducing evidence and masking additional regressions.
**Why it happens:** GitHub matrix defaults `strategy.fail-fast` to `true`.
**How to avoid:** Set `fail-fast: false` on evidence-gathering jobs and enforce required checks at branch protection layer.
**Warning signs:** PR shows canceled matrix jobs after first red cell.

### Pitfall 4: Drift Between Manifest and Oracle Coverage
**What goes wrong:** Manifest lock changes but oracle scope does not, creating false "green" gates.
**Why it happens:** Coverage list is hand-maintained instead of lock-derived.
**How to avoid:** Gate on lock-diff + generated oracle-target diff in same CI run.
**Warning signs:** Manifest symbol count changes without corresponding oracle fixture count changes.

### Pitfall 5: Diagnostics Are Logged but Not Preserved
**What goes wrong:** Runtime fields exist but are lost after CI run, blocking root-cause analysis.
**Why it happens:** Missing artifact upload or inconsistent `/mnt/data` fallback handling.
**How to avoid:** Always emit structured JSON diagnostics and upload artifact bundles for failed runs.
**Warning signs:** CI failure with no attached planner/chunk/fallback/transfer/OOM artifacts.

## Code Examples

Verified patterns from official sources and current code:

### Matrix Strategy With Full Evidence Collection
```yaml
# Source: https://docs.github.com/actions/automating-your-workflow-with-github-actions/workflow-syntax-for-github-actions
jobs:
  verify:
    strategy:
      fail-fast: false
      matrix:
        profile: [base, with-f12, with-4c1e, with-f12+with-4c1e]
    steps:
      - run: cargo hack test --workspace --feature-powerset --depth 1 --locked
```

### Nextest JUnit Artifact Output
```toml
# Source: https://nexte.st/docs/machine-readable/junit/
[profile.ci.junit]
path = "junit.xml"
```

### Runtime Metrics Extraction Contract
```rust
// Source: crates/cintx-runtime/src/metrics.rs
let stats = ExecutionStats {
    chunk_count,
    transfer_bytes,
    not0,
    fallback_reason,
    ..stats
};
write_diagnostics_json(&stats)?;
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Handwritten compatibility checks | Manifest-derived oracle targets across profile matrix | Design sections 3.3/16.2; Phase 2 foundation | Keeps verification tied to canonical API surface. |
| Monolithic `cargo test` gate | Layered gates (`cargo-hack`, oracle compare, OOM checks, scheduled GPU benches) | Current CI best practice for Rust matrix repos | Better signal separation between merge blockers and long-running suites. |
| Console-only diagnostics | Structured tracing + JSON metrics artifacts | Runtime metrics/tracing introduced in earlier phases | Enables reproducible triage of chunking/fallback/OOM regressions. |

**Deprecated/outdated:**
- Phase 3 governance workflow commands that reference missing targets in current tree.
- Stub `ci/*.yml`, `xtask/src/*`, and `benches/*.rs` placeholders without executable gate logic.

## Open Questions

1. **What exact threshold policy should trigger benchmark failures (D-11)?**
   - What we know: Threshold-based gating is required after baseline stabilization.
   - What's unclear: Per-suite slowdown ceilings, warmup windows, and variance policy.
   - Recommendation: Start with report-only calibration window, then lock numeric thresholds in code/config with explicit review policy.

2. **What runner strategy will satisfy GPU required checks in scheduled/merge-queue flows (D-07)?**
   - What we know: GPU jobs are advisory on PRs, required on scheduled/merge queues.
   - What's unclear: Hosted vs self-hosted GPU reliability and retry budget.
   - Recommendation: Define deterministic runner labels and bounded retry policy before making those checks required.

3. **Where should long-term benchmark trends live?**
   - What we know: CI artifacts are required; trend tracking is part of VERI-03.
   - What's unclear: Artifact-only retention vs committed baseline snapshots vs external dashboard.
   - Recommendation: Keep canonical baseline files in-repo and upload per-run artifacts for auditability.

## Sources

### Primary (HIGH confidence)
- Local phase constraints and requirements:
  - `.planning/phases/04-verification-release-automation/04-CONTEXT.md`
  - `.planning/REQUIREMENTS.md`
  - `.planning/ROADMAP.md`
  - `.planning/STATE.md`
- Local implementation evidence:
  - `crates/cintx-oracle/src/compare.rs`
  - `crates/cintx-oracle/src/fixtures.rs`
  - `crates/cintx-runtime/src/planner.rs`
  - `crates/cintx-runtime/src/workspace.rs`
  - `crates/cintx-runtime/src/metrics.rs`
  - `crates/cintx-ops/build.rs`
  - `crates/cintx-ops/src/resolver.rs`
  - `crates/cintx-ops/generated/compiled_manifest.lock.json`
  - `.github/workflows/compat-governance-pr.yml`
  - `.github/workflows/compat-governance-release.yml`
  - `ci/oracle-compare.yml`, `ci/feature-matrix.yml`, `ci/gpu-bench.yml`
  - `xtask/src/main.rs`, `xtask/src/manifest_audit.rs`, `xtask/src/oracle_update.rs`, `xtask/src/bench_report.rs`
  - `benches/micro_families.rs`, `benches/macro_molecules.rs`, `benches/crossover_cpu_gpu.rs`
- Official docs:
  - Cargo resolver behavior (`resolver = "3"`): https://doc.rust-lang.org/nightly/cargo/reference/resolver.html
  - Cargo `--locked` semantics: https://doc.rust-lang.org/cargo/commands/cargo-run.html
  - GitHub Actions workflow syntax (matrix, fail-fast, schedule): https://docs.github.com/actions/automating-your-workflow-with-github-actions/workflow-syntax-for-github-actions
  - Branch protection required checks: https://docs.github.com/en/enterprise-server%403.14/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule
  - Workflow artifacts: https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts
  - nextest JUnit reporting: https://nexte.st/docs/machine-readable/junit/
  - Criterion API (`save_baseline`, `retain_baseline`): https://docs.rs/criterion/latest/criterion/struct.Criterion.html
  - tracing-subscriber docs: https://docs.rs/tracing-subscriber

### Secondary (MEDIUM confidence)
- docs.rs crate metadata pages used for publish-date checks:
  - anyhow: https://docs.rs/crate/anyhow/latest/builds
  - thiserror: https://docs.rs/crate/thiserror/latest/builds
  - tracing-core versions list (for tracing line timing): https://docs.rs/crate/tracing-core/latest
  - criterion crate versions: https://docs.rs/crate/criterion/latest
  - bindgen-cli versions (bindgen release line timing): https://docs.rs/crate/bindgen-cli/latest
  - cargo-hack crate page: https://docs.rs/crate/cargo-hack/latest

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - Tool versions are verified, but docs mirrors lag for some latest publish dates.
- Architecture: HIGH - Recommendations directly map to locked decisions plus existing local oracle/runtime contracts.
- Pitfalls: HIGH - Backed by concrete local failures (missing CI targets, stubs, fail-fast oracle behavior).

**Research date:** 2026-03-28
**Valid until:** 2026-04-27
