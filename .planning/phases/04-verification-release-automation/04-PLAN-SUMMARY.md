---
phase: 04-verification-release-automation
plan: 04
subsystem: infra
tags: [criterion, github-actions, benchmark, diagnostics, xtask]
requires:
  - phase: 04-verification-release-automation-02
    provides: oracle/manifest/OOM gate command surface and CI baseline
  - phase: 04-verification-release-automation-03
    provides: PR/release governance workflow scaffolding
provides:
  - Criterion-backed micro/macro/crossover benchmark suites with structured summary rows
  - Threshold-aware `xtask bench-report` diagnostics and regression gating
  - Advisory PR and required release GPU benchmark workflow policy with artifact uploads
affects: [ci, release-automation, benchmark-governance, diagnostics]
tech-stack:
  added: [criterion-0.8.2]
  patterns: [threshold-gated-regressions, required-path-artifact-fallback, advisory-vs-required-ci-jobs]
key-files:
  created: [ci/benchmark-thresholds.json, .planning/phases/04-verification-release-automation/04-PLAN-SUMMARY.md]
  modified: [Cargo.toml, benches/micro_families.rs, benches/macro_molecules.rs, benches/crossover_cpu_gpu.rs, xtask/src/bench_report.rs, xtask/src/main.rs, ci/gpu-bench.yml, .github/workflows/compat-governance-pr.yml, .github/workflows/compat-governance-release.yml]
key-decisions:
  - "Bench regressions fail only on configured threshold exceedance, not any slowdown."
  - "Bench and runtime diagnostics artifacts must target /mnt/data with CINTX_ARTIFACT_DIR fallback metadata."
  - "PR GPU/bench execution is advisory (`continue-on-error: true`) while release/scheduled/merge-queue execution is required (`continue-on-error: false`)."
patterns-established:
  - "Bench suites append JSONL summary rows consumed by `xtask bench-report`."
  - "CI jobs upload both required-path and fallback-path artifact locations to preserve diagnostics."
requirements-completed: [VERI-03, VERI-04]
duration: 17min
completed: 2026-03-28
---

# Phase 4 Plan 4: Benchmark + GPU Governance Summary

**Criterion micro/macro/crossover benchmarks, threshold-gated bench reporting, and PR-advisory vs release-required GPU workflow policy were implemented with artifactized runtime diagnostics.**

## Performance

- **Duration:** 17 min
- **Started:** 2026-03-28T11:29:19Z
- **Completed:** 2026-03-28T11:46:31Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments
- Added `criterion = "0.8.2"` bench wiring and replaced all three benchmark stubs with compile-ready suites that emit structured summary rows for downstream report parsing.
- Replaced bench-report stub with a threshold-aware report engine that emits `/mnt/data` benchmark and runtime diagnostics artifacts with fallback metadata and fail-on-threshold-exceed semantics.
- Replaced GPU bench CI stub and wired governance workflows so PR jobs are advisory while release/scheduled/merge-queue jobs are required, both with bench/diagnostics artifact uploads.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement Phase 4 benchmark suites and Cargo bench wiring** - `578748d` (feat)
2. **Task 2: Implement threshold-based bench and diagnostics report generation** - `61405bc` (feat)
3. **Task 3: Wire GPU benchmark policy across PR-advisory and release-required workflows** - `9442a3b` (feat)

**Plan metadata:** pending final docs commit

## Files Created/Modified
- `Cargo.toml` - Added Criterion dev-dependency and explicit bench targets with `harness = false`.
- `Cargo.lock` - Locked Criterion and related benchmark dependencies.
- `benches/micro_families.rs` - Added micro-family Criterion suite with structured summary row emission.
- `benches/macro_molecules.rs` - Added macro workload Criterion suite with throughput/workspace/transfer counters.
- `benches/crossover_cpu_gpu.rs` - Added CPU-GPU crossover suite with crossover shift metrics.
- `ci/benchmark-thresholds.json` - Added versioned benchmark baselines and regression thresholds.
- `xtask/src/bench_report.rs` - Implemented threshold-aware bench and runtime diagnostics artifact generation with fallback handling.
- `xtask/src/main.rs` - Added `bench-report` command routing and CLI parsing.
- `ci/gpu-bench.yml` - Replaced stub with concrete GPU/benchmark workflow policy template and artifact upload contract.
- `.github/workflows/compat-governance-pr.yml` - Added `gpu_bench_advisory` job with `continue-on-error: true`.
- `.github/workflows/compat-governance-release.yml` - Added `gpu_bench_required` job with `continue-on-error: false` and release/schedule/merge-group trigger coverage.

## Decisions Made
- Kept regression policy fail-closed only on threshold exceedance to satisfy D-11 while avoiding any-slowdown false positives.
- Standardized bench/diagnostic artifacts to required `/mnt/data` paths plus `CINTX_ARTIFACT_DIR` fallback metadata for CI portability (D-12).
- Used explicit advisory/required split via job-level `continue-on-error` to encode D-07 and D-09 policy directly in workflow files.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `bench-report` command wiring in `xtask/src/main.rs`**
- **Found during:** Task 2 (threshold-based bench and diagnostics report generation)
- **Issue:** The plan-required verification command `cargo run --manifest-path xtask/Cargo.toml -- bench-report ...` could not run because `xtask` did not parse a `bench-report` command.
- **Fix:** Added `Command::BenchReport`, CLI parsing (`--thresholds`, `--mode`), command dispatch, and help text updates.
- **Files modified:** `xtask/src/main.rs`
- **Verification:** `cargo run --manifest-path xtask/Cargo.toml -- bench-report --thresholds ci/benchmark-thresholds.json --mode calibration`
- **Committed in:** `61405bc` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for command executability; no scope creep and fully aligned with planned deliverables.

## Issues Encountered
- Initial `cargo bench` compile check needed unsandboxed Cargo cache access due sandbox write restrictions; reran successfully and verified all three bench targets compile.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Benchmark suites, thresholds, diagnostics artifacts, and GPU policy wiring are now in place for release automation verification.
- Remaining work should focus on exercising these flows in CI with real runner availability and branch-protection required check mapping.

---
*Phase: 04-verification-release-automation*
*Completed: 2026-03-28*

## Self-Check: PASSED

- Found summary file: `.planning/phases/04-verification-release-automation/04-PLAN-SUMMARY.md`
- Verified task commits: `578748d`, `61405bc`, `9442a3b`
