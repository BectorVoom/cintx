---
phase: 04-verification-release-automation
verified: 2026-03-28T12:21:05Z
status: human_needed
score: 14/14 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 13/14
  gaps_closed:
    - "Oracle crate exposes a substantive public export surface for profile-aware report entry points consumed by Phase 4 gate tooling."
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Run `.github/workflows/compat-governance-pr.yml` on a real PR with branch protection enabled."
    expected: "`manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, and `oom_contract_gate` are merge-blocking; `gpu_bench_advisory` is non-blocking."
    why_human: "Requires live GitHub branch-protection and Actions enforcement behavior."
  - test: "Run `.github/workflows/compat-governance-release.yml` on a GPU-capable runner."
    expected: "`gpu_bench_required` is blocking and bench/diagnostics artifacts are uploaded from required/fallback paths."
    why_human: "Requires external CI runner capabilities and real workflow execution."
---

# Phase 4: Verification & Release Automation Verification Report

**Phase Goal:** Close the manifest/oracle verification loop, run multi-profile CI/benchmarks, and surface diagnostics that block regressions before release.
**Verified:** 2026-03-28T12:21:05Z
**Status:** human_needed
**Re-verification:** Yes - after gap closure

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Oracle comparison covers required merge-blocking profiles and supports optional/source inclusion when enabled. | ✓ VERIFIED | `PHASE4_APPROVED_PROFILES` and `include_unstable_source` filtering in `crates/cintx-oracle/src/fixtures.rs`; profile-aware parity in `crates/cintx-oracle/src/compare.rs`. |
| 2 | Family tolerances are explicit constants in code. | ✓ VERIFIED | `TOL_*` constants and `tolerance_for_family()` in `crates/cintx-oracle/src/compare.rs`. |
| 3 | Oracle execution persists full mismatch evidence before failing. | ✓ VERIFIED | `build_profile_parity_report()` persists report artifacts before returning; `generate_profile_parity_report()` bails only after reading `report.mismatch_count` in `crates/cintx-oracle/src/compare.rs`. |
| 4 | Maintainers can run merge-blocking verification gates from one xtask CLI entrypoint. | ✓ VERIFIED | Command router and gate dispatch in `xtask/src/main.rs`. |
| 5 | Manifest drift/profile checks run against compiled manifest lock data. | ✓ VERIFIED | `COMPILED_MANIFEST_LOCK_JSON` loading and profile/symbol diff logic in `xtask/src/manifest_audit.rs`. |
| 6 | Helper/legacy parity and OOM no-partial-write checks run as explicit CI gates. | ✓ VERIFIED | `run_helper_legacy_parity()` and `run_oom_contract_check()` in `xtask/src/oracle_update.rs`. |
| 7 | PR verification runs merge-blocking manifest/oracle/helper/OOM gates. | ✓ VERIFIED | Required jobs in `.github/workflows/compat-governance-pr.yml` invoke xtask gates directly. |
| 8 | Required profile matrix includes `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`. | ✓ VERIFIED | Profile matrix lists in `ci/oracle-compare.yml` and `ci/feature-matrix.yml`; `CINTX_REQUIRED_PROFILES` in PR workflow. |
| 9 | CI matrix execution is evidence-complete rather than fail-fast. | ✓ VERIFIED | `strategy.fail-fast: false` in `ci/oracle-compare.yml` and `ci/feature-matrix.yml`. |
| 10 | Bench automation includes micro, macro, and CPU-GPU crossover suites with artifacted rows. | ✓ VERIFIED | Implemented Criterion benches in `benches/micro_families.rs`, `benches/macro_molecules.rs`, `benches/crossover_cpu_gpu.rs` writing JSONL summary rows. |
| 11 | Benchmark gates fail only on threshold exceedance. | ✓ VERIFIED | Threshold comparison logic and fail condition in `xtask/src/bench_report.rs`; thresholds in `ci/benchmark-thresholds.json`. |
| 12 | Diagnostics artifacts include chunk/fallback/transfer/OOM-related runtime fields with required path metadata. | ✓ VERIFIED | `chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`, `/mnt/data` required paths in `xtask/src/bench_report.rs`. |
| 13 | GPU benchmark policy is advisory on PR and required on release/schedule. | ✓ VERIFIED | `gpu_bench_advisory` uses `continue-on-error: true` in PR workflow; `gpu_bench_required` uses `continue-on-error: false` in release workflow. |
| 14 | Oracle crate exposes a substantive public export surface for Phase 4 tooling. | ✓ VERIFIED | `crates/cintx-oracle/src/lib.rs` is now 20 lines and explicitly re-exports fixture/parity APIs and constants at crate root. |

**Score:** 14/14 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `crates/cintx-oracle/src/fixtures.rs` | Profile-aware fixture matrix generation from compiled lock metadata | ✓ VERIFIED | Exists; 615 lines (>=380); consumed by compare + xtask modules. |
| `crates/cintx-oracle/src/compare.rs` | Full-matrix parity evaluator with mismatch aggregation and tolerance tables | ✓ VERIFIED | Exists; 997 lines (>=760); used by xtask oracle gate. |
| `crates/cintx-oracle/src/lib.rs` | Public exports for profile-aware oracle entry points | ✓ VERIFIED | Exists; 20 lines (>=20); grouped `pub use` exports for fixtures and compare APIs/constants are present. |
| `xtask/src/main.rs` | Gate command router for Phase 4 checks | ✓ VERIFIED | Exists; 269 lines (>=130); workflow commands route through this entrypoint. |
| `xtask/src/manifest_audit.rs` | Lock drift and profile coverage checker | ✓ VERIFIED | Exists; 269 lines (>=180); wired via `manifest-audit` command. |
| `xtask/src/oracle_update.rs` | Oracle/helper/OOM gate runners | ✓ VERIFIED | Exists; 335 lines (>=220); wired via `oracle-compare`, `helper-legacy-parity`, `oom-contract-check`. |
| `ci/oracle-compare.yml` | Non-fail-fast oracle/profile CI template with artifacts | ✓ VERIFIED | Exists; 82 lines (>=70); includes matrix + gate commands. |
| `ci/feature-matrix.yml` | Manifest/OOM CI template for required profiles | ✓ VERIFIED | Exists; 77 lines (>=70); includes required profile matrix + gates. |
| `.github/workflows/compat-governance-pr.yml` | PR workflow wiring required merge-blocking gates | ✓ VERIFIED | Exists; 209 lines (>=110); required gate jobs present. |
| `benches/micro_families.rs` | Micro-family benchmark suite | ✓ VERIFIED | Exists; 169 lines (>=120); declared bench target + artifact row writes. |
| `benches/macro_molecules.rs` | Macro workload benchmark suite | ✓ VERIFIED | Exists; 167 lines (>=120); declared bench target + artifact row writes. |
| `benches/crossover_cpu_gpu.rs` | CPU-GPU crossover benchmark suite | ✓ VERIFIED | Exists; 173 lines (>=120); declared bench target + crossover metrics. |
| `xtask/src/bench_report.rs` | Threshold-aware benchmark/diagnostic report generator | ✓ VERIFIED | Exists; 755 lines (>=240); wired through xtask `bench-report` command. |
| `ci/benchmark-thresholds.json` | Versioned regression thresholds | ✓ VERIFIED | Exists; 29 lines (>=20); consumed by bench-report and workflows. |
| `ci/gpu-bench.yml` | GPU benchmark policy template with artifact upload | ✓ VERIFIED | Exists; 93 lines (>=80); policy template aligns with PR/release workflow jobs. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `crates/cintx-oracle/src/fixtures.rs` | `crates/cintx-ops/generated/compiled_manifest.lock.json` | lock-profile/stability filtering | WIRED | `compiled_manifest.lock.json`, `with-f12+with-4c1e`, and `stability` patterns present. |
| `crates/cintx-oracle/src/compare.rs` | `crates/cintx-compat/src/raw.rs` | raw-vs-legacy parity evaluation | WIRED | `raw::eval_raw` and `eval_legacy_symbol` present in parity loop. |
| `crates/cintx-oracle/src/compare.rs` | `crates/cintx-oracle/src/fixtures.rs` | persisted mismatch report path metadata | WIRED | `write_pretty_json_artifact`, `required_path`, and `mismatch_count` present. |
| `xtask/src/manifest_audit.rs` | `crates/cintx-ops/generated/compiled_manifest.lock.json` | lock/profile/symbol audit | WIRED | Compiled lock constant and profile/scope checks present. |
| `xtask/src/oracle_update.rs` | `crates/cintx-oracle/src/compare.rs` | profile-aware oracle gate invocation | WIRED | `generate_profile_parity_report` + `verify_helper_surface_coverage` imported/called. |
| `xtask/src/oracle_update.rs` | `crates/cintx-compat/src/raw.rs` | OOM regression command | WIRED | Executes `raw::tests::memory_limit_failure_keeps_output_slice_unchanged`. |
| `.github/workflows/compat-governance-pr.yml` | `xtask/src/main.rs` | direct xtask gate commands | WIRED | Runs `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, `oom-contract-check`. |
| `ci/oracle-compare.yml` | `crates/cintx-oracle/src/compare.rs` | oracle gate command template | WIRED | Uses `oracle-compare --include-unstable-source false`. |
| `ci/feature-matrix.yml` | `xtask/src/manifest_audit.rs` | manifest audit gate template | WIRED | Uses `manifest-audit`; matrix includes `with-f12+with-4c1e`. |
| `xtask/src/bench_report.rs` | `crates/cintx-runtime/src/metrics.rs` | runtime diagnostics field contract | WIRED | Reports `chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`. |
| `.github/workflows/compat-governance-pr.yml` | `ci/gpu-bench.yml` | PR advisory benchmark policy | WIRED | `gpu_bench_advisory` job with `continue-on-error: true`, command set aligned to template policy. |
| `.github/workflows/compat-governance-release.yml` | `ci/gpu-bench.yml` | release/scheduled required benchmark policy | WIRED | `gpu_bench_required` job with `continue-on-error: false`, command set aligned to template policy. |
| `crates/cintx-oracle/src/lib.rs` | `crates/cintx-oracle/src/fixtures.rs` | crate-root fixture re-exports for Phase 4 tooling | WIRED | `build_profile_representation_matrix`, `build_required_profile_matrices`, `PHASE4_APPROVED_PROFILES`, and `PHASE4_ORACLE_FAMILIES` re-exported at crate root. |
| `crates/cintx-oracle/src/lib.rs` | `crates/cintx-oracle/src/compare.rs` | crate-root parity/tolerance/helper re-exports | WIRED | `generate_profile_parity_report`, `generate_phase2_parity_report`, `verify_helper_surface_coverage`, and `tolerance_for_family` re-exported at crate root. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `VERI-01` | `01-PLAN.md`, `05-PLAN.md` | Oracle compares stable/optional APIs against vendored upstream with family tolerances | ✓ SATISFIED | Profile-aware parity + tolerance table in `crates/cintx-oracle/src/compare.rs`; fixture/profile lock coverage in `crates/cintx-oracle/src/fixtures.rs`; crate-root export gap closed in `crates/cintx-oracle/src/lib.rs`. |
| `VERI-02` | `02-PLAN.md`, `03-PLAN.md`, `05-PLAN.md` | CI blocks manifest drift, helper/legacy regressions, consistency, OOM violations | ✓ SATISFIED | xtask gates implemented (`xtask/src/main.rs`, `xtask/src/manifest_audit.rs`, `xtask/src/oracle_update.rs`) and wired in `.github/workflows/compat-governance-pr.yml`. |
| `VERI-03` | `04-PLAN.md`, `05-PLAN.md` | Benchmarks track throughput/memory/crossover regressions | ✓ SATISFIED | Bench suites + threshold report (`benches/*.rs`, `xtask/src/bench_report.rs`, `ci/benchmark-thresholds.json`); no regression after gap closure. |
| `VERI-04` | `04-PLAN.md`, `05-PLAN.md` | Diagnostics expose planner/chunking/transfer/fallback/OOM behavior | ✓ SATISFIED | Runtime diagnostics artifact includes `chunk_count`, `fallback_reason`, `transfer_bytes`, and `not0` in `xtask/src/bench_report.rs`. |
| `ORPHANED` | n/a | Additional Phase 4 requirements in `REQUIREMENTS.md` not claimed by plan frontmatter | ✓ SATISFIED | None. Plan frontmatter IDs resolve to `VERI-01`, `VERI-02`, `VERI-03`, `VERI-04`; no extra Phase 4 verification IDs found. |

### Anti-Patterns Found

No blocker/warning anti-patterns were detected in scanned Phase 4 implementation files (`TODO/FIXME`, placeholder text, empty implementations, console-log-only paths, hardcoded-empty output stubs). One regex hit in `crates/cintx-oracle/src/compare.rs` was a false positive inside formatted mismatch messages, not a stub path.

### Human Verification Required

### 1. PR Gate Enforcement in GitHub Branch Protection

**Test:** Run `.github/workflows/compat-governance-pr.yml` on a pull request and verify required check behavior with branch protection.
**Expected:** `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, and `oom_contract_gate` act as merge-blocking checks; `gpu_bench_advisory` remains non-blocking.
**Why human:** Requires live GitHub Actions + branch protection configuration, which cannot be validated from repository static analysis alone.

### 2. Release/Scheduled GPU Runner Validation

**Test:** Run `.github/workflows/compat-governance-release.yml` on GPU-capable runners and inspect uploaded bench/diagnostics artifacts.
**Expected:** `gpu_bench_required` is blocking; artifacts include bench and runtime diagnostics outputs from required/fallback paths.
**Why human:** Needs real CI infrastructure and runtime environment characteristics (runner capabilities, permissions, `/mnt/data` behavior).

### Gaps Summary

The previous must-have gap is closed: `crates/cintx-oracle/src/lib.rs` now meets the `min_lines: 20` substance gate and preserves explicit profile-aware crate-root exports. No regressions were found in previously passing artifacts or key links during re-verification. Automated Phase 4 goal checks pass; remaining validation is operational GitHub workflow behavior requiring live infrastructure.

---

_Verified: 2026-03-28T12:21:05Z_
_Verifier: Claude (gsd-verifier)_
