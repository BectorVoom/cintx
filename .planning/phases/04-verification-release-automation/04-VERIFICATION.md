---
phase: 04-verification-release-automation
verified: 2026-03-29T02:07:01Z
status: human_needed
score: 18/18 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 17/18
  gaps_closed:
    - "`gpu_bench_required` runs on an explicit GPU-capable runner contract and remains blocking."
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "PR branch protection required-check behavior"
    expected: "`manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, and `oom_contract_gate` block merge; `gpu_bench_advisory` remains non-blocking."
    why_human: "Requires live GitHub branch protection and Actions policy, which cannot be verified from local code alone."
  - test: "Release/scheduled GPU runner execution"
    expected: "`gpu_bench_required` runs on a GPU-capable runner and fails closed when benchmark/diagnostics artifact contracts are violated."
    why_human: "Requires remote GPU runner availability and real workflow execution."
  - test: "Benchmark trend tracking over time"
    expected: "Throughput/memory/crossover trend data accumulates across runs and threshold gates behave as intended with real benchmark data."
    why_human: "Trend behavior requires multiple real CI benchmark runs; static code review cannot validate longitudinal outcomes."
---

# Phase 4: Verification & Release Automation Verification Report

**Phase Goal:** Close the manifest/oracle verification loop, run multi-profile CI/benchmarks, and surface diagnostics that block regressions before release.  
**Verified:** 2026-03-29T02:07:01Z  
**Status:** human_needed  
**Re-verification:** Yes — after gap closure

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Oracle comparison covers required merge-blocking profiles and optional/source inclusion when enabled. | ✓ VERIFIED | `crates/cintx-oracle/src/fixtures.rs` contains `PHASE4_APPROVED_PROFILES`, profile builders, and lock/stability filtering; `compare.rs` contains `generate_profile_parity_report`. |
| 2 | Family tolerances remain explicit code-level constants. | ✓ VERIFIED | `crates/cintx-oracle/src/compare.rs` defines `TOL_*` constants and `tolerance_for_family()`. |
| 3 | Oracle execution writes full mismatch evidence before failure. | ✓ VERIFIED | `crates/cintx-oracle/src/compare.rs` writes report fields `fixture_count`, `mismatch_count`, `mismatches` via `write_pretty_json_artifact` before post-write failure. |
| 4 | Maintainers can run merge-blocking verification gates from one xtask entrypoint. | ✓ VERIFIED | `xtask/src/main.rs` routes `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, and `oom-contract-check`. |
| 5 | Manifest/profile drift checks are lock-authoritative. | ✓ VERIFIED | `xtask/src/manifest_audit.rs` reads `compiled_manifest.lock.json` and computes `missing_in_lock` / `missing_in_generated` / `profile_scope_mismatch`. |
| 6 | Helper/legacy parity and OOM no-partial-write checks are explicit gates. | ✓ VERIFIED | `xtask/src/oracle_update.rs` has `run_helper_legacy_parity` and `run_oom_contract_check` invoking exact regression tests. |
| 7 | PR verification runs required merge-blocking manifest/oracle/helper/OOM gates. | ✓ VERIFIED | `.github/workflows/compat-governance-pr.yml` defines required jobs and xtask invocations for all four gates. |
| 8 | Required profile matrix includes `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`. | ✓ VERIFIED | Present in `ci/oracle-compare.yml`, `ci/feature-matrix.yml`, PR workflow env, and xtask defaults. |
| 9 | Matrix workflows collect evidence without fail-fast cancellation. | ✓ VERIFIED | `strategy.fail-fast: false` in both `ci/oracle-compare.yml` and `ci/feature-matrix.yml`. |
| 10 | Bench automation covers micro, macro, and crossover suites with structured output. | ✓ VERIFIED | `benches/micro_families.rs`, `macro_molecules.rs`, `crossover_cpu_gpu.rs` include Criterion suites and structured row fields. |
| 11 | Benchmark gate fails only on configured threshold exceedance. | ✓ VERIFIED | `xtask/src/bench_report.rs` enforces failure only when `mode == enforce` and threshold exceedances exist. |
| 12 | Runtime diagnostics include chunk/fallback/transfer/OOM-relevant fields and required-path metadata. | ✓ VERIFIED | `xtask/src/bench_report.rs` emits `chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`, and required/fallback artifact metadata. |
| 13 | GPU policy is advisory on PR and required on release/scheduled flows. | ✓ VERIFIED | PR workflow has `gpu_bench_advisory` with `continue-on-error: true`; release workflow has `gpu_bench_required` with `continue-on-error: false`. |
| 14 | Oracle crate root is non-stub and export-substantive. | ✓ VERIFIED | `crates/cintx-oracle/src/lib.rs` is 20 lines and explicitly re-exports profile-aware fixture/parity APIs. |
| 15 | Profile-aware fixture/parity exports are crate-root importable. | ✓ VERIFIED | `crates/cintx-oracle/src/lib.rs` exports `build_profile_representation_matrix`, `build_required_profile_matrices`, `generate_profile_parity_report`, `tolerance_for_family`, and constants/types. |
| 16 | `gpu_bench_required` runs on explicit GPU-capable runner and remains blocking. | ✓ VERIFIED | `.github/workflows/compat-governance-release.yml` now has `runs-on: [self-hosted, linux, x64, gpu]`, `continue-on-error: false`, and file length 209 (>=180 threshold). |
| 17 | Release GPU bench fails when neither required nor fallback benchmark/diagnostic artifacts are present. | ✓ VERIFIED | `Validate bench artifact contract` exists in both release workflow and `ci/gpu-bench.yml`, checking required+fallback report/diagnostics files. |
| 18 | Release artifact uploads include bench rows/report/diagnostics from required and fallback locations. | ✓ VERIFIED | Release workflow and GPU template upload `/mnt/data/...` and `/tmp/cintx_artifacts/...` bench/diagnostic outputs. |

**Score:** 18/18 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `crates/cintx-oracle/src/fixtures.rs` | Manifest/profile-aware fixture matrix generation | ✓ VERIFIED | Exists; 615 lines (>=380). |
| `crates/cintx-oracle/src/compare.rs` | Full-matrix parity + explicit tolerance + mismatch aggregation | ✓ VERIFIED | Exists; 997 lines (>=760). |
| `crates/cintx-oracle/src/lib.rs` | Profile-aware crate-root exports | ✓ VERIFIED | Exists; 20 lines (>=20). |
| `xtask/src/main.rs` | Gate command router | ✓ VERIFIED | Exists; 269 lines (>=130). |
| `xtask/src/manifest_audit.rs` | Lock-driven drift and profile audit | ✓ VERIFIED | Exists; 269 lines (>=180). |
| `xtask/src/oracle_update.rs` | Oracle/helper/OOM gate runners | ✓ VERIFIED | Exists; 335 lines (>=220). |
| `ci/oracle-compare.yml` | Non-fail-fast oracle template | ✓ VERIFIED | Exists; 82 lines (>=70). |
| `ci/feature-matrix.yml` | Manifest/OOM template | ✓ VERIFIED | Exists; 77 lines (>=70). |
| `.github/workflows/compat-governance-pr.yml` | Required PR governance workflow | ✓ VERIFIED | Exists; 209 lines (>=110). |
| `benches/micro_families.rs` | Micro benchmark suite | ✓ VERIFIED | Exists; 169 lines (>=120). |
| `benches/macro_molecules.rs` | Macro benchmark suite | ✓ VERIFIED | Exists; 167 lines (>=120). |
| `benches/crossover_cpu_gpu.rs` | Crossover benchmark suite | ✓ VERIFIED | Exists; 173 lines (>=120). |
| `xtask/src/bench_report.rs` | Threshold-aware bench+diagnostics report engine | ✓ VERIFIED | Exists; 755 lines (>=240). |
| `ci/benchmark-thresholds.json` | Versioned regression thresholds | ✓ VERIFIED | Exists; 29 lines (>=20). |
| `ci/gpu-bench.yml` | GPU template with artifact contract checks | ✓ VERIFIED | Exists; 114 lines (>=90 and >=80). |
| `.github/workflows/compat-governance-release.yml` | Required release GPU gate + artifact governance | ✓ VERIFIED | Exists; 209 lines (>=180). Previous gap closed. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `crates/cintx-oracle/src/fixtures.rs` | `crates/cintx-ops/generated/compiled_manifest.lock.json` | lock profile/stability filtering | WIRED | Lock include path, required profiles, and stability checks are present. |
| `crates/cintx-oracle/src/compare.rs` | `crates/cintx-compat/src/raw.rs` | raw-vs-legacy parity paths | WIRED | `raw::eval_raw` and legacy symbol evaluation are present. |
| `crates/cintx-oracle/src/compare.rs` | `crates/cintx-oracle/src/fixtures.rs` | report persistence with required-path metadata | WIRED | Uses `write_pretty_json_artifact`, includes `required_path`, and tracks mismatch counts. |
| `xtask/src/manifest_audit.rs` | `crates/cintx-ops/generated/compiled_manifest.lock.json` | lock/profile scope and symbol drift checks | WIRED | Lock constant and drift/scope report fields present. |
| `xtask/src/oracle_update.rs` | `crates/cintx-oracle/src/compare.rs` | profile-aware oracle + helper checks | WIRED | Calls `generate_profile_parity_report` and `verify_helper_surface_coverage`. |
| `xtask/src/oracle_update.rs` | `crates/cintx-compat/src/raw.rs` | OOM no-partial-write regression gate | WIRED | Invokes `memory_limit_failure_keeps_output_slice_unchanged` test. |
| `.github/workflows/compat-governance-pr.yml` | `xtask/src/main.rs` | PR gate execution path | WIRED | Direct xtask calls for `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, `oom-contract-check`. |
| `ci/oracle-compare.yml` | `crates/cintx-oracle/src/compare.rs` | profile-aware oracle command contract | WIRED | `oracle-compare --include-unstable-source false` present. |
| `ci/feature-matrix.yml` | `xtask/src/manifest_audit.rs` | manifest lock and profile scope enforcement | WIRED | `manifest-audit --check-lock` with required matrix entries present. |
| `xtask/src/bench_report.rs` | `crates/cintx-runtime/src/metrics.rs` | diagnostics contract field mapping | WIRED | Diagnostics emit `chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`. |
| `.github/workflows/compat-governance-pr.yml` | `ci/gpu-bench.yml` | advisory PR GPU policy | WIRED | `gpu_bench_advisory` + `continue-on-error: true` present. |
| `.github/workflows/compat-governance-release.yml` | `ci/gpu-bench.yml` | required release GPU policy | WIRED | GPU runner labels, enforce-mode bench-report, and artifact contracts are aligned. |
| `.github/workflows/compat-governance-release.yml` | `/mnt/data` + fallback artifact paths | required/fallback contract enforcement | WIRED | Invariant checks + `Validate bench artifact contract` + upload paths cover both required/fallback locations. |
| `crates/cintx-oracle/src/lib.rs` | `crates/cintx-oracle/src/fixtures.rs` | crate-root fixture re-exports | WIRED | Fixture constants/builders re-exported. |
| `crates/cintx-oracle/src/lib.rs` | `crates/cintx-oracle/src/compare.rs` | crate-root parity/tolerance/helper re-exports | WIRED | Compare APIs/types re-exported. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `VERI-01` | `01-PLAN.md`, `05-PLAN.md` | Oracle comparison coverage with explicit tolerances | ✓ SATISFIED | Profile-aware fixture/parity logic, explicit tolerance constants, and report-first mismatch behavior are implemented. |
| `VERI-02` | `02-PLAN.md`, `03-PLAN.md`, `05-PLAN.md`, `06-PLAN.md`, `07-PLAN.md` | CI blocks manifest/helper-legacy/CubeCL/OOM regressions | ? NEEDS HUMAN | Local wiring is complete; live branch-protection and Actions gate behavior need remote verification. |
| `VERI-03` | `04-PLAN.md`, `05-PLAN.md`, `06-PLAN.md`, `07-PLAN.md` | Benchmark trend tracking across throughput/memory/crossover | ? NEEDS HUMAN | Bench suites + thresholds + report pipeline are present; trend validation needs repeated real runs. |
| `VERI-04` | `04-PLAN.md`, `05-PLAN.md`, `06-PLAN.md`, `07-PLAN.md` | Diagnostics for planner/chunk/transfer/fallback/OOM behavior | ? NEEDS HUMAN | Diagnostics schema and artifact contracts are present; runtime behavior requires executed workloads. |
| `ORPHANED` | n/a | Phase 4 VERI IDs in REQUIREMENTS.md not claimed by any plan `requirements` field | ✓ SATISFIED | None orphaned. `VERI-01`, `VERI-02`, `VERI-03`, and `VERI-04` are all represented in plan frontmatter. |

### Anti-Patterns Found

No blocker/warning anti-patterns found in scanned phase artifacts.  
One regex artifact (`{}` in format strings) appears in `crates/cintx-oracle/src/compare.rs` and is non-stub.

### Human Verification Required

### 1. PR Branch Protection Required-Check Behavior

**Test:** Run `.github/workflows/compat-governance-pr.yml` on a protected PR branch.  
**Expected:** `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, and `oom_contract_gate` are merge-blocking; `gpu_bench_advisory` remains non-blocking.  
**Why human:** Branch-protection enforcement is external GitHub policy.

### 2. Release/Scheduled GPU Runner Execution

**Test:** Run `.github/workflows/compat-governance-release.yml` on a GPU-capable runner.  
**Expected:** `gpu_bench_required` runs on `[self-hosted, linux, x64, gpu]`, remains blocking, and validates required/fallback artifact contracts.  
**Why human:** Requires external runner availability and real Actions execution.

### 3. Benchmark Trend + Diagnostics Artifact Validation

**Test:** Execute repeated benchmark/report runs and inspect generated artifacts.  
**Expected:** Threshold behavior reflects actual regressions, and diagnostics artifacts expose chunk/fallback/transfer/not0 fields across real workloads.  
**Why human:** Longitudinal and runtime behavior cannot be fully proven via static code inspection.

### Gaps Summary

No remaining code-level must-have gaps were found in re-verification.  
The previously failing artifact threshold for `.github/workflows/compat-governance-release.yml` is now closed (209 >= 180), and no regressions were detected in previously passing must-haves.

---

_Verified: 2026-03-29T02:07:01Z_  
_Verifier: Claude (gsd-verifier)_
