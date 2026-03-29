---
phase: 04-verification-release-automation
verified: 2026-03-29T00:38:49Z
status: gaps_found
score: 17/18 must-haves verified
re_verification:
  previous_status: human_needed
  previous_score: 14/14
  gaps_closed: []
  gaps_remaining:
    - "Release governance workflow artifact does not meet declared min_lines gate (168 < 180)."
  regressions:
    - "Plan 06 introduced a new must-have artifact threshold for `.github/workflows/compat-governance-release.yml` that is currently unmet."
gaps:
  - truth: "`gpu_bench_required` runs on an explicit GPU-capable runner contract and remains blocking."
    status: partial
    reason: "Runner labels and blocking semantics are wired, but the supporting artifact fails its declared substance gate."
    artifacts:
      - path: ".github/workflows/compat-governance-release.yml"
        issue: "Declared `min_lines: 180` in `06-PLAN.md`; actual file length is 168 lines."
    missing:
      - "Raise `.github/workflows/compat-governance-release.yml` to at least 180 substantive lines while preserving current runner/bench-report/artifact-contract behavior."
---

# Phase 4: Verification & Release Automation Verification Report

**Phase Goal:** Close the manifest/oracle verification loop, run multi-profile CI/benchmarks, and surface diagnostics that block regressions before release.
**Verified:** 2026-03-29T00:38:49Z
**Status:** gaps_found
**Re-verification:** Yes — previous verification existed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Oracle comparison covers required merge-blocking profiles and supports optional/source inclusion when enabled. | ✓ VERIFIED | `PHASE4_APPROVED_PROFILES` + profile builders in `crates/cintx-oracle/src/fixtures.rs`; profile-aware parity in `crates/cintx-oracle/src/compare.rs`. |
| 2 | Family tolerances are explicit code constants. | ✓ VERIFIED | `TOL_*` constants and `tolerance_for_family()` in `crates/cintx-oracle/src/compare.rs`. |
| 3 | Oracle parity persists full mismatch evidence before failing. | ✓ VERIFIED | `build_profile_parity_report()` writes report via `write_pretty_json_artifact`; `generate_profile_parity_report()` bails only after `mismatch_count` check. |
| 4 | Merge-blocking verification gates are runnable from one xtask entrypoint. | ✓ VERIFIED | `xtask/src/main.rs` routes `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, `oom-contract-check`, and `bench-report`. |
| 5 | Manifest drift/profile coverage uses compiled lock data, not hand lists. | ✓ VERIFIED | `COMPILED_MANIFEST_LOCK_JSON` and lock/profile diff logic in `xtask/src/manifest_audit.rs`. |
| 6 | Helper/legacy parity and OOM no-partial-write checks are explicit CI-callable gates. | ✓ VERIFIED | `run_helper_legacy_parity()` and `run_oom_contract_check()` in `xtask/src/oracle_update.rs`. |
| 7 | PR verification runs merge-blocking manifest/oracle/helper/OOM gates. | ✓ VERIFIED | Required jobs in `.github/workflows/compat-governance-pr.yml` call xtask gates directly. |
| 8 | Required profile matrix includes `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`. | ✓ VERIFIED | Profile set appears in `ci/oracle-compare.yml`, `ci/feature-matrix.yml`, and `CINTX_REQUIRED_PROFILES` in PR workflow. |
| 9 | Matrix workflows collect full evidence instead of fail-fast cancellation. | ✓ VERIFIED | `strategy.fail-fast: false` in `ci/oracle-compare.yml` and `ci/feature-matrix.yml`. |
| 10 | Bench automation covers micro, macro, and CPU-GPU crossover suites with reproducible row output. | ✓ VERIFIED | `benches/micro_families.rs`, `benches/macro_molecules.rs`, and `benches/crossover_cpu_gpu.rs` append JSONL summary rows. |
| 11 | Benchmark gates fail only when configured thresholds are exceeded. | ✓ VERIFIED | Threshold evaluation + enforce-mode fail policy in `xtask/src/bench_report.rs` with config in `ci/benchmark-thresholds.json`. |
| 12 | Runtime diagnostics include chunk/fallback/transfer/OOM-relevant fields with required-path metadata. | ✓ VERIFIED | `chunk_count`, `fallback_reason`, `transfer_bytes`, `not0`, and `/mnt/data` artifact policy in `xtask/src/bench_report.rs`. |
| 13 | GPU policy is advisory on PR and required on release/scheduled flows. | ✓ VERIFIED | `gpu_bench_advisory` has `continue-on-error: true`; release `gpu_bench_required` has `continue-on-error: false`. |
| 14 | Oracle crate root is substantive (>=20 lines) and re-exports profile-aware APIs. | ✓ VERIFIED | `crates/cintx-oracle/src/lib.rs` is exactly 20 lines and exports fixture/parity APIs/constants. |
| 15 | Profile-aware fixture/parity exports remain explicit and importable from crate root. | ✓ VERIFIED | `pub use fixtures::{...}` and `pub use compare::{...}` in `crates/cintx-oracle/src/lib.rs`. |
| 16 | `gpu_bench_required` runs on explicit GPU-capable runner contract and remains blocking. | ✗ FAILED | Runner/blocking wiring exists, but supporting artifact `.github/workflows/compat-governance-release.yml` fails declared `min_lines` substance gate (168 < 180). |
| 17 | Release GPU bench fails when neither required nor fallback benchmark/diagnostic artifacts exist. | ✓ VERIFIED | `Validate bench artifact contract` step checks required/fallback report + diagnostics paths in release workflow and `ci/gpu-bench.yml`. |
| 18 | Release artifact uploads include bench rows, bench report, and runtime diagnostics from required and fallback locations. | ✓ VERIFIED | Upload paths include `/mnt/data/...` and `/tmp/cintx_artifacts/...` bench/diagnostic files in release workflow and template. |

**Score:** 17/18 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `crates/cintx-oracle/src/fixtures.rs` | Profile-aware fixture generation from compiled lock metadata | ✓ VERIFIED | Exists; 615 lines (>=380); profile/stability filtering and required profile builders present. |
| `crates/cintx-oracle/src/compare.rs` | Full-matrix parity evaluator with explicit tolerances + mismatch aggregation | ✓ VERIFIED | Exists; 997 lines (>=760); report-first parity path implemented. |
| `crates/cintx-oracle/src/lib.rs` | Non-stub crate-root export hub | ✓ VERIFIED | Exists; 20 lines (>=20); explicit grouped re-exports present. |
| `xtask/src/main.rs` | Executable gate command router | ✓ VERIFIED | Exists; 269 lines (>=130); required commands and fail-closed entrypoint present. |
| `xtask/src/manifest_audit.rs` | Manifest lock drift and profile-scope checker | ✓ VERIFIED | Exists; 269 lines (>=180); lock/profile diff report implemented. |
| `xtask/src/oracle_update.rs` | Oracle/helper/OOM gate runners | ✓ VERIFIED | Exists; 335 lines (>=220); per-profile parity + OOM tests wired. |
| `ci/oracle-compare.yml` | Non-fail-fast oracle/profile template | ✓ VERIFIED | Exists; 82 lines (>=70); required matrix + artifact upload steps present. |
| `ci/feature-matrix.yml` | Manifest/OOM matrix template | ✓ VERIFIED | Exists; 77 lines (>=70); required matrix + non-fail-fast execution present. |
| `.github/workflows/compat-governance-pr.yml` | Merge-blocking PR governance workflow | ✓ VERIFIED | Exists; 209 lines (>=110); required gate jobs wired to xtask commands. |
| `benches/micro_families.rs` | Criterion micro suite | ✓ VERIFIED | Exists; 169 lines (>=120); emits structured summary rows. |
| `benches/macro_molecules.rs` | Criterion macro suite | ✓ VERIFIED | Exists; 167 lines (>=120); emits structured summary rows. |
| `benches/crossover_cpu_gpu.rs` | Criterion crossover suite | ✓ VERIFIED | Exists; 173 lines (>=120); includes crossover metrics output. |
| `xtask/src/bench_report.rs` | Threshold-aware bench + diagnostics reporter | ✓ VERIFIED | Exists; 755 lines (>=240); enforce/calibration modes + fallback artifact writing present. |
| `ci/benchmark-thresholds.json` | Versioned threshold config | ✓ VERIFIED | Exists; 29 lines (>=20); micro/macro/crossover thresholds defined. |
| `ci/gpu-bench.yml` | GPU bench policy template with artifact validation | ✓ VERIFIED | Exists; 114 lines (>=90 and >=80); GPU runner + artifact contract checks present. |
| `.github/workflows/compat-governance-release.yml` | Required release GPU gate + artifact checks | ✗ STUB | Exists but 168 lines (<180 min_lines in `06-PLAN.md`). Functional wiring exists, but declared substance gate fails. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `crates/cintx-oracle/src/fixtures.rs` | `crates/cintx-ops/generated/compiled_manifest.lock.json` | lock profile/stability filtering | WIRED | Pattern match for `compiled_manifest.lock.json`, `with-f12+with-4c1e`, `stability`. |
| `crates/cintx-oracle/src/compare.rs` | `crates/cintx-compat/src/raw.rs` | raw-vs-legacy parity | WIRED | `raw::eval_raw` and `eval_legacy_symbol` present. |
| `crates/cintx-oracle/src/compare.rs` | `crates/cintx-oracle/src/fixtures.rs` | persisted mismatch report path metadata | WIRED | `write_pretty_json_artifact`, `required_path`, `mismatch_count` present. |
| `xtask/src/manifest_audit.rs` | `crates/cintx-ops/generated/compiled_manifest.lock.json` | lock/profile/symbol audit | WIRED | Lock constant and profile scope checks present. |
| `xtask/src/oracle_update.rs` | `crates/cintx-oracle/src/compare.rs` | profile-aware oracle + helper checks | WIRED | `generate_profile_parity_report` and `verify_helper_surface_coverage` invoked. |
| `xtask/src/oracle_update.rs` | `crates/cintx-compat/src/raw.rs` | OOM failure-path regression gate | WIRED | Runs `raw::tests::memory_limit_failure_keeps_output_slice_unchanged`. |
| `.github/workflows/compat-governance-pr.yml` | `xtask/src/main.rs` | PR xtask gate execution | WIRED | Direct calls to `manifest-audit`, `oracle-compare`, `helper-legacy-parity`, `oom-contract-check`. |
| `ci/oracle-compare.yml` | `crates/cintx-oracle/src/compare.rs` | profile-aware oracle compare template command | WIRED | `oracle-compare --include-unstable-source false` present. |
| `ci/feature-matrix.yml` | `xtask/src/manifest_audit.rs` | compiled-lock drift enforcement command | WIRED | `manifest-audit` and required profile matrix entries present. |
| `xtask/src/bench_report.rs` | `crates/cintx-runtime/src/metrics.rs` | diagnostics field contract | WIRED | `chunk_count`, `fallback_reason`, `transfer_bytes`, `not0` emitted. |
| `.github/workflows/compat-governance-pr.yml` | `ci/gpu-bench.yml` | advisory GPU policy alignment | WIRED | `gpu_bench_advisory` + `continue-on-error: true` present. |
| `.github/workflows/compat-governance-release.yml` | `ci/gpu-bench.yml` | required GPU policy alignment | WIRED | `gpu_bench_required` + `continue-on-error: false` + enforce bench-report command present. |
| `crates/cintx-oracle/src/lib.rs` | `crates/cintx-oracle/src/fixtures.rs` | crate-root fixture re-exports | WIRED | Re-exports include `build_profile_representation_matrix`, `build_required_profile_matrices`, `PHASE4_APPROVED_PROFILES`, `PHASE4_ORACLE_FAMILIES`. |
| `crates/cintx-oracle/src/lib.rs` | `crates/cintx-oracle/src/compare.rs` | crate-root parity/tolerance/helper re-exports | WIRED | Re-exports include `generate_profile_parity_report`, `generate_phase2_parity_report`, `verify_helper_surface_coverage`, `tolerance_for_family`. |
| `.github/workflows/compat-governance-release.yml` | `/mnt/data/cintx_phase_04_runtime_diagnostics.json` | required/fallback artifact validation | WIRED | `Validate bench artifact contract` checks required + fallback diagnostics/report paths. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `VERI-01` | `01-PLAN.md`, `05-PLAN.md` | Oracle comparisons with family tolerances and profile coverage | ✓ SATISFIED | Profile-aware fixture/parity implementation and explicit tolerance table in oracle modules; crate-root export surface present. |
| `VERI-02` | `02-PLAN.md`, `03-PLAN.md`, `05-PLAN.md`, `06-PLAN.md` | CI blocks manifest drift/helper-legacy/CubeCL consistency/OOM regressions | ? NEEDS HUMAN | Static wiring is present in xtask + workflows; live branch-protection/Actions execution still requires remote verification. |
| `VERI-03` | `04-PLAN.md`, `05-PLAN.md`, `06-PLAN.md` | Benchmark trend tracking for throughput/memory/crossover | ? NEEDS HUMAN | Bench suites, thresholds, and report pipeline are implemented; real trend validation depends on executed benchmark runs. |
| `VERI-04` | `04-PLAN.md`, `05-PLAN.md`, `06-PLAN.md` | Structured diagnostics for planner/chunk/fallback/transfer/OOM behavior | ? NEEDS HUMAN | Diagnostics fields and artifact contract exist in code/workflows; runtime validation requires live CI execution artifacts. |
| `ORPHANED` | n/a | Phase 4 requirement IDs in REQUIREMENTS.md not referenced by any Phase 4 plan frontmatter | ✓ SATISFIED | None. `VERI-01`, `VERI-02`, `VERI-03`, and `VERI-04` are all claimed by at least one plan. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `crates/cintx-oracle/src/compare.rs` | 698, 823 | Regex false-positive on format strings containing `{}` | ℹ️ Info | No stub behavior; no TODO/FIXME/placeholder/empty-implementation anti-patterns found in phase files scanned. |

### Human Verification Required

### 1. PR Branch Protection Gate Behavior

**Test:** Run `.github/workflows/compat-governance-pr.yml` on a real PR with branch protection.
**Expected:** `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, and `oom_contract_gate` are merge-blocking; `gpu_bench_advisory` is non-blocking.
**Why human:** Requires live GitHub branch protection and Actions state.

### 2. Release/Scheduled GPU Gate Execution

**Test:** Run `.github/workflows/compat-governance-release.yml` on a GPU-capable runner.
**Expected:** `gpu_bench_required` blocks on failure and emits bench/diagnostics artifacts via required/fallback paths.
**Why human:** Requires external GPU runner environment and real Actions execution.

### Gaps Summary

Phase 4 implementation is mostly wired and substantive, but one declared must-have artifact gate is not met: `.github/workflows/compat-governance-release.yml` is 168 lines while `06-PLAN.md` requires `min_lines: 180`. This leaves one must-have unresolved, so the phase cannot be marked fully achieved yet under the current plan contract.

---

_Verified: 2026-03-29T00:38:49Z_
_Verifier: Claude (gsd-verifier)_
