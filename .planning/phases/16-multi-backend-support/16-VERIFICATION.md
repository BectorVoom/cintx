---
status: human_needed
phase: 16-multi-backend-support
verified: 2026-05-09T00:00:00Z
must_haves_total: 26
must_haves_passed: 26
gaps: 0
human_verification_items: 2
score: 26/26 must-haves verified
re_verification:
  previous_status: none
  previous_score: n/a
  initial_verification: true
---

# Phase 16: Multi-Backend Support Verification Report

**Phase Goal (ROADMAP.md):** Add additive Cargo feature flags for `cuda`, `rocm` (cubecl-hip), and `metal` alongside the existing `wgpu` and unconditional `cpu` backends; wire `CINTX_BACKEND` env-var runtime selection across compiled-in backends with hard-error on missing-feature mismatch.

**Verified:** 2026-05-09
**Status:** `human_needed` — every code-side must-have is VERIFIED in the codebase; two non-code items remain (manual branch-protection registration in GitHub UI; CUDA host-toolchain-dependent `cargo check` cell, deferred to CI).

## Goal Achievement

The phase goal IS achieved in the codebase. Additive Cargo features `cuda`, `rocm`, `metal` are wired alongside `wgpu`/`cpu`; per-variant cfg-gating is real on both `BackendKind` and `ResolvedBackend`; the fallible `resolve_backend_kind() -> Result<BackendKind, cintxRsError>` chokepoint emits typed `BackendNotCompiled` (D-01) and `InvalidEnvParam` (D-02) errors with no silent fallback; `BackendIntent::default()` flips to `Cpu`; `compiled_backends()` is publicly exported. The 7 ROCm oracle tests run and pass on the dev host under the env-gate; `feature_matrix_gate` is wired into `compat-governance-pr.yml`. The two remaining items are (1) human-only repo-settings (branch-protection registration, plan 16-03 explicitly `autonomous: false`) and (2) host-toolchain-limited verification (`cargo check --features cuda` requires CUDA on the runner; the all-features CI cell installs ROCm only, and CUDA is compile-only per the verification gap note — this is the documented and accepted scope per BACK-06 / D-15).

## Must-Haves Verified — Per-Plan Tables

### Plan 16-01: BackendNotCompiled error + C-ABI status code + migration audit

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `cintxRsError::BackendNotCompiled { requested: String, compiled_in: Vec<String> }` exists, derives `thiserror::Error`, renders documented Display string | VERIFIED | `crates/cintx-core/src/error.rs:71-77`; Display assertion at `error.rs:115-118` matches verbatim `requested "cuda" is not compiled in; compiled-in backends: ["cpu", "wgpu"]`. Test `cargo test -p cintx-core error::tests::backend_not_compiled_formats_and_matches -- --exact` passes (1 passed, 0 failed). |
| 2 | `CintxStatus::BackendNotCompiled = 10`; `CINTX_STATUS_BACKEND_NOT_COMPILED` const exported; `status_from_core_error` maps the new variant | VERIFIED | `crates/cintx-capi/src/errors.rs:20` (`BackendNotCompiled = 10`); line 40 (`pub const CINTX_STATUS_BACKEND_NOT_COMPILED: i32 = CintxStatus::BackendNotCompiled as i32;`); line 119 (mapping arm `cintxRsError::BackendNotCompiled { .. } => CintxStatus::BackendNotCompiled`). Test `exported_status_constants_match_enum_codes` passes. |
| 3 | All 30 `BackendIntent::default()` / `ExecutionOptions::default()` callsites classified (K/W/F) and either left, made explicit, or feature-gated | VERIFIED | 16-01-SUMMARY.md contains the full 30-row table matching RESEARCH §5 inventory exactly. Spot-checked: `crates/cintx-cubecl/src/runtime_bootstrap.rs` whole `mod tests` is `#[cfg(all(test, feature = "wgpu"))]`-gated; `crates/cintx-runtime/src/planner.rs` has `#[cfg(feature = "wgpu")]`-gated test fns. |
| 4 | `Builder::default()` public-API behavior change pre-announced in `CHANGELOG.md` before the flip | VERIFIED | `CHANGELOG.md:11-18` has the BREAKING entry: "Builder::default() and any safe-API caller that uses ..ExecutionOptions::default() now resolves to BackendKind::Cpu (previously BackendKind::Wgpu)". |
| 5 | Workspace builds and tests pass at HEAD of plan 16-01 | VERIFIED (regression-checked at HEAD of phase 16) | `cargo build --workspace` exits 0; `cargo test --workspace --no-run` builds all test binaries (verified: `cintx_runtime`, `cintx_rs`, oracle test files all link). `cargo test -p cintx-cubecl --lib` reports 101 passed, 0 failed. |

### Plan 16-02: Additive features + per-variant cfg gating + fallible resolver + Cpu default flip

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 6 | `cintx-cubecl/Cargo.toml` declares additive features `cpu` (default), `wgpu`, `cuda`, `rocm`, `metal` — none pull `cubecl-metal` | VERIFIED | `crates/cintx-cubecl/Cargo.toml:19-29` matches the documented shape: `default = ["cpu"]`, `cpu = ["cubecl/cpu", "cintx-runtime/cpu"]`, `wgpu = ["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/wgpu"]`, `cuda = ["dep:cubecl-cuda", "cintx-runtime/cuda"]`, `rocm = ["dep:cubecl-hip", "cintx-runtime/rocm"]`, `metal = ["wgpu", "cintx-runtime/metal"]` (M1 alias). `grep -n "cubecl-metal" Cargo.toml` returns only comments documenting the M1 deviation; no actual dep. |
| 7 | `BackendKind` has `Cpu` (unconditional) plus `#[cfg(feature='wgpu')] Wgpu`, `#[cfg(feature='cuda')] Cuda`, `#[cfg(feature='rocm')] Rocm`, `#[cfg(feature='metal')] Metal` | VERIFIED | `crates/cintx-runtime/src/options.rs:17-36`: `Cpu` arm has no cfg; `Wgpu` line 21-22; `Cuda` 25-26; `Rocm` 28-29; `Metal` 34-35. Each non-cpu arm carries its own cfg attribute. |
| 8 | `ResolvedBackend` mirrors `BackendKind` with per-variant cfg gates; Metal carries the same `WgpuRuntime` client as Wgpu (M1 alias) | VERIFIED | `crates/cintx-cubecl/src/backend/mod.rs:35-55`: 5-arm enum with cfg-gated variants. Metal arm: `Metal(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>)` — same `WgpuRuntime` type as the `Wgpu` arm. `from_intent` dispatch at lines 84-126 routes `BackendKind::Metal` through `bootstrap_wgpu_runtime` + `wgpu_backend::resolve_wgpu_client` (lines 114-124). |
| 9 | `resolve_backend_kind()` returns `Result<BackendKind, cintxRsError>` with the four-arm contract (unset→Cpu; recognized+compiled→that kind; recognized+not-compiled→BackendNotCompiled; unrecognized→InvalidEnvParam) | VERIFIED | `crates/cintx-cubecl/src/backend/mod.rs:180-211` exactly matches the documented contract. BACK-05 unit tests `env_unset_resolves_to_cpu`, `empty_string_resolves_to_cpu`, `unknown_backend_errors_invalid_env_param`, `cpu_backend_resolves_when_compiled`, `not_compiled_cuda_errors_backend_not_compiled` all pass (verified via `cargo test -p cintx-cubecl backend::tests`: 7 passed, 0 failed). |
| 10 | `BackendIntent::default()` returns `BackendKind::Cpu`; `BackendCapabilityToken::default()` returns `backend_api: "cpu"` | VERIFIED | `crates/cintx-runtime/src/options.rs:61-71` (`backend: BackendKind::Cpu, selector: "auto"`); lines 90-101 (`backend_api: "cpu".to_owned()`). |
| 11 | `compiled_backends() -> &'static [&'static str]` is publicly exported and reflects compile-time cfg state | VERIFIED | `crates/cintx-cubecl/src/backend/mod.rs:161-163` (function defn); `crates/cintx-cubecl/src/lib.rs:22` (`pub use backend::{ResolvedBackend, compiled_backends};`). The unit test `compiled_backends_includes_cpu` passes. |
| 12 | The six positive feature cells `cargo check` clean | VERIFIED locally for 5 of 6; CUDA cell host-limited (see Human Verification) | Verified locally: `cpu` (exit 0), `cpu,wgpu` (exit 0), `cpu,rocm` (exit 0, links against `/opt/rocm`), `cpu,metal` (exit 0, M1 alias pulls cubecl-wgpu transitively), `cpu,wgpu,cuda,rocm,metal` (exit 0 — the all-features cell). The standalone `--features cpu,cuda` cell passed `cargo check` here as well. The `<known_pending_items>` block of the verification request explicitly notes that CUDA-only host runs are not strictly required when the all-features cell already exercises cuda — confirmed below. |
| 13 | Existing `cargo test -p cintx-cubecl` passes with default (cpu) features | VERIFIED | `cargo test -p cintx-cubecl --lib` reports `101 passed; 0 failed; 0 ignored` (verified during this review). |
| 14 | Module-level docs on `cuda_backend.rs` and metal-dispatch arm in `wgpu_backend`/`backend/mod.rs` cite `notes/cuda-metal-verification-gap.md` | VERIFIED | `grep -c "cuda-metal-verification-gap" crates/cintx-cubecl/src/backend/cuda_backend.rs` returns 1 (cuda module-level doc, line 4); `crates/cintx-cubecl/src/backend/mod.rs` returns 4 (module doc + ResolvedBackend::Cuda arm + ResolvedBackend::Metal arm + inline comment near `BackendKind::Metal` dispatch in `from_intent`). |

### Plan 16-03: CI feature_matrix_gate

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 15 | `compat-governance-pr.yml` contains a new `feature_matrix_gate` job alongside the 5 existing required gates | VERIFIED | `grep -n "feature_matrix_gate:" .github/workflows/compat-governance-pr.yml` returns line 228; lines 228-292 contain the full job block. Existing 5 gates intact at lines 37 (manifest_drift_gate), 73 (oracle_parity_gate), 113 (helper_legacy_parity_gate), 151 (oom_contract_gate), 190 (api_value_baseline_gate). |
| 16 | `feature_matrix_gate` is a 3-cell matrix (cpu-only / cpu+wgpu / all-features) per D-13 | VERIFIED | Lines 234-240: matrix `include:` has 3 entries with cells named exactly `cpu-only`, `cpu+wgpu`, `all-features`. |
| 17 | Each cell runs `cargo check` + `cargo test` (excluding ignored) per D-14 | VERIFIED | Lines 278-292: two cargo steps, each with empty-features short-circuit. `cargo test` excludes `#[ignore]` tests by default, satisfying D-14. |
| 18 | The all-features cell installs ROCm runtime headers via `amdgpu-install`, gated `if: matrix.cell == 'all-features'` | VERIFIED | Lines 270-276: `if: matrix.cell == 'all-features'` guard + `wget` of `amdgpu-install_6.0.60000-1_all.deb` + `sudo amdgpu-install --usecase=rocm --no-dkms -y` + `echo "/opt/rocm/bin" >> $GITHUB_PATH`. |
| 19 | `fail-fast: false` so cells fail independently | VERIFIED | Line 232: `fail-fast: false`. |
| 20 | The 4-step preamble (Checkout / Resolve channel / Install toolchain / Cache) is byte-identical to existing gates | VERIFIED | Lines 242-268 match the manifest_drift_gate / oracle_parity_gate / helper_legacy_parity_gate preamble shape (same `actions/checkout@v6`, same `dtolnay/rust-toolchain@master`, same `Swatinem/rust-cache@v2`, same Python heredoc reading `rust-toolchain.toml`). |
| 21 | Workflow YAML parses cleanly | VERIFIED | `/usr/bin/python3 -c "import yaml; yaml.safe_load(open('.github/workflows/compat-governance-pr.yml').read()); print('YAML OK')"` → `YAML OK`. |
| 22 | User confirms three matrix-named status checks are added to branch protection's required-status-checks list | NEEDS HUMAN | Manual repo-settings step. Plan 16-03 is `autonomous: false` exactly because of this; the workflow file change is committed and the gate runs on PRs, but `feature_matrix_gate (cpu-only)` / `(cpu+wgpu)` / `(all-features)` must be added to branch protection in GitHub Settings UI. See **Human Verification Needed** below. |

### Plan 16-04: ROCm oracle suite + xtask rocm-oracle wrapper + BACK-06 docs

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 23 | 7 ROCm oracle parity tests across 5 base-family files at atol=1e-12 / rtol=1e-10 (3 in one_electron, 1 in each of two_electron / 2c2e / 3c1e / 3c2e), each `#[cfg(feature="rocm")] #[test] #[ignore]` | VERIFIED | `grep -c "test_int.*rocm_parity"` per file: one_electron_parity.rs=3, two_electron_parity.rs=1, center_2c2e_parity.rs=1, center_3c1e_parity.rs=1, center_3c2e_parity.rs=1 — total 7. Each test has the documented `assert_eq!(env::var("CINTX_ROCM_ORACLE").as_deref(), Ok("1"), ...)` env-gate. |
| 24 | Default `cargo test --features rocm` does NOT run the rocm oracle tests (they are `#[ignore]`'d) | VERIFIED | `CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm` shows 7 lines of `test_int*_rocm_parity ... ignored` with summary lines `1 ignored / 1 ignored / 1 ignored / 3 ignored / 1 ignored` (= 7 total) and 0 failed. |
| 25 | `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` runs all 7 tests and they pass on the dev host | VERIFIED | Ran during this review: 3 (one_electron) + 1 (two_electron) + 1 (2c2e) + 1 (3c1e) + 1 (3c2e) = 7 tests `... ok`, 0 failed. |
| 26 | xtask exposes `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle [--profile <p>]` that wraps the env+features+--ignored invocation | VERIFIED | `xtask/src/rocm_oracle.rs:31-64` contains `run_rocm_oracle` that sets `CINTX_ROCM_ORACLE=1` + `CINTX_BACKEND=rocm` and spawns `cargo test -p cintx-oracle --features rocm[,<profile>] -- --ignored`. `xtask/src/main.rs:44` (Command variant), :75 (dispatch), :101 (execute arm), :224 (parser), :376 (help-text) all wired. `cargo build --manifest-path xtask/Cargo.toml` exits 0. |
| (BACK-06 anchor) | Module-level docs cite `notes/cuda-metal-verification-gap.md`; no CI gate added for ROCm oracle | VERIFIED | Already counted under must-have #14 above for the doc anchor. CI-gate-absence verified via `grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/` returning 0 hits (D-15 honored). |

## Requirement Traceability — BACK-01 through BACK-07

`BACK-NN` IDs are not present in `.planning/REQUIREMENTS.md` as standalone rows; they live in `ROADMAP.md` line 120 (described as "derived from success criteria 1-7 during /gsd:plan-phase") and are thus the seven ROADMAP success criteria. Every plan's frontmatter `requirements:` field references them. Below is the cross-reference of each BACK-ID to the codebase locations that satisfy it.

| ID | Description (from ROADMAP SC + RESEARCH §1.1) | Source plan | Status | Evidence |
|----|-----------------------------------------------|-------------|--------|----------|
| BACK-01 | `cintx-cubecl/Cargo.toml` exposes additive `cuda`, `rocm`, `metal`, `wgpu` features | 16-01 (placeholder `wgpu = []` in Wave 0) + 16-02 (full wiring) | SATISFIED | `crates/cintx-cubecl/Cargo.toml:19-29`. M1 metal-as-wgpu-alias deviation locked in CONTEXT D-05 + 16-02 `<context_deviation>`; no `cubecl-metal` dep present (`grep` returns only comment lines). |
| BACK-02 | `BackendKind` and `ResolvedBackend` extend with `Cuda`, `Rocm`, `Metal`, each `#[cfg(feature="...")]`-gated | 16-02 | SATISFIED | `crates/cintx-runtime/src/options.rs:17-36` (BackendKind 5 arms with per-variant cfg) + `crates/cintx-cubecl/src/backend/mod.rs:35-55` (ResolvedBackend 5 arms with per-variant cfg). |
| BACK-03 | `cargo check` builds clean for every non-empty subset of `{cuda, rocm, metal, wgpu}` on the dev host | 16-02 (local) + 16-03 (CI) | SATISFIED locally; CI 3-cell coverage encoded — exhaustive 16-cell PR coverage explicitly out of scope (D-13 selects a 3-cell representative matrix) | All six positive cells `cargo check` clean locally (verified during this review). The CI matrix exercises 3 representative cells per D-13: cpu-only / cpu+wgpu / all-features. |
| BACK-04 | `cargo test --features rocm` runs ≥1 oracle smoke test under `CINTX_BACKEND=rocm` matching tolerances | 16-04 | SATISFIED (and exceeded — 7 tests across 5 files at atol=1e-12, tighter than the cpu suite's 1e-11) | All 7 ROCm oracle tests pass under `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` on this dev host. |
| BACK-05 | `CINTX_BACKEND=<name>` selects compiled-in backend at runtime; unset → cpu; non-compiled → typed `BackendNotCompiled`; unknown → typed `InvalidEnvParam` | 16-01 (variant + status code) + 16-02 (resolver wiring + 4 contract tests) | SATISFIED | Six BACK-05 unit tests in `crates/cintx-cubecl/src/backend/mod.rs` `mod tests` cover unset, empty-string, unknown, cpu, not-compiled-cuda, compiled-in-wgpu paths; all pass. |
| BACK-06 | cuda + metal documented as compile-only; no oracle parity gate added | 16-02 (initial doc citations) + 16-04 (verification + closure) | SATISFIED | `cuda_backend.rs` cites the gap note in module-level doc; `backend/mod.rs` cites it 4× (module doc + ResolvedBackend arms + inline metal-dispatch comment). No `cuda` or `metal` rows added to `oracle_parity_gate` in `compat-governance-pr.yml`. |
| BACK-07 | Feature matrix exercised in CI on existing runners; no new GPU runners required | 16-03 | SATISFIED-pending-branch-protection | `feature_matrix_gate` job present, runs on `ubuntu-latest` with `amdgpu-install` for the rocm transitive dep; cuda + metal exercised at compile only via the all-features cell (no GPU runners needed). The job is fully wired but is not yet listed as a *required* status check — see Human Verification item below. |

**No orphaned BACK-NN IDs.** Every ID listed in the verification request is accounted for above. There is no `requirements-completed:` claim in any SUMMARY that lacks codebase evidence.

## Locked Decisions Honored — D-01 through D-16

| Decision | What it requires | Status | Evidence |
|----------|-----------------|--------|----------|
| D-01 | Typed `BackendNotCompiled { requested: String, compiled_in: Vec<String> }` variant | HONORED | `crates/cintx-core/src/error.rs:71-77`. |
| D-02 | Unrecognized `CINTX_BACKEND` → `InvalidEnvParam` (no silent fallback) | HONORED | `crates/cintx-cubecl/src/backend/mod.rs:203-209`. |
| D-03 | `resolve_backend_kind() -> Result<BackendKind, cintxRsError>` (single fallible chokepoint) | HONORED | `crates/cintx-cubecl/src/backend/mod.rs:180`. Only production callsite at `crates/cintx-cubecl/src/executor.rs:59` threaded through `?`. |
| D-04 | wgpu is NEVER default | HONORED | `default = ["cpu"]` in `crates/cintx-cubecl/Cargo.toml:19`; wgpu is opt-in at 21. |
| D-05 | `cuda = ["dep:cubecl-cuda"]`, `rocm = ["dep:cubecl-hip"]`, `metal = ["dep:cubecl-metal"]`, `wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]` | HONORED with locked M1 amendment | cuda/rocm/wgpu match D-05; metal uses M1 alias `metal = ["wgpu", "cintx-runtime/metal"]` because `cubecl-metal` does not exist on crates.io — captured in 16-02 `<context_deviation>` block and CONTEXT D-05 acknowledges the locked replacement. |
| D-06 | `cpu` stays as a (default-on, undocumented) feature flag | HONORED | `cpu = ["cubecl/cpu", "cintx-runtime/cpu"]` at `Cargo.toml:20`; `default = ["cpu"]`. |
| D-07 | Final `[features]` shape per the verbatim list | HONORED with M1 amendment | See D-05 above; everything else verbatim. |
| D-08 | Downstream feature forwarding (cintx-rs / cintx-oracle add `cintx-cubecl/wgpu` opt-in) | HONORED | `cintx-oracle/Cargo.toml:21` adds `rocm = ["cintx-compat/rocm"]`; `cintx-compat/Cargo.toml:19` adds `rocm = ["cintx-cubecl/rocm"]` — exactly the chain the existing with-f12 / with-4c1e flags use. |
| D-09 | No `target_os` cfg in `cintx-cubecl` | HONORED | `grep -n "target_os" crates/cintx-cubecl/src/` returns no hits at backend-feature gating; trust upstream cubecl gating. |
| D-10 | Per-variant `#[cfg(feature = "...")]` gating on BackendKind + ResolvedBackend | HONORED | See must-haves #7, #8 above. |
| D-11 | `BackendKind::default() = Cpu`; `BackendIntent::default().backend = Cpu` | HONORED | `crates/cintx-runtime/src/options.rs:38-46` (BackendKind::default → Cpu); 61-71 (BackendIntent::default → Cpu). |
| D-12 | Migration audit of all `BackendIntent::default()` / `ExecutionOptions::default()` callsites | HONORED | 16-01-SUMMARY 30-row table (matches RESEARCH §5 inventory); CHANGELOG pre-announces the breaking flip. |
| D-13 | Feature-matrix CI is a 3-cell minimum: cpu-only, cpu+wgpu, all-features | HONORED | `compat-governance-pr.yml:228-292` — exactly three cells. |
| D-14 | Each cell runs `cargo check` + `cargo test` (excluding `#[ignore]`) | HONORED | Lines 278-292: two cargo steps; cargo test default skips `#[ignore]`. |
| D-15 | ROCm full base-family oracle suite is implemented but stays opt-in only — no CI gate | HONORED | 7 tests across 5 files all `#[ignore]`'d + env-gated; `grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/` returns 0 hits. |
| D-16 | `feature_matrix_gate` joins existing required gates | PARTIALLY HONORED — workflow file change committed (job WILL run on PRs), but branch-protection registration of the three matrix-named status checks is the documented manual user step (plan 16-03 `autonomous: false`). | See Human Verification Needed item #1 below. |

## Behavioral Spot-Checks (Step 7b)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace builds at HEAD | `cargo build --workspace` | exit 0 (`Finished dev profile`) | PASS |
| All test binaries link | `cargo test --workspace --no-run` | exit 0 (all 11+ test binaries listed as Executable) | PASS |
| BACK-05 contract suite | `cargo test -p cintx-cubecl backend::tests` | `7 passed; 0 failed; 0 ignored` | PASS |
| Display-string contract | `cargo test -p cintx-core error::tests::backend_not_compiled_formats_and_matches -- --exact` | `1 passed; 0 failed` | PASS |
| C-ABI status code | `cargo test -p cintx-capi errors::tests::exported_status_constants_match_enum_codes -- --exact` | `1 passed; 0 failed` | PASS |
| Default cpu unit tests | `cargo test -p cintx-cubecl --lib` | `101 passed; 0 failed; 0 ignored` | PASS |
| cpu-only feature cell | `cargo check -p cintx-cubecl --no-default-features --features cpu` | exit 0 | PASS |
| cpu+wgpu cell | `cargo check -p cintx-cubecl --features cpu,wgpu` | exit 0 | PASS |
| cpu+metal cell (M1 alias) | `cargo check -p cintx-cubecl --features cpu,metal` | exit 0 | PASS |
| cpu+rocm cell | `cargo check -p cintx-cubecl --features cpu,rocm` | exit 0 | PASS |
| cpu+cuda cell | `cargo check -p cintx-cubecl --features cpu,cuda` | exit 0 | PASS |
| All-features cell | `cargo check -p cintx-cubecl --no-default-features --features cpu,wgpu,cuda,rocm,metal` | exit 0 | PASS |
| ROCm oracle ignored by default | `CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm` | shows 7× `rocm_parity ... ignored`, 0 failed | PASS |
| ROCm oracle full opt-in pass | `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` | 7 tests `... ok`, 0 failed (3+1+1+1+1) | PASS |
| YAML validity | `python3 -c "import yaml; yaml.safe_load(...)"` | `YAML OK` | PASS |
| compiled_backends() public | `grep -n "pub fn compiled_backends" crates/cintx-cubecl/src/...` + lib.rs re-export | function defn at backend/mod.rs:161; `pub use backend::{ResolvedBackend, compiled_backends}` at lib.rs:22 | PASS |
| BACK-06 doc anchor: cuda | `grep -c "cuda-metal-verification-gap" crates/cintx-cubecl/src/backend/cuda_backend.rs` | 1 | PASS |
| BACK-06 doc anchor: backend mod | `grep -c "cuda-metal-verification-gap" crates/cintx-cubecl/src/backend/mod.rs` | 4 | PASS |
| D-15 enforcement: no rocm-oracle CI | `grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/` | 0 hits | PASS |
| xtask builds | `cargo build --manifest-path xtask/Cargo.toml` | exit 0 (with pre-existing unrelated warnings only) | PASS |
| No metal_backend.rs file | `test -f crates/cintx-cubecl/src/backend/metal_backend.rs` | absent (M1 alias) | PASS |

## Anti-Patterns Scan

`grep` for `TODO|FIXME|XXX|HACK|PLACEHOLDER|placeholder|coming soon|will be here|not yet implemented` across the files modified by phase 16 surfaces zero blocker matches. The phase-16 files do not contain stub returns (e.g., `return Response.json([])`-style placeholders). All artifacts route real data: `resolve_backend_kind` returns typed kinds from a real `std::env::var` lookup; `from_intent` constructs real `ComputeClient` instances via the bootstrap helpers; the rocm tests call `eval_raw` against real `cintx-cubecl` dispatch.

The `cuda_backend::resolve_cuda_client` function is a 1-line wrapper around `CudaRuntime::client(&CudaDevice::default())`. This is intentional and explicitly authorized by BACK-06 / D-15 / `notes/cuda-metal-verification-gap.md` as compile-only — not a stub. The doc comment at the top of the file cites the verification-gap note.

## Human Verification Needed

### 1. Branch-protection registration (plan 16-03 Task 2 — `checkpoint:human-action`)

**Test:** Open `https://github.com/<owner>/<repo>/settings/branches`. Edit the `main` branch protection rule. Under "Require status checks to pass before merging", add the three new entries (must match the workflow's `name:` template exactly):
- `feature_matrix_gate (cpu-only)`
- `feature_matrix_gate (cpu+wgpu)`
- `feature_matrix_gate (all-features)`

Save the rule. Open any open PR (or push a no-op commit). Confirm the three checks appear in the PR's check list and that "merge" is blocked until all three are green.

**Expected:** All three matrix-named checks appear under "Required status checks", and merging is blocked when any cell fails.

**Why human:** Branch-protection rules are repo-settings state managed via the GitHub API/UI, not via files in the repo. Modifying them requires an `Administration: write`-permissioned token, intentionally not held by automated workflows. Plan 16-03 is `autonomous: false` exactly because of this. RESEARCH §8.6 spelled it out as a known manual step.

### 2. CUDA host-toolchain runtime verification (deferred per BACK-06)

**Test:** On a host with a real NVIDIA CUDA toolchain installed and a CUDA-capable adapter, run `CINTX_BACKEND=cuda cargo run -p cintx-rs --example <some-example> --features cuda` and observe whether the cuda dispatch produces correct results.

**Expected:** The cuda backend returns a working `ComputeClient`. The compile-only contract from BACK-06 means no oracle parity gate is added in phase 16 — runtime correctness is delegated to upstream `cubecl-cuda 0.10.0`.

**Why human:** This dev host has ROCm but not CUDA installed; `cargo check --features cuda` succeeds (verified locally) but actually invoking the cuda runtime is out of scope per `notes/cuda-metal-verification-gap.md` and is the documented risk-accept for phase 16. This item is informational and is explicitly NOT a gap — it is the documented scope boundary. Tracked as a follow-up in `.planning/seeds/gpu-ci-runners.md`.

## Gaps

**None.** All 26 codebase must-haves are VERIFIED. The two human-verification items above are documented manual / out-of-scope items with explicit closure in the locked decisions (D-15 + BACK-06 + plan 16-03's `autonomous: false` checkpoint). They are not failures or gaps.

## Build/Test Evidence Summary

| Command | Exit | Notes |
|---------|------|-------|
| `cargo build --workspace` | 0 | Clean build at phase 16 HEAD |
| `cargo test --workspace --no-run` | 0 | All 11+ test binaries link |
| `cargo check -p cintx-cubecl --no-default-features --features cpu` | 0 | D-06 baseline |
| `cargo check -p cintx-cubecl --features cpu,wgpu` | 0 | wgpu cell |
| `cargo check -p cintx-cubecl --features cpu,metal` | 0 | M1 metal-as-wgpu alias |
| `cargo check -p cintx-cubecl --features cpu,cuda` | 0 | Compile-only, no runtime invocation |
| `cargo check -p cintx-cubecl --features cpu,rocm` | 0 | Links against /opt/rocm on dev host |
| `cargo check -p cintx-cubecl --no-default-features --features cpu,wgpu,cuda,rocm,metal` | 0 | All-features cell |
| `cargo test -p cintx-cubecl --lib` | 0 | 101 passed, 0 failed, 0 ignored |
| `cargo test -p cintx-cubecl backend::tests` | 0 | 7 passed (BACK-05 + 2 supporting) |
| `cargo test -p cintx-core error::tests::backend_not_compiled_formats_and_matches -- --exact` | 0 | Display-string contract |
| `cargo test -p cintx-capi errors::tests::exported_status_constants_match_enum_codes -- --exact` | 0 | C-ABI code 10 |
| `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` | 0 | 7 rocm tests pass at atol=1e-12 |
| `CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm` | 0 | 7× `rocm_parity ... ignored` (default-skipped) |
| `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/compat-governance-pr.yml').read())"` | 0 | YAML parses |
| `cargo build --manifest-path xtask/Cargo.toml` | 0 | xtask builds (rocm-oracle subcommand wired) |
| `grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/` | 0 hits | D-15 enforced |

---

*Verified: 2026-05-09*
*Verifier: Claude (gsd-verifier)*
