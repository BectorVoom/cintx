---
phase: 16-multi-backend-support
plan: 03
subsystem: ci-feature-matrix-gate
tags: [ci, github-actions, feature-matrix, rocm-install, branch-protection]

# Dependency graph
requires:
  - phase: 16-multi-backend-support
    plan: 02
    provides: cintx-cubecl/Cargo.toml additive [features] table (cpu/wgpu/cuda/rocm/metal); BackendKind 5-arm cfg gating; fallible resolve_backend_kind() chokepoint; six positive feature cells building clean locally
provides:
  - feature_matrix_gate job in .github/workflows/compat-governance-pr.yml — 3-cell matrix (cpu-only / cpu+wgpu / all-features) running cargo check + cargo test per cell on every PR
  - ROCm runtime install step (amdgpu-install, gated `if: matrix.cell == 'all-features'`) so cubecl-hip-sys's build script can find hipconfig on stock ubuntu-latest runners
  - Three new GitHub Actions status-check names that branch protection can require: `feature_matrix_gate (cpu-only)`, `feature_matrix_gate (cpu+wgpu)`, `feature_matrix_gate (all-features)`
affects:
  - 16-04-rocm-oracle-suite (runs locally only; not on this gate)
  - branch-protection-required-status-checks (manual user step; see "Manual User Step Required" below)

# Tech tracking
tech-stack:
  added:
    - "GitHub Actions matrix strategy with `include:` for explicitly named cells (cpu-only / cpu+wgpu / all-features) — extends existing oracle_parity_gate matrix shape from Phase 4"
    - "amdgpu-install 6.0.60000-1 deb installer for ROCm 6.0 runtime headers on ubuntu-latest (jammy archive)"
  patterns:
    - "Per-cell feature dispatch: `cargo check -p cintx-cubecl --features \"${{ matrix.features }}\"` with empty-string short-circuit for the default-features cell — same pattern as the per-profile dispatch in oracle_parity_gate"
    - "Conditional install step gated on a matrix cell name (`if: matrix.cell == 'all-features'`) — keeps cpu-only and cpu+wgpu cells fast (~3-5 min) while paying the ~5-7 min ROCm install cost only on the cell that needs it"
    - "4-step preamble (Checkout / Resolve pinned Rust channel / Install pinned Rust toolchain / Cache Rust artifacts) reused verbatim across all six required gates in compat-governance-pr.yml"

key-files:
  created: []
  modified:
    - .github/workflows/compat-governance-pr.yml

key-decisions:
  - "feature_matrix_gate is appended after api_value_baseline_gate (the natural sixth gate) and before the advisory jobs (gpu_bench_advisory, wgpu_capability_advisory, unstable_source_oracle). This places all six required gates first in file order, matching the existing 5-required-gate-then-advisory-jobs grouping."
  - "ROCm install step uses the planner's recommended `amdgpu-install` flow (RESEARCH §6.1) verbatim rather than the `apt-pin rocm-dev` fallback (RESEARCH §6.4). The fallback is a Wave-2 contingency only triggered if the smoke test on a feature branch shows the all-features cell flaking or exceeding 15 minutes — not a pre-emptive choice."
  - "Empty-string `features: \"\"` for the cpu-only cell, with shell-side `if [ -z ... ]` branching, rather than splitting cpu-only into a separate non-matrix job. Matches RESEARCH §6.1 exactly and keeps all three cells inside one matrix definition (so they stay in lockstep on toolchain pin, cache key, and preamble changes)."
  - "Branch-protection registration is a manual user step, not a workflow file edit. GitHub Actions auto-discovers new status checks on PR runs but does NOT auto-add them to the required-checks list — a repo admin must add `feature_matrix_gate (cpu-only)`, `feature_matrix_gate (cpu+wgpu)`, `feature_matrix_gate (all-features)` to the `main` branch protection rule. This is why the plan is `autonomous: false`."

requirements-completed: [BACK-03, BACK-07]

# Metrics
duration: ~10 min
completed: 2026-05-09
---

# Phase 16 Plan 03: Wave 2 — feature_matrix_gate CI job + ROCm install Summary

**`compat-governance-pr.yml` gains a sixth required gate `feature_matrix_gate` running a 3-cell matrix (cpu-only / cpu+wgpu / all-features) of `cargo check` + `cargo test` on every PR, with a cell-conditional `amdgpu-install` step so the all-features cell can build `cubecl-hip-sys` on stock `ubuntu-latest` runners.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-09 (sequential executor)
- **Completed:** 2026-05-09
- **Tasks:** 1 / 2 code task complete; Task 2 is the manual branch-protection step the user must perform post-merge
- **Files modified:** 1 (.github/workflows/compat-governance-pr.yml)

## Accomplishments

- `feature_matrix_gate` job appended to `.github/workflows/compat-governance-pr.yml` (lines 229-289), placed right after `api_value_baseline_gate` and before `gpu_bench_advisory`.
- 3-cell matrix per D-13: `cpu-only` (no extra features → exercises default `["cpu"]`), `cpu+wgpu` (`--features wgpu`), `all-features` (`--features wgpu,cuda,rocm,metal`). All three cells use `cargo check -p cintx-cubecl` + `cargo test -p cintx-cubecl` per D-14, exercising the exact `[features]` table Wave 1 (16-02) committed.
- `fail-fast: false` so each cell fails independently (per RESEARCH §6.3 and D-13).
- The 4-step preamble (Checkout / Resolve pinned Rust channel / Install pinned Rust toolchain / Cache Rust artifacts) is byte-identical to the preamble used by `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, `oom_contract_gate`, and `api_value_baseline_gate` — same `actions/checkout@v6`, same `dtolnay/rust-toolchain@master`, same `Swatinem/rust-cache@v2`, same Python heredoc reading `rust-toolchain.toml`.
- `if: matrix.cell == 'all-features'` step installs ROCm 6.0 runtime headers via `amdgpu-install` so `cubecl-hip-sys`'s build script can locate `hipconfig` on `/opt/rocm/bin` (added to `$GITHUB_PATH`). Step is positioned AFTER the cache step but BEFORE the cargo steps so hipconfig is on PATH before the build script runs.
- Cargo invocations use shell-side `if [ -z "${{ matrix.features }}" ]` so the cpu-only cell calls `cargo check -p cintx-cubecl` (no `--features` flag, exercising the `default = ["cpu"]` baseline) while the other two cells pass `--features "wgpu"` / `--features "wgpu,cuda,rocm,metal"`.

## Task Commits

1. **Task 1: Add feature_matrix_gate job + ROCm install step to compat-governance-pr.yml** — committed by this plan; see git log for hash.
2. **Task 2: Manual — register the three feature_matrix_gate matrix entries in branch protection** — NOT a code change; user/repo-admin action required (see "Manual User Step Required" below). This is why the plan is `autonomous: false`.

## Files Created/Modified

- `.github/workflows/compat-governance-pr.yml` — appended a new `feature_matrix_gate` job (~62 lines) after `api_value_baseline_gate`. No other gates touched. No new files created.

## YAML Parse Confirmation

Validated via `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/compat-governance-pr.yml').read()); print('YAML OK')"` using system PyYAML 6.0.1. Output:

```
YAML OK
```

Spot-check counters (all expected values):

| Marker | Expected | Actual |
|--------|----------|--------|
| `feature_matrix_gate` occurrences | ≥ 2 (job key + name) | 2 |
| `all-features` occurrences | ≥ 2 (matrix entry + `if:` guard) | 2 |
| `fail-fast: false` count | 2 (oracle_parity_gate + feature_matrix_gate) | 2 |
| `matrix.cell == 'all-features'` guard | 1 (line 271) | 1 |
| `cell:` matrix entries | 3 (cpu-only / cpu+wgpu / all-features) | 3 |

## Verbatim Job Block (committed)

```yaml
feature_matrix_gate:
    name: feature_matrix_gate (${{ matrix.cell }})
    runs-on: ubuntu-latest
    strategy:
        fail-fast: false
        matrix:
            include:
                - cell: cpu-only
                  features: ""
                - cell: cpu+wgpu
                  features: "wgpu"
                - cell: all-features
                  features: "wgpu,cuda,rocm,metal"
    steps:
        - name: Checkout
          uses: actions/checkout@v6

        - name: Resolve pinned Rust channel
          id: rust
          run: |
              python <<'PY'
              import os
              import tomllib
              from pathlib import Path

              data = tomllib.loads(Path("rust-toolchain.toml").read_text())
              toolchain = data.get("toolchain", {})
              channel = toolchain.get("channel")
              if not channel:
                  raise SystemExit("failed to resolve channel from rust-toolchain.toml")
              with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                  fh.write(f"channel={channel}\n")
              PY

        - name: Install pinned Rust toolchain
          uses: dtolnay/rust-toolchain@master
          with:
              toolchain: ${{ steps.rust.outputs.channel }}

        - name: Cache Rust artifacts
          uses: Swatinem/rust-cache@v2

        - name: Install ROCm runtime headers (cubecl-hip-sys build script needs hipconfig)
          if: matrix.cell == 'all-features'
          run: |
              wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
              sudo apt-get install -y ./amdgpu-install_6.0.60000-1_all.deb
              sudo amdgpu-install --usecase=rocm --no-dkms -y
              echo "/opt/rocm/bin" >> $GITHUB_PATH

        - name: cargo check
          run: |
              if [ -z "${{ matrix.features }}" ]; then
                cargo check -p cintx-cubecl
              else
                cargo check -p cintx-cubecl --features "${{ matrix.features }}"
              fi

        - name: cargo test (excluding ignored)
          run: |
              if [ -z "${{ matrix.features }}" ]; then
                cargo test -p cintx-cubecl
              else
                cargo test -p cintx-cubecl --features "${{ matrix.features }}"
              fi
```

## Per-Cell Expectation Table

| Cell | Cargo flags | Expected runtime | What it catches |
|------|-------------|------------------|-----------------|
| `cpu-only` | (default `["cpu"]`) | ~3 min | regressions in the cpu baseline path |
| `cpu+wgpu` | `--features wgpu` | ~5 min | regressions in `cubecl-wgpu` integration + the `[wgpu]` cfg-gated arms (Wgpu/Metal in `BackendKind` and `ResolvedBackend`) |
| `all-features` | `--features wgpu,cuda,rocm,metal` | ~10 min (ROCm install dominates) | regressions in `cubecl-cuda` / `cubecl-hip` builds + all five backend cfg arms compiling together; this cell is the regression net for `--features cuda` or `--features rocm` PR breaks |

The runtime targets are provisional — actual durations will be observed on the first PR run. If `all-features` exceeds 15 min, RESEARCH §6.4 fallbacks (apt-pin `rocm-dev` or demote to weekly cron + 4-cell matrix) are the documented contingency. The plan author chose the planner's recommended path; fallbacks are documented but not pre-applied.

## Key Link to Wave 1's `[features]` Table

The cargo invocations exercise exactly the additive `[features]` table that Wave 1 committed in `crates/cintx-cubecl/Cargo.toml`:

```toml
[features]
default = ["cpu"]
cpu = ["cubecl/cpu", "cintx-runtime/cpu"]
wgpu = ["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/wgpu"]
cuda = ["dep:cubecl-cuda", "cintx-runtime/cuda"]
rocm = ["dep:cubecl-hip", "cintx-runtime/rocm"]
metal = ["wgpu", "cintx-runtime/metal"]
```

- `cpu-only` cell exercises `default = ["cpu"]`.
- `cpu+wgpu` cell exercises `cpu` + `wgpu` (latter pulls `dep:cubecl-wgpu`, `dep:wgpu`, `cintx-runtime/wgpu`).
- `all-features` cell exercises every backend feature. Note `metal = ["wgpu", "cintx-runtime/metal"]` (the M1 alias from Wave 1) means `metal` transitively activates `wgpu`, so listing both `wgpu` and `metal` in the matrix's `--features` flag is redundant but harmless and makes the intent explicit.

`cubecl-hip-sys` (transitive dep of `cubecl-hip` 0.10.0) needs `hipconfig` on `$PATH` at build time per RESEARCH §3.2 / §8.2 — the `amdgpu-install` step in the all-features cell satisfies this.

## Verification Status

- [x] YAML parses (`python3 -c "import yaml; yaml.safe_load(...)"` exits 0)
- [x] `feature_matrix_gate:` job key present
- [x] `runs-on: ubuntu-latest`
- [x] `strategy.matrix.include` has three entries (cpu-only / cpu+wgpu / all-features)
- [x] `strategy.fail-fast: false`
- [x] 4-step preamble byte-identical to existing gates
- [x] `if: matrix.cell == 'all-features'` step installs ROCm via `amdgpu-install`
- [x] `cargo check` step + `cargo test (excluding ignored)` step both present in every cell
- [x] `grep -c "feature_matrix_gate"` returns 2
- [x] STATE.md / ROADMAP.md NOT modified (this plan is sequential-mode-on-main and the orchestrator owns those updates)
- [ ] **PENDING (post-merge, requires user action):** all three matrix-named status checks added to branch protection's required-checks list — see "Manual User Step Required" below
- [ ] **PENDING (operator-driven, not blocking this commit):** smoke test on a feature-branch push showing all three cells turn green within 15 minutes; if all-features flakes, apply RESEARCH §6.4 fallback

## Manual User Step Required (Task 2 — `checkpoint:human-action`)

GitHub Actions auto-discovers new status checks the first time the workflow runs on a PR, but **does NOT** automatically add them to branch protection's required-checks list. A repo admin must do this manually post-merge. This is why this plan is `autonomous: false`.

**Steps for the user:**

1. Push this commit to GitHub (or merge it to `main` via PR — the next PR will surface the new checks).
2. Open the repo on GitHub: `Settings → Branches → main → Edit branch protection rule`.
3. Under "Require status checks to pass before merging", click "Add" / search.
4. Add each of these three status-check names (must match the workflow's `name:` template exactly — GitHub renders `feature_matrix_gate (${{ matrix.cell }})` for each cell):
   - `feature_matrix_gate (cpu-only)`
   - `feature_matrix_gate (cpu+wgpu)`
   - `feature_matrix_gate (all-features)`
5. Save the rule.
6. Open any existing PR (or push a no-op commit). Confirm the three new status checks appear in the PR's check list and are flagged as Required.
7. Confirm "merge" is blocked until all three are green.

**Why the planner cannot do this automatically:** branch-protection rules are repo-settings state managed via the GitHub API (or UI), not via files in the repo. Modifying them requires a token with `Administration: write` permission on the repo, which is intentionally not held by automated PR workflows. RESEARCH §8.6 spelled this out as a known manual step.

**Verification command for the user (after Step 5):** open a PR and observe the three new entries listed under "Required status checks" in the PR's merge box.

## Smoke-Test Note (operator-driven, deferred)

The plan's `<acceptance_criteria>` includes "On a feature-branch push to GitHub, all three matrix cells turn green within 15 minutes." This requires a network round-trip to GitHub Actions and is not blocking this code-side commit. The operator (user) will observe the first run and capture timing. If the all-features cell exceeds 15 minutes or flakes, apply RESEARCH §6.4 fallback:

1. **Apt-pin `rocm-dev`** via Radeon's Jammy archive (smaller install footprint than `amdgpu-install`).
2. **Demote `all-features` to a weekly cron** and replace with a 4-cell matrix: `cpu+wgpu+cuda` and `cpu+wgpu+metal` as additional cells (drops the rocm runtime install requirement from per-PR).

If `cubecl-hip-sys` build fails because of an HIP version mismatch (RESEARCH §8.2 — dev host runs ROCm 7.x, CI runner gets ROCm 6.0 from amdgpu-install 6.0), HALT and report — do not commit a workaround that hides the version-skew issue.

## Deviations from Plan

None — plan executed exactly as written. The verbatim job block from `<interfaces>` was inserted at the planned position (after `api_value_baseline_gate`) with byte-identical preamble.

The plan's automated verify command suggested `python3 -c "import yaml; ..."`, which initially failed because the workspace's `python3` (`/home/user/.local/bin/python3`) is a user-installed Python lacking PyYAML. Fell back to system `/usr/bin/python3` which has PyYAML 6.0.1 — same parse, same exit-0 result. Not a code deviation; just a per-environment toolchain quirk.

## Issues Encountered

- The plan's `<read_first>` cited `compat-governance-pr.yml` lines 73 (oracle_parity_gate), 113 (helper_legacy_parity_gate), 151 (oom_contract_gate), 190 (api_value_baseline_gate) as references for the preamble. These line numbers matched exactly — no drift between Wave 1's commits and the plan's authored line citations.
- No issues during the YAML edit. The Edit tool's exact-string-match with the surrounding `Run API value baseline gate / run: ci/api-value-baseline.sh` + `gpu_bench_advisory:` boundary anchor produced a clean insertion.

## Self-Check

**Must-haves from plan `truths:` block:**

| Must-have | Status | Evidence |
|-----------|--------|----------|
| compat-governance-pr.yml contains a new feature_matrix_gate job alongside the existing 5 required gates | PASSED | `grep -n "feature_matrix_gate:"` returns line 229; existing 5 gates intact at lines 37, 73, 113, 151, 190 |
| feature_matrix_gate is a 3-cell matrix: cpu-only / cpu+wgpu / all-features | PASSED | `grep -n "cell: "` returns 3 hits at lines 235/237/239 inside `strategy.matrix.include:` |
| Each cell runs cargo check + cargo test (excluding ignored) | PASSED | Two cargo steps at lines 277-282 and 285-289; tests use `cargo test -p cintx-cubecl` which excludes `#[ignore]` by default |
| all-features cell installs ROCm runtime headers via `amdgpu-install`, gated `if: matrix.cell == 'all-features'` | PASSED | Line 271 has the guard; lines 272-275 run amdgpu-install + add /opt/rocm/bin to PATH |
| fail-fast: false | PASSED | Line 232 |
| 4-step preamble byte-identical to existing gates | PASSED | Lines 246-269 (Checkout / Resolve channel / Install toolchain / Cache) match manifest_drift_gate's preamble at lines 41-67 verbatim |
| Manual user step: register three matrix entries in branch protection | DOCUMENTED | "Manual User Step Required" section above + checkpoint message returned to orchestrator |

**File / commit existence checks:**

- `[ -f .github/workflows/compat-governance-pr.yml ]` → FOUND
- `[ -f .planning/phases/16-multi-backend-support/16-03-SUMMARY.md ]` → FOUND (this file)
- `python3 -c "import yaml; yaml.safe_load(...)"` exit 0 → CONFIRMED with system PyYAML 6.0.1

## Self-Check: PASSED

## Next Phase Readiness

- Wave 3 (16-04) — ROCm oracle suite + `xtask rocm-oracle` — does NOT depend on this gate landing in branch protection. It can land in parallel; the rocm oracle suite is opt-in (`#[ignore]` + env-gated) and runs on the dev box, not on GitHub runners.
- The branch-protection registration (Task 2) can happen any time after this commit reaches `main`; it is not blocking for Wave 3.
- Once registered, any future PR that breaks `--features cuda`, `--features rocm`, `--features metal`, or the all-features combo will fail the corresponding `feature_matrix_gate` cell and be blocked from merging — closing the regression-coverage gap that ROADMAP success criterion 7 / Phase-16 D-16 calls out.

---

*Phase: 16-multi-backend-support*
*Plan: 03*
*Completed: 2026-05-09*
