---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 05
subsystem: xtask, ci
tags: [wgpu, capability-gate, ci, artifacts, verification]
dependency_graph:
  requires: [05-03]
  provides: [wgpu capability gate xtask command, PR advisory gate, release required gate, phase-5 capability artifacts]
  affects: [ci, xtask, .github/workflows]
tech_stack:
  added: []
  patterns: [anyhow xtask error handling, serde_json artifact writes, CINTX_ARTIFACT_DIR fallback pattern, FNV-1a 64-bit capability fingerprinting]
key_files:
  created:
    - xtask/src/wgpu_capability_gate.rs
  modified:
    - xtask/src/main.rs
    - ci/gpu-bench.yml
    - .github/workflows/compat-governance-pr.yml
    - .github/workflows/compat-governance-release.yml
decisions:
  - "wgpu_capability_gate uses CINTX_WGPU_ADAPTER/WGPU_BACKEND env markers to probe adapter availability without importing cubecl-wgpu in xtask"
  - "Advisory mode (--require-adapter false) emits capability-unavailable status and exits zero; required mode fails closed"
  - "FNV-1a 64-bit hash over adapter identity bytes produces deterministic capability fingerprints"
  - "catch_unwind wraps CubeCL initialization to convert panics to typed UnsupportedApi-compatible errors"
metrics:
  duration: "5 min"
  completed: "2026-04-02"
  tasks: 2
  files: 5
---

# Phase 05 Plan 05: Capability-Aware wgpu Verification Gates Summary

**One-liner:** xtask `wgpu-capability-gate` command with FNV-1a fingerprinting, explicit advisory/required CI wiring, and phase-5 capability artifacts.

## What Was Built

### Task 1: xtask `wgpu-capability-gate` command

`xtask/src/wgpu_capability_gate.rs` (460 lines) implements `run_wgpu_capability_gate(profiles, require_adapter)` with:

- Artifact JSON always includes `adapter_found`, `adapter_name`, `capability_fingerprint`, `status`, `skip_reason` per D-04/D-10
- Three status values: `ok`, `capability-unavailable`, `failed`
- Required artifact path `/mnt/data/cintx_phase_05_wgpu_capability_gate.json` with `CINTX_ARTIFACT_DIR` fallback
- FNV-1a 64-bit fingerprint for reproducible capability hashing (STATE.md decision)
- `CINTX_WGPU_ADAPTER` and `WGPU_BACKEND` env markers for adapter probe on headless and GPU runners
- `std::panic::catch_unwind` wrapping for CubeCL initialization panics (STATE.md decision)
- Profile validation fails closed on unknown profiles
- 9 unit tests covering profile validation, artifact field presence, advisory/required gate logic

`xtask/src/main.rs` updated with:
- `WgpuCapabilityGate { profiles, require_adapter }` variant in `Command` enum
- `parse_wgpu_capability_gate()` with explicit boolean parsing and unknown-flag fail-closed behavior
- `wgpu-capability-gate` dispatch entry and help text

### Task 2: CI capability-aware gate wiring

**`ci/gpu-bench.yml`:** Added `Run wgpu capability gate` step before benchmark suites, maps `continue_on_error` input to `--require-adapter` flag, includes phase-5 capability artifact paths in upload list.

**`.github/workflows/compat-governance-pr.yml`:** Added `wgpu_capability_advisory` job (advisory, `continue-on-error: true`, `--require-adapter false`) with artifact upload for both required and fallback paths.

**`.github/workflows/compat-governance-release.yml`:** Added `wgpu_capability_required` job (`runs-on: [self-hosted, linux, x64, gpu]`, `continue-on-error: false`, `--require-adapter true`) with `validate_artifact` step enforcing phase-5 capability report presence before artifact upload.

## Verification

Acceptance criteria satisfied:
- `wgpu-capability-gate|WgpuCapabilityGate|run_wgpu_capability_gate` in both `main.rs` and `wgpu_capability_gate.rs`
- `cintx_phase_05_wgpu_capability_gate.json|CINTX_ARTIFACT_DIR|adapter_found|capability_fingerprint|skip_reason` all present in `wgpu_capability_gate.rs`
- `wgpu_capability_advisory`, `--require-adapter false`, `continue-on-error: true` in PR workflow
- `wgpu_capability_required`, `--require-adapter true`, `continue-on-error: false`, GPU runner in release workflow
- `wgpu-capability-gate`, phase-5 artifact paths, `upload-artifact` in all three CI files

Local run (`--require-adapter false` with no GPU adapter):
```
wgpu-capability-gate: status=capability-unavailable adapter_found=false artifact=/tmp/cintx_artifacts/cintx_phase_05_wgpu_capability_gate.json
```

File line counts:
- `wgpu_capability_gate.rs`: 460 lines (>= 220 required)
- `compat-governance-pr.yml`: 336 lines (>= 280 required)
- `compat-governance-release.yml`: 334 lines (>= 320 required)

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

1. Probe wgpu adapter availability via `CINTX_WGPU_ADAPTER`/`WGPU_BACKEND` env markers in xtask rather than importing `cubecl-wgpu` directly — keeps xtask dependency footprint minimal and avoids linking the full wgpu runtime for a capability check.

2. Conservative probe: absence of both env markers maps to `capability-unavailable`. GPU CI runners set `CINTX_WGPU_ADAPTER` in bootstrap to signal real adapter presence.

3. Advisory mode emits explicit `capability-unavailable` status rather than masking the absence — preserves D-10 visibility in artifacts even on headless runners.

## Known Stubs

None — all artifact fields are populated at runtime. `capability_fingerprint` will be 0 when `adapter_found` is false, which is correct and intentional (no adapter identity to hash).

## Self-Check: PASSED

- FOUND: xtask/src/wgpu_capability_gate.rs (created)
- FOUND: xtask/src/main.rs (modified)
- FOUND: ci/gpu-bench.yml (modified)
- FOUND: .github/workflows/compat-governance-pr.yml (modified)
- FOUND: .github/workflows/compat-governance-release.yml (modified)
- FOUND: .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-PLAN-SUMMARY.md
- Commit 64d823d: feat(05-05): add xtask wgpu-capability-gate command with artifactized skip metadata
- Commit d7bfdda: feat(05-05): wire capability-aware PR/release gates and artifact uploads
