---
phase: 03
slug: verification-and-compatibility-governance
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-14
updated: 2026-03-14
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` integration suites |
| **Config file** | `Cargo.toml` |
| **Quick run command** | `cargo test --workspace --test phase3_helper_transform_parity --test phase3_manifest_governance --test phase3_optimizer_equivalence --test phase3_regression_gates` |
| **Full suite command** | `cargo test --workspace --all-targets` |
| **Estimated runtime** | ~15 minutes (after phase-3 suites are added) |

---

## Sampling Rate

- **After every task commit:** Run that task's exact `<automated>` command from the map below.
- **After every plan wave:**
  - Wave 1: `cargo test --workspace --test phase3_helper_transform_parity`
  - Wave 2: `cargo test --workspace --test phase3_manifest_governance`
  - Wave 3: `cargo test --workspace --test phase3_optimizer_equivalence`
  - Wave 4: `cargo test --workspace --test phase3_regression_gates`
- **Before `$gsd-verify-work`:** `cargo test --workspace --all-targets` must be green.
- **Max feedback latency:** 15 minutes.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | COMP-02 | integration | `cargo test --workspace --test phase3_helper_transform_parity helper_matrix_parity` | ❌ planned | ⬜ pending |
| 03-01-02 | 01 | 1 | COMP-02 | integration | `cargo test --workspace --test phase3_helper_transform_parity transform_parity_cart_sph_spinor` | ❌ planned | ⬜ pending |
| 03-01-03 | 01 | 1 | VERI-03 | integration | `cargo test --workspace --test phase3_helper_transform_parity helper_diagnostics_contract` | ❌ planned | ⬜ pending |
| 03-02-01 | 02 | 2 | COMP-03 | unit | `cargo test --workspace --test phase3_manifest_governance manifest_schema_invariants` | ❌ planned | ⬜ pending |
| 03-02-02 | 02 | 2 | VERI-01 | integration | `cargo test --workspace --test phase3_manifest_governance lock_drift_requires_explicit_update` | ❌ planned | ⬜ pending |
| 03-02-03 | 02 | 2 | COMP-03, COMP-04 | integration | `cargo test --workspace --test phase3_manifest_governance profile_union_coverage` | ❌ planned | ⬜ pending |
| 03-03-01 | 03 | 3 | RAW-04 | integration | `cargo test --workspace --test phase3_optimizer_equivalence optimizer_on_off_equivalence_matrix` | ❌ planned | ⬜ pending |
| 03-03-02 | 03 | 3 | VERI-03 | integration | `cargo test --workspace --test phase3_optimizer_equivalence spinor_layout_regression` | ❌ planned | ⬜ pending |
| 03-03-03 | 03 | 3 | VERI-03 | integration | `cargo test --workspace --test phase3_optimizer_equivalence oom_error_semantics_regression` | ❌ planned | ⬜ pending |
| 03-04-01 | 04 | 4 | COMP-02, VERI-01 | integration | `cargo test --workspace --test phase3_helper_transform_parity helper_transform_parity_matrix && cargo test --workspace --test phase3_manifest_governance lock_drift_requires_explicit_update` | ✅ `.github/workflows/compat-governance-pr.yml` | ✅ green |
| 03-04-02 | 04 | 4 | VERI-02 | integration | `cargo test --workspace --test phase3_regression_gates oracle_profile_matrix_gate` | ✅ `.github/workflows/compat-governance-release.yml` | ✅ green |
| 03-04-03 | 04 | 4 | COMP-04, VERI-02 | integration | `cargo test --workspace --test phase3_regression_gates requirement_traceability_gate` | ✅ `docs/phase3-governance-gates.md` + `docs/phase2-support-matrix.md` | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing Rust test infrastructure already covers this phase. No framework/bootstrap work is required before execution.

---

## Manual-Only Verifications

All Phase 3 acceptance behaviors are expected to have automated verification targets.

---

## Validation Sign-Off

- [x] All planned tasks have `<automated>` verification commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verification
- [x] Wave 0 dependency check completed (no missing framework prerequisites)
- [x] No watch-mode flags in validation commands
- [x] `nyquist_compliant: true` set in frontmatter
- [ ] `cargo test --workspace --all-targets` green on latest phase branch

**Approval:** pending
