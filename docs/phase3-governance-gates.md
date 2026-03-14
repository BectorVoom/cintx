# Phase 3 Compatibility Governance Gates

This document defines the blocking CI governance policy for Phase 3 compatibility claims.

## Policy

- Compatibility claims are valid only when governance gates are green.
- PR gates block helper parity drift, manifest lock drift, and requirement-traceability regressions before merge.
- Release gates block publish-time claims unless full helper, manifest, oracle profile, and optimizer suites pass.
- Lock drift is blocked by default and only allowed with explicit approval rationale validated by tests.

## Gate Inventory

| Scope | Workflow | Job | Command | Requirement IDs |
|---|---|---|---|---|
| PR | `.github/workflows/compat-governance-pr.yml` | `helper_parity_gate` | `cargo test --workspace --test phase3_helper_transform_parity helper_transform_parity_matrix` | `COMP-02` |
| PR | `.github/workflows/compat-governance-pr.yml` | `manifest_governance_gate` | `cargo test --workspace --test phase3_manifest_governance lock_drift_requires_explicit_update` | `COMP-03`, `VERI-01` |
| PR | `.github/workflows/compat-governance-pr.yml` | `core_regression_gate` | `cargo test --workspace --test phase3_regression_gates requirement_traceability_gate` | `COMP-04`, `VERI-02` |
| Release | `.github/workflows/compat-governance-release.yml` | `helper_parity_release_gate` | `cargo test --workspace --test phase3_helper_transform_parity` | `COMP-02` |
| Release | `.github/workflows/compat-governance-release.yml` | `manifest_release_gate` | `cargo test --workspace --test phase3_manifest_governance` | `COMP-03`, `VERI-01` |
| Release | `.github/workflows/compat-governance-release.yml` | `oracle_profile_release_gate` | `cargo test --workspace --test phase3_regression_gates oracle_profile_matrix_gate` and `cargo test --workspace --test phase3_regression_gates requirement_traceability_gate` | `COMP-04`, `VERI-02` |
| Release | `.github/workflows/compat-governance-release.yml` | `optimizer_equivalence_release_gate` | `cargo test --workspace --test phase3_optimizer_equivalence` | `RAW-04`, `VERI-03` |

## Lock Update Policy

- Baseline manifest lock drift without approval is a hard failure (`UnapprovedLockDrift`).
- Approval rationale is mandatory for permitted lock updates (`EmptyApprovalRationale` if blank).
- Required evidence command: `cargo test --workspace --test phase3_manifest_governance lock_drift_requires_explicit_update`.

## Deterministic Release Expectations

- Release gates run on `release` events (`published`, `prereleased`) and version tags (`v*`).
- Profile-aware oracle checks are executed by `oracle_profile_matrix_gate`, which enforces approved profile scope coverage and requirement traceability.
- Any failed governance job blocks release compatibility assertions.

## Traceability Anchors

- Requirements source: `.planning/REQUIREMENTS.md`.
- Validation map: `.planning/phases/03-verification-and-compatibility-governance/03-VALIDATION.md`.
- Stable-family envelope: `docs/phase2-support-matrix.md`.
