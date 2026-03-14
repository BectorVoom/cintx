---
phase: 1
slug: contracts-and-typed-foundations
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` |
| **Config file** | none — Wave 0 installs phase-specific test layout |
| **Quick run command** | `cargo test --workspace --lib --quiet` |
| **Full suite command** | `cargo test --workspace --all-targets` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --workspace --lib --quiet`
- **After every plan wave:** Run `cargo test --workspace --all-targets`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-TBD-01 | TBD | 0 | SAFE-01 | unit | `cargo test --workspace safe_contracts::typed_inputs` | ❌ W0 | ⬜ pending |
| 01-TBD-02 | TBD | 0 | SAFE-02 | unit | `cargo test --workspace workspace_query::deterministic` | ❌ W0 | ⬜ pending |
| 01-TBD-03 | TBD | 0 | SAFE-04 | unit | `cargo test --workspace errors::typed_categories` | ❌ W0 | ⬜ pending |
| 01-TBD-04 | TBD | 0 | MEM-03 | integration | `cargo test --workspace diagnostics::structured_failures` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/phase1_contracts.rs` — typed input and workspace-query contract tests (SAFE-01, SAFE-02)
- [ ] `tests/phase1_errors.rs` — typed error taxonomy and diagnostics tests (SAFE-04, MEM-03)
- [ ] `Cargo.toml` test profile/config updates if required for deterministic quick/full commands

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
