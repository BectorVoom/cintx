---
phase: 15
slug: oracle-tolerance-unification-manifest-lock-closure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-06
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust) + cargo run --manifest-path xtask/Cargo.toml |
| **Config file** | Cargo.toml (workspace), xtask/Cargo.toml |
| **Quick run command** | `cargo test -p cintx-oracle --features cpu --lib -- tolerance` |
| **Full suite command** | `cargo test -p cintx-oracle --features cpu` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-oracle --features cpu --lib -- tolerance`
- **After every plan wave:** Run `cargo test -p cintx-oracle --features cpu`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | ORAC-01 | unit | `cargo test -p cintx-oracle --features cpu --lib -- tolerance_for_family` | ✅ | ⬜ pending |
| 15-01-02 | 01 | 1 | ORAC-01 | unit | `cargo test -p cintx-oracle --features cpu --lib -- oracle_family` | ✅ | ⬜ pending |
| 15-02-01 | 02 | 2 | ORAC-04 | integration | `cargo test -p cintx-oracle --features cpu -- oracle_gate_closure` | ✅ | ⬜ pending |
| 15-03-01 | 03 | 3 | ORAC-02 | xtask | `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit --check-lock` | ✅ | ⬜ pending |
| 15-03-02 | 03 | 3 | ORAC-03 | xtask | `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit --check-oracle-covered` | ❌ W0 | ⬜ pending |
| 15-04-01 | 04 | 4 | ORAC-03 | CI | `gh workflow run compat-governance-pr.yml` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `xtask/src/manifest_audit.rs` — add oracle_covered completeness check function
- [ ] Existing test infrastructure covers tolerance and oracle comparison

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI matrix runs all 4 profiles in parallel | ORAC-03 | Requires GitHub Actions runner | Push branch, verify Actions UI shows 4 matrix jobs |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
