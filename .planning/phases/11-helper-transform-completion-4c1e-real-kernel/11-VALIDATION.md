---
phase: 11
slug: helper-transform-completion-4c1e-real-kernel
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test / cargo nextest |
| **Config file** | `Cargo.toml` workspace |
| **Quick run command** | `cargo test -p cintx-oracle --lib -- --test-threads=1` |
| **Full suite command** | `cargo test -p cintx-oracle --features cpu -- --test-threads=1` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-oracle --lib -- --test-threads=1`
- **After every plan wave:** Run `cargo test -p cintx-oracle --features cpu -- --test-threads=1`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | HELP-01 | integration | `cargo test -p cintx-oracle helper_oracle` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | HELP-02 | integration | `cargo test -p cintx-oracle transform_oracle` | ❌ W0 | ⬜ pending |
| 11-02-01 | 02 | 1 | HELP-03 | integration | `cargo test -p cintx-oracle legacy_wrapper_oracle` | ❌ W0 | ⬜ pending |
| 11-02-02 | 02 | 1 | HELP-04 | integration | `cargo test -p cintx-oracle helper_legacy_parity` | ✅ exists | ⬜ pending |
| 11-03-01 | 03 | 2 | 4C1E-01 | integration | `cargo test -p cintx-oracle --features with-4c1e 4c1e_oracle` | ❌ W0 | ⬜ pending |
| 11-03-02 | 03 | 2 | 4C1E-02 | integration | `cargo test -p cintx-oracle --features with-4c1e 4c1e_via_2e` | ❌ W0 | ⬜ pending |
| 11-04-01 | 04 | 2 | 4C1E-03 | unit | `cargo test -p cintx-cubecl --features with-4c1e validated_4c1e` | ✅ exists | ⬜ pending |
| 11-04-02 | 04 | 2 | 4C1E-04 | integration | `cargo test -p cintx-oracle --features with-4c1e,cpu oracle_gate` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Oracle comparison tests for helpers (integer equality) in `crates/cintx-oracle/`
- [ ] Oracle comparison tests for transforms (atol=1e-12) in `crates/cintx-oracle/`
- [ ] Oracle comparison tests for legacy wrappers (atol=1e-12) in `crates/cintx-oracle/`
- [ ] Oracle comparison tests for 4c1e (atol=1e-12) in `crates/cintx-oracle/`
- [ ] Oracle comparison tests for int4c1e_via_2e_trace equivalence in `crates/cintx-oracle/`

*Existing infrastructure (compare.rs, fixtures.rs) covers framework; new test functions needed per symbol category.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI gate passes across 4 profiles | HELP-04, 4C1E-04 | Requires CI runner | Push branch, verify workflow green |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
