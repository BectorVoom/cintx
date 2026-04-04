---
phase: 9
slug: 1e-real-kernel-and-cart-to-sph-transform
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (built-in) |
| **Config file** | none (uses workspace Cargo.toml) |
| **Quick run command** | `cargo test -p cintx-cubecl --features cpu -- c2s` |
| **Full suite command** | `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu` |
| **Estimated runtime** | ~30 seconds |

---

## Wave 0 Strategy

All plans in Phase 9 use inline TDD (`tdd="true"` with `<behavior>` blocks). Tests are co-created
alongside implementation within each task — the TDD RED-GREEN-REFACTOR cycle creates test files as
part of the task itself. No separate Wave 0 plan is needed because:

- Plan 09-01 Task 1 has `tdd="true"` and creates tests inline in c2s.rs
- Plan 09-01 Task 2 creates `tests/c2s_tests.rs` as an integration test (TDD task)
- Plan 09-02 Task 1 creates `#[cfg(test)] mod tests` inline in one_electron.rs
- Plan 09-03 Task 1 has `tdd="true"` and creates `tests/one_electron_parity.rs`

Test stubs are NOT pre-created; they emerge from the TDD behavior specifications in each task.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu 2>&1 | tail -5`
- **After every plan wave:** Run `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | TDD Inline | Status |
|---------|------|------|-------------|-----------|-------------------|------------|--------|
| 09-01-01 | 01 | 1 | KERN-06 | unit | `cargo test -p cintx-cubecl -- c2s` | yes | pending |
| 09-01-02 | 01 | 1 | KERN-06 | integration | `cargo test -p cintx-cubecl --test c2s_tests` | yes | pending |
| 09-02-01 | 02 | 2 | KERN-01 | unit | `cargo test -p cintx-cubecl --features cpu -- one_electron` | yes | pending |
| 09-03-01 | 03 | 3 | VERI-05 | oracle | `cargo test -p cintx-oracle --features cpu -- one_electron_parity` | yes | pending |
| 09-03-02 | 03 | 3 | VERI-05 | artifact | `test -f /mnt/data/phase-09-1e-oracle-parity.md` | no | pending |

*Status: pending / green / red / flaky*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 handled via inline TDD (no separate stub plan needed)
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (inline TDD satisfies Nyquist contract)
