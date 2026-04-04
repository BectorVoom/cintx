---
phase: 11
slug: helper-transform-completion-4c1e-real-kernel
status: draft
nyquist_compliant: true
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
| 11-01-01 | 01 | 1 | HELP-01, D-01 | unit + integration | `cargo test -p cintx-compat --features cpu -- helpers` | YES (helpers.rs tests) | pending |
| 11-01-02 | 01 | 1 | HELP-01, HELP-02 | integration | `cargo test -p cintx-oracle --features cpu -- verify_helper` | W0 — test function created in this task | pending |
| 11-02-01 | 02 | 1 | 4C1E-01, 4C1E-03 | unit + integration | `cargo test -p cintx-cubecl --features cpu -- center_4c1e` | YES (center_4c1e tests) | pending |
| 11-03-01 | 03 | 2 | 4C1E-02 | integration | `cargo check -p cintx-compat --features "with-4c1e,cpu"` | W0 — workaround.rs created in this task | pending |
| 11-03-02 | 03 | 2 | HELP-03, 4C1E-04 | integration | `cargo test -p cintx-oracle --features "cpu,with-4c1e" -- oracle_gate && cargo test -p cintx-oracle --features cpu -- verify_helper` | W0 — oracle gate and legacy parity functions created in this task | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `verify_helper_surface_coverage` numeric comparison (created in Plan 01, Task 2 — extends existing function)
- [ ] `verify_legacy_wrapper_parity` function (created in Plan 03, Task 2)
- [ ] `oracle_gate_4c1e_parity` test (created in Plan 03, Task 2)
- [ ] workaround.rs module (created in Plan 03, Task 1)

*Each Wave 0 item is created as part of the plan task that needs it — no separate Wave 0 plan required. The test functions are created within the implementation tasks.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI gate passes across 4 profiles | HELP-04, 4C1E-04 | Requires CI runner | Push branch, verify workflow green |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands matching this map
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 items created within implementation tasks (no separate Wave 0 plan needed)
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
