---
phase: 9
slug: 1e-real-kernel-and-cart-to-sph-transform
status: draft
nyquist_compliant: false
wave_0_complete: false
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

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu 2>&1 | tail -5`
- **After every plan wave:** Run `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | KERN-06 | unit | `cargo test -p cintx-cubecl -- c2s_coeff` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 1 | KERN-06 | unit | `cargo test -p cintx-cubecl -- c2s_d_shell` | ❌ W0 | ⬜ pending |
| 09-02-01 | 02 | 2 | KERN-01 | unit | `cargo test -p cintx-cubecl --features cpu -- ovlp` | ❌ W0 | ⬜ pending |
| 09-02-02 | 02 | 2 | KERN-01 | unit | `cargo test -p cintx-cubecl --features cpu -- kinetic` | ❌ W0 | ⬜ pending |
| 09-02-03 | 02 | 2 | KERN-01 | unit | `cargo test -p cintx-cubecl --features cpu -- nuclear` | ❌ W0 | ⬜ pending |
| 09-03-01 | 03 | 3 | VERI-05 | oracle | `cargo test -p cintx-oracle --features cpu -- ovlp_parity` | ❌ W0 | ⬜ pending |
| 09-03-02 | 03 | 3 | VERI-05 | oracle | `cargo test -p cintx-oracle --features cpu -- kin_parity` | ❌ W0 | ⬜ pending |
| 09-03-03 | 03 | 3 | VERI-05 | oracle | `cargo test -p cintx-oracle --features cpu -- nuc_parity` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-cubecl/tests/c2s_tests.rs` — stubs for KERN-06 (coefficient matrix correctness)
- [ ] `crates/cintx-cubecl/tests/one_electron_tests.rs` — stubs for KERN-01 (operator post-processing)
- [ ] Oracle parity test for 1e sph family in `crates/cintx-oracle/` — stubs for VERI-05

*Existing infrastructure covers framework and test config.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
