---
phase: 13
slug: f12-stg-yp-kernels
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test + cargo nextest |
| **Config file** | `rust-toolchain.toml` (pins Rust 1.94.0) |
| **Quick run command** | `cargo test --features cpu,with-f12 -p cintx-cubecl -- math::stg` |
| **Full suite command** | `cargo test --features cpu,with-f12 -p cintx-oracle -- oracle_gate` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --features cpu,with-f12 -p cintx-cubecl -x`
- **After every plan wave:** Run `cargo test --features cpu,with-f12 --workspace`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | F12-01 | unit | `cargo test --features cpu,with-f12 -p cintx-cubecl -- stg_roots_host` | ❌ W0 | ⬜ pending |
| 13-01-02 | 01 | 1 | F12-01 | unit | `cargo test --features cpu,with-f12 -p cintx-cubecl -- stg_roots_t_clamp` | ❌ W0 | ⬜ pending |
| 13-02-01 | 02 | 1 | F12-02 | unit | `cargo test --features cpu,with-f12 -p cintx-cubecl -- f12_stg_yp_differ` | ❌ W0 | ⬜ pending |
| 13-03-01 | 03 | 2 | F12-03 | oracle | `cargo test --features cpu,with-f12 -p cintx-oracle -- oracle_gate_f12` | ❌ W0 | ⬜ pending |
| 13-03-02 | 03 | 2 | F12-03 | oracle | `cargo test --features cpu,with-f12 -p cintx-oracle -- f12_sph_only_enforcement` | ❌ W0 | ⬜ pending |
| 13-04-01 | 01 | 1 | F12-04 | unit | `cargo test -p cintx-runtime -- execution_plan_f12_zeta` | ❌ W0 | ⬜ pending |
| 13-04-02 | 01 | 1 | F12-05 | unit | `cargo test -p cintx-runtime -- f12_zeta_zero_rejected` | ❌ W0 | ⬜ pending |
| 13-05-01 | 03 | 2 | F12-05 | integration | `cargo test --features cpu,with-f12 -p cintx-oracle -- f12_zeta_zero_fixture` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-cubecl/src/math/stg.rs` — unit tests for `stg_roots_host` (F12-01)
- [ ] `crates/cintx-cubecl/src/kernels/f12.rs` — launch function tests (F12-01, F12-02)
- [ ] `crates/cintx-runtime/tests/f12_plan_tests.rs` — ExecutionPlan zeta plumbing tests (F12-04, F12-05)
- [ ] `crates/cintx-oracle/tests/oracle_gate_closure.rs` — extend with `#[cfg(feature="with-f12")]` F12 section (F12-03)

*Existing infrastructure covers test framework. New test files and sections needed for F12-specific coverage.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
