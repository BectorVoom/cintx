---
phase: 12
slug: real-spinor-transform-c2spinor-replacement
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` + `nextest` (if installed) |
| **Config file** | `rust-toolchain.toml` (toolchain pin) |
| **Quick run command** | `cargo test --package cintx-cubecl --lib transform::c2spinor` |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu` |
| **Estimated runtime** | ~60 seconds (oracle build + comparison) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --package cintx-cubecl --lib transform`
- **After every plan wave:** Run `cargo test --package cintx-cubecl && cargo test --package cintx-compat`
- **Before `/gsd:verify-work`:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu` full suite green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | SPIN-01 | unit | `cargo test --package cintx-cubecl --lib transform::c2spinor_coeffs` | ❌ W0 | ⬜ pending |
| 12-01-02 | 01 | 1 | SPIN-01 | unit | `cargo test --package cintx-cubecl --lib transform::c2spinor` | ❌ W0 (stub exists) | ⬜ pending |
| 12-02-01 | 02 | 1 | SPIN-02, SPIN-04 | unit | `cargo test --package cintx-compat --lib transform` | ❌ W0 (stub exists) | ⬜ pending |
| 12-03-01 | 03 | 2 | SPIN-03 | integration | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure spinor_1e` | ❌ W0 | ⬜ pending |
| 12-04-01 | 04 | 2 | SPIN-03 | integration | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure spinor_multi` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — CG coefficient tables extracted from libcint (SPIN-01)
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — vendor spinor integral wrappers for 1e families (SPIN-03)
- [ ] Stub tests in `c2spinor.rs` and `transform.rs` replaced with value-correctness tests

*Existing infrastructure (oracle_gate_closure.rs, fixtures.rs, compare.rs) covers test scaffolding.*

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
