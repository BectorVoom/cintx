---
phase: 10
slug: 2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) + cargo nextest |
| **Config file** | `Cargo.toml` workspace test config |
| **Quick run command** | `cargo test -p cintx-cubecl --features cpu --lib -- kernels` |
| **Full suite command** | `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu --lib -- kernels`
- **After every plan wave:** Run `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | KERN-03 | integration | `cargo test -p cintx-cubecl --features cpu -- center_2c2e` | ❌ W0 | ⬜ pending |
| 10-02-01 | 02 | 2 | KERN-04 | integration | `cargo test -p cintx-cubecl --features cpu -- center_3c1e` | ❌ W0 | ⬜ pending |
| 10-03-01 | 03 | 3 | KERN-05 | integration | `cargo test -p cintx-cubecl --features cpu -- center_3c2e` | ❌ W0 | ⬜ pending |
| 10-04-01 | 04 | 4 | KERN-02 | integration | `cargo test -p cintx-cubecl --features cpu -- two_electron` | ❌ W0 | ⬜ pending |
| 10-05-01 | 05 | 5 | VERI-05, VERI-07 | oracle | `cargo test -p cintx-oracle --features cpu` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-cubecl/src/math/rys.rs` — Add rys_root3_host, rys_root4_host, rys_root5_host wrappers
- [ ] `crates/cintx-cubecl/src/transform/c2s.rs` — Add cart_to_sph_2e, cart_to_sph_2c2e, cart_to_sph_3c1e, cart_to_sph_3c2e
- [ ] `crates/cintx-oracle/build.rs` — Extend to compile 2e/2c2e/3c1e/3c2e C sources + bindgen allowlist

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| C ABI shim on real GPU | VERI-07 | Requires physical GPU hardware | Build with wgpu backend, call cintrs_eval() on int1e_ovlp_sph, verify status == 0 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
