---
phase: 8
slug: gaussian-primitive-infrastructure-and-boys-function
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test (`#[test]`) via `cargo test` |
| **Config file** | `rust-toolchain.toml` pins `1.94.0` |
| **Quick run command** | `cargo test -p cintx-cubecl --features cpu` |
| **Full suite command** | `cargo test -p cintx-cubecl --features cpu` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu`
- **After every plan wave:** Run `cargo test -p cintx-cubecl --features cpu`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 0 | MATH-01 | compile | `cargo test -p cintx-cubecl --features cpu boys_` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 0 | MATH-02 | compile | `cargo test -p cintx-cubecl --features cpu pdata_` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 1 | MATH-01 | unit | `cargo test -p cintx-cubecl --features cpu boys_` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 1 | MATH-02 | unit | `cargo test -p cintx-cubecl --features cpu pdata_` | ❌ W0 | ⬜ pending |
| 08-03-01 | 03 | 1 | MATH-03 | unit | `cargo test -p cintx-cubecl --features cpu rys_` | ❌ W0 | ⬜ pending |
| 08-03-02 | 03 | 1 | MATH-04 | unit | `cargo test -p cintx-cubecl --features cpu os_` | ❌ W0 | ⬜ pending |
| 08-04-01 | 04 | 2 | all | integration | `cargo test -p cintx-cubecl --features cpu math_integration` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-cubecl/src/math/mod.rs` — module entry point
- [ ] `crates/cintx-cubecl/src/math/boys.rs` — Boys function skeleton
- [ ] `crates/cintx-cubecl/src/math/rys.rs` — Rys quadrature skeleton
- [ ] `crates/cintx-cubecl/src/math/obara_saika.rs` — OS recurrence skeleton
- [ ] `crates/cintx-cubecl/src/math/pdata.rs` — PairData struct skeleton
- [ ] `crates/cintx-cubecl/src/lib.rs` — add `pub mod math;`
- [ ] `crates/cintx-cubecl/Cargo.toml` — add `approx` dev-dependency
- [ ] CubeCL probe: `erf()` on wgpu, const array dynamic indexing, CubeType with array fields

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| wgpu `erf()` availability | MATH-01 | Requires GPU with WGSL backend | Run Boys test on wgpu backend; verify large-x branch works or falls back |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
