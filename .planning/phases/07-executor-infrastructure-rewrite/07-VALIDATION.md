---
phase: 7
slug: executor-infrastructure-rewrite
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test (`#[test]`) via `cargo test` |
| **Config file** | `rust-toolchain.toml` pins `1.94.0` |
| **Quick run command** | `cargo test -p cintx-cubecl --features cpu 2>&1 \| tail -20` |
| **Full suite command** | `cargo test --workspace --features cpu 2>&1 \| tail -40` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu 2>&1 | tail -20`
- **After every plan wave:** Run `cargo test --workspace --features cpu 2>&1 | tail -40`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | EXEC-06 | unit | `cargo test -p cintx-cubecl --features cpu execute_uses_direct_client_api` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | EXEC-08 | unit | `cargo test -p cintx-cubecl --features cpu resolved_backend_from_intent_selects_correct_arm` | ❌ W0 | ⬜ pending |
| 07-01-03 | 01 | 1 | EXEC-08 | unit | `cargo test -p cintx-cubecl --features cpu backend_env_var_selection` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | EXEC-07 | unit | `cargo test -p cintx-compat --features cpu eval_raw_reads_staging_directly` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | EXEC-09 | integration | `cargo test -p cintx-cubecl --features cpu` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 2 | VERI-06 | unit | `cargo test -p cintx-cubecl shader_f64_absent_returns_unsupported_api` | ❌ W0 | ⬜ pending |
| 07-reg-01 | 01 | 1 | EXEC-06 | regression | `cargo test -p cintx-cubecl` | ✅ existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-cubecl/Cargo.toml` — add `cpu = ["cubecl/cpu"]` feature flag (required before any `--features cpu` test compiles)
- [ ] `crates/cintx-cubecl/src/backend/mod.rs` — `ResolvedBackend` unit tests: `from_intent` selects correct arm, env var routing, Mutex cache prevents double-init
- [ ] `crates/cintx-cubecl/src/backend/cpu.rs` — CPU client bootstrap test (requires `--features cpu`)
- [ ] `crates/cintx-cubecl/src/executor.rs` — test: `execute_uses_direct_client_api` asserts `stage_device_buffers` is gone; direct client path runs
- [ ] `crates/cintx-compat/src/raw.rs` — test: `eval_raw_reads_staging_directly` asserts no `RecordingExecutor` in call chain; staging populated before `execute` returns
- [ ] `crates/cintx-cubecl/src/kernels/mod.rs` — update existing `family_registry_resolves_base_slice` test to compile with new `FamilyLaunchFn` signature

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| wgpu `SHADER_F64` hardware probe | VERI-06 | Requires GPU hardware with/without f64 support | Run `cargo test -p cintx-cubecl` on a machine with GPU; verify f64 test passes or returns `UnsupportedApi` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
