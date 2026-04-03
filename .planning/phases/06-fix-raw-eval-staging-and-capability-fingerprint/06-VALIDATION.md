---
phase: 06
slug: fix-raw-eval-staging-and-capability-fingerprint
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (nextest available) |
| **Config file** | none (workspace default) |
| **Quick run command** | `cargo test -p cintx-compat -- raw::tests` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-compat -- raw::tests`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | COMP-01 | unit | `cargo test -p cintx-compat -- eval_raw_writes_executor_output` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | COMP-04 | integration | `cargo test -p cintx-capi -- shim::tests::query_and_eval_wrappers_succeed_and_clear_tls_error` | ✅ | ⬜ pending |
| 06-01-03 | 01 | 1 | COMP-05 | unit | `cargo test -p cintx-compat -- raw::tests` | ✅ | ⬜ pending |
| 06-01-04 | 01 | 1 | EXEC-02 | unit | `cargo test -p cintx-compat -- eval_raw_all_base_families` | ❌ W0 | ⬜ pending |
| 06-01-05 | 01 | 1 | EXEC-04 | unit | `cargo test -p cintx-compat -- eval_raw_representation_layouts` | ❌ W0 | ⬜ pending |
| 06-01-06 | 01 | 1 | EXEC-05 | unit | `cargo test -p cintx-compat -- eval_raw_optimizer_on_off_equivalence` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | VERI-01 | integration | `cargo test -p cintx-oracle` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_writes_executor_output` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_all_base_families` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_output_is_not_all_zeros` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `query_workspace_raw_fingerprint_is_nonzero_when_gpu_available` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_representation_layouts` test
- [ ] `crates/cintx-compat/src/raw.rs` — add `eval_raw_optimizer_on_off_equivalence` test

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU adapter availability | EXEC-02 | Hardware-dependent | Run `cargo test --workspace` on a machine with wgpu-compatible GPU; verify non-zero staging output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
