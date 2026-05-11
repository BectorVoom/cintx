---
phase: 17
slug: real-integral-evaluation-in-safe-api
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-12
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (workspace) |
| **Config file** | `Cargo.toml` (workspace) |
| **Quick run command** | `cargo test --locked -p cintx-rs` |
| **Full suite command** | `cargo test --locked --workspace` |
| **Estimated runtime** | ~60–120 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command for the touched crate (`cargo test --locked -p <crate>`)
- **After every plan wave:** Run full workspace command
- **Before `/gsd-verify-work`:** Full suite must be green AND `cintx-oracle` parity tests green
- **Max feedback latency:** ~60 seconds (per-crate quick command)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 17-01 | 0 | RVAL-02 | — | N/A (Cargo edge only; T-17-01-* all `accept`) | build / Cargo edge | `cargo build -p cintx-oracle --locked && cargo build -p cintx-oracle --features cpu --locked && grep -E '^cintx-rs = \{ path = "\.\./cintx-rs", default-features = false \}$' crates/cintx-oracle/Cargo.toml` | ✅ | ⬜ pending |
| 17-02-01 | 17-02 | 1 | RVAL-01 / RVAL-03 | — | N/A (executor swap; T-17-02-* all `accept`; OOM-safe stop contract preserved via real `CubeClExecutor`) | unit (build + negative greps) | `cargo build -p cintx-rs --locked && ! grep -F 'fn fill_staging_values' crates/cintx-rs/src/api.rs && ! grep -E '^struct CubeClExecutor' crates/cintx-rs/src/api.rs && ! grep -F 'idx + 1' crates/cintx-rs/src/api.rs && ! grep -F '(idx + 1) * 0.5' crates/cintx-rs/src/api.rs && grep -F 'use cintx_cubecl::CubeClExecutor;' crates/cintx-rs/src/api.rs && grep -F 'let executor = CubeClExecutor::new();' crates/cintx-rs/src/api.rs` | ✅ | ⬜ pending |
| 17-02-02 | 17-02 | 1 | RVAL-01 | — | N/A (test-only rename + behavior assertions) | unit (test rewrite) | `cargo test -p cintx-rs --locked evaluate_returns_deterministic_nonzero_real_values -- --exact && cargo test -p cintx-rs --locked && ! grep -F 'owned_values[0], 1.0' crates/cintx-rs/src/api.rs && ! grep -F 'evaluate_runs_runtime_path_and_returns_owned_output' crates/cintx-rs/src/api.rs && grep -F 'evaluate_returns_deterministic_nonzero_real_values' crates/cintx-rs/src/api.rs` | ✅ | ⬜ pending |
| 17-03-01 | 17-03 | 2 | RVAL-02 | — | N/A (test file only; T-17-03-* all `accept`) | integration (oracle parity, cart/sph) | `cargo build -p cintx-oracle --features cpu --locked --tests && cargo build -p cintx-oracle --locked --tests && test "$(grep -cE '^#\[test\]' crates/cintx-oracle/tests/safe_api_arity2_parity.rs)" -eq 8 && grep -F 'atol = 1e-12_f64' crates/cintx-oracle/tests/safe_api_arity2_parity.rs && grep -F 'rtol = 0.0_f64' crates/cintx-oracle/tests/safe_api_arity2_parity.rs` | ❌ Wave 0 → created by Plan 17-03 Task 1 | ⬜ pending |
| 17-03-02 | 17-03 | 2 | RVAL-02 | — | N/A (test file only; spinor idempotency only) | integration (oracle, spinor idempotency) | `cargo build -p cintx-oracle --features cpu --locked --tests && CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity2_parity -- --test-threads=1 && test "$(grep -cE '^#\[test\]' crates/cintx-oracle/tests/safe_api_arity2_parity.rs)" -eq 12 && grep -F 'fn assert_safe_api_idempotent' crates/cintx-oracle/tests/safe_api_arity2_parity.rs && grep -F 'OperatorId::new(2)' crates/cintx-oracle/tests/safe_api_arity2_parity.rs && grep -F 'OperatorId::new(5)' crates/cintx-oracle/tests/safe_api_arity2_parity.rs && grep -F 'OperatorId::new(8)' crates/cintx-oracle/tests/safe_api_arity2_parity.rs && grep -F 'OperatorId::new(14)' crates/cintx-oracle/tests/safe_api_arity2_parity.rs` | ❌ Wave 0 → extended by Plan 17-03 Task 2 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Wave dependencies (post-revision): 17-01-01 (Wave 0) → 17-02-01 + 17-02-02 (Wave 1) → 17-03-01 + 17-03-02 (Wave 2). Plan 17-03 depends on both Plan 17-01 (Cargo edge) and Plan 17-02 (executor swap) — without 17-02 the vendor-parity tests in 17-03-01 fail because the synthetic executor would still be in place.*

---

## Wave 0 Requirements

- [x] Add `cintx-rs` as a `dev-dependencies` (or normal dependency under a test-only feature) of `cintx-oracle` so `safe_api_arity2_parity.rs` can call `SessionRequest::evaluate` — delivered by Plan 17-01 Task 1.
- [x] Sanity-confirm `#[cfg(has_vendor_libcint)]` and `#![cfg(any(feature = "cpu", feature = "rocm"))]` gating mirrors `one_electron_parity.rs` so the new parity file runs in the same CI matrix — enforced by Plan 17-03 Task 1 acceptance criteria.

*Wave 0 (the Cargo edge) is the only structural blocker; both Wave-1 plans (17-02) and Wave-2 plans (17-03) compile only after Plan 17-01 lands. Plan 17-03 additionally requires Plan 17-02 to be green for vendor-parity tests to pass.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual diff of `crates/cintx-rs/src/api.rs` after stub removal | RVAL-03 | SemVer-surface check requires human eyeballing of the public diff (no automated SemVer tool currently wired) | `cargo public-api --diff origin/main` if installed; otherwise `git diff origin/main -- crates/cintx-rs/src/api.rs` and confirm only private items (`CubeClExecutor`, `fill_staging_values`) change |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
