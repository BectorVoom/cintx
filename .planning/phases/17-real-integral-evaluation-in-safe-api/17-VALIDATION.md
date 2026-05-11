---
phase: 17
slug: real-integral-evaluation-in-safe-api
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| TBD-by-planner | — | — | RVAL-01 / RVAL-02 / RVAL-03 | — | — | unit / integration | `cargo test --locked` | ✅ / ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Map will be filled by `/gsd:plan-phase` once PLAN.md files are emitted — one row per task referencing the parity test and unit-test changes called out in `17-RESEARCH.md` (Validation Architecture).*

---

## Wave 0 Requirements

- [ ] Add `cintx-rs` as a `dev-dependencies` (or normal dependency under a test-only feature) of `cintx-oracle` so `safe_api_arity2_parity.rs` can call `SessionRequest::evaluate` — currently no Cargo edge exists.
- [ ] Sanity-confirm `#[cfg(has_vendor_libcint)]` and `#![cfg(any(feature = "cpu", feature = "rocm"))]` gating mirrors `one_electron_parity.rs` so the new parity file runs in the same CI matrix.

*If neither edge is needed before any task, drop these and write "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual diff of `crates/cintx-rs/src/api.rs` after stub removal | RVAL-03 | SemVer-surface check requires human eyeballing of the public diff (no automated SemVer tool currently wired) | `cargo public-api --diff origin/main` if installed; otherwise `git diff origin/main -- crates/cintx-rs/src/api.rs` and confirm only private items (`CubeClExecutor`, `fill_staging_values`) change |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
