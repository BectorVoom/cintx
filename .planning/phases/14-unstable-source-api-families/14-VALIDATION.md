---
phase: 14
slug: unstable-source-api-families
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` + `cargo nextest` |
| **Config file** | `Cargo.toml` (workspace root) |
| **Quick run command** | `cargo test --features unstable-source-api -p cintx-oracle -- unstable_source` |
| **Full suite command** | `cargo test --features unstable-source-api --workspace` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick run command (unstable_source oracle tests)
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | USRC-01 | oracle parity | `cargo test --features unstable-source-api -p cintx-oracle -- origi` | ❌ W0 | ⬜ pending |
| 14-02-01 | 02 | 1 | USRC-02 | oracle parity | `cargo test --features unstable-source-api -p cintx-oracle -- grids` | ❌ W0 | ⬜ pending |
| 14-03-01 | 03 | 2 | USRC-03 | oracle parity | `cargo test --features unstable-source-api -p cintx-oracle -- breit` | ❌ W0 | ⬜ pending |
| 14-04-01 | 04 | 2 | USRC-04 | oracle parity | `cargo test --features unstable-source-api -p cintx-oracle -- origk` | ❌ W0 | ⬜ pending |
| 14-05-01 | 05 | 2 | USRC-05 | oracle parity | `cargo test --features unstable-source-api -p cintx-oracle -- ssc` | ❌ W0 | ⬜ pending |
| 14-06-01 | 06 | 3 | USRC-06 | CI gate | CI workflow `unstable_source_oracle` job passes | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/unstable_source_parity.rs` — stubs for all USRC-* requirements
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — extend with unstable family FFI bindings
- [ ] Feature flag `unstable-source-api` propagated to `cintx-cubecl/Cargo.toml`

*Existing oracle infrastructure covers shared fixtures — only family-specific tests need adding.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Nightly CI triggers correctly | USRC-06 | Requires schedule trigger | Verify CI job definition, run manually with `workflow_dispatch` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
