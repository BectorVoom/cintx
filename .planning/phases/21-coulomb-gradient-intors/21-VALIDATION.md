---
phase: 21
slug: coulomb-gradient-intors
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-26
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Seeded from `21-RESEARCH.md` §"Validation Architecture". Per-task rows are filled by the planner during `/gsd:plan-phase` and updated by the executor.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` / `cargo nextest` (Rust integration tests in `cintx-oracle/tests/`) |
| **Config file** | none — existing workspace harness + `#[cfg(has_vendor_libcint)]` gate |
| **Quick run command** | `cargo test -p cintx-cubecl` |
| **Full suite command** | `cargo test -p cintx-oracle --features <profile>` |
| **Estimated runtime** | ~60-180 seconds (vendor libcint build cached) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl` (quick kernel feedback)
- **After every plan wave:** Run the vendor-gated `cintx-oracle` parity for families landed in that wave
- **Before `/gsd-verify-work`:** Full suite must be green across affected feature profiles
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

> Seed map keyed to the proposed plan/family breakdown. The planner replaces `{N}-NN-NN` task IDs with real ones and confirms the automated command per task.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 21-01-* | 01 | 1 | GRAD-01 | — | N/A | unit | `cargo test -p cintx-runtime rinv_orig` | ❌ W0 | ⬜ pending |
| 21-02-* | 02 | 1 | GRAD-02 | — | N/A | integration | `cargo build && <manifest-audit xtask>` | ❌ W0 | ⬜ pending |
| 21-03-* | 03 | 2 | GRAD-03, GRAD-04 | — | N/A | oracle | `cargo test -p cintx-oracle ipovlp ipkin` | ❌ W0 | ⬜ pending |
| 21-04-* | 04 | 2 | GRAD-05, GRAD-06 | — | N/A | oracle | `cargo test -p cintx-oracle ipnuc iprinv` | ❌ W0 | ⬜ pending |
| 21-05-* | 05 | 3 | GRAD-07 | — | N/A | oracle | `cargo test -p cintx-oracle int2e_ip1` | ❌ W0 | ⬜ pending |
| 21-06-* | 06 | 3 | GRAD-08 | — | N/A | oracle | `cargo test -p cintx-oracle int3c2e_ip1` | ❌ W0 | ⬜ pending |
| 21-07-* | 07 | 4 | GRAD-09 | — | N/A | oracle | `cargo test -p cintx-oracle ecp_iprinv` | ❌ W0 | ⬜ pending |
| 21-08-* | 08 | 4 | GRAD-10 | — | N/A | manual+integration | layout-vs-vendor check; doc updates | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `cintx-oracle/src/vendor_ffi.rs` — new `vendor_*_ip1` / `vendor_*iprinv` FFI wrappers around vendored libcint 6.1.3 (added per-family as kernels land)
- [ ] No new framework — the existing `#[cfg(has_vendor_libcint)]` oracle harness + `cargo test` cover the phase

*Existing infrastructure covers all phase requirements except the new vendor FFI wrappers above.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Component-leading `[3, …]` F-order matches pyscf-gto `layout_table.rs` | GRAD-07, GRAD-10 | Cross-repo layout contract not exercisable from cintx CI alone | Compare cintx staging layout against the vendor `int2e_ip1` memory layout in the oracle; confirm against pyscf-gto `layout_table.rs` |
| pyscf_rs Phase 7 `workflow_dispatch` grad arms un-gate | GRAD-10 | Cross-repo consumer acceptance | After landing, flip pyscf_rs arms to always-on; FD gate ≤1e-6 Ha/Bohr, PySCF parity ≤1e-7 Ha/Bohr |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
