---
phase: 30
slug: group-5-giao-slice-spin-giao-integrals-spinor
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-01
---

# Phase 30 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (workspace) + vendor oracle parity harness |
| **Config file** | none — existing workspace test infra |
| **Quick run command** | `cargo test -p cintx-cubecl giao` |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao` |
| **Estimated runtime** | ~120 seconds (vendor build adds ~60s on first run) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl giao` (transform/kernel-level)
- **After every plan wave:** Run the full vendor parity suite (`CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao`) — each wave is gated green before the next begins (D-04)
- **Before `/gsd-verify-work`:** Full vendor suite must be green + `cargo xtask manifest-audit` green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

> Filled per-plan by the planner. The decisive gates are the dual gauge∧kappa vendor byte-identity tests (one `vendor_*` test per family, non-skipped, double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`) and the Wave-0 gauge-gout micro-test.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 30-00-01 | 00 | 0 | GIAO-03 | — | gauge `g`-factor fold byte-identical to vendored thin family | unit/parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu gauge_gout_micro` | ❌ W0 | ⬜ pending |
| 30-01a-02 | 01a | 1 | GIAO-03 | — | spgsp (+cg/giao_sa10sp) byte-identical (spinor, atol=1e-12, non-square) | parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` | ❌ W0 | ⬜ pending |
| 30-01b-02 | 01b | 1 | GIAO-03 | — | cg/giao_sa10nucsp byte-identical (spinor, atol=1e-12, non-square) | parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` | ❌ W0 | ⬜ pending |
| 30-01c-02 | 01c | 1 | GIAO-03 | — | cg/giao_sa10sa01 (rank 9) byte-identical (spinor, atol=1e-12, non-square) | parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` | ❌ W0 | ⬜ pending |
| 30-01d-02 | 01d | 1 | GIAO-03 | — | spgnucsp + spgsa01 byte-identical; full 9-family 1e gate (spinor, atol=1e-12, non-square) | parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` | ❌ W0 | ⬜ pending |
| 30-02-01 | 02 | 2 | GIAO-03 | — | 2e GIAO×σ families byte-identical (spinor, atol=1e-12) | parity | `... cargo test ... vendor_int2e_spg` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · Task IDs are illustrative — the planner sets final per-plan IDs.*

---

## Wave 0 Requirements

- [ ] Combined gauge≠0 ∧ kappa≠0 spinor fixture (1e form) in `crates/cintx-oracle/src/fixtures.rs` — extends `build_kappa_spinor_fixture` with `with_common_origin`; non-square, ≥1 shell nctr>1, GT/LT kappa mix
- [ ] Gauge-gout byte-identity micro-test (compares the gauge `g`-factor fold to a vendored thin family `int1e_cg_sa10sp`, with a cg→giao-at-origin=0 differential check) — D-03 de-risk, lands BEFORE any family is wired
- [ ] `sigma_p.rs` gauge-origin variant scaffolding (the `CINTx1i_1e` with-origin G-tensor step) reading `PTR_COMMON_ORIG` via `eval_raw`

*Existing infrastructure (transforms, 2e si/sf suite, gauge plumbing, oracle build wiring for intor3.c/intor4.c) covers all other requirements — no new framework install.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification via the vendor byte-identity parity gate (atol=1e-12) + `manifest-audit`. No manual-only behaviors.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (combined fixture + gauge-gout micro-test)
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
