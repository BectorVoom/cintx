---
phase: 24
slug: group-3-position-multipole-moment-integrals
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-30
---

# Phase 24 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` integration tests under `crates/cintx-oracle/tests/`, gated `#![cfg(any(feature = "cpu", feature = "rocm"))]` + `has_vendor_libcint` cfg |
| **Config file** | none — cargo test discovery; per-test cfg gates |
| **Quick run command** | `cargo test -p cintx-oracle --features cpu --test <moment_parity_test> -- <name>` |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` |
| **Estimated runtime** | ~60-120 seconds (vendor build + parity suite) |

> **Vendor double-gate (MEMORY):** real parity is double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`. Without BOTH, only determinism tests run and parity silently skips. Routine `--features cpu` CI runs `--test` integration only, never `--lib` under vendor — the folded `oracle-cart-offset-vendor-zero` lib-unit failure (OQ-2) surfaces only when vendor `--lib` tests run; triage so it does not block the phase gate.

---

## Sampling Rate

- **After every task commit:** Run `cargo build -p cintx-ops` (manifest regen) + `cargo test -p cintx-oracle --features cpu --test <family>_parity` for the family just added
- **After every plan wave:** Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` (all moment parity tests)
- **Before `/gsd-verify-work`:** Full vendor suite green + `manifest-audit` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 24-A-* | A | 1 | MOM-01/02/03 | — / V5 input-validation (env-slot finiteness) | non-finite gauge origin rejected by `validate_common_orig_env_params` | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test moment_r_parity` | ❌ W0 | ⬜ pending |
| 24-B-* | B | 2 | MOM-04 (rinv/drinv) | — / V5 (`validate_rinv_orig_env_params`) | non-finite rinv_orig rejected | integration (vendor) | `... --test moment_nontensor_parity` | ❌ W0 | ⬜ pending |
| 24-C-* | C | 2 | MOM-04 (p4) | — | N/A | integration (vendor) | `... --test moment_nontensor_parity` | ❌ W0 | ⬜ pending |
| 24-D-* | D | 2 | MOM-04 (irp) | — | N/A | integration (vendor) | `... --test moment_nontensor_parity` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Plan IDs/waves are indicative (Cluster A→B/C/D per CONTEXT D-01); the planner finalizes exact plan/task IDs.*

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/moment_r_parity.rs` — covers MOM-01 (`r`, `r_origj`)
- [ ] `crates/cintx-oracle/tests/moment_low_parity.rs` — covers MOM-02 (`rr`, `r2`, `z`, `zz`, +`_origj`)
- [ ] `crates/cintx-oracle/tests/moment_high_parity.rs` — covers MOM-03 (`rrr`, `rrrr`, `r4`)
- [ ] `crates/cintx-oracle/tests/moment_nontensor_parity.rs` — covers MOM-04 (`p4`, `rinv`, `drinv`, `irp`)
- [ ] `vendor_*` safe wrappers in `crates/cintx-oracle/src/vendor_ffi.rs` for each new cart/sph symbol (mirror `vendor_int1e_iprinv_*`)
- [ ] `allowlist_function` regex extension in `crates/cintx-oracle/build.rs` for all `int1e_{r,rr,rrr,rrrr,r2,r4,z,zz,p4,irp,rinv,drinv}_{sph,cart}` + `_origj` symbols
- [ ] Test helper: a NON-ZERO `rinv_orig` setter for the rinv/drinv tests (the common-orig fixture does not set env[4..6])
- [ ] Each parity test uses a NON-SQUARE bra×ket shell pair from H2O/STO-3G (D-07 transpose guard)

*Existing `vendor_parity<FS,FC>` helpers (`one_electron_grad_both_parity.rs:307`) are the pattern to clone — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `manifest-audit` green after lock edits | MOM-01..04 (SC5) | Build-time audit, not a `#[test]` | `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`; audit derives both sides from the lock |
| OQ-2 lib-unit `CINTshells_cart_offset` triage | — | Pre-existing harness noise, confirm against pre-phase-20 commit | Run vendor `--lib` tests; if reproduced, convert to a tracked standalone oracle-harness bug so the gate is not blocked |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
