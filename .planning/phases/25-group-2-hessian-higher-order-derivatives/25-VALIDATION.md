---
phase: 25
slug: group-2-hessian-higher-order-derivatives
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-30
---

# Phase 25 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `25-RESEARCH.md` § Validation Architecture (derived verbatim from vendored libcint 6.1.3 + cintx source).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cargo test` (no nextest dep; oracle parity tests are integration tests under `crates/cintx-oracle/tests/`) |
| **Config file** | none (cargo-native); vendor gate via env + feature |
| **Quick run command** | `cargo test -p cintx-cubecl --lib` (kernel/math unit tests, fast) |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` (vendor-gated parity) |
| **Estimated runtime** | ~60–180 seconds (vendor build dominates first run) |

> ⚠ Vendor parity is **double-gated**: without BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`, parity tests silently skip (only determinism runs). See memory `reference_oracle_vendor_parity_invocation`.

---

## Sampling Rate

- **After every task commit:** `cargo test -p cintx-cubecl --lib` + the touched family's `vendor_*` test (gated).
- **After every plan wave:** full vendor-gated oracle suite + `cargo test -p cintx-runtime --lib` (FND-06) + `manifest-audit`. Confirm worktree integration with `merge-base --is-ancestor` after each cluster wave (worktree auto-merge is inconsistent — memory `feedback_worktree_auto_integration_inconsistent`).
- **Before `/gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** ~180 seconds (full vendor suite).

---

## Per-Task Verification Map

> Populated by the planner per task. Every task must map to one row with an automated command or a Wave 0 dependency.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| _TBD by planner_ | | | | | | | | | ⬜ pending |

### Requirement → Test anchors (from research)

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-02 | nroots 6..12 roots/weights byte-identical vs vendor `CINTrys_roots` | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu rys_nroots_sweep` | ❌ Wave 0 (`tests/rys_nroots_sweep_parity.rs`) |
| FND-02 | host `rys_roots_host(6..12)` no longer panics | unit | `cargo test -p cintx-cubecl --lib rys_host_nroots_ge6` | ❌ Wave 0 |
| FND-06 | rank-81 staging under memory limit → typed OOM, no partial write | unit | `cargo test -p cintx-runtime --lib rank81_oom_no_partial_write` | ❌ Wave 0 (`planner.rs` test mod, template `:1000`) |
| FND-06 | upfront assertion fires on undersized staging | unit | `cargo test -p cintx-runtime --lib staging_buffer_too_small` | ❌ Wave 0 |
| HESS-01 | `int1e_ipip{ovlp,nuc,kin,rinv}` cart+sph atol=1e-12, non-square block | integration (vendor) | `… cargo test … hess1e_ipip` | ❌ Wave 0 (`tests/hess1e_ipip_parity.rs`) |
| HESS-02 | `int2e_ipip1/ipvip1/ip1ip2/ipip1ipip2` cart+sph atol=1e-12 | integration (vendor) | `… cargo test … hess2e_ipip` | ❌ Wave 0 (`tests/hess2e_parity.rs`) |
| HESS-03 | `int2c2e_ipip1`, `int3c2e_ipip1/ipip2` cart+sph atol=1e-12 | integration (vendor) | `… cargo test … hess_multicenter_ipip` | ❌ Wave 0 |
| HESS-04 | 3rd/4th-order families cart+sph atol=1e-12, non-square + bra≠ket | integration (vendor) | `… cargo test … deriv34_ipipip` | ❌ Wave 0 (`tests/deriv34_parity.rs`) |
| ALL | `manifest-audit` green after lock edits | xtask | `cargo run -p xtask -- manifest-audit` | ✓ exists |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/rys_nroots_sweep_parity.rs` — FND-02 nroots 6..12 sweep vs vendor `CINTrys_roots` (D-02). **Highest priority — the FND-02 long-pole validation.**
- [ ] `rys.rs` unit test `rys_host_nroots_ge6` — host fn no longer panics, returns correct count.
- [ ] `planner.rs` test mod additions — `rank81_oom_no_partial_write` (D-05) + `staging_buffer_too_small` (D-04), template `planner.rs:1000`.
- [ ] `tests/hess1e_ipip_parity.rs`, `tests/hess2e_parity.rs`, `tests/hess_multicenter_ipip_parity.rs`, `tests/deriv34_parity.rs` — per-cluster `vendor_*` tests, NON-SQUARE blocks (p×d), bra≠ket for deriv4 (D-09).
- [ ] xtask `gen-rys-tables` subcommand + `--check` drift-gate (P19 precedent) for the JACOBI_*/POLY_* constant blobs.
- [ ] `build.rs` edits: add `deriv3.c`/`deriv4.c` `.file()` + extend `allowlist_function` regex with all Phase-25 cart/sph symbols (HESS-01/02/03 compile from already-built `hess.c`/`int3c2e.c` — only allowlist needed; HESS-04 needs both).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| nroots vendor ceiling (12 vs 13+) | FND-02 | Vendor build config dependent (quadmath disabled) | Plan-1 vendor probe (research open question A1) — confirm before fixing the executor l-gate upper bound |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
