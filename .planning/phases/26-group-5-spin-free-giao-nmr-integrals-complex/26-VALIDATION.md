---
phase: 26
slug: group-5-spin-free-giao-nmr-integrals-complex
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-31
---

# Phase 26 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 26-RESEARCH.md §Validation Architecture. Vendor parity is
> double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (without
> BOTH, parity tests silently skip — project memory).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (cargo test / nextest); oracle integration tests under `crates/cintx-oracle/tests/` |
| **Config file** | none (cargo default); vendor gate via env `CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu` |
| **Quick run command** | `cargo test -p <crate-touched> --features cpu` |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --locked` |
| **Estimated runtime** | ~quick: <60s per crate · full vendor-gated: several min (libcint vendor build) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-touched> --features cpu` (quick, no vendor build)
- **After every plan wave:** Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` (full vendor-gated)
- **Before `/gsd-verify-work`:** Full suite green; `manifest-audit` green; all GIAO symbols `oracle_covered=true`
- **Max feedback latency:** ~60s (quick) per task

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 26-01-* | 01 (FND-03) | 1 | FND-03 | — | manifest `complex_output` drives `complex_interleaved` + 2× staging; no silent zeroing | unit | `cargo test -p cintx-runtime build_output_layout` | ❌ W0 | ⬜ pending |
| 26-01-* | 01 (FND-03) | 1 | FND-03 | — | complex family staged real-only FAILS the contract (fail-closed) | unit | `cargo test -p cintx-oracle --lib assert_flat_buffer_contract` | ❌ W0 | ⬜ pending |
| 26-01-* | 01 (FND-03) | 1 | FND-03 | — | `int1e_igovlp` safe-API round-trip: imag non-zero, real exactly zero (D-07/D-15) | integration | `cargo test -p cintx-oracle --features cpu giao_complex_roundtrip` | ❌ W0 | ⬜ pending |
| 26-A-* | Cluster A (1e) | 2 | GIAO-01 | — | 11 spin-free 1e families byte-identical cart+sph, non-square + non-zero-gauge | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_1e_parity` | ❌ W0 | ⬜ pending |
| 26-B-* | Cluster B (2e) | 2 | GIAO-02 | — | 4 spin-free 2e families (g1, ig1, gg1, g1g2) byte-identical cart+sph | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_2e_parity` | ❌ W0 | ⬜ pending |
| (recipe) | A + B | 2 | (D-10) | — | `manifest-audit` green after registration (auto-syncs from lock) | xtask/unit | `cargo xtask manifest-audit` (or existing audit test) | ✓ existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/giao_1e_parity.rs` — covers GIAO-01 (11 families × cart/sph)
- [ ] `crates/cintx-oracle/tests/giao_2e_parity.rs` — covers GIAO-02 (4 families × cart/sph: g1, ig1, gg1, g1g2)
- [ ] `crates/cintx-oracle/tests/giao_complex_roundtrip.rs` — FND-03 safe-API D-07 assertion on `int1e_igovlp`
- [ ] `crates/cintx-runtime` unit test for manifest-driven `complex_output` → `complex_interleaved` / staging (FND-03)
- [ ] `crates/cintx-oracle` lib unit test for generalized fail-closed `assert_flat_buffer_contract` (D-04)

*Framework present — no install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Worktree wave integration to `main` | (D-09) | Background worktree auto-merge is inconsistent (project memory) | After each cluster wave, run `git merge-base --is-ancestor <wave-branch> main`; merge manually if not an ancestor |

*All phase numerical behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
