---
phase: 25
slug: group-2-hessian-higher-order-derivatives
status: draft
nyquist_compliant: true
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

> One row per task across the 6 plans (Plan 01 split Task 1 → 1a MRRR / 1b Wheeler; Plan 06 split registration → Task 2 deriv3 rank-27 / Task 3 deriv4 rank-81). File-Exists ✅ = scaffolded by a Wave-0 task; ❌ W0 = created by this task's own Wave-0 step.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-T0 vendor probe + sweep scaffold + panic-removal stub | 01 | 1 | FND-02 | unit + scaffold | `grep -c 'nroots > 5 not supported' crates/cintx-cubecl/src/math/rys.rs \| grep -qx 0 && grep -qn 'fn rys_nroots_sweep' crates/cintx-oracle/tests/rys_nroots_sweep_parity.rs` | ❌ W0 (`tests/rys_nroots_sweep_parity.rs`) | ⬜ pending |
| 01-T1a MRRR eigensolver port (eigh.c #else, ~1400 lines) | 01 | 1 | FND-02 | unit | `cargo test -p cintx-cubecl --lib eigh_mrrr_tridiag` | ❌ W0 (`math/eigh.rs`) | ⬜ pending |
| 01-T1b Flocke/Wheeler machinery + nroots 6..12 wire | 01 | 1 | FND-02 | unit + integration (vendor) | `cargo test -p cintx-cubecl --lib rys_host_nroots_ge6 && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu rys_nroots_sweep` | ✅ (T0 scaffold) | ⬜ pending |
| 01-T2 gen-rys-tables drift-gate + executor/launcher gate | 01 | 1 | FND-02 | xtask + integration (vendor) | `cargo run -p xtask -- gen-rys-tables --check && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu rys_nroots_sweep` | ✅ (T1b) | ⬜ pending |
| 02-T1 upfront staging-size assertion | 02 | 1 | FND-06 | unit | `cargo test -p cintx-runtime --lib staging_buffer_too_small` | ❌ W0 (`planner.rs` test mod) | ⬜ pending |
| 02-T2 strip 19 per-element scatter guards | 02 | 1 | FND-06 | build | `test -z "$(grep -rl 'if dst < staging.len()' crates/cintx-cubecl/src/kernels/)" && cargo build -p cintx-cubecl` | ✅ (source) | ⬜ pending |
| 02-T3 rank-81 OOM no-partial-write test | 02 | 1 | FND-06 | unit | `cargo test -p cintx-runtime --lib rank81_oom_no_partial_write` | ❌ W0 (`planner.rs` test mod) | ⬜ pending |
| 03-T0 hess1e_ipip_parity scaffold (4 families) | 03 | 2 | HESS-01 | scaffold | `grep -qc 'fn hess1e_ipip' crates/cintx-oracle/tests/hess1e_ipip_parity.rs && grep -cqE 'ipipovlp\|ipipnuc\|ipipkin\|ipiprinv' crates/cintx-oracle/tests/hess1e_ipip_parity.rs` | ❌ W0 (`tests/hess1e_ipip_parity.rs`) | ⬜ pending |
| 03-T1 register 4 rank-9 1e Hessian families | 03 | 2 | HESS-01 | build | `cargo build -p cintx-ops && grep -cqE 'INT1E_IPIPNUC' crates/cintx-compat/src/raw.rs` | ✅ (T0) | ⬜ pending |
| 03-T2 vendor FFI allowlist + parity green + audit | 03 | 2 | HESS-01 | integration (vendor) + xtask | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu hess1e_ipip && cargo run -p xtask -- manifest-audit` | ✅ (T0) | ⬜ pending |
| 04-T0 hess2e_parity scaffold (4 families incl. rank 81) | 04 | 2 | HESS-02 | scaffold | `grep -qc 'fn hess2e_ipip' crates/cintx-oracle/tests/hess2e_parity.rs` | ❌ W0 (`tests/hess2e_parity.rs`) | ⬜ pending |
| 04-T1 re-home ipip1/ipvip1 + register ip1ip2/ipip1ipip2 | 04 | 2 | HESS-02 | build | `grep -c 'unstable::source::2e' crates/cintx-ops/generated/compiled_manifest.lock.json \| grep -qx 0 && cargo build -p cintx-ops && cargo build -p cintx-cubecl` | ✅ (T0) | ⬜ pending |
| 04-T2 vendor FFI allowlist + parity green + audit | 04 | 2 | HESS-02 | integration (vendor) + xtask | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu hess2e_ipip && cargo run -p xtask -- manifest-audit` | ✅ (T0) | ⬜ pending |
| 05-T0 hess_multicenter_ipip_parity scaffold (3 families) | 05 | 2 | HESS-03 | scaffold | `grep -qc 'fn hess_multicenter_ipip' crates/cintx-oracle/tests/hess_multicenter_ipip_parity.rs` | ❌ W0 (`tests/hess_multicenter_ipip_parity.rs`) | ⬜ pending |
| 05-T1 register int2c2e_ipip1, int3c2e_ipip1/ipip2 | 05 | 2 | HESS-03 | build | `cargo build -p cintx-ops && cargo build -p cintx-cubecl && grep -cqE 'INT3C2E_IPIP2' crates/cintx-compat/src/raw.rs` | ✅ (T0) | ⬜ pending |
| 05-T2 vendor FFI allowlist + parity green + audit | 05 | 2 | HESS-03 | integration (vendor) + xtask | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu hess_multicenter_ipip && cargo run -p xtask -- manifest-audit` | ✅ (T0) | ⬜ pending |
| 06-T0 roster lock + deriv3.c/deriv4.c build wiring | 06 | 2 | HESS-04 | build | `grep -qc 'deriv3.c' crates/cintx-oracle/build.rs && grep -qc 'deriv4.c' crates/cintx-oracle/build.rs && CINTX_ORACLE_BUILD_VENDOR=1 cargo build -p cintx-oracle --features cpu` | ✅ (source) | ⬜ pending |
| 06-T1 deriv34_parity scaffold (NON-SQUARE, ranks 27/81) | 06 | 2 | HESS-04 | scaffold | `grep -qc 'fn deriv34_ipipip' crates/cintx-oracle/tests/deriv34_parity.rs && grep -cqE 'int1e_ipipipnuc\|int1e_ipipipiprinv' crates/cintx-oracle/tests/deriv34_parity.rs` | ❌ W0 (`tests/deriv34_parity.rs`) | ⬜ pending |
| 06-T2 register deriv3 (rank-27) roster + parity | 06 | 2 | HESS-04 | build + integration (vendor) | `grep -A4 'int1e_ipipipnuc' crates/cintx-ops/generated/compiled_manifest.lock.json \| grep -qc '"27"' && cargo build -p cintx-ops && cargo build -p cintx-cubecl` | ✅ (T1) | ⬜ pending |
| 06-T3 register deriv4 (rank-81) roster + dual headroom + audit | 06 | 2 | HESS-04 | integration (vendor) + xtask | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu deriv34_ipipip && cargo run -p xtask -- manifest-audit` | ✅ (T1) | ⬜ pending |

### Requirement → Test anchors (from research)

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-02 | nroots 6..12 roots/weights byte-identical vs vendor `CINTrys_roots` | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu rys_nroots_sweep` | ❌ Wave 0 (`tests/rys_nroots_sweep_parity.rs`) |
| FND-02 | host `rys_roots_host(6..12)` no longer panics | unit | `cargo test -p cintx-cubecl --lib rys_host_nroots_ge6` | ❌ Wave 0 |
| FND-02 | eigh.c #else MRRR eigensolver validated on a known tridiagonal spectrum | unit | `cargo test -p cintx-cubecl --lib eigh_mrrr_tridiag` | ❌ Wave 0 (`math/eigh.rs`) |
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
- [ ] `math/eigh.rs` unit test `eigh_mrrr_tridiag` — the ~1400-line MRRR eigensolver validated independently on a known small tridiagonal spectrum (Plan-01 Task 1a, the highest-risk FND-02 piece).
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

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
</content>
