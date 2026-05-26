---
phase: 10
slug: 2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-03
audited: 2026-05-26
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Reconciled against executed artifacts on 2026-05-26 (see Validation Audit below).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) + cargo nextest |
| **Config file** | `Cargo.toml` workspace test config |
| **Quick run command** | `cargo test -p cintx-cubecl --features cpu --lib -- kernels` |
| **Full suite command** | `cargo test -p cintx-cubecl --features cpu && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` |
| **Estimated runtime** | ~10s quick; vendor parity adds a one-time vendored-libcint `cc` build (minutes on first run) |

> **Vendor parity is double-gated.** The per-family parity tests are compiled under `#[cfg(has_vendor_libcint)]`, which is only set when **both** `--features cpu` **and** `CINTX_ORACLE_BUILD_VENDOR=1` are present. Without the env var the parity tests are cfg'd out and only determinism/structure/rocm tests run — parity silently does not execute.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu --lib -- kernels`
- **After every plan wave:** Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu`
- **Before `/gsd:verify-work`:** Full suite must be green (with `CINTX_ORACLE_BUILD_VENDOR=1`)
- **Max feedback latency:** ~10s (quick), minutes for full vendor-parity sweep

---

## Per-Task Verification Map

| Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01 | 1 | VERI-05 (shared infra) | unit | `cargo test -p cintx-cubecl --features cpu --lib -- math::rys transform::c2s` | ✅ | ✅ green |
| 10-02 | 2 | KERN-03 | oracle parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test center_2c2e_parity` | ✅ | ✅ green |
| 10-03 | 2 | KERN-04 | oracle parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test center_3c1e_parity` | ✅ | ✅ green |
| 10-04 | 2 | KERN-05 | oracle parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test center_3c2e_parity` | ✅ | ✅ green |
| 10-05 | 2 | KERN-02 | oracle parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test two_electron_parity` | ✅ | ✅ green |
| 10-06 | 3 | VERI-05, VERI-07 | gate + UAT | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test oracle_gate_closure` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Tolerances (per gate closure report):** 1e (atol 1e-11), 2e (atol 1e-12 / rtol 1e-10), 2c2e (atol 1e-9), 3c1e (atol 1e-7), 3c2e (atol 1e-9) — all `mismatch_count == 0`.

**Key test functions:**
- `test_int2c2e_sph_h2o_sto3g_vendor_parity`, `test_int3c1e_sph_h2o_sto3g_vendor_parity`, `test_center_3c2e_sph_h2o_sto3g_vendor_parity`, `test_int2e_sph_h2o_sto3g_vendor_parity`
- `oracle_gate_all_five_families`, `uat_eval_raw_returns_nonzero`, `uat_cabi_returns_status_zero`

---

## Wave 0 Requirements

All shared-infrastructure prerequisites were delivered in Plan 10-01 and are exercised by `cintx-cubecl` unit tests:

- [x] `crates/cintx-cubecl/src/math/rys.rs` — `rys_root3_host`, `rys_root4_host`, `rys_root5_host`, `rys_roots_host` dispatcher + `tests_rys_host`
- [x] `crates/cintx-cubecl/src/transform/c2s.rs` — `cart_to_sph_2e`, `cart_to_sph_2c2e`, `cart_to_sph_3c1e`, `cart_to_sph_3c2e` + 13 unit tests
- [x] `crates/cintx-oracle/build.rs` — compiles 2e/2c2e/3c1e/3c2e C sources + autocode; supplemental bindgen header declares `int2c2e_sph`/`int3c1e_sph`/`int3c2e_sph`; allowlist extended
- [x] `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int2e_sph`, `vendor_int2c2e_sph`, `vendor_int3c1e_sph`, `vendor_int3c2e_sph`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| C ABI shim on real GPU hardware | VERI-07 (supplementary) | Requires physical GPU; not reproducible in CI | Build with wgpu backend, call `cintrs_eval()` on `int1e_ovlp_sph`, verify `status == 0` |

> **Not a coverage gap.** VERI-07 is fully satisfied by automated tests (`uat_cabi_returns_status_zero` validates the C ABI `not0 > 0 ⇒ status == 0` logic via the `eval_raw` proxy on CPU, documented as intentional because `cintx-capi` is not reachable from the `cintx-oracle` crate). The row above is a supplementary hardware confirmation on real GPU — every requirement already has automated verification on the CPU path.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s (quick command)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (2026-05-26) — all six requirements (KERN-02/03/04/05, VERI-05, VERI-07) have automated verification that exists and runs green.

---

## Validation Audit 2026-05-26

State A audit of the planning-time `draft` against executed artifacts (six SUMMARYs, 10-VERIFICATION.md 13/13, `artifacts/oracle_gate_closure_report.txt` = `GATE: PASS`, on-disk parity test files, 45/45 `cintx-cubecl` kernel unit tests green).

| Metric | Count |
|--------|-------|
| Requirements audited | 6 |
| COVERED | 6 |
| PARTIAL | 0 |
| MISSING | 0 |
| Gaps found | 0 |
| Gaps resolved | 0 |
| Escalated | 0 |

**Findings:**
- The draft map had off-by-one plan numbering (KERN-03 attributed to 10-01 etc.) and listed every task as `⬜ pending` with `❌ W0`. Corrected the map to the actual plan→requirement→test mapping from each SUMMARY's `requirements-completed`.
- All parity tests exist on disk and are correctly `#[cfg(has_vendor_libcint)]`-gated; later phases also added `*_rocm_parity` and spinor/4c1e gate tests in the same files.
- Updated the full-suite/oracle commands to include `CINTX_ORACLE_BUILD_VENDOR=1` — the prior commands would have silently skipped vendor parity.
- No test files generated (none missing); only this validation doc changed.
