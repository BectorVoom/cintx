---
phase: 23
slug: group-1-remaining-1st-derivative-families-cart-sph
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-30
---

# Phase 23 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Scope: clusters A & B only (cluster C — DRV1-02, rank-9 both-side — already shipped in commit 319d055 and vendor-verified).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]`; oracle parity via `#[cfg(has_vendor_libcint)]` (cargo-nextest available but not required) |
| **Config file** | none (cargo test); CI in `.github/workflows/compat-governance-pr.yml` |
| **Quick run command** | `cargo test -p cintx-cubecl <family>` (device-vs-host host-ref, no vendor) |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test <family>_parity -- --test-threads=1` |
| **Estimated runtime** | ~60–180 s per family parity test (vendored libcint build is cached after first run) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl <family>` (device-vs-host host-ref test).
- **After every plan wave:** Run the family's `vendor_*` parity test with the double gate (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`) + `cargo build -p cintx-ops` (manifest regen) + `cargo run -p xtask -- manifest-audit` (no flags).
- **Before `/gsd-verify-work`:** All new parity tests green at atol=1e-12 + `manifest-audit` green.
- **Max feedback latency:** ~180 seconds (single family parity run, vendor cached).

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| A-int2e_ip2 | A | — | DRV1-01 | — | N/A (numeric kernel; no external input surface) | vendor parity + cubecl unit | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test int2e_ip2_parity` | ❌ W0 | ⬜ pending |
| A-int2c2e_ip1/ip2 | A | — | DRV1-04 | — | N/A | vendor parity + cubecl unit | `... --test int2c2e_ip_parity` | ❌ W0 | ⬜ pending |
| A-int3c2e_ip2 | A | — | DRV1-05 | — | N/A | vendor parity + cubecl unit | `... --test int3c2e_ip2_parity` | ❌ W0 | ⬜ pending |
| B-int3c1e_ip1/iprinv | B | — | DRV1-03 | — | N/A | vendor parity + cubecl unit | `... --test int3c1e_ip_parity` | ❌ W0 | ⬜ pending |
| (all) | A,B | — | DRV1-01/03/04/05 | — | device kernel == host reference | unit (cubecl) | `cargo test -p cintx-cubecl <family>` | ❌ W0 (clone cluster-C `test_device_*`) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Wave assignments finalized by the planner; this map tracks the requirement→test coverage, not wave order.*

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/int2e_ip2_parity.rs` — covers DRV1-01 (clone `one_electron_grad_both_parity.rs` fixture + the `int2e` shell-quad pattern)
- [ ] `crates/cintx-oracle/tests/int2c2e_ip_parity.rs` — covers DRV1-04 (both ip1 and ip2)
- [ ] `crates/cintx-oracle/tests/int3c2e_ip2_parity.rs` — covers DRV1-05
- [ ] `crates/cintx-oracle/tests/int3c1e_ip_parity.rs` — covers DRV1-03 (both ip1 overlap base + iprinv rinv base)
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — add `vendor_int{2e_ip2,2c2e_ip1,2c2e_ip2,3c2e_ip2,3c1e_ip1,3c1e_iprinv}_{sph,cart}` wrappers (clone the `vendor_int1e_ipovlpip_*` pattern at `:587`)
- [ ] `crates/cintx-cubecl/src/kernels/*` device-vs-host unit tests per family (clone cluster-C `test_device_ip{ovlpip,kinip,nucip}_matches_host_reference`)
- [ ] Framework install: none — built-in `cargo test` runner already present.

*Each new parity test must assert (a) element count = `component_rank * n_ao * n_ao` (or the 3c shape analog) to catch a too-low `component_rank` truncation (D-14), AND (b) `any_nonzero` so a stub/short buffer cannot pass parity by matching zeros, AND (c) a NON-SQUARE bra/ket block where applicable so a transposed layout cannot pass (D-05 discipline).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `int3c1e_iprinv` fail-closed at nroots>5 (fff → nroots 6) | DRV1-03 | Negative path: the device Rys ceiling rejects fff; can be asserted but the corpus (H2O/STO-3G + Cu/LANL2DZ) does not naturally exercise fff | Add an explicit unit assertion that the iprinv launcher returns `UnsupportedApi`/fails closed for an fff shell tuple (do not silently truncate). |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (5 new test files + vendor_ffi wrappers)
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
