---
phase: 19
slug: int1e-ecp-type1-type2-evaluator
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-12
---

# Phase 19 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` / `cargo nextest run` (workspace-wide) |
| **Config file** | `Cargo.toml` + `crates/*/Cargo.toml` + `.config/nextest.toml` (if present) |
| **Quick run command** | `cargo nextest run -p cintx-cubecl -p cintx-core -p cintx-compat -p cintx-ops --no-fail-fast` |
| **Full suite command** | `cargo nextest run --workspace --no-fail-fast --locked` |
| **Oracle parity command** | `cargo nextest run -p cintx-oracle --features=oracle-libcint --no-fail-fast --locked` |
| **Estimated runtime** | Quick: ~30s. Full: ~3–6 min. Oracle parity: ~2–5 min (Cu/LANL2DZ Cartesian product over ECP shells). |

---

## Sampling Rate

- **After every task commit:** Run quick command for the touched crate (e.g., `cargo nextest run -p cintx-cubecl`).
- **After every plan wave:** Run full suite + oracle parity gate.
- **Before `/gsd:verify-work`:** Full suite + oracle parity must be green at `atol=1e-12, rtol=0.0`.
- **Max feedback latency:** 60s for crate-local quick run.

---

## Eval Dimensions

| Dim | Name | Type | Oracle / Source | Threshold | Blocking |
|-----|------|------|------|-----------|----------|
| D1 | Byte-identity vs PySCF nr_ecp | Numerical parity | Vendored PySCF `nr_ecp.{c,h}` + `nr_ecp_deriv.c` via cintx-oracle FFI | `atol=1e-12, rtol=0.0` | Yes |
| D2 | Secondary cross-check vs libecpint | Numerical cross-check | libecpint (MIT, JCP 2017) — opt-in `CINTX_LIBECPINT_ORACLE=1` | `atol=1e-9` (informational envelope) | No |
| D3 | Hermiticity / symmetry properties | Property test | Compute `int1e_ecp_{sph,cart}` and verify `M = M^T` on shell-pair tuples that should be symmetric | `||M - M^T||_max < 1e-13` | Yes |
| D4 | Rotational invariance (Type-2 only) | Property test | Eigenvalues / trace invariant under SO(3) basis rotations | `|tr(M) - tr(R M R^T)| < 1e-10` for sampled R | Yes |
| D5 | Manifest / API surface coverage | Compile-time + helper-parity | `crates/cintx-ops/src/generated/compiled_manifest.lock.json` includes 4 new rows × 4 profiles; `helper_parity` test passes | All 4 profiles compile and lock matches CSV | Yes |
| D6 | SessionRequest dispatch | Integration | `safe_api_ecp_parity.rs` exercises `SessionRequest::evaluate("int1e_ecp_{sph,cart}", ...)` and matches D1 | Same as D1 | Yes |
| D7 | Gradient component correctness | Numerical parity | `int1e_ecp_ipnuc_{sph,cart}` vs PySCF `ECPscalar_ipnuc_*` — 3 components per cell | `atol=1e-12, rtol=0.0` | Yes |
| D8 | Nyquist sampling adequacy | Coverage | Every task in every plan has automated verify or maps to Wave 0 install | No 3 consecutive non-automated tasks | Yes |

---

## Per-Task Verification Map

> Filled by planner during step 8 once tasks exist. Below are anchor expectations for the planner.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 19-01-XX | 01 (math infra) | 1 | ECP-02 | — | N/A | unit | `cargo test -p cintx-cubecl --lib math::bessel` | ❌ W0 | ⬜ pending |
| 19-01-XX | 01 (math infra) | 1 | ECP-02 | — | N/A | unit | `cargo test -p cintx-cubecl --lib math::radial_quadrature` | ❌ W0 | ⬜ pending |
| 19-02-XX | 02 (core types + raw compat) | 1 | ECP-03 | — | `MissingEcpBasis` returned when ECP operator dispatched without ECP shells | unit | `cargo test -p cintx-core ecp_shell` + `cargo test -p cintx-compat ecpbas_array` | ❌ W0 | ⬜ pending |
| 19-03-XX | 03 (Type-1 kernel + parity) | 2 | ECP-01, ECP-04 | — | N/A | parity | `cargo nextest run -p cintx-oracle ecp_type1` | ❌ W0 | ⬜ pending |
| 19-04-XX | 04 (Type-2 kernel + parity) | 2 | ECP-02, ECP-04 | — | N/A | parity | `cargo nextest run -p cintx-oracle ecp_type2` | ❌ W0 | ⬜ pending |
| 19-05-XX | 05 (gradient + parity) | 3 | ECP-01, ECP-02, ECP-04 | — | N/A | parity | `cargo nextest run -p cintx-oracle ecp_ipnuc` | ❌ W0 | ⬜ pending |
| 19-06-XX | 06 (libecpint cross-check, optional) | 3 | ECP-04 | — | N/A | cross-check | `CINTX_LIBECPINT_ORACLE=1 cargo nextest run -p cintx-oracle ecp_libecpint -- --ignored` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 0 ("install / scaffold") lands once before any kernel work. It must complete before parity tests can run.

- [ ] `vendor/pyscf-nr-ecp/` (or `pyscf-nr-ecp-master/`) vendor subtree containing PySCF `pyscf/lib/gto/nr_ecp.{c,h}` + `nr_ecp_deriv.c` (Apache-2.0). Include LICENSE + provenance note.
- [ ] `crates/cintx-oracle/build.rs` extended to compile the vendored PySCF nr_ecp sources via a parallel `cc::Build`.
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` (or sibling) declares the `extern "C"` symbols for `ECPscalar_sph`, `ECPscalar_cart`, `ECPscalar_ipnuc_sph`, `ECPscalar_ipnuc_cart` (or whatever names the vendored headers actually export — confirm during execution).
- [ ] `crates/cintx-oracle/src/fixtures.rs::build_cu_lanl2dz()` builder added (LANL2DZ Cu basis + ECP, PTR_ENV_START-aligned).
- [ ] `crates/cintx-cubecl/src/math/bessel.rs` and `radial_quadrature.rs` empty-but-compiling stubs with `#[cube]` + `*_host()` signatures.
- [ ] `crates/cintx-core/src/ecp.rs` (or sibling) with empty `EcpShell` struct.
- [ ] `crates/cintx-ops/src/generated/api_manifest.csv` adds 4 new rows; `compiled_manifest.lock.json` regenerated via `cargo run -p xtask -- manifest-audit --update`.

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LANL2DZ parameter accuracy | ECP-04 | LANL2DZ basis must be sourced from a canonical reference (basissetexchange.org or PySCF basis library) — value provenance is checked once by a human against the published Hay-Wadt 1985 JCP papers before fixture commits. | Inspect `build_cu_lanl2dz()` rustdoc for source URL + published-paper citation; cross-check ~5 exponent/coefficient values against the cited source. |
| PySCF nr_ecp license + provenance | ECP-01, ECP-02, ECP-04 | License compatibility (Apache-2.0 ↔ cintx workspace license) is a human/legal judgment. | Inspect vendor subtree LICENSE file; verify NOTICE / provenance comment lists upstream commit SHA and Apache-2.0 grant. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (PySCF vendor, fixtures, math stubs)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for crate-local quick run
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
