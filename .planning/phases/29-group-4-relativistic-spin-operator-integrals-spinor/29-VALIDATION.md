---
phase: 29
slug: group-4-relativistic-spin-operator-integrals-spinor
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-31
---

# Phase 29 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `29-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `#[cfg(has_vendor_libcint)]` double-gate; oracle integration tests under `crates/cintx-oracle/tests/`. `cargo nextest` available. |
| **Config file** | none (cargo test harness); gate cfg emitted by `cintx-oracle/build.rs` |
| **Quick run command** | `cargo test -p cintx-cubecl --features cpu --lib c2spinor` (transform unit tests, fast) |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test '*spinor*'` |
| **Estimated runtime** | ~30–90 s quick; vendor parity suite minutes (cc rebuild of vendored libcint on first run) |

> **Double-gate (project memory `reference_oracle_vendor_parity_invocation`):** real parity requires BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`. Without both, vendor bodies compile out and parity silently SKIPS (determinism-only). Every new parity test MUST carry the Phase-27 D-10 NO-SILENT-SKIP assertion that FAILS (not skips) if the vendor arm did not run when the double-gate was present.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl --features cpu --lib c2spinor` + `cargo clippy`
- **After every plan wave:** Run the full vendor-parity suite for that wave's families under the double-gate; `manifest-audit` green
- **Before `/gsd-verify-work`:** Full suite must be green; all REL-01..04 families `oracle_covered=true` (spinor-only)
- **Max feedback latency:** ~90 s (quick) / wave-merge for parity

---

## Per-Wave / Per-Requirement Verification Map

> Plan-level task IDs assigned during planning; this map is the requirement→test contract the executor refines into per-task `<automated>` blocks.

| Wave | Requirement | Family / behavior | Transform path | Test Type | Automated Command | File Exists |
|------|-------------|-------------------|----------------|-----------|-------------------|-------------|
| 1 | REL-01 | `int1e_spsp` byte-identity | `c2s_sf_1e` (spin-free contraction) | vendor parity | `…--test rel_1e_sigma_parity test_spsp` | ❌ W0 |
| 1 | REL-01 | `int1e_spnucsp`/`sprinvsp` | `c2s_si_1e` (2-/1-center) | vendor parity | `…--test rel_1e_sigma_parity test_spnucsp test_sprinvsp` | ❌ W0 |
| 1 | REL-02 | `int1e_sp` (flip from `oracle_covered=false`) | `c2s_si_1e` | vendor parity | `…--test rel_1e_sigma_parity test_sp` | ❌ W0 |
| 1 | REL-02 | `int1e_srsr`/`srnucsr` | `c2s_si_1e` | vendor parity | `…test_srsr test_srnucsr` | ❌ W0 |
| 1 | REL-02 | `int1e_sr`/`sigma` | `c2s_si_1ei` (new 1e `si_2di` imaginary-ket) | vendor parity | `…test_sr test_sigma` | ❌ W0 |
| 2 | (gating) | `c2s_si_2e1/2e2(+i)` + `c2s_sf_2e1/2e2` transform byte-identity (D-03 micro-test, FIRST task) | — | transform parity | `…--test si_2e_transform_parity` (via thinnest family `int2e_spsp1_spinor`) | ❌ W0 |
| 3 | REL-03 | `int2e_spsp1`/`srsr1` | `c2s_si_2e1`+`c2s_sf_2e2` | vendor parity | `…--test rel_2e_sigma_parity test_spsp1 test_srsr1` | ❌ W0 |
| 3 | REL-03 | `int2e_spsp1spsp2`/`srsr1srsr2` | `c2s_si_2e1`+`c2s_si_2e2` | vendor parity | `…test_spsp1spsp2 test_srsr1srsr2` | ❌ W0 |
| 3 | REL-04 | `int2e_ssp1ssp2`/`sps1sps2` (needs `gaunt1.c` in build) | `c2s_si_2e1i`+`c2s_si_2e2i` | vendor parity | `…test_ssp1ssp2 test_sps1sps2` | ❌ W0 |
| 3 | REL-04 | `int2e_vsp1`/`spv1` (+2-sided) (needs `dkb.c` in build) | `c2s_si_2e1`+`c2s_sf_2e2` / +`c2s_si_2e2` | vendor parity | `…test_vsp1 test_spv1` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Sampling / Coverage Strategy

- **Primary gate fixture — `build_kappa_spinor_2e_fixture` (D-02):** 4 spinor shells, NON-SQUARE (defeats transpose symmetry), genuine kappa≠0 GT/LT mix (stresses `2l`/`2l+2` sizing, not just `4l+2`), ≥1 shell `nctr>1` (catches coeff transpose). Byte-identity gate for every REL-03/04 family.
- **Wave-1 1e fixture:** reuse Phase-28 `build_kappa_spinor_fixture` (p kappa=+1 LT × d kappa=−1 GT, nctr=2).
- **Realism cross-check — `build_heavy_atom_spinor_fixture` (secondary):** asserted finite, NOT the primary gate; guards synthetic-fixture blind spots.
- **D-03 ordering invariant:** the Wave-2 transform micro-test MUST be GREEN before any Wave-3 family wires onto the transform.

### "Covered, non-skipped" per family
A Group-4 family is legitimately `oracle_covered=true` only when: (1) its `vendor_*` shim links a real libcint 6.1.3 driver (REL-04 ⇒ `gaunt1.c`/`dkb.c` MUST be in `build.rs`); (2) its parity test runs on a kappa-bearing fixture with N>0 byte-comparisons under BOTH gate flags; (3) the NO-SILENT-SKIP assertion confirms the vendor arm executed (not `skipped`); (4) atol=1e-12 byte-identity holds. The `xtask oracle_covered_update` SC#4 guard must refuse to flip any family whose only fixture was `skipped`. Flip `oracle_covered=true` **spinor-only** (SC#5 — do not over-claim cart/sph σ intermediates).

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` — REL-01/02 (new)
- [ ] `crates/cintx-oracle/tests/si_2e_transform_parity.rs` — Wave-2 gating micro-test (new)
- [ ] `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` — REL-03/04 (new)
- [ ] `build_kappa_spinor_2e_fixture` in `crates/cintx-oracle/src/fixtures.rs` (new, D-02)
- [ ] `vendor_int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor` shims
- [ ] `vendor_int2e_{spsp1,srsr1,spsp1spsp2,srsr1srsr2,ssp1ssp2,sps1sps2,vsp1,spv1,…}_spinor` shims
- [ ] `gaunt1.c` + `dkb.c` added to `cintx-oracle/build.rs` (REL-04 — BLOCKING)
- [ ] new 1e `cart_to_spinor_si_2di` (imaginary-ket) in `c2spinor.rs` for `sr`/`sigma`
- [ ] 2e transform suite (`c2s_si_2e1/2e2(+i)` + `c2s_sf_2e1/2e2`) in `c2spinor.rs`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `int1e_sigma` output rank (component_rank 1 vs 3) | REL-02 | Empirical shape check needed before locking the manifest row (Open Q1 / Assumption A2) | Wave-1 first task: call `vendor_int1e_sigma_spinor` on the kappa fixture, measure output length vs `di*dj*2`; set `component_rank` from the measured shape. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s (quick) / wave-merge (parity)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
