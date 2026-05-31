---
phase: 27
slug: spinor-derivative-transform-gap-b1
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-31
---

# Phase 27 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 27-RESEARCH.md § Validation Architecture (confidence HIGH).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (cargo test); oracle integration tests in `crates/cintx-oracle/tests/`; `cargo nextest` available |
| **Config file** | none dedicated (workspace `Cargo.toml`; no `nextest.toml`) |
| **Quick run command** | `cargo test -p cintx-cubecl --lib transform::c2spinor && cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (smoke; no vendor) |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` + `cargo run -p xtask -- manifest-audit` |
| **Estimated runtime** | ~60–120 seconds (vendor build dominates first run) |

> **Double-gate warning (D-10 / project landmine):** real vendor byte-identity comparison requires BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`. Without both, parity tests silently degrade to determinism-only and report a false pass. The `test_no_silent_skip` assertion guards this.

---

## Sampling Rate

- **After every task commit:** `cargo test -p cintx-cubecl --lib transform::c2spinor && cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (fast, smoke arms)
- **After every plan wave:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (full vendor byte-identity)
- **Before `/gsd-verify-work`:** full vendor suite green AND `cargo run -p xtask -- manifest-audit` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

> Task IDs are assigned by the planner; this map enumerates the required verifications by behavior. The planner MUST bind each behavior below to at least one task's `<automated>` verify (or a Wave 0 dependency).

| Behavior (FND-04) | Wave | Test Type | Automated Command | File Exists | Status |
|-------------------|------|-----------|-------------------|-------------|--------|
| `cart_to_spinor_sf_derivative_2d` folds ncomp axis (rank 3 — `int1e_ipovlp_spinor`) | — | integration (vendor parity) | `... --test spinor_deriv_parity test_int1e_ipovlp_spinor_adversarial_parity` | ❌ W0 | ⬜ pending |
| sf_2d rank-9 axis-fold (`int1e_ipovlpip_spinor`) | — | integration | `... test_int1e_ipovlpip_spinor_adversarial_parity` | ❌ W0 | ⬜ pending |
| sf_2d highest-rank (27/81 `ipip*`) axis-fold | — | integration | `... test_int1e_ipip<X>_spinor_adversarial_parity` | ❌ W0 | ⬜ pending |
| sf_3c2e rank-3 axis-fold (`int3c2e_ip1_spinor`) | — | integration | `... test_int3c2e_ip1_spinor_adversarial_parity` | ❌ W0 | ⬜ pending |
| 2c2e via sf_2d (`int2c2e_ip1_spinor`) | — | integration | `... test_int2c2e_ip1_spinor_adversarial_parity` | ❌ W0 | ⬜ pending |
| nctr>1 spinor general contraction (D-08) | — | integration | covered by adversarial fixture in every test above | ❌ W0 | ⬜ pending |
| No-silent-skip coverage assertion (D-10) | — | integration | `... test_no_silent_skip` (asserts `N>0` ran + flipped families `oracle_covered=true`) | ❌ W0 | ⬜ pending |
| Manifest audit green after flip | — | xtask | `cargo run -p xtask -- manifest-audit` | ✓ | ⬜ pending |
| Wrapper unit transpose/stride correctness | — | unit | `cargo test -p cintx-cubecl --lib transform::c2spinor` | ✓ (mod exists) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Observable Signals (what proves the axis-fold is correct)
- **Per rank tier (3/9/27/81):** a `ncomp*di*dj*2` buffer splits into exactly `ncomp` non-overlapping, all-nonzero component slices (no trailing-zero, no truncation — component_rank-truncation landmine).
- **Orientation:** vendor byte-identity on a NON-SQUARE p×d block at atol=1e-12; a j-fastest reindex negative control must diverge (`mismatches>0`).
- **Both paths:** `sf_2d` (1e + 2c2e) and `sf_3c2e` (3c2e) each have ≥1 passing parity test.
- **nctr>1:** the general-contraction fixture produces vendor-identical output with contraction-major composition (no coefficient transpose).
- **No-silent-skip:** parity binary reports `running N>0 tests` under both flags; flipped families read `oracle_covered=true`; `manifest-audit` green.

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/spinor_deriv_parity.rs` — D-09 per-path×per-rank parity tests + D-10 no-silent-skip assertion (new file)
- [ ] `crates/cintx-oracle/src/fixtures.rs` — D-08 adversarial fixture builder (non-square p×d + nctr>1 + kappa=0); model on `int3c1e_genctr_parity.rs::build_genctr_fixture`
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — extern decls + wrappers for rank-9/27/81 1e ip-spinor, `int3c2e_ip1/ip2_spinor`, `int2c2e_ip1/ip2_spinor`, `int3c1e_ip1/iprinv_spinor` (only the four rank-3 1e exist today)
- [ ] `crates/cintx-cubecl/src/transform/c2spinor.rs` — `#[test]` unit coverage for the new wrappers (stride/transpose)
- [ ] Framework install: none — built-in `cargo test`; `nextest` optional

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| D-11 design spike: confirm device-emitted derivative cart block layout (`[comp][ket][bra]` component-outer) and 3c2e transpose granularity | FND-04 | Empirical layout discovery against hand-checked vendor values; precedes finalizing the `_3c2e` wrapper and nctr>1 composition | Run `/gsd:spike` exercising the full per-component axis-fold across all rank tiers and both transform paths before plan tasks (a) and the 3c2e portion are finalized |

*All shipped behaviors have automated verification; the spike above is a pre-implementation design probe, not a shippable behavior.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
