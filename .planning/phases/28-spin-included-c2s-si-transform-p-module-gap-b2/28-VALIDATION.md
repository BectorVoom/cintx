---
phase: 28
slug: spin-included-c2s-si-transform-p-module-gap-b2
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-31
---

# Phase 28 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from the D-06 hard-gate spike (28-RESEARCH.md `## Validation Architecture`).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` / `cargo nextest` (Rust 1.94.0 pinned) |
| **Config file** | `Cargo.toml` workspace; vendor parity gated on `--features cpu` + env `CINTX_ORACLE_BUILD_VENDOR=1` |
| **Quick run command** | `cargo test -p cintx-cubecl transform::c2spinor` |
| **Full suite command** | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu -p cintx-oracle` |
| **Estimated runtime** | ~60–180 seconds (vendor build dominates first run) |

> **Double-gate landmine:** Without BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`, vendor parity tests silently skip (determinism-only). The si_2d byte-identity proof MUST run under both gates. Add the no-silent-skip assertion (Phase 27 D-10 pattern).

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-cubecl transform::c2spinor` (transform-level)
- **After every plan wave:** Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu -p cintx-oracle` (vendor parity)
- **Before `/gsd-verify-work`:** Full suite green, including the `int1e_sp` si_2d byte-identity test at atol=1e-12
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

> Populated by the planner once PLAN.md files exist. Each σ·p / si_2d task maps to a Validation-Architecture spike target (A–E) in 28-RESEARCH.md.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 28-XX-XX | XX | X | FND-05 | — | N/A (pure-compute lib) | unit/parity | `cargo test ...` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `apply_bra_si_block` unit test — verifies the NEW bra-si sign convention (`a_bra_cart2spinor_si`, `cart2sph.c:3958`: `+caR*v1 +caI*vz -cbR*vy +cbI*vx`) is NOT the existing `apply_si_block` (3-of-4 cross terms differ). **Critical: reusing `apply_si_block` silently fails at atol=1e-12.**
- [ ] `build_kappa_spinor_fixture` in `fixtures.rs` — kappa≠0 (p kappa=+1 LT di=2, d kappa=−1 GT dj=6), non-square, nctr=2.
- [ ] `vendor_int1e_sp_spinor` FFI shim in `vendor_ffi.rs` for the byte-identity reference.

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Heavy-atom realism cross-check (D-05 secondary) | FND-05 | Realism guard against synthetic-fixture blind spots; not a primary gate | Run the small real heavy-atom 2c-basis fixture through the si_2d path; confirm no panic and finite output |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
