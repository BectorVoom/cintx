---
phase: 20
slug: precision-generic-f64-f32-switch
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-20
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `20-RESEARCH.md` §"Validation Architecture". Task IDs are seeded at
> the requirement level; the planner refines them to `20-NN-MM` per-task rows.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` + `cargo nextest` (if available) |
| **Config file** | `rust-toolchain.toml` (pinned 1.94.0) |
| **Quick run command** | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` |
| **Full suite command** | `CINTX_BACKEND=cpu cargo test --workspace --features cpu` |
| **Estimated runtime** | ~quick: <60s · full: minutes (oracle gates dominate) |

---

## Sampling Rate

- **After every task commit:** Run `CINTX_BACKEND=cpu cargo check --workspace --features cpu`
- **After every plan wave:** Run `CINTX_BACKEND=cpu cargo test --workspace --features cpu 2>&1 | tail -20`
- **Before `/gsd:verify-work`:** Full f64 oracle suite (all four profiles) must be green; f32 oracle gate advisory-green
- **Max feedback latency:** <60 seconds (quick `cargo check`)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 1 | PREC-01 | — | N/A | unit compile | `cargo check -p cintx-cubecl --features cpu` | ❌ W1 | ⬜ pending |
| TBD | TBD | 1 | PREC-01 | — | N/A | unit | `cargo test -p cintx-cubecl boys_f32 --features cpu` | ❌ W1 | ⬜ pending |
| TBD | TBD | 4 | PREC-02 | — | N/A | regression | `cargo test -p cintx-oracle --features cpu` | ✅ existing | ⬜ pending |
| TBD | TBD | 4 | PREC-02 | — | N/A | compile smoke | `cargo check -p cintx-rs` | ✅ existing | ⬜ pending |
| TBD | TBD | 4 | PREC-02 | — | N/A | unit | `cargo test -p cintx-rs f32_evaluate --features cpu` | ❌ W4 | ⬜ pending |
| TBD | TBD | 3 | PREC-03 | — | env/atm/bas stay `&[f64]` | regression | `cargo test -p cintx-compat --features cpu` | ✅ existing | ⬜ pending |
| TBD | TBD | 5 | PREC-04 | — | f64 byte-identity preserved | integration | `cargo test -p cintx-oracle --features cpu` | ✅ existing | ⬜ pending |
| TBD | TBD | 5 | PREC-05 | — | N/A | integration | `cargo test -p cintx-oracle f32_parity --features cpu` | ❌ W5 | ⬜ pending |
| TBD | TBD | 3 | PREC-06 | — | f32 path does NOT gate on SHADER_F64 | smoke | `CINTX_BACKEND=cpu cargo test f32_smoke --features cpu` | ❌ W3 | ⬜ pending |
| TBD | TBD | 0 | PREC-07 | — | serena symbol-aware refactor only | process gate | serena `check_onboarding_performed` | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `cintx-core/src/precision.rs` — `CintFloat` sealed trait + `PrecisionKind { F64, F32 }` enum
- [ ] `ExecutionPlan` gains a `PrecisionKind` field (non-generic, preserves `&dyn BackendExecutor` object safety)
- [ ] Confirm `num-traits` is a direct dep in `cintx-core/Cargo.toml` (currently transitive only)
- [ ] Serena onboarding: call `check_onboarding_performed` / `initial_instructions` before any Wave 1 symbol operation (D-11)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `evaluate::<f32>()` runs on a real wgpu adapter that lacks `SHADER_F64` | PREC-06 | CI runners / CPU backend cannot prove the GPU-capability bypass; needs hardware lacking native f64 | On an adapter without `SHADER_F64`, run a wgpu-backed `evaluate::<f32>()` and confirm it succeeds where `evaluate::<f64>()` fails closed |
| Serena symbol-aware refactor was used (not blind text replace) | PREC-07 | Process constraint, not a runtime assertion | Reviewer confirms refactor commits used serena `rename_symbol`/`replace_symbol_body`; deliberately-f64 ABI sites (env/atm/bas, C ABI) unchanged |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
