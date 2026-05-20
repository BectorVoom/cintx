---
phase: 20
slug: precision-generic-f64-f32-switch
status: planned
nyquist_compliant: true
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
| 20-01-T0 | 20-01 | 1 | PREC-07 | T-20-03 | serena symbol-aware refactor only (no blind replace) | process gate | serena `check_onboarding_performed` | N/A | ⬜ pending |
| 20-01-T1 | 20-01 | 1 | PREC-01 | T-20-02 | CintFloat sealed to f64\|f32 | unit | `cargo test -p cintx-core precision` | ❌ W1 | ⬜ pending |
| 20-01-T2 | 20-01 | 1 | PREC-01 | T-20-01 | A5 bytemuck cast proven sound | unit | `cargo test -p cintx-cubecl --test bytemuck_staging_cast_spike --features cpu` | ❌ W1 | ⬜ pending |
| 20-02-T1 | 20-02 | 2 | PREC-01 | T-20-04 | f64 byte-identity; f32 host finite | unit | `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu boys` | ❌ W2 | ⬜ pending |
| 20-02-T2 | 20-02 | 2 | PREC-01 | T-20-06 | f64 byte-identity; algorithm guards preserved | unit | `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu obara_saika stg pdata` | ❌ W2 | ⬜ pending |
| 20-03-T1 | 20-03 | 2 | PREC-01 | T-20-07 | rys f64 byte-identity; f32 weight-sum identity | unit | `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu rys` | ❌ W2 | ⬜ pending |
| 20-03-T2 | 20-03 | 2 | PREC-01 | T-20-09 | c2s f64 byte-identity | unit | `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu c2s` | ❌ W2 | ⬜ pending |
| 20-04-T1 | 20-04 | 3 | PREC-01,PREC-02 | T-20-10,T-20-11 | 1e/2e f64 byte-identity through precision dispatch | integration | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu one_electron two_electron` | ✅ existing | ⬜ pending |
| 20-04-T2 | 20-04 | 3 | PREC-01 | T-20-12 | 2c2e + spinor f64 byte-identity | integration | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu center_2c2e` | ✅ existing | ⬜ pending |
| 20-05-T1 | 20-05 | 4 | PREC-01 | T-20-13 | 3c1e/3c2e f64 byte-identity | integration | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu center_3c1e center_3c2e` | ✅ existing | ⬜ pending |
| 20-05-T2 | 20-05 | 4 | PREC-01,PREC-02 | T-20-14,T-20-15 | 4c1e/f12 f64 byte-identity; f12_zeta stays f64 | integration | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu,with-f12 f12` | ✅ existing | ⬜ pending |
| 20-06-T1 | 20-06 | 5 | PREC-06 | T-20-16 | f32 bypasses SHADER_F64; f64 retains it | unit | `cargo test -p cintx-cubecl --features cpu executor shader_f64` | ❌ W5 | ⬜ pending |
| 20-06-T2 | 20-06 | 5 | PREC-03 | T-20-17,T-20-18 | env/atm/bas stay &[f64]; staging over-alloc sound | regression | `CINTX_BACKEND=cpu cargo test -p cintx-compat --features cpu` | ✅ existing | ⬜ pending |
| 20-07-T1 | 20-07 | 6 | PREC-02 | T-20-19 | output structs generic, f64 default unchanged | compile/unit | `cargo test -p cintx-rs --features cpu evaluate_returns_deterministic` | ✅ existing | ⬜ pending |
| 20-07-T2 | 20-07 | 6 | PREC-02,PREC-04 | T-20-20,T-20-21 | evaluate() byte-identical f64; evaluate::<f32> returns Vec<f32> | integration | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu safe_api_arity2 safe_api_arity3 safe_api_arity4` | ✅ existing | ⬜ pending |
| 20-08-T1 | 20-08 | 7 | PREC-04 | T-20-23 | f64 tolerance model FROZEN | unit | `cargo test -p cintx-oracle --features cpu --lib f32_tolerance tolerance_for_family` | ❌ W7 | ⬜ pending |
| 20-08-T2 | 20-08 | 7 | PREC-05 | T-20-22,T-20-24 | f32 verified vs f64 libcint ref at empirical floors | integration | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --test f32_parity` | ❌ W7 | ⬜ pending |

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

**Approval:** planner-approved 2026-05-20 (all tasks have automated verify or Wave-0/Wave-1 scaffolding dependency; no 3 consecutive tasks without automated verify; latency <60s via cargo check)
