---
phase: 18
slug: sessionrequest-arity-ge3-dispatch
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-12
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `18-RESEARCH.md` § "Validation Architecture (Nyquist)".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (built-in) — workspace-wide pattern |
| **Config file** | None (per-crate `Cargo.toml`); features via `--features cpu` (and `--features cpu,with-4c1e` for the 4c1e arity-4 tests) |
| **Quick run command** | `cargo test -p cintx-rs --locked` |
| **Full suite command** | `CINTX_BACKEND=cpu cargo test -p cintx-rs --features cpu --locked && CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity --test safe_api_arity4_parity && CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu,with-4c1e --locked --test safe_api_arity4_parity` |
| **Estimated runtime** | Quick: ~10 s; Full suite: 30–90 s (cpu profile only; matrix adds rocm + four feature profiles) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cintx-rs --locked`
- **After every plan wave:** Run the full suite command above
- **Before `/gsd-verify-work`:** Full suite must be green across all four manifest profiles (`base / with-f12 / with-4c1e / with-f12+with-4c1e`) on a `has_vendor_libcint` host
- **Max feedback latency:** ~10 s per task commit

---

## Per-Task Verification Map

> Populated by the planner / `/gsd:validate-phase` after `*-PLAN.md` files land.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| _TBD_   | _TBD_ | _TBD_ | ARITY-01..05 | — | N/A | unit / integration | _TBD_ | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — per-symbol parity tests (set depends on R1 resolution: 6 if drop, 8 if manifest rows added)
- [ ] `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` — 4 per-symbol parity tests (`int4c1e_*` gated `#[cfg(feature = "with-4c1e")]`)
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — add missing wrappers `vendor_int3c1e_p2_sph` and `vendor_int3c2e_ip1_sph` (each ~22 lines)
- [ ] `crates/cintx-oracle/tests/common/` or `safe_api_helpers.rs` — shared `collect_safe_api_matrix` helper (optional but recommended)
- [ ] `crates/cintx-core/src/operator.rs` — `AoSymmetry` enum + `Display` impl
- [ ] `crates/cintx-core/src/lib.rs` — re-export `AoSymmetry`
- [ ] `crates/cintx-runtime/src/options.rs` — add `aosym: Option<AoSymmetry>` field to `ExecutionOptions`
- [ ] `crates/cintx-rs/src/api.rs` — aosym preflight in `query_workspace` + F-order rustdoc on `IntegralTensor` + `aosym_error_path` unit test
- [ ] `crates/cintx-rs/src/error.rs` — `FacadeError::UnsupportedAoSymmetry { requested: String }` + `FacadeErrorKind` variant + `kind()` match arm
- [ ] `crates/cintx-rs/src/prelude.rs` — re-export `AoSymmetry`
- [ ] `crates/cintx-ops/src/generated/api_manifest.csv` + generator inputs — add plain `int3c2e_cart` / `int3c2e_sph` operator-kind rows (R1 resolution chosen by user)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| _none_   | —           | All Phase 18 behaviors are automated via `cargo test` + the `oracle_parity_gate` CI matrix | — |

*All phase behaviors have automated verification.*

---

## Eight Nyquist Dimensions

1. **Inputs covered** — All 10 (or 12 after R1 manifest expansion) target `OperatorId` values × cart/sph representations; full Cartesian shell-tuple sweep on H2O/STO-3G (5 shells → 125 arity-3 triples × 6+ ops + 625 arity-4 quartets × 4 ops ≈ 3,250 evaluations per cpu profile run).
2. **Output classes** — byte-identity (`==`) vs vendored libcint 6.1.3 at `atol=1e-12, rtol=0.0`; any-element-nonzero sentinel guards against zero-fill regressions; `FacadeError::UnsupportedAoSymmetry` typed error path for non-`S1`; preserved `FacadeError::*` for invalid operator / out-of-envelope source / profile / F12 / 4c1e.
3. **State transitions** — `SessionRequest::new` → `query_workspace` (aosym preflight + workspace) → `evaluate` (real `CubeClExecutor` dispatch). aosym failure short-circuits at `query_workspace`.
4. **Error paths** — non-`S1` aosym → `UnsupportedAoSymmetry`; invalid operator → existing `UnsupportedApi`; ShellTuple > 4 → `ShellTupleArityError`; memory limit → existing `Memory`.
5. **Concurrency** — Tests run serially (`--test-threads=1` per Phase 17 verification; same pattern for the new files).
6. **External dependencies** — vendored libcint 6.1.3 build (`CINTX_ORACLE_BUILD_VENDOR=1` + `has_vendor_libcint` cfg); CubeCL cpu/rocm backends (`CINTX_BACKEND` env).
7. **Performance envelopes** — Per-test budget < 60 s on cpu backend (target < 5 s); gate-wide budget unchanged. Fallback to deterministic subset if empirical CI cost exceeds budget during planning.
8. **Coverage tooling** — None new. Per-symbol failure messages in CI for direct bisection. `cargo public-api` optional diff of `cintx-rs::api` / `cintx-rs::prelude` to confirm additive SemVer.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15 s per task commit
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
