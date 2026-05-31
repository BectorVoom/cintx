---
phase: 18
slug: sessionrequest-arity-ge3-dispatch
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-12
audited: 2026-05-31
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

> Populated by `/gsd:validate-phase` audit on 2026-05-31 from the four `18-0N-SUMMARY.md` artifacts and `18-VERIFICATION.md`.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| aosym-enum + preflight + typed error | 18-01, 18-02 | W0 | ARITY-04 | — | N/A | unit | `cargo test -p cintx-rs --locked aosym` | ✅ | ✅ green (2/2 on this host) |
| manifest expansion (`int3c2e_{cart,sph}`) + OperatorId shift | 18-01 | W0 | ARITY-01 | — | N/A | build + manifest-audit | `cargo build --workspace --locked` | ✅ | ✅ green |
| arity-3 dispatch + parity (`int3c1e`, `int3c1e_p2`, `int3c2e_ip1`, `int3c2e`) ×{cart,sph} | 18-03 | W0 | ARITY-01, ARITY-02, ARITY-03 | — | N/A | integration (vendor parity) | `CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity -- --test-threads=1` | ✅ | ⚠️ 6/8 green; `int3c1e_p2_{cart,sph}` ❌ red (deferred — see Manual-Only) |
| arity-4 dispatch + parity (`int2e_{cart,sph}`, `int4c1e_{cart,sph}`) | 18-04 | W0 | ARITY-01, ARITY-02, ARITY-03, ARITY-05 | — | N/A | integration (vendor parity) | `CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu,with-4c1e --locked --test safe_api_arity4_parity -- --test-threads=1` | ✅ | ✅ 4/4 green on vendor host (`int4c1e_*` buffer-size fixed in `5bd5ab0`) |
| oracle gate pickup (module gate matches `safe_api_arity2_parity.rs`) | 18-03, 18-04 | W0 | ARITY-05 | — | N/A | CI matrix (`oracle_parity_gate`) | existing matrix auto-discovers new `--test` files (CONTEXT.md D-15) | ✅ | ✅ green (no new CI job) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky/partial*

**On-host vs vendor-host:** This dev host has `has_vendor_libcint` OFF, so all 12 parity tests cfg-strip to 0 — they are exercised only on CI cells / vendor-built hosts with `CINTX_ORACLE_BUILD_VENDOR=1`. Vendor-host results above are from the `18-HUMAN-UAT.md` runtime UAT (2026-05-12). The aosym unit tests run and pass on every host.

---

## Wave 0 Requirements

- [x] `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — 8 per-symbol parity tests (R1 resolved: manifest rows added → 8 tests)
- [x] `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` — 4 per-symbol parity tests (`int4c1e_*` gated `#[cfg(feature = "with-4c1e")]`)
- [x] `crates/cintx-oracle/src/vendor_ffi.rs` — wrappers added: `vendor_int3c1e_p2_sph` + `vendor_int3c2e_cart` (`vendor_int3c2e_ip1_sph` intentionally absent — reuses `vendor_int3c2e_sph` per kernel-misnomer disposition)
- [x] `crates/cintx-core/src/operator.rs` — `AoSymmetry` enum + `Display` impl (lines 29-48)
- [x] `crates/cintx-core/src/lib.rs` — re-export `AoSymmetry`
- [x] `crates/cintx-runtime/src/options.rs` — `aosym: Option<AoSymmetry>` field on `ExecutionOptions` (line 121)
- [x] `crates/cintx-rs/src/api.rs` — aosym preflight in `query_workspace` + F-order rustdoc on `IntegralTensor` + 2 aosym unit tests (lines 919, 954)
- [x] `crates/cintx-rs/src/error.rs` — `FacadeError::UnsupportedAoSymmetry { requested: String }` + `FacadeErrorKind` variant (line 12) + `kind()` arm (line 46)
- [x] `crates/cintx-rs/src/prelude.rs` — re-export `AoSymmetry`
- [x] `crates/cintx-ops/generated/compiled_manifest.lock.json` + regenerated `api_manifest.rs` — plain `int3c2e_cart` / `int3c2e_sph` operator-kind rows added (R1: manifest rows)

*All Wave 0 artifacts delivered and verified in `18-VERIFICATION.md` (5/5 must-haves).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Byte-identity parity vs libcint 6.1.3 (all 12 tests) | ARITY-02, ARITY-03 | Requires `CINTX_ORACLE_BUILD_VENDOR=1` + vendored libcint sources; `has_vendor_libcint` cfg is OFF on dev hosts → tests cfg-strip to 0 locally. Runs automatically on CI `oracle_parity_gate` cells (the established Phase 15+ pattern). | `CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity -- --test-threads=1` and the `--features cpu,with-4c1e --test safe_api_arity4_parity` variant. Expect 6/8 + 4/4 green (see deferred row). |
| `int3c1e_p2_{cart,sph}` numeric correctness | ARITY-02 (subset) | **DEFERRED — not a coverage gap.** The automated parity tests EXIST and correctly detect a real ~1e-2..1e-4 divergence over 182 elements. Root cause is a pre-existing kernel defect (predates Phase 18, a dispatch-routing-only phase). Tracked in `18-HUMAN-UAT.md` Gap 2. | Follow-up: `/gsd:debug "int3c1e_p2_{cart,sph} kernel disagrees with vendored libcint by 1e-2 to 1e-4 in 182/N elements"`. Once the kernel is fixed, both tests turn green with no test changes. |

*All phase behaviors have automated verification. The two deferred items above are runtime-environment (vendor build) and a separately-tracked kernel defect — neither is a missing automated test.*

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

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references — zero MISSING gaps; all 5 requirements have automated tests
- [x] No watch-mode flags
- [x] Feedback latency < 15 s per task commit (aosym unit tests ~0 s; cfg-strip ~0 s)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-05-31

---

## Validation Audit 2026-05-31

| Metric | Count |
|--------|-------|
| Gaps found (MISSING) | 0 |
| Resolved (test generated) | 0 |
| Escalated (manual-only) | 2 (vendor-build runtime; deferred `int3c1e_p2` kernel defect) |

**Audit method:** State A (existing VALIDATION.md). Reconciled the stale `_TBD_` Per-Task Map against the four `18-0N-SUMMARY.md` artifacts, `18-VERIFICATION.md` (5/5 must-haves), and `18-HUMAN-UAT.md`. Confirmed on-host: 12 named parity tests present (8 arity-3 + 4 arity-4), 2 aosym unit tests pass, parity tests cfg-strip cleanly without vendor. No `gsd-nyquist-auditor` spawn needed — every requirement already has an automated test; no MISSING gaps to fill. The single PARTIAL (`int3c1e_p2_{cart,sph}`) is a coverage-complete test detecting a separately-tracked, out-of-scope kernel defect, not a validation gap.
