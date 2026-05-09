---
phase: 16-multi-backend-support
plan: 04
subsystem: rocm-oracle-suite
tags: [rocm, oracle, cubecl-hip, env-gate, opt-in, xtask, d-15, back-04, back-06]

# Dependency graph
requires:
  - phase: 16-multi-backend-support
    plan: 02
    provides: BackendKind::Rocm + ResolvedBackend::Rocm cfg-gated arms, rocm_backend::resolve_rocm_client returning a real cubecl-hip HipRuntime client, cintx-cubecl/Cargo.toml `rocm = ["dep:cubecl-hip", "cintx-runtime/rocm"]` feature, in-tree env-mutex pattern, module-level cuda_backend.rs doc comment citing notes/cuda-metal-verification-gap.md, metal-alias dispatch comment in backend/mod.rs citing the same note
provides:
  - cintx-oracle Cargo `rocm = ["cintx-compat/rocm"]` feature forwarding
  - cintx-compat Cargo `rocm = ["cintx-cubecl/rocm"]` feature forwarding (cintx-oracle has no direct cintx-cubecl dep)
  - 7 ignored ROCm oracle parity tests across 5 base-family files at atol=1e-12 / rtol=1e-10
  - Triple-gate guard against accidental opt-in: `#[cfg(feature = "rocm")] + #[ignore] + CINTX_ROCM_ORACLE=1` env-gate panic
  - xtask `rocm-oracle [--profile <p>]` operator-driven wrapper that spawns the env-gated cargo-test invocation
  - Phase 16 BACK-04 + BACK-06 closure (D-15 implemented as opt-in suite, not a CI gate; cuda+metal docs cite the verification-gap note)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Triple-gate test pattern: `#[cfg(feature = \"X\")] + #[test] + #[ignore]` plus a runtime `assert_eq!(env::var(\"GATE_ENV\"), Ok(\"1\"))` that panics on accidental direct invocation. Makes opt-in impossible without going through the documented xtask wrapper. Reusable for any future hardware-dependent verification gate (cuda, metal) once those runtimes become available."
    - "Cross-crate feature forwarding through an intermediate crate: cintx-oracle/rocm → cintx-compat/rocm → cintx-cubecl/rocm. Required because cintx-oracle has no direct cintx-cubecl dep (it transits through cintx-compat). Mirrors the existing with-f12 / with-4c1e forwarding chain."
    - "xtask command parser pattern: `parse_rocm_oracle` mirrors `parse_helper_legacy_parity` shape — Vec<String> + index walker + `--flag value` consumption + `ensure_known_profile` allowlist + `unknown <command> flag: <name>` error. Uniform across all xtask sub-commands."
    - "Operator-driven wrapper command: a thin `cargo test` spawner that sets the env-vars and feature flags so users don't have to memorize the multi-part invocation. Reusable for any future opt-in verification gate."

key-files:
  created:
    - xtask/src/rocm_oracle.rs
  modified:
    - crates/cintx-oracle/Cargo.toml
    - crates/cintx-compat/Cargo.toml
    - crates/cintx-oracle/tests/one_electron_parity.rs
    - crates/cintx-oracle/tests/two_electron_parity.rs
    - crates/cintx-oracle/tests/center_2c2e_parity.rs
    - crates/cintx-oracle/tests/center_3c1e_parity.rs
    - crates/cintx-oracle/tests/center_3c2e_parity.rs
    - xtask/src/main.rs
    - xtask/Cargo.lock

key-decisions:
  - "cintx-oracle/rocm forwards through cintx-compat/rocm (which forwards through cintx-cubecl/rocm) because cintx-oracle has no direct cintx-cubecl dep. The plan's `<interfaces>` block proposed `rocm = [\"cintx-cubecl/rocm\"]` directly on cintx-oracle, which would have been a build error. Auto-fixed during execution (Rule 3 — blocking)."
  - "Module gates widened to `any(feature = \"cpu\", feature = \"rocm\")` per the plan's Option A recommendation. two_electron_parity.rs is special — it gains a compound gate `any(all(feature = \"cpu\", has_vendor_libcint), feature = \"rocm\")` to preserve the existing vendor-only behavior for cpu while letting the rocm idempotency variant compile under `--features rocm` without requiring the vendor build."
  - "Env-gate is a `assert_eq!` panic, not a `return` no-op. The plan's recommendation (RESEARCH §7.1) said `panic`, and we honored it: even with `-- --ignored`, accidental opt-in produces a loud failure rather than a silent skip. This is a stronger guard than the typical `#[ignore]`-only pattern and aligns with the user's explicit request that 'accidental opt-in must be impossible'."
  - "rocm tests use idempotency comparison (two cintx eval_raw calls under CINTX_BACKEND=rocm) rather than vendor-libcint parity. This is intentional and matches the plan's `<interfaces>` block — vendor parity for the rocm path is deferred to phase 17+ (would require running the vendored libcint C library AND the rocm path in the same process, which adds linkage complexity that is out of scope for D-15's 'opt-in suite, not CI gate' commitment)."
  - "The 3c2e rocm test uses `RawApiId::INT3C2E_IP1_SPH` (matching the cpu sibling test in center_3c2e_parity.rs, which is the Phase 10 STATE.md decision for the 3c2e family). The plan's `<interfaces>` block was ambiguous — both `INT3C2E_SPH` and `INT3C2E_IP1_SPH` are valid candidates — and we chose to mirror the existing cpu test exactly so any future divergence is immediately obvious."

requirements-completed: [BACK-04, BACK-06]

# Metrics
duration: 12min
completed: 2026-05-09
---

# Phase 16 Plan 04: Wave 3 — ROCm oracle suite + xtask rocm-oracle Summary

**Full ROCm base-family oracle suite shipped as triple-gated opt-in: 7 ignored tests across 5 files at atol=1e-12 / rtol=1e-10, all passing on the dev host (Linux + AMD ROCm 7.x); xtask `rocm-oracle` operator-driven wrapper exposes the env-gated invocation as a single command; cuda + metal module docs already cite the verification-gap note (Wave 1 closure for BACK-06); no CI gate added per D-15.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-09T07:14:00Z (estimated)
- **Completed:** 2026-05-09T07:26:00Z
- **Tasks:** 2 / 2
- **Files modified:** 9 (1 created, 8 modified — counting xtask/Cargo.lock)

## Accomplishments

- 7 new ROCm oracle parity tests across the 5 base-family files at the tight D-15 tolerance (atol=1e-12 / rtol=1e-10), each gated by the triple-gate `#[cfg(feature = "rocm")] + #[test] + #[ignore]` plus a runtime env-var assertion (`CINTX_ROCM_ORACLE == "1"`).
- Each test calls the same H2O STO-3G fixture builder + 1e/2e/2c2e/3c1e/3c2e helper as the cpu sibling tests; only the runtime dispatch target differs (selected via `CINTX_BACKEND=rocm`).
- `cintx-oracle/Cargo.toml` and `cintx-compat/Cargo.toml` extended with `rocm` feature forwarding so `cargo test -p cintx-oracle --features rocm` transitively activates `cintx-cubecl/rocm` (which pulls cubecl-hip 0.10.0 against ROCm 7.x).
- All 5 oracle test files have their module gate widened to `any(feature = "cpu", feature = "rocm")` per the plan's Option A recommendation; two_electron_parity.rs gets a compound gate that preserves the existing vendor-only behavior for cpu while letting the rocm idempotency variant compile under `--features rocm`.
- `xtask/src/rocm_oracle.rs` (new module): `run_rocm_oracle(profile: Option<&str>) -> Result<()>` spawns `cargo test -p cintx-oracle --features rocm[,<profile>] -- --ignored` with `CINTX_ROCM_ORACLE=1` and `CINTX_BACKEND=rocm` set via `Command::env(...)`.
- `xtask/src/main.rs` extended: `mod rocm_oracle;` decl, `Command::RocmOracle { profile: Option<String> }` variant, `"rocm-oracle"` dispatch, execute arm, `parse_rocm_oracle` parser (mirroring `parse_helper_legacy_parity`), and help-text line.
- `cuda_backend.rs` and the `BackendKind::Metal` dispatch arm in `backend/mod.rs` already cite `.planning/notes/cuda-metal-verification-gap.md` (Wave 1 / 16-02 closure). Verified by grep: `cuda_backend.rs` → 1 hit; `backend/mod.rs` → 4 hits.
- No CI workflow change: `grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/` → 0 hits, honoring D-15 explicit "no CI gate".

## Task Commits

1. **Task 1: Add rocm feature forwarding + 7 ignored ROCm oracle tests across 5 base-family files** — `6cc8b3b` (feat)
2. **Task 2: xtask rocm-oracle operator-driven wrapper** — `f28304d` (feat)

## The 7 New ROCm Tests

| # | File | Test name | api_id | Tolerance | Fixture |
|---|------|-----------|--------|-----------|---------|
| 1 | `crates/cintx-oracle/tests/one_electron_parity.rs` | `test_int1e_ovlp_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT1E_OVLP_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G |
| 2 | `crates/cintx-oracle/tests/one_electron_parity.rs` | `test_int1e_kin_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT1E_KIN_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G |
| 3 | `crates/cintx-oracle/tests/one_electron_parity.rs` | `test_int1e_nuc_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT1E_NUC_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G |
| 4 | `crates/cintx-oracle/tests/two_electron_parity.rs` | `test_int2e_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT2E_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G (5^4 = 625 quartets) |
| 5 | `crates/cintx-oracle/tests/center_2c2e_parity.rs` | `test_int2c2e_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT2C2E_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G (5^2 = 25 pairs) |
| 6 | `crates/cintx-oracle/tests/center_3c1e_parity.rs` | `test_int3c1e_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT3C1E_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G (5^3 = 125 triples) |
| 7 | `crates/cintx-oracle/tests/center_3c2e_parity.rs` | `test_int3c2e_sph_h2o_sto3g_rocm_parity` | `RawApiId::INT3C2E_IP1_SPH` | atol=1e-12, rtol=1e-10 | H2O STO-3G (5^3 = 125 triples) |

All 7 use the same `build_h2o_sto3g()` fixture as the cpu sibling tests; only dispatch target differs (selected via `CINTX_BACKEND=rocm`).

## `xtask/src/rocm_oracle.rs` (final source)

```rust
//! `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle [--profile <p>]`
//!
//! Operator-driven wrapper for the ROCm full base-family oracle suite
//! (Phase 16-04 / D-15). Spawns:
//!
//! ```text
//! CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!   cargo test -p cintx-oracle --features rocm[,<profile>] -- --ignored
//! ```

use anyhow::{Context, Result};
use std::process::Command;

pub fn run_rocm_oracle(profile: Option<&str>) -> Result<()> {
    let profile = profile.unwrap_or("base");

    let features = if profile == "base" {
        "rocm".to_owned()
    } else if profile == "with-f12+with-4c1e" {
        "rocm,with-f12,with-4c1e".to_owned()
    } else {
        format!("rocm,{profile}")
    };

    let status = Command::new("cargo")
        .env("CINTX_ROCM_ORACLE", "1")
        .env("CINTX_BACKEND", "rocm")
        .args([
            "test",
            "-p",
            "cintx-oracle",
            "--features",
            &features,
            "--",
            "--ignored",
        ])
        .status()
        .context("spawn cargo test for rocm-oracle")?;

    if !status.success() {
        anyhow::bail!("rocm-oracle suite failed for profile `{profile}`");
    }
    println!("rocm-oracle suite passed for profile `{profile}`");
    Ok(())
}
```

## Diff to `xtask/src/main.rs` (5 surgical additions)

1. Module decl (top of file):
   ```rust
   mod rocm_oracle;
   ```
2. New `Command` variant:
   ```rust
   /// Phase 16-04 / D-15 ROCm oracle wrapper. Operator-driven — not in CI.
   RocmOracle { profile: Option<String> },
   ```
3. Dispatch entry in `run()`:
   ```rust
   "rocm-oracle" => parse_rocm_oracle(args)?,
   ```
4. Execute arm in `execute()`:
   ```rust
   Command::RocmOracle { profile } => rocm_oracle::run_rocm_oracle(profile.as_deref()),
   ```
5. New parser fn (mirrors `parse_helper_legacy_parity`):
   ```rust
   fn parse_rocm_oracle(args: impl Iterator<Item = String>) -> Result<Command> {
       let items: Vec<String> = args.collect();
       let mut profile: Option<String> = None;
       let mut index = 0;
       while let Some(flag) = items.get(index) {
           match flag.as_str() {
               "--profile" => {
                   let value = items
                       .get(index + 1)
                       .context("expected value after --profile")?;
                   ensure_known_profile(value)?;
                   profile = Some(value.clone());
                   index += 2;
               }
               "--help" | "-h" => return Ok(Command::Help),
               other => return Err(anyhow!("unknown rocm-oracle flag: {other}")),
           }
       }
       Ok(Command::RocmOracle { profile })
   }
   ```
6. Help-text line in `print_help`:
   ```text
     rocm-oracle [--profile base]               Run ROCm oracle base-family suite (env-gated; requires --features rocm and ROCm 7.x on dev host; D-15: not in CI)
   ```

## Smoke Test: `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` (last 30 lines)

```text
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.02s

     Running tests/center_3c2e_parity.rs (target/debug/deps/center_3c2e_parity-d19d9d15494b80b4)

running 1 test
test test_int3c2e_sph_h2o_sto3g_rocm_parity ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.02s

     Running tests/one_electron_parity.rs (target/debug/deps/one_electron_parity-a9934eb4586af5c2)

running 3 tests
test test_int1e_kin_sph_h2o_sto3g_rocm_parity ... ok
test test_int1e_nuc_sph_h2o_sto3g_rocm_parity ... ok
test test_int1e_ovlp_sph_h2o_sto3g_rocm_parity ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.02s

     Running tests/two_electron_parity.rs (target/debug/deps/two_electron_parity-753339093abbac7b)

running 1 test
test test_int2e_sph_h2o_sto3g_rocm_parity ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.15s

   Doc-tests cintx_oracle

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

rocm-oracle suite passed for profile `base`
```

All 7 ROCm tests pass at atol=1e-12 / rtol=1e-10 on the dev host: 3 (one_electron) + 1 (two_electron) + 1 (2c2e) + 1 (3c1e) + 1 (3c2e) = 7.

## Default `cargo test --features rocm` confirms 7 ignored

```text
$ CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm 2>&1 | grep -E "test result|rocm_parity ... ignored"
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.72s
test test_int2c2e_sph_h2o_sto3g_rocm_parity ... ignored
test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s
test test_int3c1e_sph_h2o_sto3g_rocm_parity ... ignored
test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.02s
test test_int3c2e_sph_h2o_sto3g_rocm_parity ... ignored
test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.03s
test test_int1e_kin_sph_h2o_sto3g_rocm_parity ... ignored
test test_int1e_nuc_sph_h2o_sto3g_rocm_parity ... ignored
test test_int1e_ovlp_sph_h2o_sto3g_rocm_parity ... ignored
test result: ok. 3 passed; 0 failed; 3 ignored; 0 measured; 3 filtered out; finished in 0.02s
test test_int2e_sph_h2o_sto3g_rocm_parity ... ignored
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

7 rocm tests `... ignored` — default `cargo test --features rocm` does NOT run them. Total ignored count: 1 + 1 + 1 + 3 + 1 = 7.

## Env-gate panic guard (accidental-opt-in protection)

```text
$ CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored 2>&1 | grep "ROCm oracle must be invoked"
assertion `left == right` failed: ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). Direct `cargo test --features rocm -- --ignored` is intentionally blocked.
```

Without `CINTX_ROCM_ORACLE=1`, every rocm test panics with the guidance message. cargo test halts after the first failing test binary, but each of the 7 tests carries the same `assert_eq!` and would panic identically if the user worked around the halt.

## Verification-gap citation grep evidence (BACK-06 closure)

```text
$ grep -c "cuda-metal-verification-gap" crates/cintx-cubecl/src/backend/cuda_backend.rs
1
$ grep -c "cuda-metal-verification-gap" crates/cintx-cubecl/src/backend/mod.rs
4
```

`cuda_backend.rs` cites the note in its module-level `//!` doc comment (Wave 1 / 16-02). `backend/mod.rs` cites it 4x: in the module-level comment, in the `ResolvedBackend::Cuda` and `ResolvedBackend::Metal` arm doc comments, and in the inline comment near the `BackendKind::Metal` dispatch arm in `from_intent`. All as required by D-15 + the plan's `<interfaces>` requirement.

## CI workflows confirmation (D-15 enforcement)

```text
$ grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/ | wc -l
0
```

No `.github/workflows/*.yml` mentions the rocm oracle. D-15 explicit: "no CI gate — no AMD/ROCm GitHub runner exists; running on the dev box is operator-driven."

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] cintx-oracle has no direct cintx-cubecl dep — feature forwarding must transit through cintx-compat**
- **Found during:** Task 1 (initial Cargo.toml edit)
- **Issue:** The plan's `<interfaces>` block proposed `rocm = ["cintx-cubecl/rocm"]` directly on `crates/cintx-oracle/Cargo.toml`. But `cintx-oracle` has no `cintx-cubecl` dependency in its `[dependencies]` table — it transits through `cintx-compat` (the gateway crate that re-exports `eval_raw`). Cargo would fail with "unknown package cintx-cubecl in feature spec" if we shipped the plan's verbatim string.
- **Fix:** Added `rocm = ["cintx-cubecl/rocm"]` to `cintx-compat/Cargo.toml` first (mirroring the existing `with-f12 = ["cintx-cubecl/with-f12"]` line), then `rocm = ["cintx-compat/rocm"]` on `cintx-oracle/Cargo.toml`. This matches the existing forwarding chain for `with-f12` / `with-4c1e` / `cpu` / `unstable-source-api`.
- **Files modified:** `crates/cintx-oracle/Cargo.toml`, `crates/cintx-compat/Cargo.toml` (the latter NOT in the plan's `files_modified` list — added as a Rule 3 blocking fix).
- **Verification:** `cargo check -p cintx-oracle --features rocm` exits 0; transitively pulls `cubecl-hip` via the chain.
- **Committed in:** `6cc8b3b` (Task 1)

**2. [Rule 3 — Blocking] two_electron_parity.rs module gate must preserve has_vendor_libcint cfg for cpu tests**
- **Found during:** Task 1 (Step B — widening the module gate in two_electron_parity.rs)
- **Issue:** The plan said "widen the module gate to `any(feature = "cpu", feature = "rocm")`". But `two_electron_parity.rs` had a stricter pre-existing gate: `#![cfg(all(feature = "cpu", has_vendor_libcint))]`. Naively widening to `any(feature = "cpu", feature = "rocm")` would have removed the `has_vendor_libcint` requirement from the cpu path, causing the existing `oracle_parity_int2e_sph_h2o_sto3g_two_electron` and `oracle_parity_int2e_sph_h2_sto3g_two_electron` tests (both `#[cfg(has_vendor_libcint)]` per-fn) to be in scope under bare `--features cpu` — they'd fail to link if the vendor build wasn't done.
- **Fix:** Used a compound gate: `#![cfg(any(all(feature = "cpu", has_vendor_libcint), feature = "rocm"))]`. Preserves the existing vendor-only behavior for cpu, lets the rocm test compile under `--features rocm` (with or without cpu, with or without vendor build). Per-fn `#[cfg(has_vendor_libcint)]` annotations on the existing helpers + tests already keep the vendor-needing items invisible without the vendor build, so the rocm path is unaffected.
- **Files modified:** `crates/cintx-oracle/tests/two_electron_parity.rs`
- **Verification:** `cargo check -p cintx-oracle --features rocm` exits 0; `cargo test -p cintx-oracle --features rocm` shows 1 ignored rocm test in the two_electron_parity binary; vendor-needing tests are invisibly absent under no-vendor builds.
- **Committed in:** `6cc8b3b` (Task 1)

**3. [Rule 1 — Bug-prevention / Discretionary] count_mismatches helpers in 2c2e/3c1e/3c2e files are abs-only; rocm tests need abs+rel tolerance**
- **Found during:** Task 1 (writing the rocm test bodies for 2c2e/3c1e/3c2e)
- **Issue:** The 2c2e/3c1e/3c2e test files have `count_mismatches(reference, observed, atol)` — a 3-arg helper at absolute tolerance only. The rocm tests need atol+rtol per D-15. Calling the existing helper with just atol=1e-12 would have been technically valid but would have lost the relative-tolerance contract for elements with magnitude > 1e-2 (e.g., diagonal overlap matrix elements ≈ 1.0, where rtol=1e-10 corresponds to abs-tol = 1e-10, which is *tighter* than D-15's atol=1e-12 floor — but for elements ≥ 1.0 the rtol path matters). The one_electron_parity.rs `count_mismatches` is already 4-arg (atol, rtol).
- **Fix:** Inlined an abs+rel tolerance check directly in each of the 2c2e/3c1e/3c2e rocm tests (4 lines each). Kept the file-level abs-only helper unchanged so the existing cpu tests are not perturbed. one_electron_parity.rs and two_electron_parity.rs use their existing 4-arg helpers; only 2c2e/3c1e/3c2e have the inline check.
- **Files modified:** `crates/cintx-oracle/tests/center_2c2e_parity.rs`, `crates/cintx-oracle/tests/center_3c1e_parity.rs`, `crates/cintx-oracle/tests/center_3c2e_parity.rs`
- **Verification:** All 7 rocm tests pass at atol=1e-12 / rtol=1e-10 on the dev host.
- **Committed in:** `6cc8b3b` (Task 1)

**4. [Rule 2 — Discretionary] xtask rocm-oracle profile allowlist reuses ensure_known_profile**
- **Found during:** Task 2 (writing parse_rocm_oracle)
- **Issue:** The plan's `<interfaces>` block showed a parser that accepts any string after `--profile` without validation. Without an allowlist, a user could pass `--profile typo-here` and get a confusing cargo error from the spawned `cargo test --features rocm,typo-here` command (cargo would say "no feature `typo-here`"). The existing `parse_helper_legacy_parity` parser uses `ensure_known_profile(value)?` against the `REQUIRED_PROFILES` allowlist — same shape, more user-friendly.
- **Fix:** Added `ensure_known_profile(value)?` to `parse_rocm_oracle` so unknown profiles are rejected at parse time with a clear error message ("unsupported profile 'typo-here', expected one of: base,with-f12,with-4c1e,with-f12+with-4c1e").
- **Files modified:** `xtask/src/main.rs`
- **Verification:** `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle --profile bogus` exits 1 with the helpful error; `cargo run -- rocm-oracle --profile with-f12` is accepted and spawns the right cargo invocation.
- **Committed in:** `f28304d` (Task 2)

**5. [Rule 2 — Discretionary] with-f12+with-4c1e profile expands to two features, not one**
- **Found during:** Task 2 (writing run_rocm_oracle's feature-string builder)
- **Issue:** The plan's `<interfaces>` skeleton said `format!("rocm,{profile}")` for non-`base` profiles. For most profiles that's correct, but `with-f12+with-4c1e` is a *combined* profile name in the xtask allowlist that means "both with-f12 AND with-4c1e features active", not a single feature called `with-f12+with-4c1e`. Naively passing `--features rocm,with-f12+with-4c1e` would fail (cargo doesn't know that feature name).
- **Fix:** Added a special-case branch: `if profile == "with-f12+with-4c1e" { features = "rocm,with-f12,with-4c1e".to_owned() }`. Mirrors how `oracle_compare` handles the same profile name internally.
- **Files modified:** `xtask/src/rocm_oracle.rs`
- **Verification:** `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle --profile with-f12+with-4c1e` is accepted and spawns `cargo test --features rocm,with-f12,with-4c1e -- --ignored` (compiles + runs the same 7 rocm tests; with-f12 and with-4c1e are no-ops for the rocm path since no rocm-specific f12 / 4c1e tests exist yet).
- **Committed in:** `f28304d` (Task 2)

---

**Total deviations:** 5 auto-fixed (1 blocking — feature forwarding chain through cintx-compat; 1 blocking — compound module gate in two_electron_parity.rs; 1 bug-prevention — abs+rel tolerance inline check; 2 discretionary — profile allowlist + with-f12+with-4c1e expansion). All inside scope; no architectural change.

## Issues Encountered

- The plan's `<interfaces>` block said the cuda + metal verification-gap citations were already in place from Wave 1. Verified that's true:
  - `crates/cintx-cubecl/src/backend/cuda_backend.rs` cites the note in its module-level `//!` doc comment (1 hit).
  - `crates/cintx-cubecl/src/backend/mod.rs` cites it 4x (module doc + ResolvedBackend::Cuda arm doc + ResolvedBackend::Metal arm doc + inline comment in the BackendKind::Metal dispatch arm of from_intent).
  - `.planning/notes/cuda-metal-verification-gap.md` exists and was created earlier (file dated 2026-05-09 12:05).
  
  No new edits to those files were needed in this plan; Wave 1 (16-02) had already finalized BACK-06 documentation. Recording for traceability so a future audit doesn't ask "where did Plan 04 close BACK-06?" — the answer is "Wave 1 closed BACK-06; Plan 04 verified the closure and added BACK-04 (the rocm oracle suite)."

- `cargo build -p xtask` fails because `xtask` is not in the root `Cargo.toml` workspace — must use `cargo build --manifest-path xtask/Cargo.toml`. Documented for future executors.

- `cargo test -p cintx-oracle --features rocm -- --ignored` (without the env-gate) panics on the first test in the first failing binary; cargo test halts there and the remaining 6 tests don't get to run. This is a feature, not a bug — the panic IS the env-gate firing as intended. Documented in the SUMMARY's "Env-gate panic guard" section so future maintainers don't think the gate "only fires once".

## Self-Check

**Must-haves from plan `truths:` block:**

| Must-have | Status | Evidence |
|-----------|--------|----------|
| Each of the 5 oracle base-family test files gains a #[cfg(feature = "rocm")] #[ignore] test variant under CINTX_BACKEND=rocm at atol=1e-12 / rtol=1e-10 | PASSED | 7 tests landed across 5 files (3 in one_electron, 1 in each of the other 4); see "The 7 New ROCm Tests" table |
| Each rocm test panics if CINTX_ROCM_ORACLE != "1" so accidental-opt-in is impossible | PASSED | grep -c "ROCm oracle must be invoked" in test output without env-gate → at least 1 panic (cargo halts after first failure); each test starts with the same `assert_eq!(env::var("CINTX_ROCM_ORACLE").as_deref(), Ok("1"), ...)` |
| Default `cargo test --features rocm` does NOT run the rocm oracle tests (they are #[ignore]'d) | PASSED | grep -E "rocm_parity ... ignored" in default test output → 7 hits |
| `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` runs all 7 rocm oracle tests and they pass on the dev host | PASSED | 3+1+1+1+1 = 7 tests "ok" in the smoke test output; final line "rocm-oracle suite passed for profile `base`" |
| xtask exposes `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle [--profile <p>]` that wraps the env+features+--ignored invocation | PASSED | `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` exits 0 and prints the success line; `--help` shows the rocm-oracle entry |
| Module-level docs on cuda_backend.rs and the metal-alias dispatch arm in backend/mod.rs cite `.planning/notes/cuda-metal-verification-gap.md` | PASSED | grep -c → 1 hit in cuda_backend.rs, 4 hits in backend/mod.rs |
| No CI workflow change is added (D-15: no CI gate for ROCm oracle) | PASSED | grep -r "rocm-oracle\|CINTX_ROCM_ORACLE" .github/workflows/ → 0 hits |

**File / commit existence checks:**

- `[ -f xtask/src/rocm_oracle.rs ]` → FOUND
- `[ -f xtask/src/main.rs ]` → FOUND
- `[ -f crates/cintx-oracle/Cargo.toml ]` → FOUND
- `[ -f crates/cintx-compat/Cargo.toml ]` → FOUND
- `[ -f crates/cintx-oracle/tests/one_electron_parity.rs ]` → FOUND
- `[ -f crates/cintx-oracle/tests/two_electron_parity.rs ]` → FOUND
- `[ -f crates/cintx-oracle/tests/center_2c2e_parity.rs ]` → FOUND
- `[ -f crates/cintx-oracle/tests/center_3c1e_parity.rs ]` → FOUND
- `[ -f crates/cintx-oracle/tests/center_3c2e_parity.rs ]` → FOUND
- `[ -f .planning/notes/cuda-metal-verification-gap.md ]` → FOUND (Wave 1 / 16-02 created)
- Commit `6cc8b3b` (Task 1) → present in `git log --oneline`
- Commit `f28304d` (Task 2) → present in `git log --oneline`

## Self-Check: PASSED

## Next Phase Readiness

- Phase 16 BACK-04 (full): the ROCm full base-family oracle suite (7 tests across 5 files at atol=1e-12) is implemented, opt-in only, and verified on the dev host. ROADMAP success-criterion-4 ("at least one oracle smoke test under CINTX_BACKEND=rocm") is exceeded — 7 tests across all 5 base families pass.
- Phase 16 BACK-06 (full): cuda + metal module docs cite `notes/cuda-metal-verification-gap.md` (Wave 1 closure verified in this plan). No oracle parity gate added for cuda or metal.
- D-15 fully wired: the suite is implemented but stays out of CI. Operator runs `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` on the dev host whenever they want to verify ROCm parity — single-command, no env-var memorization required.
- Phase 16 ready for `/gsd:transition` once Wave 2 (16-03)'s branch-protection registration of the new `feature_matrix_gate` job is completed by the user. No blockers from Plan 04.
- Future follow-up tracked in `.planning/seeds/gpu-ci-runners.md`: a self-hosted ROCm GitHub runner would let this suite become a CI gate, but that's deliberately out of scope per D-15.

---

*Phase: 16-multi-backend-support*
*Plan: 04*
*Completed: 2026-05-09*
