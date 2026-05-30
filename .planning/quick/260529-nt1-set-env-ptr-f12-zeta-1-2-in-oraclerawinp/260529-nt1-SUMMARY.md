---
phase: quick-260529-nt1
plan: 01
subsystem: oracle-fixtures
tags: [oracle, f12, env-layout, vendor-gate, libcint-parity]
requires: [260529-ne7 (conformant PTR_ENV_START=20 sample() env layout)]
provides: "OracleRawInputs::sample() supplies a legitimate F12 zeta on env[PTR_F12_ZETA]=env[9]; all 4 vendor gate profiles pass clean"
affects: [crates/cintx-oracle/src/fixtures.rs]
key-files:
  modified:
    - crates/cintx-oracle/src/fixtures.rs
decisions:
  - "env[PTR_F12_ZETA]=1.2 set on its designated libcint reserved slot in sample(); mirrors build_h2o_sto3g_f12; env[PTR_RANGE_OMEGA]=env[8] stays 0.0"
metrics:
  duration: ~6 min
  completed: 2026-05-29
---

# Phase quick-260529-nt1: Set env[PTR_F12_ZETA]=1.2 in OracleRawInputs::sample() Summary

Injected the legitimate F12/STG/YP correlation-factor exponent (`env[PTR_F12_ZETA]=env[9]=1.2`) into `OracleRawInputs::sample()` so the shared-inputs.env profile parity can evaluate the F12-family fixtures (`int2e_stg_*`/`int2e_yp_*`) instead of fail-closing on the runtime's `PTR_F12_ZETA must be non-zero` validator — all four vendor oracle gate profiles now pass clean (mismatch_count=0).

## What Changed

- **`OracleRawInputs::sample()`** (fixtures.rs ~514): one new line `env[PTR_F12_ZETA] = 1.2;` (using the named, already-imported `PTR_F12_ZETA` constant == 9), placed right after `env[NGRIDS] = 1.0;`. Nothing else in `sample()` changed: `env[PTR_RANGE_OMEGA]=env[8]` stays 0.0 (the ne7 fix), and the physical basis, shls2/3/4, grid slots, exp/coeff pointers, and atm/bas are byte-for-byte as ne7 left them.
- **Guard test `sample_env_reserves_libcint_global_slots`** (fixtures.rs ~1122): added an `else if slot == PTR_F12_ZETA { assert_eq!(inputs.env[slot], 1.2, ...) }` branch to the reserved-region loop, exempting env[9] from the "must be 0.0" rule and asserting it equals 1.2. The standalone `assert_eq!(inputs.env[PTR_RANGE_OMEGA], 0.0, ...)` assertion was KEPT verbatim (not weakened); NGRIDS/PTR_GRIDS branches and all shls/angular-momenta assertions unchanged.

## Why

After quick task 260529-ne7 made `sample()`'s env conformant (PTR_ENV_START=20, reserved slots zeroed), `env[PTR_F12_ZETA]` (env[9]) became 0.0. The shared-inputs.env profile parity evaluates ALL fixtures — including the 10 `int2e_stg_*`/`int2e_yp_*` sph F12-family symbols — against `OracleRawInputs::sample().env`. The cintx runtime validator (`validate_f12_env_params`) correctly fail-closes F12-family plans when `env[PTR_F12_ZETA]` is `None`/`0.0`, tripping every F12 symbol. env[9] is the DESIGNATED libcint reserved slot for the F12/STG correlation-factor exponent; setting it to 1.2 (the documented typical zeta, mirroring `build_h2o_sto3g_f12(zeta)`) is its intended use. Both cintx and vendored libcint read the SAME `inputs.env`, so the comparison stays apples-to-apples at zeta=1.2. Non-F12 integrals ignore env[9], so base/with-4c1e and all 1e/2e-Coulomb results are unchanged.

## Verification

### Fast cpu suite (no vendor)

`cargo test -p cintx-oracle --features cpu` — all green, no regressions. The guard test `fixtures::tests::sample_env_reserves_libcint_global_slots` passes (lib unittests: 15 passed; 0 failed). All integration test binaries pass with 0 failures.

### Final mandatory vendor gate (run VERBATIM, exit code 0)

```
CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false
```

GATE_EXIT=0, overall summary status: `ok`. Verbatim per-profile result (from `/tmp/cintx_artifacts/cintx_phase_04_oracle_compare_summary.json`):

| Profile | Status | mismatch_count | fixture_count |
|---------|--------|----------------|---------------|
| base | pass | 0 | 39 |
| with-f12 | pass | 0 | 49 |
| with-4c1e | pass | 0 | 41 |
| with-f12+with-4c1e | pass | 0 | 51 |

- The `PTR_F12_ZETA` / `InvalidEnvParam` fail-closed error is GONE for the `with-f12` and `with-f12+with-4c1e` profiles (grep for `PTR_F12_ZETA`/`InvalidEnvParam`/`must be non-zero` in the gate output returns nothing).
- ALL 4 PROFILES PASS CLEAN (mismatch_count=0 each). base + with-4c1e stayed clean as required.
- The F12/STG/YP `int2e_stg_*`/`int2e_yp_*` symbols now EVALUATE and MATCH vendor libcint 6.1.3 numerically at the unified atol — there is NO residual F12-kernel parity issue to defer. fixture_count rose from 39 (base) to 49 (with-f12) and 51 (with-f12+with-4c1e), confirming the F12 family is now exercised rather than skipped/fail-closed.

## Deviations from Plan

None - plan executed exactly as written. The TDD task collapsed to a single atomic guard-test+source edit (the guard test is the behavior check); both the source change and the test update landed together and the exact-named test run passed first try.

## Commits

- `aba9af5`: fix(260529-nt1): set env[PTR_F12_ZETA]=1.2 in OracleRawInputs::sample() for with-f12 gate

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/src/fixtures.rs (env[PTR_F12_ZETA] = 1.2 present; guard test branch present)
- FOUND: commit aba9af5
- Vendor gate exit 0; 4/4 profiles pass clean.
