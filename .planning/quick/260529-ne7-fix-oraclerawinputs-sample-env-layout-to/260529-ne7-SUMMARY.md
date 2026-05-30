---
phase: quick-260529-ne7
plan: 01
subsystem: oracle-fixtures
tags: [oracle, libcint-parity, env-layout, 2e-integrals, fixtures]
requires:
  - "cintx_compat::raw::{PTR_ENV_START, NGRIDS, PTR_GRIDS, ATM/BAS slot constants}"
provides:
  - "OracleRawInputs::sample() with conformant PTR_ENV_START-aligned env layout (env[8]=PTR_RANGE_OMEGA == 0.0)"
  - "sample_env_reserves_libcint_global_slots guard test"
affects:
  - "All cintx-oracle parity fixtures that consume OracleRawInputs::sample()"
key-files:
  modified:
    - crates/cintx-oracle/src/fixtures.rs
decisions:
  - "Reserve env[0..PTR_ENV_START=20]=0 except the two legitimate grid slots env[NGRIDS]/env[PTR_GRIDS]; place coords/exps/coeffs/grid at >=20."
  - "PTR_RANGE_OMEGA (env[8]) is declared as a local test-only const (not exported by cintx-compat, carries no cintx semantics) to assert the specific collision being fixed."
metrics:
  tasks: 2
  files: 1
  completed: 2026-05-29
---

# Phase quick Plan 260529-ne7: Conformant OracleRawInputs::sample() env layout Summary

Rebuilt `OracleRawInputs::sample()` with a libcint-conformant `PTR_ENV_START`-aligned `env` layout so that shell coefficients no longer collide with libcint's reserved global-parameter region; this eliminates the `cint2e_sph`/`cint2e_cart` legacy-parity divergence (vendored libcint was computing range-separated Coulomb because shell-2's coeff 0.6 landed on `env[8]=PTR_RANGE_OMEGA`).

## What changed

- `crates/cintx-oracle/src/fixtures.rs`:
  - `OracleRawInputs::sample()` rewritten to mirror the conformant `build_h2o_sto3g()` pattern: start with `vec![0.0_f64; PTR_ENV_START]` (20 reserved zero slots), set `env[NGRIDS]=1.0`, then append the atom coordinate, the four per-shell `(exp, coeff)` pairs, and the grid coordinate at indices `>= PTR_ENV_START`, recording the grid-coord index into `env[PTR_GRIDS]`.
  - `atm` built with `ATM_SLOTS` length using `CHARGE_OF / PTR_COORD / NUC_MOD_OF=POINT_NUC / PTR_ZETA` constants (single point-charge atom Z=1 at origin).
  - `bas` built with `4 * BAS_SLOTS` using the named slot constants, with per-shell `ptr_exp`/`ptr_coeff` pointing at the appended scalars.
  - Doc comment rewritten to describe the conformant layout and the specific `PTR_RANGE_OMEGA` collision being fixed.
  - Added top-level imports `NGRIDS`, `PTR_GRIDS`.
  - Added guard test `sample_env_reserves_libcint_global_slots` asserting: `env[8]==0.0`; the whole reserved region `env[0..20]` is zero except `env[NGRIDS]==1.0` and `env[PTR_GRIDS]==grid-coord-index (>=20)`; `shls2/3/4` unchanged; 4 shells with angular momenta `[0,1,0,1]`.

**Physical basis preserved exactly:** single atom Z=1 at origin; shells `0:(l=0,exp=1.0,coeff=1.0)`, `1:(l=1,exp=0.9,coeff=0.8)`, `2:(l=0,exp=0.7,coeff=0.6)`, `3:(l=1,exp=0.5,coeff=0.4)`; one grid point at origin; `shls2=[0,1]`, `shls3=[0,1,2]`, `shls4=[0,1,2,3]`. Only the env layout + atm/bas pointers changed.

## Verification

- Guard test `sample_env_reserves_libcint_global_slots` passes.
- Full `cargo test -p cintx-oracle --features cpu` (lib + all integration tests): **all green, 0 failures** — no regressions from the shared-fixture change. No test asserted the OLD polluted env values, so no test rewrites were needed.

## Final vendor gate (run verbatim)

Command (verbatim):
```
CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false
```

Per-profile result (from `/tmp/cintx_artifacts/cintx_phase_04_oracle_compare_summary.json`):
```
base                 -> pass    mismatch_count=0
with-f12             -> failed  mismatch_count=10
with-4c1e            -> pass    mismatch_count=0
with-f12+with-4c1e   -> failed  mismatch_count=10
```

Gate exit line (verbatim):
```
xtask gate failed: oracle parity gate failed for 2 profile(s): with-f12: oracle parity failed with 10 mismatches | with-f12+with-4c1e: oracle parity failed with 10 mismatches
```

### cint2e divergence: GONE (confirmed)

The target blocker — the `cint2e_sph` / `cint2e_cart` constant-per-quartet ratio divergence (cintx=1.709e-3 vs vendor=3.22e-4, ratio ~5.308) — is **eliminated**. The `base` profile (which contains `int2e_sph`, `int2e_cart`, `int2e_ip1_*`, `int2e_spinor`, etc.) now reports **mismatch_count=0, status=pass**, as does `with-4c1e`. This was the last known helper/legacy-wrapper blocker for the base/4c1e profiles, and it is now clean.

### Newly-surfaced downstream blocker (NOT fixed — noted for follow-up)

`with-f12` and `with-f12+with-4c1e` now fail with 10 mismatches each. All 10 are `kind=raw_eval` rejections on the F12 family symbols:

```
int2e_stg_sph, int2e_stg_ip1_sph, int2e_stg_ipip1_sph, int2e_stg_ipvip1_sph, int2e_stg_ip1ip2_sph,
int2e_yp_sph,  int2e_yp_ip1_sph,  int2e_yp_ipip1_sph,  int2e_yp_ipvip1_sph,  int2e_yp_ip1ip2_sph
```

with detail (verbatim, representative):
```
raw eval for `int2e_stg_ip1_sph`: invalid env parameter PTR_F12_ZETA: env[9] (PTR_F12_ZETA) must be non-zero for F12/STG/YP integrals
```

**Root cause:** This is a *consequence of the layout fix*, not a regression in correctness. The F12 oracle fixtures run their parity comparison against the shared `sample()` `env` directly (`crates/cintx-oracle/src/compare.rs` lines ~1211-1223 use `&inputs.env` for all symbols; there is no per-family F12-zeta injection). The OLD polluted layout *accidentally* placed shell-1's coefficient `0.8` on `env[9]=PTR_F12_ZETA`, which happened to satisfy the cintx engine's (correct) fail-closed requirement that F12/STG/YP integrals have a non-zero F12 zeta. The new conformant layout zeros the reserved region, so `env[9]=0.0` and the fail-closed `InvalidEnvParam` validation now (correctly) trips.

This is therefore a **newly-surfaced downstream blocker** caused by the F12 fixture path relying on coefficient pollution for its zeta. Per the task constraints, it was **NOT fixed here** (do not weaken the engine's fail-closed F12 validation; do not silently re-pollute the reserved slot). The correct follow-up is to make the F12 oracle path inject a legitimate non-zero `env[PTR_F12_ZETA]` (e.g. via the existing `build_h2o_sto3g_f12(zeta)` builder pattern, or a sample()-variant with an explicit F12 zeta) and then re-verify vendor parity of the STG/YP values at that zeta. That is a distinct parity question (vendor would also see the zeta) and is out of scope for this env-layout fix.

No `/tmp/cintx_artifacts` matrix-json artifact-path issue surfaced — all profile artifacts wrote to `used_required_path=true` (`/tmp/cintx_artifacts/...`).

## Deviations from Plan

None — plan executed exactly as written. The plan anticipated that a new downstream blocker might surface after the base divergence cleared and instructed to report it verbatim and stop without fixing it; that is exactly what happened (F12 zeta fail-closed rejection).

## Commits

- `3b4ced5` fix(260529-ne7): conformant PTR_ENV_START env layout for OracleRawInputs::sample()

## Self-Check: PASSED

- `crates/cintx-oracle/src/fixtures.rs` exists and contains `sample_env_reserves_libcint_global_slots` + the rewritten `sample()` (PTR_ENV_START-aligned).
- Commit `3b4ced5` present in `git log`.
- `cargo test -p cintx-oracle --features cpu` all green; guard test passes.
- Vendor gate run verbatim; base + with-4c1e pass clean (cint2e divergence gone); with-f12 / with-f12+with-4c1e fail on a newly-surfaced F12-zeta fail-closed rejection, reported verbatim and NOT fixed.
