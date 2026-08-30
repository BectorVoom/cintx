# Precision & Accuracy Tolerance Table

## Unified Tolerance Policy

All integral families evaluated in `cintx` are governed by the unified tolerance policy:

| Parameter | Value | Definition |
|---|---|---|
| `UNIFIED_ATOL` | `1.0e-12` | Absolute tolerance for all standard double-precision float comparisons vs libcint 6.1.3 |
| `UNIFIED_RTOL` | `1.0e-12` | Relative tolerance for all standard double-precision float comparisons vs libcint 6.1.3 |
| f64 near-zero regime | none (`0.0`) | f64 uses the same mixed comparison at every finite magnitude; at exactly zero this is naturally absolute-only |
| f32 `ZERO_THRESHOLD` | `1.0e-18` | Frozen D-09 policy; not owned or changed by the f64 precision plan |

## Per-Family Tolerance Table

Every family evaluated across all supported execution profiles (`base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`, `unstable-source`) adheres to this table without exception:

| Family | Sub-families / Forms | `atol` | `rtol` | `zero_threshold` | Exception Status |
|---|---|---|---|---|---|
| `1e` | Cart, Sph, Spinor, GIAO, $\sigma$, Grids | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `2e` | Cart, Sph, Spinor, GIAO | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `2c2e` | Cart, Sph, Spinor, GIAO | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `3c1e` | Cart, Sph, Spinor, GIAO | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `3c2e` | Cart, Sph, Spinor, GIAO | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `4c1e` | Cart, Sph, Spinor | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `f12` | Cart, Sph, Spinor, 2e/3c2e/2c2e | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |
| `unstable::source::*` | 1e, 2e, 3c1e, 3c2e, Breit, SSC | `1.0e-12` | `1.0e-12` | none | **No exception required** (passes at 1e-12) |

### Exception Statement
**Zero exceptions above 1.0e-12 are required across the entire codebase.**
All families pass oracle comparison tests against vendored `libcint 6.1.3` within `atol = 1.0e-12` and `rtol = 1.0e-12`.
Historical tolerance relaxations (`1e-11`, `1e-9`, `1e-7`) from initial prototyping have been fully retired.

## Measurement and ratchet

The concrete measurement is recorded in `artifacts/cintx_precision_budget.json`.
Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo run --release --manifest-path
xtask/Cargo.toml --features cpu -- error-budget --check-headroom` in CI: it
fails if any recorded operation grows, even when it remains inside the raw
tolerance. Refreshing that reviewed baseline requires the explicit `--record`
flag; the command refuses to record its deliberate `--perturb-test` mode.
