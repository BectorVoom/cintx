---
phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
plan: 03
subsystem: kernels
tags: [three-center, overlap, vrr, hrr, cart-to-sph, oracle, libcint]

# Dependency graph
requires:
  - phase: 10-01
    provides: cart_to_sph_3c1e, vendor_int3c1e_sph FFI wrapper

provides:
  - Real 3c1e overlap kernel via fill_g_tensor_3c1e + contract_3c1e_ovlp
  - int3c1e_sph and int3c1e_cart manifest entries for eval_raw dispatch
  - Oracle parity test: mismatch_count=0 vs libcint 6.1.3 at atol=1e-7

affects:
  - 10-04 through 10-06 (other kernel plans can reuse 3c1e as reference)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Three-center overlap VRR: combined j+k dimension filled first, then HRR separates i and k"
    - "G-tensor g_alloc = max(dli*dlj*dlk, dli*vrr_nmax) — parentheses required for correct operator precedence"
    - "3c1e overlap registered in manifest as operator/stable so eval_raw dispatches through launch_center_3c1e"

key-files:
  created:
    - crates/cintx-oracle/tests/center_3c1e_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-compat/src/raw.rs

key-decisions:
  - "Add int3c1e_sph/int3c1e_cart to manifest (Rule 2 deviation): the plan specified eval_raw for the oracle test but the operator was absent from the manifest; adding it is required for correctness"
  - "Implement g_alloc as (dli*dlj*dlk).max(dli*vrr_nmax) not dli*dlj*dlk.max(vrr_nmax) to match libcint g_size = MAX(dli*dlj*dlk, dli*nmax)"

# Metrics
duration: 12min
completed: 2026-04-03
tasks: 2
files: 5
---

# Phase 10 Plan 03: 3c1e Real Kernel and Oracle Gate Closure Summary

**Real three-center one-electron overlap kernel implementing CINTg3c1e_ovlp VRR+HRR algorithm, with oracle parity test passing at atol=1e-7 against vendored libcint 6.1.3 for H2O STO-3G across all 125 shell triples**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-03T08:03:08Z
- **Completed:** 2026-04-03T08:15:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Replaced zero-returning stub in `center_3c1e.rs` with real `CINTg3c1e_ovlp` algorithm
- Implemented `fill_g_tensor_3c1e`: three-center Gaussian overlap VRR in combined j+k direction, i-HRR, then k-separation HRR
- Implemented `contract_3c1e_ovlp`: reads G-tensor indices `g[ix + jx*dj + kx*dk]` × `g[iy + jy*dj + ky*dk]` × `g[iz + jz*dj + kz*dk]`
- Applied `common_fac_sp(li)*common_fac_sp(lj)*common_fac_sp(lk)` scaling and `cart_to_sph_3c1e` transform
- Added `int3c1e_sph` and `int3c1e_cart` to manifest CSV and RS files so `eval_raw` dispatches to the kernel
- Created `center_3c1e_parity.rs` with self-consistency test (nonzero + idempotency) and vendor parity test
- Fixed operator precedence bug in `g_alloc` formula (found during vendor parity testing)

## Task Commits

1. **Task 1: Implement 3c1e kernel** — `da2a513` (feat)
2. **Task 2: Add oracle parity test + manifest + bug fix** — `af2ea35` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` — Full kernel implementation replacing stub
- `crates/cintx-oracle/tests/center_3c1e_parity.rs` — Oracle parity test (self-consistency + vendor)
- `crates/cintx-ops/src/generated/api_manifest.csv` — Added int3c1e_sph and int3c1e_cart rows
- `crates/cintx-ops/src/generated/api_manifest.rs` — Added int3c1e_sph and int3c1e_cart ManifestEntry structs
- `crates/cintx-compat/src/raw.rs` — Added INT3C1E_SPH and INT3C1E_CART constants to RawApiId

## Decisions Made

- **int3c1e_sph added to manifest:** The plan specified `eval_raw` for oracle comparisons. Since `int3c1e_sph` (overlap) was absent from the manifest (only `int3c1e_p2_sph` kinetic existed), `eval_raw` would fail with `MissingSymbol`. Added both cart and sph overlap variants as `operator/stable` entries with `canonical_family: "3c1e"` so eval_raw dispatches through `launch_center_3c1e`.
- **g_alloc formula:** Used `(dli*dlj*dlk).max(dli*vrr_nmax)` matching libcint's `MAX(dli*dlj*dlk, dli*nmax)`. Rust's method-call precedence would otherwise compute `dli*dlj*(dlk.max(vrr_nmax))` — wrong.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Functionality] Added int3c1e_sph/int3c1e_cart to manifest**
- **Found during:** Task 2 (oracle test design)
- **Issue:** Plan specified `eval_raw(RawApiId for "int3c1e_sph")` but the manifest only contained `int3c1e_p2_sph` (kinetic). Overlap was missing.
- **Fix:** Added `int3c1e_sph` and `int3c1e_cart` to manifest CSV, RS, and `RawApiId` constants. Added `int3c1e_sph` as the primary constant used in tests.
- **Files modified:** `api_manifest.csv`, `api_manifest.rs`, `raw.rs`
- **Commit:** af2ea35

**2. [Rule 1 - Bug] Fixed g_alloc operator precedence error**
- **Found during:** Task 2 (vendor parity revealed 74 mismatches with p-shells)
- **Issue:** `g_alloc = dli * dlj * dlk.max(vrr_nmax)` computed wrong allocation when `dlk < vrr_nmax`. For s-p-s: g_alloc=4 instead of 2, causing gy/gz offsets to be wrong in `contract_3c1e_ovlp`.
- **Fix:** `g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax)` matches libcint's `MAX(dli*dlj*dlk, dli*nmax)`.
- **Files modified:** `center_3c1e.rs`
- **Commit:** af2ea35

## Known Stubs

None. The 3c1e overlap kernel fully computes non-zero values for all shell triples.

## Verification Results

- `cargo build -p cintx-cubecl`: Finished with 0 errors
- `cargo test -p cintx-cubecl --features cpu -- center_3c1e`: 2 unit tests pass
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test center_3c1e_parity`:
  - `test_int3c1e_sph_h2o_sto3g_nonzero`: PASS (178 nonzero elements, idempotency verified)
  - `test_int3c1e_sph_h2o_sto3g_vendor_parity`: PASS (mismatch_count=0 at atol=1e-7)
