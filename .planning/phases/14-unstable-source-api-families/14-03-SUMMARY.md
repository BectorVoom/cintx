---
phase: 14-unstable-source-api-families
plan: 03
subsystem: grids-kernel-launch
tags: [grids, kernel, oracle, parity, manifest, dispatch]
dependency_graph:
  requires:
    - 14-01 (infrastructure: GridsEnvParams, manifest entries, dispatch stubs)
  provides:
    - launch_grids kernel for all 5 grids operator variants
    - grids oracle parity smoke tests (5 symbols)
    - vendor oracle parity gates (5 symbols, cfg(has_vendor_libcint))
  affects:
    - crates/cintx-cubecl/src/kernels/unstable.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-compat/Cargo.toml
    - crates/cintx-runtime/src/dispatch.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/tests/unstable_source_parity.rs
tech_stack:
  added: []
  patterns:
    - Rys quadrature grids variant (grid center as pseudo-nucleus, +1 charge, no Z factor)
    - nabla_i / nabla_j G-tensor derivative chain on host
    - spvsp derived from ipvip (4-component combination: s5-s7, s6-s2, s1-s3, s0+s4+s8)
    - ipip column-transposed gout layout per libcint autocode
key_files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/unstable.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-compat/Cargo.toml
    - crates/cintx-runtime/src/dispatch.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/tests/unstable_source_parity.rs
decisions:
  - "grids kernel uses GridsEnvParams.grid_coords (not env ptr) so kernel is env-independent"
  - "DispatchFamily::UnstableSource added for grids/origi/breit/origk/ssc in planner dispatch"
  - "validate_profile_and_source_gate: source-only symbols skip profile check when feature enabled"
  - "manifest component_rank corrected: grids_ip=3, grids_ipvip=9, grids_ipip=9, grids_spvsp=4"
metrics:
  duration_minutes: 90
  completed_date: "2026-04-05"
  tasks_completed: 2
  files_modified: 8
---

# Phase 14 Plan 03: Grids Kernel Launch and Oracle Parity Tests Summary

**One-liner:** Host-side Rys quadrature grids kernel for 5 operator variants with grid-displaced centers, NGRIDS dimension multiplier, and 5 oracle parity smoke tests.

## Tasks Completed

1. **Task 1: Implement launch_grids kernel** — Complete Rys quadrature implementation for grids base, ip, ipvip, spvsp, and ipip variants. Uses GridsEnvParams.grid_coords for kernel-level grid access without env pointer. Staging buffer sized as `ncomp * ngrids * nsi * nsj`. Dispatch through `DispatchFamily::UnstableSource` planner variant.

2. **Task 2: Oracle parity tests for grids family** — 5 non-vendor smoke tests (`test_int1e_grids_*_nonzero`) plus 5 vendor oracle parity gates (`oracle_parity_int1e_grids_*`) in `unstable_source_parity.rs`. H2O/STO-3G fixture extended with ngrids grid points appended after PTR_ENV_START.

## Commits

- `99caf1e`: feat(14-03): implement grids kernel launch for 5 operator variants
- `6e2b372`: test(14-03): add grids oracle parity tests with 5 smoke tests

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed stale `env: &[f64]` parameter from launch_grids_kernel**
- **Found during:** Task 1 continuation
- **Issue:** Previous session had incomplete refactoring — `env` parameter still used `ptr_grids + g*3` indexing
- **Fix:** Replaced `env` parameter with direct use of `grids_params.grid_coords[g]`
- **Files modified:** `crates/cintx-cubecl/src/kernels/unstable.rs`
- **Commit:** 99caf1e

**2. [Rule 1 - Bug] Fixed unstable-source-api feature not propagating to cintx-cubecl**
- **Found during:** Task 2 test run
- **Issue:** `cintx-compat/Cargo.toml` declared `unstable-source-api = []` (empty feature); executor `supports_canonical_family("grids")` checked `cfg!(feature = "unstable-source-api")` in cintx-cubecl, which was always false
- **Fix:** Changed to `unstable-source-api = ["cintx-cubecl/unstable-source-api"]`
- **Files modified:** `crates/cintx-compat/Cargo.toml`
- **Commit:** 6e2b372

**3. [Rule 1 - Bug] Fixed validate_profile_and_source_gate rejecting source-only symbols**
- **Found during:** Task 2 test run (error: "not compiled in active profile base")
- **Issue:** Profile check required active profile ("base") to be in `compiled_in_profiles: ["unstable-source"]`; these two profiles are orthogonal gating mechanisms
- **Fix:** Source-only symbols skip the profile check when `unstable_source_api_enabled()` returns true
- **Files modified:** `crates/cintx-compat/src/raw.rs`
- **Commit:** 6e2b372

**4. [Rule 1 - Bug] Added DispatchFamily::UnstableSource for grids/origi/breit/origk/ssc**
- **Found during:** Task 2 test run (error: "unsupported dispatch family grids")
- **Issue:** `DispatchDecision::from_manifest_family("grids")` had no match for unstable families
- **Fix:** Added `UnstableSource` variant, mapped from all 5 unstable family names
- **Files modified:** `crates/cintx-runtime/src/dispatch.rs`
- **Commit:** 6e2b372

**5. [Rule 1 - Bug] Fixed grids manifest component_rank values**
- **Found during:** Task 2 test run (BufferTooSmall errors on multi-component variants)
- **Issue:** Plan 01 set `component_rank: "1"` for grids_ip/ipvip and `"2"` for ipip. These are literal component counts, not exponents. Correct values: ip=3, ipvip=9, ipip=9, spvsp=4
- **Fix:** Updated both `api_manifest.rs` and `api_manifest.csv`
- **Files modified:** `crates/cintx-ops/src/generated/api_manifest.rs`, `crates/cintx-ops/src/generated/api_manifest.csv`
- **Commit:** 6e2b372

**6. [Rule 2 - Missing functionality] Fixed GridsEnvParams test constructors in validator.rs**
- **Found during:** `cargo test -p cintx-runtime` after adding `grid_coords` field
- **Issue:** Tests constructed `GridsEnvParams { ngrids, ptr_grids }` without the new `grid_coords` field
- **Fix:** Added `grid_coords: vec![]` and `grid_coords: vec![[0.0,0.0,0.0]; 5]` to test structs
- **Files modified:** `crates/cintx-runtime/src/validator.rs`
- **Commit:** 6e2b372

## Known Stubs

The following grids-adjacent stubs remain in `unstable.rs` (by design, other plans):
- `launch_origi` — returns `UnsupportedApi` (Plan 02)
- `launch_breit` — returns `UnsupportedApi` (Plan 03 next wave)
- `launch_origk` — returns `UnsupportedApi` (Plan 03 next wave)
- `launch_ssc` — returns `UnsupportedApi` (Plan 04)

Vendor oracle parity gates in `unstable_source_parity.rs` require `CINTX_ORACLE_BUILD_VENDOR=1` and are not yet run. Non-vendor smoke tests confirm non-zero output.

## Self-Check: PASSED
