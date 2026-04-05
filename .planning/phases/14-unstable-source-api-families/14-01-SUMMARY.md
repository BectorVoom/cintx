---
phase: 14-unstable-source-api-families
plan: 01
subsystem: unstable-source-api infrastructure
tags: [feature-gate, manifest, kernel-dispatch, oracle, validator]
dependency_graph:
  requires: []
  provides:
    - unstable-source-api feature gate in cintx-cubecl
    - 18 ManifestEntry items (origi/grids/breit/origk/ssc families)
    - GridsEnvParams struct and validate_grids_env_params
    - kernel dispatch stubs for all 5 unstable families
    - oracle build/FFI/test scaffold for 18 symbols
  affects:
    - crates/cintx-cubecl (new feature + module)
    - crates/cintx-ops (manifest expansion)
    - crates/cintx-runtime (planner + validator extension)
    - crates/cintx-oracle (build, FFI, test scaffold)
    - crates/cintx-rs (feature forwarding)
tech_stack:
  added: []
  patterns:
    - unstable-source-api cfg-gated dispatch mirroring with-f12 pattern
    - GridsEnvParams/validate_grids_env_params mirroring F12 env params pattern
    - dynamic unresolved_families() replacing static cfg-gated const arrays
key_files:
  created:
    - crates/cintx-cubecl/src/kernels/unstable.rs
    - crates/cintx-oracle/tests/unstable_source_parity.rs
  modified:
    - crates/cintx-cubecl/Cargo.toml
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-runtime/src/lib.rs
    - crates/cintx-oracle/Cargo.toml
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-rs/Cargo.toml
decisions:
  - "Convert unresolved_families() from static &[&str] to Vec<&str> to support dynamic 3-feature combination without 8 cfg variants"
  - "Use compact single-line ManifestEntry format for Phase 14 entries to avoid another very large Edit operation"
  - "Grids FFI wrappers use [i32; 4] shls to match libcint cint1e_grids.c signature (i, j, grid_start, grid_end)"
metrics:
  duration_minutes: 16
  completed_date: "2026-04-05"
  tasks_completed: 2
  files_modified: 11
  files_created: 2
---

# Phase 14 Plan 01: Unstable-Source API Infrastructure Summary

Complete infrastructure for all 5 unstable-source families: feature gate propagation from cintx-rs through cintx-cubecl, 18 manifest entries with `compiled_in_profiles: &["unstable-source"]`, GridsEnvParams plumbing with NGRIDS validation, oracle build/FFI/test scaffold, and kernel dispatch stubs.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Feature gates, manifest entries, GridsEnvParams, kernel stubs | d9f20ee | 9 files |
| 2 | Oracle build.rs supplemental header, vendor FFI, test scaffold | 0aa712e | 4 files |

## What Was Built

### Task 1: Feature gates, manifest entries, GridsEnvParams, kernel dispatch stubs

**Feature gate propagation (D-01, D-04):**
- `unstable-source-api = []` added to `crates/cintx-cubecl/Cargo.toml`
- `cintx-cubecl/unstable-source-api` added to cintx-rs forwarding chain
- Feature now propagates: cintx-rs -> cintx-cubecl -> kernels module

**Manifest (D-01, D-02, D-03):**
- 18 new `ManifestEntry` items at indices 112-129 in `api_manifest.rs`
- 18 matching `OperatorDescriptor` entries
- `PROFILE_SCOPE_APPROVED` extended to include `"unstable-source"`
- Families: origi (4), grids (5), breit (2), origk (6), ssc (1)
- All have `compiled_in_profiles: &["unstable-source"]` per D-01/D-02
- Breit entries use `RepresentationSupport::new(false, false, true)` per D-07

**GridsEnvParams plumbing (D-05, D-06):**
- `GridsEnvParams { ngrids: usize, ptr_grids: usize }` added to `planner.rs`
- `grids_params: Option<GridsEnvParams>` field added to `OperatorEnvParams`
- `validate_grids_env_params` added to `validator.rs` — rejects None and ngrids==0
- 4 tests: none rejected, ngrids=0 rejected, valid passes, non-grids skips
- Exported from cintx-runtime lib.rs

**Kernel dispatch (D-04):**
- `#[cfg(feature = "unstable-source-api")] pub mod unstable;` added to mod.rs
- 5 dispatch arms added to `resolve_family_name` (origi, grids, breit, origk, ssc)
- `supports_canonical_family` extended for all 5 families
- `unresolved_families()` converted from static const arrays (4 cfg variants) to dynamic `Vec<&str>` function — cleaner and handles 3-feature combinations without 8 cfg variants

**Stub kernels:**
- `crates/cintx-cubecl/src/kernels/unstable.rs` created with 5 stub `pub fn launch_*` functions
- All return `Err(cintxRsError::UnsupportedApi { requested: "family: stub" })`

### Task 2: Oracle build.rs supplemental header, vendor FFI, test scaffold

**Oracle Cargo.toml:**
- `unstable-source-api = ["cintx-compat/unstable-source-api"]` feature added

**build.rs C source files:**
- `cint1e_grids.c` + `g1e_grids.c` added (int1e_grids* family)
- `breit.c` added (int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor)
- `cint3c1e_a.c` added (int3c1e_r2/r4/r6_origk + ip1 derivatives)
- `cint1e_a.c` and `cint3c2e.c` already present (origi and ssc)
- Rerun triggers added for all new files

**Supplemental header (18 extern declarations):**
- 4 origi symbols (cint1e_a.c)
- 5 grids symbols (cint1e_grids.c — not in cint_funcs.h per Pitfall 4)
- 2 breit spinor symbols (breit.c)
- 6 origk symbols (cint3c1e_a.c)
- 1 ssc symbol (cint3c2e.c)

**Bindgen allowlist:** All 18 symbol names added to the pipe-separated allowlist.

**vendor_ffi.rs:**
- 18 new safe wrapper functions appended
- 1e/3c1e/3c2e wrappers use `shls: &[i32; N]` matching family arity
- Grids wrappers use `shls: &[i32; 4]` (i, j, grid_start, grid_end) per cint1e_grids.c

**Oracle test scaffold:**
- `tests/unstable_source_parity.rs` created with 5 per-family modules
- Gated behind `#![cfg(feature = "cpu")]` + `#![cfg(feature = "unstable-source-api")]`
- Empty module stubs — implementations added in Wave 2 plans

## Verification Results

All success criteria passed:

- `cargo check --features unstable-source-api -p cintx-cubecl -p cintx-runtime -p cintx-ops` exits 0
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo check -p cintx-oracle --features cpu,unstable-source-api` exits 0
- `cargo test -p cintx-runtime -- validate_grids` — 4 tests passed
- grep confirms 18 new entries in api_manifest.rs with `compiled_in_profiles: &["unstable-source"]`
- All 21 acceptance criteria checks passed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] OperatorEnvParams struct literal tests broken after adding grids_params field**
- **Found during:** Task 1 test run
- **Issue:** Existing validator tests used `OperatorEnvParams { f12_zeta: ... }` struct literals which become compile errors when `grids_params` field was added
- **Fix:** Updated 3 existing test struct literals to use `..OperatorEnvParams::default()` struct update syntax
- **Files modified:** `crates/cintx-runtime/src/validator.rs`
- **Commit:** d9f20ee (included in Task 1 commit)

**2. [Rule 2 - Missing functionality] unresolved_families() needed dynamic behavior for 3 features**
- **Found during:** Task 1 implementation
- **Issue:** Plan said to "expand from 4 to 8 cfg combinations" or "convert to function that builds the list dynamically". 8 cfg combinations is fragile; dynamic function is cleaner and correct.
- **Fix:** Replaced static `UNSUPPORTED_FOLLOW_ON_FAMILIES` const with 4 cfg variants with a `unresolved_families() -> Vec<&'static str>` function
- **Files modified:** `crates/cintx-cubecl/src/kernels/mod.rs`
- **Commit:** d9f20ee

**3. [Rule 3 - Blocking issue] Manifest entries needed compact single-line format**
- **Found during:** Task 1 manifest editing
- **Issue:** Previous multi-line Edit attempts failed silently due to large replacement size; the "File has been modified since read" error appears when multiple edits are interleaved
- **Fix:** Used compact single-line ManifestEntry format to fit in a single Edit operation
- **Files modified:** `crates/cintx-ops/src/generated/api_manifest.rs`
- **Commit:** d9f20ee

## Known Stubs

The following stub functions exist intentionally — Wave 2 plans will implement real kernels:

| Stub | File | Purpose |
|------|------|---------|
| `launch_origi` | `crates/cintx-cubecl/src/kernels/unstable.rs` | Kernel pending Phase 14 Plan 02 |
| `launch_grids` | `crates/cintx-cubecl/src/kernels/unstable.rs` | Kernel pending Phase 14 Plan 02 |
| `launch_breit` | `crates/cintx-cubecl/src/kernels/unstable.rs` | Kernel pending Phase 14 Plan 03 |
| `launch_origk` | `crates/cintx-cubecl/src/kernels/unstable.rs` | Kernel pending Phase 14 Plan 03 |
| `launch_ssc` | `crates/cintx-cubecl/src/kernels/unstable.rs` | Kernel pending Phase 14 Plan 04 |
| `mod origi_parity {}` | `crates/cintx-oracle/tests/unstable_source_parity.rs` | Tests pending Plan 02 |
| `mod grids_parity {}` | `crates/cintx-oracle/tests/unstable_source_parity.rs` | Tests pending Plan 02 |
| `mod breit_parity {}` | `crates/cintx-oracle/tests/unstable_source_parity.rs` | Tests pending Plan 03 |
| `mod origk_parity {}` | `crates/cintx-oracle/tests/unstable_source_parity.rs` | Tests pending Plan 03 |
| `mod ssc_parity {}` | `crates/cintx-oracle/tests/unstable_source_parity.rs` | Tests pending Plan 04 |

These stubs are intentional and expected — Plan 01's goal is infrastructure setup for Wave 2.

## Self-Check: PASSED
