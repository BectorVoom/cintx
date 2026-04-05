---
phase: 14-unstable-source-api-families
plan: "04"
subsystem: cintx-cubecl/kernels + cintx-oracle
tags: [breit, spinor, oracle-parity, unstable-source-api, two-electron]
dependency_graph:
  requires: ["14-01"]
  provides: ["USRC-03"]
  affects: ["cintx-cubecl", "cintx-compat", "cintx-runtime", "cintx-ops", "cintx-oracle"]
tech_stack:
  added: []
  patterns:
    - "c2s_sf_2e1i + c2s_sf_2e2i → negate cart_buf (i^2 = -1 phase convention)"
    - "VRR/HRR g-tensor with elevated dims for derivative headroom"
    - "9-term gout contraction using nabla1i/j/l + x1j/x1l operators"
key_files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/unstable.rs
    - crates/cintx-oracle/tests/unstable_source_parity.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-runtime/src/dispatch.rs
    - crates/cintx-compat/Cargo.toml
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-ops/generated/compiled_manifest.lock.json
decisions:
  - "Breit uses single direct gout kernel (not three-stage BREIT0 composite) for r1p2 and r2p2"
  - "iket phase: negate cart_buf instead of implementing separate iket spinor transform path"
  - "family_name uses 'unstable::source::breit' prefix; canonical_family stays 'breit' for dispatch routing"
metrics:
  duration: "2 sessions (context carryover)"
  completed_date: "2026-04-05"
  tasks_completed: 2
  files_modified: 8
---

# Phase 14 Plan 04: Breit Kernel and Oracle Parity Summary

Implemented `launch_breit` for `int2e_breit_r1p2_spinor` and `int2e_breit_r2p2_spinor` — spinor-only Breit 2e integrals using VRR/HRR g-tensor construction with elevated angular momenta, 9-term gout contraction, and iket phase negation to match libcint's `c2s_sf_2e1i + c2s_sf_2e2i` spinor transform convention.

## Tasks Completed

| Task | Commit | Description |
|------|--------|-------------|
| 1: Breit kernel | `ebdc69a` | `launch_breit` with BreitShape, VRR/HRR, nabla/position operators, gout r1p2/r2p2 |
| 2: Oracle parity | `38d96ea` | Both Breit spinor oracle tests pass at atol=1e-12; all blocking fixes applied |

## Acceptance Criteria

- `int2e_breit_r1p2_spinor`: vendor parity PASS, n_elem=32, mismatches=0, any_nonzero=true (atol=1.0e-12)
- `int2e_breit_r2p2_spinor`: vendor parity PASS, n_elem=32, mismatches=0, any_nonzero=true (atol=1.0e-12)
- Cart/sph representations rejected by manifest forms guard (forms: `["spinor"]` only)
- All workspace tests pass with `--features cpu,unstable-source-api`

## Deviations from Plan

### Plan Description Correction

The plan described Breit as a "three-stage composite kernel" (BREIT0 macro). After reading `breit.c`, the target symbols `int2e_breit_r1p2_spinor` and `int2e_breit_r2p2_spinor` are standard single-pass 2e integrals with their own gout functions — NOT the composite three-stage drivers. The composite `BREIT0` macro defines `int2e_breit_ssp1ssp2_spinor` etc., not `int2e_breit_r1p2_spinor`. Implementation used single-pass gout approach.

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing DispatchFamily variants for Phase 14 families**
- **Found during:** Task 2 testing
- **Issue:** `cintx-runtime/src/dispatch.rs` had no match arms for "breit", "origi", "grids", "origk", "ssc" families — returned UnsupportedApi
- **Fix:** Added Origi/Grids/Breit/Origk/Ssc variants with "unstable::source::" family_name prefix
- **Files modified:** `crates/cintx-runtime/src/dispatch.rs`
- **Commit:** `38d96ea`

**2. [Rule 1 - Bug] Feature propagation broken — unstable-source-api not reaching cintx-cubecl**
- **Found during:** Task 2 testing — executor.supports() returned false for breit plans
- **Issue:** `cintx-compat/Cargo.toml` had `unstable-source-api = []` instead of `= ["cintx-cubecl/unstable-source-api"]`, so the kernel dispatch module (guarded by `#[cfg(feature = "unstable-source-api")]`) never compiled in the oracle test binary
- **Fix:** Changed feature definition to propagate to cubecl
- **Files modified:** `crates/cintx-compat/Cargo.toml`
- **Commit:** `38d96ea`

**3. [Rule 1 - Bug] api_manifest.rs family_name missing "unstable::source::" prefix**
- **Found during:** Task 2 — source_only_symbols_are_identifiable test failed
- **Issue:** Phase 14 entries used short `family_name: "breit"` etc. but the test invariant requires `family_name.starts_with("unstable::source::")`
- **Fix:** Updated all Phase 14 `family_name` fields to use "unstable::source::{short}" prefix while keeping `canonical_family` as the short dispatch name
- **Files modified:** `crates/cintx-ops/src/generated/api_manifest.rs`, `api_manifest.csv`
- **Commit:** `38d96ea`

**4. [Rule 1 - Bug] compiled_manifest.lock.json missing Phase 14 entries**
- **Found during:** Task 2 — oracle fixtures.rs reported "manifest lock metadata missing" for Phase 14 symbols
- **Issue:** The compiled manifest lock had only 112 entries (0-111); Phase 14 added 18 new symbols (112-129)
- **Fix:** Added all 18 Phase 14 entries to the lock JSON (130 entries total)
- **Files modified:** `crates/cintx-ops/generated/compiled_manifest.lock.json`
- **Commit:** `38d96ea`

**5. [Rule 1 - Bug] Sign error — missing iket phase convention**
- **Found during:** Task 2 oracle tests — 4 of 32 elements had correct magnitude but wrong sign
- **Issue:** libcint uses `c2s_sf_2e1i + c2s_sf_2e2i` (iket variants) for Breit, which applies a factor of `i` to both the j-ket (step 1) and l-ket (step 2). Combined phase: `i_j * i_l = i^2 = -1`. My code used regular `cart_to_spinor_sf_4d` without this phase
- **Fix:** Negate the staging buffer after `cart_to_spinor_sf_4d` to match the `-1` iket phase
- **Files modified:** `crates/cintx-cubecl/src/kernels/unstable.rs`
- **Commit:** `38d96ea`

**6. [Rule 1 - Bug] raw.rs validate_profile_and_source_gate rejected source-only symbols**
- **Found during:** Earlier session — "raw api is not compiled in active profile base"
- **Issue:** Profile gate checked `is_compiled_in_profile("base")` before checking source-only path; breit entries have `compiled_in_profiles: ["unstable-source"]` so this always failed
- **Fix:** Short-circuit for source-only symbols when `unstable_source_api_enabled()` returns true
- **Files modified:** `crates/cintx-compat/src/raw.rs`
- **Commit:** `38d96ea`

## Key Technical Decisions

1. **Single-pass gout approach**: `int2e_breit_r1p2_spinor` uses a direct gout (not BREIT0 composite). The gout uses nabla1i/j/l operators and x1j/x1l position operators applied to an elevated g-tensor (ng={2,2,0,1}) producing a 9-term sum.

2. **iket phase negation**: Rather than implementing a separate `cart_to_spinor_sf_4d_iket` function, we negate the staging buffer. This is mathematically equivalent and simpler.

3. **family_name convention**: Phase 14 families use "unstable::source::{name}" in `family_name` (satisfying the test invariant) and the short name in `canonical_family` (for dispatch routing via `kernels/mod.rs` which reads `entry.canonical_family`).

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/unstable.rs
- FOUND: crates/cintx-oracle/tests/unstable_source_parity.rs
- FOUND: commit ebdc69a (Task 1: Breit kernel)
- FOUND: commit 38d96ea (Task 2: Oracle parity + fixes)
