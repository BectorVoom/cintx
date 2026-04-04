---
phase: 11-helper-transform-completion-4c1e-real-kernel
plan: 03
subsystem: oracle
tags: [4c1e, workaround, vendor-ffi, legacy-oracle, oracle-gate, int4c1e_sph, int4c1e_via_2e_trace]

# Dependency graph
requires:
  - phase: 11-helper-transform-completion-4c1e-real-kernel/11-01
    provides: unified UNIFIED_ATOL=1e-12 tolerance, helper surface coverage, oracle baseline
  - phase: 11-helper-transform-completion-4c1e-real-kernel/11-02
    provides: real 4c1e kernel (center_4c1e.rs) producing non-zero output via CubeCL
provides:
  - 4c1e workaround via 2e trace contraction (crates/cintx-compat/src/workaround.rs)
  - vendor_int4c1e_sph/cart FFI wrappers for oracle comparison (crates/cintx-oracle/src/vendor_ffi.rs)
  - verify_legacy_wrapper_parity function comparing 7 sph integral symbols at atol=1e-12
  - oracle_gate_4c1e_nonzero_output test (non-vendor) proving real 4c1e kernel path
  - oracle_gate_4c1e_parity test (vendor) for with-4c1e profile gate closure
  - with-4c1e and with-f12 feature gates forwarded through cintx-oracle Cargo.toml
  - helper-legacy-parity gate passing all four feature profiles
affects:
  - 4c1e oracle gate CI jobs (with-4c1e profile)
  - helper-legacy-parity CI gate (all four profiles)
  - cintx-oracle Cargo.toml (with-4c1e/with-f12 features)
  - Phase 12+ spinor transform plans that extend oracle harness

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "4c1e workaround via 2e trace: eval_raw(INT2E_SPH) followed by sum over k=l diagonal gives 4c1e equivalence"
    - "verify_legacy_wrapper_parity gated on has_vendor_libcint cfg — compares sph integral symbols at UNIFIED_ATOL"
    - "Oracle feature gates forwarded through cintx-oracle Cargo.toml so CI can test all four profiles"

key-files:
  created:
    - crates/cintx-compat/src/workaround.rs
  modified:
    - crates/cintx-compat/src/lib.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/src/compare.rs
    - crates/cintx-oracle/tests/oracle_gate_closure.rs
    - crates/cintx-oracle/Cargo.toml
    - crates/cintx-oracle/build.rs

key-decisions:
  - "int4c1e_via_2e_trace uses unsafe eval_raw(INT2E_SPH) and traces over k=l diagonal — no new CompatDims needed"
  - "verify_legacy_wrapper_parity compares 7 sph integral families (not all 45 symbols) since cart/spinor/optimizer variants are covered by surface checks and the sph comparison validates the shared kernel path"
  - "cintx-oracle Cargo.toml gains with-4c1e and with-f12 features forwarding to cintx-compat so CI can run helper-legacy-parity gate with all four profiles"
  - "oracle_gate_4c1e_parity needs both cfg(has_vendor_libcint) and cfg(feature=with-4c1e) gates since it requires both the real kernel and the vendored libcint reference"

requirements-completed: [HELP-03, 4C1E-02, 4C1E-04]

# Metrics
duration: 8min
completed: 2026-04-04
---

# Phase 11 Plan 03: 4c1e Workaround, Legacy Oracle, and Gate Closure Summary

**4c1e workaround via int2e trace contraction, vendor 4c1e FFI wrappers (int4c1e_sph/cart), legacy numeric oracle comparing 7 sph integral families at atol=1e-12, and oracle gate for with-4c1e profile — all four helper-legacy-parity CI profiles passing**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-04T08:21:59Z
- **Completed:** 2026-04-04T08:30:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Created `crates/cintx-compat/src/workaround.rs` with `int4c1e_via_2e_trace` which evaluates INT2E_SPH and traces over the k=l diagonal to produce 4c1e-equivalent output
- Added `vendor_int4c1e_sph` and `vendor_int4c1e_cart` to `vendor_ffi.rs`; updated `build.rs` to compile `cint4c1e.c`/`g4c1e.c` and generate FFI bindings for both symbols
- Added `verify_legacy_wrapper_parity` in `compare.rs` (gated on `has_vendor_libcint`) comparing 7 sph integral families (ovlp, kin, nuc, 2e, 2c2e, 3c1e, 3c2e) against vendored libcint at UNIFIED_ATOL=1e-12; wired into `verify_helper_surface_coverage`
- Added `oracle_gate_4c1e_nonzero_output` (non-vendor, runs in CI) and `oracle_gate_4c1e_parity` (vendor + with-4c1e) tests in oracle gate closure test suite
- Added `with-4c1e` and `with-f12` features to cintx-oracle Cargo.toml forwarding to cintx-compat
- All four CI profiles pass: `cpu`, `cpu+with-4c1e`, `cpu+with-f12`, `cpu+with-f12+with-4c1e`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create workaround module with int4c1e_via_2e_trace** - `3a5533b` (feat)
2. **Task 2: Add vendor 4c1e FFI, legacy numeric oracle, and close gates** - `0d3b2f8` (feat)

## Files Created/Modified

- `/home/chemtech/workspace/cintx/crates/cintx-compat/src/workaround.rs` - int4c1e_via_2e_trace function; 4c1e workaround by INT2E_SPH + trace contraction; gated on with-4c1e feature
- `/home/chemtech/workspace/cintx/crates/cintx-compat/src/lib.rs` - Added `#[cfg(feature = "with-4c1e")] pub mod workaround`
- `/home/chemtech/workspace/cintx/crates/cintx-oracle/src/vendor_ffi.rs` - Added vendor_int4c1e_sph and vendor_int4c1e_cart functions
- `/home/chemtech/workspace/cintx/crates/cintx-oracle/src/compare.rs` - Added verify_legacy_wrapper_parity (under has_vendor_libcint) and wired into verify_helper_surface_coverage
- `/home/chemtech/workspace/cintx/crates/cintx-oracle/tests/oracle_gate_closure.rs` - Added oracle_gate_4c1e_parity and oracle_gate_4c1e_nonzero_output tests
- `/home/chemtech/workspace/cintx/crates/cintx-oracle/Cargo.toml` - Added with-4c1e and with-f12 features forwarding to cintx-compat
- `/home/chemtech/workspace/cintx/crates/cintx-oracle/build.rs` - Added cint4c1e.c/g4c1e.c to vendor build; int4c1e_sph/cart in bindgen allowlist and supplemental header

## Decisions Made

- `int4c1e_via_2e_trace` calls `unsafe eval_raw` directly (matching the raw.rs pattern) — no need for a CompatDims since we pass `dims: None` and size the output buffer manually
- `verify_legacy_wrapper_parity` compares 7 sph integral families rather than all 45 symbol names: cart/spinor/optimizer variants share the same kernel path as sph and the sph comparison is the meaningful oracle check; spinor wrappers properly return UnsupportedApi
- Oracle feature flags forwarded from cintx-compat through cintx-oracle so all four CI profiles work without changes to the workspace root

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added with-4c1e and with-f12 features to cintx-oracle Cargo.toml**
- **Found during:** Task 2 (feature flag check for oracle tests)
- **Issue:** Plan's verification commands use `--features "cpu,with-4c1e"` on cintx-oracle but the crate had no such features
- **Fix:** Added `with-4c1e = ["cintx-compat/with-4c1e"]` and `with-f12 = ["cintx-compat/with-f12"]` to oracle's Cargo.toml
- **Files modified:** crates/cintx-oracle/Cargo.toml
- **Verification:** `cargo check -p cintx-oracle --features "cpu,with-4c1e"` exits 0
- **Committed in:** 0d3b2f8 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical feature declaration)
**Impact on plan:** Required for CI to run the with-4c1e profile tests. No scope creep.

## Issues Encountered

None — both tasks compiled cleanly on first attempt after the missing oracle feature gate was added.

## Next Phase Readiness

- Phase 11 plan 03 complete: all three plans in Phase 11 done
- 4c1e gate: `oracle_gate_4c1e_nonzero_output` passes unconditionally; `oracle_gate_4c1e_parity` (vendor comparison) runs when CINTX_ORACLE_BUILD_VENDOR=1
- helper-legacy-parity gate: all four profiles (cpu, cpu+with-4c1e, cpu+with-f12, cpu+with-f12+with-4c1e) pass
- Requirements HELP-03, 4C1E-02, 4C1E-04 satisfied
- Phase 12 (spinor transform) can proceed — oracle harness and workaround infrastructure complete

## Self-Check: PASSED

All files found:
- FOUND: crates/cintx-compat/src/workaround.rs
- FOUND: crates/cintx-oracle/src/vendor_ffi.rs
- FOUND: .planning/phases/11-helper-transform-completion-4c1e-real-kernel/11-03-SUMMARY.md

All commits found:
- FOUND: 3a5533b (Task 1 commit)
- FOUND: 0d3b2f8 (Task 2 commit)

---
*Phase: 11-helper-transform-completion-4c1e-real-kernel*
*Completed: 2026-04-04*
