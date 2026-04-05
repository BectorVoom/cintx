---
phase: 13-f12-stg-yp-kernels
plan: "01"
subsystem: math
tags: [f12, stg, yp, gaussian-geminal, quadrature, roots, cubecl, error-handling, manifest]

requires:
  - phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
    provides: kernel dispatch infrastructure and oracle parity for base integral families

provides:
  - CINTstg_roots Rust port (stg_roots_host) with Clenshaw/DCT algorithm
  - COS_14_14 static table (196 elements) and helper functions
  - roots_xw.dat embedded as binary data accessible via bytemuck cast
  - InvalidEnvParam error variant on cintxRsError
  - OperatorEnvParams struct with f12_zeta field on ExecutionPlan
  - validate_f12_env_params function rejecting zeta==0 for f12 family
  - canonical_family="f12" for all 10 F12/STG/YP manifest entries
  - f12 dispatch arm in resolve_family_name gated by with-f12 feature
  - f12 stub kernel returning UnsupportedApi

affects:
  - 13-02-PLAN.md (builds F12 kernel on this foundation)
  - cintx-cubecl math infrastructure
  - cintx-runtime validator and planner contracts
  - cintx-ops manifest dispatch routing

tech-stack:
  added: [bytemuck (cast_slice for binary data tables)]
  patterns:
    - Binary data embedding via include_bytes! with AlignedBytes wrapper for f64 alignment
    - Host-side Rust port of C Clenshaw recurrence matching stg_roots.c exactly
    - OperatorEnvParams as extensible env parameter carrier on ExecutionPlan
    - Validator gate pattern: validate_f12_env_params before kernel launch

key-files:
  created:
    - crates/cintx-cubecl/src/math/stg.rs
    - crates/cintx-cubecl/src/math/roots_xw_data.rs
    - crates/cintx-cubecl/src/math/roots_xw_x.bin
    - crates/cintx-cubecl/src/math/roots_xw_w.bin
    - crates/cintx-cubecl/src/kernels/f12.rs
  modified:
    - crates/cintx-cubecl/src/math/mod.rs
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-core/src/error.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-runtime/src/lib.rs

key-decisions:
  - "Use include_bytes! with AlignedBytes<N> wrapper for roots_xw.dat: avoids include! macro expression limitation for 1.7M-element arrays; bytemuck::cast_slice provides zero-copy &[f64] view"
  - "T_MAX clamp const is exactly 19682.99_f64 per D-07 and stg_roots.c line 416"
  - "OperatorEnvParams defaults to all-None; populated by caller (raw compat reads env[9], safe API via ExecutionOptions) — not in ExecutionPlan::new()"
  - "validate_f12_env_params treats both None and Some(0.0) as invalid zeta, returns InvalidEnvParam with PTR_F12_ZETA param name"
  - "UNSUPPORTED_FOLLOW_ON_FAMILIES uses four cfg combinations for both with-4c1e and with-f12 to keep the list always accurate"

patterns-established:
  - "Binary table embedding: static AlignedBytes<{N * 8}> with include_bytes! + bytemuck::cast_slice for large f64 tables"
  - "Clenshaw recurrence host port: mirror C function signatures exactly, use macro_rules! for step unrolling"
  - "Feature-gated family stub: create minimal module that returns UnsupportedApi until real kernel lands in next plan"

requirements-completed: [F12-01, F12-04, F12-05]

duration: 15min
completed: 2026-04-05
---

# Phase 13 Plan 01: F12/STG/YP Foundation Summary

**CINTstg_roots ported to Rust with 1.7M-element binary tables, F12 manifest dispatch routing, OperatorEnvParams plumbing, and zeta==0 validator gate ready for Plan 02 kernel implementation.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-05T02:53:00Z
- **Completed:** 2026-04-05T03:08:11Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments

- Ported `CINTstg_roots` algorithm to host Rust as `stg_roots_host(nroots, ta, ua)` with Clenshaw/DCT recurrence, COS_14_14 table, and T_MAX clamp
- Embedded `roots_xw.dat` (1,783,600 f64 values each for DATA_X and DATA_W) as binary blobs via `include_bytes!` with `AlignedBytes` wrapper, accessible via `bytemuck::cast_slice`
- Wired F12 family through manifest and kernel dispatch: 10 entries now have `canonical_family="f12"`, `resolve_family_name` returns `Some(launch_f12)` under `with-f12` feature
- Added `OperatorEnvParams { f12_zeta }` on `ExecutionPlan` and `validate_f12_env_params` rejecting zero/None zeta before kernel launch

## Task Commits

1. **Task 1: Port CINTstg_roots and embed roots_xw.dat tables** - `3c88448` (feat)
2. **Task 2: Add InvalidEnvParam, manifest update, OperatorEnvParams, validator, f12 dispatch** - `1cac900` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/math/stg.rs` - CINTstg_roots host port with COS_14_14, _clenshaw_dc, _matmul_14_14, _clenshaw_d1
- `crates/cintx-cubecl/src/math/roots_xw_data.rs` - Binary table loader via include_bytes! + bytemuck
- `crates/cintx-cubecl/src/math/roots_xw_x.bin` - DATA_X binary (1783600 f64 LE values)
- `crates/cintx-cubecl/src/math/roots_xw_w.bin` - DATA_W binary (1783600 f64 LE values)
- `crates/cintx-cubecl/src/kernels/f12.rs` - Stub launch_f12 returning UnsupportedApi
- `crates/cintx-cubecl/src/math/mod.rs` - Added pub mod stg and roots_xw_data
- `crates/cintx-cubecl/src/kernels/mod.rs` - f12 arm in resolve_family_name and supports_canonical_family
- `crates/cintx-core/src/error.rs` - InvalidEnvParam variant on cintxRsError
- `crates/cintx-ops/src/generated/api_manifest.rs` - 10 F12 entries: canonical_family "2e" -> "f12"
- `crates/cintx-runtime/src/planner.rs` - OperatorEnvParams struct + field on ExecutionPlan
- `crates/cintx-runtime/src/validator.rs` - validate_f12_env_params function
- `crates/cintx-runtime/src/lib.rs` - Re-export OperatorEnvParams

## Decisions Made

- Used `include_bytes!` + `AlignedBytes<{N*8}>` wrapper for binary table embedding: the `include!` macro cannot accept comma-separated expressions (rejects at `incomplete_include` lint), so binary + bytemuck is the correct approach for 1.7M-element tables
- T_MAX constant is `19682.99_f64` exactly per D-07 requirement
- OperatorEnvParams defaults to all-None in `ExecutionPlan::new()`; callers are responsible for populating f12_zeta from `env[9]`
- Four cfg combinations used for UNSUPPORTED_FOLLOW_ON_FAMILIES to handle the cross-product of with-4c1e and with-f12

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed include! macro approach for data embedding**
- **Found during:** Task 1 (roots_xw_data.rs)
- **Issue:** Plan said to use `include!` macro for data, but Rust's `include!` rejects comma-separated values inside slice literals with `incomplete_include` lint error
- **Fix:** Switched to binary encoding + `include_bytes!` with `AlignedBytes<N>` alignment wrapper and `bytemuck::cast_slice` for zero-copy `&[f64]` view
- **Files modified:** crates/cintx-cubecl/src/math/roots_xw_data.rs
- **Verification:** `cargo check --features cpu,with-f12 -p cintx-cubecl` passes cleanly
- **Committed in:** 3c88448

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Functionally equivalent to plan intent; binary embedding is more efficient for large numeric tables than text include anyway.

## Issues Encountered

None beyond the include! macro approach (resolved via deviation rule above).

## Known Stubs

- `crates/cintx-cubecl/src/kernels/f12.rs`: `launch_f12` always returns `UnsupportedApi`. This is intentional — Plan 13-02 implements the real F12/STG/YP kernel. The stub is not a blocker for this plan's goal (infrastructure foundation).

## Next Phase Readiness

- Plan 13-02 can now implement F12 kernel entry points: `stg_roots_host` is working, dispatch routing is in place, `OperatorEnvParams` carries `f12_zeta`, and validator gate is ready
- Both `--features cpu` and `--features cpu,with-f12` compile without errors
- 6 stg math tests, 4 validator f12 tests, and 1 kernels/mod.rs f12 test pass

---
*Phase: 13-f12-stg-yp-kernels*
*Completed: 2026-04-05*
