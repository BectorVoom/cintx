---
phase: 02-cpu-compatibility-execution
plan: "02"
subsystem: api
tags: [raw-compat, libcint-layout, validator, diagnostics]
requires:
  - phase: 02-cpu-compatibility-execution
    provides: deterministic CPU linkage and shared execution request contracts from 02-01
provides:
  - Raw libcint-style view wrappers for atm/bas/env/shls/dims/cache/opt contracts
  - Centralized RAW-01 validator boundary with shell/dims/env/cache/opt invariants
  - Diagnostics-backed raw compatibility API entrypoint for pre-dispatch contract checks
affects: [phase-2-raw-runtime, raw-api, runtime-validation]
tech-stack:
  added: []
  patterns: [raw borrowed views, fail-fast contract validator, diagnostics-backed API boundary]
key-files:
  created: [src/runtime/raw/mod.rs, src/runtime/raw/views.rs, src/runtime/raw/validator.rs, tests/phase2_raw_contracts.rs]
  modified: [src/runtime/mod.rs, src/api/raw.rs, tests/phase2_raw_contracts.rs]
key-decisions:
  - "Keep legacy typed raw query API intact while adding a dedicated libcint-compatible validation surface."
  - "Enforce dims contracts at the validation boundary by requiring provided dims to match natural contracted-shell dims."
  - "Require cache presence when opt is provided to keep optional optimizer/cache handling explicit and deterministic."
patterns-established:
  - "All libcint raw layout interpretation is routed through runtime/raw views and validator modules."
  - "Raw compatibility API failures return QueryError with diagnostics payload instead of bare LibcintRsError."
requirements-completed: [RAW-01]
duration: 16 min
completed: 2026-03-14
---

# Phase 2 Plan 02: Raw Validation Boundary Summary

**RAW-01 now has a strict validation boundary for libcint-style raw buffers, with diagnostics-backed API entrypoints that fail malformed layouts before runtime dispatch.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-03-14T06:28:35Z
- **Completed:** 2026-03-14T06:44:45Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Added `src/runtime/raw/views.rs` borrowed raw view types with slot divisibility checks, signed offset guards, and env range validation for `atm`/`bas`.
- Added `src/runtime/raw/validator.rs` centralized RAW-01 contract validation for shell tuple arity, contracted dims invariants, env offsets, and cache/opt coupling.
- Added `raw.compat.query_workspace` in `src/api/raw.rs` so callers can validate libcint-style requests and receive typed diagnostics before execution integration.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build raw compatibility view types for libcint buffers** - `026861f` (feat)
2. **Task 2: Implement RAW-01 validator boundary rules** - `54233bb` (feat)
3. **Task 3: Expose validation-backed raw API surface and tests** - `4af320f` (feat)

## Files Created/Modified
- `src/runtime/raw/mod.rs` - New raw runtime module surface and re-exports.
- `src/runtime/raw/views.rs` - Borrowed libcint layout view wrappers with slot/offset/range guards.
- `src/runtime/raw/validator.rs` - RAW-01 validator entrypoint and contracted-shell dims/caching invariants.
- `src/runtime/mod.rs` - Registered the `runtime::raw` module.
- `src/api/raw.rs` - Added `RawCompatRequest` and diagnostics-backed `query_workspace_compat`.
- `tests/phase2_raw_contracts.rs` - Added RAW-01 contract regression coverage for views, validator, and API boundary.

## Decisions Made
- Kept the Phase 1 typed `raw::query_workspace` API unchanged to avoid cross-plan breakage, and introduced a dedicated compat entrypoint for libcint layouts.
- Bound dims to natural contracted-shell dimensions at validation time to guarantee deterministic shape contracts before planner/backend integration.
- Encoded a strict `opt -> cache required` invariant in the validator to make optional execution-state requirements explicit and testable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bootstrapped the missing `phase2_raw_contracts` test target during Task 1**
- **Found during:** Task 1 (Build raw compatibility view types for libcint buffers)
- **Issue:** The required verification command referenced `tests/phase2_raw_contracts.rs`, which did not exist.
- **Fix:** Created the integration test target and added `raw_layout_slot_and_offset_checks` coverage.
- **Files modified:** `tests/phase2_raw_contracts.rs`
- **Verification:** `cargo test --workspace --test phase2_raw_contracts raw_layout_slot_and_offset_checks`
- **Committed in:** `026861f`

**2. [Rule 1 - Bug] Removed invalid `Eq` derive from `RawEnvView`**
- **Found during:** Task 1 verification
- **Issue:** `RawEnvView` stored `&[f64]`; deriving `Eq` failed because `f64` does not implement `Eq`.
- **Fix:** Reduced derive to `PartialEq` for `RawEnvView`.
- **Files modified:** `src/runtime/raw/views.rs`
- **Verification:** Re-ran Task 1 verification command successfully.
- **Committed in:** `026861f`

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes were scoped unblockers for mandated verification and did not expand scope beyond RAW-01 contract validation.

## Issues Encountered
- Initial Task 1 compile failed due to invalid trait derivation on `RawEnvView`; fixed immediately and verification was rerun successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Raw validation primitives are now in place for 02-07 raw query/evaluate integration.
- RAW-01 contract failures are test-locked before planner/backend execution semantics are introduced.

---
*Phase: 02-cpu-compatibility-execution*
*Completed: 2026-03-14*

## Self-Check: PASSED
- Found `.planning/phases/02-cpu-compatibility-execution/02-02-SUMMARY.md`.
- Verified commits `026861f`, `54233bb`, and `4af320f` in `git log --oneline --all`.
