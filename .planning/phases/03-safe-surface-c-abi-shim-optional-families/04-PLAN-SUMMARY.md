---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 04
subsystem: api
tags: [rust, c-abi, ffi, tls, compat]
requires:
  - phase: 03-safe-surface-c-abi-shim-optional-families-03
    provides: Stable-only C ABI boundary and safe/raw integration contracts used by the shim.
provides:
  - Stable `#[repr(i32)]` C status taxonomy exports with explicit `CINTX_STATUS_*` constants.
  - Thread-local last-error code/message/api/family/representation copy-out APIs for C callers.
  - Panic-bounded `extern "C"` query/eval wrappers over compat raw APIs with fail-closed behavior.
  - Null-pointer guardrails for nonzero-length eval buffers to avoid silent success on invalid pointers.
affects: [cintx-capi, compat-ffi, phase-04-verification-release-automation]
tech-stack:
  added: []
  patterns: [stable-status-constants, tls-last-error-copyout, fail-closed-pointer-validation]
key-files:
  created: []
  modified:
    - crates/cintx-capi/src/errors.rs
    - crates/cintx-capi/src/lib.rs
    - crates/cintx-capi/src/shim.rs
key-decisions:
  - "Expose `CINTX_STATUS_*` constants beside `CintxStatus` so C callers can bind stable integer codes without enum layout assumptions."
  - "Treat `(ptr == NULL && len > 0)` for eval output/cache as `NullPointer` to preserve fail-closed semantics for invalid C call sites."
patterns-established:
  - "Shim path pattern: decode API -> catch_unwind boundary -> map typed status -> update TLS last-error report."
  - "Copy-out diagnostics pattern: return required NUL-terminated byte lengths even when output buffer is NULL."
requirements-completed: [COMP-04]
duration: 4m
completed: 2026-03-28
---

# Phase 03 Plan 04: Optional C ABI Shim Summary

**C ABI shim wrappers now expose stable status constants and stricter fail-closed pointer validation while preserving thread-local copy-out diagnostics over compat raw APIs.**

## Performance

- **Duration:** 4m
- **Started:** 2026-03-28T05:44:13Z
- **Completed:** 2026-03-28T05:47:51Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added explicit `CINTX_STATUS_*` integer constants for the complete C ABI status taxonomy and re-exported them from crate root.
- Extended status mapping coverage with regression tests to lock enum-to-integer code consistency.
- Hardened `cintrs_eval` to return `NullPointer` when output/cache pointers are null but nonzero lengths are supplied.
- Added shim regression tests for invalid API-id TLS reporting and null-pointer fail-closed behavior.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build C status taxonomy and thread-local last-error copy-out APIs** - `447ddf2` (feat)
2. **Task 2: Implement thin compat-style extern wrappers with fail-closed status + TLS reporting** - `8141f0e` (fix)

## Files Created/Modified
- `crates/cintx-capi/src/errors.rs` - Added exported status constants and status-code regression coverage.
- `crates/cintx-capi/src/lib.rs` - Re-exported status constants and aligned crate-level success constant with shared status taxonomy.
- `crates/cintx-capi/src/shim.rs` - Added null-pointer guards for nonzero-length eval buffers and new shim failure-path tests.

## Decisions Made
- Stabilize integer status access via named constants (`CINTX_STATUS_*`) in addition to enum variants.
- Reject null output/cache pointers paired with nonzero lengths at the C ABI boundary rather than allowing implicit `out=None` execution.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Prevented silent success on null eval pointers with nonzero lengths**
- **Found during:** Task 2
- **Issue:** `cintrs_eval` treated null `out`/`cache` pointers as optional slices even when lengths were nonzero, which could mask caller bugs.
- **Fix:** Added explicit `NullPointer` checks for `(out == NULL && out_len > 0)` and `(cache == NULL && cache_len > 0)` before compat dispatch.
- **Files modified:** `crates/cintx-capi/src/shim.rs`
- **Verification:** `cargo test -p cintx-capi --lib`
- **Committed in:** `8141f0e`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix tightened C ABI fail-closed semantics without expanding scope beyond the optional shim contract.

## Issues Encountered

None.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- C callers now have stable integer status constants, typed nonzero failures, and thread-local copy-out diagnostics.
- Shim wrappers enforce stricter invalid-pointer handling for eval failure paths, reducing integration ambiguity for downstream C consumers.

## Known Stubs

None.

## Self-Check: PASSED

- Found `.planning/phases/03-safe-surface-c-abi-shim-optional-families/04-PLAN-SUMMARY.md`.
- Found `crates/cintx-capi/src/errors.rs`, `crates/cintx-capi/src/lib.rs`, and `crates/cintx-capi/src/shim.rs`.
- Found task commits `447ddf2` and `8141f0e` in git history.
