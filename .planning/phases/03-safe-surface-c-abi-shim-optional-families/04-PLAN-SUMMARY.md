---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 04
subsystem: api
tags: [rust, c-abi, tls, compat, ffi, error-handling]
requires:
  - phase: 03-safe-surface-c-abi-shim-optional-families-01
    provides: Stable-only C ABI crate boundary and phase-3 safe-surface scaffolding.
  - phase: 03-safe-surface-c-abi-shim-optional-families-02
    provides: Optional-family/runtime-envelope compat behavior consumed by raw wrappers.
provides:
  - Stable integer C status taxonomy with `Success = 0` and typed nonzero failure codes.
  - Thread-local last-error reports with copy-out C APIs (`code`, `message`, `api`, `family`, `representation`, `clear`).
  - Panic-bounded `extern "C"` query/eval wrappers over `query_workspace_raw` and `eval_raw`.
  - Stable crate-root C ABI export surface for status/error + shim wrappers.
affects: [phase-03-plan-03, phase-04, cintx-capi, migration-interop, compat-ffi]
tech-stack:
  added: []
  patterns: [tls-last-error-state, fail-closed-c-shim-wrapper, panic-to-status-mapping]
key-files:
  created:
    - .planning/phases/03-safe-surface-c-abi-shim-optional-families/deferred-items.md
  modified:
    - crates/cintx-capi/src/errors.rs
    - crates/cintx-capi/src/shim.rs
    - crates/cintx-capi/src/lib.rs
key-decisions:
  - "Use a numeric C API id (`CintxRawApi`) that maps directly to `RawApiId` constants, avoiding dynamic symbol-lifetime problems in FFI."
  - "Return required byte count (including trailing NUL) for all copy-out last-error accessors so callers can preflight/resize safely."
  - "Keep C ABI stable-only by exporting shim/error symbols through crate root and explicitly pinning `CAPI_EXPOSES_UNSTABLE_SOURCE_API = false`."
patterns-established:
  - "C wrapper boundary pattern: validate pointers -> catch_unwind -> call compat raw API -> map typed status -> update TLS report."
  - "Fail-closed reporting pattern: write caller outputs only after compat call success; failures return status and TLS diagnostics only."
requirements-completed: [COMP-04]
duration: 10m
completed: 2026-03-28
---

# Phase 03 Plan 04: Optional C ABI Shim Summary

**C ABI query/eval wrappers now provide deterministic integer statuses and thread-local copy-out diagnostics over compat raw paths with panic-safe fail-closed behavior.**

## Performance

- **Duration:** 10m
- **Started:** 2026-03-28T00:27:08Z
- **Completed:** 2026-03-28T00:37:27Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Implemented `errors.rs` status taxonomy (`#[repr(i32)]`, `Success = 0`) plus TLS `LAST_ERROR` storage and copy-out C APIs.
- Implemented `shim.rs` thin `extern "C"` wrappers that call `query_workspace_raw` / `eval_raw` inside panic boundaries and set/clear TLS diagnostics consistently.
- Finalized `lib.rs` stable export surface by re-exporting shim/error symbols and asserting stable-only boundary behavior in tests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build C status taxonomy and thread-local last-error copy-out APIs** - `21b730a` (feat)
2. **Task 2: Implement thin compat-style extern wrappers with fail-closed status + TLS reporting** - `dfbabaf` (feat)

Additional corrective commit:

1. **Follow-up: finalize stable crate-root C ABI exports** - `90571c6` (fix)

## Files Created/Modified
- `crates/cintx-capi/src/errors.rs` - Added typed C status codes, TLS report model, error mapping, copy-out APIs, C exports, and unit tests.
- `crates/cintx-capi/src/shim.rs` - Added C API id mapping, panic-bounded query/eval wrappers, pointer validation, compat dispatch, and shim tests.
- `crates/cintx-capi/src/lib.rs` - Added crate-root stable re-exports/constants and regression test for stable-only boundary.
- `.planning/phases/03-safe-surface-c-abi-shim-optional-families/deferred-items.md` - Logged out-of-scope workspace issues discovered during execution.

## Decisions Made
- Keep wrapper ABI centered on integer API identifiers (`i32`) rather than dynamic C strings to preserve strict mapping to manifest-backed `RawApiId` constants.
- Expose copy-out diagnostics for all required report fields (`message`, `api`, `family`, `representation`) and make truncation detectable via required-length return values.
- Keep C ABI failure semantics fail-closed: wrapper code does not emit partial success writes, and all failures route through nonzero status + TLS report.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Filled missing stable crate-root export surface**
- **Found during:** Task 2 final verification
- **Issue:** `crates/cintx-capi/src/lib.rs` remained near-stub and did not reflect the stable C ABI export surface described by plan artifacts.
- **Fix:** Added stable re-exports for error/shim symbols, explicit success/status constants, and a crate-level stable-only regression test.
- **Files modified:** `crates/cintx-capi/src/lib.rs`
- **Verification:** `cargo test -p cintx-capi --lib`
- **Committed in:** `90571c6`

**2. [Rule 3 - Blocking] Scoped formatting to avoid unrelated workspace rustfmt failure**
- **Found during:** Task 1 verification
- **Issue:** `cargo fmt --all` failed because an unrelated workspace module file (`crates/cintx-rs/src/error.rs`) was missing while module resolution ran.
- **Fix:** Switched to scoped formatting for touched files (`rustfmt crates/cintx-capi/src/*.rs`) and continued plan verification.
- **Files modified:** None (workflow adjustment only)
- **Verification:** `cargo test -p cintx-capi --lib errors::tests:: -- --nocapture` and `cargo test -p cintx-capi --lib`
- **Committed in:** N/A (no source change)

---

**Total deviations:** 2 auto-handled (1 missing critical, 1 blocking)
**Impact on plan:** Deviations stayed within plan scope and were necessary to satisfy the required stable C ABI surface and complete verification reliably.

## Issues Encountered
- Concurrent parallel execution left unrelated files modified in the worktree; this plan staged only `crates/cintx-capi/src/{errors.rs,shim.rs,lib.rs}` and logged scope boundaries in `deferred-items.md`.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- C callers can now invoke shim query/eval entry points and read deterministic typed diagnostics via TLS copy-out APIs.
- Phase 3/4 consumers can build on the stable-only C ABI boundary without exposing unstable source-only symbols.

## Known Stubs

None.

---
*Phase: 03-safe-surface-c-abi-shim-optional-families*
*Completed: 2026-03-28*
