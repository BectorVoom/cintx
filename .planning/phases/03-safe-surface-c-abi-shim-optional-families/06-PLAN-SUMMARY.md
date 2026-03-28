---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 06
subsystem: api
tags: [rust, safe-facade, compat, optional-families, unsupported-api]
requires:
  - phase: 03-safe-surface-c-abi-shim-optional-families-02
    provides: Raw compat optional/source envelope gates and UnsupportedApi reason contracts.
  - phase: 03-safe-surface-c-abi-shim-optional-families-05
    provides: Typed safe query_workspace/evaluate scaffolding and facade error mapping.
provides:
  - Shared compat-policy preflight for safe facade evaluate flow.
  - Safe-facade UnsupportedApi parity with raw gates for with-f12, Validated4C1E, and source-only requests.
  - Regression tests proving the safe-to-compat key link behavior.
affects: [phase-03-verification, phase-04-verification-and-release-automation, safe-facade-policy-gates]
tech-stack:
  added: []
  patterns:
    - Safe facade delegates optional/source policy decisions to compat raw gate helpers.
    - Two-stage safe preflight (before and after ExecutionPlan construction) to preserve source-only reason text and extents-aware checks.
key-files:
  created: []
  modified:
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-rs/src/api.rs
    - crates/cintx-rs/Cargo.toml
    - Cargo.lock
key-decisions:
  - "Use cintx_compat::raw::enforce_safe_facade_policy_gate as the single UnsupportedApi policy source for safe evaluate preflight."
  - "Run a compat-policy preflight before ExecutionPlan::new and again after plan construction so source-only families fail with compat-origin text before planner dispatch-family rejection."
  - "Make cintx-rs depend directly on cintx-compat and cintx-ops so resolver metadata and shared policy gates are available in all safe-facade builds."
patterns-established:
  - "Facade policy checks should route through compat gate helpers instead of duplicating profile/source/optional logic in safe-executor code."
requirements-completed: [EXEC-01, OPT-01, OPT-02, OPT-03]
duration: 8 min
completed: 2026-03-28
---

# Phase 3 Plan 6: Safe Facade Compat Policy Link Summary

**Safe evaluate now enforces compat raw policy gates, so optional/source UnsupportedApi outcomes use shared raw reason text instead of facade-local checks.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-28T07:01:06Z
- **Completed:** 2026-03-28T07:09:20Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added a public compat raw helper (`enforce_safe_facade_policy_gate`) that applies profile/source-only gates plus with-f12 and Validated4C1E envelope validation using existing raw logic.
- Wired safe `SessionQuery::evaluate()` to call compat policy gates before and after `ExecutionPlan::new`, preserving typed contracts while ensuring compat-origin UnsupportedApi reasons.
- Added safe-facade regression tests for required reason phrases: `with-f12 sph envelope`, `outside Validated4C1E`, and `unstable-source-api`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Expose compat-raw safe-facade policy gate as a reusable public helper** - `b0c9acf` (feat)
2. **Task 2: Wire safe query/evaluate flow to compat-raw policy gate and remove orphaned link gap** - `c81f897` (feat)

**Plan metadata:** Pending final docs commit in this execution.

## Files Created/Modified

- `crates/cintx-compat/src/raw.rs` - Added shared safe-facade policy helper, refactored profile/source checks, and added gate-specific raw tests.
- `crates/cintx-rs/src/api.rs` - Routed safe evaluate through compat preflight gates and added phrase-level regression tests.
- `crates/cintx-rs/Cargo.toml` - Promoted compat dependency to direct safe-facade dependency and added direct ops dependency for resolver preflight.
- `Cargo.lock` - Updated lockfile edge for new direct `cintx-rs -> cintx-ops` dependency.

## Decisions Made

- Centralized safe UnsupportedApi policy derivation in `cintx_compat::raw` to close the verifier-reported key-link gap.
- Kept safe `query_workspace()`/`evaluate()` typed API shape unchanged while shifting optional/source policy enforcement to compat preflight.
- Treated pre-plan source-only rejection as correctness-critical to avoid leaking planner-internal `"unsupported dispatch family"` reasons.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Source-only families failed before post-plan policy gate could execute**
- **Found during:** Task 2
- **Issue:** `ExecutionPlan::new` rejected `unstable::source::*` families with planner dispatch-family text before the post-plan compat gate ran.
- **Fix:** Added an early compat preflight using resolver descriptor metadata before `ExecutionPlan::new`, then retained the post-plan gate for natural-extents-aware checks.
- **Files modified:** `crates/cintx-rs/src/api.rs`, `crates/cintx-rs/Cargo.toml`, `Cargo.lock`
- **Verification:** `cargo test -p cintx-rs --lib`, `cargo test -p cintx-rs --lib --features with-f12`, `cargo test -p cintx-rs --lib --features with-4c1e`
- **Committed in:** `c81f897`

**2. [Rule 3 - Blocking] Roadmap plan-progress command reported success without mutating Phase 3 row**
- **Found during:** State/roadmap update step
- **Issue:** `gsd-tools roadmap update-plan-progress "03"` returned `"updated": true` but left `.planning/ROADMAP.md` at `5/6`.
- **Fix:** Manually synchronized the Phase 3 checklist, plan count/list, and progress row to `6/6 Complete`.
- **Files modified:** `.planning/ROADMAP.md`
- **Verification:** Confirmed roadmap now includes checked `06-PLAN.md` and `| Phase 3 ... | 6/6 | Complete | 2026-03-28 |`.
- **Committed in:** Final docs metadata commit for this plan.

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** No scope creep; both fixes were required to preserve policy parity and planning-state consistency.

## Issues Encountered

- Safe source-only requests initially returned planner dispatch-family errors before compat preflight. Resolved by adding the early resolver-backed compat gate.
- `roadmap update-plan-progress` produced a no-op update response, so Phase 3 roadmap state was synchronized manually.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The `api.rs -> compat/raw.rs` key link is now concrete and test-covered.
- Optional/source UnsupportedApi phrasing is compat-policy-sourced across default and optional feature profiles.
- Phase 03 plan sequence is fully complete and ready for phase-level verification/next-phase planning.

## Self-Check: PASSED

- FOUND: `.planning/phases/03-safe-surface-c-abi-shim-optional-families/06-PLAN-SUMMARY.md`
- FOUND: `b0c9acf`
- FOUND: `c81f897`

---
*Phase: 03-safe-surface-c-abi-shim-optional-families*
*Completed: 2026-03-28*
