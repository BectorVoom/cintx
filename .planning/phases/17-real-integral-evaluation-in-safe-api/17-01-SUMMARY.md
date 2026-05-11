---
phase: 17-real-integral-evaluation-in-safe-api
plan: 01
subsystem: cargo
tags: [cargo, dependency, cintx-oracle, cintx-rs, wave-0]

# Dependency graph
requires:
  - phase: 17-real-integral-evaluation-in-safe-api (context)
    provides: D-01..D-12 decisions locking executor-swap approach and oracle test shape
provides:
  - cintx-rs path-dep edge in crates/cintx-oracle/Cargo.toml enabling safe-API parity tests to compile
affects: [17-02, 17-03, safe_api_arity2_parity.rs test compilation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "path-dep with default-features = false for intra-workspace safe-facade deps (mirrors cintx-rs/Cargo.toml line 12)"

key-files:
  created: []
  modified:
    - crates/cintx-oracle/Cargo.toml
    - Cargo.lock

key-decisions:
  - "Add cintx-rs path-dep with default-features = false; cpu/rocm backends reach cintx-cubecl transitively through existing cintx-compat/cpu chain without new feature entries"

patterns-established:
  - "Intra-workspace path-dep with default-features = false: keeps the safe-facade boundary lean under future cintx-rs feature additions"

requirements-completed: [RVAL-02]

# Metrics
duration: 8min
completed: 2026-05-11
---

# Phase 17 Plan 01: Add cintx-rs Cargo dep to cintx-oracle Summary

**cintx-rs intra-workspace path-dep added to cintx-oracle with `default-features = false`, enabling Plan 03 safe-API parity test file to compile against `SessionRequest::evaluate`**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-05-11T23:09:00Z
- **Completed:** 2026-05-11T23:17:49Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Added `cintx-rs = { path = "../cintx-rs", default-features = false }` to `[dependencies]` in `crates/cintx-oracle/Cargo.toml` in alphabetical order between `cintx-ops` and `serde_json`
- Updated `Cargo.lock` with the new intra-workspace dep edge (cintx-rs already a workspace member; no network fetch required)
- Verified all four build profiles: `cargo build -p cintx-oracle --locked`, `--features cpu --locked` both succeed
- Verified `[features]` table is byte-identical to the pre-edit version; `rocm` feature still only lists `cintx-compat/rocm`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add cintx-rs path-dep to cintx-oracle [Cargo.toml]** - `02c9563` (chore)

**Plan metadata:** (see final commit below)

## Files Created/Modified
- `crates/cintx-oracle/Cargo.toml` - Added `cintx-rs = { path = "../cintx-rs", default-features = false }` to [dependencies]
- `Cargo.lock` - Updated with new intra-workspace dep edge for cintx-oracle -> cintx-rs

## Decisions Made
- Used `default-features = false` mirroring the `cintx-compat` path-dep pattern in `crates/cintx-rs/Cargo.toml` line 12. The cpu/rocm backends are already reachable transitively through the existing `cpu = ["cintx-compat/cpu"]` chain; no new feature forwarding entry needed.
- Ran `cargo build -p cintx-oracle` without `--locked` first to update the lockfile, then confirmed `--locked` succeeds. This is expected per the plan's success criteria note: "No Cargo.lock entry-list churn beyond the one new cintx-rs graph edge."

## Deviations from Plan

None - plan executed exactly as written.

Note: The `--locked` verification required a two-step approach (first update lockfile, then verify with `--locked`) because the Cargo.lock entry for cintx-oracle's dependencies needed the new edge added. This is expected behavior for an intra-workspace dep addition; no deviation rule applies.

## Issues Encountered
- `cargo build -p cintx-oracle --locked` initially failed with "cannot update the lock file" since the lockfile entry for cintx-oracle did not yet include `cintx-rs`. Resolved by running `cargo build -p cintx-oracle` first (no extra flags) to update the lockfile, then confirming `--locked` succeeds. This is normal workflow for any new dep addition.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Wave 0 complete: the Cargo edge exists; Plan 02 (executor swap in cintx-rs/src/api.rs) and Plan 03 (safe_api_arity2_parity.rs new test file) can now proceed in parallel as Wave 1.
- Plans 02 and 03 both depend on this Cargo edge being merged; in worktree isolation, both agents will have this change available.

---
*Phase: 17-real-integral-evaluation-in-safe-api*
*Completed: 2026-05-11*

## Self-Check: PASSED
- `crates/cintx-oracle/Cargo.toml`: FOUND (modified with cintx-rs dep)
- `Cargo.lock`: FOUND (updated)
- Commit `02c9563`: FOUND
- `cargo build -p cintx-oracle --locked`: exits 0
- `cargo build -p cintx-oracle --features cpu --locked`: exits 0
- `[features]` table: byte-identical to pre-edit (diff produces no output)
- `grep -c '^cintx-'` in Cargo.toml: 4 (cintx-compat, cintx-core, cintx-ops, cintx-rs)
