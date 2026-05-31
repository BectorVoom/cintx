---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 06
subsystem: runtime
tags: [giao, chunk-staging, complex-interleaved, memory-limit, planner, wr-01, fnd-06]

# Dependency graph
requires:
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    provides: FND-03 complex-interleaved output layout (complex_output manifest flag → 2× staging)
  - phase: 25-group-2-hessian-higher-order-derivatives
    provides: FND-06 fail-closed full-block chunk staging precedent in eval_raw
provides:
  - "Full-block per-chunk staging in the safe-API evaluate path for monolithic complex/GIAO whole-block writers (mirrors eval_raw raw.rs:1061-1070)"
  - "Runtime test locking GIAO operability under memory-limit chunking (chunk_count > 1)"
affects: [giao-nmr, complex-output-families, memory-limit-chunking, safe-api-evaluate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "complex_interleaved families short-circuit staging_elements_for_chunk to the full block (monolithic whole-block writers are not chunk-partitionable)"

key-files:
  created: []
  modified:
    - crates/cintx-runtime/src/planner.rs

key-decisions:
  - "Chose the full-block staging approach (preferred per WR-01) over marking GIAO non-chunkable, preserving availability and matching eval_raw exactly"
  - "Branch on plan.output_layout.complex_interleaved in staging_elements_for_chunk rather than per-family special-casing — the layout flag is the single source of truth for whole-block complex writers"

patterns-established:
  - "Monolithic-writer chunk staging: any complex_interleaved family gets the FULL staging_elements per chunk; real families keep the sliced suffix-prefix partition"

requirements-completed: [GIAO-01, GIAO-02, FND-03]

# Metrics
duration: 12min
completed: 2026-05-31
---

# Phase 26 Plan 06: GIAO Chunk-Staging Gap Closure Summary

**Safe-API `evaluate` now allocates the FULL interleaved block per chunk for monolithic GIAO/complex writers, so a chunk-forcing `memory_limit_bytes` succeeds (or fails up front with `MemoryLimitExceeded`) instead of returning a per-chunk `BufferTooSmall` (WR-01).**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-31
- **Completed:** 2026-05-31
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Closed WR-01: GIAO / monolithic-complex families are once again operable under memory-limit chunking. `staging_elements_for_chunk` now returns the full `plan.output_layout.staging_elements` whenever `complex_interleaved` is set, mirroring the proven `eval_raw` precedent (raw.rs:1061-1070, FND-06).
- Real (non-complex) families keep the genuine chunk-partitioned `suffix - prefix` staging — the fix is scoped to whole-block complex writers only.
- Added `evaluate_giao_complex_family_survives_memory_chunking`, a runtime test driving `int1e_giao_irjxp_cart` through `evaluate` with `memory_limit_bytes = 192` (forcing `chunk_count > 1`) and asserting the call SUCCEEDS with the `memory_limit` fallback — locking the behavior against regression.
- No `if dst < staging.len()` partial-write guard introduced (project-memory non-negotiable).

## Task Commits

Each task was committed atomically:

1. **Task 1: Size evaluate's per-chunk staging to the full block (mirror eval_raw)** - `05ff68e` (fix)
2. **Task 2: Runtime test — GIAO family through evaluate with chunk_count > 1** - `111b64b` (test)

## Files Created/Modified
- `crates/cintx-runtime/src/planner.rs` - Added a `complex_interleaved` short-circuit branch to `staging_elements_for_chunk` (returns full `staging_elements` for monolithic complex writers) with a WR-01/FND-06 doc comment citing the eval_raw precedent; added the `evaluate_giao_complex_family_survives_memory_chunking` runtime test.

## Decisions Made
- **Full-block staging over non-chunkable marking:** WR-01 allowed either fixing the chunk staging OR marking complex families non-chunkable (yielding `MemoryLimitExceeded` up front). Chose the full-block approach because it preserves availability and matches `eval_raw` byte-for-byte; the upfront workspace check still fails closed with `MemoryLimitExceeded` if the full block cannot fit, so the OOM-safe contract is preserved with no partial writes.
- **Layout-flag branch, not per-family logic:** the `complex_interleaved` flag (set from the manifest `complex_output` column via `build_output_layout`) is the single source of truth, so the fix automatically covers every present and future complex/GIAO family without family enumeration.

## Deviations from Plan

None - plan executed exactly as written. Task 1 took the preferred full-block approach (not the non-chunkable fallback), so Task 2 asserts the SUCCESS path as specified.

## Issues Encountered
None. The GIAO cart family `int1e_giao_irjxp_cart` (arity 2, cart, `complex_output=true`) validates cleanly against the existing `sample_basis(Cart)` two-shell fixture, so no new test fixture was needed.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- WR-01 availability gap is closed and test-locked. The safe-API `evaluate` path and the raw `eval_raw` path now share identical full-block chunk-staging semantics for monolithic complex writers.
- No blockers. Remaining phase-26 gap-closure work (one_electron.rs in 26-04/05/08) does not touch planner.rs — no overlap.

## Threat Model Coverage
- **T-26-09 (DoS / availability):** mitigated — full-block staging per chunk (Task 1) restores GIAO operability under chunking; the runtime test (Task 2) locks the behavior.
- **T-26-10 (silent partial write):** held — no partial-write path introduced; monolithic writers still fail closed on `BufferTooSmall` if a full block cannot fit, and OOM yields `MemoryLimitExceeded` up front (no buffer touched).

## Self-Check: PASSED

- `crates/cintx-runtime/src/planner.rs` — FOUND
- `.planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-06-SUMMARY.md` — FOUND
- Commit `05ff68e` (Task 1) — FOUND
- Commit `111b64b` (Task 2) — FOUND
- Commit `9378041` (docs) — FOUND

---
*Phase: 26-group-5-spin-free-giao-nmr-integrals-complex*
*Completed: 2026-05-31*
