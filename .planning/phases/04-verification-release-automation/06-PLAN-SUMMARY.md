---
phase: 04-verification-release-automation
plan: 06
subsystem: infra
tags: [github-actions, gpu-bench, release-gates, artifacts]
requires:
  - phase: 04-verification-release-automation
    provides: "UAT diagnosis for Test 2 runner/artifact contract gap."
provides:
  - "Required and template GPU bench jobs run on explicit GPU-capable runner labels."
  - "Release and template workflows fail fast if bench/diagnostics artifacts are missing from required and fallback paths."
affects: [phase-04-verification, ci-release-gates, gpu-bench-template]
tech-stack:
  added: []
  patterns:
    - "Explicit GPU runner contract: [self-hosted, linux, x64, gpu]"
    - "Post-report artifact contract validation before upload-artifact"
key-files:
  created:
    - .planning/phases/04-verification-release-automation/06-PLAN-SUMMARY.md
  modified:
    - .github/workflows/compat-governance-release.yml
    - ci/gpu-bench.yml
key-decisions:
  - "Bind `gpu_bench_required` and `gpu_bench_template` to the explicit GPU runner label set."
  - "Enforce required/fallback artifact existence before upload via a dedicated validation step."
patterns-established:
  - "Required release GPU jobs remain blocking (`continue-on-error: false`) while enforcing evidence contracts."
requirements-completed: [VERI-02, VERI-03, VERI-04]
duration: 2 min
completed: 2026-03-29
---

# Phase 04 Plan 06: GPU Bench Runner + Artifact Contract Summary

**GPU-required release/template workflows are now explicitly GPU-bound and fail when benchmark or diagnostics evidence is missing from both required and fallback artifact paths.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-29T00:28:07Z
- **Completed:** 2026-03-29T00:30:43Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Bound `gpu_bench_required` and `gpu_bench_template` to `runs-on: [self-hosted, linux, x64, gpu]`.
- Preserved required release blocking semantics with `continue-on-error: false` on `gpu_bench_required`.
- Added `Validate bench artifact contract` checks to both workflows so runs fail if report/diagnostics files are missing from both `/mnt/data` and `/tmp/cintx_artifacts`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Bind required GPU bench jobs to a GPU-capable runner contract** - `928cc13` (fix)
2. **Task 2: Enforce required/fallback artifact contract before upload** - `b0eb565` (fix)

## Files Created/Modified

- `.github/workflows/compat-governance-release.yml` - Required release GPU bench gate now runs on GPU-capable runner labels and validates artifact contract before upload.
- `ci/gpu-bench.yml` - Reusable GPU bench template now runs on GPU-capable runner labels and validates artifact contract before upload.
- `.planning/phases/04-verification-release-automation/06-PLAN-SUMMARY.md` - Plan execution summary with task commits, decisions, and verification outcomes.

## Decisions Made

- Enforced a concrete GPU runner contract (`[self-hosted, linux, x64, gpu]`) for both required release and template GPU jobs to satisfy D-07/D-08 semantics.
- Added explicit artifact contract validation after `bench-report` to enforce D-12 before artifact upload.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 04 Test 2 gap is closed at workflow configuration level.
- Required GPU gate is explicitly GPU-bound and fail-fast on missing bench/diagnostics evidence.
- Ready for remote workflow rerun/UAT confirmation.

## Self-Check: PASSED

- Found `.planning/phases/04-verification-release-automation/06-PLAN-SUMMARY.md`.
- Found task commit `928cc13`.
- Found task commit `b0eb565`.

---
*Phase: 04-verification-release-automation*
*Completed: 2026-03-29*
