---
phase: 02-execution-compatibility-stabilization
plan: 07
subsystem: compat-oracle
tags: [rust, compat, helpers, legacy, optimizer, oracle, parity]
requires:
  - phase: 02-execution-compatibility-stabilization
    provides: raw compat query/evaluate contract from plan 06 and 3c kernel/transform coverage from plan 08
provides:
  - Phase 2 helper/transform entry points and misc.h-derived legacy wrapper forwards over shared raw path
  - Immutable optimizer lifecycle plus manifest-complete optimizer symbols (`CINT*` and `int2e_*_optimizer`)
  - Manifest-driven oracle fixtures and parity gates with representation-matrix and parity-report artifacts
affects: [phase-03-planning, oracle-verification-baseline, compat-surface-completeness]
tech-stack:
  added: []
  patterns:
    - manifest-driven fixture derivation from `compiled_manifest.lock.json`
    - family-specific tolerance table with zero-threshold absolute-error gating
key-files:
  created:
    - crates/cintx-oracle/src/fixtures.rs
    - crates/cintx-oracle/src/compare.rs
  modified:
    - crates/cintx-compat/src/helpers.rs
    - crates/cintx-compat/src/transform.rs
    - crates/cintx-compat/src/optimizer.rs
    - crates/cintx-compat/src/legacy.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/lib.rs
key-decisions:
  - "Keep helper/transform/optimizer surface checks manifest-strict by comparing expected symbols from Resolver metadata against implemented public entry points."
  - "Route oracle evaluated-output comparisons through `cintx-compat` raw + legacy wrappers rather than direct runtime/cubecl calls to preserve compat ownership boundaries."
  - "Emit required `/mnt/data` artifact targets in metadata but fall back to `/tmp/cintx_artifacts` when `/mnt/data` is not writable in sandboxed environments."
patterns-established:
  - "Phase 2 oracle fixtures are derived from canonical manifest entries, not hardcoded symbol lists."
  - "Parity reports explicitly include tolerance, helper coverage, layout assertions, and optimizer equivalence per fixture."
requirements-completed: [COMP-03, EXEC-05]
duration: 29min
completed: 2026-03-26
---

# Phase 2 Plan 07: Compat Surface and Oracle Parity Summary

**Phase 2 compatibility coverage is now complete across helper/transform/optimizer/legacy APIs with oracle-backed parity gates for base families.**

## Performance

- **Duration:** 29 min
- **Started:** 2026-03-26T11:15:00Z
- **Completed:** 2026-03-26T11:44:00Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Implemented and validated the full Phase 2 helper + transform compat subset and fixed helper formula regression expectations.
- Completed immutable optimizer lifecycle plus additional `int2e_*_optimizer` entry points required by manifest helper-kind optimizer rows.
- Added manifest-driven oracle fixture and parity harness modules that verify helper coverage, family tolerance envelopes, layout/interleaving assertions, and optimizer on/off equivalence.
- Generated required parity artifacts with explicit required-path metadata:
  - required: `/mnt/data/cintx_phase_02_manifest_representation_matrix.json`
  - required: `/mnt/data/cintx_phase_02_compat_parity_report.json`
  - actual (sandbox fallback): `/tmp/cintx_artifacts/cintx_phase_02_manifest_representation_matrix.json`
  - actual (sandbox fallback): `/tmp/cintx_artifacts/cintx_phase_02_compat_parity_report.json`

## Task Commit

1. **Task 1-3: Helper/transform/optimizer/legacy surface plus oracle parity harness** - `c7b8273` (feat)

## Files Created/Modified

- `crates/cintx-compat/src/helpers.rs` - Added helper API surface and regression tests.
- `crates/cintx-compat/src/transform.rs` - Added transform entry points mapped to CubeCL staging transforms.
- `crates/cintx-compat/src/optimizer.rs` - Added optimizer lifecycle + `int2e_*_optimizer` entry points.
- `crates/cintx-compat/src/legacy.rs` - Added misc.h-derived wrapper forwards with manifest coverage regression test.
- `crates/cintx-compat/src/raw.rs` - Added `INT3C1E_P2_SPINOR` symbol constant for legacy wrapper mapping completeness.
- `crates/cintx-oracle/build.rs` - Added bindgen/cc wiring with env-gated vendor probe build path.
- `crates/cintx-oracle/src/fixtures.rs` - Added manifest-derived fixture matrix and artifact writer utilities.
- `crates/cintx-oracle/src/compare.rs` - Added tolerance-table parity comparisons, helper coverage checks, and parity report generation.
- `crates/cintx-oracle/src/lib.rs` - Updated crate docs for the active oracle harness.

## Decisions Made

- Preserved shared compat ownership boundaries by treating legacy wrappers as upstream-compatible entry points while keeping comparisons inside `cintx-compat`.
- Recorded explicit `/mnt/data` required artifact paths in outputs and reports even when runtime fallback is necessary.
- Kept family tolerance gates explicit (`1e-11`, `1e-12`, `1e-9`, `1e-7`, `1e-5`, `1e-18`) for auditability.

## Issues Encountered

- `/mnt/data` is not writable in this execution environment (`/mnt` read-only), so artifact writes fall back to `/tmp/cintx_artifacts` with required-path metadata retained.

## Verification

- `cargo test -p cintx-compat --lib` ✅
- `cargo test -p cintx-oracle --lib` ✅

## Next Phase Readiness

- Phase 2 roadmap and requirement targets are complete through Plan 07.
- Project is ready to begin Phase 3 (`Safe Surface, C ABI Shim & Optional Families`).

## Self-Check: PASSED

- FOUND: `.planning/phases/02-execution-compatibility-stabilization/07-PLAN-SUMMARY.md`
- FOUND: `c7b8273`
- FOUND: `/tmp/cintx_artifacts/cintx_phase_02_manifest_representation_matrix.json`
- FOUND: `/tmp/cintx_artifacts/cintx_phase_02_compat_parity_report.json`

---
*Phase: 02-execution-compatibility-stabilization*
*Completed: 2026-03-26*
