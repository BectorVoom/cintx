---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Phase 3 context gathered
last_updated: "2026-03-27T23:05:16.544Z"
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 9
  completed_plans: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-21)

**Core value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.  
**Current focus:** Phase 03 — safe-surface-c-abi-optional-families

## Current Position

Phase: 02 (execution-compatibility-stabilization) — COMPLETE
Plan: 7 of 7

## Performance Metrics

**Velocity:**

- Total plans completed: 3
- Average duration: 21 min
- Total execution time: 1.0 hours

**By Phase:**
| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | 27 min | 13.5 min |
| 02 | 7 | 107 min | 15.3 min |

**Recent Trend:**

- Last 5 plans: 7 min, 10 min, 26 min, 8 min, 29 min
- Trend: Improved after raw/oracle stabilization

| Phase 01-manifest-planner-foundation P01 | 18min | 2 tasks | 15 files |
| Phase 01-manifest-planner-foundation P02 | 9min | 2 tasks | 10 files |
| Phase 02-execution-compatibility-stabilization P02 | 18min | 2 tasks | 8 files |
| Phase 02 P03 | 9 min | 2 tasks | 6 files |
| Phase 02 P04 | 7 min | 2 tasks | 5 files |
| Phase 02 P05 | 10 min | 2 tasks | 9 files |
| Phase 02 P06 | 26 min | 3 tasks | 3 files |
| Phase 02 P08 | 8 min | 2 tasks | 8 files |
| Phase 02 P07 | 29 min | 3 tasks | 9 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md and summarized here for continuity.

- [Phase 01-manifest-planner-foundation]: Always derive the manifest arity from the family (1e/2c2e=2, 3c1e/3c2e=3, 2e/4c1e=4) to align with the documented dims contract.
- [Phase 01-manifest-planner-foundation]: Represent FeatureFlag, Stability, and HelperKind with Cow<'static, str> so generated metadata and runtime parsers can share 'static data without lifetime issues.
- [Phase 01-manifest-planner-foundation]: Keep the canonical lock in crates/cintx-ops/generated and implicitly validate the support matrix before emitting resolver tables.
- [Phase 01-manifest-planner-foundation]: Persist exact chunk layouts inside `WorkspaceQuery` and reject evaluate-time planning drift instead of silently replanning.
- [Phase 01-manifest-planner-foundation]: Clamp `chunk_size_override` to the maximum work units that fit inside the effective memory limit.
- [Phase 01-manifest-planner-foundation]: Surface bad shell atom references through `InvalidShellAtomIndex` instead of `ChunkPlanFailed`.
- [Phase 02-execution-compatibility-stabilization]: Keep Phase 2 workspace scope limited to core/ops/runtime/compat/cubecl/oracle and defer cintx-rs/cintx-capi membership.
- [Phase 02-execution-compatibility-stabilization]: Require explicit crate edges compat->cubecl and oracle->compat instead of implicit transitive wiring.
- [Phase 02-execution-compatibility-stabilization]: Resolve CubeCL kernels module ambiguity by pinning lib export to kernels/mod.rs during workspace activation.
- [Phase 02]: Treat helper/transform/optimizer-lifecycle and legacy-wrapper rows as first-class canonical manifest entries with explicit helper_kind/category metadata.
- [Phase 02]: Derive expected legacy wrappers from in-scope base symbols plus misc.h macro classification to fail on missing or extra wrapper rows.
- [Phase 02]: Expose resolver helper_kind filters and kind-aware symbol lookup so helper/legacy resolution stays manifest-driven.
- [Phase 02]: Keep the runtime execution contract backend-neutral and enforce OutputOwnership as BackendStagingOnly -> CompatFinalWrite at planner/dispatch boundaries.
- [Phase 02]: Route evaluate() through deterministic runtime scheduling and centralized run metrics (chunk_count, peak_workspace_bytes, transfer_bytes, not0) instead of backend-owned policy.
- [Phase 02-execution-compatibility-stabilization]: Pinned the initial executable CubeCL profile to CUBECL_RUNTIME_PROFILE=cpu and exposed a concrete constructor through CubeClExecutor::new.
- [Phase 02-execution-compatibility-stabilization]: Kept backend execution fail-closed to canonical 1e/2e/2c2e registry entries and returned UnsupportedApi for follow-on families.
- [Phase 02-execution-compatibility-stabilization]: Preserved planner output ownership as BackendStagingOnly -> CompatFinalWrite; transfer planning stages metadata/workspace/output buffers only.
- [Phase 02]: Use symbol-backed RawApiId resolved through Resolver — Keeps raw dispatch manifest-driven and avoids hardcoding operator ids in compat.
- [Phase 02]: Map RawOptimizerHandle workspace hints to runtime memory limits — Enables deterministic chunking and MemoryLimitExceeded validation without extending raw function signatures.
- [Phase 02]: Enable 3c1e/3c2e in kernel registry while keeping 4c1e unsupported — Completes Phase 2 base-family execution envelope without expanding unsupported scope.
- [Phase 02]: Extend compat optimizer coverage with `int2e_cart_optimizer`, `int2e_sph_optimizer`, and `int2e_optimizer` so helper-kind optimizer symbols remain manifest-complete.
- [Phase 02]: Drive parity fixtures from the canonical `compiled_manifest.lock.json` and emit representation matrices plus parity reports with `/mnt/data` required-path metadata.
- [Phase 02]: Verify family-specific tolerance envelopes and optimizer on/off equivalence through compat raw + legacy wrapper comparisons while asserting final flat-buffer and spinor interleaving contracts.

### Pending Todos

None yet.

### Blockers/Concerns

None currently.

## Session Continuity

Last session: 2026-03-27T23:05:16.542Z
Stopped at: Phase 3 context gathered
Resume file: .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md
