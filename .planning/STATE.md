---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: "Milestone: Full libcint 6.1.3 Family Parity"
status: executing
stopped_at: Phase 30 Wave 1 re-planned (30-01a/b/c/d) — ready to execute
last_updated: "2026-06-01T00:00:00.000Z"
last_activity: 2026-06-01
progress:
  total_phases: 21
  completed_phases: 19
  total_plans: 104
  completed_plans: 102
  percent: 98
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-05)

**Core value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.  
**Current focus:** Phase 30 — group-5-giao-slice-spin-giao-integrals-spinor

## Current Position

Phase: 30 (group-5-giao-slice-spin-giao-integrals-spinor) — EXECUTING (Wave 1 re-planned, ready to execute)
Plan: 30-00 complete; Wave 1 split into 30-01a/b/c/d (sequential a→b→c→d, each its own vendor gate); 30-02 not started
Status: Ready to execute — next is 30-01a (NEW 8-G-tensor London overlap engine: int1e_spgsp)
Seed/design: .planning/notes/phase-30-wave1-engine-class-split-PLAN.md (engine-class sub-wave breakdown)

Phase 30 Wave 1 re-plan COMPLETE (2026-06-01):

  - Monolithic 30-01-PLAN.md removed; replaced by 4 engine-class sub-wave plans, all wave:1, depends
    a→b→c→d (all on 00), GIAO-03 in each. Plan-checker: 4 Wave-1 plans internally sound; 2 blockers
    were dangling 30-02 handoff refs (depends_on:[01], 30-01-SUMMARY.md) — both re-pointed to 01d.
    30-01a spgsp (rank3); 30-01b cg/giao_sa10nucsp (rank3); 30-01c cg/giao_sa10sa01 (rank9, REAL
    c2s_si_1e); 30-01d spgnucsp+spgsa01 → closes full 9-family 1e gate. Reuses 3b68ff1 scaffolding.

Phase 30 Wave 1 pause (2026-06-01):

  - 30-00 COMPLETE: gauge x1i-with-origin fold + int1e_cg_sa10sp rank-3 gout variant in sigma_p.rs,
    combined gauge∧kappa spinor fixture, giao_sigma_1e_parity micro-test (byte-identity + cg→giao
    collapse). De-risk gate green.

  - 30-01 PAUSED at decision checkpoint (commit 3b68ff1 = safe registration scaffolding only).
    Executor verified vs libcint source that Wave 1 is NOT a transcription: only cg_sa10sp/giao_sa10sp
    (2 of 9) are proven; the other 7 need ~6 net-new device kernels (Rys+gauge + 8-G London engine
    classes, rank-9 36-comp gouts). All 9 manifest rows registered oracle_covered=false; bindgen
    allowlist + 7 vendor shims added; compiles. STATE/ROADMAP NOT advanced for 30-01.

  - DECISION (user, 2026-06-01): re-plan Wave 1 into engine-class sub-waves (30-01a overlap spgsp;
    30-01b Rys-gauge sp/nucsp; 30-01c Rys-gauge sa01 rank-9; 30-01d spg-Rys/London), each with its
    own vendor gate. See .planning/notes/phase-30-wave1-engine-class-split-PLAN.md. Reuse 3b68ff1
    scaffolding as-is. GIAO-03 still closes at end of Wave 2.

Phase 27 outcome (FND-04 / Gap B1, sf-derivative spinor transform):

  - 27-05 flipped oracle_covered=true for the 20 vendor-backed sf-derivative spinor families
    (18 arity-2 1e sf_2d ranks 3/9/27/81 + int3c2e_ip1/ip2 sf_3c2e rank-3); component_rank
    verified against the rank-tier table before each flip (no rank edits).

  - 4 D-12 vendor-stub arms (int2c2e_ip1/ip2_spinor + int3c1e_ip1/iprinv_spinor) stay
    oracle_covered=false — libcint 6.1.3 ships them as return-0 / exit(1) stubs (no byte-identity
    reference). 6 D-03 arity-4 int2e_ip* + 1 D-04 int1e_ecp_iprinv_spinor also stay false.

  - D-10 no-silent-skip assertion completed (test_no_silent_skip reads MANIFEST_ENTRIES at runtime,
    asserts the FLIPPED=true / DEFERRED=false split); manifest-audit green; full vendor parity suite
    green under both gate flags (6 passed, 0 failed, 3 ignored). No capi/legacy surface added.
Deferred follow-up: finite-difference verification of the 4 D-12 vendor-stub arms (FD of cintx scalar
  int2c2e_spinor / int3c1e_spinor), then flip under an FD-tolerance gate. D-03 needs an sf_4d derivative
  wrapper; D-04 belongs to the relativistic/ECP-spinor track.
Last activity: 2026-05-31

**v1.4 phase sequence (dependency-ordered):**
22 Gap A (FND-01) → 23 Group 1 1st-deriv (DRV1) → 24 Group 3 moments (MOM) →
25 Group 2 Hessian (HESS, FND-02 Wheeler, FND-06 fail-closed) →
26 Group 5 spin-free GIAO (GIAO-01/02, FND-03 complex) →
27 Gap B1 spinor-derivative (FND-04) → 28 Gap B2 c2s_si + σ·p (FND-05) →
29 Group 4 relativistic σ (REL) → 30 Group 5 GIAO×σ (GIAO-03) →
31 Group 6 gauge/Breit-Gaunt + full-parity (BREIT, PARITY-01).

Phases 23 and 24 can run in parallel after 22; phase 27 can parallel 26.

## Performance Metrics

**Velocity:**

- Total plans completed: 60
- Average duration: 15.6 min
- Total execution time: 1.3 hours

**By Phase:**
| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | 27 min | 13.5 min |
| 02 | 7 | 107 min | 15.3 min |
| 19 | 3 | 28 min | 9.3 min |
| 20 | 11 | - | - |
| 21 | 8 | - | - |
| 22 | 2 | - | - |
| 23 | 5 | - | - |
| 24 | 5 | - | - |
| 26 | 8 | - | - |
| 27 | 6 | - | - |
| 28 | 4 | - | - |
| 29 | 6 | - | - |

**Recent Trend:**

- Last 5 plans: 8 min, 29 min, 13 min, 10 min, 5 min
- Trend: Faster; Phase 19 Plan 03 typed-surface work executed in 5 min because Plan 01 had already scaffolded the placeholder files (ecp.rs stub, manifest rows at fixed OperatorIds) and the pattern was a 1:1 lift from Phase 18 D-04 UnsupportedAoSymmetry.

| Phase 01-manifest-planner-foundation P01 | 18min | 2 tasks | 15 files |
| Phase 01-manifest-planner-foundation P02 | 9min | 2 tasks | 10 files |
| Phase 02-execution-compatibility-stabilization P02 | 18min | 2 tasks | 8 files |
| Phase 02 P03 | 9 min | 2 tasks | 6 files |
| Phase 02 P04 | 7 min | 2 tasks | 5 files |
| Phase 02 P05 | 10 min | 2 tasks | 9 files |
| Phase 02 P06 | 26 min | 3 tasks | 3 files |
| Phase 02 P08 | 8 min | 2 tasks | 8 files |
| Phase 02 P07 | 29 min | 3 tasks | 9 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P01 | 3 min | 2 tasks | 9 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P02 | 11m | 2 tasks | 11 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P03 | 14 min | 2 tasks | 3 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P04 | 10m | 2 tasks | 3 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P01 | 4 min | 2 tasks | 3 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P02 | 62m | 2 tasks | 1 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P04 | 4m | 2 tasks | 3 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P03 | 9 min | 2 tasks | 1 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P05 | 34 min | 2 tasks | 4 files |
| Phase 03-safe-surface-c-abi-shim-optional-families P06 | 8 min | 2 tasks | 4 files |
| Phase 04-verification-release-automation P01 | 9 min | 2 tasks | 3 files |
| Phase 04-verification-release-automation P02 | 21m | 3 tasks | 6 files |
| Phase 04-verification-release-automation P03 | 2m | 2 tasks | 3 files |
| Phase 04-verification-release-automation P04 | 17min | 3 tasks | 11 files |
| Phase 04-verification-release-automation P05 | 2min | 1 tasks | 1 files |
| Phase 04-verification-release-automation P06 | 2 min | 2 tasks | 2 files |
| Phase 04-verification-release-automation P07 | 3 min | 2 tasks | 1 files |
| Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend P01 | 3 | 2 tasks | 5 files |
| Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend P02 | 7 | 2 tasks | 4 files |
| Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend P03 | 29 | 2 tasks | 3 files |
| Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend P04 | 25 | 2 tasks | 4 files |
| Phase 06-fix-raw-eval-staging-and-capability-fingerprint P01 | 8 | 2 tasks | 2 files |
| Phase 06-fix-raw-eval-staging-and-capability-fingerprint P02 | 4 | 1 tasks | 1 files |
| Phase 08-gaussian-primitive-infrastructure-and-boys-function P01 | 8 | 2 tasks | 9 files |
| Phase 08 P03 | 4 | 2 tasks | 3 files |
| Phase 08 P04 | 8 | 2 tasks | 3 files |
| Phase 09 P02 | 573 | 1 tasks | 2 files |
| Phase 09-1e-real-kernel-and-cart-to-sph-transform P03 | 25 | 3 tasks | 4 files |
| Phase 09-1e-real-kernel-and-cart-to-sph-transform P04 | 180 | 2 tasks | 7 files |
| Phase 09-1e-real-kernel-and-cart-to-sph-transform P05 | 1 | 1 tasks | 2 files |
| Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure P01 | 12 | 2 tasks | 4 files |
| Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure P03 | 12 | 2 tasks | 5 files |
| Phase 10 P02 | 196 | 2 tasks | 4 files |
| Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure P04 | 8m | 2 tasks | 2 files |
| Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure P06 | 8 | 1 tasks | 2 files |
| Phase 12 P01 | 11 | 2 tasks | 4 files |
| Phase 12 P02 | 5 | 2 tasks | 5 files |
| Phase 12-real-spinor-transform-c2spinor-replacement P05 | 90 | 1 tasks | 3 files |
| Phase 13-f12-stg-yp-kernels P01 | 15 | 2 tasks | 12 files |
| Phase 13-f12-stg-yp-kernels P02 | 90 | 2 tasks | 6 files |
| Phase 13-f12-stg-yp-kernels P03 | 45 | 2 tasks | 9 files |
| Phase 13-f12-stg-yp-kernels P04 | 90 | 2 tasks | 6 files |
| Phase 14-unstable-source-api-families P01 | 16 | 2 tasks | 13 files |
| Phase 14-unstable-source-api-families P05 | 5 | 2 tasks | 6 files |
| Phase 15-oracle-tolerance-unification-manifest-lock-closure P01 | 8 | 2 tasks | 4 files |
| Phase 15 P02 | 7 | 2 tasks | 4 files |
| Phase 15-oracle-tolerance-unification-manifest-lock-closure P03 | 2 | 2 tasks | 2 files |
| Phase 19-int1e-ecp-type1-type2-evaluator P01 | 13 min | 3 tasks | 20 files |
| Phase 19-int1e-ecp-type1-type2-evaluator P02 | 10 min | 2 tasks | 2 files |
| Phase 19-int1e-ecp-type1-type2-evaluator P03 | 5 min | 3 tasks | 8 files |
| Phase 20 P01 | 5 | 3 tasks | 5 files |
| Phase 20 P10 | 7 | 4 tasks | 11 files |
| Phase 20-precision-generic-f64-f32-switch P11 | 15 | 2 tasks | 3 files |
| Phase 21 P02 | 15 | 2 tasks | 5 files |
| Phase 21 P03 | 25 min | 2 tasks | 4 files |
| Phase 24 P24-01 | 18 | 3 tasks | 8 files |
| Phase 24 P24-02 | 70min | 2 tasks | 9 files |
| Phase 24 P24-03 | 28 | 1 tasks | 5 files |
| Phase 24 P24-04 | 38 | 1 tasks | 5 files |
| Phase 24 P24-05 | 22 | 1 tasks | 4 files |
| Phase 25 P02 | 24 | 3 tasks | 7 files |
| Phase 25 P03 | 70 | 3 tasks | 6 files |
| Phase 25 P04 | 55 | 3 tasks | 7 files |
| Phase 25 P05 | 10 | 3 tasks | 7 files |
| Phase 26 P01 | 12 | 3 tasks | 9 files |
| Phase 26 P02 | 95min | 3 tasks | 9 files |
| Phase 26 P03 | 70 | 3 tasks | 8 files |
| Phase 28 P01 | 12 | 2 tasks | 1 files |
| Phase 28 P02 | 9 | 1 tasks | 2 files |
| Phase 28 P03 | 38 | 2 tasks | 8 files |
| Phase 28 P04 | 42min | 2 tasks | 3 files |
| Phase 29 P02 | 100min | 3 tasks | 8 files |
| Phase 29 P03 | 35min | 3 tasks | 2 files |
| Phase 29 P04 | 55min | 3 tasks | 6 files |
| Phase 29 P05 | 7 | 3 tasks | 6 files |
| Phase 29 P06 | 95min | 3 tasks | 7 files |
| Phase 29 P6 | 95min | 3 tasks | 7 files |
| Phase 30 P00 | 35 | 3 tasks | 4 files |

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
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Top-level with-f12/with-4c1e gates explicitly map to libcint with_f12/with_4c1e to prevent feature-profile drift.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: cintx-rs unstable source APIs are exposed only via cfg(feature = "unstable-source-api") namespace to preserve stable defaults.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: cintx-capi remains a stable-only export boundary in plan 01 with no unstable-source C exports.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Treat optional-family availability as manifest-profile plus runtime-envelope dual gates.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Keep source-only rows manifest-visible but reject them unless unstable-source-api is enabled.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Allow runtime dispatch family 4c1e so validated with-4c1e calls can execute through the shared planner path.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Treat lockfile drift in Phase 3 wiring as correctness debt and regenerate immediately.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Keep unstable promotion policy encoded in source docs at both safe and C ABI boundaries.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Keep Task 2 as verification-only because optional/unstable runtime gates already satisfied plan contracts in this branch state.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Add explicit resolver MissingSymbol checks for F12/STG/YP cart and spinor symbols to harden sph-only manifest envelope enforcement.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Expose CINTX_STATUS_* constants so C callers can bind stable integer codes independent of Rust enum layout.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Treat (ptr == NULL && len > 0) in cintrs_eval as NullPointer to keep C ABI fail-closed semantics explicit.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Expose accessor methods on WorkspaceExecutionToken to keep contract metadata stable without exposing private fields.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Capture owned safe output directly from runtime backend staging via RecordingExecutor instead of rebuilding buffers after evaluate.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Record safe/capi feature-forwarding and stability contracts in package.metadata.cintx for manifest-level audits.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Use SessionBuilder::from_request with typed composition helpers to rebuild requests immutably while preserving query/evaluate invariants.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Keep prelude unstable exports behind cfg(feature = unstable-source-api) while expanding grouped stable re-exports.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Use cintx_compat::raw::enforce_safe_facade_policy_gate as the single UnsupportedApi policy source for safe evaluate preflight.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Run a compat-policy preflight before ExecutionPlan::new and again after plan construction so source-only families fail with compat-origin text before planner dispatch-family rejection.
- [Phase 03-safe-surface-c-abi-shim-optional-families]: Make cintx-rs depend directly on cintx-compat and cintx-ops so resolver metadata and shared policy gates are available in all safe-facade builds.
- [Phase 04-verification-release-automation]: Promote oracle fixture generation to profile-scoped APIs backed by compiled-manifest lock profile/stability metadata.
- [Phase 04-verification-release-automation]: Aggregate parity mismatches across the full fixture matrix, persist report artifacts first, then fail with mismatch_count.
- [Phase 04-verification-release-automation]: Default merge-blocking parity mode keeps include_unstable_source=false, requiring explicit opt-in for unstable-source coverage.
- [Phase 04-verification-release-automation]: Keep xtask verification gates fail-closed with non-zero exits on drift/parity/OOM regressions.
- [Phase 04-verification-release-automation]: Scope manifest lock diffing to oracle operator/source symbols to avoid helper/legacy false positives.
- [Phase 04-verification-release-automation]: Persist profile-specific oracle artifacts for each approved profile even when a profile fails parity.
- [Phase 04-verification-release-automation]: Keep required PR verification as four explicit jobs: manifest_drift_gate, oracle_parity_gate, helper_legacy_parity_gate, and oom_contract_gate.
- [Phase 04-verification-release-automation]: Resolve Rust channel from rust-toolchain.toml in each required job to avoid toolchain drift.
- [Phase 04-verification-release-automation]: Exercise helper/legacy and OOM gates across base,with-f12,with-4c1e,with-f12+with-4c1e profiles through deterministic loop execution.
- [Phase 04-verification-release-automation]: Bench regressions fail only when configured thresholds are exceeded.
- [Phase 04-verification-release-automation]: Bench and runtime diagnostics artifacts must target /mnt/data with CINTX_ARTIFACT_DIR fallback metadata.
- [Phase 04-verification-release-automation]: PR GPU/bench jobs stay advisory while release/scheduled/merge-queue jobs are required via explicit continue-on-error policy.
- [Phase 04-verification-release-automation]: Keep pub mod compare/fixtures intact while exporting profile-aware fixture/parity APIs explicitly from crate root.
- [Phase 04-verification-release-automation]: Preserve compile-edge export smoke coverage while expanding crate-root re-exports for Phase 4 gate consumers.
- [Phase 04-verification-release-automation]: Bound gpu_bench_required and gpu_bench_template to [self-hosted, linux, x64, gpu] to enforce the required GPU runner contract.
- [Phase 04-verification-release-automation]: Added Validate bench artifact contract checks so bench report and runtime diagnostics must exist in /mnt/data or /tmp/cintx_artifacts before artifact upload.
- [Phase 04-verification-release-automation]: Centralize required and fallback artifact paths in workflow-level env variables to reduce silent drift risk.
- [Phase 04-verification-release-automation]: Add a dedicated release policy invariant step that inspects committed workflow markers and fails closed.
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: BackendIntent defaults to BackendKind::Wgpu with selector 'auto' per D-03; Cpu variant kept for oracle/test use only
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: planning_matches() compares all four contract fields (memory, chunk_size, backend_intent, capability_token) so any backend policy drift fails evaluate closed (D-08)
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: BackendCapabilityToken fingerprint defaults to 0; later plans will populate with real wgpu adapter capability hash during device selection
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Use FNV-1a 64-bit hash over sorted feature/limit lists plus adapter identity fields for reproducible capability fingerprints
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Wrap cubecl init_setup with std::panic::catch_unwind to convert CubeCL panic-based adapter failures into typed UnsupportedApi errors
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Keep selector format simple (auto/discrete:N/integrated:N) aligned with CubeCL WgpuDevice enum variants
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Gate ensure_validated_4c1e and validated_4c1e_error under cfg(feature = with-4c1e) to eliminate dead_code warnings in default builds
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: kernels::resolve_family now returns UnsupportedApi with unsupported_representation:<repr> instead of UnsupportedRepresentation struct to keep D-12 taxonomy consistent across executor and kernels
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Transfer adapter label sourced from backend_intent.selector rather than static runtime_profile string per D-04 reproducibility
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Add cintx-cubecl as direct dep in cintx-rs so safe facade imports CubeClExecutor without indirection
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: WorkspaceExecutionToken clones backend_intent and backend_capability_token at query time for drift detection at evaluate time
- [Phase 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend]: Tests for eval/evaluate paths accept wgpu-capability fail-closed errors so CI passes without GPU
- [Phase 06-fix-raw-eval-staging-and-capability-fingerprint]: Scope RecordingExecutor locally in raw.rs rather than sharing — avoids coupling cintx-compat internals to cintx-rs internal pattern
- [Phase 06-fix-raw-eval-staging-and-capability-fingerprint]: execution_options_from_opt returns Result<ExecutionOptions, cintxRsError> so wgpu bootstrap failures propagate cleanly to all callers
- [Phase 06-fix-raw-eval-staging-and-capability-fingerprint]: Bootstrap-before-query pattern: always call bootstrap_wgpu_runtime before runtime_query_workspace to ensure planning_matches has a real fingerprint anchor
- [Phase 06-fix-raw-eval-staging-and-capability-fingerprint]: Assert bytes_written > 0 for staging path tests — query.bytes is workspace size not output size; bytes_written is output elements × sizeof(f64)
- [Phase 06-fix-raw-eval-staging-and-capability-fingerprint]: Use INT3C1E_P2_SPH and INT3C2E_IP1_SPH as 3c1e/3c2e regression test family representatives
- [Phase 08-gaussian-primitive-infrastructure-and-boys-function]: Pass TURNOVER_POINT[m] as scalar parameter to #[cube] boys_gamma_inc to avoid runtime const array indexing ambiguity in CubeCL 0.9.x
- [Phase 08-gaussian-primitive-infrastructure-and-boys-function]: Use as usize cast pattern for Array<f64> indexing in #[cube]: u32 loop counters with as usize at index sites — established pattern for all Phase 8+ math modules
- [Phase 08-gaussian-primitive-infrastructure-and-boys-function]: Host wrapper + #[cube] pair pattern: every math function has *_host() counterpart callable from tests without GPU context
- [Phase 08]: vrr_step guards nmax>=1 to avoid s-shell no-op array writes, mirrors g1e.c early return pattern
- [Phase 08]: Integration tests use host-side wrappers only (not CubeCL CPU backend launch) to avoid cond_br MLIR limitation discovered in Plan 02
- [Phase 08]: Add rys_root1_host as a pure-Rust host wrapper replicating #[cube] rys_root1 branching logic exactly
- [Phase 08]: Wire Rys-Boys weight-sum identity crosscheck at large/moderate/small x domains with appropriate tolerances
- [Phase 09-1e-real-kernel-and-cart-to-sph-transform]: Applied -0.5 (not +0.5) factor in kinetic contraction: D_j^2 of Gaussian is negative, so -0.5*D_j^2 yields positive kinetic energy
- [Phase 09-1e-real-kernel-and-cart-to-sph-transform]: Used vrr_2e_step_host for nuclear attraction VRR (root-dependent c00/b10), not vrr_step_host (which uses fixed center displacement)
- [Phase 09-1e-real-kernel-and-cart-to-sph-transform]: Use idempotency check (two eval_raw calls) as oracle parity method since upstream libcint is not compiled by default
- [Phase 09-1e-real-kernel-and-cart-to-sph-transform]: Kinetic G-tensor derivative acts on bra VRR i-index (ix+2) not HRR j-level (jx+2); nmax=li+lj+2 provides the needed VRR headroom
- [Phase 09-1e-real-kernel-and-cart-to-sph-transform]: Commit oracle parity artifact to repository artifacts/ directory since /mnt/data is unavailable in this environment
- [Phase 09-04]: Kinetic D_j^2 derivative steps ±2 j-levels in ket direction; formula jx*(jx-1)*g0[jx-2] - 2*aj*(2*jx+1)*g0[jx] + 4*aj^2*g0[jx+2] requires HRR to lj+2 and nmax=li+lj+2
- [Phase 09-04]: C2S_L1 is identity matrix (px/py/pz order); CINTcommon_fac_sp normalization for s/p applied separately in primitive loop, not in transform coefficients
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: Keep weight-sum identity tests at large x (asymptotic regime) where sum(w_i)==sqrt(PIE4/x) exactly; polynomial-fit branches do not satisfy this identity
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: Use supplemental bindgen header to declare int2c2e_sph/int3c1e_sph/int3c2e_sph which are in .c files but not in cint_funcs.h
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: Added int3c1e_sph/int3c1e_cart to manifest so eval_raw dispatches 3c1e overlap through launch_center_3c1e
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: g_alloc uses (dli*dlj*dlk).max(dli*vrr_nmax) matching libcint MAX formula — parentheses required for Rust operator precedence
- [Phase 10-02]: env user data MUST start at PTR_ENV_START=20 — PTR_RANGE_OMEGA=env[8] is read by all 2e+ libcint integrals; placing H2 z-coord there caused range-separated Coulomb to activate
- [Phase 10-02]: 2c2e kernel algorithm is correct — common_factor includes fac_sp per g2c2e.c; parity failures must be checked for env layout before kernel correctness
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: Use PTR_ENV_START-aligned env layout in int2e oracle tests to preserve libcint global env semantics.
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: Plan 10-04 canonicalizes 3c2e ij evaluation to li>=lj and transposes back to preserve caller shell order while matching ibase behavior.
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: 3c2e oracle fixtures now reserve libcint env global slots with PTR_ENV_START for correct 2e-family reference behavior.
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: Use shells (3,4,0)=H1-1s/H2-1s/O-1s for 3c1e/3c2e gate triples — same-center s-s-p is physically zero by angular symmetry
- [Phase 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure]: UAT item 2 tests eval_raw kernel path (not0>0 = C ABI status==0) since cintx-capi is not directly testable from cintx-oracle integration tests
- [Phase 12]: CG coefficient extraction from libcint g_trans_cart2jR/g_trans_cart2jI at g_c2s[] documented offsets for l=0..4, verified via Python parsing
- [Phase 12]: Four separate c2spinor code paths (sf, iket_sf, si, iket_si) per D-03; kappa<0=GT, kappa>0=LT, kappa==0=both; iket=multiply by i: (re,im)->(-im,re)
- [Phase 12]: cart_to_spinor_interleaved_staging kept as no-op (not deleted) for staging API compatibility; executor l/kappa wiring gap documented as TODO for Phase 12-02/03
- [Phase 12]: 2D c2spinor sf transform bra step uses conjugate convention (saI += -caI * v1) matching libcint a_bra_cart2spinor_sf; ket step uses complex multiply matching a_ket_cart2spinor
- [Phase 12]: Spinor buffer layout: column-major interleaved (j_spinor outer, i_spinor inner), staging[(j*di+i)*2] = re, +1 = im — matches libcint zcopy_ij
- [Phase 12]: kappa=0 spinor ordering: LT block first then GT, matching libcint implicit memory layout convention
- [Phase 12]: executor.rs skips apply_representation_transform for Spinor representation: kernel launchers own spinor transforms per Plan 04 design
- [Phase 13-f12-stg-yp-kernels]: Use include_bytes! with AlignedBytes wrapper for roots_xw.dat binary tables: include! macro rejects comma-separated expressions, binary + bytemuck is correct approach for 1.7M-element f64 tables
- [Phase 13-f12-stg-yp-kernels]: OperatorEnvParams defaults to all-None in ExecutionPlan::new(); callers (raw compat, safe API) populate f12_zeta from env[9]
- [Phase 13-f12-stg-yp-kernels]: manifest canonical_family for STG/YP is '2e' not 'f12'; F12 detection uses symbol prefix (int2e_stg/int2e_yp)
- [Phase 13-f12-stg-yp-kernels]: launch_f12 passes 'f12' explicitly to validate_f12_env_params; operator_name strips 'stg'/'yp' prefix for variant suffix
- [Phase 13-f12-stg-yp-kernels]: F12 canonical_family changed from '2e' to 'f12' in manifest so kernel dispatch routes to f12::launch_f12 via resolve_family_name in kernels/mod.rs
- [Phase 13-f12-stg-yp-kernels]: F12 base variant oracle parity at atol=1e-12 confirmed vs libcint 6.1.3; derivative variants use idempotency tests due to unimplemented multi-component sph transform
- [Phase 13-f12-stg-yp-kernels]: nabla1i_2e uses ceil angular momenta for G tensor headroom; base used for gout loops and sph transforms
- [Phase 13-f12-stg-yp-kernels]: gout_ipip1 applies column-major transposition matching libcint autocode; ipvip1/ip1ip2 do not
- [Phase 13-f12-stg-yp-kernels]: grad2.c and hess.c required in oracle build.rs — cint2e_f12.c only declares extern forward references
- [Phase 14-unstable-source-api-families]: Convert unresolved_families() from static &[&str] to Vec<&str> to support dynamic 3-feature combination without 8 cfg variants
- [Phase 14-unstable-source-api-families]: Use compact single-line ManifestEntry format for Phase 14 entries to avoid another very large Edit operation
- [Phase 14-unstable-source-api-families]: Grids FFI wrappers use [i32; 4] shls to match libcint cint1e_grids.c signature (i, j, grid_start, grid_end)
- [Phase 14-unstable-source-api-families]: unstable-source profile runs standalone, never combined with standard profiles
- [Phase 14-unstable-source-api-families]: Nightly CI is advisory-only, not a merge blocker
- [Phase 15-oracle-tolerance-unification-manifest-lock-closure]: tolerance_for_family drops Result wrapper: catch-all arm with Box::leak ensures any family gets unified atol=1e-12 without bail
- [Phase 15-oracle-tolerance-unification-manifest-lock-closure]: manifest_oracle_families() reads compiled_manifest.lock.json at runtime — replaces hardcoded PHASE4_ORACLE_FAMILIES for oracle eligibility checks in fixtures and xtask
- [Phase 15]: oracle-covered-update stamps helper/transform/optimizer/legacy entries unconditionally as covered because verify_helper_surface_coverage passes as part of generate_profile_parity_report
- [Phase 15]: manifest-audit check_oracle_coverage only checks stability=stable entries per D-07; should_fail now includes !uncovered_stable.is_empty() for hard CI gate
- [Phase 15-oracle-tolerance-unification-manifest-lock-closure]: Accept any non-empty subset of standard profiles in validate_required_profile_scope; CI matrix covers full coverage across parallel jobs
- [Phase 15-oracle-tolerance-unification-manifest-lock-closure]: Use fail-fast: false in oracle_parity_gate matrix so all four profile jobs report independently even when one fails (D-09)
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: PySCF master HEAD commit 60cd9022b5158b0eef46ded606a03b111a0ad08c pinned as the vendored nr_ecp source baseline (no nr_ecp-specific release tag); SHA recorded in vendor/pyscf-nr-ecp/NOTICE for byte-identity reproducibility.
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: Ship a cintx-authored dgemm_ reference shim (vendor/pyscf-nr-ecp/src/dgemm_shim.c) rather than depend on system BLAS — dev host has libblas.so.3 but no .so symlink. Future builds can drop the shim and link system BLAS without touching the rest of the vendor tree.
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: Use -std=gnu99 for the PySCF parallel cc::Build (vs libcint chain's gnu89) because nr_ecp.c uses C99 mid-block for-loop init declarations and <complex.h>; two distinct static libs keep the flag choice isolated.
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: crates/cintx-ops/generated/compiled_manifest.lock.json is the canonical manifest source — crates/cintx-ops/build.rs regenerates api_manifest.csv + api_manifest.rs from it on every build. The xtask manifest-audit subcommand has no --update flag; edit the lock JSON directly to add rows.
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: ECP rows land at OperatorIds 26..=29 (int1e_ecp_{cart,sph,ipnuc_cart,ipnuc_sph}); INT4C1E_CART_OPERATOR_ID=24 preserved. Test-only constants INT2E_STG_SPH_OPERATOR_ID and INT2E_IPIP1_SPH_OPERATOR_ID shift +4 to 106 and 116 — Plan 03 owns updating them (they live in #[cfg(test)] modules so cargo check still passes).
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: Parse fixture parameter JSON at runtime via serde_json::from_str(include_str!(...)) + OnceLock rather than hardcode literals (basis data stays auditable JSON, single source of truth).
- [Phase 19-int1e-ecp-type1-type2-evaluator P01]: Cu/LANL2DZ general-contraction blocks from BSE are split into single-NCTR libcint bas rows (libcint requires NCTR_OF=1 per row for distinct contraction coefficients). 3 BSE shells → 8 libcint bas rows.
- [Phase 19-int1e-ecp-type1-type2-evaluator P02]: Modified spherical Bessel i_l(x) evaluated DIRECTLY in all three branches (small-x, moderate-x Taylor, large-x asymptotic) — no upward or downward recurrence. Direct evaluation is numerically stable across the full Phase 19 envelope (l ∈ [0, ECP_LMAX=5], x ∈ [0, ∞)) and trivially parallel per-l. Boundaries (x=1e-7, x=16) mirror PySCF nr_ecp.c::ECPsph_ine verbatim with the exp(-z) scaling dropped.
- [Phase 19-int1e-ecp-type1-type2-evaluator P02]: Gauss-Chebyshev second-kind generator copied VERBATIM from vendor/pyscf-nr-ecp/src/nr_ecp.c::ECPgauss_chebyshev (lines 4848-4865). No precommitted binary table; runtime closed-form evaluation runs ~10 μs at LEVEL_MAX=2047 and is called once per shell per launch (not in a hot inner loop). Direct evaluation keeps byte-identity with PySCF trivially provable.
- [Phase 19-int1e-ecp-type1-type2-evaluator P02]: Gauss-Hermite hardcoded n=1..=8 reference table from DLMF 18.16.4 (physicists' convention, weight e^{-x²}). n > 8 panics with a clear out-of-range message — Phase 19's working set (Cu/LANL2DZ + standard ECP basis catalog) does not exceed n=8, so fail-fast is preferred to silently dropping precision. A future plan can add Golub-Welsch tridiagonal eigensolver fallback for higher n.
- [Phase 19-int1e-ecp-type1-type2-evaluator P02]: Plan 02 Test 2 correction — PySCF's ECPgauss_chebyshev weights include the radial-transform Jacobian dr/du, so they sum to ~23.5 at LEVEL0 (the truncated grid length on [0, ∞)), NOT to π/2. The plan's "sum to π/2 at atol=1e-12" assertion was replaced with the substantive ∫_0^∞ e^{-r} dr = 1 identity (rel ~3.8e-11 at LEVEL0, ~1.9e-14 at LEVEL_MAX) — exercises both nodes and weights simultaneously.
- [Phase 19-int1e-ecp-type1-type2-evaluator P02]: CubeCL 0.10 natural-log discovery — `f64::log` is the std two-arg version `log(self, base)` which cubecl does NOT override; the unary natural log is registered as `ln` (cubecl-core unary.rs:197 `impl_unary_func!(Log, ln, ...)`). Use `f64::ln(x)` inside `#[cube]` for natural log. Recorded in radial_quadrature.rs module rustdoc for future math modules.
- [Phase 19-int1e-ecp-type1-type2-evaluator P02]: CubeCL prelude shadows host f64 methods inside #[cfg(test)] modules of cintx-cubecl — calling `x.sinh()` inside a test resolves to the cubecl Cube intrinsic and panics with "Unexpanded Cube functions should not be called". Use precomputed reference f64 literals in tests instead of computing trigonometric/hyperbolic reference values at runtime.
- [Phase 19-int1e-ecp-type1-type2-evaluator P03]: OperatorId integers for the four ECP constants are read-and-derived from `OPERATOR_DESCRIPTORS` in `crates/cintx-ops/src/generated/api_manifest.rs` (positional pairing: `OPERATOR_DESCRIPTORS[K].id == OperatorId::new(K)`) — never hardcoded. A typed `ecp_operator_ids_match_constants` `#[test]` in cintx-ops/src/resolver.rs enforces manifest ↔ constants agreement at test-run time, plus the `int4c1e_cart → OperatorId::new(24)` preservation invariant.
- [Phase 19-int1e-ecp-type1-type2-evaluator P03]: BasisSet::try_new keeps its existing signature (SemVer-preserving); the new try_new_with_ecp(atoms, shells, ecp_shells) is the ECP-aware constructor. The empty-ECP default in try_new delegates to try_new_with_ecp with `Arc::from(Vec::<Arc<EcpShell>>::new().into_boxed_slice())`. Validates EcpShell::atom_index against atoms.len() with the existing CoreError::MissingAtomIndex variant (no new variant for atom-index validation).
- [Phase 19-int1e-ecp-type1-type2-evaluator P03]: EcpShell::try_new added one new CoreError variant — EcpAngularMomentumTooHigh { requested, max } — for the Projected(l > ECP_LMAX=5) check. Existing variants (InvalidShellCounts, ShellPrimitiveMismatch, InvalidNuclearDetail) cover the length/finiteness cases verbatim from Shell::try_new.
- [Phase 19-int1e-ecp-type1-type2-evaluator P03]: EcpBasArray::new reuses cintxRsError::InvalidBasLayout (the same variant RawBasView::new uses for length-not-multiple-of-BAS_SLOTS) — no new compat error variant. Phase 19 D-05's "ecpbas reuses BAS_SLOTS=8" decision makes the slab shape contract identical to ordinary bas rows.
- [Phase 19-int1e-ecp-type1-type2-evaluator P03]: FacadeError::MissingEcpBasis is facade-only — not emitted by From<cintxRsError>. SessionRequest::query_workspace preflight runs in order: aosym (Phase 18) → ECP (Phase 19) → runtime_query_workspace. Each preflight is independent and fails fast before runtime. The MissingEcpBasis variant carries `operator: String` resolved via Resolver::descriptor with a defensive fallback to OperatorId Display so the safe API never panics on a missing manifest entry.
- [Phase ?]: [Phase 20-01]: CintFloat::from_f64_lossy for f32 uses x as f32 truncation (documented lossy at threshold boundary only)
- [Phase ?]: [Phase 20-01]: A5 bytemuck staging cast proven SOUND — Wave 3 can use bytemuck::cast_slice_mut without Vec<F> fallback
- [Phase ?]: [Phase 20-01]: PrecisionKind defaults to F64; ExecutionPlan.precision field byte-identical on f64 path (D-08/D-12)
- [Phase 20-10]: CR-01 fix shape: out_elems captured pre-cast (f64 staging.len()) in outer dispatcher; &mut staging_f32[..out_elems] passed to typed inner so all copy_len/not0 in inner are automatically correct
- [Phase 20-10]: WR-05 device path: used F::EPSILON (CubeCL Float const, verified from cubecl-core-0.10.0) not F::new(f64::EPSILON as f32 * 0.5); host uses num_traits F::epsilon(); both yield the same type-appropriate epsilon
- [Phase 20-10]: WR-05 f64 impact: F::epsilon() for f64 == 2.22e-16 vs old DBL_EPSILON_HALF 1.11e-16; factor-of-2 within f64 oracle guard band (atol=1e-12); no precision branch needed
- [Phase 20-10]: WR-03: compute_pdata_host converts ALL inputs to f64 first; Gaussian-product exponential fac = (-ai*aj/zeta_ab*rr).exp() always f64-precision regardless of F; trailing .to_f64().unwrap_or() removed (values already f64 after input conversion)
- [Phase ?]: nmax headroom for ipovlp: nmax = li+lj+1 (one extra bra level so nabla ix+1 access is valid)
- [Phase ?]: nmax headroom for ipkin: nmax = li+lj+3 (kinetic +2 for D_j^2 jx+2 access + nabla +1 for ix+1 access)
- [Phase ?]: Spinor gradient returns UnsupportedApi (R5/D-03): guard placed before gradient compute path
- [Phase ?]: [Phase 24-01]: Only the 12 source-confirmed _origj symbols (r/rr/r2/r4/z/zz) registered — no rrr_origj/rrrr_origj exist in libcint 6.1.3 intor1.c (OQ-3)
- [Phase ?]: [Phase 24-01]: rinv/drinv parity tests inject a NON-ZERO rinv center via env_with_rinv_origin (PTR_RINV_ORIG, not PTR_COMMON_ORIG) per D-04/OQ-1; zero origin trivially-passing and disallowed
- [Phase ?]: [Phase 24-01]: OQ-2 cart_offset lib-unit failure reproduced at pre-phase-20 commit 8997703 → standalone harness bug; Phase 24 integration gate de-blocked
- [Phase ?]: [Phase 24-01]: rank-parameterized vendor_parity sizes every buffer rank*ni*nj (D-08); parity #[test] bodies gated on has_vendor_libcint as the Nyquist RED target for plans 02-05
- [Phase 24]: Cluster A moment kernel: ONE parameterized #[cube] kernel for r/rr/rrr/rrrr/r2/r4/z/zz + _origj; per-axis moment ladder m_p = Sum_t C(p,t) drj^(p-t) overlap[jx+t] (closed-form of libcint CINTx1j_1e) reproduces verbatim gout order, proven by atol=1e-12 vendor parity
- [Phase 24]: origin-source branch (D-02) is host-side: drj = rj - origin (common_orig for base, rj for _origj so drj=0 = libcint G1E_R_J pointer shift); no new env code
- [Phase 24]: _origj parity tests use a CROSS-center non-square block (H1-1s x O-2p); same-center even-moment _origj integrals are identically zero (vendor included)
- [Phase ?]: [Phase 24-03]: rinv/drinv read env[PTR_RINV_ORIG] (env[4..6]) NOT PTR_COMMON_ORIG (D-04/OQ-1); separate is_rinv_family_symbol gate. int1e_rinv = scalar nuclear Rys arm with atom-loop dropped to single rinv-center origin, charge=+1 no -Z_C; int1e_drinv = D_I+D_J of the rinv G-tensor (transl-invariance grad), rank 3, bra+1/ket+1 headroom; both fail-closed nroots>5. Vendor parity 0 at atol=1e-12 cart+sph
- [Phase 24]: int1e_p4 (∇⁴, rank 1) = Laplacian-of-Laplacian on the overlap G-tensor (no Rys), BOTH-side +2 headroom (ng={2,2,...}, nmax=li+lj+4); built from d_i_1e_into/d_j_1e_into as four tensors (g0, D_J², D_I², D_I²·D_J²); rank-1 contraction s0+2s4+2s8+s40+2s44+s80 verbatim from intor1.c:2534; even+origin-free → CROSS-center non-square parity block (H1-1s × O-2p); fail-closed li+lj+4>8; vendor parity 0 at atol=1e-12 cart+sph.
- [Phase 25]: FND-06: single upfront assert_staging_size() BufferTooSmall contract point in planner.rs evaluate() replaces all per-element scatter guards (D-04); 20 guards stripped across 6 kernel files
- [Phase 25]: FND-06: rank-81 OOM no-partial-write proven via int1e_rrrr_cart driver + sentinel-survives-typed-stop test (D-05)
- [Phase 25]: oracle-cart-offset-vendor-zero CONFIRMED pre-existing at pre-phase-20 commit 00771ab (CINTshells_cart_offset[4] cintx=8 vendor=0); not a Phase-25 regression; does not block family gate (integration --test passes)
- [Phase 25-03]: HESS-01 — int1e_ipip{ovlp,nuc,kin,rinv} are bra-only ∇² = the Phase-23 first-order D_I engine applied twice (g1=D_I(g0,i+1), g2=D_I(g0,i), g3=D_I(g1,i)); ovlp uses the no-Rys overlap base, nuc/rinv the nuclear Rys base, all sharing ONE gout-permutation helper (gradgrad_bra_contract) with the verbatim hess.c order [s0,s3,s6,s1,s4,s7,s2,s5,s8]
- [Phase 25-03]: ipipkin's -½ kinetic factor must be folded into cintx's gout (observed=2× vendor without it): libcint emits -(s) and scales by ½ in CINT1e_drv, but cintx contracts s directly into staging. ng={2,2,...} (nmax=li+lj+4, lj_ext=lj+2) vs ng={2,0,...} for ovlp/nuc/rinv
- [Phase 25-03]: ipiprinv parity REQUIRES a PTR_ENV_START-aligned fixture (env[0..20] reserved) so the rinv origin at env[4..6] is not clobbered by atom coords; inject a nonzero rinv origin via env_with_rinv_origin (zero origin is trivially-passing)
- [Phase 25-03]: xtask is a standalone cargo project (own Cargo.lock), NOT a workspace member — run `cd xtask && cargo run -- manifest-audit`, not `cargo run -p xtask` from the workspace root
- [Phase ?]: [Phase 25-04]: 2e Hessian gout permutation is identical between F12 and plain Coulomb — reuse the Phase-13 gout_ipip1/ipvip1/ip1ip2 helpers verbatim (pub(crate)); only the rank-81 gout_ipip1ipip2 was new
- [Phase ?]: [Phase 25-04]: D-07 re-home preserves the two source-only-gate raw.rs tests by repointing to int2e_breit_r1p2_spinor (still source-only) instead of deleting coverage
- [Phase 25-05]: HESS-03 — int3c2e_ipip1 == gout_ipip1 verbatim (identical s[] + column-major reorder); ipip2 = same with G2E_D_K, so one new gout_ipip2_l (nabla1l_2e on the 2e `ll` slot, ll+2) covers the ket case. ipip2 KET headroom (lk+2) distinct from ipip1's bra li+2
- [Phase 25-05]: 3c2e host Hessian launchers MUST include the per-primitive Gaussian-overlap prefactors (fac_env = common_factor * pdata_ij.fac * pdata_kl.fac), exactly like the device ip1/ip2 host bridge host_ip1_cart_blocks; bare common_factor scales 3c2e output wrong. 2c2e is immune (phantom j,l coincide with i,k → overlap fac=1)
- [Phase 25-05]: manifest lock stores component_rank only, NOT the ng[] tuple — ket k_inc=2 is a code contract (build_2e_shape(li,lj,0,lk+2)) verified by byte-identity parity vs vendor G2E_D_K, not a lock grep
- [Phase 26-01]: complex_output is a per-family lock.json bool defaulting false; only spinor operator rows backfilled true. Planner build_output_layout + assert_flat_buffer_contract re-key off this flag (complex_interleaved) instead of Representation::Spinor, so GIAO cart/sph families size 2x from manifest data.
- [Phase ?]: [Phase 26-02]: Append new manifest families at the END to preserve all positional OperatorIds (zero-shift registration).
- [Phase ?]: [Phase 26-02]: GIAO families per-family nuclear model: gnuc/ignuc atom-sum -Z (int1e_type=2); ia01p/a01gp/cg_a11part/giao_a11part single rinv center +1 (int1e_type=1).
- [Phase ?]: [Phase 26-02]: complex_output families emit REAL device output materialized host-side as [re=0, im=value]; vendor parity extracts imaginary half, asserts real==0 (D-07/D-15).
- [Phase ?]: [Phase 26-03 GIAO-02]: int2e_g1g2 component_rank derived from intor2.c ng[7]=9 (D-16, not guessed); rank-9 both-electron gauge family byte-identical first try.
- [Phase ?]: [Phase 26-03 GIAO-02]: 2e GIAO families host-routed via fill_g_tensor_2e (Hess2e analog) with new r0i_2e/r0k_2e position operators in f12.rs; complex-interleaved [re=0,im=value] 4-shell staging.
- [Phase ?]: [Phase 28-01]: si 2D bra step (cart_to_spinor_si_2d) uses a_bra_cart2spinor_si signs (+ca_i*vz/-cb_r*vy/+cb_i*vx), NOT apply_si_block's CINTc2s_ket_spinor_si1 convention; apply_si_block left untouched for the single-block helper surface.
- [Phase ?]: [Phase 28-01]: cart_to_spinor_si_2d owns the KET->BRA transpose internally per gc block (Phase-27 D-06), reuses ordinary apply_ket_transform verbatim, sizes all buffers via spinor_len (never hardcoded 4l+2), and fail-closes before any write.
- [Phase ?]: [Phase 28]: σ·p assembler emits PRE-BLOCKED component-leading gc[comp*block_len+n] on-device (not interleaved gout[n*4+comp]) so cart_to_spinor_si_2d reads gc_x=block0..gc_1=block3 with no host transpose; rank-parameterized via #[comptime] tensor_rank for int1e_sigma reuse (D-03)
- [Phase ?]: Phase 28-03: int1e_sp_spinor registered infrastructure-only (oracle_covered=false, appended last so OperatorId 347, no positional shift); SC#4 enforced via is_skipped_spinor_fixture so oracle-covered-update refuses to flip it; vendor_int1e_sp_spinor FFI shim + bindgen allowlist added; D-01 honored (σ flips deferred to Phase 29).
- [Phase ?]: FND-05 proven byte-identical: int1e_sp σ·p assembler → cart_to_spinor_si_2d vs vendor int1e_sp_spinor at atol=1e-12 (no manifest flip, D-01)
- [Phase ?]: int1e_sigma is component_rank=3 (3 stacked Pauli σ-matrices), empirically measured — 29-01 rank-1 prior disproven
- [Phase ?]: 29-02: 7 1e Group-4 σ families byte-identical to libcint at atol=1e-12 via overlap+Rys nuclear #[cube] engines; 8 rows oracle_covered=true spinor-only
- [Phase 29-03]: Built the 2e cart→spinor transform suite as 6 composable per-electron fns (electron-1 producing opij, electron-2 consuming it) matching libcint's c2s_si_2e1/2e2 driver split, so Wave-3 launchers can pair electron-1×electron-2 transforms per family
- [Phase 29-03]: apply_2d_spinor_zi transcribes the 2×2 Pauli σ·n expansion verbatim from cart2sph.c:4118-4186; the σ-mix is bra1-only so the ket1 step reuses apply_ket1_block_all_kappa unchanged
- [Phase 29-03]: build_kappa_spinor_2e_fixture is a 4-shell non-square (2,6,2,4) GT/LT-mix nctr>1 quartet (D-02); 29-03 delivers compiling structural code only — 2e byte-identity is the 29-04 [BLOCKING] micro-test
- [Phase ?]: [Phase 29-04]: int2e_spsp1 reuses the ipvip1 (nabla_i nabla_j) s[0..8] triple-product tensor; its sigma-p1 gout (gc_x=s5-s7, gc_y=s6-s2, gc_z=s1-s3, gc_1=s0+s4+s8) is a different linear fold of the SAME tensor + headroom (i+1,j+1), proven byte-identical to vendored libcint at atol=1e-12. The D-03 BLOCKING gate is GREEN; Wave 3 unblocked.
- [Phase ?]: [Phase 29-05]: REL-04 ssp/sps drivers live in gaunt1.c and vsp/spv in dkb.c — NEITHER was in oracle build.rs; added both .file() entries (corrects CONTEXT.md, trusts RESEARCH Pitfall 1). Without them REL-04 vendor shims have no symbol to link.
- [Phase ?]: [Phase 29-05]: inserted 15 remaining 2e Group-4 manifest rows after the spsp1 row (index >=349), past every hardcoded OperatorId const (<=106) — no positional drift; all component_rank=1 (σ fold internal to c2s), spinor-only, oracle_covered=false until 29-06.
- [Phase ?]: [Phase 29-05]: rel_2e_sigma_parity.rs is a RED scaffold (cintx launchers land in 29-06: cintx collector is a panic-stub, 15 byte-identity gates #[ignore]'d); the always-on no-silent-skip sweep runs all 15 vendor arms NON-SKIPPED, proving the gaunt1.c/dkb.c build wiring linked a real driver per family.
- [Phase ?]: 29-06: all 16 2e Group-4 σ families byte-identical to libcint 6.1.3 (atol 1e-12); oracle_covered=true spinor-only; 24/24 Group-4 covered. Group 4 complete. Key: 2-sided σ⊗σ headroom {1,1,1,1}.
- [Phase ?]: [Phase 30-00]: Gauge fold ported as CINTx1i_1e recurrence f[i]=g[i+1]+origin*g[i] in a separate sigma_p_cg_sa10sp_kernel; int1e_sp path byte-identical; proven via int1e_cg_sa10sp vendor byte-identity at atol=1e-12 + cg->giao collapse at common_orig=[0,0,0].

### Roadmap Evolution

- Phase 5 added: Re-implement detailed-design GPU path with CubeCL (wgpu backend)
- v1.1 roadmap created: Phases 7-10 (executor rewrite, math infrastructure, 1e kernel, 2e+ kernels and oracle gate)

### Pending Todos

None yet.

### Blockers/Concerns

- ~~[Phase 19, 2026-05-20] **ECP byte-identity blocked on missing K-Taylor port.**~~ **RESOLVED 2026-05-20.** The K-Taylor replan (19-05..08) shipped and is verified: 19-05 ported PySCF's exact table-interpolation radial machinery (`ecpsph_ine_opt_host`, `ecprad_part_host`, `type1_rad_part_host`, `type2_facs_rad_host`) with the two K-Taylor tables embedded as byte-locked `.bin` blobs + an xtask drift-gate; 19-06 rewired the scalar `compute_type1/2_pair` onto it (byte-identity atol=1e-12 confirmed); 19-07 ported `nr_ecp_deriv.c::_deriv1_cart` for the gradient (byte-identity atol=1e-12 confirmed); 19-08 added the optional non-blocking libecpint cross-check. All four ECP rows are `oracle_covered=true`. The dropped-into-generic-parity-matrix regression and the CLAUDE.md fail-closed hardening gaps surfaced during the execute-phase gates were both fixed (commits f589c3d/52b5086/cbe95bb).
- int1e_a01gp (rank-9 GIAO) deferred: 2x ket-element parity discrepancy; 10/11 GIAO-01 families byte-identical.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260509-c6d | update cubecl version 0.10.0 in all repository | 2026-05-08 | aa59ceb | [260509-c6d-update-cubecl-version-0-10-0-in-all-repo](./quick/260509-c6d-update-cubecl-version-0-10-0-in-all-repo/) |
| 260529-r2g | refactor center_2c2e to a generic-float #[cube] device kernel + rocm random oracle test | 2026-05-29 | 194edae | [260529-r2g-center-2c2e-cubecl-gpu-kernel](./quick/260529-r2g-center-2c2e-cubecl-gpu-kernel/) |
| 260529-e69 | refactor center_3c1e to a generic-float CubeCL #[cube] device kernel + rocm random oracle test | 2026-05-29 | bd24b8e | [260529-e69-refactor-center-3c1e-rs-to-cubecl-kernel](./quick/260529-e69-refactor-center-3c1e-rs-to-cubecl-kernel/) |
| 260529-exs | refactor center_3c2e to generic-float CubeCL #[cube] device kernels (scalar + int3c2e_ip1) + rocm random oracle test | 2026-05-29 | cc83ec3 | [260529-exs-center-3c2e-cubecl-kernel](./quick/260529-exs-center-3c2e-cubecl-kernel/) |
| 260529-fsa | refactor center_4c1e to a generic-float CubeCL #[cube] device kernel + rocm random oracle test | 2026-05-29 | b7f5519 | [260529-fsa-refactor-center-4c1e-rs-to-cubecl-kernel](./quick/260529-fsa-refactor-center-4c1e-rs-to-cubecl-kernel/) |
| 260529-gbf | refactor ecp.rs to a generic-float CubeCL #[cube] device kernel (Type-1 angular splice) + rocm random oracle test | 2026-05-29 | 55b4a88 | [260529-gbf-refactor-ecp-rs-to-cubecl-kernel-with-ge](./quick/260529-gbf-refactor-ecp-rs-to-cubecl-kernel-with-ge/) |
| 260529-hin | port the ECP Type-2 two-dgemm angular splice to a generic-float CubeCL #[cube] device kernel + rocm oracle | 2026-05-29 | d3405f4 | [260529-hin-port-the-ecp-type-2-two-dgemm-angular-sp](./quick/260529-hin-port-the-ecp-type-2-two-dgemm-angular-sp/) |
| 260529-i2q | refactor f12.rs base Cartesian contraction to a generic-float CubeCL #[cube] device kernel + rocm random oracle test | 2026-05-29 | 45a4a17 | [260529-i2q-refactor-f12-rs-to-cubecl-device-kernel-](./quick/260529-i2q-refactor-f12-rs-to-cubecl-device-kernel-/) |
| 260529-imi | refactor one_electron.rs scalar operators (ovlp/kin/nuc) to a generic-float CubeCL #[cube] device kernel + rocm random oracle test (mismatch_count=0, 48 cases) | 2026-05-29 | 23eb85d | [260529-imi-refactor-one-electron-rs-to-cubecl-kerne](./quick/260529-imi-refactor-one-electron-rs-to-cubecl-kerne/) |
| 260529-j7d | port the 1e gradient operators (ipovlp/ipkin/ipnuc/iprinv) in one_electron.rs to generic-float CubeCL #[cube] device kernels + rocm random oracle test (mismatch_count=0, 48 cases × 4 ops) | 2026-05-29 | 9f3b9b2 | [260529-j7d-port-the-1e-gradient-operators-ipovlp-ip](./quick/260529-j7d-port-the-1e-gradient-operators-ipovlp-ip/) |
| 260529-jtd | implement spinor int1e gradient (ipovlp/ipkin/ipnuc/iprinv) via on-device cart gradient + host per-component cart→spinor transform; was UnsupportedApi (R5/D-03). Vendor parity vs libcint 6.1.3 = 0 mismatches, all 4 ops | 2026-05-29 | fb02060 | [260529-jtd-implement-spinor-int1e-gradient-ipovlp-i](./quick/260529-jtd-implement-spinor-int1e-gradient-ipovlp-i/) |
| 260529-kke | fix scalar spinor int1e cart→spinor block-orientation bug (transpose ket-major→bra-major before cart_to_spinor_sf_2d); proven via asymmetric p×d vendor parity (ovlp/kin/nuc: 232→0 mismatches vs libcint 6.1.3) | 2026-05-29 | f4230c6 | [260529-kke-fix-scalar-spinor-int1e-cart-to-spinor-b](./quick/260529-kke-fix-scalar-spinor-int1e-cart-to-spinor-b/) |
| 260529-lbr | fix CINTshells_{cart,spheric,spinor}_offset to match libcint i<nbas semantics (write nbas start-offsets, drop trailing ao_loc[nbas] total). Unblocks vendor oracle gate past helper check; surfaced a separate pre-existing CINTgto_norm helper mismatch (follow-up) | 2026-05-29 | 3bf0682 | [260529-lbr-fix-cintshells-cart-spheric-spinor-offse](./quick/260529-lbr-fix-cintshells-cart-spheric-spinor-offse/) |
| 260529-mfh | fix CINTgto_norm to libcint misc.c closed form (was inverted formula, all 15 (l,a) failed). Vendor parity 0 mismatches; gate now advances to next pre-existing blocker CINTc2s_bra_sph | 2026-05-29 | 8db9fcb | [260529-mfh-fix-cintgto-norm-to-match-libcint-misc-c](./quick/260529-mfh-fix-cintgto-norm-to-match-libcint-misc-c/) |
| 260529-mqo | fix CINTc2s_bra_sph both defects: vendor FFI wrapper now honors libcint's returned *mut f64 (l<2 returns gcart, not gsph), and cintx stub now applies real per-l bra c2s via pub c2s_coeff. Vendor parity 0 mismatches; gate CLEARS the full helper/transform block → now hits a pre-existing cint2e_sph/cart 2e-integral divergence (cintx=1.709e-3 vs vendor=3.22e-4) in verify_legacy_wrapper_parity | 2026-05-29 | 1e5bf0a | [260529-mqo-fix-cintc2s-bra-sph-both-defects-vendor-](./quick/260529-mqo-fix-cintc2s-bra-sph-both-defects-vendor-/) |
| 260529-ne7 | fix OracleRawInputs::sample() to libcint-conformant PTR_ENV_START=20 env layout (was packing data onto reserved slots; env[8]=PTR_RANGE_OMEGA made vendor compute range-separated 2e). cint2e divergence GONE; base + with-4c1e gate profiles PASS CLEAN. Exposed that the old layout accidentally supplied env[9]=PTR_F12_ZETA → with-f12 profiles now correctly fail-closed (F12 fixtures need a real zeta injection, follow-up) | 2026-05-29 | 3b4ced5 | [260529-ne7-fix-oraclerawinputs-sample-env-layout-to](./quick/260529-ne7-fix-oraclerawinputs-sample-env-layout-to/) |
| 260529-nt1 | set env[PTR_F12_ZETA]=1.2 in OracleRawInputs::sample() (conformant designated F12-zeta slot) so F12/STG/YP oracle parity has a valid zeta. **Full vendor oracle gate now GREEN: all 4 profiles (base, with-f12, with-4c1e, with-f12+with-4c1e) PASS CLEAN, mismatch_count=0 each** vs libcint 6.1.3; F12 family matches vendor numerically | 2026-05-29 | aba9af5 | [260529-nt1-set-env-ptr-f12-zeta-1-2-in-oraclerawinp](./quick/260529-nt1-set-env-ptr-f12-zeta-1-2-in-oraclerawinp/) |
| 260529-q4k | refactor two_electron.rs scalar int2e (4-center ERI) to a generic-float CubeCL #[cube(launch)] device kernel (G-tensor build + all 4 HRR branches + Cartesian contraction on-device, comptime nroots, host-computed strides as runtime u32) dispatched on all 5 backends incl. ROCm/HIP; int2e_ip1 gradient stays host. ROCm random vendor parity: mismatch_count=0, any_nonzero=true across 96 quartets (int2e_sph+cart) vs libcint 6.1.3. f64 byte-identity preserved (safe_api_arity4 + two_electron_ip1 + 13/13 lib tests green) | 2026-05-29 | b967cc4 | [260529-q4k-two-electron-gpu-port](./quick/260529-q4k-two-electron-gpu-port/) |
| 260529-twi | refactor ALL of unstable.rs (origi/grids/breit/origk/ssc) to generic-float CubeCL #[cube(launch)] device kernels + rocm oracles. First split the 3511-line file into a per-family module dir (unstable/{mod,shared,origi,grids,breit,origk,ssc}.rs, behavior-preserving). Then ported each family's scalar path on-device (host keeps c2s/spinor + AO scatter; derivative sub-paths origi-ip2/origk-ip1/grids-derivs + breit gout/spinor deferred-to-host, documented). ROCm vendor parity mismatch_count=0: origi 96, origk 144, ssc 48, breit 48 (spinor, non-square) cases. grids vendor oracle BLOCKED by pre-existing eval_raw InvalidShellTuple{expected:2,got:4} (baseline noise, not this port) → validated by direct in-crate device-vs-host on ROCm (64 cases, 0 mismatch) + a blocker-lock test. 16/16 in-crate device tests green; unstable_source_parity baseline 8/15 unchanged (no regression) | 2026-05-30 | 5a6d71f | [260529-twi-refactor-unstable-rs-kernels-to-generic-](./quick/260529-twi-refactor-unstable-rs-kernels-to-generic-/) |
| 260530-9ay | fix + GPU-port the deferred unstable derivative sub-paths (origi-ip2, origk-ip1, grids ip/ipip/ipvip/spvsp). ROOT CAUSE was a manifest bug, not math: these ops had component_rank=1 in compiled_manifest.lock.json so the planner under-allocated the workspace and launch_* dropped all components past the first (zeros). Corrected ranks (origi-ip2/origk-ip1=3, grids ip=3, ipip/ipvip=9, spvsp=4) → CPU vendor parity restored for origi-ip2 r2/r4 + origk-ip1 r2/r4. Then ported all to #[cube(launch)] device kernels (origi_ip2_kernel, origk_ip1_kernel, one grids_deriv_kernel comptime op_kind) on all 5 backends. ROCm: origi-ip2 vendor parity 96/0; origk-ip1 r2/r4 0; grids derivs device-vs-host 0 (vendor blocked by pre-existing InvalidShellTuple). RESIDUAL (not fixed, documented): origk-ip1 r6 diverges ~6% from libcint on the y-component at top k-power — cintx self-consistent (ip1=grad(scalar), FD-verified) + scalar matches vendor, but ip1_r6 ≠ vendor; root cause not isolated. 26/26 in-crate device tests green; unstable_source_parity 12/23 (was 8/23), no regressions | 2026-05-30 | 51017b9 | [260530-9ay-fix-gpu-port-deferred-unstable-derivativ](./quick/260530-9ay-fix-gpu-port-deferred-unstable-derivativ/) |
| 260530-iiq | fix WR-03: int3c1e general-contraction (nctr>1) support — nctr-blocked output for the scalar + ip1 + iprinv launchers (center_3c1e.rs). The mandated empirical block-ordering step uncovered a deeper root cause: the libcint env coefficient block is COLUMN-MAJOR (env[ci*nprim+ip]) while cintx Shells are ROW-MAJOR (coeff[ip*nctr+ci]); the eval_raw env→Shell parse copied verbatim, silently TRANSPOSING nctr>1 coefficients for EVERY family's raw path (latent — no prior nctr>1 parity test existed). Fixed with a transpose at the env→Shell boundary in raw.rs. New int3c1e_genctr_parity vendor test 6/6 byte-identical at atol=1e-12 (cart+sph, non-square p(nctr=2)×d×s) for scalar+ip1+iprinv. nctr==1 preserved across families (int3c1e_ip 5/5, center_3c1e 2/2, one_electron 6/6, two_electron 2/2; cubecl --lib 280, compat 43, ops 11). Device #[cube] kernels unchanged. Closes WR-03. | 2026-05-30 | 9be0141 | [260530-iiq-int3c1e-general-contraction-nctr-1-suppo](./quick/260530-iiq-int3c1e-general-contraction-nctr-1-suppo/) |
| 260530-j62 | fix CR-01: make the unstable-source profile-membership check in raw.rs validate_profile_and_source_gate REACHABLE. Two is_source_only() blocks collapsed to one — the first returned Ok() early, making the second's is_compiled_in_profile check dead code (a source-only op in no available profile was silently accepted). The code-review's literal suggestion (check only "unstable-source") was WRONG: active_manifest_profile() never returns "unstable-source" and two source-only int2e symbols ship in BASE profiles, so the correct rule is accept iff compiled in the active profile OR unstable-source, else reject. Verified: compat 43, cubecl --lib 280, unstable_source_parity 23/23 (--features cpu,unstable-source-api + vendor) — no source family falsely rejected. | 2026-05-30 | 108dfe2 | [260530-j62-fix-cr-01-unreachable-unstable-source-pr](./quick/260530-j62-fix-cr-01-unreachable-unstable-source-pr/) |
| 260530-k0s | fix the pre-existing grids derivative host-fallback Rys-root undersize bug (surfaced by the ROCm device oracle). grids_contract_{nuclear_like,ip,ipip,ipvip} computed nrys_roots up to 5 but fetched only 2 roots (rys_root1/2_host), so the device->host fallback for nroots>2 panicked 'index out of bounds: len 2 index 2'. Replaced with rys_roots_host(nrys_roots) (byte-identical for nroots<=2, correct 3..=5); spvsp delegates to ipvip (covered). Verified: grids_random_rocm_parity now passes on gfx1152 (was panicking); unstable_source_parity 23/23; cubecl --lib 280. | 2026-05-30 | 1395aae | [260530-k0s-fix-grids-ipvip-rys-root-array-undersize](./quick/260530-k0s-fix-grids-ipvip-rys-root-array-undersize/) |
| 260531-aw1 | force-port the remaining host-side math (eigh.rs symmetric-tridiagonal eigensolver + rys_wheeler.rs long-double Rys nroots>=6 engine, previously host-by-design per FND-02) to CubeCL #[cube] CPU kernels. eigh `cint_diagonalize` now a #[cube] kernel (bit-identical to host MRRR/QL+Rayleigh+Sturm ref, MAXDIFF=0 over 2000 random tridiagonals). FMA fidelity probe verdict FUSED (CubeCL 0.10.0 CpuRuntime lowers fma() bit-for-bit = host mul_add) → double-double two_prod uses the device fma intrinsic, no Dekker-split fallback. Rys nroots 8..12 (double-double Jacobi/Laguerre/Schmidt via a DdDev CubeType) ported fully ON-DEVICE, byte-identical to host dd path + vendor at the documented split. New in-crate vendor reference-table test rys_roots_host_nroots6to12_matches_libcint (gold from libcint harness). **Vendor parity preserved: 29/29 family suite byte-identical (center_2c2e 2, center_3c1e 2, deriv34 14, hess1e_ipip 8, hess2e 2, hess_multicenter_ipip 2, int2c2e_ip 4) + rys_nroots_sweep GREEN; NO tolerance loosened, NO reference value edited** (orchestrator-run vendor gate, --features cpu + CINTX_ORACLE_BUILD_VENDOR=1). DEVIATION (parity-honest escape hatch, plan-sanctioned): Rys nroots **6,7** production dispatch kept HOST — the f64 device kernels are bit-identical in isolation but a CubeCL CpuRuntime launch in the family hot path perturbs subsequent HOST g-tensor accumulation ~1e-11 (FP-environment side effect), tripping the flat-1e-12 family gate; device kernels retained in-module. Follow-up: root-cause the CubeCL launch FP-environment side effect to land 6,7 on-device too. Diff confined to crates/cintx-cubecl/src/math/. | 2026-05-31 | 93b7840 | [260531-aw1-port-host-eigh-and-rys-wheeler-to-cubecl](./quick/260531-aw1-port-host-eigh-and-rys-wheeler-to-cubecl/) |

## Session Continuity

Last session: 2026-05-31T22:57:26.415Z
Stopped at: Phase 30 context gathered
Resume file: None
