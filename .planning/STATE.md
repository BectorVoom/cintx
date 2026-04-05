---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: "Milestone: CubeCL Direct Client API & Real Kernel Compute"
status: Milestone complete
stopped_at: Completed 12-05-PLAN.md
last_updated: "2026-04-05T00:21:45.172Z"
progress:
  total_phases: 10
  completed_phases: 10
  total_plans: 47
  completed_plans: 48
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.  
**Current focus:** Phase 12 — real-spinor-transform-c2spinor-replacement

## Current Position

Phase: 12
Plan: Not started

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

### Roadmap Evolution

- Phase 5 added: Re-implement detailed-design GPU path with CubeCL (wgpu backend)
- v1.1 roadmap created: Phases 7-10 (executor rewrite, math infrastructure, 1e kernel, 2e+ kernels and oracle gate)

### Pending Todos

None yet.

### Blockers/Concerns

None currently.

## Session Continuity

Last session: 2026-04-04T23:58:54.471Z
Stopped at: Completed 12-05-PLAN.md
Resume file: None
