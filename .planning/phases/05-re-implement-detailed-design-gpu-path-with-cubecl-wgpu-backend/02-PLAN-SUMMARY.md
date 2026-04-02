---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 02
subsystem: cubecl
tags: [rust, cubecl, wgpu, capability, bootstrap, backend-preflight, fingerprint]

# Dependency graph
requires:
  - phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
    plan: 01
    provides: BackendIntent/BackendCapabilityToken contract in ExecutionOptions + WorkspaceQuery

provides:
  - WgpuCapabilitySnapshot struct with adapter name/api/type/vendor/device/features/limits fields
  - CapabilityReason enum with MissingAdapter/MissingFeature/LimitTooLow/FamilyUnsupported/RepresentationUnsupported variants
  - WgpuPreflightReport combining snapshot, fingerprint, and unsatisfied reasons
  - capability_fingerprint() using FNV-1a 64-bit hash over all snapshot fields for deterministic drift detection
  - bootstrap_wgpu_runtime(intent) -> Result<WgpuPreflightReport, cintxRsError> selector parsing and adapter preflight
  - Fail-closed selector parsing for auto/default/discrete:N/integrated:N patterns
  - std::panic::catch_unwind wrapper to convert CubeCL adapter panics to typed UnsupportedApi errors
  - Exported capability and bootstrap modules from cintx-cubecl lib.rs
  - cubecl-wgpu = 0.9.0 and cubecl-runtime = 0.9.0 explicit dependency wiring

affects:
  - 05-03 (executor rewrite will consume bootstrap_wgpu_runtime and WgpuPreflightReport)
  - cintx-runtime (BackendCapabilityToken.capability_fingerprint will be populated with real fingerprint from bootstrap)
  - Any caller of CubeClExecutor that needs typed capability failure context

# Tech tracking
tech-stack:
  added:
    - cubecl-wgpu = 0.9.0 (explicit dep in cintx-cubecl for wgpu bootstrap APIs)
    - cubecl-runtime = 0.9.0 (explicit dep for CubeCL runtime contract)
    - wgpu = 26.0.1 (for wgpu::Adapter feature/limit inspection)
    - cubecl with wgpu feature enabled
  patterns:
    - FNV-1a 64-bit fingerprint over capability snapshot fields for deterministic drift detection
    - reason-prefixed diagnostic strings (missing_adapter, missing_feature:<name>, limit_too_low:<name>:<actual>/<required>) for typed D-12 taxonomy
    - std::panic::catch_unwind at CubeCL adapter setup boundary to convert panics to typed UnsupportedApi errors
    - Selector parsing with explicit format contract (auto/default/discrete:N/integrated:N)

key-files:
  created:
    - crates/cintx-cubecl/src/capability.rs
    - crates/cintx-cubecl/src/runtime_bootstrap.rs
  modified:
    - crates/cintx-cubecl/Cargo.toml
    - crates/cintx-cubecl/src/lib.rs

key-decisions:
  - "Use FNV-1a 64-bit hash over sorted feature/limit lists plus adapter identity fields for reproducible capability fingerprints"
  - "Wrap cubecl init_setup with std::panic::catch_unwind to prevent adapter-absent environments from panicking instead of returning typed errors"
  - "Keep selector format strings simple (auto, discrete:N, integrated:N) matching WgpuDevice enum variants as documented in cubecl-wgpu 0.9.0"
  - "Collect a fixed set of named wgpu::Features flags into capability snapshot rather than raw bitfield to keep fingerprint human-auditable"

patterns-established:
  - "Capability taxonomy uses reason-prefixed strings per D-12: missing_adapter / missing_feature:<name> / limit_too_low:<name>:<actual>/<required>"
  - "Bootstrap returns WgpuPreflightReport; callers use is_capable() and first_reason() to drive error propagation"
  - "wgpu::Adapter.features()/limits() used for capability snapshot collection; calls placed after init_setup returns WgpuSetup"

requirements-completed:
  - EXEC-02
  - COMP-05
  - VERI-04

# Metrics
duration: 7min
completed: 2026-04-02
---

# Phase 05 Plan 02: CubeCL Capability Snapshot and WGPU Bootstrap Summary

**Typed wgpu capability snapshot with FNV-1a fingerprint, D-12 reason taxonomy, and fail-closed bootstrap_wgpu_runtime selector parsing using cubecl-wgpu 0.9.0**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-02T07:36:09Z
- **Completed:** 2026-04-02T07:44:05Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Defined `WgpuCapabilitySnapshot` with all adapter identity and compute-relevant feature/limit fields; deterministic FNV-1a 64-bit `capability_fingerprint()` function that changes when any field changes
- Defined `CapabilityReason` enum with all D-12 reason variants and `to_reason_string()` producing reason-prefixed diagnostic strings (`missing_adapter`, `missing_feature:<name>`, `limit_too_low:<name>:<actual>/<required>`, `family_unsupported:<family>`, `representation_unsupported:<repr>`)
- Implemented `bootstrap_wgpu_runtime(intent)` with fail-closed selector parsing (`auto`/`default`/`discrete:N`/`integrated:N`) and `std::panic::catch_unwind` wrapper for adapter-absent environments
- Added explicit `cubecl-wgpu = "0.9.0"`, `cubecl-runtime = "0.9.0"`, and `wgpu = "26.0.1"` dependencies with `cubecl` `wgpu` feature enabled

## Task Commits

Each task was committed atomically:

1. **Task 1: Define CubeCL capability snapshot and reason taxonomy contracts** - `564da55` (feat)
2. **Task 2: Implement wgpu bootstrap preflight and dependency wiring** - `329e714` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/capability.rs` - WgpuCapabilitySnapshot, CapabilityReason, WgpuPreflightReport, capability_fingerprint (342 lines)
- `crates/cintx-cubecl/src/runtime_bootstrap.rs` - bootstrap_wgpu_runtime, parse_selector, selector_to_wgpu_device, collect_feature_names, collect_limit_entries (389 lines)
- `crates/cintx-cubecl/Cargo.toml` - Added cubecl wgpu feature + cubecl-wgpu/cubecl-runtime/wgpu deps
- `crates/cintx-cubecl/src/lib.rs` - Exported capability and runtime_bootstrap modules

## Decisions Made

- FNV-1a 64-bit hash over sorted feature/limit lists plus adapter identity fields for deterministic, reproducible capability fingerprints — matches D-04 without external hashing libraries
- Wrap `cubecl::wgpu::init_setup` with `std::panic::catch_unwind` to convert CubeCL panic-based adapter failures into typed `UnsupportedApi` errors per D-02 (avoids Pitfall 1 from research notes)
- Keep selector format simple (`auto`/`discrete:N`/`integrated:N`) aligned with CubeCL's `WgpuDevice` enum variants for clear forward-compatibility
- Collect a named set of `wgpu::Features` flags into the snapshot rather than serializing the raw bitfield to keep fingerprints human-auditable in diagnostics

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Worktree missing Plan 01 changes**
- **Found during:** Initial build (before Task 1)
- **Issue:** Worktree branch `worktree-agent-adc699a6` did not have Plan 01's BackendIntent/BackendCapabilityToken types — the merge from main hadn't propagated to the worktree
- **Fix:** Added `local-main` remote pointing to `/home/chemtech/workspace/cintx` and merged `local-main/temp-reset` into the worktree branch
- **Files modified:** (structural — brought in Plan 01 runtime files)
- **Verification:** `cintx_runtime::BackendIntent` and `BackendKind` became available for import
- **Committed in:** Merge commit (before task commits)

**2. [Rule 1 - Bug] wgpu::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING does not exist**
- **Found during:** Task 2 (initial build)
- **Issue:** The feature flag name was wrong for wgpu 26.0.1; correct name is `STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING`
- **Fix:** Used correct feature flag name from wgpu 26.0.1 API
- **Files modified:** `crates/cintx-cubecl/src/runtime_bootstrap.rs`
- **Verification:** Build succeeds; all tests pass

---

**Total deviations:** 2 auto-fixed (Rule 1 - blocking compile errors)
**Impact on plan:** Both fixes required for compilation. No scope creep.

## Issues Encountered

- The worktree was initialized from the `temp-reset` merge point (95c3cbf) before Plan 01 was committed to the `temp-reset` branch in the main repo, so BackendIntent/BackendCapabilityToken types were absent. Resolved by merging `local-main/temp-reset`.

## Next Phase Readiness

- `bootstrap_wgpu_runtime(intent)` is ready for consumption by the Phase 5 Plan 03 executor rewrite
- `WgpuPreflightReport.fingerprint` provides the `capability_fingerprint` field needed to populate `BackendCapabilityToken.capability_fingerprint` in the real executor path
- `WgpuPreflightReport.snapshot.adapter_name` and `backend_api` provide the remaining `BackendCapabilityToken` fields
- All capability tests pass; selector parsing is deterministic and fail-closed

---
*Phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend*
*Completed: 2026-04-02*
