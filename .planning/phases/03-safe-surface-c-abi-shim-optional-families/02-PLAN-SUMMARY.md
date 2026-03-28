---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 02
subsystem: api
tags: [rust, manifest, resolver, cubecl, feature-gates, compat]
requires:
  - phase: 03-safe-surface-c-abi-shim-optional-families-01
    provides: Top-level feature naming and stable-surface defaults for optional/unstable exposure.
provides:
  - Manifest entries for sph-only with-f12 STG/YP, optional with-4c1e, and unstable source-only symbols.
  - Resolver helpers for profile-aware symbol checks and source-only identification.
  - Compat runtime envelope gates for with-f12 sph-only behavior and Validated4C1E constraints.
  - CubeCL feature-gated center_4c1e launch path with fail-closed validation checks.
affects: [phase-03-plan-03, phase-03-plan-04, compat, cubecl, resolver, feature-matrix-ci]
tech-stack:
  added: []
  patterns: [manifest-driven feature gating, fail-closed optional envelopes, feature-profile-aware raw dispatch]
key-files:
  created: []
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-ops/src/resolver.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-cubecl/src/executor.rs
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-cubecl/src/kernels/center_4c1e.rs
    - crates/cintx-compat/Cargo.toml
    - crates/cintx-cubecl/Cargo.toml
    - crates/cintx-runtime/src/dispatch.rs
key-decisions:
  - "Treat optional-family availability as manifest-profile + runtime-envelope dual gates."
  - "Keep source-only rows manifest-visible but reject them unless unstable-source-api is enabled."
  - "Allow runtime dispatch family 4c1e so validated with-4c1e calls can execute through the shared planner path."
patterns-established:
  - "Profile-gated raw resolution: resolve symbol, enforce compiled_in_profiles against active feature profile, then continue."
  - "Envelope-first rejection: with-f12 and Validated4C1E checks return explicit UnsupportedApi reasons before backend launch."
  - "Feature-matrix crate wiring: package-level with-f12/with-4c1e features propagate to dependent backend crates."
requirements-completed: [OPT-01, OPT-02, OPT-03]
duration: 11m
completed: 2026-03-28
---

# Phase 03 Plan 02: Optional Family and Unstable Source Gates Summary

**Manifest-driven with-f12/with-4c1e/unstable-source gating now enforces sph-only F12 behavior, Validated4C1E boundaries, and deterministic unsupported paths in compat/CubeCL.**

## Performance

- **Duration:** 11m
- **Started:** 2026-03-28T00:12:42Z
- **Completed:** 2026-03-28T00:24:10Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Expanded canonical manifest inventory with sph-only STG/YP optional operators and explicit unstable source-only rows.
- Added resolver helpers/tests for profile-aware symbol availability and source-only classification.
- Enforced optional-family runtime envelopes in compat and enabled feature-gated 4c1e backend launch handling in CubeCL.

## Task Commits

Each task was committed atomically:

1. **Task 1: Expand manifest/resolver metadata for optional and unstable-source families** - `85faa34` (feat)
2. **Task 2: Enforce optional-family and unstable-source envelopes in compat and CubeCL execution** - `2e9e33a` (feat)

## Files Created/Modified
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - Added optional/unstable rows and normalized 4c1e metadata.
- `crates/cintx-ops/src/generated/api_manifest.rs` - Regenerated manifest table for resolver/runtime consumers.
- `crates/cintx-ops/src/generated/api_manifest.csv` - Regenerated auditable symbol matrix with sph-only STG/YP rows.
- `crates/cintx-ops/src/resolver.rs` - Added profile/source helper APIs and regression tests.
- `crates/cintx-compat/src/raw.rs` - Added active-profile checks, with-f12 and Validated4C1E validators, and source-only gating tests.
- `crates/cintx-cubecl/src/executor.rs` - Added validated 4c1e acceptance/rejection checks under feature gating.
- `crates/cintx-cubecl/src/kernels/mod.rs` - Added cfg-gated center_4c1e registry wiring.
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` - Implemented validated 4c1e kernel launch stub with explicit rejection reasons.
- `crates/cintx-compat/Cargo.toml` - Added with-f12/with-4c1e/unstable-source-api crate features.
- `crates/cintx-cubecl/Cargo.toml` - Added with-f12/with-4c1e backend crate features.
- `crates/cintx-runtime/src/dispatch.rs` - Added runtime dispatch family support for 4c1e.

## Decisions Made
- Runtime raw resolution now derives the active profile from Cargo features and blocks symbols absent from `compiled_in_profiles`.
- with-f12 requests outside sph/natural-dims envelope now fail with explicit `UnsupportedApi` mentioning the with-f12 sph envelope.
- Validated4C1E checks now enforce cart/sph, scalar rank, natural dims, `max(l)<=4`, and CPU runtime assumptions before backend launch.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added package feature wiring required by the plan verification matrix**
- **Found during:** Task 2
- **Issue:** `cargo test -p cintx-compat --features with-f12/with-4c1e` required crate-defined features and backend propagation paths that were missing.
- **Fix:** Added feature sections to `cintx-compat` and `cintx-cubecl` Cargo manifests.
- **Files modified:** `crates/cintx-compat/Cargo.toml`, `crates/cintx-cubecl/Cargo.toml`
- **Verification:** Full compat/cubecl feature-matrix test commands passed.
- **Committed in:** `2e9e33a`

**2. [Rule 3 - Blocking] Enabled runtime dispatch family for 4c1e to allow validated execution path**
- **Found during:** Task 2
- **Issue:** Shared planner dispatch rejected `4c1e`, preventing with-4c1e validated calls from reaching CubeCL.
- **Fix:** Added `DispatchFamily::Center4c1e` and manifest-family mapping for `4c1e`.
- **Files modified:** `crates/cintx-runtime/src/dispatch.rs`
- **Verification:** `cargo test -p cintx-compat --lib --features with-4c1e` and `cargo test -p cintx-cubecl --lib --features with-4c1e` passed.
- **Committed in:** `2e9e33a`

**3. [Rule 1 - Bug] Corrected scalar operator component-rank encoding for optional/source manifest rows**
- **Found during:** Task 2 verification
- **Issue:** New rows used `component_rank = "scalar"`, but planner scalar parsing expects empty rank for operator component multiplier.
- **Fix:** Normalized component rank to empty string and regenerated manifest artifacts.
- **Files modified:** `crates/cintx-ops/generated/compiled_manifest.lock.json`, `crates/cintx-ops/src/generated/api_manifest.rs`, `crates/cintx-ops/src/generated/api_manifest.csv`
- **Verification:** compat with-f12/with-4c1e test matrix and ops tests passed.
- **Committed in:** `2e9e33a`

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** Deviations were required for correctness and to satisfy the declared feature-matrix verification commands; scope remained within optional/unstable family gating.

## Issues Encountered
- A temporary regression appeared when 4-shell fixture data was reused by existing 3-shell tests; fixed by keeping legacy fixtures intact and introducing dedicated 4-shell fixture data only for new envelope tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Optional-family manifest and runtime gates are now explicit and test-covered across base/with-f12/with-4c1e slices.
- Safe facade and C ABI work in subsequent Phase 3 plans can rely on resolver profile/source helper contracts and deterministic rejection behavior.

## Known Stubs

None.

---
*Phase: 03-safe-surface-c-abi-shim-optional-families*
*Completed: 2026-03-28*

## Self-Check: PASSED

FOUND: .planning/phases/03-safe-surface-c-abi-shim-optional-families/02-PLAN-SUMMARY.md
FOUND: 85faa34
FOUND: 2e9e33a
