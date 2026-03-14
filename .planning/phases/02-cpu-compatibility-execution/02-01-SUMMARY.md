---
phase: 02-cpu-compatibility-execution
plan: "01"
subsystem: runtime
tags: [libcint, cpu-backend, execution-plan, routing-contract]
requires:
  - phase: 01-contracts-and-typed-foundations
    provides: typed operator/representation models, workspace-query validation, diagnostics taxonomy
provides:
  - Deterministic vendored libcint CPU linkage for stable-family baseline symbols
  - Canonical execution request and dispatch model shared by safe and raw runtime paths
  - Stable-family routing obligation matrix tests that lock mandatory envelopes for 02-06
affects: [phase-2-routing, safe-runtime, raw-runtime, cpu-backend]
tech-stack:
  added: [cc]
  patterns: [template-driven libcint header generation, normalized execution request contracts, matrix obligation tests]
key-files:
  created: [build.rs, src/runtime/execution_plan.rs, tests/phase2_cpu_backend_routing.rs]
  modified: [Cargo.toml, Cargo.lock, src/runtime/mod.rs, src/lib.rs]
key-decisions:
  - "Compile the upstream libcint CPU source baseline with deterministic ordered inputs."
  - "Generate cint.h and cint_config.h from vendored templates inside build.rs to keep local/CI linkage hermetic."
  - "Use ExecutionRequest::from_safe and ExecutionRequest::from_raw to normalize planner inputs across APIs."
  - "Keep 3c1e spinor in the mandatory stable-family envelope matrix for 02-06 routing."
patterns-established:
  - "Build script diagnostics use [cintx-build:<kind>] typed failure prefixes for missing vendored artifacts."
  - "Stable-family coverage obligations are codified as executable matrix tests before router implementation."
requirements-completed: [EXEC-01]
duration: 7 min
completed: 2026-03-14
---

# Phase 2 Plan 01: CPU Compatibility Execution Foundation Summary

**Deterministic libcint CPU linkage plus a shared execution-request dispatch contract were established, with explicit stable-family routing obligations locked for downstream router work.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-14T06:16:03Z
- **Completed:** 2026-03-14T06:23:34Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Added deterministic CPU backend build/linkage for vendored libcint stable-family sources via `build.rs`.
- Introduced canonical execution request/dispatch structs and re-exported them through runtime and crate root.
- Added executable routing-obligation matrix tests for `1e/2e/2c2e/3c1e/3c2e x cart/sph/spinor`, including mandatory `3c1e` spinor.

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire deterministic CPU backend build and linkage for stable families** - `d47172f` (feat)
2. **Task 2: Introduce canonical execution request model shared by safe/raw layers** - `030228c` (feat)
3. **Task 3: Codify stable-family routing obligations for downstream implementation** - `38ace60` (test)

## Files Created/Modified
- `build.rs` - Compiles vendored libcint CPU sources, generates `cint.h/cint_config.h`, and emits deterministic link metadata.
- `Cargo.toml` - Added `cc` as a build dependency for C compilation.
- `Cargo.lock` - Locked new build dependency versions.
- `src/runtime/execution_plan.rs` - Added shared execution request, memory options, backend, and dispatch structs.
- `src/runtime/mod.rs` - Re-exported execution plan contracts.
- `src/lib.rs` - Re-exported execution request/dispatch types from crate root.
- `tests/phase2_cpu_backend_routing.rs` - Added symbol-link checks, execution-request contract test, and stable-family matrix obligations.

## Decisions Made
- Mirror upstream libcint CMake baseline CPU sources in a deterministic build-order list to avoid local/CI symbol drift.
- Keep header generation in build script (instead of committed generated files) so template drift fails during compile-time.
- Normalize safe/raw execution planning inputs through one struct model to prevent parallel dispatch contract drift.
- Lock all stable-family representation envelopes as required routing targets ahead of 02-06 implementation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added bootstrap phase-2 test target required by task verification command**
- **Found during:** Task 1
- **Issue:** Task 1 verification command references `--test phase2_cpu_backend_routing`, but the target file did not exist.
- **Fix:** Created `tests/phase2_cpu_backend_routing.rs` with `cpu_backend_symbols_link`.
- **Files modified:** `tests/phase2_cpu_backend_routing.rs`
- **Verification:** `cargo test --workspace --test phase2_cpu_backend_routing cpu_backend_symbols_link`
- **Committed in:** `d47172f`

**2. [Rule 3 - Blocking] Explicitly linked the generated CPU static library in integration tests**
- **Found during:** Task 1
- **Issue:** First verification run failed with undefined C symbols because the integration test crate did not link `cint_phase2_cpu` automatically.
- **Fix:** Added `#[link(name = "cint_phase2_cpu", kind = "static")]` and `#[link(name = "m")]` in the test extern block.
- **Files modified:** `tests/phase2_cpu_backend_routing.rs`
- **Verification:** Re-ran task verification successfully after link annotations.
- **Committed in:** `d47172f`

**3. [Rule 3 - Blocking] Replaced `BTreeSet` with `HashSet` in matrix contract test**
- **Found during:** Task 3
- **Issue:** `IntegralFamily`/`Representation` do not implement `Ord`, causing Task 3 test compile failure.
- **Fix:** Switched uniqueness tracking to `HashSet`.
- **Files modified:** `tests/phase2_cpu_backend_routing.rs`
- **Verification:** `cargo test --workspace --test phase2_cpu_backend_routing stable_family_required_matrix_contract`
- **Committed in:** `38ace60`

---

**Total deviations:** 3 auto-fixed (3 blocking)
**Impact on plan:** All deviations were minimal unblockers required to satisfy the prescribed verification commands and did not expand scope beyond EXEC-01 foundations.

## Issues Encountered
- Initial symbol-link verification failed due to missing integration-test link attributes; resolved in-task and verified.
- Matrix uniqueness check initially used `BTreeSet` requiring `Ord`; corrected to `HashSet`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CPU linkage and execution-request contract scaffolding are ready for downstream routing/planner implementation.
- Stable-family routing obligations are now explicit and test-enforced for plan `02-06`.

---
*Phase: 02-cpu-compatibility-execution*
*Completed: 2026-03-14*

## Self-Check: PASSED
- Found `.planning/phases/02-cpu-compatibility-execution/02-01-SUMMARY.md`.
- Verified commits `d47172f`, `030228c`, and `38ace60` in `git log --oneline --all`.
