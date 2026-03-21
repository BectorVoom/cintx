---
phase: manifest-planner-foundation
plan: 01
subsystem: api
tags: [rust, manifest, resolver, metadata]
# Dependency graph
requires: []
provides:
  - canonical manifest lock living under crates/cintx-ops/generated
  - metadata-driven resolver descriptors with `Cow`-backed enums
affects:
  - manifest-planner-foundation
tech-stack:
  added: []
  patterns: [metadata-driven resolver lookups, Arc-backed domain validations]
key-files:
  created:
    - crates/cintx-ops/generated/api_manifest.rs
    - crates/cintx-ops/generated/api_manifest.csv
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/mod.rs
    - crates/cintx-ops/src/lib.rs
  modified:
    - crates/cintx-core/src/atom.rs
    - crates/cintx-core/src/basis.rs
    - crates/cintx-core/src/env.rs
    - crates/cintx-core/src/error.rs
    - crates/cintx-core/src/lib.rs
    - crates/cintx-core/src/shell.rs
    - crates/cintx-ops/build.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-ops/src/resolver.rs
requirements-completed: [BASE-01, BASE-02, BASE-03]
# Metrics
duration: 18 min
completed: 2026-03-21T06:22:50Z
---

# Phase 01: Manifest Planner Foundation Summary

**Manifest generation landed under `crates/cintx-ops/generated`, resolvers became metadata-driven, and core domain primitives gained rigorous validation guards plus regression tests.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-21T06:04:55Z
- **Completed:** 2026-03-21T06:22:50Z
- **Tasks:** 2
- **Files modified:** 15

## Accomplishments
- Hardened `Atom`, `Shell`, `BasisSet`, and `EnvParams` constructors with proper guards plus regression tests, and re-exported `CoreError`.
- Rebuilt `cintx-ops` build script to read the canonical lock, enforce the four-profile union, compute representation-aware arity, and emit manifest/CSV/resolver metadata.
- Updated the resolver to use `Cow`-backed enums plus metadata-driven descriptors/tests so lookups never rely on raw symbols.

## Task Commits

Each task was committed atomically:

1. **Task 1: Harden domain primitives** - `35d470b` (fix)
2. **Task 2: Regenerate manifest metadata base** - `e7a42f2` (feat)

**Plan metadata:** pending final summary/state commit

## Files Created/Modified
- `crates/cintx-core/src/atom.rs` - tightened validation guards and added atomic-number/dimension tests.
- `crates/cintx-ops/build.rs` - canonical manifest writer now emits accurate arity, `Cow`-wrapped metadata, and CSV output.
- `crates/cintx-ops/src/generated/api_manifest.rs` - new table of manifest entries plus metadata-rich descriptors.
- `crates/cintx-ops/src/resolver.rs` - metadata-driven resolver, `Cow`-backed enums, and regression tests.

## Decisions Made
- Always derive the manifest arity from the family (`1e/2c2e=2`, `3c1e/3c2e=3`, `2e/4c1e=4`) to align with the documented dims contract.
- Represent `FeatureFlag`, `Stability`, and `HelperKind` with `Cow<'static, str>` so generated metadata and runtime parsers can share `'static` data without lifetimes issues.
- Keep the canonical lock in `crates/cintx-ops/generated` and implicitly validate the support matrix before emitting resolver tables.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Domain constructor rigidity**
- **Found during:** Task 1 (core primitives)
- **Issue:** `Atom`, `Shell`, `EnvParams`, and `BasisSet` lacked the guards/tests needed to reject bad metadata, and Clippy demanded `is_empty` helpers.
- **Fix:** Folded nested `if`s into guarded patterns, derived `EnvUnits::Default`, exposed `is_empty` helpers, added regression tests, and re-exported `CoreError`.
- **Files modified:** `crates/cintx-core/src/{atom,basis,env,error,shell,lib}.rs`
- **Verification:** `cargo test -p cintx-core --lib`
- **Committed in:** `35d470b`

**2. [Rule 1 - Bug] Metadata-driven manifest/resolver**
- **Found during:** Task 2 (manifest generation)
- **Issue:** Generated data had incorrect arity, lifetimes, and symbol-dependent lookups which clashed with resolver expectations.
- **Fix:** Updated `build.rs` to compute actual arity, enforce the support matrix, output `Cow`-wrapped enums, and regenerated the canonical manifest/CSV/resolver with extensive metadata plus tests.
- **Files modified:** `crates/cintx-ops/{build.rs,generated/*,src/resolver.rs}`
- **Verification:** `cargo test -p cintx-ops --lib`
- **Committed in:** `e7a42f2`

## Issues Encountered
- `cargo fmt --manifest-path crates/cintx-core/Cargo.toml --all -- --check` and `cargo fmt --manifest-path crates/cintx-ops/Cargo.toml --all -- --check` fail because `rustfmt` cannot resolve the duplicate `cintx-cubecl/src/kernels.{rs,mod.rs}` layout and attempts to reformat the generated manifest table; the commands stop before the generated files can be re-styled.
- Git operations required `require_escalated` permissions because the sandbox enforces `.git/index.lock` creation rules; this only affected staging/committing commands.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Domain primitives and manifest metadata now emit canonical metadata ready for planner execution.
- No additional blockers remain for the next phase; the resolver and manifest locks can be consumed by downstream components.

---
*Phase: manifest-planner-foundation*
*Completed: 2026-03-21*
