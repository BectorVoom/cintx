---
phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
plan: 01
subsystem: api
tags: [gauge-origin, env-slot, ptr-common-orig, validator, raw-api, safe-api, giao, moments]

# Dependency graph
requires:
  - phase: 21-plain-coulomb-gradient-integral-families
    provides: "PTR_RINV_ORIG env-slot end-to-end precedent (const + plan field + validator + ExecutionOptions + builder setter + api propagation)"
provides:
  - "PTR_COMMON_ORIG = 1 const in cintx-compat raw.rs"
  - "OperatorEnvParams.common_orig plan field (Option<[f64;3]>, defaults None)"
  - "validate_common_orig_env_params finiteness validator (D-01), module-path reachable"
  - "operator-agnostic eval_raw env[1..3] read under bounds guard (D-02)"
  - "ExecutionOptions.common_orig safe-API option"
  - "with_common_origin builder setter + api.rs options->plan propagation"
affects: [24-moments, 26-giao, 30-giao]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gauge-origin env slot plumbed field-for-field on the Phase-21 PTR_RINV_ORIG precedent"
    - "D-01: finiteness-only validator (None is valid, defaults to [0,0,0]) — diverges from rinv presence check"
    - "D-02: operator-agnostic env-read (no operator-name guard) — read unconditionally under bounds guard"
    - "Validator family reached by module path cintx_runtime::validator::, NOT crate-root re-export"

key-files:
  created: []
  modified:
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-runtime/src/options.rs
    - crates/cintx-rs/src/builder.rs
    - crates/cintx-rs/src/api.rs

key-decisions:
  - "D-01: validate_common_orig_env_params is a finiteness check, not a presence check — None is valid (defaults to [0,0,0])"
  - "D-02: eval_raw reads env[1..3] operator-agnostically (no is_iprinv-style guard); only a bounds guard prevents OOB"
  - "validate_common_orig_env_params reachable by module path only (not added to lib.rs re-export), matching the rinv/grids/f12 convention"

patterns-established:
  - "Pattern: new gauge/origin env slots follow the rinv 6-site plumbing (const, plan field, validator, ExecutionOptions, setter, api propagation)"
  - "Pattern: finiteness validators prefix unused operator_name with _ to keep signature parity with the validator family"

requirements-completed: [FND-01]

# Metrics
duration: 6min
completed: 2026-05-29
---

# Phase 22 Plan 01: PTR_COMMON_ORIG Gauge-Origin Env Slot Summary

**PTR_COMMON_ORIG (env[1..3]) gauge-origin slot plumbed end-to-end through both the raw/compat path (const + operator-agnostic env-read + finiteness validator) and the safe-API path (ExecutionOptions.common_orig + with_common_origin setter + plan propagation), on the Phase-21 PTR_RINV_ORIG precedent with D-01 finiteness and D-02 operator-agnostic divergences.**

## Performance

- **Duration:** ~6 min
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- `PTR_COMMON_ORIG = 1` const and operator-agnostic, bounds-guarded `env[1..3]` read in `eval_raw`, populating `plan.operator_env_params.common_orig`
- `validate_common_orig_env_params` finiteness validator (D-01: None valid, NaN/inf in `Some` rejected with `InvalidEnvParam{param:"PTR_COMMON_ORIG"}`); reachable by module path `cintx_runtime::validator::validate_common_orig_env_params`, NOT added to the crate-root re-export
- Safe-API path: `ExecutionOptions.common_orig`, `with_common_origin([x,y,z])` builder setter, and the `api.rs` options->plan propagation block
- 4 D-01 unit tests pass (default None, accepts None, accepts finite Some, rejects non-finite)

## Task Commits

Each task was committed atomically:

1. **Task 2: Finiteness validator (D-01) + D-01 unit tests + planner field** - `d4abef8` (feat)
2. **Task 1: PTR_COMMON_ORIG const + operator-agnostic env[1..3] read** - `e6ef71b` (feat)
3. **Task 3: ExecutionOptions.common_orig + with_common_origin setter + api.rs propagation** - `ede8597` (feat)

_Note: Tasks were committed in the order 2 → 1 → 3. Task 2's validator and the `OperatorEnvParams.common_orig` plan field are interdependent (the validator reads the field; Task 1's eval_raw call site references the validator by module path), so the field+validator+tests were committed first to keep each commit independently compilable._

## Files Created/Modified
- `crates/cintx-compat/src/raw.rs` - Added `PTR_COMMON_ORIG = 1` const; operator-agnostic env[1..3] read in eval_raw under `env.len() >= PTR_COMMON_ORIG + 3` bounds guard; unconditional finiteness validate call via module path
- `crates/cintx-runtime/src/planner.rs` - Added `OperatorEnvParams.common_orig: Option<[f64;3]>` field (defaults None via derive(Default))
- `crates/cintx-runtime/src/validator.rs` - Added `validate_common_orig_env_params` (finiteness-only, operator-agnostic) + 4 D-01 unit tests
- `crates/cintx-runtime/src/options.rs` - Added `ExecutionOptions.common_orig: Option<[f64;3]>` field
- `crates/cintx-rs/src/builder.rs` - Added `with_common_origin([f64;3])` setter (mirrors `with_rinv_origin`)
- `crates/cintx-rs/src/api.rs` - Added `common_orig` options->plan propagation block after the rinv block

## Decisions Made
- **D-01 (finiteness, not presence):** `validate_common_orig_env_params` treats `None` as valid (gauge origin defaults to `[0,0,0]`; libcint reads unset env as zero). Only a `Some([..])` containing a non-finite component is rejected. This deliberately diverges from `validate_rinv_orig_env_params`, which rejects `None` for iprinv operators.
- **D-02 (operator-agnostic):** `eval_raw` reads env[1..3] unconditionally (no `is_iprinv_family_symbol`-style operator-name guard). No dispatchable consumer exists yet; moments/GIAO add their own dispatch in Phases 24/26. Only the bounds guard prevents OOB indexing (T-22-01-01).
- **Module-path reachability:** Kept the validator out of the `lib.rs` `pub use validator::{...}` re-export, matching the established rinv/grids/f12 convention. Reached as `cintx_runtime::validator::validate_common_orig_env_params`.
- **Unused `operator_name` parameter:** Prefixed with `_` to silence the lint while preserving signature parity with the validator family.

## Deviations from Plan

None - plan executed exactly as written. Task ordering (2 before 1) was the plan-anticipated path: the plan's Task 1 note explicitly says "If Task 2's validator is not yet compiled when you write this, write Task 2 first — same plan, same wave."

## Issues Encountered
None during planned work. (Tooling note: initial Read/Edit calls resolved to the shared checkout path rather than the isolated worktree copy; corrected by reading and editing the worktree copies. No code impact.)

## Threat Surface Scan
No new security-relevant surface beyond the plan's threat_model. The two registered mitigations are implemented:
- **T-22-01-01** (OOB-panic DoS): `env.len() >= PTR_COMMON_ORIG + 3` bounds guard before indexing (Task 1).
- **T-22-01-02** (NaN/inf tampering): `validate_common_orig_env_params` finiteness check rejects NaN/±inf before plan consumption (Task 2).

## Next Phase Readiness
- FND-01 gauge-origin slot is delivered and available for Phase 24 (moments) and Phases 26/30 (GIAO) to consume.
- Plan 22-02's round-trip test can import `validate_common_orig_env_params` by the module path as designed.
- No blockers.

## Self-Check: PASSED

---
*Phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig*
*Completed: 2026-05-29*
