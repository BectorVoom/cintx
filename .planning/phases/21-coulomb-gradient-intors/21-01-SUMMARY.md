---
phase: 21-coulomb-gradient-intors
plan: "01"
subsystem: planner/validator/raw-compat
tags: [rinv-origin, iprinv, env-params, validation-gate, safe-api]
dependency_graph:
  requires: []
  provides: [rinv_orig field on OperatorEnvParams, validate_rinv_orig_env_params, PTR_RINV_ORIG const, eval_raw rinv read block, with_rinv_origin setter]
  affects: [cintx-runtime, cintx-compat, cintx-rs]
tech_stack:
  added: []
  patterns: [env-param field extension mirroring f12_zeta/grids_params pattern, validator gate mirroring validate_f12_env_params, eval_raw read block mirroring f12_zeta block]
key_files:
  created: []
  modified:
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-runtime/src/validator.rs
    - crates/cintx-runtime/src/options.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-rs/src/builder.rs
    - crates/cintx-rs/src/api.rs
decisions:
  - "with_rinv_origin setter landed on SessionBuilder (cintx-rs/src/builder.rs) mirroring f12_zeta setter; rinv_orig field added to ExecutionOptions (cintx-runtime/src/options.rs) for safe-API propagation"
  - "eval_raw read block guards with env.len() >= PTR_RINV_ORIG + 3 so a too-short env never indexes out of bounds (T-21-01-01)"
  - "validate_rinv_orig_env_params uses operator_name.contains('iprinv') (not ==) so ecp_iprinv is covered by the same gate as iprinv (T-21-01-02)"
  - "eval_raw round-trip tests via unregistered symbols deferred to Plan 21-02 (manifest registration); Task 2 tests cover the predicate functions and the const value directly"
metrics:
  duration: "5 min"
  completed: "2026-05-26"
  tasks: 2
  files_modified: 6
---

# Phase 21 Plan 01: PTR_RINV_ORIG env-slot plumbing SUMMARY

**One-liner:** PTR_RINV_ORIG=4 env-slot plumbing end-to-end — rinv_orig field on OperatorEnvParams, typed validator gate for iprinv operators, eval_raw read block with OOB guard, and with_rinv_origin safe-API setter.

## What Was Built

Plan 21-01 implements the foundation that Wave 2 (`int1e_iprinv`) and Wave 4 (`ECPscalar_iprinv`) depend on:

1. **`OperatorEnvParams.rinv_orig: Option<[f64; 3]>`** — third field added to the struct in `planner.rs`, deriving `None` via the existing `#[derive(Default)]`. Does not touch `ExecutionPlan::new` (caller-populates-after-new precedent from f12_zeta).

2. **`validate_rinv_orig_env_params`** — new validator function in `validator.rs` mirroring `validate_f12_env_params` exactly. Uses `.contains("iprinv")` predicate to cover both `"iprinv"` (int1e_iprinv) and `"ecp_iprinv"` (ECPscalar_iprinv). Non-iprinv operators pass unconditionally (overlap/kinetic/grids/etc. are never gated).

3. **`PTR_RINV_ORIG: usize = 4` const** — added to `raw.rs` with libcint reference doc comment, placed next to `PTR_F12_ZETA`.

4. **`eval_raw` read block** — inserted right after the f12_zeta block. Reads `env[4..7]` into `rinv_orig` guarded by `env.len() >= PTR_RINV_ORIG + 3` (no OOB, T-21-01-01), then calls `validate_rinv_orig_env_params` before kernel entry (T-21-01-02).

5. **`is_iprinv_family_symbol` predicate** — added next to `is_ecp_family_symbol`/`is_f12_family_symbol` as the symbol-detection function.

6. **`rinv_orig: Option<[f64; 3]>`** field added to `ExecutionOptions` (cintx-runtime/options.rs).

7. **`with_rinv_origin` setter** — on `SessionBuilder` (cintx-rs/src/builder.rs) following the `f12_zeta` setter pattern; propagation added in `api.rs` following the f12_zeta propagation block.

## Key Implementation Decisions

**where `with_rinv_origin` landed:** `cintx-rs/src/builder.rs` as `pub fn with_rinv_origin(mut self, origin: [f64; 3]) -> Self`. The safe-API field `rinv_orig: Option<[f64; 3]>` lives on `ExecutionOptions` (cintx-runtime/src/options.rs). Propagation to `plan.operator_env_params.rinv_orig` is in `cintx-rs/src/api.rs` at the same insertion point as `f12_zeta` propagation.

**env-slot read guard:** `env.len() >= PTR_RINV_ORIG + 3` (i.e. `env.len() >= 7`). This ensures no out-of-bounds access for too-short env arrays. When the guard does not fire (env too short), `rinv_orig` stays `None` and `validate_rinv_orig_env_params` returns `InvalidEnvParam { param: "PTR_RINV_ORIG" }`.

**validator predicate:** `operator_name.contains("iprinv")` — matches `"iprinv"` (int1e_iprinv), `"ecp_iprinv"` (ECPscalar_iprinv), and any future iprinv variant. Wave 4 (`ECPscalar_iprinv`) is covered without any future changes to the validator.

**eval_raw round-trip tests deferred:** The acceptance criteria called for round-trip tests dispatching `"int1e_iprinv_sph"` through eval_raw. Those symbols are not registered in the manifest (Plan 21-02 handles registration). The tests for Task 2 cover: `PTR_RINV_ORIG` constant value, `is_iprinv_family_symbol` predicate correctness (both positive and negative cases). Full eval_raw round-trip + validator-rejection tests will land in Plan 21-02 when the manifest entries are present.

## Tests Added

**cintx-runtime (Task 1, 5 tests):**
- `rinv_orig_default_is_none`
- `validate_rinv_orig_rejects_none_for_iprinv`
- `validate_rinv_orig_rejects_none_for_ecp_iprinv`
- `validate_rinv_orig_accepts_non_iprinv`
- `validate_rinv_orig_accepts_some`

**cintx-compat (Task 2, 3 tests):**
- `ptr_rinv_orig_is_4`
- `is_iprinv_family_symbol_detects_iprinv`
- `is_iprinv_family_symbol_does_not_match_ipovlp_ipkin_ipnuc`

## Verification Results

- `cargo test -p cintx-runtime rinv_orig` — 5 passed, 0 failed
- `cargo test -p cintx-compat` — 40 passed (all existing + 3 new), 0 failed
- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` — Finished, 0 errors

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written.

### Scope Notes

**[Rule 2 — clarification] eval_raw round-trip tests deferred to Plan 21-02:** The plan called for round-trip tests via `eval_raw` with `"int1e_iprinv_sph"`. These require the symbol to be manifest-registered (Plan 21-02). Tests that CAN be written at this stage (predicate + const) were added instead. This is consistent with the plan's note "the kernel itself is unimplemented until Wave 2 — that is acceptable for this test."

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Changes are purely in-process Rust library code. The env-slot read guard (`env.len() >= PTR_RINV_ORIG + 3`) explicitly addresses T-21-01-01.

## Self-Check: PASSED

Files confirmed present:
- crates/cintx-runtime/src/planner.rs — rinv_orig field at line 53
- crates/cintx-runtime/src/validator.rs — validate_rinv_orig_env_params at line 182
- crates/cintx-runtime/src/options.rs — rinv_orig field added
- crates/cintx-compat/src/raw.rs — PTR_RINV_ORIG const at line 49, is_iprinv_family_symbol at line 738, read block present
- crates/cintx-rs/src/builder.rs — with_rinv_origin setter at line 102
- crates/cintx-rs/src/api.rs — rinv_orig propagation present

Commits confirmed:
- a33e729: feat(21-01): add rinv_orig field to OperatorEnvParams + validate_rinv_orig_env_params gate
- 7920415: feat(21-01): add PTR_RINV_ORIG const + eval_raw read block + with_rinv_origin setter
