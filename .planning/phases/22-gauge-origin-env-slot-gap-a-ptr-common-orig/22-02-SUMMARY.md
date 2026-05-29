---
phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
plan: 02
subsystem: oracle
tags: [gauge-origin, env-slot, ptr-common-orig, fixture, round-trip, oracle, fnd-01, d-03]

# Dependency graph
requires:
  - phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
    plan: 01
    provides: "PTR_COMMON_ORIG const, OperatorEnvParams.common_orig field, eval_raw env[1..3] read, validate_common_orig_env_params"
provides:
  - "build_h2o_sto3g_common_orig / build_h2o_sto3g_common_orig_at fixture builders (non-zero env[1..3])"
  - "COMMON_ORIG_FIXTURE_ORIGIN = [0.5,-0.3,0.8] shared fixture constant"
  - "common_orig_roundtrip plain-#[test] proving the Plan-01 slot wiring is live and distinguishable from the [0,0,0] default"
affects: [24-moments, 26-giao, 30-giao]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-slot oracle fixture wrapper modeled on build_h2o_sto3g_f12 (base fixture + one env-slot write)"
    - "D-03: slot verified via public env->OperatorEnvParams round-trip + eval_raw Ok, NOT a vendor byte-identity parity test (no consuming kernel yet)"
    - "Non-zero fixture origin + assert_ne!(Some([0,0,0])) closes the false-pass-from-zero gap (T-22-02-01)"

key-files:
  created:
    - crates/cintx-oracle/tests/common_orig_roundtrip.rs
  modified:
    - crates/cintx-oracle/src/fixtures.rs

key-decisions:
  - "Origin literal [0.5,-0.3,0.8]: non-zero on all three axes so a slot reading the default is impossible to mistake for a populated one (CONTEXT line 103); value is Claude's discretion per D-03/CONTEXT line 48"
  - "No vendor harness stub added: the fixture builder is the deliverable; a wired vendor call is explicitly optional this phase (no consuming kernel to compare against until Phases 24/26)"
  - "Round-trip verified two complementary ways since eval_raw returns RawEvalSummary (no plan, no common_orig field) and the internal plan path is unreachable from cintx-oracle/tests/: (1) eval_raw returns Ok on the non-zero fixture; (2) public env->OperatorEnvParams.common_orig read mirroring raw.rs:604-615 + validator"

requirements-completed: [FND-01]

# Metrics
duration: 4min
completed: 2026-05-29
---

# Phase 22 Plan 02: Gauge-Origin Oracle Fixture + Slot Round-Trip Summary

**Stood up the committed NON-ZERO H2O/STO-3G gauge-origin fixture (`build_h2o_sto3g_common_orig`, env[1..3]=[0.5,-0.3,0.8]) and a plain-`#[test]` round-trip proving the Plan-22-01 PTR_COMMON_ORIG wiring is live — `eval_raw` accepts the non-zero fixture and the public env→`OperatorEnvParams.common_orig` surface reads back `Some([0.5,-0.3,0.8])`, distinguishable from the `[0,0,0]` default (D-03: slot verification only, no consuming kernel yet).**

## Performance

- **Duration:** ~4 min
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- `build_h2o_sto3g_common_orig()` and `build_h2o_sto3g_common_orig_at(origin)` fixture builders in `fixtures.rs`, modeled on the `build_h2o_sto3g_f12` single-slot wrapper; both write `env[PTR_COMMON_ORIG..+3]` (D-03 data infra for Phases 24/26)
- `COMMON_ORIG_FIXTURE_ORIGIN = [0.5,-0.3,0.8]` shared constant — non-zero on all three axes so a populated `common_orig` is provably distinguishable from the `[0,0,0]` default
- `PTR_COMMON_ORIG` added to the `use cintx_compat::raw::{...}` import block in `fixtures.rs`
- `common_orig_roundtrip.rs` plain-`#[test]` (no `#[cfg(has_vendor_libcint)]`, gated `#![cfg(feature = "cpu")]`): non-zero fixture round-trips to `Some(COMMON_ORIG_FIXTURE_ORIGIN)` and `!= Some([0,0,0])`; base fixture reads back `Some([0,0,0])` proving the read is unconditional (D-02); both validate; both `eval_raw` calls return Ok
- 2 tests pass under `--features cpu`

## Task Commits

Each task was committed atomically:

1. **Task 1: Non-zero gauge-origin fixture builder (data infra)** - `ef9f6d8` (feat)
2. **Task 2: Gauge-origin slot raw<->plan round-trip test (D-03)** - `433cc7c` (test)

## Files Created/Modified
- `crates/cintx-oracle/src/fixtures.rs` (modified) - Added `PTR_COMMON_ORIG` to raw imports; added `COMMON_ORIG_FIXTURE_ORIGIN` const + `build_h2o_sto3g_common_orig` / `build_h2o_sto3g_common_orig_at` builders writing `env[PTR_COMMON_ORIG..+3]`
- `crates/cintx-oracle/tests/common_orig_roundtrip.rs` (created) - Plain-`#[test]` round-trip: `eval_raw` Ok on the non-zero fixture + public env→`OperatorEnvParams.common_orig` read (mirroring raw.rs:604-615) + finiteness validator, asserting the non-zero value and `!= [0,0,0]`, plus the base-fixture zero-default read

## Decisions Made
- **Origin literal `[0.5,-0.3,0.8]`:** non-zero on all three axes per CONTEXT line 103; value is Claude's discretion (D-03 / CONTEXT line 48). Closes the false-pass-from-zero gap (T-22-02-01).
- **No vendor harness stub:** the fixture builder is the deliverable; a wired `vendor_*` call is explicitly optional this phase (D-03 — no consuming kernel to compare against until Phases 24/26). Not added.
- **Two-way round-trip observation:** `eval_raw` returns `RawEvalSummary` (no plan, no `common_orig` field) and the internal `prepare_raw_call`/`ExecutionPlan` path is unreachable from `cintx-oracle/tests/`, so the round-trip is verified by (1) `eval_raw` returning Ok on the non-zero fixture (the live env[1..3] read does not error) and (2) the public `env -> OperatorEnvParams.common_orig` read mirroring the internal `eval_raw` read at raw.rs:604-615, followed by the module-path validator.

## Deviations from Plan

None - plan executed exactly as written. The copy-ready test code in the plan matched the real APIs verbatim (`RawApiId::INT1E_OVLP_SPH`, `unsafe eval_raw(...)`, `cintx_core::cintxRsError`, module-path `cintx_runtime::validator::validate_common_orig_env_params`); no symbol-resolution or signature adjustments were needed.

## Issues Encountered
None during planned work. (Tooling note: the first Edit call resolved to the shared checkout path rather than the isolated worktree copy; corrected by editing the worktree copy. No code impact.)

## TDD Gate Compliance
Task 2 is `tdd="true"`. The implementation under test (PTR_COMMON_ORIG slot wiring, `OperatorEnvParams.common_orig`, validator) was delivered by merged Plan 22-01, and the consumed fixture by this plan's Task 1 (commit `ef9f6d8`, RED-providing data). The round-trip `test(...)` commit (`433cc7c`) follows the feature commits it exercises; the two tests passed on first run because the slot wiring they verify already existed (this is a slot-verification test of pre-built wiring, D-03, not a new feature implemented test-first). No false RED occurred — the tests assert observable behavior of the merged slot, exactly as designed.

## Threat Surface Scan
No new security-relevant surface beyond the plan's threat_model. Registered dispositions honored:
- **T-22-02-01** (false-pass from a zero fixture, mitigate): fixture origin is non-zero on all three axes and the non-zero test asserts `!= Some([0,0,0])`, so a slot that silently reads the default cannot pass.
- **T-22-02-02** (OOB on short env, accept): already mitigated upstream in Plan 22-01 (`env.len() >= PTR_COMMON_ORIG + 3` guard); the fixture env is full-length (PTR_ENV_START=20), no new surface.
- **T-22-02-03** (info/auth/injection, accept): test-only data, no network/secret/auth surface.

## Next Phase Readiness
- `build_h2o_sto3g_common_orig` is committed data infrastructure ready for Phase 24 (moments) and Phases 26/30 (GIAO) to point a consuming kernel at; the gated byte-identity parity test lands when that kernel exists (D-03).
- No blockers.

## Self-Check: PASSED

---
*Phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig*
*Completed: 2026-05-29*
