---
phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
reviewed: 2026-05-29T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-rs/src/api.rs
  - crates/cintx-rs/src/builder.rs
  - crates/cintx-runtime/src/options.rs
  - crates/cintx-runtime/src/planner.rs
  - crates/cintx-runtime/src/validator.rs
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/tests/common_orig_roundtrip.rs
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 22: Code Review Report

**Reviewed:** 2026-05-29T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 22 wires the `PTR_COMMON_ORIG` gauge-origin env slot (`env[1..3]`) through the full
stack: a new `PTR_COMMON_ORIG` constant + unconditional env read in `eval_raw`, an
`OperatorEnvParams.common_orig` field, an `ExecutionOptions.common_orig` field, a
`SessionBuilder::with_common_origin` setter, safe-API propagation in `SessionQuery`, a
finiteness-only validator `validate_common_orig_env_params`, and an oracle fixture +
round-trip integration test. Per D-03 no kernel consumes the value yet.

The diff under review changes only ~22 lines of non-test code; it is small, well-bounded,
and follows the established `rinv_orig` precedent closely. The bounds guard correctly
prevents OOB indexing, and the validator's finiteness-only semantics (D-01) are implemented
and tested as specified.

The defects below are not crashes or data corruption in this phase (no consumer exists), but
they are real correctness/robustness gaps that will bite in Phases 24/26 when a kernel starts
reading `common_orig`, plus several documentation accuracy problems. The most important is an
asymmetry: the raw path validates the gauge origin but the safe-API path does NOT, despite the
builder doc comment promising finiteness validation. That promise is currently false on the
safe path.

## Warnings

### WR-01: Safe-API path propagates `common_orig` but never validates it

**File:** `crates/cintx-rs/src/api.rs:197-200`
**Issue:** The safe-API `evaluate_generic` path copies `options.common_orig` into
`plan.operator_env_params.common_orig` but never calls `validate_common_orig_env_params`. The
raw path (`raw.rs:634-637`) does validate. The builder doc comment at `builder.rs:111-114`
explicitly states the value is "finiteness-validated (NaN/inf rejected) by
`validate_common_orig_env_params`", and `options.rs:124-127` repeats the same claim. On the
safe path that guarantee is not met: a caller passing `with_common_origin([f64::NAN, 0.0, 0.0])`
will have the NaN silently threaded into the plan with no error. Today no kernel consumes it so
nothing crashes, but when Phase 24/26 reads `common_orig` the unchecked NaN/inf will propagate
into the integral with no typed error — exactly the "no garbage-origin evaluation" contract the
sibling `rinv_orig`/`f12_zeta` gates were built to enforce. The validate call is cheap and the
function already exists; the omission is the bug.
**Fix:** After the propagation block, validate before building the executor:
```rust
if let Some(origin) = self.request.options().common_orig {
    plan.operator_env_params.common_orig = Some(origin);
}
// Validate finiteness on the safe path too (parity with raw.rs:634-637).
cintx_runtime::validator::validate_common_orig_env_params(
    plan.descriptor.operator_name(),
    &plan.operator_env_params,
)
.map_err(FacadeError::from)?;
```

### WR-02: Gauge origin is not validated on the query/workspace-only path

**File:** `crates/cintx-compat/src/raw.rs:578-584`
**Issue:** When `eval_raw` is called with `out = None` (the workspace-sizing mode), it returns
at line 579 BEFORE the `common_orig` read+validate block at 628-637. `query_workspace_raw`
(549-560) likewise never touches `common_orig`. So a non-finite gauge origin in `env[1..3]`
passes the workspace-query phase cleanly and is only caught on the subsequent `out=Some`
evaluation. This matches the pre-existing `f12_zeta`/`rinv_orig` placement (all after the
early return), so it is internally consistent, but it means the documented "reject before
kernel entry" contract is only honored on the evaluation call, not at query time. A caller
that uses the query result to commit to an allocation/plan before evaluating gets no early
signal. Recommend documenting this as a deliberate limitation or moving the read+validate
ahead of the `out`-is-None early return so query and evaluate agree.
**Fix:** Move the `common_orig` read + `validate_common_orig_env_params` call to just after
`ExecutionPlan::new` and before the `out.is_none()` early return, or add an explicit comment at
line 578 noting that env-param validation is intentionally evaluation-only and query_workspace
does not surface it.

### WR-03: `validate_rinv_orig_env_params` uses a `match` with a single arm where `if let` is correct

**File:** `crates/cintx-runtime/src/validator.rs:186-198`
**Issue:** (Adjacent to the Phase 22 work and exercised by the same validator module.) The
`match params.rinv_orig { None => {...}, _ => {} }` is an awkward single-meaningful-arm match
that clippy flags (`single_match`/`match like if let`). The new `validate_common_orig_env_params`
correctly uses `if let Some(...)`; the rinv sibling did not get the same cleanup, leaving two
inconsistent styles side by side in the file the phase touches. Not a correctness bug, but it
degrades the maintainability the phase otherwise improves, and may break a `-D warnings` clippy
gate.
**Fix:**
```rust
if operator_name.contains("iprinv") && params.rinv_orig.is_none() {
    return Err(cintxRsError::InvalidEnvParam {
        param: "PTR_RINV_ORIG",
        reason: "env[4..6] (PTR_RINV_ORIG) must be set for iprinv operators".to_owned(),
    });
}
Ok(())
```

## Info

### IN-01: Stale line-number references in the round-trip test doc comments

**File:** `crates/cintx-oracle/tests/common_orig_roundtrip.rs:6, 23, 68`
**Issue:** The test repeatedly cites "the internal eval_raw read at raw.rs:604-615" (and
"raw.rs:604-615" again at lines 23 and 68) as the code it mirrors. In the current `raw.rs`,
lines 604-615 are the `f12_zeta` validate call and the start of the `rinv_orig` block; the
actual `common_orig` read it mirrors is at `raw.rs:628-633`. Hardcoded line references rot
quickly and here are already wrong, misdirecting a future reader to the wrong block.
**Fix:** Reference the symbol/marker instead of line numbers, e.g. "mirrors the
`Phase 22 FND-01` `common_orig` read block in `eval_raw`", with no line number.

### IN-02: `common_orig` read in `eval_raw` duplicates the `rinv_orig` read logic verbatim

**File:** `crates/cintx-compat/src/raw.rs:613-633`
**Issue:** The `common_orig` block (628-633) is a near-exact copy of the `rinv_orig` block
(613-619): same bounds guard shape, same three-element destructure, same `Some([x,y,z])`
assignment. A small private helper (e.g. `read_origin_slot(env, start) -> Option<[f64;3]>`)
would remove the duplication and make the next origin slot (Phase 24/26) a one-liner.
**Fix:** Extract `fn read_origin_slot(env: &[f64], start: usize) -> Option<[f64; 3]>` and call
it for both `PTR_RINV_ORIG` and `PTR_COMMON_ORIG`.

### IN-03: Builder lacks a `clear_common_origin` / `clear_rinv_origin` helper for symmetry

**File:** `crates/cintx-rs/src/builder.rs:102-115`
**Issue:** `SessionBuilder` provides `clear_*` helpers for `profile_label`, `memory_limit`, and
`chunk_size`, but `with_rinv_origin` and the new `with_common_origin` have no `clear_*`
counterpart. A caller that built up options via `from_request` cannot un-set a gauge origin
through the builder. Minor API-surface inconsistency; not required by this phase but worth
noting while the setter is fresh.
**Fix:** Add `pub fn clear_common_origin(mut self) -> Self { self.options.common_orig = None; self }`
(and a matching `clear_rinv_origin`) if builder symmetry is desired.

### IN-04: No safe-API/builder test exercises `with_common_origin` propagation

**File:** `crates/cintx-rs/src/builder.rs:217-232` (test module)
**Issue:** `builder.rs` has `builder_f12_zeta_propagates_into_options` verifying `f12_zeta`
lands in `ExecutionOptions`, but no equivalent test was added for `with_common_origin` (or
`with_rinv_origin`). The only `common_orig` coverage is the raw round-trip integration test;
the safe builder->options->plan propagation added in this phase (api.rs:197-200) is untested.
Combined with WR-01 (missing safe-path validation), the safe path is the least-covered part of
the change.
**Fix:** Add a builder test asserting `SessionBuilder::new(...).with_common_origin([0.5,-0.3,0.8])
.build().options().common_orig == Some([0.5,-0.3,0.8])`, and ideally a safe-API test asserting a
non-finite origin is rejected once WR-01 is fixed.

---

_Reviewed: 2026-05-29T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
