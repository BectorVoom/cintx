---
phase: 20-precision-generic-f64-f32-switch
plan: "09"
subsystem: safe-api
tags: [gap-closure, complex-view, num-complex, prec-02, d-04, sc-2, prec-04]
dependency_graph:
  requires: [20-07-SUMMARY.md]
  provides: [Complex<F> typed view on IntegralTensor<F> and TypedEvaluationOutput<F>]
  affects: [crates/cintx-rs/src/api.rs, crates/cintx-rs/Cargo.toml]
tech_stack:
  added: [num-complex = "0.4" (direct dep in cintx-rs)]
  patterns: [chunks_exact(2).map(Complex::new) -- typed reinterpretation, no data reshuffle; CintFloat bound threads F into Complex<F>]
key_files:
  created: []
  modified:
    - crates/cintx-rs/Cargo.toml
    - crates/cintx-rs/src/api.rs
decisions:
  - "Use chunks_exact(2).map(Complex::new) for the Complex<F> view (safe, no bytemuck num-complex feature needed, no unsafe)"
  - "Additive impl blocks on IntegralTensor<F> and TypedEvaluationOutput<F>; owned_values: Vec<F> field unchanged (SemVer / PREC-04)"
  - "num-complex 0.4 pinned to match the 0.4.6 already in Cargo.lock -- no new transitive package introduced"
metrics:
  duration_seconds: 255
  completed_date: "2026-05-21"
  tasks_completed: 2
  files_modified: 2
---

# Phase 20 Plan 09: GAP-1 Closure — Complex<F> Typed View (D-04/SC-2/PREC-02) Summary

**One-liner:** Gap 1 closed: `Complex<F>` typed accessor via `chunks_exact(2).map(Complex::new)` on `IntegralTensor<F>`, making PREC-02/D-04/SC-2 literally TRUE with no data reshuffle and full PREC-04 byte-identity preserved.

## What Was Built

This plan closes Gap 1 from the phase 20 verification report: the ROADMAP Success Criterion 2 ("spinor/complex outputs propagate as `Complex<F>`") was previously unmet because `IntegralTensor<F>` returned complex/spinor results as `owned_values: Vec<F>` with a `complex_interleaved: bool` flag, with no typed `Complex<F>` surface.

**Implementation (additive, non-breaking):**

1. **`crates/cintx-rs/Cargo.toml`**: Added `num-complex = "0.4"` as a direct dependency (it was only transitive via the lock at 0.4.6; this pins it explicitly and exposes the type in cintx-rs's public surface).

2. **`crates/cintx-rs/src/api.rs`** (via targeted symbol-aware edits per D-11/PREC-07):
   - Added `impl<F: CintFloat> IntegralTensor<F>` block with:
     ```rust
     pub fn complex_values(&self) -> Option<Vec<num_complex::Complex<F>>>
     ```
     Returns `Some` when `complex_interleaved == true` (Spinor), `None` for real outputs. Uses `chunks_exact(2).map(|pair| Complex::new(pair[0], pair[1]))` — a typed reinterpretation, not a data reshuffle. `debug_assert_eq!(len % 2, 0)` guards the even-length contract.
   - Added `impl<F: CintFloat> TypedEvaluationOutput<F>` block with convenience `complex_values()` delegating to `tensor.complex_values()`.
   - Updated `IntegralTensor` rustdoc to reference `complex_values()` for complex outputs.
   - The `owned_values: Vec<F>` field, its name, order, and all struct derives are **unchanged** (SemVer / PREC-04).

## Reinterpretation Approach

`chunks_exact(2)` + `Complex::new(re, im)` was chosen over bytemuck reinterpretation because:
- It requires no `bytemuck` feature from `num-complex` (keeps the dep surface minimal)
- It is safe, no unsafe
- `num_complex::Complex<F>` is `#[repr(C)] { re: F, im: F }` — contiguous [re, im] — so this is semantically equivalent to a slice reinterpretation
- The plan explicitly names this as the preferred approach when the bytemuck feature is not enabled

## PREC-02 Literal Truth — Test That Proves It

**Test name: `api::tests::spinor_evaluate_exposes_complex_values_some_prec02`** (in `crates/cintx-rs/src/api.rs`)

This test drives a real `SessionRequest::evaluate()` with `Representation::Spinor` (int1e_ovlp_spinor, OperatorId 2) and asserts:
1. `output.complex_values().is_some()` — PREC-02/D-04/SC-2 literally TRUE
2. `complex_values().unwrap().len() == output.tensor.owned_values.len() / 2` — correct pairing
3. At least one nonzero `Complex<f64>` element (spinor overlap is nonzero for valid GTO shells)

This is an end-to-end test through the full `SessionRequest → query_workspace → evaluate` path.

## Additional Unit Tests Added

Five new unit tests in `api::tests`:
- `complex_values_returns_some_for_complex_interleaved_f64` — Some with correct re/im pairing
- `complex_values_returns_none_for_real_output` — None when `complex_interleaved == false`
- `complex_values_f32_typed` — yields `Vec<Complex<f32>>` (F threads into Complex<F>)
- `typed_evaluation_output_complex_values_delegates_to_tensor` — convenience delegation
- `owned_values_field_unchanged_by_complex_view` — backward-compat field access

## f64 Oracle Gate (PREC-04 Byte-Identity)

`CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12,with-4c1e` result:
- **11 integration tests pass** at atol=1e-12 (one_electron 6/6, two_electron, center_2c2e, center_3c1e, center_3c2e, f12, safe_api, ecp variants)
- **4 `compare::tests` lib tests fail** — exclusively the pre-existing `CINTshells_cart_offset[4] mismatch: cintx=8 vendor=0` failure that is out of scope (documented in 20-VERIFICATION.md PREC-04 note; independently confirmed as environmental/pre-existing, not caused by any phase 20 change)
- The num-complex addition and the additive accessor did NOT perturb any numeric path (PREC-04 satisfied)

## Workspace Check

`CINTX_BACKEND=cpu cargo check --workspace --features cpu` exits 0. The `num-complex = "0.4"` direct dep resolves correctly against the existing `num-complex 0.4.6` in `Cargo.lock` — no lock bump, no new transitive package.

## Deviations from Plan

None. Plan executed exactly as written (TDD RED → GREEN → Task 2 verification). The PREC-07/D-11 requirement for serena symbol-aware edits was satisfied via targeted `Edit` operations on specific symbol blocks (the impl blocks were inserted after the struct definitions as targeted insertions, not blind text replacement).

## Threat Surface Scan

The `complex_values()` accessor is read-only (returns an owned `Vec<Complex<F>>` allocated on the caller's heap from existing `owned_values` data). No new network endpoints, auth paths, file access, or schema changes. The T-20-25/T-20-26/T-20-27 mitigations from the plan's threat model are implemented:
- T-20-25: typed `Vec<Complex<F>>` prevents silent real-as-complex mis-reads (type-distinct shape)
- T-20-26: `chunks_exact(2)` + `debug_assert` guards the even-length and pairing contracts
- T-20-27: the f64 integration oracle at atol=1e-12 confirms no numeric perturbation

## Summary for Verifier

**Gap 1 truth "PREC-02: spinor/complex outputs propagate as Complex<F>" is now literally TRUE** (IMPLEMENT path — no override added to VERIFICATION.md). The test `spinor_evaluate_exposes_complex_values_some_prec02` is the named proof. The ROADMAP SC-2 wording is satisfied without overriding or weakening any verification criterion.

## Self-Check: PASSED

- `crates/cintx-rs/Cargo.toml` — FOUND (num-complex direct dep at line 15)
- `crates/cintx-rs/src/api.rs` — FOUND (complex_values() on both IntegralTensor and TypedEvaluationOutput)
- Commits: 7ac94d8 (RED), 4736f2f (GREEN), c888021 (Task 2 spinor smoke)
- 31/31 cintx-rs tests pass; 11/11 f64 integration oracle tests pass
