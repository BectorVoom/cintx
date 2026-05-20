---
phase: 20-precision-generic-f64-f32-switch
verified: 2026-05-21T10:00:00Z
status: gaps_found
score: 5/7
overrides_applied: 0
gaps:
  - truth: "PREC-02: spinor/complex outputs propagate as Complex<F>"
    status: partial
    reason: "ROADMAP SC-2 states 'spinor/complex outputs propagate as Complex<F>'. The implementation uses Vec<F> with complex_interleaved: bool (interleaved layout). The plan 07 must_haves do not mention Complex<F> and intentionally chose a different approach — but the ROADMAP success criterion wording is not met literally. The spinor transform (c2spinor.rs) is generic over F: CintFloat and does thread F correctly, but the output type is Vec<F> with an interleaved flag, not num_complex::Complex<F>."
    artifacts:
      - path: "crates/cintx-rs/src/api.rs"
        issue: "IntegralTensor<F>.owned_values: Vec<F> with complex_interleaved: bool flag, not Complex<F>"
    missing:
      - "Either update ROADMAP SC-2 to reflect that spinor outputs use Vec<F> with complex_interleaved (the existing design choice), or implement Complex<F> wrapping. If the interleaved approach is the accepted design, an override should be added."
  - truth: "PREC-05: f32 path has a separate oracle gate verified against libcint — just not byte-identical"
    status: partial
    reason: "The f32 oracle gate exists (f32_parity.rs, 11 vendor tests + 1 smoke) and passes for all tested base-family scalar cases. However, code review CR-01 and CR-02 document a correctness defect in the f32 staging buffer length contract: after bytemuck::cast_slice_mut the staging has chunk_len*2 f32 lanes, but copy_len and not0 scan derive from staging.len() (== chunk_len*2), not the true output element count. For multi-component/spinor/derivative outputs where staging_elements > chunk_len, this writes past the meaningful region and returns wrong f32 values silently. The f32 oracle tests do NOT cover these cases. The f12 F32 path also sizes a temporary f64 buffer to the doubled f32 lane count (CR-02). The gate passes for tested families but the f32 path is not fully verified — specifically, multi-component and f12 derivative cases are untested and likely wrong."
    artifacts:
      - path: "crates/cintx-cubecl/src/kernels/one_electron.rs"
        issue: "CR-01: copy_len = staging.len().min(cart_buf.len()) uses staging.len() == chunk_len*2 on f32 path, not the true output element count"
      - path: "crates/cintx-cubecl/src/kernels/f12.rs"
        issue: "CR-02: staging_f64 allocated at staging.len() (doubled for f32); sub-kernel receives doubled buffer; not0 scan may count stale lanes"
    missing:
      - "Add CR-01 fix: pass true output element count to typed inner, bound copy + not0 scan to it, return BufferTooSmall if f32 view cannot hold output"
      - "Add CR-02 fix: size staging_f64 to true out_elems (not staging.len()); bound readback and not0 to out_elems"
      - "Add f32 oracle tests for multi-component and f12 derivative operators once fixes are in place"
human_verification: []
---

# Phase 20: Generic Float Precision (f64/f32 Switch) — Verification Report

**Phase Goal:** cintx parameterizes its compute path over a generic float type `F: Float` so callers evaluate integrals in f64 (default, byte-identity) or f32 (loose-tolerance, unlocks adapters lacking `SHADER_F64`). Precision is chosen at the call site via a method-level generic; `evaluate()` continues to mean f64 and every existing call site compiles unchanged.
**Verified:** 2026-05-21T10:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PREC-01: generic `F: Float` threaded through full compute path | VERIFIED | boys/rys/obara_saika/stg/pdata all have `<F: Float>` device fns and `<F: CintFloat>` host wrappers; all kernel launchers (one_electron/two_electron/center_2c2e/3c1e/3c2e/4c1e/f12) dispatch on plan.precision; c2s/c2spinor transforms are generic over F; f64 tables FROZEN as `[f64; N]` injected via `from_f64_lossy`; f64 monomorphization default preserved |
| 2 | PREC-02: evaluate_generic::<F>() / evaluate() f64 shim / TypedEvaluationOutput<F = f64> / call sites unchanged | PARTIAL | evaluate_generic::<F: CintFloat + Pod>() exists; evaluate() delegates to evaluate_generic::<f64>(); TypedEvaluationOutput<F=f64> and IntegralTensor<F=f64> with owned_values: Vec<F> and f64 defaults exist; plan.precision = F::PRECISION wiring is correct. DEVIATION: ROADMAP SC-2 states "spinor/complex outputs propagate as Complex<F>" — implementation uses Vec<F> with complex_interleaved: bool (interleaved layout, not num_complex::Complex<F>). The plan 07 must_haves do not mention Complex<F> and the plan intentionally chose the existing interleaved approach. Core F-threading works; Complex<F> wording is unmet. |
| 3 | PREC-03: raw compat env/atm/bas + C ABI remain f64 | VERIFIED | cintx-compat functions use `&[f64]` for env/bas throughout; cintx-capi shim.rs exports `env: *const f64`, `out: *mut f64`; no generic F in any compat/capi signature |
| 4 | PREC-04: f64 path byte-identical; existing oracle gates/manifest locks/tests pass unchanged | VERIFIED (with note) | Integration oracle: one_electron 6/6, two_electron 2/2, center_2c2e/3c1e/3c2e, f12 15/15, safe_api, ecp — all pass at atol=1e-12. NOTE: 4 compare::tests lib unit tests FAIL on `CINTshells_cart_offset[4] mismatch: cintx=8 vendor=0` — independently confirmed as pre-existing (git diff shows phase 20 did not touch cart_offset code; the identical failure reproduces at pre-phase-20 commit 8997703 under vendor build; tracked in .planning/todos/pending/oracle-cart-offset-vendor-zero.md). PREC-04 integration oracle is satisfied; the lib unit test failure is environmental/pre-existing. |
| 5 | PREC-05: f32 path has a separate oracle gate at ~1e-4 rtol verified against libcint | PARTIAL | f32_parity.rs exists with 11 vendor-gated tests covering 1e/2c2e/3c1e/3c2e/2e base families — all pass. F32_UNIFIED_RTOL=1e-4; f32_tolerance_for_family() exists in compare.rs. BLOCKER: CR-01 documents an unsound length contract — on the F32 arm, copy_len and not0 scan use staging.len() (== chunk_len*2 after bytemuck cast) instead of the true output element count. For multi-component/spinor/derivative outputs where staging_elements > chunk_len, wrong f32 values are returned silently. CR-02 is f12-specific: staging_f64 is sized to the doubled f32 lane count. These cases are NOT covered by f32_parity.rs tests. The gate passes for tested base-family scalar cases but the f32 path is not fully verified. |
| 6 | PREC-06: f32 path does NOT gate on SHADER_F64 | VERIFIED | executor.rs check_capability() returns Ok(()) immediately for PrecisionKind::F32 before any check_shader_f64_in_features call; check_f64_capability is unchanged for the F64 path |
| 7 | PREC-07: refactor via serena MCP symbol-aware tools, not blind text replacement | VERIFIED (process gate) | .serena/ directory confirmed present; SUMMARY 01 records serena initial_instructions confirmation and D-11 mandate active; subsequent plan SUMMARYs confirm serena tool usage; code structure (precise targeted edits, FROZEN f64 tables unmodified, env/atm/bas untouched) is consistent with symbol-aware editing. PREC-07 is a process gate — not verifiable from code alone beyond evidence of correct outcomes. |

**Score:** 5/7 truths fully verified (2 partial; both contribute to gaps_found)

### Deferred Items

None. No gaps identified in later milestone phases.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-core/src/precision.rs` | CintFloat sealed trait + PrecisionKind enum | VERIFIED | pub trait CintFloat with sealed::Sealed, from_f64_lossy, PRECISION const; PrecisionKind { F64, F32 } with #[default] F64; both re-exported from cintx-core root |
| `crates/cintx-cubecl/tests/bytemuck_staging_cast_spike.rs` | A5 bytemuck cast soundness proof | VERIFIED | 5 assertions pass (f64 identity, u8→f32 write-read, f64→2×f32 lanes, alignment, size constants) |
| `crates/cintx-runtime/src/planner.rs` | ExecutionPlan.precision: PrecisionKind field | VERIFIED | pub precision: PrecisionKind field present, default via PrecisionKind::default() |
| `crates/cintx-cubecl/src/math/boys.rs` | Generic boys_gamma_inc<F: Float> + boys_gamma_inc_host<F: CintFloat> | VERIFIED | Both generic signatures present; SQRTPIE4/TURNOVER_POINT remain [f64; N] |
| `crates/cintx-cubecl/src/math/rys.rs` | Generic rys_root1-5<F: Float> + rys_root*_host<F: CintFloat> | VERIFIED | All 14 #[cube] functions generic over F: Float; 5 host wrappers generic over F: CintFloat |
| `crates/cintx-cubecl/src/math/obara_saika.rs` | Generic VRR/HRR steps | VERIFIED | vrr_step<F: Float>, hrr_step<F: Float>, host wrappers <F: CintFloat> |
| `crates/cintx-cubecl/src/math/stg.rs` | stg_roots_host<F: CintFloat> | VERIFIED | stg_roots_host<F: CintFloat> present; no #[cube] fn needed (noted in module doc) |
| `crates/cintx-cubecl/src/math/pdata.rs` | compute_pdata<F: Float> + compute_pdata_host<F: CintFloat> | VERIFIED | Both generic signatures present |
| `crates/cintx-cubecl/src/transform/c2s.rs` | cart_to_sph_*<F: CintFloat> | VERIFIED | All cart_to_sph variants generic over F: CintFloat; coefficient tables FROZEN f64 |
| `crates/cintx-cubecl/src/transform/c2spinor.rs` | cart_to_spinor_*<F: CintFloat> | VERIFIED | All spinor transform functions generic over F: CintFloat |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | Precision-dispatched launcher with typed inner | VERIFIED (CR-01 warning) | launch_one_electron dispatches on plan.precision to launch_one_electron_typed::<F>; bytemuck cast present. CR-01 staging length contract defect noted. |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | Precision-dispatched launcher | VERIFIED (CR-01 warning) | Same pattern as one_electron; CR-01 applies |
| `crates/cintx-cubecl/src/kernels/center_2c2e.rs` | Precision-dispatched launcher | VERIFIED (CR-01 warning) | Same pattern; CR-01 applies |
| `crates/cintx-cubecl/src/kernels/center_3c1e.rs` | Precision-dispatched launcher | VERIFIED (CR-01 warning) | Same pattern; CR-01 applies |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` | Precision-dispatched launcher | VERIFIED (CR-01 warning) | Same pattern; CR-01 applies |
| `crates/cintx-cubecl/src/kernels/center_4c1e.rs` | Precision-dispatched launcher | VERIFIED (CR-01 warning) | Same pattern; CR-01 applies |
| `crates/cintx-cubecl/src/kernels/f12.rs` | Precision-dispatched f12 launcher | VERIFIED (CR-01+CR-02 warning) | Dispatches on plan.precision; bytemuck cast present. CR-02 doubles the severity: staging_f64 sized to staging.len() not out_elems. |
| `crates/cintx-cubecl/src/executor.rs` | check_capability bypassing SHADER_F64 for f32 | VERIFIED | check_capability returns Ok() early for PrecisionKind::F32; check_f64_capability unchanged |
| `crates/cintx-runtime/src/options.rs` | ExecutionOptions.precision: PrecisionKind field | VERIFIED | precision: PrecisionKind field present at line 131 |
| `crates/cintx-rs/src/api.rs` | evaluate_generic::<F>() + evaluate() shim + TypedEvaluationOutput<F=f64> | VERIFIED (PREC-02 note) | All three present; see PREC-02 note re Complex<F> deviation |
| `crates/cintx-oracle/src/compare.rs` | f32_tolerance_for_family + F32 tolerance constants | VERIFIED | F32_UNIFIED_RTOL=1e-4, F32_UNIFIED_ATOL=1e-7, f32_tolerance_for_family() present; f64 UNIFIED_ATOL=1e-12 FROZEN |
| `crates/cintx-oracle/tests/f32_parity.rs` | Separate f32 oracle gate driving evaluate_generic::<f32>() | VERIFIED (PREC-05 note) | 11 vendor-gated tests + 1 unconditional smoke; passes for base families. CR-01/CR-02 indicate untested multi-component/f12 derivative cases may produce wrong results. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `cintx-rs/src/api.rs evaluate_generic::<F>` | `ExecutionOptions.precision` | `plan.precision = F::PRECISION` | WIRED | F::PRECISION const maps f64→F64, f32→F32; set immediately after ExecutionPlan::new() |
| `cintx-rs/src/api.rs evaluate()` | `evaluate_generic::<f64>()` | thin shim delegation | WIRED | `pub fn evaluate(self) -> Result<TypedEvaluationOutput<f64>, FacadeError> { self.evaluate_generic::<f64>() }` |
| `cintx-cubecl/src/executor.rs check_capability` | `plan.precision` | early Ok for F32 | WIRED | `if plan.precision == PrecisionKind::F32 { return Ok(()); }` before check_f64_capability |
| `cintx-cubecl/src/kernels/one_electron.rs` | `boys_gamma_inc::<F>` | boys_gamma_inc::<F> call in typed inner | WIRED | Grep confirms `boys_gamma_inc::<F>` in typed kernel body |
| `cintx-oracle/tests/f32_parity.rs` | `cintx_rs SessionRequest::evaluate_generic::<f32>` | evaluate_generic::<f32>() call | WIRED | Confirmed at lines 125, 168, 198, 942 |
| `cintx-oracle/tests/f32_parity.rs` | `compare::f32_tolerance_for_family` | tolerance lookup | WIRED | `use cintx_oracle::compare::f32_tolerance_for_family` present |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `cintx-rs/src/api.rs TypedEvaluationOutput<F>` | owned_values: Vec<F> | executor.execute() writes into chunk_staging (Vec<f64>); read back via bytemuck::cast_slice::<f64,F> | Yes — kernel writes real integral values; oracle tests confirm nonzero output | FLOWING |
| `cintx-oracle/tests/f32_parity.rs` | f32_matrix: Vec<f32> | collect_safe_api_matrix_f32() calls evaluate_generic::<f32>() | Yes for tested base families; UNCERTAIN for multi-component/spinor/f12 derivative | FLOWING (base families) / UNCERTAIN (edge cases per CR-01/CR-02) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CintFloat/PrecisionKind unit tests | `cargo test -p cintx-core precision` | 4/4 green (SUMMARY 01) | PASS |
| A5 bytemuck spike | `cargo test -p cintx-cubecl --test bytemuck_staging_cast_spike --features cpu` | 5/5 green (SUMMARY 01) | PASS |
| f64 workspace check | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | exit 0 (multiple SUMMARYs) | PASS |
| f32 parity oracle (base families) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test f32_parity` | 11/11 pass (SUMMARY 08, orchestrator confirmation) | PASS |
| f64 integration oracle (all families) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12,with-4c1e` | All integration tests pass (orchestrator confirmation) | PASS |

### Probe Execution

No explicit probes declared in phase PLAN files. Behavioral spot-checks above serve as the verification gate.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PREC-01 | Plans 01, 02, 03, 04, 05 | F: Float through full compute path | SATISFIED | All math modules and kernel launchers verified generic |
| PREC-02 | Plans 04, 05, 07 | evaluate_generic::<F>() method-level generic | PARTIALLY SATISFIED | Core evaluate_generic::<F>() works; Complex<F> spinor wording in ROADMAP SC-2 unmet (implementation uses Vec<F> + interleaved flag) |
| PREC-03 | Plan 06 | Raw compat env/atm/bas + C ABI stay f64 | SATISFIED | cintx-compat and cintx-capi confirmed f64 throughout |
| PREC-04 | Plans 07, 08 | f64 byte-identical; oracle gates unchanged | SATISFIED | Integration oracle all pass; compare::tests lib failures are pre-existing environmental |
| PREC-05 | Plan 08 | f32 oracle gate at ~1e-4 rtol | PARTIALLY SATISFIED | Gate exists and passes for base family scalar cases; CR-01/CR-02 defects leave multi-component and f12 derivative cases unverified and potentially incorrect |
| PREC-06 | Plan 06 | f32 path bypasses SHADER_F64 | SATISFIED | check_capability() early-returns Ok for F32 |
| PREC-07 | Plans 01-08 | Serena symbol-aware tools, no blind text replacement | SATISFIED (process gate) | .serena/ present; SUMMARY evidence; FROZEN values preserved; env/atm/bas untouched |

**Note:** PREC requirements are defined in ROADMAP.md (derived from 20-CONTEXT.md decisions) and are NOT listed in .planning/REQUIREMENTS.md (which covers v1.2/v1.3 requirements through Phase 15). No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | ~646-648 | CR-01: `copy_len = staging.len().min(cart_buf.len())` uses doubled f32 lane count | WARNING | Wrong f32 values for multi-component/spinor outputs; undetected by current tests |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | ~742-745 | CR-01 pattern repeat | WARNING | Same as above |
| `crates/cintx-cubecl/src/kernels/center_2c2e.rs` | ~423-426 | CR-01 pattern repeat | WARNING | Same as above |
| `crates/cintx-cubecl/src/kernels/center_3c1e.rs` | ~492-495 | CR-01 pattern repeat | WARNING | Same as above |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` | ~500-503 | CR-01 pattern repeat | WARNING | Same as above |
| `crates/cintx-cubecl/src/kernels/center_4c1e.rs` | ~752-755 | CR-01 pattern repeat | WARNING | Same as above |
| `crates/cintx-cubecl/src/kernels/f12.rs` | ~1555-1595 | CR-02: staging_f64 sized to staging.len() (doubled); not0 scans stale lanes | WARNING | Wrong not0 stat + potential stale-lane truncation in f12 f32 derivative path |
| `crates/cintx-cubecl/src/math/pdata.rs` | ~140-176 | WR-03: compute_pdata_host<F> computes exponential in F arithmetic (not f64) | WARNING | Contradicts "all intermediates stay f64" contract; only harmless because kernel callers pass f64 inputs |
| `crates/cintx-cubecl/src/math/boys.rs` | ~141, ~233 | WR-05: convergence tolerance uses f64::EPSILON on f32 path | WARNING | Loop may spin extra iterations; host/device convergence criteria differ |
| Multiple kernels | ~600 etc. | WR-06: nonzero_threshold=1e-18 scans chunk_len*2 lanes on f32 path (combined with CR-01) | WARNING | not0 count inflated by stale upper-half lanes |

No unreferenced TBD/FIXME/XXX debt markers found in phase 20 modified files.

### Human Verification Required

None. All must-haves are either verified or failed programmatically via code inspection and orchestrator-confirmed test results.

### Gaps Summary

Two gaps block the `passed` verdict:

**Gap 1: PREC-02 — Complex<F> spinor output (PARTIAL)**

ROADMAP success criterion 2 states "spinor/complex outputs propagate as `Complex<F>`". The implementation uses `Vec<F>` with `complex_interleaved: bool`. The plan 07 must_haves do not mention `Complex<F>` — the plan intentionally chose the existing interleaved layout. The F-threading through the spinor transform (`c2spinor.rs` is generic over F) is correct for the actual design. This is a wording mismatch between the ROADMAP SC and the executed plan.

**This looks intentional.** The plan 07 explicitly chose `complex_interleaved: bool` over `num_complex::Complex<F>` to preserve the existing memory layout contract. To accept this deviation, add to VERIFICATION.md frontmatter:

```yaml
overrides:
  - must_have: "spinor/complex outputs propagate as Complex<F>"
    reason: "Plan 07 intentionally uses Vec<F> with complex_interleaved: bool (existing interleaved layout) — avoids breaking the existing num_complex-free memory layout; spinor transform c2spinor.rs is correctly generic over F; this satisfies the F-threading intent without introducing a num_complex dependency in the output type"
    accepted_by: "{your name}"
    accepted_at: "{ISO timestamp}"
```

**Gap 2: PREC-05 — f32 oracle gate incomplete for multi-component/spinor/f12 derivative cases (PARTIAL)**

The f32 oracle gate (`f32_parity.rs`) passes for base-family scalar cases (11/11). However, code review CR-01 and CR-02 identify an unsound staging buffer length contract:

- CR-01: On the F32 arm, `copy_len = staging.len().min(cart_buf.len())` where `staging.len() == chunk_len*2` (doubled by the bytemuck cast). For multi-component or spinor outputs where `staging_elements > chunk_len`, this silently produces wrong f32 values. The 7 currently covered families use simple/scalar cases where this does not trigger.
- CR-02: In f12, `staging_f64` is allocated at `staging.len()` (doubled f32 lane count), compounding the length contract violation for f12 derivative cases.

These are not merely theoretical — they are observable for any operator with `component_count > 1` or for f12 derivative operators. The "verified against libcint" claim in PREC-05 holds only for the tested subset.

**Required before PREC-05 can be marked fully satisfied:**
1. Fix CR-01 in all 7 affected kernel files: pass the true output element count, bound copy + not0 scan to it, return `BufferTooSmall` if the f32 view cannot accommodate the output.
2. Fix CR-02 in f12: size `staging_f64` to `out_elems` (not `staging.len()`).
3. Add f32 oracle tests for at least one multi-component operator (e.g., int1e_nuc_sph derivative or a spinor operator) to confirm the fix works.

---

_Verified: 2026-05-21T10:00:00Z_
_Verifier: Claude (gsd-verifier)_
