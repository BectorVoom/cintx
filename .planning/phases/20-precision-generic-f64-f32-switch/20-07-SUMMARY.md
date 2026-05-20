---
phase: 20-precision-generic-f64-f32-switch
plan: 07
subsystem: safe-facade-generic-evaluate
tags: [precision, CintFloat, evaluate_generic, TypedEvaluationOutput, IntegralTensor, bytemuck, wave6, PREC-02, PREC-04, PREC-07]
dependency_graph:
  requires: [20-01 (CintFloat sealed trait + PrecisionKind), 20-06 (ExecutionOptions.precision, plan.precision threading)]
  provides: [evaluate_generic::<F: CintFloat>() method-level generic, TypedEvaluationOutput<F = f64> generic struct, IntegralTensor<F = f64> generic struct, CintFloat::PRECISION const, evaluate() f64 shim]
  affects: [crates/cintx-core/src/precision.rs, crates/cintx-rs/src/api.rs, crates/cintx-rs/Cargo.toml]
tech_stack:
  added: [bytemuck = { version = "1", features = ["derive"] } in cintx-rs]
  patterns: [method-level generic with f64 default shim (D-03/D-12), CintFloat::PRECISION const for zero-cost F->PrecisionKind mapping, bytemuck::cast_slice::<f64,F> for read-back of kernel output, OOM-safe fallible-alloc + no-partial-writes preserved]
key_files:
  created: []
  modified:
    - crates/cintx-core/src/precision.rs (const PRECISION: PrecisionKind added to CintFloat trait and impls)
    - crates/cintx-rs/src/api.rs (IntegralTensor<F=f64>, TypedEvaluationOutput<F=f64>, evaluate_generic::<F>, evaluate() shim)
    - crates/cintx-rs/Cargo.toml (bytemuck dep added)
decisions:
  - "f64 delegation mechanism: thin non-generic evaluate() shim calls evaluate_generic::<f64>() (D-03/D-12); every existing call site compiles unchanged without a turbofish"
  - "CintFloat::PRECISION: const associated constant F64 for f64, F32 for f32; zero-cost mapping at compile time; added to the sealed trait so external implementors cannot exist"
  - "Staging buffer pattern: chunk_staging always Vec<f64> (kernel dispatchers expect &mut [f64] — frozen interface); output read back via bytemuck::cast_slice::<f64, F> (f64: identity, f32: yields 2*N lanes, take first N)"
  - "F: CintFloat + bytemuck::Pod bound on evaluate_generic — bytemuck::Pod is satisfied by both f32 and f64 (proven Plan 01 A5 spike)"
  - "OOM-safe fallible alloc contract (try_reserve_exact + HostAllocationFailed) preserved verbatim; only float type parameterized"
metrics:
  duration: "~9 min"
  completed: "2026-05-20"
  tasks_completed: 2
  files_changed: 4
---

# Phase 20 Plan 07: Generic evaluate::<F: CintFloat>() Safe API Surface Summary

Exposed the precision choice at the public safe API by making `SessionQuery::evaluate_generic::<F>()` a method-level generic returning `TypedEvaluationOutput<F>`, while keeping the unparameterized `evaluate()` byte-identical via a thin f64 shim. Both output structs (`IntegralTensor`, `TypedEvaluationOutput`) are now generic over `F = f64` so existing call sites compile unchanged.

## Tasks Completed

| Task | Name | Commit (RED) | Commit (GREEN/impl) | Key Files |
|------|------|--------------|---------------------|-----------|
| 1 | Make TypedEvaluationOutput and IntegralTensor generic over F = f64 | — (combined) | 72ba738 | `crates/cintx-rs/src/api.rs` |
| 2 | Make evaluate a method-level generic + f64 delegation + CintFloat::PRECISION | — (combined) | 4199fe8 | `crates/cintx-core/src/precision.rs`, `crates/cintx-rs/src/api.rs`, `crates/cintx-rs/Cargo.toml` |

## Task 1: Generic Output Structs

### What Was Changed

`IntegralTensor` and `TypedEvaluationOutput` now carry a type parameter `F = f64`:

```rust
pub struct IntegralTensor<F = f64> {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<F>,   // was Vec<f64> — changed to Vec<F>
}

pub struct TypedEvaluationOutput<F = f64> {
    pub tensor: IntegralTensor<F>,
    pub stats: EvaluationStats,
    pub workspace_bytes: usize,
    pub chunk_count: usize,
    pub bytes_written: usize,
}
```

The `f64` defaults mean every existing `output.tensor.owned_values` reference (type `Vec<f64>`) compiles unchanged — D-12 preserved. Existing derives (`Clone, Debug, Default, PartialEq`) retained; f32/f64 both satisfy the implicit bounds.

### Unit Tests Added

- `integral_tensor_default_type_param_is_f64` — unparameterized resolves to f64
- `typed_evaluation_output_default_type_param_is_f64` — same for TEO
- `integral_tensor_f32_is_constructible` — `IntegralTensor::<f32>` and `TypedEvaluationOutput::<f32>` constructible

## Task 2: Generic evaluate_generic::<F>() + f64 Shim + CintFloat::PRECISION

### PART A: CintFloat::PRECISION const

Added `const PRECISION: PrecisionKind` to the `CintFloat` sealed trait:

```rust
pub trait CintFloat: ... {
    const PRECISION: PrecisionKind;  // NEW
    fn from_f64_lossy(x: f64) -> Self;
}

impl CintFloat for f64 { const PRECISION: PrecisionKind = PrecisionKind::F64; ... }
impl CintFloat for f32 { const PRECISION: PrecisionKind = PrecisionKind::F32; ... }
```

This provides zero-cost `F → PrecisionKind` mapping at compile time without runtime branching.

### PART B: Generic evaluate_generic::<F>() Body

The original `evaluate()` body moved into `evaluate_generic::<F: CintFloat + bytemuck::Pod>()`. Key changes:

| Old (f64 hardcoded) | New (F-generic) |
|---|---|
| `owned_values: Vec<f64>` | `owned_values: Vec<F>` |
| `F::zero()` initial fill | `F::zero()` (from `num_traits::Float`) |
| `size_of::<f64>()` | `size_of::<F>()` |
| `chunk_staging: Vec<f64>` | `chunk_staging: Vec<f64>` (unchanged — see staging note) |
| No plan.precision | `plan.precision = F::PRECISION` |
| `&mut chunk_staging` to ExecutionIo | `&mut chunk_staging` (Vec<f64>, frozen interface) |
| `copy_from_slice` direct | `bytemuck::cast_slice::<f64, F>` read-back |

**Staging buffer design rationale:**  
The kernel dispatchers (`one_electron.rs`, `two_electron.rs`) receive `&mut [f64]` (frozen ExecutionIo interface), then internally do `bytemuck::cast_slice_mut::<f64, f32>` for the f32 arm, writing `staging_elements` f32 values at indices 0..N. The facade therefore keeps `chunk_staging` as `Vec<f64>` (over-allocates for f32: N*2 f32 lanes >= N needed, proven Plan 01 A5 and Plan 06 T06-2d), then reads back via `bytemuck::cast_slice::<f64, F>` to extract exactly `chunk_len` F values. For f64 this is a zero-cost identity cast.

**Precision wiring** (Plan 07 wiring note satisfied):  
`plan.precision = F::PRECISION` is set immediately after `ExecutionPlan::new()`, following the `f12_zeta` caller-populates-after-new precedent from Plan 06.

**OOM-safe contract preserved verbatim:**  
`try_reserve_exact` + `HostAllocationFailed` + no-partial-writes — not touched.

### PART C: f64 Delegation Mechanism

Chosen mechanism: **thin non-generic shim**

```rust
/// Delegates to evaluate_generic::<f64>() — byte-identical f64 path (D-12).
pub fn evaluate(self) -> Result<TypedEvaluationOutput<f64>, FacadeError> {
    self.evaluate_generic::<f64>()
}
```

Every existing `req.evaluate()` call compiles unchanged with f64 behavior. The alternative (default type parameter on `evaluate<F = f64>`) was considered but requires a turbofish at most call sites to override, whereas the shim+rename approach makes `evaluate_generic::<f32>()` the explicit opt-in surface.

### Unit Tests Added

- `cintfloat_precision_const_f64_is_f64` — `f64::PRECISION == PrecisionKind::F64`
- `cintfloat_precision_const_f32_is_f32` — `f32::PRECISION == PrecisionKind::F32`
- `evaluate_generic_f32_returns_vec_f32_with_nonzero_element` — f32 smoke: Vec<f32> with nonzero elements
- `evaluate_unparameterized_delegates_to_f64_path` — `evaluate()` == `evaluate_generic::<f64>()` byte-for-byte

## Precision Plumbing (End-to-End, After Plan 07)

```
caller: req.evaluate_generic::<f32>()
    ↓ plan.precision = F32   (Plan 07: F::PRECISION wiring)
    ↓ check_capability → Ok() early for F32   (Plan 06 D-10)
    ↓ kernel dispatch: launch_*_typed::<f32>()   (Plans 04/05)
    ↓ kernel writes f32 values into f64 staging buffer
    ↓ bytemuck::cast_slice::<f64, f32> read-back
TypedEvaluationOutput<f32> { owned_values: Vec<f32> }

caller: req.evaluate()   (no turbofish — existing call sites)
    ↓ evaluate_generic::<f64>() shim
    ↓ plan.precision = F64 (default)
    ↓ byte-identical to pre-generic evaluate()
TypedEvaluationOutput<f64> { owned_values: Vec<f64> }
```

## Note for Plan 08

`req.evaluate_generic::<f32>()` is now the entry point that the f32 oracle gate (Plan 08) drives. The method returns `TypedEvaluationOutput<f32>` with `owned_values: Vec<f32>`. The oracle gate should cast `f32` values to `f64` before differencing against the libcint f64 reference, using `f32_tolerance_for_family` (from 20-PATTERNS.md §compare.rs).

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| cargo check cintx-rs | `cargo check -p cintx-rs` | exit 0 |
| All cintx-rs unit tests (25) | `cargo test -p cintx-rs` | 25/25 pass |
| f64 safe API oracle (PREC-04) | `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --test safe_api_arity2_parity` | 4/4 pass |
| Workspace check | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | exit 0 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] bytemuck cast_slice_mut from &mut [f32] to &mut [f64] panics on odd chunk_len**
- **Found during:** Task 2 GREEN phase (runtime panic in f32 test)
- **Issue:** The plan's interfaces section suggested `bytemuck::cast_slice_mut::<F, f64>(&mut chunk_staging)` where `chunk_staging: Vec<F>`. For `F = f32` with odd `chunk_len`, the byte size `chunk_len * 4` is not divisible by 8, causing a bytemuck `OutputSliceWouldHaveSlop` panic.
- **Fix:** Keep `chunk_staging` as `Vec<f64>` always (matching the frozen `ExecutionIo::new` interface), and use `bytemuck::cast_slice::<f64, F>` for read-back instead. This matches how the kernel dispatchers already work (they receive `&mut [f64]` and internally cast to `&mut [f32]`).
- **Files modified:** `crates/cintx-rs/src/api.rs`
- **Impact:** None — the staging buffer design is equivalent (f64 over-allocates for f32, proven Plan 01 A5); the read-back is correct because the kernel writes f32 values at the start of the f32 view of the f64 buffer.

## Known Stubs

None. `evaluate_generic::<f32>()` is fully wired end-to-end: `plan.precision = F32` propagates to kernel dispatchers which write real f32 values. The smoke test (`evaluate_generic_f32_returns_vec_f32_with_nonzero_element`) asserts nonzero output.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries.

T-20-19: MITIGATED — `evaluate_generic::<f32>()` returns `TypedEvaluationOutput<f32>` (`owned_values: Vec<f32>`); the type system makes f32/f64 confusion a compile error.
T-20-20: MITIGATED — buffer sized in `size_of::<F>()` for output; chunk_staging Vec<f64> over-allocates for f32 (Plan 01 A5); OOM-safe `try_reserve_exact` contract unchanged.
T-20-21: MITIGATED — `evaluate()` shim preserves f64 byte-identity; existing oracle tests (4/4) pass unchanged at atol=1e-12.
T-20-SC: ACCEPTED — bytemuck = "1" added (already in workspace via cintx-cubecl; legitimate package).

## Self-Check: PASSED

- `crates/cintx-core/src/precision.rs` has `const PRECISION: PrecisionKind`: FOUND (lines 53, 65, 75)
- `crates/cintx-rs/src/api.rs` has `struct IntegralTensor<F = f64>`: FOUND (line 544)
- `crates/cintx-rs/src/api.rs` has `struct TypedEvaluationOutput<F = f64>`: FOUND (line 552)
- `crates/cintx-rs/src/api.rs` has `pub fn evaluate_generic<F:`: FOUND (line 151)
- `crates/cintx-rs/src/api.rs` has `pub fn evaluate(` shim: FOUND (line 131)
- `plan.precision = F::PRECISION` in evaluate_generic: FOUND (line 187)
- `bytemuck::cast_slice::<f64, F>` read-back: FOUND (line 306)
- Commit 72ba738 (Task 1): FOUND
- Commit 4199fe8 (Task 2): FOUND
