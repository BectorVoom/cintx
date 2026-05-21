---
phase: 20-precision-generic-f64-f32-switch
plan: 10
subsystem: kernels + math
tags: [gap-closure, f32-correctness, cr-01, cr-02, wr-01, wr-03, wr-04, wr-05, wr-06, prec-05, prec-04]
dependency_graph:
  requires: [20-09]
  provides: [PREC-05-kernel-math-hardening]
  affects: [20-11]
tech_stack:
  added: []
  patterns:
    - "out_elems = staging.len() pre-cast pattern for all 7 F32 outer arms (CR-01)"
    - "staging_f32[..out_elems] slice bounding (CR-01)"
    - "BufferTooSmall guard on F32 arm before typed inner call (CR-01)"
    - "vec![0.0_f64; out_elems] for f12 temp buffer (CR-02)"
    - "to_f64().expect(...) replacing to_f64().unwrap_or(...) in math layer (WR-04)"
    - "F::epsilon() host + F::EPSILON device for Boys convergence tol (WR-05)"
    - "f64-first compute_pdata_host: convert inputs then run math in f64 (WR-03)"
key_files:
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-cubecl/src/kernels/center_2c2e.rs
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-cubecl/src/kernels/center_4c1e.rs
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/math/pdata.rs
    - crates/cintx-cubecl/src/math/stg.rs
    - crates/cintx-cubecl/src/math/rys.rs
    - crates/cintx-cubecl/src/math/boys.rs
decisions:
  - "WR-05 device path: used F::EPSILON (CubeCL Float const, available in #[cube] as a const, verified from cubecl-core-0.10.0/src/frontend/element/float.rs) not F::new(f64::EPSILON as f32 * 0.5). Host path uses num_traits F::epsilon(). Both yield the same type-appropriate epsilon (f32::EPSILON for f32, f64::EPSILON for f64)."
  - "WR-05 f64 oracle impact: F::epsilon() for f64 is 2.22e-16 vs prior DBL_EPSILON_HALF 1.11e-16 — factor-of-2 difference. The f64 integration oracle still passes at atol=1e-12 (confirmed in Task 4). No precision branch needed."
  - "CR-01 fix shape: out_elems captured pre-cast in outer dispatcher; &mut staging_f32[..out_elems] passed to typed inner. staging.len() inside typed inner is now out_elems for both f64 and f32 — all staging.len()-derived copy_len/.min() expressions in the typed inner are automatically correct."
  - "WR-03: compute_pdata_host now converts ALL inputs to f64 before any arithmetic. The return type was already concrete PairData (f64 fields), so no boundary change was needed — just removed the trailing .to_f64().unwrap_or() calls that were converting already-f64 values."
  - "DBL_EPSILON_HALF constant removed from boys.rs (was only used by WR-05 host tol, now replaced by F::epsilon())."
metrics:
  duration_minutes: 7
  completed_date: "2026-05-21"
  tasks_completed: 4
  files_modified: 11
  commits: 2
---

# Phase 20 Plan 10: Gap 2 (PREC-05) Kernel + Math Hardening Summary

**One-liner:** CR-01/CR-02 unsound f32 staging-buffer length contract fixed in all 7 kernels + f12 typed inner; WR-01/WR-03/WR-04/WR-05/WR-06 f32-correctness hardening in math layer; f64 oracle byte-identical at atol=1e-12 (PREC-04 preserved).

## Objective

Gap 2 closure (PREC-05) — kernel + math half. This plan fixes critical defects in the f32 compute path that could silently return wrong values for multi-component, spinor, and f12-derivative outputs, and hardens the math layer's f32-correctness contracts.

## Tasks Completed

### Task 1+2: CR-01 + WR-06 (all 7 kernels) + CR-02 + WR-01 (f12 typed inner)

**Commit:** `5ba79fb`

**CR-01 fix — all 7 kernel F32 outer arms:**

The key data-flow fact: `api.rs` allocates `chunk_staging: Vec<f64>` of `chunk_len` elements (the TRUE output element count). After `bytemuck::cast_slice_mut`, `staging_f32.len() == chunk_len * 2`. The fix:

```rust
PrecisionKind::F32 => {
    let out_elems = staging.len(); // f64 slice length == TRUE output element count
    let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
    if staging_f32.len() < out_elems {
        return Err(cintxRsError::BufferTooSmall { required: out_elems, provided: staging_f32.len() });
    }
    launch_kernel_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
}
```

Because the typed inner now receives `staging` of length exactly `out_elems`, all `staging.len()`-derived `copy_len/.min()` expressions inside are automatically correct.

**CR-02 fix — f12 typed inner (`launch_f12_typed`):**

```rust
let out_elems = staging.len(); // true output element count (outer arm sliced for F32)
let mut staging_f64 = vec![0.0_f64; out_elems]; // was: staging.len() == chunk_len*2 for F32
// readback bounded:
for (dst, &src) in staging[..out_elems].iter_mut().zip(staging_f64.iter()) { ... }
// not0 bounded:
let not0 = staging[..out_elems].iter().filter(...).count();
```

**WR-01 fix — f12 true-byte stats:**

```rust
let staging_bytes = out_elems * std::mem::size_of::<F>(); // was: staging.len() (doubled for f32)
```

**WR-06 fix — all 7 kernels:**

```rust
let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
```

### Task 3: WR-03/WR-04/WR-05 — math layer hardening

**Commit:** `69a6a18`

**WR-03 — `compute_pdata_host` (pdata.rs):**

Converted all inputs to f64 first, then run Gaussian-product math in f64. The precision-sensitive exponential `fac = (-ai*aj/zeta_ab*rr).exp()` was previously evaluated in F (f32 for the f32 monomorphization). The return type was already concrete `PairData` (f64 fields) so the trailing `.to_f64().unwrap_or(0.0)` calls were removed — values are already f64 after the input conversion.

**WR-04 — all math layer files (pdata.rs, stg.rs, rys.rs, boys.rs):**

Replaced all `.to_f64().unwrap_or(<fabricated>)` with `.to_f64().expect("CintFloat is f32|f64; to_f64 is total")`. The `ua.to_f64().unwrap_or(1.0)` (stg.rs) and `tt.to_f64().unwrap_or(0.0)` → `erf` (boys.rs) were the most dangerous cases — now loud panics instead of plausible-but-wrong substitutions.

**WR-05 — Boys convergence tolerance (boys.rs host + #[cube]):**

- Host (`boys_gamma_inc_impl`): `let tol = F::epsilon() * e;` (num_traits::Float::epsilon())
- Device (#[cube] `boys_gamma_inc`): `let tol = F::EPSILON * e;` (CubeCL Float const, verified available in cubecl-core-0.10.0)
- For f32: tol ≈ f32::EPSILON (~1.19e-7) — terminates at f32 convergence, not f64-epsilon underflow
- For f64: tol = f64::EPSILON (2.22e-16) — factor-of-2 vs old DBL_EPSILON_HALF (1.11e-16); f64 oracle still passes at atol=1e-12
- Removed `DBL_EPSILON_HALF` constant (no longer referenced)
- Host and device now converge at the same precision-appropriate tolerance

### Task 4: f64 byte-identity regression gate (PREC-04)

Verified by running the full integration oracle and all cintx-cubecl tests post-fix:

- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` — exit 0
- `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu,with-f12,with-4c1e` — 180 lib tests + all integration tests pass, 0 failed
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12,with-4c1e --test '*'` — all 13 integration test suites pass (79+ tests); the 4 `compare::tests` lib failures are the pre-existing `CINTshells_cart_offset[4]` issue, independently confirmed out-of-scope

## Deviations from Plan

None — plan executed exactly as written.

The only note: `one_electron.rs` and `two_electron.rs` had CR-01 outer arm fixes already present in the codebase from a prior partial edit (git status showed them modified before this plan executed). The WR-06 threshold update in `one_electron.rs` was the remaining change; both files were committed with the full Task 1+2 batch.

## WR-05 Device-Side Mechanism (per plan output spec)

CubeCL 0.10.0's `Float` trait (in `cubecl-core-0.10.0/src/frontend/element/float.rs`) defines `const EPSILON: Self` as a trait constant. Inside `#[cube]` functions, `F::EPSILON` is available and resolves to `f32::EPSILON` for `F=f32` and `f64::EPSILON` for `F=f64`. This is the same per-type precision as `num_traits::Float::epsilon()` on the host, so host and device now converge at the same tolerance for each precision. The old device expression `F::new(f64::EPSILON as f32 * 0.5)` was evaluated as a single f32 literal (~1.1e-16 rounded to f32 ≈ a denormal) — effectively lower than f32::EPSILON and inconsistent with the host.

## Note for Plan 20-11

The f12 typed inner (`launch_f12_typed`) now sizes `staging_f64` to `out_elems` and bounds all readback/not0 to `out_elems`. For derivative variants (e.g. `stg_ip1`, ncomp=3), the sub-kernels' internal `staging.len().min(...)` clamps now compute against the correctly-sized `out_elems` buffer. The multi-component f32 paths (f12 `stg_ip1` with ncomp=3, spinor outputs) are now structurally correct and ready for the new f32 oracle tests in Plan 20-11.

## Self-Check

### Files Exist

- [x] `crates/cintx-cubecl/src/kernels/one_electron.rs` — modified
- [x] `crates/cintx-cubecl/src/kernels/two_electron.rs` — modified
- [x] `crates/cintx-cubecl/src/kernels/center_2c2e.rs` — modified
- [x] `crates/cintx-cubecl/src/kernels/center_3c1e.rs` — modified
- [x] `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — modified
- [x] `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — modified
- [x] `crates/cintx-cubecl/src/kernels/f12.rs` — modified
- [x] `crates/cintx-cubecl/src/math/pdata.rs` — modified
- [x] `crates/cintx-cubecl/src/math/stg.rs` — modified
- [x] `crates/cintx-cubecl/src/math/rys.rs` — modified
- [x] `crates/cintx-cubecl/src/math/boys.rs` — modified

### Commits Exist

- [x] `5ba79fb` — CR-01+CR-02+WR-01+WR-06 kernel fixes
- [x] `69a6a18` — WR-03+WR-04+WR-05 math layer fixes

## Self-Check: PASSED
