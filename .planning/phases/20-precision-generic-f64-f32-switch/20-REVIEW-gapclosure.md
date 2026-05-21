---
phase: 20-gapclosure
reviewed: 2026-05-21T00:00:00Z
depth: deep
files_reviewed: 15
files_reviewed_list:
  - crates/cintx-rs/src/api.rs
  - crates/cintx-rs/Cargo.toml
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-cubecl/src/kernels/center_4c1e.rs
  - crates/cintx-cubecl/src/kernels/f12.rs
  - crates/cintx-cubecl/src/math/boys.rs
  - crates/cintx-cubecl/src/math/pdata.rs
  - crates/cintx-cubecl/src/math/rys.rs
  - crates/cintx-cubecl/src/math/stg.rs
  - crates/cintx-oracle/tests/f32_parity.rs
  - crates/cintx-cubecl/src/executor.rs
findings:
  critical: 0
  warning: 4
  info: 2
  total: 6
status: issues_found
---

# Phase 20 Gap-Closure: Code Review Report

**Reviewed:** 2026-05-21
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found (4 warnings, 2 info; no blockers)

---

## Summary

This review covers the gap-closure diff (`cd63035..HEAD`) spanning three plans: Plan 20-09
(`complex_values()` typed view on `IntegralTensor<F>`), Plan 20-10 (CR-01/CR-02/WR-03..WR-06
in the 7 kernel outer dispatchers and math helpers), and Plan 20-11 (f32 parity gate for
`int2e_stg_ip1_sph`).

**CR-01 fix shape (6 non-f12 kernels):** Correct. `out_elems = staging.len()` (f64 count
before bytemuck cast) is the true output element count on the F32 arm. The typed inner
receives `&mut staging_f32[..out_elems]`, bounding the copy and not0 scan correctly in all 6
kernels. The copy itself uses `staging.len().min(...)` which, with staging already bounded,
cannot exceed `out_elems`.

**CR-01 fix in f12:** Correct. The outer dispatcher captures `out_elems = staging.len()` before
the bytemuck cast, slices to `&mut staging_f32[..out_elems]`, and the inner `launch_f12_typed`
sizes its temporary f64 buffer to `out_elems`, bounds the readback loop, and bounds the not0
scan. All three bounds are present.

**Complex view (Plan 20-09):** Functionally correct. `chunks_exact(2)` on a buffer guaranteed
even-length by the planner produces the right `Complex<F>` pairs. No panic risk on the normal
code path. One structural concern noted below.

**WR-04 (`.expect()` on `to_f64()`):** Safe in practice. `num_traits::ToPrimitive::to_f64()`
for both `f32` and `f64` always returns `Some` under all finite and non-finite (NaN, Inf)
inputs; neither primitive implementation returns `None`. The `CintFloat` sealed trait guarantees
only `f32 | f64` implement it. The `.expect()` calls will not panic on real inputs.

**WR-05 (Boys tolerance):** Behaviorally correct for f32. The f64 path changes slightly — see
WR-02 below.

No critical (BLOCKER) issues found. All fixes are structurally sound; findings below are
quality issues, one precision-contract concern, and documentation inaccuracies.

---

## Warnings

### WR-01: Both `BufferTooSmall` guards in the outer F32 arm are always-false dead code

**Files:**
- `crates/cintx-cubecl/src/kernels/one_electron.rs:653`
- `crates/cintx-cubecl/src/kernels/two_electron.rs:746`
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs:429`
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs:498`
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs:506`
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs:758`
- `crates/cintx-cubecl/src/kernels/f12.rs:1650`

**Issue:** In each outer F32 dispatcher arm, the guard reads:

```rust
let out_elems = staging.len();          // staging: &mut [f64], len = chunk_len
let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
if staging_f32.len() < out_elems {      // staging_f32.len() = 2 * chunk_len ALWAYS
    return Err(cintxRsError::BufferTooSmall { ... });
}
```

`bytemuck::cast_slice_mut` on a `&mut [f64]` of length N always produces a `&mut [f32]` of
length 2N. Therefore `staging_f32.len()` (= `2 * out_elems`) is always ≥ `out_elems`, making
the condition `staging_f32.len() < out_elems` permanently false. Clippy may flag this as dead
code under `clippy::absurd_extreme_comparisons` or similar.

Additionally, in `launch_f12_typed` (f12.rs line 1563), the inner guard is:

```rust
let out_elems = staging.len();
if staging.len() < out_elems { ... }   // x < x, always false
```

This is tautologically dead code (comparing a value to itself).

The actual protection is the downstream slice bound `&mut staging_f32[..out_elems]` which
panics at runtime if `staging_f32.len() < out_elems` — but since that condition is impossible,
neither the guard nor the slice panic is ever reached.

**Fix:** Remove the unreachable guard blocks. Replace with a debug-only assertion to document
the invariant without shipping dead branches:

```rust
// F32 arm: bytemuck doubles the slice length; out_elems is always <= staging_f32.len().
let out_elems = staging.len();
let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
debug_assert!(staging_f32.len() >= out_elems, "bytemuck cast invariant violated");
launch_xxx_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
```

---

### WR-02: WR-05 doubles the f64 Boys convergence tolerance on both host and device paths — potential PREC-04 f64 last-bit drift

**Files:**
- `crates/cintx-cubecl/src/math/boys.rs:139` (host `boys_gamma_inc_impl`)
- `crates/cintx-cubecl/src/math/boys.rs:234` (device `boys_gamma_inc` `#[cube]`)

**Issue:** The old host tolerance was `DBL_EPSILON_HALF = f64::EPSILON * 0.5 ≈ 1.11e-16`.
The old device tolerance was `F::new(f64::EPSILON as f32 * 0.5)` which, for F=f64, evaluates
to approximately the same 1.11e-16. Both are now replaced with `F::epsilon()` / `F::EPSILON`
which for F=f64 equals `f64::EPSILON ≈ 2.22e-16` — a factor of 2 larger.

A larger convergence tolerance causes the power-series loop to terminate one iteration earlier
in borderline cases. For the f64 path, this changes the final Boys function value at the
last 1–2 ULPs. The comment acknowledges the factor-of-2 difference is "within the guard band of
the f64 oracle (atol=1e-12)", which is true — no oracle test will fail because `atol=1e-12` >>
`f64::EPSILON`. However, the stated PREC-04 design goal is f64 byte-identity with libcint at
the bit level, and this change makes that goal harder to verify.

The f32 change is clearly correct (old device code used a ~f32-precision constant even for f64,
which accidentally over-converged the f32 path and under-converged nothing but wasted iterations).

**Fix (advisory):** If strict f64 byte-identity is required, restore the f64 host path tolerance
to `DBL_EPSILON_HALF` (or equivalently `F::epsilon() * F::from_f64_lossy(0.5)` for f64) and
keep the new f32 path using `F::epsilon()`. The device path change is harder to revert cleanly
since `F::EPSILON * 0.5` is available in CubeCL. If the tolerance change is deliberately
accepted (it's within oracle tolerance), add a comment explicitly documenting that PREC-04
byte-identity is relaxed to atol=1e-12 for the Boys function host path, to prevent future
confusion.

---

### WR-03: `complex_values()` uses only `debug_assert` to enforce the even-length invariant — release builds silently truncate on user-constructed odd-length buffers

**File:** `crates/cintx-rs/src/api.rs:582`

**Issue:** `IntegralTensor<F>` is a fully public struct with all public fields. A caller can
construct:

```rust
let t = IntegralTensor {
    complex_interleaved: true,
    owned_values: vec![1.0f64, 2.0, 3.0],  // odd length
    ..Default::default()
};
let cv = t.complex_values();  // returns Some(vec![Complex(1,2)]) silently dropping 3.0
```

In debug builds the `debug_assert_eq!(self.owned_values.len() % 2, 0)` fires. In release
builds, `chunks_exact(2)` silently returns `floor(3/2) = 1` element, discarding the third
value. This is not a panic, but it is silent data loss from a public API — the caller receives
fewer complex elements than they might expect.

When `IntegralTensor` is produced by `evaluate()` / `evaluate_generic()`, the planner
guarantees `staging_elements` is always even for Spinor outputs (since `staging_elements =
base_elements * component_count * 2`), so the normal code path is safe. The risk is from
external construction.

**Fix:** Promote the `debug_assert` to a runtime check that returns `None` (or an error) for
odd-length buffers, since a `Some` with silently truncated results is worse than `None`:

```rust
pub fn complex_values(&self) -> Option<Vec<num_complex::Complex<F>>> {
    if !self.complex_interleaved {
        return None;
    }
    if self.owned_values.len() % 2 != 0 {
        // Malformed buffer: odd length for a complex-interleaved tensor.
        // Return None rather than silently truncating.
        return None;
    }
    Some(
        self.owned_values
            .chunks_exact(2)
            .map(|pair| num_complex::Complex::new(pair[0], pair[1]))
            .collect(),
    )
}
```

---

### WR-04: `complex_values()` docstring claims "typed reinterpretation" (zero-copy) but the implementation allocates a new `Vec` on every call

**File:** `crates/cintx-rs/src/api.rs:565-589`

**Issue:** The docstring states:

> "the contiguous interleaved `[re, im, re, im, ...]` `owned_values` buffer is reinterpreted
> element-for-element into `Complex<F>` (num_complex's `Complex<F>` is `#[repr(C)] { re, im }`,
> contiguous — this is a typed reinterpretation, not a data reshuffle)"

The phrase "typed reinterpretation" strongly implies a zero-copy view (e.g., via
`bytemuck::cast_slice`). The actual implementation calls `.collect()` on a `chunks_exact(2)`
iterator, allocating a new `Vec<Complex<F>>` on every invocation. This is a copy, not a
reinterpretation. A user calling `complex_values()` in a hot loop (e.g., per-chunk) will
incur a heap allocation and copy proportional to the output size each time, which they would
not expect from the word "reinterpretation".

The layout compatibility between `[F; 2]` (or interleaved `F`) and `num_complex::Complex<F>`
is correct, so a zero-copy path via `bytemuck::cast_slice` IS possible if `num_complex` is
built with the `"bytemuck"` feature. Without that feature, the allocating approach is correct —
but the documentation must say "copy" not "reinterpretation".

**Fix (documentation):** Revise the docstring to remove the zero-copy implication:

```
/// Returns `Some(Vec<Complex<F>>)` when `complex_interleaved == true`: each consecutive
/// `[re, im]` pair in `owned_values` is packaged into a `num_complex::Complex<F>` value.
/// The result is a newly allocated `Vec`; `owned_values` is unchanged.
```

**Fix (long-term, optional):** Add `num-complex` with `bytemuck` feature, then return
`Cow<[Complex<F>]>` backed by a zero-copy cast when the layout is compatible.

---

## Info

### IN-01: `to_f64().expect()` uses a runtime panic to enforce a compile-time-verifiable sealed-trait invariant

**Files:**
- `crates/cintx-cubecl/src/math/boys.rs:161`
- `crates/cintx-cubecl/src/math/pdata.rs:158-167`
- `crates/cintx-cubecl/src/math/rys.rs:820,935,1183,1332,1459,1603`
- `crates/cintx-cubecl/src/math/stg.rs:374-375`

**Issue:** `CintFloat` is sealed to `f64 | f32`. `num_traits::ToPrimitive::to_f64()` always
returns `Some` for both types (the primitive implementations never return `None`). The
`.expect("CintFloat is f32|f64; to_f64 is total")` pattern replaces the previous
`.unwrap_or(fabricated_fallback)` anti-pattern, which is an improvement.

However, in a public library, any `.expect()` / `.unwrap()` in non-test code is a latent DoS
surface if the invariant is ever violated (e.g., if `num_traits` changes its contract for these
types, or if the sealed trait is somehow bypassed). The message is clear and sufficient for
debugging, but a stronger pattern would avoid the `Option` entirely.

This is informational only — the current code is correct and the invariant holds for all
current and foreseeable `CintFloat` implementors.

**Suggestion:** If `num_traits` 0.2 exposes a `to_f64_raw` or similar infallible method in a
future version, migrate to it. For now, a helper function that wraps the pattern once and
documents the invariant avoids repeating the comment string 10+ times:

```rust
/// Convert `CintFloat` to f64; total for sealed {f64, f32} (num_traits ToPrimitive).
#[inline(always)]
fn cint_float_to_f64<F: CintFloat>(v: F) -> f64 {
    v.to_f64().expect("CintFloat is sealed to f32|f64; to_f64 is infallible")
}
```

---

### IN-02: `num-complex` version constraint is unpinned (`"0.4"`) while other public-API dependencies use pinned versions

**File:** `crates/cintx-rs/Cargo.toml:15`

**Issue:** The new dependency is declared as:

```toml
num-complex = "0.4"
```

This resolves to `^0.4`, allowing any `0.4.x` patch. CLAUDE.md specifies pinned versions for
all core and public-API dependencies (`thiserror = "2.0.18"`, `anyhow = "1.0.102"`, etc.) to
support reproducible oracle and manifest results. `num-complex` is now part of the public API
surface (it appears in the return type of the public `complex_values()` method), so callers
must have compatible versions. A `0.4.x` patch to `num-complex` that changes `Complex<F>`'s
`PartialEq` behavior or serialization would affect users without a lockfile bump.

**Suggestion:** Pin to the specific patch version currently resolved in `Cargo.lock`:

```toml
num-complex = "0.4.7"   # (or whatever is in Cargo.lock)
```

---

_Reviewed: 2026-05-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Commits reviewed: cd63035..HEAD (gap-closure, Plans 20-09/20-10/20-11)_
_This review is ADVISORY and non-blocking per the plan scope statement._
