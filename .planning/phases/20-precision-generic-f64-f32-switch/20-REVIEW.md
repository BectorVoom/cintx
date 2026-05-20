---
phase: 20-precision-generic-f64-f32-switch
reviewed: 2026-05-21T00:00:00Z
depth: standard
files_reviewed: 31
files_reviewed_list:
  - crates/cintx-core/Cargo.toml
  - crates/cintx-core/src/lib.rs
  - crates/cintx-core/src/precision.rs
  - crates/cintx-cubecl/src/executor.rs
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-cubecl/src/kernels/center_4c1e.rs
  - crates/cintx-cubecl/src/kernels/f12.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/math/boys.rs
  - crates/cintx-cubecl/src/math/obara_saika.rs
  - crates/cintx-cubecl/src/math/pdata.rs
  - crates/cintx-cubecl/src/math/rys.rs
  - crates/cintx-cubecl/src/math/stg.rs
  - crates/cintx-cubecl/src/transform/c2s.rs
  - crates/cintx-cubecl/src/transform/c2spinor.rs
  - crates/cintx-cubecl/tests/boys_tests.rs
  - crates/cintx-cubecl/tests/bytemuck_staging_cast_spike.rs
  - crates/cintx-cubecl/tests/obara_saika_tests.rs
  - crates/cintx-cubecl/tests/pdata_tests.rs
  - crates/cintx-cubecl/tests/rys_tests.rs
  - crates/cintx-oracle/src/compare.rs
  - crates/cintx-oracle/src/lib.rs
  - crates/cintx-oracle/tests/f32_parity.rs
  - crates/cintx-rs/Cargo.toml
  - crates/cintx-rs/src/api.rs
  - crates/cintx-runtime/src/options.rs
  - crates/cintx-runtime/src/planner.rs
findings:
  critical: 2
  warning: 6
  info: 5
  total: 13
status: issues_found
---

# Phase 20: Code Review Report

**Reviewed:** 2026-05-21
**Depth:** standard
**Files Reviewed:** 31 (4 test files in scope were inspected for reliability only)
**Status:** issues_found

## Summary

Phase 20 makes the compute path generic over `F: CintFloat` (f64/f32). The core
design — keeping all intermediate math in `f64`, converting only at the staging
write via `F::from_f64_lossy`, and reinterpreting the `&mut [f64]` staging buffer
as `&mut [f32]` via `bytemuck::cast_slice_mut` on the f32 arms — is consistent
across the 1e/2e/2c2e/3c1e/3c2e/4c1e kernels and preserves the FROZEN f64 const
tables. The `FamilyLaunchFn` signature (`fn(..., &mut [f64]) -> ...`) is left
intact on every outer dispatcher, and the precision tag is correctly threaded
`ExecutionOptions.precision -> plan.precision -> match` without making
`BackendExecutor` generic.

However, the **f32 staging-buffer math correctness rests on an unstated
invariant that the code does not enforce** (CR-01): the kernels write up to
`staging.len()` f32 lanes into a buffer that is sized in *f64* elements by the
caller, and the `bytemuck` cast doubles the lane count. The whole-buffer
truncating threshold comparison and the f12 temporary-buffer sizing
(`staging_len = staging.len()`) only happen to land on valid data because of the
specific way `api.rs` reads back exactly `chunk_len` lanes — there is no
defensive bound check, and one boundary case (CR-02) can read uninitialized /
stale f32 lanes into the f12 `not0` statistic and the truncating copy.

The f64 path is preserved byte-for-byte (verified by the `.to_bits()` parity
tests embedded in each kernel), and the f32 oracle gate (`f32_parity.rs`) is
additive and does not touch the frozen f64 tolerance constants. Several quality
defects (duplicated `common_fac_sp`/`cart_comps` across seven files, dead `nf_base`
binding, misleading doc comments) accompany the precision work.

## Critical Issues

### CR-01: f32 kernels read/write past the meaningful output region of the reinterpreted staging buffer

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:644-648` (pattern repeated in `two_electron.rs:742-745`, `center_2c2e.rs:423-426`, `center_3c1e.rs:492-495`, `center_3c2e.rs:500-503`, `center_4c1e.rs:752-755`, `f12.rs:1626-1629`)

**Issue:** On the F32 arm, the outer dispatcher does:

```rust
let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
launch_one_electron_typed::<f32>(backend, plan, specialization, staging_f32)
```

`staging` is a `Vec<f64>` of `chunk_len` elements allocated by the caller
(`api.rs:269-277` allocates `chunk_staging: Vec<f64>` of `chunk_len`; the runtime
`planner.rs:333` `try_alloc_staging` does the same). After the cast, `staging_f32`
has length `chunk_len * 2`. But the *real* output for the chunk is only
`staging_elements` (≤ `chunk_len`) elements. The typed inner then does:

```rust
let copy_len = staging.len().min(cart_buf.len());   // staging.len() == chunk_len*2
for (dst, &src) in staging[..copy_len].iter_mut().zip(cart_buf[..copy_len].iter()) { ... }
...
let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;  // scans chunk_len*2 lanes
```

The integral results are written correctly to f32 lanes `0..min(cart_buf.len(), chunk_len*2)`,
and `api.rs:306-311` reads back exactly `chunk_len` lanes — so for the common
single-chunk case the *values consumed by the caller* are correct. The defect is
that the kernel's own contract is unsound: nothing checks that the writeable f32
region (`chunk_len*2` lanes) is large enough or that the consumed region
(`chunk_len` lanes) is fully written. When `cart_buf.len()` (the true element
count) lies strictly between `chunk_len` and `chunk_len*2` — which happens for
multi-component / spinor outputs whose `staging_elements` exceeds the per-chunk
f64 allotment — the kernel will silently overwrite f32 lanes that `api.rs` then
re-slices, returning wrong values without any error. The "A5 proven sound" claim
in the doc comments covers only *alignment/Pod-ness*, not the *length contract*.

**Fix:** Make the writeable region explicit and assert it covers the output.
Pass the true element count down (it is already available as
`plan.output_layout.staging_elements` for the chunk) and bound the copy + `not0`
scan to it, returning `BufferTooSmall` when the f32 view cannot hold the output:

```rust
PrecisionKind::F32 => {
    let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
    let needed = expected_output_elements(plan); // == cart/sph/spinor element count for this chunk
    if staging_f32.len() < needed {
        return Err(cintxRsError::BufferTooSmall { required: needed, provided: staging_f32.len() });
    }
    launch_one_electron_typed::<f32>(backend, plan, specialization, &mut staging_f32[..needed])
}
```

and inside the typed inner replace every `staging.len()`-derived `copy_len` /
`not0` scan with the bounded `needed` length so the f32 path can never touch
lanes outside the real output.

### CR-02: f12 F32 path allocates and scans a temporary f64 buffer sized to the *doubled* f32 lane count, polluting `not0` and risking stale-lane truncation

**File:** `crates/cintx-cubecl/src/kernels/f12.rs:1555-1595`

**Issue:** `launch_f12_typed::<F>` does not write into `staging` directly; it
allocates a private `staging_f64` and runs the f64 sub-kernel there:

```rust
let staging_len = staging.len();          // F32: chunk_len * 2  (doubled by the bytemuck cast)
let mut staging_f64 = vec![0.0_f64; staging_len];
let stats = ... launch_stg_base(..., &mut staging_f64, zeta) ...?;
for (dst, &src) in staging.iter_mut().zip(staging_f64.iter()) {  // writes chunk_len*2 lanes
    *dst = F::from_f64_lossy(src);
}
let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;
```

For the F32 arm, `staging.len() == chunk_len * 2`, so `staging_f64` is allocated
**twice the size** the f12 sub-kernels expect. The sub-kernel (`f12_kernel_core`)
fills only the first `staging_elements` slots and leaves the upper half at `0.0`.
That upper half is then (a) iterated by the `not0` filter — harmless for zeros
but wrong if the sub-kernel's `copy_from_slice` ever lands data there, and more
importantly (b) the `staging_f64` over-allocation means the sub-kernel's own
`staging.len().min(sph.len())` clamps (e.g. `f12.rs:1278`, `1374`) compute against
the doubled length, so for a derivative variant whose component layout exceeds
`chunk_len` the sub-kernel writes component data into lanes that the f32 view
then truncates and `api.rs` re-slices incorrectly. Unlike the other five kernels
(which at least write through the same buffer the caller reads), f12 introduces a
*second* buffer whose size is derived from the already-doubled f32 lane count,
compounding the CR-01 length-contract violation. The f64 path is unaffected
(`staging_len == chunk_len`), which is why the `test_f12_parity_f64`
`.to_bits()` test passes and hides this.

**Fix:** Size the temporary f64 buffer to the true element count, not the
reinterpreted f32 lane count, and bound the readback/`not0` to it:

```rust
let out_elems = plan.output_layout.staging_elements_for_chunk(...); // true count
let mut staging_f64 = vec![0.0_f64; out_elems];
// ... run sub-kernel into staging_f64 ...
for (dst, &src) in staging[..out_elems].iter_mut().zip(staging_f64.iter()) {
    *dst = F::from_f64_lossy(src);
}
let not0 = staging[..out_elems].iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;
```

Add an explicit `staging.len() >= out_elems` guard returning `BufferTooSmall`
before the run so the OOM-safe / no-partial-write contract is preserved on the
f32 path.

## Warnings

### WR-01: `staging_bytes` / `peak_workspace_bytes` reported in f12 inner uses `size_of::<f64>()` regardless of `F`

**File:** `crates/cintx-cubecl/src/kernels/f12.rs:1398`

**Issue:** Inside `f12_kernel_core` the stats are computed as
`let staging_bytes = staging.len() * std::mem::size_of::<f64>();`. That is fine
*there* because `f12_kernel_core` always operates on the temporary f64 buffer.
But `launch_f12_typed` then overrides with `staging.len() * size_of::<F>()`
(line 1597), and `staging.len()` on the F32 arm is the doubled lane count, so the
reported `peak_workspace_bytes`/`transfer_bytes` for f32 f12 is `chunk_len*2*4 =
chunk_len*8` — i.e. it reports the full f64 buffer size, double the genuine f32
output bytes. Every other kernel reports `staging.len() * size_of::<F>()` where
`staging.len()` is the (already correct for them) f32 view length, so this is an
f12-specific stats inconsistency that will mis-report transfer accounting once
CR-02 is fixed. Recompute against the true output element count.

**Fix:** After fixing CR-02, set `let staging_bytes = out_elems * size_of::<F>();`
in `launch_f12_typed`.

### WR-02: `nf_base` is computed but never used in the f12 base-variant path

**File:** `crates/cintx-cubecl/src/kernels/f12.rs:1199`

**Issue:** `let nf_base = nfi_base * nfj_base * nfk_base * nfl_base;` is bound and
used by the derivative path (`gout_contracted = vec![0.0; ncomp * nf_base]`,
line 1303) but in the `ncomp == 1` base branch it is dead. It is not behind a
cfg, so if the derivative branch is ever feature-gated out this becomes an unused
binding. More importantly the dual use of base vs ceil ncart sizing
(`nfi_ceil` vs `nfi_base`) in the same function is a correctness foot-gun — a
future edit could pass `nf_base` where `nfi_ceil*...` is required. Document the
two regimes or split the function.

**Fix:** Annotate intent or scope `nf_base` to the derivative branch:
`let nf_base = if ncomp == 1 { 0 } else { nfi_base * nfj_base * nfk_base * nfl_base };`
is not ideal; prefer moving the binding inside the `else { ... }` block.

### WR-03: `compute_pdata_host::<f32>` performs the Gaussian-product math in f32, not f64, despite the "all intermediates stay f64" contract

**File:** `crates/cintx-cubecl/src/math/pdata.rs:140-176`

**Issue:** The module doc and every kernel doc claim "intermediate computations
remain f64; precision conversion happens only at the final staging write." But
`compute_pdata_host<F: CintFloat>` computes `zeta_ab`, `center_p`, `rr`, and the
exponential `fac = (-ai*aj/zeta_ab*rr).exp()` in `F` arithmetic and only converts
to f64 *after* the math (`.to_f64().unwrap_or(0.0)`). On the f32 arm the inputs
arrive as f32 (exponents/coords passed as `F`), so the pair-data exponential —
the single most precision-sensitive quantity in the whole pipeline — is evaluated
in f32 and then widened to f64, which does NOT match "compute in f64." This is the
likely dominant source of f32 error and contradicts the stated design. Note the
`#[cube]` `compute_pdata` has the same property but that is device code; the host
wrapper is what the f32 oracle gate exercises via the kernels (the kernels call
`compute_pdata_host(ai, aj, ...)` where `ai/aj` are already f64 because the
kernels read `shell.exponents[..]: f64` — so in practice the *kernel* callers pass
f64 and the f32 monomorphization of this function is only hit by the
`pdata_tests.rs` smoke test). Verify the kernel call sites never instantiate
`compute_pdata_host::<f32>`; if they can, the f32 results silently lose precision
versus the documented f64-intermediate contract.

**Fix:** Make the host wrapper always compute in f64 by converting inputs first:
`let ai = ai.to_f64().unwrap_or(0.0); ...` then run the existing f64 body. The
return type is already concrete `PairData` (f64), so this is a localized change
that makes the f32 monomorphization genuinely f64-intermediate as documented.

### WR-04: `to_f64().unwrap_or(...)` silently substitutes a fabricated value on conversion failure across the math layer

**File:** `crates/cintx-cubecl/src/math/stg.rs:374-375`, `crates/cintx-cubecl/src/math/pdata.rs:167-175`, `crates/cintx-cubecl/src/math/rys.rs:819`, `crates/cintx-cubecl/src/math/boys.rs:162`

**Issue:** Every `F -> f64` conversion in the math layer uses
`.to_f64().unwrap_or(<fallback>)`. For `f32`/`f64` inputs `num_traits::Float::to_f64`
never returns `None`, so the fallback is dead today. But the fallbacks are not
neutral: `stg.rs:375` uses `ua.to_f64().unwrap_or(1.0)` and then divides by
`ua_f64.sqrt()` and takes `ua_f64.log10()`; a fabricated `1.0` would silently
produce a finite-but-wrong root rather than an error. `boys.rs:162`
`tt.to_f64().unwrap_or(0.0)` would feed `erf_host(0.0)` and produce a bogus
`F_0`. These mask any future trait widening (e.g. an `f16` arm) behind silent
wrong numbers on the OOM-safe path instead of a typed failure. Per CLAUDE.md the
library path must use `thiserror` failures, not best-effort substitution.

**Fix:** Since the sealed trait guarantees lossless `to_f64`, replace
`.unwrap_or(x)` with `.expect("CintFloat is f32|f64; to_f64 is total")` to convert
the impossible case into a loud panic during development, or thread a
`Result<_, cintxRsError>` if a fallible contract is desired. At minimum the
fallback constants must not be values that produce plausible-but-wrong output
(`1.0` for `ua`, `0.0` for a Boys argument).

### WR-05: Boys power-series convergence tolerance is hard-pinned to f64 epsilon on the f32 path, contradicting the inline comment

**File:** `crates/cintx-cubecl/src/math/boys.rs:141` and `:233`

**Issue:** The host `boys_gamma_inc_impl` sets
`let tol = F::from_f64_lossy(DBL_EPSILON_HALF) * e;` where
`DBL_EPSILON_HALF = f64::EPSILON * 0.5 ≈ 1.1e-16`. For `F = f32` this tolerance
(~1.1e-16, then stored in an f32) underflows/rounds to a value far below f32
machine epsilon (~6e-8). The loop `while x > tol { x = x*t/bi; ... }` then runs
until `x` underflows to subnormal/zero rather than until f32 convergence — the
comment "for f32, this tolerance is tighter ... safe for correctness" is
optimistic: it can spin many extra iterations and accumulate f32 rounding in `s`,
and `tol` itself may round to `0.0_f32` making the loop termination depend solely
on `x` underflowing. The `#[cube]` twin at line 233 uses
`F::new(f64::EPSILON as f32 * 0.5)` which evaluates `f64::EPSILON as f32` (≈ 2.2e-16
→ rounds to ~2.2e-16 stored in f32, effectively a denormal-ish tiny number) — a
different value from the host path, so the host and device Boys functions are NOT
guaranteed to converge identically on f32. This is a host/device divergence on
the very function the precision refactor is built around.

**Fix:** Use a precision-appropriate tolerance: `let tol = F::epsilon() * e;`
(num_traits provides `Float::epsilon()` per-type), and mirror the same expression
in the `#[cube]` path so host and device share one convergence criterion.

### WR-06: Loop-carried `not0` / threshold uses `F::from_f64_lossy(1e-18)` which underflows to 0.0 in f32, defeating the anti-zero-fill sentinel

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:600`, `two_electron.rs:709`, `center_2c2e.rs:390`, `center_3c1e.rs:454`, `center_3c2e.rs:466`, `center_4c1e.rs:716`, `f12.rs:1591`

**Issue:** Each kernel computes
`let nonzero_threshold = F::from_f64_lossy(1e-18_f64);` then counts
`staging.iter().filter(|&&v| v.abs() > nonzero_threshold)`. For `F = f32`,
`1e-18_f64 as f32` is representable (~1e-18 is above f32's smallest subnormal
~1.4e-45), so it does not flatten to 0.0 — but it is far below f32's *useful*
precision floor (~1e-7 relative). The sentinel therefore counts as "nonzero" any
f32 lane holding leftover/garbage as small as 1e-18, which combined with CR-01/CR-02
(scanning `chunk_len*2` lanes including the uninitialized upper half) inflates the
`not0` statistic. The f32 oracle gate's own anti-zero-fill check uses `1e-18_f32`
(`f32_parity.rs:519`) so it agrees with the kernel, but neither catches the
"counted stale lanes" case. The threshold should reflect the f32 noise floor.

**Fix:** Bound the `not0` scan to the true output region (see CR-01) and pick a
precision-aware sentinel, e.g. `F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 })`,
or simply scan only the genuinely-written elements so stale lanes cannot register.

## Info

### IN-01: `common_fac_sp` and `cart_comps` are copy-pasted verbatim across seven kernel files

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:41-67`, `two_electron.rs:23-46`, `center_2c2e.rs:45-71`, `center_3c1e.rs:44-70`, `center_3c2e.rs:34-59`, `center_4c1e.rs:26-49`, `f12.rs:31-54`

**Issue:** Identical `common_fac_sp(l: u8) -> f64` (with the same two magic
constants `0.282094791773878143` / `0.488602511902919921`) and near-identical
`cart_comps` helpers are duplicated in seven files. The 3c2e copy returns
`(usize, usize, usize)` while the others return `(u8, u8, u8)` — a subtle drift
already. A single shared `crate::math::common_fac_sp` and `cart_comps` would
remove the drift risk and the magic-number repetition.

**Fix:** Extract both to a shared module (e.g. `crate::transform::c2s` already
hosts `ncart`/`nsph`) and import from each kernel.

### IN-02: `SQRTPI` literal redefined in five kernel files instead of using `std::f64::consts`

**File:** `crates/cintx-cubecl/src/kernels/center_2c2e.rs:37`, `center_3c1e.rs:36`, `center_3c2e.rs:26`, `center_4c1e.rs:23`, `one_electron.rs:26`, `f12.rs:28`

**Issue:** `const SQRTPI: f64 = 1.7724538509055159_f64;` is hand-written in six
files. The last digit differs from `std::f64::consts::PI.sqrt()`
(1.7724538509055160…), and a hand-typed constant is exactly the kind of FROZEN
value the precision doc warns about. Centralize it next to `common_fac_sp`.

**Fix:** Define once in a shared `math` constants module and re-export.

### IN-03: Misleading doc comment claims f12 internal math "stays f64 throughout" while pdata math can run in F

**File:** `crates/cintx-cubecl/src/kernels/f12.rs:1518-1525`

**Issue:** The doc states "all intermediates stay `f64` and only the final staging
write uses `F::from_f64_lossy`." As noted in WR-03, `compute_pdata_host` actually
computes in `F`. The comment is accurate only because the f12 kernel passes f64
exponents (`shell_*.exponents[..]` are f64) — but the doc asserts a guarantee the
type system does not provide. Tighten the wording to "kernel callers pass f64
inputs, so the f64 path is exercised" or fix WR-03.

**Fix:** Reword the doc or implement WR-03 so the claim becomes true by construction.

### IN-04: `#[allow(unused_assignments)]` blanket-applied at module scope in rys.rs masks real dead-store bugs

**File:** `crates/cintx-cubecl/src/math/rys.rs:12`

**Issue:** `#![allow(unused_assignments)]` is applied to the whole module to
silence the `let mut rt1 = F::new(0.0);` init-then-overwrite pattern. A
module-wide allow also suppresses genuine dead-store warnings that could indicate
a missed branch in the large piecewise polynomial functions (e.g. a segment that
forgets to set `ww2`). Prefer a localized `#[allow]` on the specific functions or
restructure so the initial values are not dead.

**Fix:** Move the allow to the individual `rys_root*` functions, or initialize
the accumulators only in branches that need them.

### IN-05: `T_MAX` and the magic scaling constant `0.9102392266268373` in stg.rs are undocumented magic numbers

**File:** `crates/cintx-cubecl/src/math/stg.rs:31`, `:382`

**Issue:** `const T_MAX: f64 = 19682.99` has a source comment but the value's
provenance (why 19682.99 specifically) is not derivable from the comment; and
`tt = t.ln() * 0.9102392266268373 + 1.0` carries an inline comment `// log(3)+1
scaling` that does not match the literal (`1/ln(3) = 0.91024…`, not `log(3)+1`).
The comment is wrong even if the constant is right, which will mislead a future
maintainer porting/validating the STG grid.

**Fix:** Correct the inline comment to identify the constant as `1/ln(3)` (or the
true derivation) and cite the exact `stg_roots.c` line for `T_MAX`.

---

_Reviewed: 2026-05-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
