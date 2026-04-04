# Phase 11: Helper/Transform Completion & 4c1e Real Kernel - Research

**Researched:** 2026-04-04
**Domain:** Oracle harness coverage for helper/transform/legacy-wrapper symbols; 4c1e Rys quadrature kernel implementation; tolerance unification; workaround module creation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Tighten ALL per-family tolerances to atol=1e-12 in this phase, not just new symbols. Replace the per-family constants in `compare.rs` (TOL_1E_ATOL, TOL_2E_ATOL, TOL_2C2E_3C2E_ATOL, TOL_3C1E_ATOL, TOL_4C1E_ATOL) with a single unified constant.
- **D-02:** CINTgto_norm (the only float-returning helper) uses float atol=1e-12. All other helpers (count, offset, norm-count) use exact integer equality. If a family fails at 1e-12, the kernel is buggy and must be fixed — tolerance is immutable.
- **D-03:** Adapt existing 2e Rys quadrature infrastructure (rys_roots, rys_weights, VRR, HRR) for the 4-center 1-electron operator. Do not create a parallel code path; reuse the patterns from `two_electron.rs` and `center_3c1e.rs` with 4-center routing modifications.
- **D-04:** `int4c1e_via_2e_trace` workaround lives in a new `cintx-compat::workaround` module. It calls eval_raw with a 2e symbol, then traces/contracts the result to produce 4c1e-equivalent output. Clean separation from the real kernel in `cintx-cubecl`.
- **D-05:** Validated4C1E envelope stays as-is: cart/sph representation, scalar component_rank, max(l)<=4. Spinor 4c1e returns UnsupportedApi unconditionally — classifier checks representation before angular momentum (v1.2 roadmap decision).
- **D-06:** Use manifest-driven gap analysis to identify missing symbols. Query compiled_manifest.lock.json for all helper_kind entries (helper, transform, legacy-wrapper, optimizer). Diff against IMPLEMENTED_*_SYMBOLS arrays in `compare.rs` to produce an exact gap list.
- **D-07:** Legacy wrappers use eval-based comparison: call each wrapper via eval_raw with test fixtures, compare output buffers against vendored libcint at atol=1e-12. Same approach as base family oracle tests.
- **D-08:** Extend the existing `helper_legacy_parity_gate` from Phase 4 in-place to cover newly oracle-wired symbols. No new CI jobs — expand what the existing gate tests across all four feature profiles.
- **D-09:** 4c1e oracle parity runs inside the existing `oracle_parity_gate` when with-4c1e profile is active. Already profile-gated, no new CI jobs needed.

### Claude's Discretion

- Exact Rys quadrature adaptation details for 4c1e (center routing, pair data construction)
- Oracle fixture molecule/shell choices for new symbols
- Internal module organization within the workaround module
- Order of implementation (helpers first vs 4c1e first vs parallel)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HELP-01 | Oracle harness compares every helper symbol against vendored libcint with atol=1e-12 | All 17 helper symbols already implemented in `helpers.rs`; oracle comparison in `compare.rs::verify_helper_surface_coverage` does existence checks but NOT numeric oracle comparison against vendored libcint; gap: add numeric helper oracle tests |
| HELP-02 | Oracle harness compares every transform symbol against vendored libcint with atol=1e-12 | 7 transform symbols in manifest and `compare.rs::IMPLEMENTED_TRANSFORM_SYMBOLS`; same gap as HELP-01 — surface coverage check exists but numeric oracle comparison missing |
| HELP-03 | Oracle harness compares every legacy wrapper symbol against vendored libcint with atol=1e-12 | 45 legacy wrapper symbols in manifest, matching `LEGACY_WRAPPER_SYMBOLS` in `legacy.rs`; `eval_legacy_symbol` in compare.rs maps them for fixture parity but 4c1e and spinor legacy wrappers may not be covered |
| HELP-04 | CI helper-legacy-parity gate passes with 0 mismatches across all four feature profiles | `helper_legacy_parity_gate` CI job exists in `compat-governance-pr.yml` looping over all 4 profiles; currently calls `verify_helper_surface_coverage` which checks sets but does no numeric oracle comparison; tolerance unification and numeric comparisons must be added to make the gate meaningful |
| 4C1E-01 | int4c1e_sph produces real Rys quadrature results matching libcint at atol=1e-12 within Validated4C1E envelope | `center_4c1e.rs` is a stub; algorithm is `g4c1e.c` / `cint4c1e.c` from vendored libcint — uses same VRR/HRR infrastructure as 2e but with a different G-tensor prefactor (no Rys roots — it IS the 4-center Gaussian overlap) |
| 4C1E-02 | int4c1e_via_2e_trace workaround produces results matching direct 4c1e evaluation | New `cintx-compat::workaround` module needed; calls eval_raw with a 2e operator symbol then traces/contracts over auxiliary index |
| 4C1E-03 | Out-of-envelope 4c1e inputs return UnsupportedApi; spinor 4c1e returns UnsupportedApi unconditionally | `validate_4c1e_envelope` already exists in `raw.rs` and checks representation then angular momentum, but currently checks representation AFTER feature gate; spinor check must be FIRST per D-05 |
| 4C1E-04 | Oracle parity CI gate for with-4c1e profile passes with 0 mismatches at atol=1e-12 | `oracle_parity_gate` CI already runs with-4c1e profile; `raw_api_for_symbol` in compare.rs already maps `int4c1e_cart` and `int4c1e_sph`; 4c1e family tolerance constant must be updated to 1e-12 (D-01) |
</phase_requirements>

## Summary

Phase 11 has two parallel workstreams: oracle coverage completion for helper/transform/legacy-wrapper symbols, and replacing the `center_4c1e.rs` stub with a real Rys quadrature kernel.

**Helper/transform/legacy-wrapper coverage (HELP-01 through HELP-04):** The manifest has 17 helper symbols, 7 transform symbols, and 45 legacy wrapper symbols — all of which already have matching implementations in `helpers.rs`, `transform.rs`, and `legacy.rs`. The gap is not in the implementations but in the oracle comparison: `verify_helper_surface_coverage` in `compare.rs` does set-equality checks (do the right symbols exist?) but does NOT perform numeric comparison against vendored libcint. Phase 11 must add numeric oracle assertions — exact integer equality for count/offset helpers, float atol=1e-12 for CINTgto_norm and transforms, and eval-based atol=1e-12 for legacy wrappers. Additionally, the six per-family tolerance constants in `compare.rs` must be collapsed into a single `UNIFIED_ATOL = 1e-12` constant (D-01).

**4c1e real kernel (4C1E-01 through 4C1E-04):** The `center_4c1e.rs` file is a pure stub (staging remains zeros). The libcint algorithm is in `g4c1e.c` / `cint4c1e.c` — it is NOT a Rys quadrature integral. It is a 4-center Gaussian overlap integral using the same G-tensor shape as 2e (ibase/kbase adaptive layout, same 4-branch HRR) but with a different prefactor: `fac = 1 / (aijkl * sqrt(aijkl))` with polynomial recurrence instead of Rys roots. The 2e VRR/HRR infrastructure from `two_electron.rs` can be reused for the G-tensor fill after replacing the Rys-rooted c00/b10/b01/b00 parameters with the 4c1e polynomial recurrence parameters from `g4c1e.c`. The workaround module (D-04) calls eval_raw with int2e_sph then traces the auxiliary (k,l) index pair to contract to 4c1e-equivalent output.

**Primary recommendation:** Implement in this order — (1) tolerance unification in compare.rs, (2) numeric helper oracle assertions, (3) 4c1e kernel, (4) workaround module, (5) gate CI validation.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust toolchain | 1.94.0 (pinned) | Reproducible builds | Project constraint from CLAUDE.md |
| `thiserror` | 2.0.18 | Library-facing typed errors | Project constraint — not anyhow in public API |
| `anyhow` | 1.0.102 | xtask/oracle tooling errors | Project convention |
| `serde_json` | workspace | Oracle artifact JSON | Already used in compare.rs and oracle_update.rs |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `approx` | workspace | Float comparison in tests | Use for atol/rtol assertions |
| `cc` | 1.2.x | Vendored libcint build | Oracle parity tests requiring `has_vendor_libcint` |

**Installation:** No new dependencies required — all needed crates are already in workspace.

## Architecture Patterns

### Recommended Project Structure

No structural changes to existing crate layout. New additions:

```
crates/cintx-compat/src/
├── workaround.rs           # NEW: int4c1e_via_2e_trace workaround (D-04)
└── lib.rs                  # Add: pub mod workaround (behind with-4c1e feature)

crates/cintx-cubecl/src/kernels/
└── center_4c1e.rs          # REPLACE: stub -> real 4c1e Gaussian overlap kernel

crates/cintx-oracle/src/
└── compare.rs              # MODIFY: tolerance unification + numeric helper oracle
```

### Pattern 1: Tolerance Unification (D-01)

Replace six per-family constants with one unified constant. All `tolerance_for_family()` arms use it.

```rust
// In compare.rs — replace all TOL_*_ATOL / TOL_*_RTOL constants:
const UNIFIED_ATOL: f64 = 1e-12;
const UNIFIED_RTOL: f64 = 1e-10;   // retain relative tolerance for families that use it
const ZERO_THRESHOLD: f64 = 1e-18; // unchanged
```

The `tolerance_for_family()` function arms for `1e`, `2e`, `2c2e`, `3c2e`, `3c1e`, `4c1e` all point to `UNIFIED_ATOL`.

### Pattern 2: Numeric Helper Oracle Assertions (HELP-01)

Integer-returning helpers are oracle-compared with exact equality against vendored libcint FFI values. CINTgto_norm uses float atol=1e-12. The existing `verify_helper_surface_coverage` function is extended to perform numeric comparisons using the same `OracleRawInputs::sample()` fixture data.

```rust
// In verify_helper_surface_coverage — numeric comparisons block:
let cintx_len = CINTlen_cart(2)?;
let vendor_len = vendor_ffi::vendor_CINTlen_cart(2);  // add to vendor_ffi.rs
assert_eq!(cintx_len, vendor_len as usize, "CINTlen_cart mismatch");

let cintx_norm = CINTgto_norm(1, 0.5);
let vendor_norm = vendor_ffi::vendor_CINTgto_norm(1, 0.5);
assert!((cintx_norm - vendor_norm).abs() <= 1e-12, "CINTgto_norm mismatch");
```

**Key insight for CINTgto_norm:** The current implementation in `helpers.rs` uses a lightweight approximation `(2.0 * a).powf((n as f64 + 1.5) * 0.5)`. This is NOT the full libcint formula. Vendored libcint `CINTgto_norm` includes a gamma function factor: `sqrt(fac * sqrt(M_PI) / pow(2*a, n+1.5))` where `fac = factorial2(2n-1)`. The current implementation will NOT pass oracle comparison at 1e-12. This is a correctness gap that must be fixed.

### Pattern 3: 4c1e G-tensor Kernel

The 4c1e overlap is NOT a Rys quadrature integral. From `g4c1e.c`: the G-tensor is filled with a polynomial recurrence in a 1D scratch buffer then remapped into the 4D G-tensor via the same ibase/kbase adaptive layout used by 2e. No Rys roots or weights are needed.

**Key algorithm difference from 2e:**

```
// 2e: requires Rys roots/weights, c00/b10/b01/b00 parameters per root
// 4c1e: single Gaussian overlap prefactor, polynomial recurrence in 1D buf

aijkl = aij + akl
fac = envs->fac[0] / (aijkl * sqrt(aijkl))   // gs = 1

// 1D scratch buf of size db*(max(nmax,mmax)+1) where db = nmax+mmax+1
// buf[0]=1, buf[1]=-r1r12*buf[0], buf[i+1]= 0.5*i/aijkl*buf[i-1] - r1r12*buf[i]
// then 2D shift fill, then map to g[i*dn + j*dm]
```

The G-tensor shape (ibase/kbase, di/dk/dl/dj strides, g_size) is identical to 2e. Reuse `build_2e_shape()` from `two_electron.rs` directly.

```rust
// In center_4c1e.rs — reuse build_2e_shape, replace fill with 4c1e polynomial:
fn fill_4c1e_g_tensor(
    g: &mut [f64],       // 3 * g_size (x/y/z interleaved)
    shape: &TwoEShape,   // reused from two_electron.rs
    ri: [f64; 3], rj: [f64; 3], rk: [f64; 3], rl: [f64; 3],
    ai: f64, aj: f64, ak: f64, al: f64,
    fac: f64,            // contraction prefactor
) { ... }
```

The HRR phase after G-tensor fill is the SAME 4-branch HRR from `two_electron.rs` — `hrr_lj2d_4d`, `hrr_kj2d_4d`, `hrr_il2d_4d`, `hrr_ik2d_4d`. These are reusable without modification.

### Pattern 4: workaround Module Structure (D-04)

```rust
// crates/cintx-compat/src/workaround.rs
// Feature-gated: #[cfg(feature = "with-4c1e")]

use crate::raw::{RawApiId, eval_raw};

/// Computes 4c1e-equivalent results by evaluating int2e_sph and tracing
/// over the (k,l) auxiliary shell pair.
pub unsafe fn int4c1e_via_2e_trace(
    out: &mut [f64],
    shls_4c: &[i32; 4],    // [i, j, k, l]
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<(), cintxRsError> { ... }
```

The trace/contraction: call `eval_raw(INT2E_SPH, ...)` with shls = [i, j, k, l], then sum over the kl diagonal. The output element `out[i,j]` = sum_m `int2e[i,j,m,m]`.

### Pattern 5: Spinor 4c1e Classification (D-05, 4C1E-03)

Current `validate_4c1e_envelope` in `raw.rs` checks: (1) with-4c1e feature, (2) representation cart/sph, (3) component rank, (4) dims, (5) max(l)<=4. Per D-05, spinor must be rejected FIRST (before the feature gate check), because the rejection reason should be about representation, not about missing feature flag.

```rust
// Updated order in validate_4c1e_envelope:
// 1. Check representation — if Spinor, return UnsupportedApi immediately
if matches!(representation, Representation::Spinor) {
    return Err(validated_4c1e_error("spinor representation not supported for 4c1e"));
}
// 2. Then check with-4c1e feature gate
if !cfg!(feature = "with-4c1e") { ... }
// 3. Then remaining checks (scalar rank, dims, max(l)<=4)
```

The same ordering change must be applied in `center_4c1e.rs::ensure_validated_4c1e`.

### Pattern 6: Oracle Fixture Coverage for 4c1e (4C1E-04)

The `raw_api_for_symbol` function in `compare.rs` already maps `int4c1e_cart` and `int4c1e_sph` to `RawApiId::INT4C1E_CART` and `INT4C1E_SPH`. The fixture matrix will include 4c1e symbols when the `with-4c1e` profile is active. No new symbol routing is needed — only the stub needs to be replaced with a real kernel so the comparison produces non-zero output.

The `vendor_ffi.rs` needs a `vendor_int4c1e_sph` wrapper (following the pattern of `vendor_int2e_sph`) for oracle gate testing under `CINTX_ORACLE_BUILD_VENDOR=1`.

### Anti-Patterns to Avoid

- **Separate code path for 4c1e:** D-03 requires reusing 2e VRR/HRR infrastructure. Do not duplicate `fill_g_tensor` or `build_2e_shape`.
- **Parallel tolerance constants:** D-01 requires a single `UNIFIED_ATOL`. Do not leave any per-family constant in compare.rs.
- **CINTgto_norm approximation:** The existing implementation in `helpers.rs` is wrong — it is a placeholder and must be replaced with the full libcint formula before HELP-01 can pass.
- **Tolerance as a knob:** D-02 says if a family fails at 1e-12, the kernel is buggy. Do not add per-symbol tolerance exceptions.
- **Spinor 4c1e after feature check:** Check spinor representation BEFORE checking the with-4c1e feature gate (D-05).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| G-tensor shape metadata for 4c1e | New shape struct | `build_2e_shape()` from `two_electron.rs` | Identical layout (ibase/kbase, same strides) |
| HRR for 4c1e | Separate HRR functions | `hrr_lj2d_4d`, `hrr_kj2d_4d`, `hrr_il2d_4d`, `hrr_ik2d_4d` from `two_electron.rs` | Identical algorithm, same 4-branch dispatch |
| Cartesian component enumeration | Custom iterator | `cart_comps()` from `two_electron.rs`/`center_3c1e.rs` | Same ordering, already tested |
| Fixture comparison infrastructure | Custom comparison | `diff_summary()`, `FamilyTolerance` from `compare.rs` | Already handles atol/rtol/zero_threshold |
| Vendored libcint FFI scaffolding | New build.rs logic | Existing `vendor_ffi.rs` pattern with `#[cfg(has_vendor_libcint)]` | Add new wrapper functions, don't change build |

**Key insight:** The 4c1e kernel shares its G-tensor layout (shape struct) and HRR phase exactly with 2e. Only the G-tensor fill differs — polynomial recurrence vs Rys quadrature. This is the pattern described in D-03.

## CINTgto_norm Correctness Gap

**This is the highest-risk item in the phase.** The current `helpers.rs` implementation:

```rust
pub fn CINTgto_norm(n: i32, a: f64) -> f64 {
    // Lightweight stable approximation used by compat parity checks.
    (2.0 * a).powf((n as f64 + 1.5) * 0.5)
}
```

The real libcint formula from `cint_bas.h` / `misc.c`:

```c
// CINTgto_norm: normalization of GTO with angular momentum n and exponent a
// = sqrt(fac * sqrt(pi) / pow(2*a, n+1.5))
// where fac = factorial2(2n-1) = (2n-1)!! = 1*3*5*...*(2n-1)
// Special case: fac for n=0 is 1, for n=1 is 1, for n=2 is 3, etc.
```

The correct Rust implementation must compute double factorial `(2n-1)!!` and use it in the normalization factor. The existing approximation will produce incorrect values for any `n >= 1` and will fail the atol=1e-12 oracle comparison. This must be fixed in this phase (HELP-01 requires it).

**Correct formula:**
```rust
pub fn CINTgto_norm(n: i32, a: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 || n < 0 {
        return 0.0;
    }
    // double factorial (2n-1)!! = 1 for n=0, 1 for n=1, 3 for n=2, 15 for n=3...
    let fac = double_factorial_2n_minus1(n);
    (fac * std::f64::consts::PI.sqrt() / (2.0 * a).powf(n as f64 + 1.5)).sqrt()
}
```

## Common Pitfalls

### Pitfall 1: 4c1e is Not Rys Quadrature
**What goes wrong:** Implementing 4c1e by analogy with 2e using Rys roots/weights produces zero or incorrect output.
**Why it happens:** The 4c1e operator is a four-center overlap integral (no 1/r_12 factor), so no electron-repulsion denominator means no Rys quadrature is needed.
**How to avoid:** `g4c1e.c` uses a 1D polynomial recurrence with a single prefactor `1/(aijkl*sqrt(aijkl))`. The buf[i] recurrence is `buf[i+1] = 0.5*i/aijkl*buf[i-1] - r1r12*buf[i]`. The HRR phase is identical to 2e.
**Warning signs:** If you see `rys_roots_host` calls in the 4c1e path, it is wrong.

### Pitfall 2: CINTgto_norm Formula
**What goes wrong:** The existing approximation passes positive-value checks but fails atol=1e-12 oracle comparison.
**Why it happens:** The approximation was written as a placeholder with a comment "Lightweight stable approximation used by compat parity checks" — it is not the libcint formula.
**How to avoid:** Implement the double factorial formula from libcint's `CINTgto_norm`. For n=0: norm = sqrt(sqrt(pi)/(2a)^1.5), for n=1: same as n=0 (df=1), for n=2: df=3, etc.
**Warning signs:** Any test calling `CINTgto_norm(2, 0.5)` and comparing against libcint will fail immediately.

### Pitfall 3: Spinor 4c1e Rejection Order
**What goes wrong:** Spinor 4c1e requests return "with-4c1e feature disabled" instead of "spinor representation not supported".
**Why it happens:** Current `validate_4c1e_envelope` checks feature gate before representation.
**How to avoid:** Move the spinor check to first position before the feature gate check. This is D-05 and required by 4C1E-03.
**Warning signs:** Tests for spinor rejection that assert the error contains "spinor" will fail if order is wrong.

### Pitfall 4: Legacy Wrapper Oracle Completeness
**What goes wrong:** `verify_helper_surface_coverage` passes (set equality) but numeric parity fails for some wrappers.
**Why it happens:** The existing gate only checks that the right symbols exist — it does not call each wrapper and compare output numerically.
**How to avoid:** Extend the helper_legacy_parity logic to call each wrapper with test fixtures and compare against vendored libcint output at atol=1e-12 (D-07). The `eval_legacy_symbol` dispatch in compare.rs already has mappings for 24 operator legacy symbols — verify coverage is complete.
**Warning signs:** Gate passes but oracle_parity_gate fails with mismatch for legacy symbols.

### Pitfall 5: nroots=1 for 4c1e
**What goes wrong:** Setting nroots for the G-tensor shape following the 2e formula produces wrong layout.
**Why it happens:** The 4c1e G-tensor in libcint uses `nrys_roots = 1` hardcoded (it does not use Rys quadrature at all). The shape struct from `build_2e_shape` derives nroots from `(li+lj+lk+ll)/2 + 1`, but this is wrong for 4c1e.
**How to avoid:** In the 4c1e kernel, force `nroots = 1` in the shape struct (or create a `build_4c1e_shape` that sets nroots=1 with the same ibase/kbase layout). The 1D scratch buffer `b_size = db * (max(nmax, mmax) + 1)` is separate from the G-tensor and only needed internally.
**Warning signs:** G-tensor indexing produces out-of-bounds panics or incorrect strides.

### Pitfall 6: Tolerance Unification Breaking 2c2e/3c1e
**What goes wrong:** Tightening 2c2e (was 1e-9) and 3c1e (was 1e-7) to 1e-12 fails existing oracle tests.
**Why it happens:** These families may have accumulated numerical error in the kernel implementation.
**How to avoid:** Run `cargo test -p cintx-oracle --features cpu` after the tolerance change to identify failures before committing. If 2c2e or 3c1e fail at 1e-12, the kernel must be fixed — D-02 is explicit: "the tolerance is immutable."
**Warning signs:** Oracle gate closure test failures for 2c2e or 3c1e immediately after tolerance unification.

## Code Examples

### 4c1e G-tensor Fill (from libcint g4c1e.c)

```rust
// Source: libcint-master/src/g4c1e.c::CINTg4c1e_ovlp
// Key: uses polynomial 1D recurrence in scratch buf, NOT Rys roots

fn fill_4c1e_g_tensor(
    g: &mut [f64],         // 3 * g_size elements (x, y, z axes concatenated)
    shape: &FourC1eShape,  // ibase/kbase, dn/dm, g_size — nroots=1
    ri: [f64; 3], rk: [f64; 3],    // base centers (ibase ? ri : rj, kbase ? rk : rl)
    rij: [f64; 3], rkl: [f64; 3],  // pair centers (weighted midpoints)
    aijkl: f64,
    fac: f64,              // = 1.0 / (aijkl * sqrt(aijkl)) * contraction_prefactors
) {
    let nmax = shape.li + shape.lj;    // from angular momenta
    let mmax = shape.lk + shape.ll;
    let db = nmax + mmax + 1;
    let b_size = db * (nmax.max(mmax) + 1);
    let mut buf = vec![0.0_f64; b_size * 3];  // x/y/z scratch

    for axis in 0..3 {
        let bx = &mut buf[axis * b_size..][..b_size];
        bx[0] = if axis == 2 { fac } else { 1.0 };

        let (r1r12, r1r2) = if nmax >= mmax {
            // center = weighted midpoint from i side
            let r1r12 = ri[axis] - (shape.aij * rij[axis] + shape.akl * rkl[axis]) / aijkl;
            let r1r2 = ri[axis] - rk[axis];
            (r1r12, r1r2)
        } else {
            let r1r12 = rk[axis] - (shape.aij * rij[axis] + shape.akl * rkl[axis]) / aijkl;
            let r1r2 = rk[axis] - ri[axis];
            (r1r12, r1r2)
        };

        if nmax + mmax > 0 {
            bx[1] = -r1r12 * bx[0];
        }
        for i in 1..(nmax + mmax) {
            bx[i + 1] = 0.5 * i as f64 / aijkl * bx[i - 1] - r1r12 * bx[i];
        }
        // 2D shift fill then remap to g[i*dn + j*dm]
        // ... (follow g4c1e.c pattern exactly)
    }
}
```

### Tolerance Unification (compare.rs)

```rust
// Source: compare.rs — replace all per-family constants
const UNIFIED_ATOL: f64 = 1e-12;
const UNIFIED_RTOL: f64 = 1e-10;
const ZERO_THRESHOLD: f64 = 1e-18;

pub fn tolerance_for_family(family: &str) -> Result<FamilyTolerance> {
    let family_name: &'static str = match family {
        "1e" => "1e",
        "2e" | "unstable::source::2e" => "2e",
        "2c2e" => "2c2e",
        "3c2e" => "3c2e",
        "3c1e" => "3c1e",
        "4c1e" => "4c1e",
        other => bail!("missing family tolerance for `{other}`"),
    };
    Ok(FamilyTolerance {
        family: family_name,
        atol: UNIFIED_ATOL,
        rtol: UNIFIED_RTOL,
        zero_threshold: ZERO_THRESHOLD,
    })
}
```

### Workaround Module Structure (D-04)

```rust
// Source: decision D-04 from CONTEXT.md
// crates/cintx-compat/src/workaround.rs

#[cfg(feature = "with-4c1e")]
pub mod int4c1e_compat {
    use super::*;

    /// Compute 4c1e integral (i,j|kl) by evaluating the 2e ERI (ij|kl)
    /// and tracing over the k=l diagonal.
    ///
    /// Result matches direct int4c1e evaluation at atol=1e-12 for
    /// cart/sph representations within the Validated4C1E envelope.
    pub unsafe fn int4c1e_via_2e_trace(
        out: &mut [f64],
        shls_4c: &[i32; 4],
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
    ) -> Result<(), cintxRsError> {
        // Allocate 2e buffer of size ni * nj * nk * nl
        // Call eval_raw(INT2E_SPH, ...) with same shls
        // Contract: out[a] = sum_m int2e_buf[a, m, m] (trace over kl)
        todo!("implement workaround trace")
    }
}
```

### CINTgto_norm Correct Formula

```rust
// Source: libcint misc.c / cint_bas.h
pub fn CINTgto_norm(n: i32, a: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 || n < 0 {
        return 0.0;
    }
    // fac = (2n-1)!! = double factorial
    let fac = {
        let mut f = 1.0_f64;
        let mut k = 2 * n - 1;
        while k > 0 {
            f *= k as f64;
            k -= 2;
        }
        f
    };
    (fac * std::f64::consts::PI.sqrt() / (2.0 * a).powf(n as f64 + 1.5)).sqrt()
}
```

## Gap Analysis Summary

Running the manifest-based gap analysis (D-06) against current compare.rs arrays:

| Kind | Manifest count | compare.rs IMPLEMENTED_* count | Gap |
|------|---------------|--------------------------------|-----|
| Helper | 17 | 17 (IMPLEMENTED_HELPER_SYMBOLS) | None — symbols complete; oracle numeric comparison missing |
| Transform | 7 | 7 (IMPLEMENTED_TRANSFORM_SYMBOLS) | None — symbols complete; oracle numeric comparison missing |
| Optimizer | 7 | 7 (IMPLEMENTED_OPTIMIZER_SYMBOLS) | None — symbols complete |
| Legacy | 45 | 45 (LEGACY_WRAPPER_SYMBOLS in legacy.rs) | None — symbols complete; numeric oracle may be incomplete |
| Operator (4c1e) | 2 | 2 mapped in raw_api_for_symbol | Kernel stub — replaces zeros with real output |

The key insight: **all symbols are already implemented** at the code level. Phase 11's work is:
1. Adding numeric oracle comparison (HELP-01, HELP-02, HELP-03)
2. Fixing CINTgto_norm formula
3. Replacing center_4c1e.rs stub with real kernel
4. Adding workaround module
5. Unifying tolerances
6. Fixing spinor 4c1e classification order
7. Verifying CI gates pass at 1e-12

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-family tolerance constants | Unified atol=1e-12 | Phase 11 (D-01) | Stricter verification across all families |
| center_4c1e.rs stub (zeros) | Real polynomial recurrence kernel | Phase 11 | int4c1e_sph produces libcint-compatible values |
| Spinor check after feature gate | Spinor check first in classifier | Phase 11 (D-05) | Correct error message for spinor 4c1e requests |
| Helper oracle = set equality only | Helper oracle = set + numeric comparison | Phase 11 | HELP-01/02/03 requirements satisfied |

## Runtime State Inventory

Step 2.5: SKIPPED — This is a correctness/kernel-implementation phase, not a rename/refactor/migration phase. No stored data, live service config, OS-registered state, secrets, or build artifacts carry renamed strings.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust 1.94.0 | All kernel code | Check via `rustup show` | Pinned 1.94.0 | None — pin is mandatory |
| CubeCL cpu feature | Oracle tests (#[cfg(feature="cpu")]) | Available | 0.9.x | None — cpu feature needed for oracle gate |
| libcint-master source | Oracle build (CINTX_ORACLE_BUILD_VENDOR=1) | Available at `.claude/worktrees/*/libcint-master` and standard path | 6.1.3 | Idempotency tests (no vendored FFI) |
| `has_vendor_libcint` cfg | `oracle_gate_all_five_families` test, new helper oracle tests | Set by CINTX_ORACLE_BUILD_VENDOR=1 | — | Skip numeric vendor comparison; use idempotency |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** Vendored libcint FFI — if CINTX_ORACLE_BUILD_VENDOR=1 is not set, numeric oracle comparisons against upstream are skipped; idempotency tests still run.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + cargo nextest |
| Config file | `rust-toolchain.toml` (1.94.0 pin) |
| Quick run command | `cargo test -p cintx-compat -p cintx-oracle --features cpu` |
| Full suite command | `cargo test --workspace --features cpu && cargo run --manifest-path xtask/Cargo.toml -- helper-legacy-parity --profile base` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HELP-01 | Helper symbols return integer-exact values matching libcint | unit | `cargo test -p cintx-oracle --features cpu -- helper_numeric_oracle` | No — Wave 0 |
| HELP-02 | Transform symbols return float values matching libcint at 1e-12 | unit | `cargo test -p cintx-oracle --features cpu -- transform_numeric_oracle` | No — Wave 0 |
| HELP-03 | Legacy wrapper symbols match libcint at 1e-12 | integration | `cargo test -p cintx-oracle --features cpu` (via parity report) | Yes (partial) |
| HELP-04 | helper-legacy-parity gate passes all 4 profiles | integration | `cargo run --manifest-path xtask/Cargo.toml -- helper-legacy-parity --profile base` | Yes |
| 4C1E-01 | int4c1e_sph real kernel matches libcint at 1e-12 | unit | `cargo test -p cintx-cubecl --features cpu -- center_4c1e` | No — Wave 0 |
| 4C1E-02 | workaround matches real kernel at 1e-12 | unit | `cargo test -p cintx-compat --features with-4c1e,cpu -- workaround` | No — Wave 0 |
| 4C1E-03 | Spinor 4c1e returns UnsupportedApi before checking angular momentum | unit | `cargo test -p cintx-compat --features with-4c1e -- int4c1e_spinor_rejected` | Partial (wrong order) |
| 4C1E-04 | oracle_parity_gate with-4c1e profile passes at 1e-12 | integration | `cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles with-4c1e --include-unstable-source false` | Yes (fails until kernel fixed) |

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-compat -p cintx-cubecl -p cintx-oracle --features cpu`
- **Per wave merge:** Full suite + xtask helper-legacy-parity for base profile
- **Phase gate:** All 4 profiles oracle-compare + helper-legacy-parity + oracle_gate_closure before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/src/compare.rs` — numeric helper oracle comparison functions (covers HELP-01, HELP-02)
- [ ] `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — real kernel test (covers 4C1E-01)
- [ ] `crates/cintx-compat/src/workaround.rs` — new file with workaround test (covers 4C1E-02)

## Open Questions

1. **2c2e/3c1e tolerance unification risk**
   - What we know: 2c2e was at atol=1e-9, 3c1e was at atol=1e-7; both are now required to be 1e-12 per D-01.
   - What's unclear: Whether the Phase 10 kernel implementations for these families can actually achieve 1e-12 or whether there are residual numerical gaps.
   - Recommendation: Run the oracle gate immediately after tolerance unification before doing other work. If 2c2e or 3c1e fail at 1e-12, diagnose and fix the kernel before proceeding. D-02 says this is mandatory.

2. **CINTgto_norm double factorial boundary**
   - What we know: The correct formula requires `(2n-1)!!` for n >= 0.
   - What's unclear: Exact libcint treatment of n=0 (double factorial of -1 is 1 by convention) — need to verify against libcint's misc.c.
   - Recommendation: Read `libcint-master/src/misc.c::CINTgto_norm` directly to confirm n=0 handling before implementing.

3. **Workaround trace correctness for multi-contracted shells**
   - What we know: The trace contracts k=l diagonal of a 2e ERI result.
   - What's unclear: Whether the trace over contracted indices is a simple diagonal sum or requires handling multiple contraction components.
   - Recommendation: Start with single-contraction shells for oracle validation; generalize once single-ctr case passes at 1e-12.

## Sources

### Primary (HIGH confidence)
- `crates/cintx-oracle/src/compare.rs` — Current tolerance constants, IMPLEMENTED_*_SYMBOLS arrays, gap analysis
- `crates/cintx-compat/src/helpers.rs` — Current helper implementations including CINTgto_norm approximation
- `crates/cintx-compat/src/legacy.rs` — LEGACY_WRAPPER_SYMBOLS, macro pattern
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — Stub implementation
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — Rys quadrature + VRR/HRR to reuse
- `.claude/worktrees/agent-a269eba5/libcint-master/src/g4c1e.c` — Authoritative 4c1e G-tensor algorithm
- `.claude/worktrees/agent-a269eba5/libcint-master/src/cint4c1e.c` — CINT4c1e_loop_nopt contraction loop
- `crates/cintx-ops/src/generated/api_manifest.rs` — Manifest symbol counts (17 Helper, 7 Transform, 7 Optimizer, 45 Legacy, 2 Operator 4c1e)
- `.github/workflows/compat-governance-pr.yml` — Existing CI gate structure (helper_legacy_parity_gate, oracle_parity_gate)
- `.planning/phases/11-helper-transform-completion-4c1e-real-kernel/11-CONTEXT.md` — All locked decisions D-01 through D-09

### Secondary (MEDIUM confidence)
- `xtask/src/oracle_update.rs` — run_helper_legacy_parity behavior (currently calls verify_helper_surface_coverage only)
- `crates/cintx-oracle/src/vendor_ffi.rs` — Existing vendor FFI wrapper pattern for adding new helper/4c1e wrappers

### Tertiary (LOW confidence)
- None — all claims are grounded in direct source code inspection.

## Metadata

**Confidence breakdown:**
- Gap analysis: HIGH — confirmed by direct source inspection of manifest and compare.rs arrays
- 4c1e algorithm: HIGH — confirmed by reading g4c1e.c and cint4c1e.c source
- CINTgto_norm bug: HIGH — current implementation has explicit "approximation" comment and wrong formula
- Spinor ordering bug: HIGH — validated_4c1e_envelope code read directly
- Tolerance unification risk: MEDIUM — whether 2c2e/3c1e kernels pass at 1e-12 is untested

**Research date:** 2026-04-04
**Valid until:** Stable — no external library changes expected; valid until Phase 12 changes the codebase
