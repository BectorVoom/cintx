# Phase 12: Real Spinor Transform (c2spinor Replacement) - Research

**Researched:** 2026-04-04
**Domain:** Clebsch-Gordan cart-to-spinor transform; libcint 6.1.3 c2spinor coefficient extraction; spinor oracle parity
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Extract Clebsch-Gordan coupling coefficient tables directly from upstream libcint's `cart2sph.c` (`g_trans_cart2jR[]` / `g_trans_cart2jI[]` arrays), mirroring the approach used in `c2s.rs` which extracted from `cart2sph.c`. This guarantees oracle parity by construction.
- **D-02:** Coefficients live in a separate `c2spinor_coeffs.rs` file within `crates/cintx-cubecl/src/transform/`. The `c2spinor.rs` module imports and applies them.
- **D-03:** All four `CINTc2s_*spinor*` variants (`ket_spinor_sf1`, `iket_spinor_sf1`, `ket_spinor_si1`, `iket_spinor_si1`) get distinct code paths matching upstream `c2spinor.c`. `ket` vs `iket` differs by conjugation sign; `_sf` (spin-free) vs `_si` (spin-included) differs by which CG coupling matrix is applied. No shared-core-with-flags abstraction.
- **D-04:** Spinor staging buffer maintains the existing interleaved real/imaginary layout `[re0, im0, re1, im1, ...]`.
- **D-05:** Verification is sequenced: land 1e spinor oracle parity first (overlap, kinetic, nuclear attraction), then extend to 2e, 2c2e, 3c1e, 3c2e in a second plan.
- **D-06:** Spinor oracle tests expand the existing `oracle_gate_closure.rs` by adding spinor representation to the fixture generation loop.
- **D-07:** kappa selects which rows of the CG coupling matrix to apply. `kappa < 0` uses the j=l+1/2 block (`cart2j_gt_lR/I`), `kappa > 0` uses the j=l-1/2 block (`cart2j_lt_lR/I`), `kappa = 0` uses both blocks (total components = 4l+2).
- **D-08:** Existing stub tests (amplitude-averaging, buffer-length-only) are deleted and replaced with value-correctness tests that compare against known CG-transformed outputs.

### Claude's Discretion

- Internal factoring of coefficient application loops within each variant
- Oracle fixture molecule/shell choices for spinor tests (likely reuse existing H2O/STO-3G fixtures with kappa variants)
- Order of variant implementation within each plan
- Exact plan boundaries between 1e and multi-center family coverage

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SPIN-01 | Cart-to-spinor transform implements real Clebsch-Gordan coupling coefficients for all angular momenta up to g-function (l=4) | Coefficient tables are in `libcint-master/src/cart2sph.c` at known offsets for l=0..4; struct `g_c2s[]` maps l to (cart2j_gt_lR, cart2j_gt_lI, cart2j_lt_lR, cart2j_lt_lI). |
| SPIN-02 | All CINTc2s_*spinor* transform variants implemented | Four functions identified in cart2sph.c: `CINTc2s_ket_spinor_sf1`, `CINTc2s_iket_spinor_sf1`, `CINTc2s_ket_spinor_si1`, `CINTc2s_iket_spinor_si1`. All four already exist as stubs in `cintx-compat/src/transform.rs`. |
| SPIN-03 | Spinor-form base family evaluations match libcint to atol=1e-12 | Oracle infrastructure exists for 1e and multi-center families; vendor FFI spinor helpers already wrapped; interleaved buffer layout already supported in `fixtures.rs` via `complex_interleaved`. |
| SPIN-04 | kappa parameter correctly interpreted in spinor transform dispatch | `spinor_len(l, kappa)` in `shell.rs` already correct; kappa routing matches libcint `_len_spinor()` exactly. |
</phase_requirements>

---

## Summary

Phase 12 replaces a provably incorrect amplitude-averaging stub in `c2spinor.rs` with a complete Clebsch-Gordan cart-to-spinor transform. The upstream libcint source (`libcint-master/src/cart2sph.c`) is available in the repository and provides the authoritative CG coefficient tables (`g_trans_cart2jR[]` / `g_trans_cart2jI[]`) as flat arrays indexed by known offsets. The extraction pattern is identical to what `c2s.rs` already does for Condon-Shortley coefficients, so the structural template is validated and understood.

The key insight from reading the upstream source is that the four `CINTc2s_*spinor*1` public functions differ in exactly two independent axes: (1) `sf` vs `si` determines whether the sigma (Pauli) coupling terms (vx, vy, vz) are included in the matrix-vector product, and (2) `ket` vs `iket` determines the sign/order of real and imaginary parts in the output accumulation. This means the implementation can factor the CG matrix lookup and loop structure into helper functions while keeping the four accumulation formulas distinct. The compat entry points in `transform.rs` already carry the correct signatures but delegate to the stub; they need their interiors replaced with the real transform.

Oracle verification infrastructure is fully in place: vendored libcint FFI is built with `CINTX_ORACLE_BUILD_VENDOR=1`, spinor symbols (`int1e_ovlp_spinor`, etc.) are present in `raw_api_for_symbol` and `eval_legacy_symbol`, `OracleFixture` already sets `complex_interleaved: true` for spinor representations, and `assert_flat_buffer_contract` has a spinor-specific check. The only gaps are the missing vendor FFI wrappers for spinor integral functions and the fact that `oracle_gate_closure.rs` currently has no test exercising spinor representation.

**Primary recommendation:** Extract CG coefficient tables from `cart2sph.c` into `c2spinor_coeffs.rs` using the documented offsets in `g_c2s[]`, implement the four transform variants in `c2spinor.rs` following the libcint formulas exactly, then expand `oracle_gate_closure.rs` with spinor representation for 1e families first.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `libcint-master/src/cart2sph.c` | 6.1.3 (vendored) | Authoritative CG coefficient source | The upstream source is in the repo; extracting from it guarantees parity by construction |
| Rust std f64 arithmetic | stable | Coefficient application loops | No GPU launch needed — this is a CPU-side staging transform |
| `cintx-core` | workspace | `cintxRsError`, `Representation` | Error surface and enum already established |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cintx-oracle` cpu feature | workspace | Oracle comparison gate | Test path: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu` |
| `approx` | workspace | Floating-point comparison in unit tests | Use for coefficient value sanity checks in `c2spinor_coeffs.rs` tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extracting from `g_trans_cart2jR[]` | Computing CG coefficients analytically | Analytic computation is error-prone and harder to audit; extraction from upstream tables is O(0) risk |
| Four distinct code paths | Shared-core-with-flags | Flags create subtle bugs when upstream diverges; distinct paths match libcint exactly and are easier to diff |

---

## Architecture Patterns

### Recommended Project Structure

```
crates/cintx-cubecl/src/transform/
├── c2spinor_coeffs.rs   # NEW: CG coefficient arrays (gt/lt, R/I, l=0..4)
├── c2spinor.rs          # REWRITE: four transform variants using coeffs
└── mod.rs               # unchanged — dispatch already routes Spinor arm
```

```
crates/cintx-compat/src/transform.rs   # REWRITE interiors of four spinor entry points
crates/cintx-oracle/src/vendor_ffi.rs  # ADD vendor wrappers for spinor integrals
crates/cintx-oracle/tests/oracle_gate_closure.rs  # ADD spinor fixture loop
```

### Pattern 1: Coefficient Table Layout (from libcint g_c2s struct)

**What:** Each angular momentum l has four sub-arrays: `cart2j_gt_lR`, `cart2j_gt_lI`, `cart2j_lt_lR`, `cart2j_lt_lI`. Each is a flat array indexed as `[spinor_row * nf * 2 + alpha_or_beta_half * nf + cart_col]` where alpha half = first nf entries, beta half = second nf entries.

**Layout (from libcint g_c2s offsets):**

| l | `cart2j_gt_lR` offset | `cart2j_lt_lR` offset | gt rows (2l+2) | lt rows (2l) |
|---|---|---|---|---|
| 0 | 0   | 0   | 2  | 0 (empty) |
| 1 | 4   | 16  | 4  | 2 |
| 2 | 40  | 88  | 6  | 4 |
| 3 | 160 | 280 | 8  | 6 |
| 4 | 440 | 680 | 10 | 8 |

`gt` means j=l+1/2 (kappa<0), `lt` means j=l-1/2 (kappa>0).

**Rust const layout:**

```rust
// Source: libcint-master/src/cart2sph.c g_trans_cart2jR[]/g_trans_cart2jI[]
// For each l: [[alpha_row_0: [cart_0..ncart-1], beta_row_0: [cart_0..ncart-1]], row_1..., ...]
// gt (j=l+1/2, kappa<0): nd = 2l+2 spinor rows, each row has 2*ncart(l) entries (alpha+beta)
// lt (j=l-1/2, kappa>0): nd = 2l   spinor rows, each row has 2*ncart(l) entries (alpha+beta)
pub const CJ_GT_L2_R: [[f64; 12]; 6] = [...];  // 6 rows × (2×6) = 6 rows × 12
pub const CJ_GT_L2_I: [[f64; 12]; 6] = [...];
pub const CJ_LT_L2_R: [[f64; 12]; 4] = [...];  // 4 rows × 12
pub const CJ_LT_L2_I: [[f64; 12]; 4] = [...];
```

### Pattern 2: sf (spin-free) Transform Core (from libcint CINTc2s_ket_spinor_sf1)

**What:** Apply CG matrix to scalar cartesian integral values only (no vx/vy/vz components).

```rust
// Source: libcint-master/src/cart2sph.c CINTc2s_ket_spinor_sf1 (line 6741)
// gsp_re[j + i*lds] = sum_n caR[i,n] * v1[n]
// gsp_im[j + i*lds] = sum_n caI[i,n] * v1[n]   (note: upstream sign is +caI, not -caI for _sf1)
// gspb_re[j + i*lds] = sum_n cbR[i,n] * v1[n]
// gspb_im[j + i*lds] = sum_n cbI[i,n] * v1[n]
fn apply_sf(coeff_r: &[f64], coeff_i: &[f64], cart: &[f64], nd: usize, nf: usize) -> Vec<f64> { ... }
```

**CRITICAL: sign convention in sf1 vs iket_sf1:**

From the libcint source (confirmed by reading lines 6741-6837):
- `ket_spinor_sf1`: `gspaz_re += caR * v1`, `gspaz_im += caI * v1` (positive imaginary)
- `iket_spinor_sf1`: `gspaz_re -= caI * v1`, `gspaz_im += caR * v1` (i-multiplied — swapped and sign-flipped)

The imaginary unit `i` acts on the spinor component: `iket` = multiply spinor output by `i`, which maps `(re, im) -> (-im, re)`.

### Pattern 3: si (spin-included) Transform Core

**What:** Apply CG matrix to all four Pauli-coupled cartesian components (v1, vx, vy, vz).

```rust
// Source: libcint-master/src/cart2sph.c CINTc2s_ket_spinor_si1 (line 6839)
// alpha spinor row i:
//   gspaz_re += caR*v1 - caI*vz + cbR*vy - cbI*vx
//   gspaz_im += caI*v1 + caR*vz + cbI*vy + cbR*vx
// beta spinor row i:
//   gspbz_re += cbR*v1 + cbI*vz - caR*vy - caI*vx
//   gspbz_im += cbI*v1 - cbR*vz - caI*vy + caR*vx
```

**iket_spinor_si1 sign convention** (lines 6899-6957):
- Same Pauli terms but output multiplied by `i`: `(re,im) -> (-im, re)` for each spinor component.

### Pattern 4: kappa Dispatch

```rust
// Source: libcint-master/src/cart2sph.c _len_spinor() line 3537
// Mirrors existing spinor_len() in cintx-core/src/shell.rs
fn select_cg_block(l: u8, kappa: i32) -> (&'static [f64], &'static [f64]) {
    if kappa < 0 {
        // j = l + 1/2, use cart2j_gt_lR / cart2j_gt_lI
        (CG_GT_R[l], CG_GT_I[l])
    } else {
        // j = l - 1/2, use cart2j_lt_lR / cart2j_lt_lI
        (CG_LT_R[l], CG_LT_I[l])
    }
    // kappa == 0: both blocks applied sequentially; nd = 4l+2
}
```

### Pattern 5: Compat Entry Point Signature Preservation

The four compat functions in `transform.rs` already have the correct signatures from upstream. Their interiors need to change from calling the stub to calling the real transform with `l` and `kappa` passed through:

```rust
// Source: crates/cintx-compat/src/transform.rs (current stubs)
pub fn CINTc2s_ket_spinor_sf1(
    gsp: &mut [f64], gcart: &[f64], lds: i32, ldc: i32, nctr: i32, l: i32, kappa: i32
) -> Result<(), cintxRsError> { ... }
```

The interior calls `c2spinor::cart_to_spinor_sf(gsp, gcart, lds as usize, ldc as usize, nctr as usize, l as u8, kappa)`.

### Anti-Patterns to Avoid

- **Sharing one accumulation loop with a "is_iket" bool flag:** Libcint uses distinct functions for a reason — the accumulation formulas are different enough that shared code obscures the distinction and risks copy-paste sign errors.
- **Placing CG tables inline in c2spinor.rs:** The tables are large (hundreds of f64 values for l=3 and l=4). Keeping them in `c2spinor_coeffs.rs` follows the D-02 decision and mirrors the `c2s.rs` pattern.
- **Confusing alpha/beta spinor halves:** Upstream uses `gspa` and `gspb` as separate pointers for alpha and beta halves. In the interleaved staging layout, alpha half occupies `buf[0..nd*2]` and beta half occupies `buf[nd*2..nd*4]` (where `nd = spinor_len(l, kappa)` and each element is an interleaved re/im pair).
- **Forgetting that the staging buffer has size `spinor_len(l, kappa) * 2` (interleaved re/im), doubled again for alpha+beta:** Total staging elements = `spinor_len(l, kappa) * 4`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CG coefficient values | Custom CG computation formulas | Direct extraction from `g_trans_cart2jR[]` / `g_trans_cart2jI[]` in `cart2sph.c` | Upstream tables are already validated against oracle; manual computation has floating-point precision risk |
| kappa component count | Custom formula | Existing `spinor_len(l, kappa)` in `shell.rs` | Already correct and tested |
| Oracle comparison infrastructure | New parity test harness | Extend existing `oracle_gate_closure.rs` per D-06 | Full oracle infrastructure is in place; adding spinor is adding one representation to the fixture loop |

**Key insight:** The libcint source tree is local at `libcint-master/`. There is no need to look up CG coefficients from a reference text — the numbers can be read directly from the source at known offsets verified in `g_c2s[]`.

---

## Common Pitfalls

### Pitfall 1: Confusing sf and si accumulation formulas

**What goes wrong:** Copy-pasting the sf formula into si and forgetting to add the vx/vy/vz sigma coupling terms. Produces wrong results for any non-s orbital but passes s-shell checks (since vx=vy=vz=0 for s).
**Why it happens:** Both sf and si look structurally similar at the loop level.
**How to avoid:** Implement si by diffing against the upstream `CINTc2s_ket_spinor_si1` code directly (line 6839). Add at least one p-shell or d-shell value-correctness test that would catch sigma-term omission.
**Warning signs:** All `int1e_ovlp_spinor` results (s-shell dominated) pass but `int1e_kin_spinor` p-shell results fail.

### Pitfall 2: Wrong alpha/beta buffer split in staging layout

**What goes wrong:** The alpha and beta halves of the spinor output are placed back-to-back in libcint's output, but cintx uses interleaved re/im pairs. The mapping is:
- alpha component `i`: `staging[i*2]` (re), `staging[i*2+1]` (im)
- beta component `i`: `staging[(nd+i)*2]` (re), `staging[(nd+i)*2+1]` (im)
**Why it happens:** Libcint internally uses separate `gspaR/gspaI/gspbR/gspbI` arrays then interleaves for output. The staging buffer must mirror this interleaving exactly.
**How to avoid:** Write a unit test with known-good CG values and verify element positions explicitly.
**Warning signs:** Even indices (re parts) pass but odd indices (im parts) are mismatched.

### Pitfall 3: iket sign confusion

**What goes wrong:** `iket` multiplies the spinor output by the imaginary unit `i`. This maps `(re, im) -> (-im, re)`, not `(im, -re)` or any other permutation.
**Why it happens:** The sign transformation for complex multiplication by `i` is easy to mis-remember.
**How to avoid:** Check against libcint lines 6826-6829 for `iket_spinor_sf1` and lines 6944-6947 for `iket_spinor_si1`. The real part of the output accumulates `-(caI * v1)` (for sf) instead of `caR * v1`.
**Warning signs:** `ket` variants pass oracle but `iket` variants are off by sign on the imaginary component.

### Pitfall 4: kappa=0 handling

**What goes wrong:** When `kappa == 0`, both j=l+1/2 and j=l-1/2 blocks are applied and the output has `nd = 4l+2` components. Implementations that only check `kappa < 0` vs `kappa > 0` will produce a wrong-sized output for `kappa = 0`.
**Why it happens:** The zero case is easy to miss since most practical use has `kappa != 0`.
**How to avoid:** Handle three branches explicitly: `kappa < 0` (gt block only, nd=2l+2), `kappa > 0` (lt block only, nd=2l), `kappa == 0` (both blocks, nd=4l+2). Confirm against `_len_spinor()` in libcint (line 3537-3546) which matches `spinor_len()` in `shell.rs`.
**Warning signs:** `kappa=0` tests have buffer length mismatches.

### Pitfall 5: Staging buffer size for multi-center spinor

**What goes wrong:** The 1e staging buffer is `spinor_len(l, kappa) * 2` (re/im interleaved) per shell pair. For multi-center families (2e, 3c1e, etc.) the buffer must be the product of all shell component counts, each multiplied by 2 for interleaving. The `* 2` factor applies to the full multi-dimensional output, not per-shell.
**Why it happens:** c2s staging for sph is simpler (real only); spinor introduces the complex interleaving as a new concern.
**How to avoid:** Reference `OracleFixture.required_elements()` in `fixtures.rs` which already calls `complex_interleaved.then(|| base * 2)` correctly.
**Warning signs:** Buffer too-small errors from executor staging path on 2e+ spinor tests.

---

## Code Examples

### Extracting CG Coefficients from Upstream Arrays

```rust
// Source: libcint-master/src/cart2sph.c, g_c2s[] struct at line 3561
// For l=2: cart2j_gt_lR is at g_trans_cart2jR+40, with 6 spinor rows (2l+2=6)
// Each row has 2*ncart(2) = 12 entries (alpha half: 6, beta half: 6)
// Total: 6 * 12 = 72 entries
//
// Layout: coeff_R[spinor_row * nf * 2 + alpha_or_beta * nf + cart_col]
// where alpha_or_beta=0 for alpha spinor half, =1 for beta spinor half
pub const CJ_GT_L2_R: [[f64; 12]; 6] = [
    // row 0 (spinor component 0): [alpha_cart0..5, beta_cart0..5]
    [...],
    // rows 1..5
];
```

### sf Transform Implementation Pattern

```rust
// Source: libcint-master/src/cart2sph.c CINTc2s_ket_spinor_sf1 lines 6741-6788
pub fn cart_to_spinor_sf(
    gsp: &mut [f64],   // interleaved re/im output, len = nd * 4 (alpha+beta, re+im)
    cart: &[f64],      // flat cartesian input, len = ncart(l)
    l: u8,
    kappa: i32,
) {
    let nf = ncart(l);
    let nd = spinor_len(l as usize, kappa as i16);
    let (coeff_r, coeff_i) = select_cg_block(l, kappa);

    // Alpha half occupies gsp[0..nd*2], beta half gsp[nd*2..nd*4]
    for i in 0..nd {
        let mut sa_re = 0.0f64;
        let mut sa_im = 0.0f64;
        let mut sb_re = 0.0f64;
        let mut sb_im = 0.0f64;
        for n in 0..nf {
            let v1 = cart[n];
            let ca_r = coeff_r[i * nf * 2 + n];           // alpha row
            let ca_i = coeff_i[i * nf * 2 + n];
            let cb_r = coeff_r[i * nf * 2 + nf + n];      // beta row
            let cb_i = coeff_i[i * nf * 2 + nf + n];
            sa_re += ca_r * v1;
            sa_im += ca_i * v1;   // note: upstream _sf1 uses +caI, not -caI
            sb_re += cb_r * v1;
            sb_im += cb_i * v1;
        }
        gsp[i * 2]             = sa_re;
        gsp[i * 2 + 1]         = sa_im;
        gsp[(nd + i) * 2]      = sb_re;
        gsp[(nd + i) * 2 + 1]  = sb_im;
    }
}
```

### iket Conjugation Pattern

```rust
// Source: libcint-master/src/cart2sph.c CINTc2s_iket_spinor_sf1 lines 6790-6837
// iket = multiply spinor output by i: (re, im) -> (-im, re)
// In the accumulation: instead of caR*v1, caI*v1 -> write: -(caI*v1), (caR*v1)
gsp[i * 2]     -= ca_i * v1;   // re = -(caI * v1)
gsp[i * 2 + 1] += ca_r * v1;  // im =  (caR * v1)
```

### Oracle Fixture Extension (spinor representation loop)

```rust
// Source: pattern from existing oracle_gate_closure.rs + fixtures.rs
// Add to build_h2o_sto3g() fixture loop — representation already handled by:
//   representation: Representation::Spinor in shell/bas construction
// The spinor_len() helper already gives correct CINTcgto_spinor(shell, bas) count
// vendor_ffi.rs needs wrappers:
pub fn vendor_int1e_ovlp_spinor(out: &mut [f64], shls: &[i32; 2], ...) -> i32 {
    unsafe { ffi::int1e_ovlp(out.as_mut_ptr(), ptr::null_mut(), ...) }
}
// Note: libcint spinor integrals output complex (interleaved doubles),
// so output buffer must be sized ni*nj*2 where ni = CINTcgto_spinor(shls[0])
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Amplitude-averaging stub (wrong) | Real CG coupling (this phase) | Phase 12 | All spinor oracle results become valid |
| Per-family tolerance exceptions | Unified atol=1e-12 everywhere | v1.2 roadmap | Spinor must hit 1e-12 — no exceptions |
| Buffer-length-only stub tests | Value-correctness tests against CG-known outputs | Phase 12 (D-08) | Regressions detectable at coefficient level |

**Deprecated/outdated:**
- The amplitude-averaging stub `(pair[0].abs() + pair[1].abs()) * 0.5` is provably wrong per STATE.md v1.2 roadmap decision. All tests that only check buffer length (not values) must be deleted.

---

## Open Questions

1. **kappa=0 production callers**
   - What we know: `spinor_len(l, 0) = 4l+2`; both CG blocks must be applied sequentially; `_len_spinor(0, l)` in libcint also returns `4l+2`.
   - What's unclear: Whether any existing eval_raw call in the test suite uses `kappa=0` shells to exercise this path.
   - Recommendation: Implement kappa=0 branch regardless (it is the general case), and add a unit test with `l=1, kappa=0` to verify the combined output has length 6 (4*1+2=6).

2. **Spinor vendor FFI wrappers for multi-center families**
   - What we know: `vendor_ffi.rs` has no spinor integral wrappers; only `CINTcgto_spinor` helpers exist.
   - What's unclear: Whether libcint's spinor integrals for 2e/3c use interleaved doubles or complex arrays at the C level.
   - Recommendation: From libcint source, spinor integrals use `double complex *` (which is interleaved) — confirm by checking the function signatures in `cint1e.c` for `int1e_ovlp` vs `int1e_ovlp_sph`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | All Rust compilation | Yes | 1.92.0 | — |
| CubeCL cpu backend | Oracle testing | Yes (cpu feature) | 0.9.x | — |
| Vendored libcint | Oracle parity tests | Yes (`CINTX_ORACLE_BUILD_VENDOR=1`) | 6.1.3 | Non-vendor path (idempotency only) |
| libcint-master source | Coefficient extraction | Yes (in repo) | 6.1.3 | — |

**Missing dependencies with no fallback:** None that block implementation.

**Notes:**
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu` confirms the vendor path builds and the `oracle_gate_all_five_families` test runs.
- The Rust toolchain at 1.92.0 is slightly behind the pinned 1.94.0 in `rust-toolchain.toml`; the pinned version will be used when running via `cargo` in the repo due to toolchain resolution.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` + `nextest` (if installed) |
| Config file | `rust-toolchain.toml` (toolchain), no separate test config |
| Quick run command | `cargo test --package cintx-cubecl --lib transform::c2spinor` |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SPIN-01 | CG coefficients correct for l=0..4 | unit | `cargo test --package cintx-cubecl --lib transform::c2spinor_coeffs` | No — Wave 0 gap |
| SPIN-01 | sf/si transform produces correct spinor values | unit | `cargo test --package cintx-cubecl --lib transform::c2spinor` | Partial (stub tests exist, must be replaced) |
| SPIN-02 | All four compat entry points delegate correctly | unit | `cargo test --package cintx-compat --lib transform` | Partial (stub tests exist, must be replaced) |
| SPIN-03 1e | Spinor 1e parity against libcint 6.1.3 | integration/oracle | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure spinor_1e` | No — Wave 0 gap |
| SPIN-03 2e+ | Spinor 2e/2c2e/3c1e/3c2e parity against libcint 6.1.3 | integration/oracle | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure spinor_multi_center` | No — Wave 0 gap |
| SPIN-04 | kappa dispatch produces correct component count | unit | `cargo test --package cintx-cubecl --lib transform::c2spinor -- kappa` | No — Wave 0 gap |

### Sampling Rate

- **Per task commit:** `cargo test --package cintx-cubecl --lib transform`
- **Per wave merge:** `cargo test --package cintx-cubecl && cargo test --package cintx-compat`
- **Phase gate:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu` full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — covers SPIN-01 coefficient correctness
- [ ] `crates/cintx-cubecl/src/transform/c2spinor.rs` (rewrite) — replaces SPIN-01/02/04 stub tests
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` additions — vendor FFI wrappers for spinor integrals (SPIN-03)
- [ ] Oracle gate spinor tests in `oracle_gate_closure.rs` — covers SPIN-03

---

## Sources

### Primary (HIGH confidence)

- `libcint-master/src/cart2sph.c` — Full upstream libcint 6.1.3 source; verified CG tables (`g_trans_cart2jR[]`, `g_trans_cart2jI[]`) and all four `CINTc2s_*spinor*1` implementations read directly from lines 808–6958.
- `crates/cintx-cubecl/src/transform/c2s.rs` — Established structural template for coefficient extraction and transform function layout; verified by reading the full file.
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — Current stub; confirmed amplitude-averaging implementation to be deleted.
- `crates/cintx-compat/src/transform.rs` — Four compat entry points with correct signatures; confirmed they delegate to stub.
- `crates/cintx-core/src/shell.rs` — `spinor_len()` verified to match libcint `_len_spinor()` exactly.
- `crates/cintx-oracle/src/vendor_ffi.rs` — Verified vendor FFI has no spinor integral wrappers yet (only spinor helper wrappers).
- `crates/cintx-oracle/src/fixtures.rs` — Verified `complex_interleaved: true` for spinor in `OracleFixture`; `required_elements()` doubles for interleaved.
- `crates/cintx-oracle/src/compare.rs` — Verified `tolerance_for_family()`, `IMPLEMENTED_TRANSFORM_SYMBOLS` (all four spinor symbols listed), `assert_flat_buffer_contract()` spinor check.
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` — Verified no spinor representation in fixture loop; `oracle_gate_all_five_families` test confirmed to run with `CINTX_ORACLE_BUILD_VENDOR=1`.

### Secondary (MEDIUM confidence)

- `crates/cintx-cubecl/src/executor.rs` — Confirmed `transform::apply_representation_transform()` is called post-kernel on the staging buffer; spinor path routed through `c2spinor::cart_to_spinor_interleaved_staging()` stub.
- `.planning/STATE.md` — Confirmed v1.2 roadmap decision: "c2spinor.rs stub (amplitude-averaging) is provably wrong and must be treated as todo!()".

---

## Metadata

**Confidence breakdown:**
- CG coefficient source and extraction pattern: HIGH — tables verified by direct reading from vendored libcint source in repo
- Transform variant formulas (sf/si/ket/iket): HIGH — all four function bodies read from upstream source with sign conventions documented
- Oracle infrastructure readiness: HIGH — vendor FFI, fixture generation, and comparison infrastructure verified by reading relevant files
- kappa=0 behavior: HIGH — `_len_spinor()` in libcint and `spinor_len()` in `shell.rs` both verified
- Staging buffer layout (alpha+beta interleaving): MEDIUM — inferred from libcint `gspa/gspb` pointer arithmetic and `OF_CMPLX` macro; `OF_CMPLX=2` confirmed in libcint headers

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (libcint 6.1.3 is vendored and pinned; no drift risk)
