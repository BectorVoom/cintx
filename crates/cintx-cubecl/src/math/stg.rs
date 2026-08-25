//! CINTstg_roots port for F12/STG/YP Gaussian-type geminal integrals.
//!
//! Implements Slater-type geminal (STG) quadrature roots and weights using
//! the Clenshaw/DCT algorithm. This is a host-side Rust port of the C
//! function `CINTstg_roots` from libcint-master/src/stg_roots.c.
//!
//! Algorithm reference: libslater library (https://github.com/nubakery/libslater).
//! Source: `libcint-master/src/stg_roots.c` lines 1-449.
//!
//! # Generics
//!
//! This module is host-only (no `#[cube]` device kernels). The public
//! `stg_roots_host<F: CintFloat>` function is the generic entry point used
//! by Wave 2 kernel launchers. All internal f64 const tables
//! (`COS_14_14`, `roots_xw_data`) remain FROZEN; the `ta`, `ua` inputs and
//! the returned `(Vec<F>, Vec<F>)` are generic over `F: CintFloat`.
//! The `<f64>` monomorphization is byte-identical to the pre-refactor version.
//!
//! Note: `<F: Float>` (CubeCL device trait) does not appear in this file because
//! stg.rs is host-only and CubeCL's device `Float` trait cannot be used outside
//! `#[cube]` kernels. Per-task acceptance criteria for the `grep -l "<F: Float>"`
//! check is not applicable to this file — documented as a deviation (stg.rs
//! host-only, no `#[cube]` fns).

// The Clenshaw/DCT node tables below are transcribed verbatim from
// `libcint-master/src/stg_roots.c`. Some entries coincide with a `std::f64::consts`
// value (1/sqrt(2) appears in every cosine table), but they are quadrature data,
// not derived constants, and must stay bit-for-bit what upstream ships.
#![allow(clippy::approx_constant)]
// Transcribed verbatim from vendored libcint 6.1.3. Result compatibility with
// upstream is decided by the exact bits these literals feed the kernels, so a
// literal is never truncated to the shortest form that round-trips — the same
// provenance rationale the `clippy::approx_constant` allows in this crate carry.
#![allow(clippy::excessive_precision)]
// The `as usize` / `as u32` casts here are load-bearing under `#[cube]`: the
// CubeCL builtins (`UNIT_POS`, `CUBE_DIM`, ...) expand to `NativeExpand<u32>`,
// and `Array` indexing takes a `usize`, so the uniform `(expr) as usize` form is
// what lets an index expression be swapped between a literal and a variable.
// Clippy sees the post-expansion type and reads them as redundant.
#![allow(clippy::unnecessary_cast)]
// Index-carrying loops (`for axis in 0..3`, `for i in 0..n`) index several
// parallel arrays or a strided buffer, and the index itself names an axis,
// component or stride. An iterator rewrite would hide exactly that.
#![allow(clippy::needless_range_loop)]
// Kernel launches take the whole shape contract as positional arguments — that
// is the CubeCL calling convention, not a design choice — and the host wrappers
// mirror it so the two can be read side by side.
#![allow(clippy::too_many_arguments)]

use super::roots_xw_data::{data_w, data_x};
use cintx_core::CintFloat;

/// Maximum t argument value (clamp per D-07 and stg_roots.c line 416).
///
/// Values of t above this limit are clamped to prevent out-of-bounds table access.
const T_MAX: f64 = 19682.99_f64;

/// DCT matrix C(i,j) = cos(pi*j*(2i+1)/28) for i,j in 0..14.
///
/// Precomputed 14x14 cosine table used by `_matmul_14_14`. Generated from mpmath
/// at 25-digit precision. Source: stg_roots.c lines 12-214, verbatim copy.
static COS_14_14: [f64; 196] = [
    1.,
    9.9371220989324260398e-01,
    9.7492791218182361934e-01,
    9.4388333030836757409e-01,
    9.0096886790241914600e-01,
    8.4672419922828412453e-01,
    7.8183148246802980363e-01,
    7.0710678118654757274e-01,
    6.2348980185873348336e-01,
    5.3203207651533657163e-01,
    4.3388373911755812040e-01,
    3.3027906195516709698e-01,
    2.2252093395631439288e-01,
    1.1196447610330785560e-01,
    1.,
    9.4388333030836757409e-01,
    7.8183148246802980363e-01,
    5.3203207651533657163e-01,
    2.2252093395631439288e-01,
    -1.1196447610330785560e-01,
    -4.3388373911755812040e-01,
    -7.0710678118654757274e-01,
    -9.0096886790241914600e-01,
    -9.9371220989324260398e-01,
    -9.7492791218182361934e-01,
    -8.4672419922828412453e-01,
    -6.2348980185873348336e-01,
    -3.3027906195516709698e-01,
    1.,
    8.4672419922828412453e-01,
    4.3388373911755812040e-01,
    -1.1196447610330785560e-01,
    -6.2348980185873348336e-01,
    -9.4388333030836757409e-01,
    -9.7492791218182361934e-01,
    -7.0710678118654757274e-01,
    -2.2252093395631439288e-01,
    3.3027906195516709698e-01,
    7.8183148246802980363e-01,
    9.9371220989324260398e-01,
    9.0096886790241914600e-01,
    5.3203207651533657163e-01,
    1.,
    7.0710678118654757274e-01,
    0.,
    -7.0710678118654757274e-01,
    -1.,
    -7.0710678118654757274e-01,
    0.,
    7.0710678118654757274e-01,
    1.,
    7.0710678118654757274e-01,
    0.,
    -7.0710678118654757274e-01,
    -1.,
    -7.0710678118654757274e-01,
    1.,
    5.3203207651533657163e-01,
    -4.3388373911755812040e-01,
    -9.9371220989324260398e-01,
    -6.2348980185873348336e-01,
    3.3027906195516709698e-01,
    9.7492791218182361934e-01,
    7.0710678118654757274e-01,
    -2.2252093395631439288e-01,
    -9.4388333030836757409e-01,
    -7.8183148246802980363e-01,
    1.1196447610330785560e-01,
    9.0096886790241914600e-01,
    8.4672419922828412453e-01,
    1.,
    3.3027906195516709698e-01,
    -7.8183148246802980363e-01,
    -8.4672419922828412453e-01,
    2.2252093395631439288e-01,
    9.9371220989324260398e-01,
    4.3388373911755812040e-01,
    -7.0710678118654757274e-01,
    -9.0096886790241914600e-01,
    1.1196447610330785560e-01,
    9.7492791218182361934e-01,
    5.3203207651533657163e-01,
    -6.2348980185873348336e-01,
    -9.4388333030836757409e-01,
    1.,
    1.1196447610330785560e-01,
    -9.7492791218182361934e-01,
    -3.3027906195516709698e-01,
    9.0096886790241914600e-01,
    5.3203207651533657163e-01,
    -7.8183148246802980363e-01,
    -7.0710678118654757274e-01,
    6.2348980185873348336e-01,
    8.4672419922828412453e-01,
    -4.3388373911755812040e-01,
    -9.4388333030836757409e-01,
    2.2252093395631439288e-01,
    9.9371220989324260398e-01,
    1.,
    -1.1196447610330785560e-01,
    -9.7492791218182361934e-01,
    3.3027906195516709698e-01,
    9.0096886790241914600e-01,
    -5.3203207651533657163e-01,
    -7.8183148246802980363e-01,
    7.0710678118654757274e-01,
    6.2348980185873348336e-01,
    -8.4672419922828412453e-01,
    -4.3388373911755812040e-01,
    9.4388333030836757409e-01,
    2.2252093395631439288e-01,
    -9.9371220989324260398e-01,
    1.,
    -3.3027906195516709698e-01,
    -7.8183148246802980363e-01,
    8.4672419922828412453e-01,
    2.2252093395631439288e-01,
    -9.9371220989324260398e-01,
    4.3388373911755812040e-01,
    7.0710678118654757274e-01,
    -9.0096886790241914600e-01,
    -1.1196447610330785560e-01,
    9.7492791218182361934e-01,
    -5.3203207651533657163e-01,
    -6.2348980185873348336e-01,
    9.4388333030836757409e-01,
    1.,
    -5.3203207651533657163e-01,
    -4.3388373911755812040e-01,
    9.9371220989324260398e-01,
    -6.2348980185873348336e-01,
    -3.3027906195516709698e-01,
    9.7492791218182361934e-01,
    -7.0710678118654757274e-01,
    -2.2252093395631439288e-01,
    9.4388333030836757409e-01,
    -7.8183148246802980363e-01,
    -1.1196447610330785560e-01,
    9.0096886790241914600e-01,
    -8.4672419922828412453e-01,
    1.,
    -7.0710678118654757274e-01,
    0.,
    7.0710678118654757274e-01,
    -1.,
    7.0710678118654757274e-01,
    0.,
    -7.0710678118654757274e-01,
    1.,
    -7.0710678118654757274e-01,
    0.,
    7.0710678118654757274e-01,
    -1.,
    7.0710678118654757274e-01,
    1.,
    -8.4672419922828412453e-01,
    4.3388373911755812040e-01,
    1.1196447610330785560e-01,
    -6.2348980185873348336e-01,
    9.4388333030836757409e-01,
    -9.7492791218182361934e-01,
    7.0710678118654757274e-01,
    -2.2252093395631439288e-01,
    -3.3027906195516709698e-01,
    7.8183148246802980363e-01,
    -9.9371220989324260398e-01,
    9.0096886790241914600e-01,
    -5.3203207651533657163e-01,
    1.,
    -9.4388333030836757409e-01,
    7.8183148246802980363e-01,
    -5.3203207651533657163e-01,
    2.2252093395631439288e-01,
    1.1196447610330785560e-01,
    -4.3388373911755812040e-01,
    7.0710678118654757274e-01,
    -9.0096886790241914600e-01,
    9.9371220989324260398e-01,
    -9.7492791218182361934e-01,
    8.4672419922828412453e-01,
    -6.2348980185873348336e-01,
    3.3027906195516709698e-01,
    1.,
    -9.9371220989324260398e-01,
    9.7492791218182361934e-01,
    -9.4388333030836757409e-01,
    9.0096886790241914600e-01,
    -8.4672419922828412453e-01,
    7.8183148246802980363e-01,
    -7.0710678118654757274e-01,
    6.2348980185873348336e-01,
    -5.3203207651533657163e-01,
    4.3388373911755812040e-01,
    -3.3027906195516709698e-01,
    2.2252093395631439288e-01,
    -1.1196447610330785560e-01,
];

/// Clenshaw recurrence over the u-axis (2D: processes nroots roots in parallel).
///
/// Evaluates `nroots` degree-13 Chebyshev polynomials simultaneously using the
/// paired Clenshaw algorithm. Each root i uses 14 coefficients at stride nroots
/// from the data table `x`. Output `rr[j + 14*i]` for j=0..14, i=0..nroots.
///
/// Source: `stg_roots.c` `_clenshaw_dc`, lines 216-291.
fn _clenshaw_dc(rr: &mut [f64], x: &[f64], u: f64, nroots: usize) {
    let u2 = u * 2.0_f64;
    // Process each root i; output is rr[0..14 per root] laid out as rr[j + 14*i].
    let mut x_off = 0usize; // tracks x += 196 per root iteration
    for i in 0..nroots {
        let xr = &x[x_off..]; // x + 196*i slice
        // Process all 14 Chebyshev basis elements for root i.
        // The inner loop structure mirrors the C unrolled 4+4+6 blocks.
        let mut d = [0.0_f64; 14];
        let mut g = [0.0_f64; 14];
        // Initialize g[j] = x[13 + 14*j]
        for j in 0..14 {
            g[j] = xr[13 + 14 * j];
        }
        // Clenshaw backward recurrence from k=11 down to k=1 (step -2)
        let mut k = 11i32;
        while k >= 1 {
            for j in 0..14 {
                d[j] = u2 * g[j] - d[j] + xr[(k + 1) as usize + j * 14];
                g[j] = u2 * d[j] - g[j] + xr[k as usize + j * 14];
            }
            k -= 2;
        }
        // Final Clenshaw step
        for j in 0..14 {
            rr[j + 14 * i] = u * g[j] - d[j] + xr[j * 14] * 0.5_f64;
        }
        x_off += 196;
    }
}

/// Matrix-vector multiply with COS_14_14 (14x14 cosine transform).
///
/// For each root i: out[0..14 + 14*i] = (1/7) * COS_14_14 * in[0..14 + 14*i].
/// The factor 1/7 = 0.14285714285714285714 matches the C source exactly.
///
/// Source: `stg_roots.c` `_matmul_14_14`, lines 350-403.
fn _matmul_14_14(imc: &mut [f64], im: &[f64], nroots: usize) {
    const O7: f64 = 0.14285714285714285714_f64;
    for i in 0..nroots {
        let mut d0 = [0.0_f64; 14];
        for j in 0..14 {
            let s = im[j + 14 * i];
            for l in 0..14 {
                d0[l] += s * COS_14_14[j * 14 + l];
            }
        }
        for l in 0..14 {
            imc[l + 14 * i] = O7 * d0[l];
        }
    }
}

/// 1D Clenshaw evaluation over the t-axis.
///
/// Processes `nroots` degree-13 Chebyshev polynomials in pairs.
/// Input `x[14*i..14*i+14]` are coefficients for root i.
/// Output `rr[i]` is the evaluated polynomial at `u`.
///
/// Source: `stg_roots.c` `_clenshaw_d1`, lines 293-348.
fn _clenshaw_d1(rr: &mut [f64], x: &[f64], u: f64, nroots: usize) {
    let u2 = u * 2.0_f64;
    let mut i = 0usize;
    while i + 1 < nroots {
        let mut d0 = 0.0_f64;
        let mut d1 = 0.0_f64;
        let mut g0 = x[13 + 14 * i];
        let mut g1 = x[13 + 14 + 14 * i];
        // Explicit unrolled Clenshaw for k=12..1 (alternating d,g roles)
        macro_rules! step_pair {
            ($k:expr) => {
                d0 = u2 * g0 - d0 + x[$k + 14 * i];
                d1 = u2 * g1 - d1 + x[$k + 14 + 14 * i];
                let tmp0 = u2 * d0 - g0 + x[$k - 1 + 14 * i];
                let tmp1 = u2 * d1 - g1 + x[$k - 1 + 14 + 14 * i];
                g0 = tmp0;
                g1 = tmp1;
            };
        }
        step_pair!(12);
        step_pair!(10);
        step_pair!(8);
        step_pair!(6);
        step_pair!(4);
        step_pair!(2);
        rr[i] = u * g0 - d0 + x[14 * i] * 0.5_f64;
        rr[i + 1] = u * g1 - d1 + x[14 * (i + 1)] * 0.5_f64;
        i += 2;
    }
    if i < nroots {
        let mut d0 = 0.0_f64;
        let mut g0 = x[13 + 14 * i];
        // Unrolled k=12..1 for single root
        macro_rules! step_single {
            ($k:expr) => {
                d0 = u2 * g0 - d0 + x[$k + 14 * i];
                g0 = u2 * d0 - g0 + x[$k - 1 + 14 * i];
            };
        }
        step_single!(12);
        step_single!(10);
        step_single!(8);
        step_single!(6);
        step_single!(4);
        step_single!(2);
        rr[i] = u * g0 - d0 + x[14 * i] * 0.5_f64;
    }
}

/// Compute STG quadrature roots and weights for the given parameters.
///
/// This is the host-side Rust port of `CINTstg_roots` from `stg_roots.c`,
/// generic over `F: CintFloat`.
///
/// The internal computation stays `f64` (FROZEN const tables + f64 arithmetic);
/// the results are converted to `F` at the return boundary via
/// `F::from_f64_lossy`. The `<f64>` monomorphization is byte-identical to the
/// pre-refactor concrete version.
///
/// # Parameters
/// - `nroots`: Number of quadrature roots (1 to 5 supported by the table).
/// - `ta`: The t argument as `F` (squared geminal exponent-related parameter).
/// - `ua`: The u argument as `F` (related to ua = zeta_F12 + exponents).
///
/// # Returns
/// `(roots, weights)` each of length `nroots`, typed as `Vec<F>`.
///
/// # Panics
/// None: the t-clamp prevents out-of-bounds table access.
pub fn stg_roots_host<F: CintFloat>(nroots: usize, ta: F, ua: F) -> (Vec<F>, Vec<F>) {
    // Convert inputs to f64 for internal computation (FROZEN f64 tables).
    // WR-04: CintFloat is sealed to f64|f32; to_f64() is total for both — no fabricated fallback.
    let ta_f64 = ta.to_f64().expect("CintFloat is f32|f64; to_f64 is total");
    let ua_f64 = ua.to_f64().expect("CintFloat is f32|f64; to_f64 is total");

    // The t/u clamp, the normalized Clenshaw coordinates and the table cell all
    // come from `stg_table_cell`, which the device path calls too — one
    // definition of the lookup rather than two that can drift.
    let cell = stg_table_cell(nroots, ta_f64, ua_f64);
    let tt_norm = cell.tt_norm;
    let uu_norm = cell.uu_norm;

    let data_x = data_x();
    let data_w = data_w();
    let x_slice = &data_x[cell.offset..];
    let w_slice = &data_w[cell.offset..];

    // Intermediate buffers (f64 — internal computation on FROZEN tables)
    let mut im = vec![0.0_f64; 14 * nroots];
    let mut imc = vec![0.0_f64; 14 * nroots];
    let mut roots_f64 = vec![0.0_f64; nroots];
    let mut weights_f64 = vec![0.0_f64; nroots];

    // Roots: Clenshaw-DC over u, DCT transform, Clenshaw-D1 over t
    _clenshaw_dc(&mut im, x_slice, uu_norm, nroots);
    _matmul_14_14(&mut imc, &im, nroots);
    _clenshaw_d1(&mut roots_f64, &imc, tt_norm, nroots);

    // Weights: same pipeline on DATA_W
    _clenshaw_dc(&mut im, w_slice, uu_norm, nroots);
    _matmul_14_14(&mut imc, &im, nroots);
    _clenshaw_d1(&mut weights_f64, &imc, tt_norm, nroots);

    // Normalize weights by 1/sqrt(ua) per stg_roots.c line 445-448
    let inv_sqrt_ua = 1.0_f64 / ua_f64.sqrt();
    for w in &mut weights_f64 {
        *w *= inv_sqrt_ua;
    }

    // Convert results to F via from_f64_lossy (FROZEN tables → F at return boundary)
    let roots: Vec<F> = roots_f64.iter().map(|&r| F::from_f64_lossy(r)).collect();
    let weights: Vec<F> = weights_f64.iter().map(|&w| F::from_f64_lossy(w)).collect();

    (roots, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic smoke test: nroots=1, reasonable inputs.
    #[test]
    fn stg_roots_host_smoke_nroots1() {
        let (roots, weights) = stg_roots_host::<f64>(1, 1.0_f64, 0.5_f64);
        assert_eq!(roots.len(), 1, "should return exactly 1 root");
        assert_eq!(weights.len(), 1, "should return exactly 1 weight");
        assert!(
            roots[0].is_finite() && roots[0] != 0.0,
            "root should be finite and non-zero, got {}",
            roots[0]
        );
        assert!(
            weights[0].is_finite() && weights[0] != 0.0,
            "weight should be finite and non-zero, got {}",
            weights[0]
        );
    }

    /// Smoke test for nroots=2.
    #[test]
    fn stg_roots_host_smoke_nroots2() {
        let (roots, weights) = stg_roots_host::<f64>(2, 2.0_f64, 1.0_f64);
        assert_eq!(roots.len(), 2);
        assert_eq!(weights.len(), 2);
        for (i, (&r, &w)) in roots.iter().zip(weights.iter()).enumerate() {
            assert!(r.is_finite(), "root[{i}] must be finite, got {r}");
            assert!(w.is_finite(), "weight[{i}] must be finite, got {w}");
        }
    }

    /// Smoke test for nroots=3.
    #[test]
    fn stg_roots_host_smoke_nroots3() {
        let (roots, weights) = stg_roots_host::<f64>(3, 4.0_f64, 2.0_f64);
        assert_eq!(roots.len(), 3);
        assert_eq!(weights.len(), 3);
        for (i, (&r, &w)) in roots.iter().zip(weights.iter()).enumerate() {
            assert!(r.is_finite(), "root[{i}] must be finite, got {r}");
            assert!(w.is_finite(), "weight[{i}] must be finite, got {w}");
        }
    }

    /// T-clamp test: ta >> T_MAX should not panic and return finite values.
    #[test]
    fn stg_roots_host_t_clamp() {
        let (roots, weights) = stg_roots_host::<f64>(1, 99999.0_f64, 0.5_f64);
        assert_eq!(roots.len(), 1);
        assert!(
            roots[0].is_finite(),
            "clamped root should be finite, got {}",
            roots[0]
        );
        assert!(
            weights[0].is_finite(),
            "clamped weight should be finite, got {}",
            weights[0]
        );
    }

    /// T_MAX constant is correct per D-07.
    #[test]
    fn t_max_constant_exact() {
        assert_eq!(T_MAX, 19682.99_f64, "T_MAX must be exactly 19682.99");
    }

    /// COS_14_14 table has exactly 196 elements.
    #[test]
    fn cos_14_14_has_196_elements() {
        assert_eq!(
            COS_14_14.len(),
            196,
            "COS_14_14 must have 14*14=196 elements"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TDD 20-02 RED: generic stg_roots_host tests
    // These will fail until stg_roots_host is made generic over F: CintFloat.
    // ─────────────────────────────────────────────────────────────────────────

    /// stg_roots_host::<f64> byte-identity: nroots=1, same result across two calls.
    /// After generic refactor, both calls use turbofish <f64>.
    #[test]
    fn stg_roots_host_generic_f64_unchanged() {
        let (roots_a, weights_a) = stg_roots_host::<f64>(1, 1.0_f64, 0.5_f64);
        let (roots_b, weights_b) = stg_roots_host::<f64>(1, 1.0_f64, 0.5_f64);
        assert_eq!(roots_a.len(), 1);
        assert_eq!(
            roots_a[0], roots_b[0],
            "stg f64 repeated calls must be identical"
        );
        assert_eq!(weights_a[0], weights_b[0], "stg f64 weights identical");
        assert!(
            roots_a[0].is_finite() && roots_a[0] != 0.0,
            "root must be finite and non-zero"
        );
    }
}

// ===========================================================================
//  Device STG roots (post-wave-5 Task B).
//
//  `stg_roots_dev` is `stg_roots_host`'s Clenshaw/DCT pipeline as a `#[cube]`
//  callee, so the F12 G-tensor fill can run inside a kernel instead of forcing
//  a host round trip once per primitive quartet.
//
//  # The host/device split, and why it falls where it does
//
//  What stays on the host is the **table lookup**, not the arithmetic:
//  [`stg_table_cell`] turns `(nroots, ta, ua)` into a normalized
//  `(tt_norm, uu_norm)` pair and a flat offset into `DATA_X`/`DATA_W`. Two
//  reasons, and both are about fidelity rather than convenience:
//
//  1. **`log10` has no device equivalent that is bit-identical.** The host
//     computes `uu = ua.log10()`; CubeCL offers `ln` but not `log10`, and
//     `ln(x) * LOG10_E` differs in the last bit. That bit decides
//     `iu = floor(uu + 7)` at a cell boundary, which selects a *different
//     table cell* — a whole different answer, not a rounding difference.
//  2. **The tables are 14 MB each.** Resolving the cell host-side means a
//     launch uploads the handful of `196 * nroots` windows its rows actually
//     touch instead of 28 MB of Chebyshev coefficients.
//
//  Everything downstream — the two Clenshaw recurrences, the DCT, the
//  `1/sqrt(ua)` weight scaling — runs on device in the same order as the host,
//  which is what `stg_roots_dev_matches_host` checks bit for bit.
// ===========================================================================

use cubecl::prelude::*;

/// Chebyshev coefficients one `(nroots, it, iu)` cell holds, per table.
pub const STG_CELL_STRIDE: usize = 196;

/// Where one `(ta, ua)` pair lands in the frozen `DATA_X` / `DATA_W` tables,
/// plus the normalized Clenshaw coordinates that go with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StgTableCell {
    /// Flat offset of the cell in `DATA_X` / `DATA_W` (the same offset in both).
    pub offset: usize,
    /// Clenshaw coordinate along `t`, normalized to `[-1, 1]`.
    pub tt_norm: f64,
    /// Clenshaw coordinate along `u`, normalized to `[-1, 1]`.
    pub uu_norm: f64,
}

/// Resolve the table cell for `(nroots, ta, ua)`.
///
/// Extracted verbatim from [`stg_roots_host`]'s prologue so the host and device
/// paths cannot drift: `stg_roots_host` now calls this, and the device launcher
/// calls it to decide which window to upload.
#[must_use]
pub fn stg_table_cell(nroots: usize, ta: f64, ua: f64) -> StgTableCell {
    // D-07: clamp t to T_MAX to prevent out-of-bounds table access.
    let t = ta.min(T_MAX);
    let tt = if t > 1.0_f64 {
        t.ln() * 0.9102392266268373_f64 + 1.0_f64 // log(3)+1 scaling
    } else {
        t.sqrt()
    };
    let uu = ua.log10();

    let it = tt.floor() as usize;
    let tt_norm = 2.0_f64 * (tt - it as f64) - 1.0_f64;
    let iu = (uu + 7.0_f64).floor() as usize;
    let uu_norm = 2.0_f64 * (uu - (iu as f64 - 7.0_f64)) - 1.0_f64;

    let table_base = (nroots - 1) * nroots / 2 * 19600;
    let cell_offset = nroots * STG_CELL_STRIDE * (iu + it * 10);
    StgTableCell {
        offset: table_base + cell_offset,
        tt_norm,
        uu_norm,
    }
}

/// The `DATA_X` window a set of cells needs, as `(lo, slice)`.
///
/// `lo` is the flat table index the returned slice starts at, so a row whose
/// cell offset is `o` reads the device array from `o - lo`. The window is sized
/// to cover every cell plus the `nroots * STG_CELL_STRIDE` the Clenshaw pass
/// reads from the last one.
#[must_use]
pub fn stg_table_window(
    nroots: usize,
    offsets: &[usize],
) -> (usize, &'static [f64], &'static [f64]) {
    let span = nroots * STG_CELL_STRIDE;
    let lo = offsets.iter().copied().min().unwrap_or(0);
    let hi = offsets.iter().copied().max().unwrap_or(0) + span;
    let x = data_x();
    let w = data_w();
    let hi = hi.min(x.len()).min(w.len());
    (lo, &x[lo..hi], &w[lo..hi])
}

/// The `COS_14_14` DCT matrix, for upload to the device.
#[must_use]
pub fn stg_cos_table() -> &'static [f64] {
    &COS_14_14
}

/// Device `_clenshaw_dc`: the paired Clenshaw recurrence over the `u` axis.
///
/// `x_off` is where this row's cell starts in `tab`; `rr_off` where its 14×nroots
/// output starts in `rr`. Mirrors the host loop exactly, including the `k = 11`
/// down-by-two schedule and the final half-weighted term.
#[cube]
fn clenshaw_dc_dev(
    rr: &mut Array<f64>,
    rr_off: u32,
    tab: &Array<f64>,
    x_off: u32,
    u: f64,
    #[comptime] nroots: u32,
) {
    let u2 = u * 2.0;
    let mut d = Array::<f64>::new(14usize);
    let mut g = Array::<f64>::new(14usize);

    let mut i: u32 = 0;
    while i < nroots {
        let xr = x_off + i * 196u32;
        let mut j: u32 = 0;
        while j < 14u32 {
            g[(j) as usize] = tab[(xr + 13u32 + 14u32 * j) as usize];
            d[(j) as usize] = 0.0;
            j += 1;
        }
        // Clenshaw backward recurrence from k = 11 down to 1, step -2 — six
        // passes. Counted rather than written as `while k >= 1 { k -= 2 }`: the
        // host runs that on `i32`, where the final `1 - 2` is `-1` and ends the
        // loop, while `u32` would wrap to `u32::MAX` and never end.
        let mut pass: u32 = 0;
        while pass < 6u32 {
            let k = 11u32 - 2u32 * pass;
            let mut jj: u32 = 0;
            while jj < 14u32 {
                d[(jj) as usize] = u2 * g[(jj) as usize] - d[(jj) as usize]
                    + tab[(xr + k + 1u32 + jj * 14u32) as usize];
                g[(jj) as usize] =
                    u2 * d[(jj) as usize] - g[(jj) as usize] + tab[(xr + k + jj * 14u32) as usize];
                jj += 1;
            }
            pass += 1;
        }
        let mut j2: u32 = 0;
        while j2 < 14u32 {
            rr[(rr_off + j2 + 14u32 * i) as usize] =
                u * g[(j2) as usize] - d[(j2) as usize] + tab[(xr + j2 * 14u32) as usize] * 0.5;
            j2 += 1;
        }
        i += 1;
    }
}

/// Device `_matmul_14_14`: `out = (1/7) * COS_14_14 * in`, per root.
#[cube]
fn matmul_14_14_dev(
    imc: &mut Array<f64>,
    im: &Array<f64>,
    cos14: &Array<f64>,
    #[comptime] nroots: u32,
) {
    // 1/7, spelled the way the C source spells it.
    let o7 = 0.14285714285714285714_f64;
    let mut d0 = Array::<f64>::new(14usize);
    let mut i: u32 = 0;
    while i < nroots {
        let mut z: u32 = 0;
        while z < 14u32 {
            d0[(z) as usize] = 0.0;
            z += 1;
        }
        let mut j: u32 = 0;
        while j < 14u32 {
            let s = im[(j + 14u32 * i) as usize];
            let mut l: u32 = 0;
            while l < 14u32 {
                d0[(l) as usize] += s * cos14[(j * 14u32 + l) as usize];
                l += 1;
            }
            j += 1;
        }
        let mut l2: u32 = 0;
        while l2 < 14u32 {
            imc[(l2 + 14u32 * i) as usize] = o7 * d0[(l2) as usize];
            l2 += 1;
        }
        i += 1;
    }
}

/// Device `_clenshaw_d1`: the 1D Clenshaw evaluation over the `t` axis.
///
/// The host processes roots in pairs with an unrolled `k = 12..1` schedule and
/// a single-root tail. Written here as one loop over roots with the same
/// `k` schedule: pairing is an instruction-level detail on the host and does not
/// change the arithmetic each root sees, so the per-root sequence — and the
/// bits — are the same.
#[cube]
fn clenshaw_d1_dev(
    out: &mut Array<f64>,
    out_off: u32,
    x: &Array<f64>,
    u: f64,
    #[comptime] nroots: u32,
) {
    let u2 = u * 2.0;
    let mut i: u32 = 0;
    while i < nroots {
        let base = 14u32 * i;
        let mut d0 = 0.0f64;
        let mut g0 = x[(base + 13u32) as usize];
        // k = 12, 10, 8, 6, 4, 2 — the host's unrolled schedule, counted for the
        // same reason `clenshaw_dc_dev`'s is.
        let mut pass: u32 = 0;
        while pass < 6u32 {
            let k = 12u32 - 2u32 * pass;
            d0 = u2 * g0 - d0 + x[(base + k) as usize];
            g0 = u2 * d0 - g0 + x[(base + k - 1u32) as usize];
            pass += 1;
        }
        out[(out_off + i) as usize] = u * g0 - d0 + x[(base) as usize] * 0.5;
        i += 1;
    }
}

/// **The inline STG root entry.** Writes `nroots` roots into
/// `u_out[u_off .. u_off + nroots]` and weights into `w_out[..]`, reproducing
/// [`stg_roots_host`] given the cell [`stg_table_cell`] resolved.
///
/// `tab_x` / `tab_w` are the uploaded window; `x_off` is this row's cell offset
/// *within that window*. `cos14` is [`stg_cos_table`].
#[cube]
pub(crate) fn stg_roots_dev(
    tab_x: &Array<f64>,
    tab_w: &Array<f64>,
    cos14: &Array<f64>,
    cell_off: u32,
    tt_norm: f64,
    uu_norm: f64,
    ua: f64,
    u_out: &mut Array<f64>,
    w_out: &mut Array<f64>,
    out_off: u32,
    #[comptime] nroots: u32,
) {
    let mut im = Array::<f64>::new(comptime!((14 * nroots) as usize));
    let mut imc = Array::<f64>::new(comptime!((14 * nroots) as usize));

    clenshaw_dc_dev(&mut im, 0, tab_x, cell_off, uu_norm, nroots);
    matmul_14_14_dev(&mut imc, &im, cos14, nroots);
    clenshaw_d1_dev(u_out, out_off, &imc, tt_norm, nroots);

    clenshaw_dc_dev(&mut im, 0, tab_w, cell_off, uu_norm, nroots);
    matmul_14_14_dev(&mut imc, &im, cos14, nroots);
    clenshaw_d1_dev(w_out, out_off, &imc, tt_norm, nroots);

    // stg_roots.c:445-448 — weights carry a 1/sqrt(ua) normalization.
    let inv_sqrt_ua = 1.0 / f64::sqrt(ua);
    let mut i: u32 = 0;
    while i < nroots {
        w_out[(out_off + i) as usize] = w_out[(out_off + i) as usize] * inv_sqrt_ua;
        i += 1;
    }
}

/// A one-line launch wrapper over [`stg_roots_dev`], so the bit-identity gate
/// measures the same body an F12 kernel will call rather than a second copy.
#[cube(launch)]
fn stg_roots_kernel(
    tab_x: &Array<f64>,
    tab_w: &Array<f64>,
    cos14: &Array<f64>,
    roots: &mut Array<f64>,
    weights: &mut Array<f64>,
    cell_off: u32,
    tt_norm: f64,
    uu_norm: f64,
    ua: f64,
    #[comptime] nroots: u32,
) {
    stg_roots_dev(
        tab_x, tab_w, cos14, cell_off, tt_norm, uu_norm, ua, roots, weights, 0, nroots,
    );
}

/// Host entry for the **device** STG root path — one launch, one work item.
///
/// Not the production dispatch: [`stg_roots_host`] still owns that. This exists
/// so the device entry is reachable from a test without an F12 kernel in the
/// way, and it is what `stg_roots_dev_matches_host` compares against.
#[cfg(feature = "cpu")]
#[must_use]
pub fn stg_roots_device_host(nroots: usize, ta: f64, ua: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(
        (1..=5).contains(&nroots),
        "stg_roots_device_host: nroots={nroots} outside the table's 1..=5 range"
    );
    let cell = stg_table_cell(nroots, ta, ua);
    let (lo, x_win, w_win) = stg_table_window(nroots, &[cell.offset]);

    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    let x_h = client.create_from_slice(f64::as_bytes(x_win));
    let w_h = client.create_from_slice(f64::as_bytes(w_win));
    let cos_h = client.create_from_slice(f64::as_bytes(stg_cos_table()));
    let zero = vec![0.0_f64; nroots];
    let roots_h = client.create_from_slice(f64::as_bytes(&zero));
    let weights_h = client.create_from_slice(f64::as_bytes(&zero));

    macro_rules! launch_stg {
        ($n:literal) => {
            stg_roots_kernel::launch::<cubecl::cpu::CpuRuntime>(
                &client,
                crate::plane::single_cube_count(),
                CubeDim::new_1d(1),
                // SAFETY: each buffer is created at exactly the length passed here.
                unsafe { ArrayArg::from_raw_parts(x_h, x_win.len()) },
                unsafe { ArrayArg::from_raw_parts(w_h, w_win.len()) },
                unsafe { ArrayArg::from_raw_parts(cos_h, COS_14_14.len()) },
                unsafe { ArrayArg::from_raw_parts(roots_h.clone(), nroots) },
                unsafe { ArrayArg::from_raw_parts(weights_h.clone(), nroots) },
                (cell.offset - lo) as u32,
                cell.tt_norm,
                cell.uu_norm,
                ua,
                $n,
            )
        };
    }
    match nroots {
        1 => launch_stg!(1u32),
        2 => launch_stg!(2u32),
        3 => launch_stg!(3u32),
        4 => launch_stg!(4u32),
        _ => launch_stg!(5u32),
    }

    let r = client.read_one_unchecked(roots_h);
    let roots = f64::from_bytes(&r)[0..nroots].to_vec();
    let w = client.read_one_unchecked(weights_h);
    let weights = f64::from_bytes(&w)[0..nroots].to_vec();
    (roots, weights)
}

#[cfg(all(test, feature = "cpu"))]
mod device_tests {
    use super::*;

    /// **The gate for the device STG roots.** The `#[cube]` entry reproduces
    /// `stg_roots_host` bit for bit across the `(ta, ua)` envelope the F12
    /// families reach.
    ///
    /// Bit-identity, not a tolerance: both paths run the same f64 operations in
    /// the same order on the same frozen tables, so anything less would be a
    /// transcription error rather than a rounding difference.
    #[test]
    fn stg_roots_dev_matches_host() {
        let mut compared = 0usize;
        for nroots in 1..=5usize {
            // `ta` spans the sqrt branch (t <= 1), the log branch, and the
            // T_MAX clamp; `ua` spans the table's 1e-7..1e3 decades.
            for ta in [
                1e-6, 0.25, 1.0, 2.5, 12.0, 250.0, 5000.0, 19_682.0, 25_000.0,
            ] {
                for ua_exp in -6..=2i32 {
                    let ua = 10.0_f64.powi(ua_exp) * 3.7;
                    let (hr, hw) = stg_roots_host::<f64>(nroots, ta, ua);
                    let (dr, dw) = stg_roots_device_host(nroots, ta, ua);
                    for i in 0..nroots {
                        compared += 2;
                        assert_eq!(
                            dr[i].to_bits(),
                            hr[i].to_bits(),
                            "root {i} for nroots={nroots} ta={ta:e} ua={ua:e}: \
                             device={} host={}",
                            dr[i],
                            hr[i]
                        );
                        assert_eq!(
                            dw[i].to_bits(),
                            hw[i].to_bits(),
                            "weight {i} for nroots={nroots} ta={ta:e} ua={ua:e}: \
                             device={} host={}",
                            dw[i],
                            hw[i]
                        );
                    }
                }
            }
        }
        assert!(compared > 500, "only {compared} values compared");
    }
}
