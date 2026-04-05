//! CINTstg_roots port for F12/STG/YP Gaussian-type geminal integrals.
//!
//! Implements Slater-type geminal (STG) quadrature roots and weights using
//! the Clenshaw/DCT algorithm. This is a host-side Rust port of the C
//! function `CINTstg_roots` from libcint-master/src/stg_roots.c.
//!
//! Algorithm reference: libslater library (https://github.com/nubakery/libslater).
//! Source: `libcint-master/src/stg_roots.c` lines 1-449.

use super::roots_xw_data::{data_w, data_x};

/// Maximum t argument value (clamp per D-07 and stg_roots.c line 416).
///
/// Values of t above this limit are clamped to prevent out-of-bounds table access.
const T_MAX: f64 = 19682.99_f64;

/// DCT matrix C(i,j) = cos(pi*j*(2i+1)/28) for i,j in 0..14.
///
/// Precomputed 14x14 cosine table used by `_matmul_14_14`. Generated from mpmath
/// at 25-digit precision. Source: stg_roots.c lines 12-214, verbatim copy.
static COS_14_14: [f64; 196] = [
     1.                       ,
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
     1.                       ,
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
     1.                       ,
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
     1.                       ,
     7.0710678118654757274e-01,
     0.                       ,
    -7.0710678118654757274e-01,
    -1.                       ,
    -7.0710678118654757274e-01,
     0.                       ,
     7.0710678118654757274e-01,
     1.                       ,
     7.0710678118654757274e-01,
     0.                       ,
    -7.0710678118654757274e-01,
    -1.                       ,
    -7.0710678118654757274e-01,
     1.                       ,
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
     1.                       ,
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
     1.                       ,
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
     1.                       ,
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
     1.                       ,
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
     1.                       ,
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
     1.                       ,
    -7.0710678118654757274e-01,
     0.                       ,
     7.0710678118654757274e-01,
    -1.                       ,
     7.0710678118654757274e-01,
     0.                       ,
    -7.0710678118654757274e-01,
     1.                       ,
    -7.0710678118654757274e-01,
     0.                       ,
     7.0710678118654757274e-01,
    -1.                       ,
     7.0710678118654757274e-01,
     1.                       ,
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
     1.                       ,
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
     1.                       ,
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
/// This is the host-side Rust port of `CINTstg_roots` from `stg_roots.c`.
///
/// # Parameters
/// - `nroots`: Number of quadrature roots (1 to 5 supported by the table).
/// - `ta`: The t argument (squared geminal exponent-related parameter).
/// - `ua`: The u argument (related to ua = zeta_F12 + exponents).
///
/// # Returns
/// `(roots, weights)` each of length `nroots`.
///
/// # Panics
/// None: the t-clamp prevents out-of-bounds table access.
pub fn stg_roots_host(nroots: usize, ta: f64, ua: f64) -> (Vec<f64>, Vec<f64>) {
    // D-07: clamp t to T_MAX to prevent out-of-bounds table lookup.
    let t = ta.min(T_MAX);

    // Compute normalized t coordinate (tt)
    let tt = if t > 1.0_f64 {
        t.ln() * 0.9102392266268373_f64 + 1.0_f64 // log(3)+1 scaling
    } else {
        t.sqrt()
    };

    // Compute normalized u coordinate (uu)
    let uu = ua.log10();

    // Compute t grid index and normalized t in [-1, 1]
    let it = tt.floor() as usize;
    let tt_norm = 2.0_f64 * (tt - it as f64) - 1.0_f64;

    // Compute u grid index and normalized u in [-1, 1]
    // iu range: 0..=9 (corresponds to u in [1e-7, 1e3])
    let iu = (uu + 7.0_f64).floor() as usize;
    let uu_norm = 2.0_f64 * (uu - (iu as f64 - 7.0_f64)) - 1.0_f64;

    // Table offset: stride is nroots * 196 per (it, iu) cell.
    // DATA_X base offset: (nroots-1)*nroots/2 * 19600 (skips earlier nroots tables).
    let table_base = (nroots - 1) * nroots / 2 * 19600;
    let cell_offset = nroots * 196 * (iu + it * 10);

    let data_x = data_x();
    let data_w = data_w();
    let x_slice = &data_x[table_base + cell_offset..];
    let w_slice = &data_w[table_base + cell_offset..];

    // Intermediate buffers
    let mut im = vec![0.0_f64; 14 * nroots];
    let mut imc = vec![0.0_f64; 14 * nroots];
    let mut roots = vec![0.0_f64; nroots];
    let mut weights = vec![0.0_f64; nroots];

    // Roots: Clenshaw-DC over u, DCT transform, Clenshaw-D1 over t
    _clenshaw_dc(&mut im, x_slice, uu_norm, nroots);
    _matmul_14_14(&mut imc, &im, nroots);
    _clenshaw_d1(&mut roots, &imc, tt_norm, nroots);

    // Weights: same pipeline on DATA_W
    _clenshaw_dc(&mut im, w_slice, uu_norm, nroots);
    _matmul_14_14(&mut imc, &im, nroots);
    _clenshaw_d1(&mut weights, &imc, tt_norm, nroots);

    // Normalize weights by 1/sqrt(ua) per stg_roots.c line 445-448
    let inv_sqrt_ua = 1.0_f64 / ua.sqrt();
    for w in &mut weights {
        *w *= inv_sqrt_ua;
    }

    (roots, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic smoke test: nroots=1, reasonable inputs.
    #[test]
    fn stg_roots_host_smoke_nroots1() {
        let (roots, weights) = stg_roots_host(1, 1.0_f64, 0.5_f64);
        assert_eq!(roots.len(), 1, "should return exactly 1 root");
        assert_eq!(weights.len(), 1, "should return exactly 1 weight");
        assert!(roots[0].is_finite() && roots[0] != 0.0, "root should be finite and non-zero, got {}", roots[0]);
        assert!(weights[0].is_finite() && weights[0] != 0.0, "weight should be finite and non-zero, got {}", weights[0]);
    }

    /// Smoke test for nroots=2.
    #[test]
    fn stg_roots_host_smoke_nroots2() {
        let (roots, weights) = stg_roots_host(2, 2.0_f64, 1.0_f64);
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
        let (roots, weights) = stg_roots_host(3, 4.0_f64, 2.0_f64);
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
        let (roots, weights) = stg_roots_host(1, 99999.0_f64, 0.5_f64);
        assert_eq!(roots.len(), 1);
        assert!(roots[0].is_finite(), "clamped root should be finite, got {}", roots[0]);
        assert!(weights[0].is_finite(), "clamped weight should be finite, got {}", weights[0]);
    }

    /// T_MAX constant is correct per D-07.
    #[test]
    fn t_max_constant_exact() {
        assert_eq!(T_MAX, 19682.99_f64, "T_MAX must be exactly 19682.99");
    }

    /// COS_14_14 table has exactly 196 elements.
    #[test]
    fn cos_14_14_has_196_elements() {
        assert_eq!(COS_14_14.len(), 196, "COS_14_14 must have 14*14=196 elements");
    }
}
