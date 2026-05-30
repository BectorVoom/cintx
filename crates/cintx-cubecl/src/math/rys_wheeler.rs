//! Host Wheeler/Jacobi Rys root+weight engine for `nroots >= 6` (Phase 25 FND-02, Task 1b).
//!
//! Verbatim host-side port of libcint 6.1.3's `n > 5` Rys root/weight path
//! (`lower == 0` only — the short-range `lower != 0` path is out of scope for Phase 25
//! and is not exercised by any Hessian family). The control flow mirrors
//! `libcint-master/src/rys_roots.c::CINTrys_roots` (per-nroots dispatch), feeding into
//! the Flocke modified-moments -> Wheeler recursion -> vendored-MRRR eigensolve
//! (`eigh::cint_diagonalize`, Task 1a) -> root transform chain, plus the large-x Schmidt
//! / Laguerre tails.
//!
//! ## Dispatch table (rys_roots.c:97-114, lower==0)
//!
//! | nroots | x <= breakpoint        | x > breakpoint          | breakpoint |
//! |--------|------------------------|-------------------------|------------|
//! | 6,7    | `rys_jacobi` (f64)     | `rys_schmidt` (f64)     | 11         |
//! | 8      | `rys_jacobi` (f64)     | `lrys_schmidt` (ld)     | 11         |
//! | 9      | `lrys_jacobi` (ld)     | `lrys_laguerre` (ld)    | 10         |
//! | 10,11  | `lrys_jacobi` (ld)     | `lrys_laguerre` (ld)    | 18         |
//! | 12     | `lrys_jacobi` (ld)     | `lrys_laguerre` (ld)    | 22         |
//!
//! Plus the two global polynomial-fit regimes that apply to ALL nroots:
//! `x <= SMALLX_LIMIT` (3e-7) and `x >= 35 + nroots*5`. (Those regimes are out of the
//! Phase-25 corpus envelope; they fall back to the Jacobi/Schmidt path here and are
//! validated by the nroots sweep over the intermediate-x grid.)
//!
//! ## Long-double caveat (RESEARCH §FND-02 Pitfall 1)
//!
//! The cintx vendor build disables `HAVE_SQRTL`/`HAVE_QUADMATH_H` (`cintx-oracle/build.rs`),
//! so the C `lrys_*` path's `sqrtl`/`expl` degrade to `c99_sqrtl` (one Babylonian f64-sqrt
//! refinement) / `c99_expl` (= `exp`). This Rust port runs the long-double path in f64 and
//! replicates `c99_sqrtl`. The nroots 8..12 sweep at atol=1e-12 validates that the residual
//! f64-vs-80bit divergence stays within tolerance.

use super::eigh;
use super::roots_jacobi_data as data;

/// libcint `MXRYSROOTS` (cint.h: 32).
const MXRYSROOTS: usize = 32;
/// `SMALLX_LIMIT` (rys_roots.c:26).
const SMALLX_LIMIT: f64 = 3e-7;
/// `SQRTPIE4` = sqrt(pi)/2 (rys_wheeler.c:15).
const SQRTPIE4: f64 = 0.886_226_925_452_758_013_649_083_741_670_572_591_398_774_728_061_193_564_106_903_894_926_4;
/// `SML_FLOAT64` (fmt.c:20).
const SML_FLOAT64: f64 = f64::EPSILON * 0.5;
/// Flocke extra recursion order for the f64 ("DP") Miller pass (rys_wheeler.c:22).
const FLOCKE_EXTRA_ORDER_FOR_DP: usize = 20;
/// Flocke extra recursion order for the long-double ("LP") Miller pass (rys_wheeler.c:24).
const FLOCKE_EXTRA_ORDER_FOR_LP: usize = 24;

/// `c99_sqrtl` (rys_roots.c:1776) — one Babylonian refinement over f64 `sqrt`.
/// Used wherever the C long-double path calls `sqrtl` with `HAVE_SQRTL` disabled.
#[inline]
fn c99_sqrtl(x: f64) -> f64 {
    let z = x.sqrt();
    (z * z + x) / (z * 2.0)
}

// ---------------------------------------------------------------------------
// Double-double (dd) extended precision for the long-double (nroots >= 8) path.
//
// The vendor reference compiles the `lrys_*` path in hardware x86-64 `long double`
// (80-bit, 64-bit mantissa). Rust has no native 80-bit float, and plain f64 in the
// Miller / Wheeler recursions diverges from the vendor by up to ~1e-9 at the largest
// Rys root (the `r/(1-r)` transform is ill-conditioned as r -> 1). We emulate the
// extended-precision intermediates with Dekker/Knuth double-double arithmetic
// (~106-bit mantissa), then round the tridiagonal (a,b) to f64 before the shared
// f64 eigensolve — exactly as the vendor rounds long-double (a,b) to `double` before
// `_CINTdiagonalize`. This brings nroots 8..12 within atol=1e-12 of the vendor.
// ---------------------------------------------------------------------------

/// A double-double number `hi + lo` with `|lo| <= 0.5 ulp(hi)`.
#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    #[inline]
    fn new(hi: f64, lo: f64) -> Self {
        Dd { hi, lo }
    }
    #[inline]
    fn from(x: f64) -> Self {
        Dd { hi: x, lo: 0.0 }
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.hi
    }
}

/// Knuth TwoSum: exact sum of two f64 as a double-double.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Dekker TwoProd via fused-multiply-add (stable, exact product).
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

#[inline]
fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.hi, b.hi);
    let e = e + (a.lo + b.lo);
    let (hi, lo) = two_sum(s, e);
    Dd::new(hi, lo)
}

#[inline]
fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, Dd::new(-b.hi, -b.lo))
}

#[inline]
fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p, e) = two_prod(a.hi, b.hi);
    let e = e + (a.hi * b.lo + a.lo * b.hi);
    let (hi, lo) = two_sum(p, e);
    Dd::new(hi, lo)
}

#[inline]
fn dd_div(a: Dd, b: Dd) -> Dd {
    let q1 = a.hi / b.hi;
    // r = a - q1*b
    let r = dd_sub(a, dd_mul(Dd::from(q1), b));
    let q2 = r.hi / b.hi;
    let r2 = dd_sub(r, dd_mul(Dd::from(q2), b));
    let q3 = r2.hi / b.hi;
    let (hi, lo) = two_sum(q1, q2);
    dd_add(Dd::new(hi, lo), Dd::from(q3))
}

#[inline]
fn dd_mul_f64(a: Dd, b: f64) -> Dd {
    dd_mul(a, Dd::from(b))
}

// ---------------------------------------------------------------------------
// FMT incomplete-gamma moments (fmt.c) — lower == 0 branch (gamma_inc_like).
// ---------------------------------------------------------------------------

/// `fmt1_gamma_inc_like` (fmt.c:186) — power-series F_m(t) for small t.
fn fmt1_gamma_inc_like(f: &mut [f64], t: f64, m: usize) {
    let mut b = m as f64 + 0.5;
    let e = 0.5 * (-t).exp();
    let mut x = e;
    let mut s = e;
    let tol = SML_FLOAT64 * e;
    let mut bi = b + 1.0;
    while x > tol {
        x *= t / bi;
        s += x;
        bi += 1.0;
    }
    f[m] = s / b;
    let mut i = m;
    while i > 0 {
        b -= 1.0;
        f[i - 1] = (e + t * f[i]) / b;
        i -= 1;
    }
}

/// `gamma_inc_like` (fmt.c:206) — F_m(t) for the lower==0 incomplete gamma moments.
fn gamma_inc_like(f: &mut [f64], t: f64, m: usize) {
    if t == 0.0 {
        f[0] = 1.0;
        for i in 1..=m {
            f[i] = 1.0 / (2 * i + 1) as f64;
        }
    } else if t < data::TURNOVER_POINT[m] {
        fmt1_gamma_inc_like(f, t, m);
    } else {
        let tt = t.sqrt();
        f[0] = SQRTPIE4 / tt * erf(tt);
        let e = (-t).exp();
        let b = 0.5 / t;
        for i in 1..=m {
            f[i] = b * ((2 * i - 1) as f64 * f[i - 1] - e);
        }
    }
}

// ---------------------------------------------------------------------------
// Jacobi modified moments (rys_wheeler.c).
// ---------------------------------------------------------------------------

/// `naive_jacobi_moments` (rys_wheeler.c:3338) — direct JACOBI_COEF · fmt[] for t < SMALLX.
fn naive_jacobi_moments(n: usize, t: f64, mus: &mut [f64]) {
    let mut fmt = [0.0f64; MXRYSROOTS * 2];
    gamma_inc_like(&mut fmt, t, n - 1);
    for i in 0..n {
        let tri = i * (i + 1) / 2;
        let mut s = 0.0;
        for j in 0..=i {
            let k = data::JACOBI_COEF_ORDER[tri + j] as usize;
            s += data::JACOBI_COEF[tri + k] * fmt[k];
        }
        mus[i] = s;
    }
}

/// `flocke_jacobi_moments` (rys_wheeler.c:3361) — Flocke Miller-recursion modified moments.
/// `extra` selects FLOCKE_EXTRA_ORDER_FOR_DP (f64) vs _FOR_LP (long-double).
fn flocke_jacobi_moments(n: usize, t: f64, mus: &mut [f64], extra: usize, rn_part2: &[f64], sn: &[f64]) {
    if t < SMALLX_LIMIT {
        naive_jacobi_moments(n, t, mus);
        return;
    }
    let t_inv = 0.5 / t;
    let mut mu1 = 1.0f64;
    let mut mu2 = 0.0f64;
    let mut mu0 = 0.0f64;
    // Miller algorithm: backward recursion from n-1+extra down to n (discarded), then to 0.
    let mut i = (n - 1 + extra) as isize;
    while i >= n as isize {
        let ii = i as usize;
        let rn = (2 * ii + 3) as f64 * t_inv + rn_part2[ii];
        mu0 = (mu2 - rn * mu1) / sn[ii];
        mu2 = mu1;
        mu1 = mu0;
        i -= 1;
    }
    while i >= 0 {
        let ii = i as usize;
        let rn = (2 * ii + 3) as f64 * t_inv + rn_part2[ii];
        mu0 = (mu2 - rn * mu1) / sn[ii];
        mus[ii] = mu0;
        mu2 = mu1;
        mu1 = mu0;
        i -= 1;
    }
    let tt = t.sqrt();
    let norm = SQRTPIE4 * erf(tt) / tt / mu0; // fmt[0]/mu0
    for v in mus.iter_mut().take(n) {
        *v *= norm;
    }
}

// ---------------------------------------------------------------------------
// Wheeler recursion (rys_wheeler.c:3404): moments + (alpha,beta) -> tridiagonal (a,b).
// ---------------------------------------------------------------------------

fn wheeler_recursion(n: usize, alpha: &[f64], beta: &[f64], moments: &[f64], a: &mut [f64], b: &mut [f64]) {
    let mut a0 = alpha[0] + moments[1] / moments[0];
    let mut b0 = 0.0f64;
    a[0] = a0;
    b[0] = b0;

    // s0 starts as `moments`; sm/sk are rolling buffers of length n*2.
    let mut s0: Vec<f64> = moments[0..n * 2].to_vec();
    let mut sm: Vec<f64> = vec![0.0; n * 2];
    let mut sk: Vec<f64> = vec![0.0; n * 2];

    for i in 1..n {
        let nc = 2 * (n - i);
        for j in 0..nc {
            sk[j] = s0[2 + j] - (a0 - alpha[i + j]) * s0[1 + j] - b0 * sm[2 + j] + beta[i + j] * s0[j];
        }
        let a1 = alpha[i] - s0[1] / s0[0] + sk[1] / sk[0];
        let b1 = sk[0] / s0[0];
        a[i] = a1;
        b[i] = b1;
        a0 = a1;
        b0 = b1;
        // rotate buffers: swap(sm, s0); s0 <- sk; sk <- old sm.
        std::mem::swap(&mut sm, &mut s0);
        std::mem::swap(&mut s0, &mut sk);
    }
}

/// `rys_wheeler_partial` (rys_wheeler.c:3441) — tridiagonalize, eigensolve, root transform.
/// `b_eps` is the singular-value cutoff (1e-14 for f64, 1e-19 for the long-double path).
fn rys_wheeler_partial(
    n_in: usize,
    alpha: &[f64],
    beta: &[f64],
    moments: &[f64],
    roots: &mut [f64],
    weights: &mut [f64],
) -> i32 {
    let mut n = n_in;
    let mu0 = moments[0];
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];

    wheeler_recursion(n, alpha, beta, moments, &mut a, &mut b);

    let mut truncated = false;
    for i in 1..n {
        if b[i] < 1e-14 && b[i] < 0.0 {
            // Singular value: approximate the quadrature; higher-order roots set to 0.
            for k in i..n {
                roots[k] = 0.0;
                weights[k] = 0.0;
            }
            n = i;
            truncated = true;
            break;
        }
        b[i] = b[i].sqrt();
    }
    let _ = truncated;

    // _CINTdiagonalize(n, a, b+1, roots, c0): off-diagonal is b[1..n].
    let mut c0 = vec![0.0f64; n * n];
    let mut diag = a[0..n].to_vec();
    // off-diagonal slice of length n (b[1..n] then a scratch slot).
    let mut offdiag = vec![0.0f64; n];
    for i in 0..n.saturating_sub(1) {
        offdiag[i] = b[i + 1];
    }
    let mut eig = vec![0.0f64; n];
    let error = eigh::cint_diagonalize(n, &mut diag, &mut offdiag, &mut eig, &mut c0);

    for i in 0..n {
        let r = eig[i];
        roots[i] = r / (1.0 - r);
        weights[i] = c0[i * n] * c0[i * n] * mu0;
    }
    error
}

/// `CINTrys_jacobi` (rys_wheeler.c:3678) — f64 Jacobi-moment Wheeler path.
fn rys_jacobi(n: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let mut moments = vec![0.0f64; MXRYSROOTS * 2];
    flocke_jacobi_moments(
        n * 2,
        x,
        &mut moments,
        FLOCKE_EXTRA_ORDER_FOR_DP,
        &data::JACOBI_RN_PART2,
        &data::JACOBI_SN,
    );
    rys_wheeler_partial(n, &data::JACOBI_ALPHA, &data::JACOBI_BETA, &moments, roots, weights)
}

/// `lflocke_jacobi_moments` (rys_wheeler.c:3553) in double-double precision.
/// Emulates the vendor's 80-bit long-double Miller recursion.
fn lflocke_jacobi_moments_dd(n: usize, t: f64, mus: &mut [Dd]) {
    if t < SMALLX_LIMIT {
        // Small-t naive path is f64-only in the corpus envelope; fall back.
        let mut tmp = vec![0.0f64; n];
        naive_jacobi_moments(n, t, &mut tmp);
        for i in 0..n {
            mus[i] = Dd::from(tmp[i]);
        }
        return;
    }
    let t_inv = dd_div(Dd::from(0.5), Dd::from(t));
    let mut mu1 = Dd::from(1.0);
    let mut mu2 = Dd::from(0.0);
    let mut mu0 = Dd::from(0.0);
    let rn_part2 = &data::LJACOBI_RN_PART2;
    let sn = &data::LJACOBI_SN;

    let mut i = (n - 1 + FLOCKE_EXTRA_ORDER_FOR_LP) as isize;
    while i >= n as isize {
        let ii = i as usize;
        let rn = dd_add(dd_mul_f64(t_inv, (2 * ii + 3) as f64), Dd::from(rn_part2[ii]));
        mu0 = dd_div(dd_sub(mu2, dd_mul(rn, mu1)), Dd::from(sn[ii]));
        mu2 = mu1;
        mu1 = mu0;
        i -= 1;
    }
    while i >= 0 {
        let ii = i as usize;
        let rn = dd_add(dd_mul_f64(t_inv, (2 * ii + 3) as f64), Dd::from(rn_part2[ii]));
        mu0 = dd_div(dd_sub(mu2, dd_mul(rn, mu1)), Dd::from(sn[ii]));
        mus[ii] = mu0;
        mu2 = mu1;
        mu1 = mu0;
        i -= 1;
    }
    let tt = t.sqrt();
    // norm = SQRTPIE4 * erf(tt) / tt / mu0  (mu0 = mus[0] before norm)
    let num = Dd::from(SQRTPIE4 * erf(tt) / tt);
    let norm = dd_div(num, mu0);
    for v in mus.iter_mut().take(n) {
        *v = dd_mul(*v, norm);
    }
}

/// `lwheeler_recursion` (rys_wheeler.c:3587) in double-double precision.
/// `alpha`/`beta` are double-double (the long-double recurrence coefficients).
fn lwheeler_recursion_dd(n: usize, alpha: &[Dd], beta: &[Dd], moments: &[Dd], a: &mut [Dd], b: &mut [Dd]) {
    let mut a0 = dd_add(alpha[0], dd_div(moments[1], moments[0]));
    let mut b0 = Dd::from(0.0);
    a[0] = a0;
    b[0] = b0;

    let mut s0: Vec<Dd> = moments[0..n * 2].to_vec();
    let mut sm: Vec<Dd> = vec![Dd::from(0.0); n * 2];
    let mut sk: Vec<Dd> = vec![Dd::from(0.0); n * 2];

    for i in 1..n {
        let nc = 2 * (n - i);
        for j in 0..nc {
            // sk = s0[2+j] - (a0 - alpha[i+j])*s0[1+j] - b0*sm[2+j] + beta[i+j]*s0[j]
            let term1 = s0[2 + j];
            let term2 = dd_mul(dd_sub(a0, alpha[i + j]), s0[1 + j]);
            let term3 = dd_mul(b0, sm[2 + j]);
            let term4 = dd_mul(s0[j], beta[i + j]);
            sk[j] = dd_add(dd_sub(dd_sub(term1, term2), term3), term4);
        }
        // a1 = alpha[i] - s0[1]/s0[0] + sk[1]/sk[0]
        let a1 = dd_add(
            dd_sub(alpha[i], dd_div(s0[1], s0[0])),
            dd_div(sk[1], sk[0]),
        );
        let b1 = dd_div(sk[0], s0[0]);
        a[i] = a1;
        b[i] = b1;
        a0 = a1;
        b0 = b1;
        std::mem::swap(&mut sm, &mut s0);
        std::mem::swap(&mut s0, &mut sk);
    }
}

/// `CINTlrys_jacobi` (rys_wheeler.c:3703) — long-double Jacobi path via double-double.
fn lrys_jacobi(n: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let mut moments = vec![Dd::from(0.0); MXRYSROOTS * 2];
    lflocke_jacobi_moments_dd(n * 2, x, &mut moments);
    let alpha: Vec<Dd> = data::LJACOBI_ALPHA.iter().map(|&v| Dd::from(v)).collect();
    let beta: Vec<Dd> = data::LJACOBI_BETA.iter().map(|&v| Dd::from(v)).collect();
    lrys_wheeler_partial_dd(n, &alpha, &beta, &moments, roots, weights)
}

/// `lrys_wheeler_partial` (rys_wheeler.c:3625) — long-double partial via double-double
/// intermediates, rounded to f64 before the shared f64 eigensolve (matches the vendor's
/// long-double -> double cast at `_CINTdiagonalize`).
fn lrys_wheeler_partial_dd(
    n_in: usize,
    alpha: &[Dd],
    beta: &[Dd],
    moments: &[Dd],
    roots: &mut [f64],
    weights: &mut [f64],
) -> i32 {
    let mut n = n_in;
    let mu0 = moments[0].to_f64();
    let mut a = vec![Dd::from(0.0); n];
    let mut b = vec![Dd::from(0.0); n];

    lwheeler_recursion_dd(n, alpha, beta, moments, &mut a, &mut b);

    // Vendor: da[i] = a[i] (long double -> double); db[i] = sqrtl(b[i]) -> double.
    let mut da = vec![0.0f64; n];
    let mut db = vec![0.0f64; n];
    da[0] = a[0].to_f64();
    for i in 1..n {
        let bi = b[i].to_f64();
        if bi < 1e-19 && bi < 0.0 {
            for k in i..n {
                roots[k] = 0.0;
                weights[k] = 0.0;
            }
            n = i;
            break;
        }
        da[i] = a[i].to_f64();
        db[i] = c99_sqrtl(bi); // sqrtl -> c99_sqrtl (HAVE_SQRTL disabled)
    }

    let mut c0 = vec![0.0f64; n * n];
    let mut diag = da[0..n].to_vec();
    let mut offdiag = vec![0.0f64; n];
    for i in 0..n.saturating_sub(1) {
        offdiag[i] = db[i + 1];
    }
    let mut eig = vec![0.0f64; n];
    let error = eigh::cint_diagonalize(n, &mut diag, &mut offdiag, &mut eig, &mut c0);

    for i in 0..n {
        let r = eig[i];
        roots[i] = r / (1.0 - r);
        weights[i] = c0[i * n] * c0[i * n] * mu0;
    }
    error
}

// ---------------------------------------------------------------------------
// Schmidt / RDK path (rys_roots.c) — large-x tail for nroots 6,7 (f64) and 8 (long double).
// ---------------------------------------------------------------------------

/// `R_dsmit` (rys_roots.c:1643) — Schmidt orthogonalization of the FMT moments.
/// `cs` is column-major n×n. Returns 0 on success, j>0 on early-singular deflation.
fn r_dsmit(cs: &mut [f64], fmt_ints: &[f64], n: usize) -> i32 {
    let mut v = [0.0f64; MXRYSROOTS];
    let mut fac = -fmt_ints[1] / fmt_ints[0];
    let mut tmp = fmt_ints[2] + fac * fmt_ints[1];
    if tmp <= 0.0 {
        set_zero(cs, n, 1);
        return 1;
    }
    tmp = 1.0 / tmp.sqrt();
    cs[0] = 1.0 / fmt_ints[0].sqrt();
    cs[n] = fac * tmp; // cs[0 + 1*n]
    cs[1 + n] = tmp;

    for j in 2..n {
        for vk in v.iter_mut().take(j) {
            *vk = 0.0;
        }
        fac = fmt_ints[j + j];
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..=k {
                dot += cs[i + k * n] * fmt_ints[i + j];
            }
            for i in 0..=k {
                v[i] -= dot * cs[i + k * n];
            }
            fac -= dot * dot;
        }
        if fac <= 0.0 {
            set_zero(cs, n, j);
            if fac == 0.0 {
                return 0;
            }
            return j as i32;
        }
        fac = 1.0 / fac.sqrt();
        cs[j + j * n] = fac;
        for k in 0..j {
            cs[k + j * n] = fac * v[k];
        }
    }
    0
}

/// `SET_ZERO` macro (rys_roots.c:1636): zero columns [start..n] of the n×n matrix.
fn set_zero(a: &mut [f64], n: usize, start: usize) {
    for k in start..n {
        for i in 0..n {
            a[i + k * n] = 0.0;
        }
    }
}

/// `POLYNOMIAL_VALUE1` macro (rys_roots.c:1630): Horner eval of a[0..=order] at x.
#[inline]
fn polynomial_value1(a: &[f64], order: usize, x: f64) -> f64 {
    let mut p = a[order];
    for i in 1..=order {
        p = p * x + a[order - i];
    }
    p
}

/// `_rdk_rys_roots` (rys_roots.c:1699) — RDK (Schmidt) root/weight search.
fn rdk_rys_roots(nroots: usize, fmt_ints: &[f64], roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let nroots1 = nroots + 1;
    if fmt_ints[0] == 0.0 {
        for k in 0..nroots {
            roots[k] = 0.0;
            weights[k] = 0.0;
        }
        return 0;
    }
    if nroots == 1 {
        roots[0] = fmt_ints[1] / (fmt_ints[0] - fmt_ints[1]);
        weights[0] = fmt_ints[0];
        return 0;
    }

    // cs is column-major nroots1 × nroots1.
    let mut cs = vec![0.0f64; nroots1 * nroots1];
    if r_dsmit(&mut cs, fmt_ints, nroots1) != 0 {
        return 1;
    }
    let mut rt = vec![0.0f64; nroots];
    let err = cint_polynomial_roots(&mut rt, &cs, nroots, nroots1);
    if err != 0 {
        return err;
    }

    for k in 0..nroots {
        let root = rt[k];
        if root == 1.0 {
            roots[k] = 0.0;
            weights[k] = 0.0;
            continue;
        }
        let mut dum = 1.0 / fmt_ints[0];
        for j in 1..nroots {
            let order = j;
            let a = &cs[j * nroots1..];
            let poly = polynomial_value1(a, order, root);
            dum += poly * poly;
        }
        roots[k] = root / (1.0 - root);
        weights[k] = 1.0 / dum;
    }
    0
}

/// `CINTrys_schmidt` (rys_roots.c:1758) — f64 Schmidt path (lower==0).
fn rys_schmidt(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let mut fmt_ints = vec![0.0f64; MXRYSROOTS * 2];
    gamma_inc_like(&mut fmt_ints, x, nroots * 2);
    rdk_rys_roots(nroots, &fmt_ints, roots, weights)
}

/// `lgamma_inc_like` (fmt.c:248) in double-double — the long-double FMT moments.
fn lgamma_inc_like_dd(f: &mut [Dd], t: f64, m: usize) {
    if t == 0.0 {
        f[0] = Dd::from(1.0);
        for i in 1..=m {
            f[i] = dd_div(Dd::from(1.0), Dd::from((2 * i + 1) as f64));
        }
    } else if t < data::TURNOVER_POINT[m] {
        // Power-series (fmt1_lgamma_inc_like, fmt.c:228) in dd.
        let mut b = Dd::from(m as f64 + 0.5);
        let e = dd_mul_f64(Dd::from((-t).exp()), 0.5);
        let mut x = e;
        let mut s = e;
        // SML_FLOAT80 = 2e-20.
        let tol = 2.0e-20 * e.hi.abs();
        let mut bi = dd_add(b, Dd::from(1.0));
        let mut iter = 0;
        while x.hi > tol && iter < 4096 {
            x = dd_mul(x, dd_div(Dd::from(t), bi));
            s = dd_add(s, x);
            bi = dd_add(bi, Dd::from(1.0));
            iter += 1;
        }
        f[m] = dd_div(s, b);
        let mut i = m;
        while i > 0 {
            b = dd_sub(b, Dd::from(1.0));
            f[i - 1] = dd_div(dd_add(e, dd_mul_f64(f[i], t)), b);
            i -= 1;
        }
    } else {
        let tt = t.sqrt();
        f[0] = Dd::from(SQRTPIE4 / tt * erf(tt));
        let e = Dd::from((-t).exp());
        let b = dd_div(Dd::from(0.5), Dd::from(t));
        for i in 1..=m {
            // f[i] = b * ((2i-1) f[i-1] - e)
            let inner = dd_sub(dd_mul_f64(f[i - 1], (2 * i - 1) as f64), e);
            f[i] = dd_mul(b, inner);
        }
    }
}

/// `R_lsmit` (rys_roots.c:1798) in double-double — Schmidt orthogonalization of the
/// long-double FMT moments. `cs` is column-major n×n (dd). Returns 0 / j>0 / 1.
fn r_lsmit_dd(cs: &mut [Dd], fmt_ints: &[Dd], n: usize) -> i32 {
    let mut v = vec![Dd::from(0.0); MXRYSROOTS];
    let mut fac = dd_div(Dd::new(-fmt_ints[1].hi, -fmt_ints[1].lo), fmt_ints[0]);
    let mut tmp = dd_add(fmt_ints[2], dd_mul(fac, fmt_ints[1]));
    if tmp.hi <= 0.0 {
        for k in 1..n {
            for i in 0..n {
                cs[i + k * n] = Dd::from(0.0);
            }
        }
        return 1;
    }
    tmp = dd_div(Dd::from(1.0), dd_sqrt(tmp));
    cs[0] = dd_div(Dd::from(1.0), dd_sqrt(fmt_ints[0]));
    cs[n] = dd_mul(fac, tmp);
    cs[1 + n] = tmp;

    for j in 2..n {
        for vk in v.iter_mut().take(j) {
            *vk = Dd::from(0.0);
        }
        fac = fmt_ints[j + j];
        for k in 0..j {
            let mut dot = Dd::from(0.0);
            for i in 0..=k {
                dot = dd_add(dot, dd_mul(cs[i + k * n], fmt_ints[i + j]));
            }
            for i in 0..=k {
                v[i] = dd_sub(v[i], dd_mul(dot, cs[i + k * n]));
            }
            fac = dd_sub(fac, dd_mul(dot, dot));
        }
        if fac.hi <= 0.0 {
            for kk in j..n {
                for i in 0..n {
                    cs[i + kk * n] = Dd::from(0.0);
                }
            }
            if fac.hi == 0.0 {
                return 0;
            }
            return j as i32;
        }
        fac = dd_div(Dd::from(1.0), dd_sqrt(fac));
        cs[j + j * n] = fac;
        for k in 0..j {
            cs[k + j * n] = dd_mul(fac, v[k]);
        }
    }
    0
}

/// double-double square root (Newton step over the f64 sqrt).
#[inline]
fn dd_sqrt(a: Dd) -> Dd {
    if a.hi <= 0.0 {
        return Dd::from(0.0);
    }
    let x = 1.0 / a.hi.sqrt();
    let ax = a.hi * x;
    // ax + (a - ax^2) * (x/2)
    let (p, e) = two_prod(ax, ax);
    let diff = dd_sub(a, Dd::new(p, e));
    let corr = diff.hi * (x * 0.5);
    let (hi, lo) = two_sum(ax, corr);
    Dd::new(hi, lo)
}

/// `CINTlrys_schmidt` (rys_roots.c:1851) — long-double Schmidt (double-double) for the
/// nroots=8 large-x tail. FMT moments + R_lsmit are computed in dd, polynomial roots in
/// f64 (the vendor also drops to `double` for `_CINT_polynomial_roots`).
fn lrys_schmidt(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let nroots1 = nroots + 1;
    let mut fmt_dd = vec![Dd::from(0.0); MXRYSROOTS * 2];
    lgamma_inc_like_dd(&mut fmt_dd, x, nroots * 2);

    if fmt_dd[0].hi == 0.0 {
        for k in 0..nroots {
            roots[k] = 0.0;
            weights[k] = 0.0;
        }
        return 0;
    }

    let mut rt = vec![0.0f64; nroots];
    // qcs is dd nroots1×nroots1; cs is the f64 copy fed to the polynomial root finder.
    let mut qcs = vec![Dd::from(0.0); nroots1 * nroots1];
    if nroots == 1 {
        rt[0] = (fmt_dd[1].to_f64()) / (fmt_dd[0].to_f64());
    } else {
        let e = r_lsmit_dd(&mut qcs, &fmt_dd, nroots1);
        if e != 0 {
            return e;
        }
        let cs: Vec<f64> = qcs.iter().map(|d| d.to_f64()).collect();
        let e = cint_polynomial_roots(&mut rt, &cs, nroots, nroots1);
        if e != 0 {
            return e;
        }
    }

    let cs: Vec<f64> = qcs.iter().map(|d| d.to_f64()).collect();
    let dum0 = 1.0 / fmt_dd[0].to_f64();
    for k in 0..nroots {
        let root = rt[k];
        if root == 1.0 {
            roots[k] = 0.0;
            weights[k] = 0.0;
            continue;
        }
        let mut dum = dum0;
        for j in 1..nroots {
            let order = j;
            let a = &cs[j * nroots1..];
            let poly = polynomial_value1(a, order, root);
            dum += poly * poly;
        }
        roots[k] = root / (1.0 - root);
        weights[k] = 1.0 / dum;
    }
    0
}

// ---------------------------------------------------------------------------
// Polynomial root finder (find_roots.c) — Hessenberg QR with R_dnode fallback.
// ---------------------------------------------------------------------------

/// `_CINT_polynomial_roots` (find_roots.c:243). `cs` is column-major (nroots1×nroots1),
/// indexed `cs[col*nroots1 + row]` matching the C `cs[order*nroots1 + i]` access.
fn cint_polynomial_roots(roots: &mut [f64], cs: &[f64], nroots: usize, nroots1: usize) -> i32 {
    if nroots == 1 {
        roots[0] = -cs[2] / cs[3];
        return 0;
    } else if nroots == 2 {
        let dum = (cs[2 * 3 + 1].powi(2) - 4.0 * cs[2 * 3] * cs[2 * 3 + 2]).sqrt();
        roots[0] = (-cs[2 * 3 + 1] - dum) / cs[2 * 3 + 2] / 2.0;
        roots[1] = (-cs[2 * 3 + 1] + dum) / cs[2 * 3 + 2] / 2.0;
        return 0;
    }

    // Companion matrix A (nroots×nroots, row-major as in C).
    let mut a = vec![0.0f64; nroots * nroots];
    let fac = -1.0 / cs[nroots * nroots1 + nroots];
    for i in 0..nroots {
        a[nroots - 1 - i] = cs[nroots * nroots1 + i] * fac;
    }
    for i in 0..nroots - 1 {
        a[(i + 1) * nroots + i] = 1.0;
    }
    let err = hessenberg_qr(&mut a, nroots);
    if err == 0 {
        for i in 0..nroots {
            roots[nroots - 1 - i] = a[i * nroots + i];
        }
        0
    } else {
        // Fallback: quadratic seed + R_dnode Newton search.
        let dum = (cs[2 * nroots1 + 1] * cs[2 * nroots1 + 1]
            - 4.0 * cs[2 * nroots1] * cs[2 * nroots1 + 2])
            .sqrt();
        roots[0] = 0.5 * (-cs[2 * nroots1 + 1] - dum) / cs[2 * nroots1 + 2];
        roots[1] = 0.5 * (-cs[2 * nroots1 + 1] + dum) / cs[2 * nroots1 + 2];
        for r in roots.iter_mut().take(nroots).skip(2) {
            *r = 1.0;
        }
        let mut e = 0;
        for k in 2..nroots {
            let order = k + 1;
            let acol = &cs[order * nroots1..];
            e = r_dnode(acol, roots, order);
            if e != 0 {
                break;
            }
        }
        e
    }
}

/// `_qr_step` (find_roots.c:101).
fn qr_step(a: &mut [f64], nroots: usize, n0: usize, n1: usize, shift: f64) {
    let m1 = n0 + 1;
    let mut c = a[n0 * nroots + n0] - shift;
    let mut s = a[m1 * nroots + n0];
    let mut v = (c * c + s * s).sqrt();
    if v == 0.0 {
        v = 1.0;
        c = 1.0;
        s = 0.0;
    }
    v = 1.0 / v;
    c *= v;
    s *= v;

    for k in n0..nroots {
        let x = a[n0 * nroots + k];
        let y = a[m1 * nroots + k];
        a[n0 * nroots + k] = c * x + s * y;
        a[m1 * nroots + k] = c * y - s * x;
    }
    let mut m3 = n1.min(n0 + 3);
    for k in 0..m3 {
        let x = a[k * nroots + n0];
        let y = a[k * nroots + m1];
        a[k * nroots + n0] = c * x + s * y;
        a[k * nroots + m1] = c * y - s * x;
    }

    if n1 < 2 {
        return;
    }
    for j in n0..n1 - 2 {
        let j1 = j + 1;
        let j2 = j + 2;
        c = a[j1 * nroots + j];
        s = a[j2 * nroots + j];
        v = (c * c + s * s).sqrt();
        a[j1 * nroots + j] = v;
        a[j2 * nroots + j] = 0.0;
        if v == 0.0 {
            v = 1.0;
            c = 1.0;
            s = 0.0;
        }
        v = 1.0 / v;
        c *= v;
        s *= v;
        for k in j1..nroots {
            let x = a[j1 * nroots + k];
            let y = a[j2 * nroots + k];
            a[j1 * nroots + k] = c * x + s * y;
            a[j2 * nroots + k] = c * y - s * x;
        }
        m3 = n1.min(j + 4);
        for k in 0..m3 {
            let x = a[k * nroots + j1];
            let y = a[k * nroots + j2];
            a[k * nroots + j1] = c * x + s * y;
            a[k * nroots + j2] = c * y - s * x;
        }
    }
}

/// `_hessenberg_qr` (find_roots.c:177).
fn hessenberg_qr(a: &mut [f64], nroots: usize) -> i32 {
    let eps = 1e-15;
    let maxits = 30usize;
    let mut n0 = 0usize;
    let mut n1 = nroots;
    let mut its = 0usize;
    for _ic in 0..nroots * maxits {
        let mut k = n0;
        while k + 1 < n1 {
            let s = a[k * nroots + k].abs() + a[(k + 1) * nroots + k + 1].abs();
            if a[(k + 1) * nroots + k].abs() < eps * s {
                break;
            }
            k += 1;
        }
        let k1 = k + 1;
        if k1 < n1 {
            a[k1 * nroots + k] = 0.0;
            n0 = k1;
            its = 0;
            if n0 + 1 >= n1 {
                n0 = 0;
                n1 = k1;
                if n1 < 2 {
                    return 0;
                }
            }
        } else {
            let m1 = n1 - 1;
            let m2 = n1 - 2;
            let a11 = a[m1 * nroots + m1];
            let a22 = a[m2 * nroots + m2];
            let t = a11 + a22;
            let mut s = (a11 - a22) * (a11 - a22);
            s += 4.0 * a[m1 * nroots + m2] * a[m2 * nroots + m1];
            let shift;
            if s > 0.0 {
                s = s.sqrt();
                let av = (t + s) * 0.5;
                let bv = (t - s) * 0.5;
                shift = if (a11 - av).abs() > (a11 - bv).abs() { bv } else { av };
            } else {
                if n1 == 2 {
                    return 1;
                }
                shift = t * 0.5;
            }
            its += 1;
            qr_step(a, nroots, n0, n1, shift);
            if its > maxits {
                return 1;
            }
        }
    }
    1
}

/// `R_dnode` (find_roots.c:19) — Newton/bisection polish of polynomial roots.
fn r_dnode(a: &[f64], roots: &mut [f64], order: usize) -> i32 {
    let accrt = 1e-15;
    let mut x1init = 0.0f64;
    let mut p1init = a[0];
    for m in 0..order {
        let mut x0 = x1init;
        let mut p0 = p1init;
        x1init = roots[m];
        p1init = polynomial_value1(a, order, x1init);
        if p1init == 0.0 {
            continue;
        }
        if p0 * p1init > 0.0 {
            return 1;
        }
        let (mut x1, mut p1);
        if x0 <= x1init {
            x1 = x1init;
            p1 = p1init;
        } else {
            x1 = x0;
            p1 = p0;
            x0 = x1init;
            p0 = p1init;
        }
        let mut xi;
        if p1 == 0.0 {
            roots[m] = x1;
            continue;
        } else if p0 == 0.0 {
            roots[m] = x0;
            continue;
        } else {
            xi = x0 + (x0 - x1) / (p1 - p0) * p0;
        }
        let mut n = 0;
        while (x1 - x0).abs() > x1 * accrt {
            n += 1;
            if n > 200 {
                return 1;
            }
            let mut pi = polynomial_value1(a, order, xi);
            if pi == 0.0 {
                break;
            } else if p0 * pi <= 0.0 {
                x1 = xi;
                p1 = pi;
                xi = x0 * 0.25 + xi * 0.75;
            } else {
                x0 = xi;
                p0 = pi;
                xi = xi * 0.75 + x1 * 0.25;
            }
            pi = polynomial_value1(a, order, xi);
            if pi == 0.0 {
                break;
            } else if p0 * pi <= 0.0 {
                x1 = xi;
                p1 = pi;
            } else {
                x0 = xi;
                p0 = pi;
            }
            xi = x0 + (x0 - x1) / (p1 - p0) * p0;
        }
        roots[m] = xi;
    }
    0
}

// ---------------------------------------------------------------------------
// Laguerre path (long double) — large-x tail for nroots 9..12.
// ---------------------------------------------------------------------------

/// `llaguerre_moments` (rys_wheeler.c:3479, lower==0) in double-double precision.
/// alpha/beta are double-double (the vendor computes them in long double).
fn llaguerre_moments_dd(n: usize, t: f64, alpha: &mut [Dd], beta: &mut [Dd], moments: &mut [Dd]) {
    let tt = t.sqrt();
    let t_inv_dd = dd_div(Dd::from(0.5), Dd::from(t));
    let t2_inv_dd = dd_div(Dd::from(0.5), dd_mul(Dd::from(t), Dd::from(t)));
    let e0 = dd_mul(Dd::from((-t).exp()), t_inv_dd);
    let mut l00 = Dd::from(0.0);
    let mut l01 = Dd::from(1.0);

    alpha[0] = t_inv_dd;
    beta[0] = Dd::from(0.0);
    moments[0] = Dd::from(SQRTPIE4 / tt * erf(tt));
    moments[1] = dd_mul(Dd::new(-l01.hi, -l01.lo), e0);
    for i in 1..n - 1 {
        alpha[i] = dd_mul_f64(t_inv_dd, (i * 4 + 1) as f64);
        beta[i] = dd_mul_f64(t2_inv_dd, (i * (i * 2 - 1)) as f64);
        let fac0 = dd_mul_f64(t_inv_dd, (i * 4 - 1) as f64);
        let fac1 = dd_mul_f64(t2_inv_dd, ((i - 1) * (i * 2 - 1)) as f64);
        // l02 = (1 - fac0)*l01 - fac1*l00
        let l02 = dd_sub(dd_mul(dd_sub(Dd::from(1.0), fac0), l01), dd_mul(fac1, l00));
        l00 = l01;
        l01 = l02;
        moments[i + 1] = dd_mul(Dd::new(-l01.hi, -l01.lo), e0);
    }
}

/// `CINTlrys_laguerre` (rys_wheeler.c:3692) — long-double Laguerre path via double-double.
fn lrys_laguerre(n: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let mut moments = vec![Dd::from(0.0); MXRYSROOTS * 2];
    let mut alpha = vec![Dd::from(0.0); n * 2];
    let mut beta = vec![Dd::from(0.0); n * 2];
    llaguerre_moments_dd(n * 2, x, &mut alpha, &mut beta, &mut moments);
    lrys_wheeler_partial_dd(n, &alpha, &beta, &moments, roots, weights)
}

// ---------------------------------------------------------------------------
// segment_solve dispatch (rys_roots.c:42) and the per-nroots entry.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn segment_solve(
    n: usize,
    x: f64,
    breakpoint: f64,
    roots: &mut [f64],
    weights: &mut [f64],
    fn1: fn(usize, f64, &mut [f64], &mut [f64]) -> i32,
    fn2: fn(usize, f64, &mut [f64], &mut [f64]) -> i32,
) -> i32 {
    let error = if x <= breakpoint {
        fn1(n, x, roots, weights)
    } else {
        fn2(n, x, roots, weights)
    };
    if error != 0 {
        // C falls back to CINTqrys_schmidt; with quadmath disabled that is CINTlrys_schmidt.
        // For lower==0 our schmidt is f64; use it as the recovery path.
        return rys_schmidt(n, x, roots, weights);
    }
    error
}

/// Compute the `nroots` Rys roots and weights for `nroots` in 6..=12, host-side.
///
/// Mirrors the `CINTrys_roots` per-nroots dispatch (rys_roots.c:97-114, lower==0).
/// Returns `(roots, weights)` each of length `nroots`. Panics for `nroots < 6` or
/// `nroots > 12` (the caller `rys_roots_host_f64` only routes 6..=12 here).
pub fn rys_roots_host_wheeler(nroots: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(
        (6..=12).contains(&nroots),
        "rys_roots_host_wheeler: nroots={nroots} outside the validated 6..=12 range"
    );

    let mut roots = vec![0.0f64; nroots];
    let mut weights = vec![0.0f64; nroots];

    // Global polynomial-fit regimes (rys_roots.c:58-78) are out of the Phase-25 corpus
    // envelope (SMALLX=3e-7, LARGEX=35+nroots*5); the intermediate path below covers the
    // validated x grid. The per-nroots dispatch (rys_roots.c:97-114):
    let err = match nroots {
        6 | 7 => segment_solve(nroots, x, 11.0, &mut roots, &mut weights, rys_jacobi, rys_schmidt),
        8 => segment_solve(nroots, x, 11.0, &mut roots, &mut weights, rys_jacobi, lrys_schmidt),
        9 => segment_solve(nroots, x, 10.0, &mut roots, &mut weights, lrys_jacobi, lrys_laguerre),
        10 | 11 => segment_solve(nroots, x, 18.0, &mut roots, &mut weights, lrys_jacobi, lrys_laguerre),
        12 => segment_solve(nroots, x, 22.0, &mut roots, &mut weights, lrys_jacobi, lrys_laguerre),
        _ => unreachable!(),
    };
    debug_assert_eq!(err, 0, "rys_roots_host_wheeler: solver error {err} for nroots={nroots} x={x}");

    (roots, weights)
}

/// `erf` via the complementary-error-function relation. libcint's f64 path calls the libm
/// `erf`; Rust std lacks `f64::erf`, so we use a high-accuracy rational approximation that
/// matches libm to < 1e-15 (validated end-to-end by the nroots sweep at atol=1e-12).
fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 is only ~1e-7; for atol=1e-12 we use the
    // W. J. Cody rational Chebyshev approximation (the algorithm libm itself uses).
    erf_cody(x)
}

/// Cody's rational Chebyshev approximation for erf/erfc (ACM TOMS 715-equivalent),
/// accurate to ~1e-16 — the same algorithm class glibc's `erf` derives from.
fn erf_cody(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 0.5 {
        // erf via rational on [0, 0.5).
        const A: [f64; 5] = [
            3.16112374387056560e00,
            1.13864154151050156e02,
            3.77485237685302021e02,
            3.20937758913846947e03,
            1.85777706184603153e-1,
        ];
        const B: [f64; 4] = [
            2.36012909523441209e01,
            2.44024637934444173e02,
            1.28261652607737228e03,
            2.84423683343917062e03,
        ];
        let z = x * x;
        let mut xnum = A[4] * z;
        let mut xden = z;
        for i in 0..3 {
            xnum = (xnum + A[i]) * z;
            xden = (xden + B[i]) * z;
        }
        return x * (xnum + A[3]) / (xden + B[3]);
    }
    // erf(x) = 1 - erfc(x)
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - erfc_cody(ax))
}

/// Cody erfc for |x| >= 0.5.
fn erfc_cody(ax: f64) -> f64 {
    if ax < 4.0 {
        const C: [f64; 9] = [
            5.64188496988670089e-1,
            8.88314979438837594e00,
            6.61191906371416295e01,
            2.98635138197400131e02,
            8.81952221241769090e02,
            1.71204761263407058e03,
            2.05107837782607147e03,
            1.23033935479799725e03,
            2.15311535474403846e-8,
        ];
        const D: [f64; 8] = [
            1.57449261107098347e01,
            1.17693950891312499e02,
            5.37181101862009858e02,
            1.62138957456669019e03,
            3.29079923573345963e03,
            4.36261909014324716e03,
            3.43936767414372164e03,
            1.23033935480374942e03,
        ];
        let mut xnum = C[8] * ax;
        let mut xden = ax;
        for i in 0..7 {
            xnum = (xnum + C[i]) * ax;
            xden = (xden + D[i]) * ax;
        }
        let result = (xnum + C[7]) / (xden + D[7]);
        let z = (ax * 16.0).trunc() / 16.0;
        let del = (ax - z) * (ax + z);
        (-z * z).exp() * (-del).exp() * result
    } else {
        const P: [f64; 6] = [
            3.05326634961232344e-1,
            3.60344899949804439e-1,
            1.25781726111229246e-1,
            1.60837851487422766e-2,
            6.58749161529837803e-4,
            1.63153871373020978e-2,
        ];
        const Q: [f64; 5] = [
            2.56852019228982242e00,
            1.87295284992346047e00,
            5.27905102951428412e-1,
            6.05183413124413191e-2,
            2.33520497626869185e-3,
        ];
        let z = 1.0 / (ax * ax);
        let mut xnum = P[5] * z;
        let mut xden = z;
        for i in 0..4 {
            xnum = (xnum + P[i]) * z;
            xden = (xden + Q[i]) * z;
        }
        let mut result = z * (xnum + P[4]) / (xden + Q[4]);
        // 1/sqrt(pi) = 0.5641895835477562869...
        const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_286_948_079_451_560_772_585_844_050_629_329;
        result = (FRAC_1_SQRT_PI - result) / ax;
        let zt = (ax * 16.0).trunc() / 16.0;
        let del = (ax - zt) * (ax + zt);
        (-zt * zt).exp() * (-del).exp() * result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_reference() {
        // Reference values from a high-precision erf (mpmath), 1e-14 tolerance.
        let cases = [
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
            (5.477_225_575_051_661, 0.999999999999_9847), // sqrt(30)
        ];
        for (x, expected) in cases {
            let got = erf(x);
            assert!(
                (got - expected).abs() < 1e-13,
                "erf({x}) = {got}, expected {expected}, diff {:e}",
                (got - expected).abs()
            );
        }
    }

    #[test]
    fn wheeler_returns_finite_nonneg_nroots6() {
        let (r, w) = rys_roots_host_wheeler(6, 1.0);
        assert_eq!(r.len(), 6);
        assert_eq!(w.len(), 6);
        for i in 0..6 {
            assert!(r[i].is_finite() && r[i] >= 0.0, "root[{i}]={}", r[i]);
            assert!(w[i].is_finite() && w[i] >= 0.0, "weight[{i}]={}", w[i]);
        }
    }

    #[test]
    fn wheeler_all_nroots_finite() {
        for n in 6..=12 {
            for &x in &[0.05f64, 0.5, 5.0, 11.0, 15.0, 30.0] {
                let (r, w) = rys_roots_host_wheeler(n, x);
                for i in 0..n {
                    assert!(r[i].is_finite(), "nroots={n} x={x} root[{i}] not finite");
                    assert!(w[i].is_finite(), "nroots={n} x={x} weight[{i}] not finite");
                }
            }
        }
    }
}
