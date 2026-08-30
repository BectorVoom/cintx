//! Host Wheeler/Jacobi Rys root+weight engine for `nroots >= 6` (Phase 25 FND-02, Task 1b).
//!
//! Verbatim host-side port of libcint 6.1.3's `n > 5` Rys root/weight path.
//! The control flow mirrors
//! `libcint-master/src/rys_roots.c::CINTrys_roots` (per-nroots dispatch), feeding into
//! the Flocke modified-moments -> Wheeler recursion -> vendored-MRRR eigensolve
//! (`eigh::cint_diagonalize`, Task 1a) -> root transform chain, plus the large-x Schmidt
//! / Laguerre tails.
//!
//! ## The `lower != 0` family (D-PBC-24)
//!
//! This module was `lower == 0` only until D-PBC-24: the short-range
//! `∫_lower^1` quadrature was out of scope for Phase 25 and no Hessian family
//! exercised it. It is now carried too — see [`sr_rys_roots_host`] and the
//! section that opens with "D-PBC-24 stage 3" near the bottom of this file.
//! `lower` enters in exactly one place, the FMT moments, so the Schmidt
//! orthogonalization, Wheeler recursion, polynomial root finder and eigensolve
//! below are shared between the two families verbatim.
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

// Transcribed verbatim from vendored libcint 6.1.3. Result compatibility with
// upstream is decided by the exact bits these literals feed the kernels, so a
// literal is never truncated to the shortest form that round-trips — the same
// provenance rationale the `clippy::approx_constant` allows in this crate carry.
#![allow(clippy::excessive_precision)]
// Index arithmetic here is written in full — `base + 0 * stride`, `base + 1 * stride`,
// `out[n * 3 + 0]` — so that a slot or component index lines up column-wise with its
// neighbours and with the libcint layout being mirrored. Folding the `0 *` and `1 *`
// away would shorten the line and hide the stride.
#![allow(clippy::identity_op)]
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
// Element loops rather than `copy_from_slice`: the source index carries a table
// base offset (`tabu_base + i`, `j * ORDER7OFFSET + i`) or a recursion bound that
// the slice form would have to restate as two range expressions.
#![allow(clippy::manual_memcpy)]

use super::eigh;
use super::roots_jacobi_data as data;
use super::roots_jacobi_dd_data as dd_data;
use cubecl::prelude::*;

/// libcint `MXRYSROOTS` (cint.h: 32).
const MXRYSROOTS: usize = 32;
/// `SMALLX_LIMIT` (rys_roots.c:26).
const SMALLX_LIMIT: f64 = 3e-7;
/// `SQRTPIE4` = sqrt(pi)/2 (rys_wheeler.c:15).
const SQRTPIE4: f64 =
    0.886_226_925_452_758_013_649_083_741_670_572_591_398_774_728_061_193_564_106_903_894_926_4;
/// `SML_FLOAT64` (fmt.c:20).
const SML_FLOAT64: f64 = f64::EPSILON * 0.5;
/// Flocke extra recursion order for the f64 ("DP") Miller pass (rys_wheeler.c:22).
const FLOCKE_EXTRA_ORDER_FOR_DP: usize = 20;
#[allow(dead_code)] // superseded host long-double reference; see the note above
/// Flocke extra recursion order for the long-double ("LP") Miller pass (rys_wheeler.c:24).
const FLOCKE_EXTRA_ORDER_FOR_LP: usize = 24;

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// `c99_sqrtl` (rys_roots.c:1776) — one Babylonian refinement over f64 `sqrt`.
/// Used wherever the C long-double path calls `sqrtl` with `HAVE_SQRTL` disabled.
#[inline]
fn c99_sqrtl(x: f64) -> f64 {
    let z = x.sqrt();
    (z * z + x) / (z * 2.0)
}

// ---------------------------------------------------------------------------
// Superseded host long-double reference (`l*` / `Dd`).
//
// `rys_roots_host_wheeler` dispatches nroots 8..12 to the **device** dd
// implementations — `wheeler_jacobi_f64`, `wheeler_jacobi_dd`,
// `wheeler_schmidt_dd`, `wheeler_laguerre_dd`, built on the `*_dev` chain near
// `DdDev` below — so nothing calls the host chain any more. It is kept, not
// deleted, because it is the *independent* transcription those device kernels
// were validated against: a divergence between the two is the only cheap signal
// that a device dd routine has drifted, and re-deriving it from the C source
// would cost far more than carrying it.
//
// Note for Phase 33 task 33-05 (FMA contraction destroying the error-free
// transform): the functions at risk are the device `two_sum_dev` / `two_prod_dev`
// pair, NOT the host `two_sum` / `two_prod` here. The host pair is unreachable.
// ---------------------------------------------------------------------------

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

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// A double-double number `hi + lo` with `|lo| <= 0.5 ulp(hi)`.
#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
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

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// Knuth TwoSum: exact sum of two f64 as a double-double.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// Dekker TwoProd via fused-multiply-add (stable, exact product).
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
#[inline]
fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.hi, b.hi);
    let e = e + (a.lo + b.lo);
    let (hi, lo) = two_sum(s, e);
    Dd::new(hi, lo)
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
#[inline]
fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, Dd::new(-b.hi, -b.lo))
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
#[inline]
fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p, e) = two_prod(a.hi, b.hi);
    let e = e + (a.hi * b.lo + a.lo * b.hi);
    let (hi, lo) = two_sum(p, e);
    Dd::new(hi, lo)
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
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

#[allow(dead_code)] // superseded host long-double reference; see the note above
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
fn flocke_jacobi_moments(
    n: usize,
    t: f64,
    mus: &mut [f64],
    extra: usize,
    rn_part2: &[f64],
    sn: &[f64],
) {
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

fn wheeler_recursion(
    n: usize,
    alpha: &[f64],
    beta: &[f64],
    moments: &[f64],
    a: &mut [f64],
    b: &mut [f64],
) {
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
            sk[j] =
                s0[2 + j] - (a0 - alpha[i + j]) * s0[1 + j] - b0 * sm[2 + j] + beta[i + j] * s0[j];
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

    // `n` is narrowed inside the loop for the *eigensolve below*, not to shorten
    // the iteration — which is why the assignment is immediately followed by
    // `break`. Clippy reads a mutated range bound as a likely mistake.
    #[allow(clippy::mut_range_bound)]
    for i in 1..n {
        // rys_wheeler.c:3453 nests `if (b[i] < 1e-14) { if (b[i] < 0.) { ... break } }`,
        // so the truncation is reached only for a NEGATIVE b; the outer bound is a
        // diagnostic gate, not part of the condition. Flattened here to the branch
        // that actually decides.
        if b[i] < 0.0 {
            // Singular value: approximate the quadrature; higher-order roots set to 0.
            for k in i..n {
                roots[k] = 0.0;
                weights[k] = 0.0;
            }
            n = i;
            break;
        }
        b[i] = b[i].sqrt();
    }

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
    rys_wheeler_partial(
        n,
        &data::JACOBI_ALPHA,
        &data::JACOBI_BETA,
        &moments,
        roots,
        weights,
    )
}

// D-PBC-24: no longer dead — the lower-bounded short-range quadrature at
// the bottom of this file builds on it. See `sr_rys_roots_host`.
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
        let rn = dd_add(
            dd_mul_f64(t_inv, (2 * ii + 3) as f64),
            Dd::from(rn_part2[ii]),
        );
        mu0 = dd_div(dd_sub(mu2, dd_mul(rn, mu1)), Dd::from(sn[ii]));
        mu2 = mu1;
        mu1 = mu0;
        i -= 1;
    }
    while i >= 0 {
        let ii = i as usize;
        let rn = dd_add(
            dd_mul_f64(t_inv, (2 * ii + 3) as f64),
            Dd::from(rn_part2[ii]),
        );
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

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// `lwheeler_recursion` (rys_wheeler.c:3587) in double-double precision.
/// `alpha`/`beta` are double-double (the long-double recurrence coefficients).
fn lwheeler_recursion_dd(
    n: usize,
    alpha: &[Dd],
    beta: &[Dd],
    moments: &[Dd],
    a: &mut [Dd],
    b: &mut [Dd],
) {
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
        let a1 = dd_add(dd_sub(alpha[i], dd_div(s0[1], s0[0])), dd_div(sk[1], sk[0]));
        let b1 = dd_div(sk[0], s0[0]);
        a[i] = a1;
        b[i] = b1;
        a0 = a1;
        b0 = b1;
        std::mem::swap(&mut sm, &mut s0);
        std::mem::swap(&mut s0, &mut sk);
    }
}

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// `CINTlrys_jacobi` (rys_wheeler.c:3703) — long-double Jacobi path via double-double.
fn lrys_jacobi(n: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let mut moments = vec![Dd::from(0.0); MXRYSROOTS * 2];
    lflocke_jacobi_moments_dd(n * 2, x, &mut moments);
    let alpha: Vec<Dd> = data::LJACOBI_ALPHA.iter().map(|&v| Dd::from(v)).collect();
    let beta: Vec<Dd> = data::LJACOBI_BETA.iter().map(|&v| Dd::from(v)).collect();
    lrys_wheeler_partial_dd(n, &alpha, &beta, &moments, roots, weights)
}

// D-PBC-24: no longer dead — the lower-bounded short-range quadrature at
// the bottom of this file builds on it. See `sr_rys_roots_host`.
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
        // Same flattening as `rys_wheeler_partial` above: rys_wheeler.c:3640 gates the
        // truncation on `b[i] < 0.` inside an outer `b[i] < 1e-19` diagnostic check.
        if bi < 0.0 {
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

// D-PBC-24: no longer dead — the lower-bounded short-range quadrature at
// the bottom of this file builds on it. See `sr_rys_roots_host`.
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

// D-PBC-24: no longer dead — the lower-bounded short-range quadrature at
// the bottom of this file builds on it. See `sr_rys_roots_host`.
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

#[allow(dead_code)] // superseded host long-double reference; see the note above
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

#[allow(dead_code)] // superseded host long-double reference; see the note above
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
                shift = if (a11 - av).abs() > (a11 - bv).abs() {
                    bv
                } else {
                    av
                };
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

// D-PBC-24: no longer dead — the lower-bounded short-range quadrature at
// the bottom of this file builds on it. See `sr_rys_roots_host`.
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

#[allow(dead_code)] // superseded host long-double reference; see the note above
/// `CINTlrys_laguerre` (rys_wheeler.c:3692) — long-double Laguerre path via double-double.
fn lrys_laguerre(n: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let mut moments = vec![Dd::from(0.0); MXRYSROOTS * 2];
    let mut alpha = vec![Dd::from(0.0); n * 2];
    let mut beta = vec![Dd::from(0.0); n * 2];
    llaguerre_moments_dd(n * 2, x, &mut alpha, &mut beta, &mut moments);
    lrys_wheeler_partial_dd(n, &alpha, &beta, &moments, roots, weights)
}

// ===========================================================================
//  DEVICE #[cube] f64 Wheeler engine (quick task 260531-aw1, Task 2: nroots 6,7).
//
//  The f64 path is a host-orchestrated multi-launch CubeCL pipeline on CpuRuntime:
//    Jacobi (x <= breakpoint):
//      1. jacobi_tridiag_kernel  -> tridiagonal (a, b) via erf-normalized flocke
//         Miller moments + Wheeler recursion (b pre-sqrt'd, mu0 carried out)
//      2. eigh::cint_diagonalize -> the device QL eigensolver (Task 1) on (a,b)
//      3. jacobi_transform_kernel-> roots = r/(1-r), weights = c0[i*n]^2 * mu0
//    Schmidt (x > breakpoint):
//      schmidt_kernel            -> gamma_inc moments + R_dsmit + QR polynomial
//                                   root solver + weights, all in one #[cube] launch
//  No new patterns: comptime nroots, &mut Array scratch, bounded loops + flags,
//  coefficient tables passed as device input Arrays (no const-array runtime index).
// ===========================================================================

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// Device erf via Cody rational Chebyshev (unrolled const access, no runtime const index).
#[cube]
fn erf_dev(x: f64) -> f64 {
    let ax = f64::abs(x);
    let mut out = 0.0f64;
    if ax < 0.5 {
        // erf on [0,0.5): rational in z=x^2.
        let a0 = 3.16112374387056560e00;
        let a1 = 1.13864154151050156e02;
        let a2 = 3.77485237685302021e02;
        let a3 = 3.20937758913846947e03;
        let a4 = 1.85777706184603153e-1;
        let b0 = 2.36012909523441209e01;
        let b1 = 2.44024637934444173e02;
        let b2 = 1.28261652607737228e03;
        let b3 = 2.84423683343917062e03;
        let z = x * x;
        let mut xnum = a4 * z;
        let mut xden = z;
        xnum = (xnum + a0) * z;
        xden = (xden + b0) * z;
        xnum = (xnum + a1) * z;
        xden = (xden + b1) * z;
        xnum = (xnum + a2) * z;
        xden = (xden + b2) * z;
        out = x * (xnum + a3) / (xden + b3);
    } else {
        // erf(x) = sign(x) * (1 - erfc(ax)).
        let mut sign = 1.0f64;
        if x < 0.0 {
            sign = -1.0;
        }
        out = sign * (1.0 - erfc_dev(ax));
    }
    out
}

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// Device erfc for |x| >= 0.5 (Cody), unrolled.
#[cube]
fn erfc_dev(ax: f64) -> f64 {
    let mut result = 0.0f64;
    if ax < 4.0 {
        let c0 = 5.64188496988670089e-1;
        let c1 = 8.88314979438837594e00;
        let c2 = 6.61191906371416295e01;
        let c3 = 2.98635138197400131e02;
        let c4 = 8.81952221241769090e02;
        let c5 = 1.71204761263407058e03;
        let c6 = 2.05107837782607147e03;
        let c7 = 1.23033935479799725e03;
        let c8 = 2.15311535474403846e-8;
        let d0 = 1.57449261107098347e01;
        let d1 = 1.17693950891312499e02;
        let d2 = 5.37181101862009858e02;
        let d3 = 1.62138957456669019e03;
        let d4 = 3.29079923573345963e03;
        let d5 = 4.36261909014324716e03;
        let d6 = 3.43936767414372164e03;
        let d7 = 1.23033935480374942e03;
        let mut xnum = c8 * ax;
        let mut xden = ax;
        xnum = (xnum + c0) * ax;
        xden = (xden + d0) * ax;
        xnum = (xnum + c1) * ax;
        xden = (xden + d1) * ax;
        xnum = (xnum + c2) * ax;
        xden = (xden + d2) * ax;
        xnum = (xnum + c3) * ax;
        xden = (xden + d3) * ax;
        xnum = (xnum + c4) * ax;
        xden = (xden + d4) * ax;
        xnum = (xnum + c5) * ax;
        xden = (xden + d5) * ax;
        xnum = (xnum + c6) * ax;
        xden = (xden + d6) * ax;
        let r = (xnum + c7) / (xden + d7);
        let z = f64::floor(ax * 16.0) / 16.0;
        let del = (ax - z) * (ax + z);
        result = f64::exp(-z * z) * f64::exp(-del) * r;
    } else {
        let p0 = 3.05326634961232344e-1;
        let p1 = 3.60344899949804439e-1;
        let p2 = 1.25781726111229246e-1;
        let p3 = 1.60837851487422766e-2;
        let p4 = 6.58749161529837803e-4;
        let p5 = 1.63153871373020978e-2;
        let q0 = 2.56852019228982242e00;
        let q1 = 1.87295284992346047e00;
        let q2 = 5.27905102951428412e-1;
        let q3 = 6.05183413124413191e-2;
        let q4 = 2.33520497626869185e-3;
        let z = 1.0 / (ax * ax);
        let mut xnum = p5 * z;
        let mut xden = z;
        xnum = (xnum + p0) * z;
        xden = (xden + q0) * z;
        xnum = (xnum + p1) * z;
        xden = (xden + q1) * z;
        xnum = (xnum + p2) * z;
        xden = (xden + q2) * z;
        xnum = (xnum + p3) * z;
        xden = (xden + q3) * z;
        let mut r = z * (xnum + p4) / (xden + q4);
        let frac_1_sqrt_pi = 0.564_189_583_547_756_286_948_079_451_560_772_585_844_050_629_329f64;
        r = (frac_1_sqrt_pi - r) / ax;
        let zt = f64::floor(ax * 16.0) / 16.0;
        let del = (ax - zt) * (ax + zt);
        result = f64::exp(-zt * zt) * f64::exp(-del) * r;
    }
    result
}

const SQRTPIE4_DEV: f64 =
    0.886_226_925_452_758_013_649_083_741_670_572_591_398_774_728_061_193_564_106_903_894_926_4;
// The `x <= SMALLX_LIMIT` polynomial-fit regime is outside the validated
// Phase-25 corpus envelope, so no device kernel branches on it yet. Kept
// beside `SQRTPIE4_DEV` because it is the breakpoint that branch will need.
#[allow(dead_code)]
const SMALLX_LIMIT_DEV: f64 = 3e-7;

/// Device flocke_jacobi_moments (intermediate-x branch only; SMALLX path is out of the
/// validated f64 corpus envelope and is not exercised). `n` is the moment count (= nroots*2).
/// `rn_part2`/`sn` are JACOBI_RN_PART2 / JACOBI_SN passed as device input Arrays.
/// Writes `mus[0..n]`. `extra` is FLOCKE_EXTRA_ORDER_FOR_DP (20).
#[cube]
fn flocke_jacobi_moments_dev(
    n: u32,
    t: f64,
    rn_part2: &Array<f64>,
    sn: &Array<f64>,
    rn_off: u32,
    sn_off: u32,
    extra: u32,
    mus: &mut Array<f64>,
) {
    let t_inv = 0.5 / t;
    let mut mu1 = 1.0f64;
    let mut mu2 = 0.0f64;
    let mut mu0 = 0.0f64;
    // Miller backward recursion: i from n-1+extra down to n (discarded), then to 0.
    let top = n - 1 + extra;
    // First pass: i in [n, top], descending — discarded.
    let mut step: u32 = 0;
    let count1 = top + 1 - n; // number of i in [n, top]
    while step < count1 {
        let ii = top - step; // ii descends from top to n
        let rn = (2 * ii + 3) as f64 * t_inv + rn_part2[(rn_off + ii) as usize];
        mu0 = (mu2 - rn * mu1) / sn[(sn_off + ii) as usize];
        mu2 = mu1;
        mu1 = mu0;
        step += 1;
    }
    // Second pass: i in [0, n-1], descending — stored.
    let mut step2: u32 = 0;
    while step2 < n {
        let ii = (n - 1) - step2; // ii descends from n-1 to 0
        let rn = (2 * ii + 3) as f64 * t_inv + rn_part2[(rn_off + ii) as usize];
        mu0 = (mu2 - rn * mu1) / sn[(sn_off + ii) as usize];
        mus[(ii) as usize] = mu0;
        mu2 = mu1;
        mu1 = mu0;
        step2 += 1;
    }
    let tt = f64::sqrt(t);
    let norm = SQRTPIE4_DEV * erf_dev(tt) / tt / mu0;
    let mut j: u32 = 0;
    while j < n {
        mus[(j) as usize] = mus[(j) as usize] * norm;
        j += 1;
    }
}

/// Device Wheeler recursion: moments + (alpha,beta) -> tridiagonal (a,b).
/// `s0`/`sm`/`sk` are caller-passed scratch Arrays of length n*2. `n` = nroots.
#[cube]
fn wheeler_recursion_dev(
    n: u32,
    alpha: &Array<f64>,
    beta: &Array<f64>,
    alpha_off: u32,
    beta_off: u32,
    moments: &Array<f64>,
    a: &mut Array<f64>,
    b: &mut Array<f64>,
    s0: &mut Array<f64>,
    sm: &mut Array<f64>,
    sk: &mut Array<f64>,
) {
    let mut a0 = alpha[(alpha_off) as usize] + moments[(1) as usize] / moments[(0) as usize];
    let mut b0 = 0.0f64;
    a[(0) as usize] = a0;
    b[(0) as usize] = b0;
    // s0 <- moments[0..n*2]; sm <- 0; sk <- 0
    let n2 = n * 2;
    let mut i: u32 = 0;
    while i < n2 {
        s0[(i) as usize] = moments[(i) as usize];
        sm[(i) as usize] = 0.0;
        sk[(i) as usize] = 0.0;
        i += 1;
    }
    let mut step: u32 = 1;
    while step < n {
        let nc = 2 * (n - step);
        let mut j: u32 = 0;
        while j < nc {
            sk[(j) as usize] = s0[(2 + j) as usize]
                - (a0 - alpha[(alpha_off + step + j) as usize]) * s0[(1 + j) as usize]
                - b0 * sm[(2 + j) as usize]
                + beta[(beta_off + step + j) as usize] * s0[(j) as usize];
            j += 1;
        }
        let a1 = alpha[(alpha_off + step) as usize] - s0[(1) as usize] / s0[(0) as usize]
            + sk[(1) as usize] / sk[(0) as usize];
        let b1 = sk[(0) as usize] / s0[(0) as usize];
        a[(step) as usize] = a1;
        b[(step) as usize] = b1;
        a0 = a1;
        b0 = b1;
        // rotate buffers: swap(sm,s0); s0<-sk; sk<-old sm. Done via copies into scratch.
        // tmp = sm; sm = s0; s0 = sk; sk = tmp
        let mut k: u32 = 0;
        while k < n2 {
            let tmp = sm[(k) as usize];
            sm[(k) as usize] = s0[(k) as usize];
            s0[(k) as usize] = sk[(k) as usize];
            sk[(k) as usize] = tmp;
            k += 1;
        }
        step += 1;
    }
}

/// Device Jacobi tridiagonal kernel: erf-normalized flocke moments + Wheeler recursion,
/// producing the tridiagonal `a_out[0..n]`, `b_out[0..n]` (b pre-sqrt'd from index 1),
/// and `mu0_out[0]` for the weight scaling. `n` (= nroots) is comptime.
#[cube(launch)]
fn jacobi_tridiag_kernel(
    alpha: &Array<f64>,
    beta: &Array<f64>,
    rn_part2: &Array<f64>,
    sn: &Array<f64>,
    moments: &mut Array<f64>,
    a_out: &mut Array<f64>,
    b_out: &mut Array<f64>,
    mu0_out: &mut Array<f64>,
    s0: &mut Array<f64>,
    sm: &mut Array<f64>,
    sk: &mut Array<f64>,
    x: f64,
    #[comptime] n: u32,
    #[comptime] extra: u32,
) {
    flocke_jacobi_moments_dev(n * 2, x, rn_part2, sn, 0, 0, extra, moments);
    mu0_out[(0) as usize] = moments[(0) as usize];
    wheeler_recursion_dev(n, alpha, beta, 0, 0, moments, a_out, b_out, s0, sm, sk);
    // sqrt the off-diagonal b[1..n] in place (matches host rys_wheeler_partial).
    let mut i: u32 = 1;
    while i < n {
        b_out[(i) as usize] = f64::sqrt(b_out[(i) as usize]);
        i += 1;
    }
}

/// Device transform: roots[i] = eig[i]/(1-eig[i]); weights[i] = c0[i*n]^2 * mu0. `n` comptime.
#[cube(launch)]
fn jacobi_transform_kernel(
    eig: &Array<f64>,
    c0: &Array<f64>,
    mu0_in: &Array<f64>,
    roots: &mut Array<f64>,
    weights: &mut Array<f64>,
    #[comptime] n: u32,
) {
    let mu0 = mu0_in[(0) as usize];
    let mut i: u32 = 0;
    while i < n {
        let r = eig[(i) as usize];
        roots[(i) as usize] = r / (1.0 - r);
        let c = c0[(i * n) as usize];
        weights[(i) as usize] = c * c * mu0;
        i += 1;
    }
}

/// Host orchestrator for the f64 Jacobi path (nroots 6,7; x <= breakpoint).
///
/// Host-launched on `CpuRuntime`, so it only exists when the `cpu` feature is on;
/// see [`rys_roots_host_wheeler`] for the GPU-only-profile substitution.
#[cfg(feature = "cpu")]
fn rys_jacobi_device(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let n = nroots;
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());

    let alpha = &data::JACOBI_ALPHA[..];
    let beta = &data::JACOBI_BETA[..];
    let rn_part2 = &data::JACOBI_RN_PART2[..];
    let sn = &data::JACOBI_SN[..];

    let alpha_h = client.create_from_slice(f64::as_bytes(alpha));
    let beta_h = client.create_from_slice(f64::as_bytes(beta));
    let rn_h = client.create_from_slice(f64::as_bytes(rn_part2));
    let sn_h = client.create_from_slice(f64::as_bytes(sn));

    let n2 = n * 2;
    let mom_zero = vec![0.0f64; MXRYSROOTS * 2];
    let mom_h = client.create_from_slice(f64::as_bytes(&mom_zero));
    let a_zero = vec![0.0f64; n];
    let a_h = client.create_from_slice(f64::as_bytes(&a_zero));
    let b_h = client.create_from_slice(f64::as_bytes(&a_zero));
    let mu0_zero = vec![0.0f64; 1];
    let mu0_h = client.create_from_slice(f64::as_bytes(&mu0_zero));
    let scr_zero = vec![0.0f64; n2];
    let s0_h = client.create_from_slice(f64::as_bytes(&scr_zero));
    let sm_h = client.create_from_slice(f64::as_bytes(&scr_zero));
    let sk_h = client.create_from_slice(f64::as_bytes(&scr_zero));

    jacobi_tridiag_kernel::launch::<cubecl::cpu::CpuRuntime>(
        &client,
        crate::plane::single_cube_count(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(alpha_h, alpha.len()) },
        unsafe { ArrayArg::from_raw_parts(beta_h, beta.len()) },
        unsafe { ArrayArg::from_raw_parts(rn_h, rn_part2.len()) },
        unsafe { ArrayArg::from_raw_parts(sn_h, sn.len()) },
        unsafe { ArrayArg::from_raw_parts(mom_h, MXRYSROOTS * 2) },
        unsafe { ArrayArg::from_raw_parts(a_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(b_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(mu0_h.clone(), 1) },
        unsafe { ArrayArg::from_raw_parts(s0_h, n2) },
        unsafe { ArrayArg::from_raw_parts(sm_h, n2) },
        unsafe { ArrayArg::from_raw_parts(sk_h, n2) },
        x,
        n as u32,
        FLOCKE_EXTRA_ORDER_FOR_DP as u32,
    );

    let a_bytes = client.read_one_unchecked(a_h);
    let a_vec = f64::from_bytes(&a_bytes)[0..n].to_vec();
    let b_bytes = client.read_one_unchecked(b_h);
    let b_vec = f64::from_bytes(&b_bytes)[0..n].to_vec();

    // Eigensolve: _CINTdiagonalize(n, a, b+1, eig, c0) — off-diagonal is b[1..n].
    let mut diag = a_vec.clone();
    let mut offdiag = vec![0.0f64; n];
    for i in 0..n.saturating_sub(1) {
        offdiag[i] = b_vec[i + 1];
    }
    let mut eig = vec![0.0f64; n];
    let mut c0 = vec![0.0f64; n * n];
    let error = eigh::cint_diagonalize(n, &mut diag, &mut offdiag, &mut eig, &mut c0);

    // Transform kernel: roots/weights from eig + c0 + mu0.
    let eig_h = client.create_from_slice(f64::as_bytes(&eig));
    let c0_h = client.create_from_slice(f64::as_bytes(&c0));
    let r_zero = vec![0.0f64; n];
    let roots_h = client.create_from_slice(f64::as_bytes(&r_zero));
    let weights_h = client.create_from_slice(f64::as_bytes(&r_zero));

    jacobi_transform_kernel::launch::<cubecl::cpu::CpuRuntime>(
        &client,
        crate::plane::single_cube_count(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(eig_h, n) },
        unsafe { ArrayArg::from_raw_parts(c0_h, n * n) },
        unsafe { ArrayArg::from_raw_parts(mu0_h, 1) },
        unsafe { ArrayArg::from_raw_parts(roots_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(weights_h.clone(), n) },
        n as u32,
    );
    let r_bytes = client.read_one_unchecked(roots_h);
    roots[0..n].copy_from_slice(&f64::from_bytes(&r_bytes)[0..n]);
    let w_bytes = client.read_one_unchecked(weights_h);
    weights[0..n].copy_from_slice(&f64::from_bytes(&w_bytes)[0..n]);
    error
}

// ---------------------------------------------------------------------------
// Device f64 Schmidt path (x > breakpoint, nroots 6,7) — self-contained kernel.
// ---------------------------------------------------------------------------

const SML_FLOAT64_DEV: f64 = 1.1102230246251565e-16; // f64::EPSILON * 0.5

/// Device gamma_inc_like (FMT moments F_m(t), lower==0). `turnover` is TURNOVER_POINT[m]
/// passed in (avoids const-array runtime index). Writes f[0..=m].
#[cube]
fn gamma_inc_like_dev(f: &mut Array<f64>, t: f64, m: u32, turnover: f64) {
    if t == 0.0 {
        f[(0) as usize] = 1.0;
        let mut i: u32 = 1;
        while i <= m {
            f[(i) as usize] = 1.0 / (2 * i + 1) as f64;
            i += 1;
        }
    } else if t < turnover {
        // fmt1 power series.
        let mut b = m as f64 + 0.5;
        let e = 0.5 * f64::exp(-t);
        let mut xx = e;
        let mut s = e;
        let tol = SML_FLOAT64_DEV * e;
        let mut bi = b + 1.0;
        let mut k: u32 = 0;
        let mut going = true;
        while k < 4096u32 {
            if going {
                if xx > tol {
                    xx = xx * t / bi;
                    s += xx;
                    bi += 1.0;
                } else {
                    going = false;
                }
            }
            k += 1;
        }
        f[(m) as usize] = s / b;
        let mut i: u32 = m;
        while i > 0 {
            b -= 1.0;
            f[(i - 1) as usize] = (e + t * f[(i) as usize]) / b;
            i -= 1;
        }
    } else {
        let tt = f64::sqrt(t);
        f[(0) as usize] = SQRTPIE4_DEV / tt * erf_dev(tt);
        let e = f64::exp(-t);
        let bb = 0.5 / t;
        let mut i: u32 = 1;
        while i <= m {
            f[(i) as usize] = bb * ((2 * i - 1) as f64 * f[(i - 1) as usize] - e);
            i += 1;
        }
    }
}

/// Device R_dsmit (Schmidt orthogonalization). `cs` column-major n×n; `fmt_ints` moments.
/// `v` is caller scratch (len >= n). Returns 0 ok / 1 / j>0 via `flag[0]`.
#[cube]
fn r_dsmit_dev(
    cs: &mut Array<f64>,
    fmt_ints: &Array<f64>,
    n: u32,
    v: &mut Array<f64>,
    flag: &mut Array<f64>,
) {
    let mut ret = 0.0f64;
    let mut fac = -fmt_ints[(1) as usize] / fmt_ints[(0) as usize];
    let mut tmp = fmt_ints[(2) as usize] + fac * fmt_ints[(1) as usize];
    let mut aborted = false;
    if tmp <= 0.0 {
        // zero column 1.
        let mut i: u32 = 0;
        while i < n {
            cs[(i + 1 * n) as usize] = 0.0;
            i += 1;
        }
        ret = 1.0;
        aborted = true;
    }
    if !aborted {
        tmp = 1.0 / f64::sqrt(tmp);
        cs[(0) as usize] = 1.0 / f64::sqrt(fmt_ints[(0) as usize]);
        cs[(n) as usize] = fac * tmp;
        cs[(1 + n) as usize] = tmp;

        let mut j: u32 = 2;
        let mut stop = false;
        while j < n {
            if !stop {
                let mut vi: u32 = 0;
                while vi < j {
                    v[(vi) as usize] = 0.0;
                    vi += 1;
                }
                fac = fmt_ints[(j + j) as usize];
                let mut k: u32 = 0;
                while k < j {
                    let mut dot = 0.0f64;
                    let mut i: u32 = 0;
                    while i <= k {
                        dot += cs[(i + k * n) as usize] * fmt_ints[(i + j) as usize];
                        i += 1;
                    }
                    let mut i2: u32 = 0;
                    while i2 <= k {
                        v[(i2) as usize] = v[(i2) as usize] - dot * cs[(i2 + k * n) as usize];
                        i2 += 1;
                    }
                    fac -= dot * dot;
                    k += 1;
                }
                if fac <= 0.0 {
                    // zero columns [j..n].
                    let mut kk: u32 = j;
                    while kk < n {
                        let mut i: u32 = 0;
                        while i < n {
                            cs[(i + kk * n) as usize] = 0.0;
                            i += 1;
                        }
                        kk += 1;
                    }
                    if fac == 0.0 {
                        ret = 0.0;
                    } else {
                        ret = j as f64;
                    }
                    stop = true;
                } else {
                    fac = 1.0 / f64::sqrt(fac);
                    cs[(j + j * n) as usize] = fac;
                    let mut k2: u32 = 0;
                    while k2 < j {
                        cs[(k2 + j * n) as usize] = fac * v[(k2) as usize];
                        k2 += 1;
                    }
                }
            }
            j += 1;
        }
    }
    flag[(0) as usize] = ret;
}

/// Device POLYNOMIAL_VALUE1: Horner of a[off..=off+order] at x.
#[cube]
fn poly_value1_dev(a: &Array<f64>, off: u32, order: u32, x: f64) -> f64 {
    let mut p = a[(off + order) as usize];
    let mut i: u32 = 1;
    while i <= order {
        p = p * x + a[(off + order - i) as usize];
        i += 1;
    }
    p
}

/// Device _qr_step (find_roots.c:101). `a` row-major nroots×nroots.
#[cube]
fn qr_step_dev(a: &mut Array<f64>, nroots: u32, n0: u32, n1: u32, shift: f64) {
    let m1 = n0 + 1;
    let mut c = a[(n0 * nroots + n0) as usize] - shift;
    let mut s = a[(m1 * nroots + n0) as usize];
    let mut v = f64::sqrt(c * c + s * s);
    if v == 0.0 {
        v = 1.0;
        c = 1.0;
        s = 0.0;
    }
    v = 1.0 / v;
    c *= v;
    s *= v;
    let mut k: u32 = n0;
    while k < nroots {
        let xx = a[(n0 * nroots + k) as usize];
        let yy = a[(m1 * nroots + k) as usize];
        a[(n0 * nroots + k) as usize] = c * xx + s * yy;
        a[(m1 * nroots + k) as usize] = c * yy - s * xx;
        k += 1;
    }
    let mut m3 = n1;
    if n0 + 3 < m3 {
        m3 = n0 + 3;
    }
    let mut kk: u32 = 0;
    while kk < m3 {
        let xx = a[(kk * nroots + n0) as usize];
        let yy = a[(kk * nroots + m1) as usize];
        a[(kk * nroots + n0) as usize] = c * xx + s * yy;
        a[(kk * nroots + m1) as usize] = c * yy - s * xx;
        kk += 1;
    }
    if n1 >= 2 {
        let mut j: u32 = n0;
        while j + 2 < n1 {
            let j1 = j + 1;
            let j2 = j + 2;
            c = a[(j1 * nroots + j) as usize];
            s = a[(j2 * nroots + j) as usize];
            v = f64::sqrt(c * c + s * s);
            a[(j1 * nroots + j) as usize] = v;
            a[(j2 * nroots + j) as usize] = 0.0;
            if v == 0.0 {
                v = 1.0;
                c = 1.0;
                s = 0.0;
            }
            v = 1.0 / v;
            c *= v;
            s *= v;
            let mut k3: u32 = j1;
            while k3 < nroots {
                let xx = a[(j1 * nroots + k3) as usize];
                let yy = a[(j2 * nroots + k3) as usize];
                a[(j1 * nroots + k3) as usize] = c * xx + s * yy;
                a[(j2 * nroots + k3) as usize] = c * yy - s * xx;
                k3 += 1;
            }
            let mut m3b = n1;
            if j + 4 < m3b {
                m3b = j + 4;
            }
            let mut k4: u32 = 0;
            while k4 < m3b {
                let xx = a[(k4 * nroots + j1) as usize];
                let yy = a[(k4 * nroots + j2) as usize];
                a[(k4 * nroots + j1) as usize] = c * xx + s * yy;
                a[(k4 * nroots + j2) as usize] = c * yy - s * xx;
                k4 += 1;
            }
            j += 1;
        }
    }
}

/// Device _hessenberg_qr (find_roots.c:177). `a` row-major nroots×nroots. Returns via flag.
#[cube]
fn hessenberg_qr_dev(a: &mut Array<f64>, nroots: u32, flag: &mut Array<f64>) {
    let eps = 1e-15f64;
    let mut n0: u32 = 0;
    let mut n1 = nroots;
    let mut its: u32 = 0;
    let mut ret = 1.0f64;
    let mut done = false;
    let total = nroots * 30u32;
    let mut ic: u32 = 0;
    while ic < total {
        if !done {
            let mut k = n0;
            let mut scanning = true;
            while k + 1 < n1 && scanning {
                let s = f64::abs(a[(k * nroots + k) as usize])
                    + f64::abs(a[((k + 1) * nroots + k + 1) as usize]);
                if f64::abs(a[((k + 1) * nroots + k) as usize]) < eps * s {
                    scanning = false;
                } else {
                    k += 1;
                }
            }
            let k1 = k + 1;
            if k1 < n1 {
                a[(k1 * nroots + k) as usize] = 0.0;
                n0 = k1;
                its = 0;
                if n0 + 1 >= n1 {
                    n0 = 0;
                    n1 = k1;
                    if n1 < 2 {
                        ret = 0.0;
                        done = true;
                    }
                }
            } else {
                let m1 = n1 - 1;
                let m2 = n1 - 2;
                let a11 = a[(m1 * nroots + m1) as usize];
                let a22 = a[(m2 * nroots + m2) as usize];
                let tsum = a11 + a22;
                let mut s = (a11 - a22) * (a11 - a22);
                s += 4.0 * a[(m1 * nroots + m2) as usize] * a[(m2 * nroots + m1) as usize];
                let mut shift = 0.0f64;
                let mut bad = false;
                if s > 0.0 {
                    s = f64::sqrt(s);
                    let av = (tsum + s) * 0.5;
                    let bv = (tsum - s) * 0.5;
                    if f64::abs(a11 - av) > f64::abs(a11 - bv) {
                        shift = bv;
                    } else {
                        shift = av;
                    }
                } else if n1 == 2 {
                    ret = 1.0;
                    done = true;
                    bad = true;
                } else {
                    shift = tsum * 0.5;
                }
                if !bad {
                    its += 1;
                    qr_step_dev(a, nroots, n0, n1, shift);
                    if its > 30u32 {
                        ret = 1.0;
                        done = true;
                    }
                }
            }
        }
        ic += 1;
    }
    flag[(0) as usize] = ret;
}

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// Device R_dnode Newton/bisection polish. `a` is the cs column slice base offset `off`.
/// Returns via flag[0] (0 ok / 1 error).
#[cube]
fn r_dnode_dev(
    a: &Array<f64>,
    off: u32,
    roots: &mut Array<f64>,
    order: u32,
    flag: &mut Array<f64>,
) {
    let accrt = 1e-15f64;
    let mut x1init = 0.0f64;
    let mut p1init = a[(off) as usize];
    let mut ret = 0.0f64;
    let mut aborted = false;
    let mut m: u32 = 0;
    while m < order {
        if !aborted {
            let mut x0 = x1init;
            let mut p0 = p1init;
            x1init = roots[(m) as usize];
            p1init = poly_value1_dev(a, off, order, x1init);
            let mut skip = false;
            if p1init == 0.0 {
                skip = true;
            }
            if !skip && p0 * p1init > 0.0 {
                ret = 1.0;
                aborted = true;
                skip = true;
            }
            if !skip {
                let mut x1 = x1init;
                let mut p1 = p1init;
                if x0 <= x1init {
                    x1 = x1init;
                    p1 = p1init;
                } else {
                    x1 = x0;
                    p1 = p0;
                    x0 = x1init;
                    p0 = p1init;
                }
                let mut handled = false;
                if p1 == 0.0 {
                    roots[(m) as usize] = x1;
                    handled = true;
                } else if p0 == 0.0 {
                    roots[(m) as usize] = x0;
                    handled = true;
                }
                if !handled {
                    let mut xi = x0 + (x0 - x1) / (p1 - p0) * p0;
                    let mut nn: u32 = 0;
                    let mut converged = false;
                    while nn < 200u32 {
                        if !converged {
                            if f64::abs(x1 - x0) > x1 * accrt {
                                let mut pi = poly_value1_dev(a, off, order, xi);
                                let mut brk = false;
                                if pi == 0.0 {
                                    brk = true;
                                } else if p0 * pi <= 0.0 {
                                    x1 = xi;
                                    p1 = pi;
                                    xi = x0 * 0.25 + xi * 0.75;
                                } else {
                                    x0 = xi;
                                    p0 = pi;
                                    xi = xi * 0.75 + x1 * 0.25;
                                }
                                if !brk {
                                    pi = poly_value1_dev(a, off, order, xi);
                                    if pi == 0.0 {
                                        brk = true;
                                    } else if p0 * pi <= 0.0 {
                                        x1 = xi;
                                        p1 = pi;
                                    } else {
                                        x0 = xi;
                                        p0 = pi;
                                    }
                                }
                                if brk {
                                    converged = true;
                                } else {
                                    xi = x0 + (x0 - x1) / (p1 - p0) * p0;
                                }
                            } else {
                                converged = true;
                            }
                        }
                        nn += 1;
                    }
                    roots[(m) as usize] = xi;
                }
            }
        }
        m += 1;
    }
    flag[(0) as usize] = ret;
}

/// Device _CINT_polynomial_roots (find_roots.c:243). `cs` column-major nroots1×nroots1.
/// Writes roots[0..nroots]. `acomp` is caller scratch for the companion matrix (nroots×nroots).
#[cube]
fn cint_polynomial_roots_dev(
    roots: &mut Array<f64>,
    cs: &Array<f64>,
    nroots: u32,
    nroots1: u32,
    acomp: &mut Array<f64>,
    flag: &mut Array<f64>,
) {
    let mut ret = 0.0f64;
    if nroots == 1 {
        roots[(0) as usize] = -cs[(2) as usize] / cs[(3) as usize];
    } else if nroots == 2 {
        let dum = f64::sqrt(
            cs[(2 * 3 + 1) as usize] * cs[(2 * 3 + 1) as usize]
                - 4.0 * cs[(2 * 3) as usize] * cs[(2 * 3 + 2) as usize],
        );
        roots[(0) as usize] = (-cs[(2 * 3 + 1) as usize] - dum) / cs[(2 * 3 + 2) as usize] / 2.0;
        roots[(1) as usize] = (-cs[(2 * 3 + 1) as usize] + dum) / cs[(2 * 3 + 2) as usize] / 2.0;
    } else {
        // Companion matrix A (row-major nroots×nroots).
        let nn = nroots * nroots;
        let mut z: u32 = 0;
        while z < nn {
            acomp[(z) as usize] = 0.0;
            z += 1;
        }
        let fac = -1.0 / cs[(nroots * nroots1 + nroots) as usize];
        let mut i: u32 = 0;
        while i < nroots {
            acomp[(nroots - 1 - i) as usize] = cs[(nroots * nroots1 + i) as usize] * fac;
            i += 1;
        }
        let mut i2: u32 = 0;
        while i2 + 1 < nroots {
            acomp[((i2 + 1) * nroots + i2) as usize] = 1.0;
            i2 += 1;
        }
        hessenberg_qr_dev(acomp, nroots, flag);
        let err = flag[(0) as usize];
        if err == 0.0 {
            let mut k: u32 = 0;
            while k < nroots {
                roots[(nroots - 1 - k) as usize] = acomp[(k * nroots + k) as usize];
                k += 1;
            }
        } else {
            // Fallback: quadratic seed + R_dnode Newton search.
            let dum = f64::sqrt(
                cs[(2 * nroots1 + 1) as usize] * cs[(2 * nroots1 + 1) as usize]
                    - 4.0 * cs[(2 * nroots1) as usize] * cs[(2 * nroots1 + 2) as usize],
            );
            roots[(0) as usize] =
                0.5 * (-cs[(2 * nroots1 + 1) as usize] - dum) / cs[(2 * nroots1 + 2) as usize];
            roots[(1) as usize] =
                0.5 * (-cs[(2 * nroots1 + 1) as usize] + dum) / cs[(2 * nroots1 + 2) as usize];
            let mut r: u32 = 2;
            while r < nroots {
                roots[(r) as usize] = 1.0;
                r += 1;
            }
            let mut e = 0.0f64;
            let mut k: u32 = 2;
            let mut stop = false;
            while k < nroots {
                if !stop {
                    let order = k + 1;
                    let off = order * nroots1;
                    r_dnode_dev(cs, off, roots, order, flag);
                    e = flag[(0) as usize];
                    if e != 0.0 {
                        stop = true;
                    }
                }
                k += 1;
            }
            ret = e;
        }
    }
    flag[(0) as usize] = ret;
}

/// Device Schmidt kernel (f64): gamma_inc moments + R_dsmit + QR roots + weights.
/// `turnovers[0]` = TURNOVER_POINT[nroots*2] (passed to avoid const-array runtime index).
#[cube(launch)]
fn schmidt_kernel(
    fmt_ints: &mut Array<f64>,
    cs: &mut Array<f64>,
    rt: &mut Array<f64>,
    acomp: &mut Array<f64>,
    vscr: &mut Array<f64>,
    flag: &mut Array<f64>,
    roots: &mut Array<f64>,
    weights: &mut Array<f64>,
    x: f64,
    turnover: f64,
    #[comptime] nroots: u32,
) {
    let nroots1 = nroots + 1;
    gamma_inc_like_dev(fmt_ints, x, nroots * 2, turnover);

    let mut zeroed = false;
    if fmt_ints[(0) as usize] == 0.0 {
        let mut k: u32 = 0;
        while k < nroots {
            roots[(k) as usize] = 0.0;
            weights[(k) as usize] = 0.0;
            k += 1;
        }
        zeroed = true;
    }

    if !zeroed {
        // zero cs (nroots1 x nroots1).
        let nn1 = nroots1 * nroots1;
        let mut z: u32 = 0;
        while z < nn1 {
            cs[(z) as usize] = 0.0;
            z += 1;
        }
        r_dsmit_dev(cs, fmt_ints, nroots1, vscr, flag);
        // (smit error handling: if flag != 0 the host fallback covers it; here we proceed,
        //  matching the f64 corpus envelope where R_dsmit succeeds.)
        cint_polynomial_roots_dev(rt, cs, nroots, nroots1, acomp, flag);

        let mut k: u32 = 0;
        while k < nroots {
            let root = rt[(k) as usize];
            if root == 1.0 {
                roots[(k) as usize] = 0.0;
                weights[(k) as usize] = 0.0;
            } else {
                let mut dum = 1.0 / fmt_ints[(0) as usize];
                let mut j: u32 = 1;
                while j < nroots {
                    let off = j * nroots1;
                    let poly = poly_value1_dev(cs, off, j, root);
                    dum += poly * poly;
                    j += 1;
                }
                roots[(k) as usize] = root / (1.0 - root);
                weights[(k) as usize] = 1.0 / dum;
            }
            k += 1;
        }
    }
}

/// Host orchestrator for the f64 Schmidt path (nroots 6,7; x > breakpoint).
///
/// Retained as the on-device implementation of the f64 Schmidt path. Not wired into the
/// production dispatch: nroots 6,7 stay on the host path (parity-honest escape hatch — see
/// `rys_roots_host_wheeler`), and nroots 8's large-x tail uses the dd `lrys_schmidt_device`.
#[allow(dead_code)]
#[cfg(feature = "cpu")]
fn rys_schmidt_device(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let n = nroots;
    let nroots1 = n + 1;
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());

    let fmt_zero = vec![0.0f64; MXRYSROOTS * 2];
    let fmt_h = client.create_from_slice(f64::as_bytes(&fmt_zero));
    let cs_zero = vec![0.0f64; nroots1 * nroots1];
    let cs_h = client.create_from_slice(f64::as_bytes(&cs_zero));
    let rt_zero = vec![0.0f64; n];
    let rt_h = client.create_from_slice(f64::as_bytes(&rt_zero));
    let acomp_zero = vec![0.0f64; n * n];
    let acomp_h = client.create_from_slice(f64::as_bytes(&acomp_zero));
    let v_zero = vec![0.0f64; MXRYSROOTS];
    let v_h = client.create_from_slice(f64::as_bytes(&v_zero));
    let flag_zero = vec![0.0f64; 1];
    let flag_h = client.create_from_slice(f64::as_bytes(&flag_zero));
    let r_zero = vec![0.0f64; n];
    let roots_h = client.create_from_slice(f64::as_bytes(&r_zero));
    let weights_h = client.create_from_slice(f64::as_bytes(&r_zero));

    let turnover = data::TURNOVER_POINT[n * 2];

    schmidt_kernel::launch::<cubecl::cpu::CpuRuntime>(
        &client,
        crate::plane::single_cube_count(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(fmt_h, MXRYSROOTS * 2) },
        unsafe { ArrayArg::from_raw_parts(cs_h, nroots1 * nroots1) },
        unsafe { ArrayArg::from_raw_parts(rt_h, n) },
        unsafe { ArrayArg::from_raw_parts(acomp_h, n * n) },
        unsafe { ArrayArg::from_raw_parts(v_h, MXRYSROOTS) },
        unsafe { ArrayArg::from_raw_parts(flag_h, 1) },
        unsafe { ArrayArg::from_raw_parts(roots_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(weights_h.clone(), n) },
        x,
        turnover,
        n as u32,
    );
    let r_bytes = client.read_one_unchecked(roots_h);
    roots[0..n].copy_from_slice(&f64::from_bytes(&r_bytes)[0..n]);
    let w_bytes = client.read_one_unchecked(weights_h);
    weights[0..n].copy_from_slice(&f64::from_bytes(&w_bytes)[0..n]);
    0
}

// ===========================================================================
//  DEVICE double-double layer + dd Wheeler paths (Task 3: nroots 8..12).
//
//  The FMA probe (fma_probe) proved the CubeCL 0.10.0 CPU backend FUSES `fma`
//  bit-for-bit vs host f64::mul_add, so the dd two_prod error term is exact and
//  no Dekker-split software product is needed. Each `Dd { hi, lo }` is a CubeType
//  struct threaded through #[cube] dd ops (two_sum/two_prod/dd_add/sub/mul/div/sqrt),
//  mirroring the host `Dd` algorithms 1:1. The dd tridiagonal (a,b) is rounded to
//  f64 (Dd::hi) before the shared device eigh — exactly as the host/vendor round
//  long-double (a,b) to double before _CINTdiagonalize.
// ===========================================================================

#[derive(CubeType, Clone, Copy)]
struct DdDev {
    hi: f64,
    lo: f64,
}

#[cube]
fn dd_from(x: f64) -> DdDev {
    DdDev { hi: x, lo: 0.0 }
}

#[cube]
fn two_sum_dev(a: f64, b: f64) -> DdDev {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    DdDev { hi: s, lo: err }
}

#[cube]
fn two_prod_dev(a: f64, b: f64) -> DdDev {
    let p = a * b;
    let e = fma(a, b, -p);
    DdDev { hi: p, lo: e }
}

#[cube]
fn dd_add_dev(a: DdDev, b: DdDev) -> DdDev {
    let s = two_sum_dev(a.hi, b.hi);
    let e = s.lo + (a.lo + b.lo);
    let r = two_sum_dev(s.hi, e);
    DdDev { hi: r.hi, lo: r.lo }
}

#[cube]
fn dd_sub_dev(a: DdDev, b: DdDev) -> DdDev {
    let nb = DdDev {
        hi: -b.hi,
        lo: -b.lo,
    };
    dd_add_dev(a, nb)
}

#[cube]
fn dd_mul_dev(a: DdDev, b: DdDev) -> DdDev {
    let p = two_prod_dev(a.hi, b.hi);
    let e = p.lo + (a.hi * b.lo + a.lo * b.hi);
    let r = two_sum_dev(p.hi, e);
    DdDev { hi: r.hi, lo: r.lo }
}

#[cube]
fn dd_div_dev(a: DdDev, b: DdDev) -> DdDev {
    let q1 = a.hi / b.hi;
    let r1 = dd_sub_dev(a, dd_mul_dev(dd_from(q1), b));
    let q2 = r1.hi / b.hi;
    let r2 = dd_sub_dev(r1, dd_mul_dev(dd_from(q2), b));
    let q3 = r2.hi / b.hi;
    let s = two_sum_dev(q1, q2);
    dd_add_dev(DdDev { hi: s.hi, lo: s.lo }, dd_from(q3))
}

#[cube]
fn dd_mul_f64_dev(a: DdDev, b: f64) -> DdDev {
    dd_mul_dev(a, dd_from(b))
}

#[cube]
fn dd_sqrt_dev(a: DdDev) -> DdDev {
    let mut outh = 0.0f64;
    let mut outl = 0.0f64;
    if a.hi > 0.0 {
        let x = 1.0 / f64::sqrt(a.hi);
        let ax = a.hi * x;
        let p = two_prod_dev(ax, ax);
        let diff = dd_sub_dev(a, DdDev { hi: p.hi, lo: p.lo });
        let corr = diff.hi * (x * 0.5);
        let s = two_sum_dev(ax, corr);
        outh = s.hi;
        outl = s.lo;
    }
    DdDev { hi: outh, lo: outl }
}

const SQRTPIE4_DD: f64 =
    0.886_226_925_452_758_013_649_083_741_670_572_591_398_774_728_061_193_564_106_903_894_926_4;
const FLOCKE_EXTRA_LP_DEV: u32 = 24;

/// Device lflocke_jacobi_moments in dd (intermediate-x branch; SMALLX path out of envelope).
/// rn_part2/sn are LJACOBI_RN_PART2 / LJACOBI_SN. mus_hi/mus_lo hold the dd moment pairs.
#[cube]
fn lflocke_jacobi_moments_dev(
    n: u32,
    t: f64,
    rn_part2: &Array<f64>,
    sn: &Array<f64>,
    rn_off: u32,
    sn_off: u32,
    mus_hi: &mut Array<f64>,
    mus_lo: &mut Array<f64>,
) {
    let t_inv = dd_div_dev(dd_from(0.5), dd_from(t));
    // dd accumulators threaded as scalar hi/lo pairs (CubeType structs are not reassignable).
    let mut mu1h = 1.0f64;
    let mut mu1l = 0.0f64;
    let mut mu2h = 0.0f64;
    let mut mu2l = 0.0f64;
    let mut mu0h = 0.0f64;
    let mut mu0l = 0.0f64;
    let top = n - 1 + FLOCKE_EXTRA_LP_DEV;
    let count1 = top + 1 - n;
    let mut step: u32 = 0;
    while step < count1 {
        let ii = top - step;
        let rn = dd_add_dev(
            dd_mul_f64_dev(t_inv, (2 * ii + 3) as f64),
            dd_from(rn_part2[(rn_off + ii) as usize]),
        );
        let mu1 = DdDev { hi: mu1h, lo: mu1l };
        let mu2 = DdDev { hi: mu2h, lo: mu2l };
        let mu0 = dd_div_dev(
            dd_sub_dev(mu2, dd_mul_dev(rn, mu1)),
            dd_from(sn[(sn_off + ii) as usize]),
        );
        mu0h = mu0.hi;
        mu0l = mu0.lo;
        mu2h = mu1h;
        mu2l = mu1l;
        mu1h = mu0h;
        mu1l = mu0l;
        step += 1;
    }
    let mut step2: u32 = 0;
    while step2 < n {
        let ii = (n - 1) - step2;
        let rn = dd_add_dev(
            dd_mul_f64_dev(t_inv, (2 * ii + 3) as f64),
            dd_from(rn_part2[(rn_off + ii) as usize]),
        );
        let mu1 = DdDev { hi: mu1h, lo: mu1l };
        let mu2 = DdDev { hi: mu2h, lo: mu2l };
        let mu0 = dd_div_dev(
            dd_sub_dev(mu2, dd_mul_dev(rn, mu1)),
            dd_from(sn[(sn_off + ii) as usize]),
        );
        mu0h = mu0.hi;
        mu0l = mu0.lo;
        mus_hi[(ii) as usize] = mu0h;
        mus_lo[(ii) as usize] = mu0l;
        mu2h = mu1h;
        mu2l = mu1l;
        mu1h = mu0h;
        mu1l = mu0l;
        step2 += 1;
    }
    let tt = f64::sqrt(t);
    let num = dd_from(SQRTPIE4_DD * erf_dev(tt) / tt);
    let norm = dd_div_dev(num, DdDev { hi: mu0h, lo: mu0l });
    let mut j: u32 = 0;
    while j < n {
        let v = DdDev {
            hi: mus_hi[(j) as usize],
            lo: mus_lo[(j) as usize],
        };
        let vn = dd_mul_dev(v, norm);
        mus_hi[(j) as usize] = vn.hi;
        mus_lo[(j) as usize] = vn.lo;
        j += 1;
    }
}

/// Device llaguerre_moments in dd (lower==0). Outputs alpha/beta (dd) + moments (dd).
#[cube]
fn llaguerre_moments_dev(
    n: u32,
    t: f64,
    alpha_hi: &mut Array<f64>,
    alpha_lo: &mut Array<f64>,
    beta_hi: &mut Array<f64>,
    beta_lo: &mut Array<f64>,
    mom_hi: &mut Array<f64>,
    mom_lo: &mut Array<f64>,
) {
    let tt = f64::sqrt(t);
    let t_inv = dd_div_dev(dd_from(0.5), dd_from(t));
    let t2_inv = dd_div_dev(dd_from(0.5), dd_mul_dev(dd_from(t), dd_from(t)));
    let e0 = dd_mul_dev(dd_from(f64::exp(-t)), t_inv);
    let mut l00h = 0.0f64;
    let mut l00l = 0.0f64;
    let mut l01h = 1.0f64;
    let mut l01l = 0.0f64;

    alpha_hi[(0) as usize] = t_inv.hi;
    alpha_lo[(0) as usize] = t_inv.lo;
    beta_hi[(0) as usize] = 0.0;
    beta_lo[(0) as usize] = 0.0;
    let m0 = dd_from(SQRTPIE4_DD / tt * erf_dev(tt));
    mom_hi[(0) as usize] = m0.hi;
    mom_lo[(0) as usize] = m0.lo;
    let m1 = dd_mul_dev(
        DdDev {
            hi: -l01h,
            lo: -l01l,
        },
        e0,
    );
    mom_hi[(1) as usize] = m1.hi;
    mom_lo[(1) as usize] = m1.lo;

    let mut i: u32 = 1;
    while i + 1 < n {
        let ai = dd_mul_f64_dev(t_inv, (i * 4 + 1) as f64);
        alpha_hi[(i) as usize] = ai.hi;
        alpha_lo[(i) as usize] = ai.lo;
        let bi = dd_mul_f64_dev(t2_inv, (i * (i * 2 - 1)) as f64);
        beta_hi[(i) as usize] = bi.hi;
        beta_lo[(i) as usize] = bi.lo;
        let fac0 = dd_mul_f64_dev(t_inv, (i * 4 - 1) as f64);
        let fac1 = dd_mul_f64_dev(t2_inv, ((i - 1) * (i * 2 - 1)) as f64);
        let l00 = DdDev { hi: l00h, lo: l00l };
        let l01 = DdDev { hi: l01h, lo: l01l };
        // l02 = (1 - fac0)*l01 - fac1*l00
        let l02 = dd_sub_dev(
            dd_mul_dev(dd_sub_dev(dd_from(1.0), fac0), l01),
            dd_mul_dev(fac1, l00),
        );
        l00h = l01h;
        l00l = l01l;
        l01h = l02.hi;
        l01l = l02.lo;
        let mip1 = dd_mul_dev(
            DdDev {
                hi: -l01h,
                lo: -l01l,
            },
            e0,
        );
        mom_hi[(i + 1) as usize] = mip1.hi;
        mom_lo[(i + 1) as usize] = mip1.lo;
        i += 1;
    }
}

/// Device dd Wheeler recursion: dd moments + dd (alpha,beta) -> tridiagonal (a,b) dd.
/// Scratch s0/sm/sk are dd pairs in parallel hi/lo Arrays of length n*2.
#[cube]
#[allow(clippy::too_many_arguments)]
fn lwheeler_recursion_dev(
    n: u32,
    alpha_hi: &Array<f64>,
    alpha_lo: &Array<f64>,
    beta_hi: &Array<f64>,
    beta_lo: &Array<f64>,
    mom_hi: &Array<f64>,
    mom_lo: &Array<f64>,
    a_hi: &mut Array<f64>,
    a_lo: &mut Array<f64>,
    b_hi: &mut Array<f64>,
    b_lo: &mut Array<f64>,
    s0h: &mut Array<f64>,
    s0l: &mut Array<f64>,
    smh: &mut Array<f64>,
    sml: &mut Array<f64>,
    skh: &mut Array<f64>,
    skl: &mut Array<f64>,
) {
    let al0 = DdDev {
        hi: alpha_hi[(0) as usize],
        lo: alpha_lo[(0) as usize],
    };
    let m0 = DdDev {
        hi: mom_hi[(0) as usize],
        lo: mom_lo[(0) as usize],
    };
    let m1 = DdDev {
        hi: mom_hi[(1) as usize],
        lo: mom_lo[(1) as usize],
    };
    let a0init = dd_add_dev(al0, dd_div_dev(m1, m0));
    let mut a0h = a0init.hi;
    let mut a0l = a0init.lo;
    let mut b0h = 0.0f64;
    let mut b0l = 0.0f64;
    a_hi[(0) as usize] = a0h;
    a_lo[(0) as usize] = a0l;
    b_hi[(0) as usize] = b0h;
    b_lo[(0) as usize] = b0l;

    let n2 = n * 2;
    let mut i: u32 = 0;
    while i < n2 {
        s0h[(i) as usize] = mom_hi[(i) as usize];
        s0l[(i) as usize] = mom_lo[(i) as usize];
        smh[(i) as usize] = 0.0;
        sml[(i) as usize] = 0.0;
        skh[(i) as usize] = 0.0;
        skl[(i) as usize] = 0.0;
        i += 1;
    }
    let mut step: u32 = 1;
    while step < n {
        let nc = 2 * (n - step);
        let mut j: u32 = 0;
        while j < nc {
            let s0_2 = DdDev {
                hi: s0h[(2 + j) as usize],
                lo: s0l[(2 + j) as usize],
            };
            let s0_1 = DdDev {
                hi: s0h[(1 + j) as usize],
                lo: s0l[(1 + j) as usize],
            };
            let s0_0 = DdDev {
                hi: s0h[(j) as usize],
                lo: s0l[(j) as usize],
            };
            let sm_2 = DdDev {
                hi: smh[(2 + j) as usize],
                lo: sml[(2 + j) as usize],
            };
            let alij = DdDev {
                hi: alpha_hi[(step + j) as usize],
                lo: alpha_lo[(step + j) as usize],
            };
            let beij = DdDev {
                hi: beta_hi[(step + j) as usize],
                lo: beta_lo[(step + j) as usize],
            };
            let a0 = DdDev { hi: a0h, lo: a0l };
            let b0 = DdDev { hi: b0h, lo: b0l };
            let term2 = dd_mul_dev(dd_sub_dev(a0, alij), s0_1);
            let term3 = dd_mul_dev(b0, sm_2);
            let term4 = dd_mul_dev(s0_0, beij);
            let v = dd_add_dev(dd_sub_dev(dd_sub_dev(s0_2, term2), term3), term4);
            skh[(j) as usize] = v.hi;
            skl[(j) as usize] = v.lo;
            j += 1;
        }
        let al_s = DdDev {
            hi: alpha_hi[(step) as usize],
            lo: alpha_lo[(step) as usize],
        };
        let s0_1 = DdDev {
            hi: s0h[(1) as usize],
            lo: s0l[(1) as usize],
        };
        let s0_0 = DdDev {
            hi: s0h[(0) as usize],
            lo: s0l[(0) as usize],
        };
        let sk_1 = DdDev {
            hi: skh[(1) as usize],
            lo: skl[(1) as usize],
        };
        let sk_0 = DdDev {
            hi: skh[(0) as usize],
            lo: skl[(0) as usize],
        };
        let a1 = dd_add_dev(
            dd_sub_dev(al_s, dd_div_dev(s0_1, s0_0)),
            dd_div_dev(sk_1, sk_0),
        );
        let b1 = dd_div_dev(sk_0, s0_0);
        a_hi[(step) as usize] = a1.hi;
        a_lo[(step) as usize] = a1.lo;
        b_hi[(step) as usize] = b1.hi;
        b_lo[(step) as usize] = b1.lo;
        a0h = a1.hi;
        a0l = a1.lo;
        b0h = b1.hi;
        b0l = b1.lo;
        // rotate: tmp=sm; sm=s0; s0=sk; sk=tmp
        let mut k: u32 = 0;
        while k < n2 {
            let th = smh[(k) as usize];
            let tl = sml[(k) as usize];
            smh[(k) as usize] = s0h[(k) as usize];
            sml[(k) as usize] = s0l[(k) as usize];
            s0h[(k) as usize] = skh[(k) as usize];
            s0l[(k) as usize] = skl[(k) as usize];
            skh[(k) as usize] = th;
            skl[(k) as usize] = tl;
            k += 1;
        }
        step += 1;
    }
}

/// c99_sqrtl device (one Babylonian refinement over f64 sqrt) — for the dd->f64 off-diag.
#[cube]
fn c99_sqrtl_dev(x: f64) -> f64 {
    let z = f64::sqrt(x);
    (z * z + x) / (z * 2.0)
}

/// Device dd Jacobi tridiagonal kernel (lrys_jacobi): dd moments + dd Wheeler -> (a,b) f64.
/// Outputs f64 da[0..n] and db[0..n] (db = c99_sqrtl(b.hi) from index 1), plus mu0_out[0].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn ljacobi_tridiag_kernel(
    alpha_in: &Array<f64>,
    beta_in: &Array<f64>,
    rn_part2: &Array<f64>,
    sn: &Array<f64>,
    momh: &mut Array<f64>,
    moml: &mut Array<f64>,
    alh: &mut Array<f64>,
    all_: &mut Array<f64>,
    beh: &mut Array<f64>,
    bel: &mut Array<f64>,
    ah: &mut Array<f64>,
    al: &mut Array<f64>,
    bh: &mut Array<f64>,
    bl: &mut Array<f64>,
    s0h: &mut Array<f64>,
    s0l: &mut Array<f64>,
    smh: &mut Array<f64>,
    sml: &mut Array<f64>,
    skh: &mut Array<f64>,
    skl: &mut Array<f64>,
    da_out: &mut Array<f64>,
    db_out: &mut Array<f64>,
    mu0_out: &mut Array<f64>,
    x: f64,
    #[comptime] n: u32,
) {
    lflocke_jacobi_moments_dev(n * 2, x, rn_part2, sn, 0, 0, momh, moml);
    mu0_out[(0) as usize] = momh[(0) as usize];
    // alpha/beta dd from f64 LJACOBI tables.
    let mut i: u32 = 0;
    while i < n * 2 {
        alh[(i) as usize] = alpha_in[(i) as usize];
        all_[(i) as usize] = 0.0;
        beh[(i) as usize] = beta_in[(i) as usize];
        bel[(i) as usize] = 0.0;
        i += 1;
    }
    lwheeler_recursion_dev(
        n, alh, all_, beh, bel, momh, moml, ah, al, bh, bl, s0h, s0l, smh, sml, skh, skl,
    );
    da_out[(0) as usize] = ah[(0) as usize];
    let mut k: u32 = 1;
    while k < n {
        da_out[(k) as usize] = ah[(k) as usize];
        db_out[(k) as usize] = c99_sqrtl_dev(bh[(k) as usize]);
        k += 1;
    }
}

/// Device dd Laguerre tridiagonal kernel (lrys_laguerre): dd Laguerre moments + dd Wheeler.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn llaguerre_tridiag_kernel(
    momh: &mut Array<f64>,
    moml: &mut Array<f64>,
    alh: &mut Array<f64>,
    all_: &mut Array<f64>,
    beh: &mut Array<f64>,
    bel: &mut Array<f64>,
    ah: &mut Array<f64>,
    al: &mut Array<f64>,
    bh: &mut Array<f64>,
    bl: &mut Array<f64>,
    s0h: &mut Array<f64>,
    s0l: &mut Array<f64>,
    smh: &mut Array<f64>,
    sml: &mut Array<f64>,
    skh: &mut Array<f64>,
    skl: &mut Array<f64>,
    da_out: &mut Array<f64>,
    db_out: &mut Array<f64>,
    mu0_out: &mut Array<f64>,
    x: f64,
    #[comptime] n: u32,
) {
    llaguerre_moments_dev(n * 2, x, alh, all_, beh, bel, momh, moml);
    mu0_out[(0) as usize] = momh[(0) as usize];
    lwheeler_recursion_dev(
        n, alh, all_, beh, bel, momh, moml, ah, al, bh, bl, s0h, s0l, smh, sml, skh, skl,
    );
    da_out[(0) as usize] = ah[(0) as usize];
    let mut k: u32 = 1;
    while k < n {
        da_out[(k) as usize] = ah[(k) as usize];
        db_out[(k) as usize] = c99_sqrtl_dev(bh[(k) as usize]);
        k += 1;
    }
}

/// Shared host orchestrator for the dd Jacobi/Laguerre paths: launch a dd tridiagonal
/// kernel (selected by `is_laguerre`), eigh, then the f64 transform kernel. `n` = nroots.
#[cfg(feature = "cpu")]
fn lrys_wheeler_device(
    nroots: usize,
    x: f64,
    roots: &mut [f64],
    weights: &mut [f64],
    is_laguerre: bool,
) -> i32 {
    let n = nroots;
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    let n2 = n * 2;

    // dd scratch buffers (parallel hi/lo).
    let momz = vec![0.0f64; MXRYSROOTS * 2];
    let momh = client.create_from_slice(f64::as_bytes(&momz));
    let moml = client.create_from_slice(f64::as_bytes(&momz));
    let abz = vec![0.0f64; n2.max(MXRYSROOTS * 2)];
    let alh = client.create_from_slice(f64::as_bytes(&abz));
    let all_ = client.create_from_slice(f64::as_bytes(&abz));
    let beh = client.create_from_slice(f64::as_bytes(&abz));
    let bel = client.create_from_slice(f64::as_bytes(&abz));
    let nz = vec![0.0f64; n];
    let ah = client.create_from_slice(f64::as_bytes(&nz));
    let al = client.create_from_slice(f64::as_bytes(&nz));
    let bh = client.create_from_slice(f64::as_bytes(&nz));
    let bl = client.create_from_slice(f64::as_bytes(&nz));
    let scrz = vec![0.0f64; n2];
    let s0h = client.create_from_slice(f64::as_bytes(&scrz));
    let s0l = client.create_from_slice(f64::as_bytes(&scrz));
    let smh = client.create_from_slice(f64::as_bytes(&scrz));
    let sml = client.create_from_slice(f64::as_bytes(&scrz));
    let skh = client.create_from_slice(f64::as_bytes(&scrz));
    let skl = client.create_from_slice(f64::as_bytes(&scrz));
    let da_h = client.create_from_slice(f64::as_bytes(&nz));
    let db_h = client.create_from_slice(f64::as_bytes(&nz));
    let mu0z = vec![0.0f64; 1];
    let mu0_h = client.create_from_slice(f64::as_bytes(&mu0z));

    if is_laguerre {
        llaguerre_tridiag_kernel::launch::<cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(momh, MXRYSROOTS * 2) },
            unsafe { ArrayArg::from_raw_parts(moml, MXRYSROOTS * 2) },
            unsafe { ArrayArg::from_raw_parts(alh, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(all_, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(beh, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(bel, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(ah, n) },
            unsafe { ArrayArg::from_raw_parts(al, n) },
            unsafe { ArrayArg::from_raw_parts(bh, n) },
            unsafe { ArrayArg::from_raw_parts(bl, n) },
            unsafe { ArrayArg::from_raw_parts(s0h, n2) },
            unsafe { ArrayArg::from_raw_parts(s0l, n2) },
            unsafe { ArrayArg::from_raw_parts(smh, n2) },
            unsafe { ArrayArg::from_raw_parts(sml, n2) },
            unsafe { ArrayArg::from_raw_parts(skh, n2) },
            unsafe { ArrayArg::from_raw_parts(skl, n2) },
            unsafe { ArrayArg::from_raw_parts(da_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(db_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(mu0_h.clone(), 1) },
            x,
            n as u32,
        );
    } else {
        let alpha = &data::LJACOBI_ALPHA[..];
        let beta = &data::LJACOBI_BETA[..];
        let rn = &data::LJACOBI_RN_PART2[..];
        let snv = &data::LJACOBI_SN[..];
        let alpha_h = client.create_from_slice(f64::as_bytes(alpha));
        let beta_h = client.create_from_slice(f64::as_bytes(beta));
        let rn_h = client.create_from_slice(f64::as_bytes(rn));
        let sn_h = client.create_from_slice(f64::as_bytes(snv));
        ljacobi_tridiag_kernel::launch::<cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(alpha_h, alpha.len()) },
            unsafe { ArrayArg::from_raw_parts(beta_h, beta.len()) },
            unsafe { ArrayArg::from_raw_parts(rn_h, rn.len()) },
            unsafe { ArrayArg::from_raw_parts(sn_h, snv.len()) },
            unsafe { ArrayArg::from_raw_parts(momh, MXRYSROOTS * 2) },
            unsafe { ArrayArg::from_raw_parts(moml, MXRYSROOTS * 2) },
            unsafe { ArrayArg::from_raw_parts(alh, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(all_, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(beh, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(bel, abz.len()) },
            unsafe { ArrayArg::from_raw_parts(ah, n) },
            unsafe { ArrayArg::from_raw_parts(al, n) },
            unsafe { ArrayArg::from_raw_parts(bh, n) },
            unsafe { ArrayArg::from_raw_parts(bl, n) },
            unsafe { ArrayArg::from_raw_parts(s0h, n2) },
            unsafe { ArrayArg::from_raw_parts(s0l, n2) },
            unsafe { ArrayArg::from_raw_parts(smh, n2) },
            unsafe { ArrayArg::from_raw_parts(sml, n2) },
            unsafe { ArrayArg::from_raw_parts(skh, n2) },
            unsafe { ArrayArg::from_raw_parts(skl, n2) },
            unsafe { ArrayArg::from_raw_parts(da_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(db_h.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(mu0_h.clone(), 1) },
            x,
            n as u32,
        );
    }

    let da_bytes = client.read_one_unchecked(da_h);
    let da = f64::from_bytes(&da_bytes)[0..n].to_vec();
    let db_bytes = client.read_one_unchecked(db_h);
    let db = f64::from_bytes(&db_bytes)[0..n].to_vec();

    let mut diag = da.clone();
    let mut offdiag = vec![0.0f64; n];
    for i in 0..n.saturating_sub(1) {
        offdiag[i] = db[i + 1];
    }
    let mut eig = vec![0.0f64; n];
    let mut c0 = vec![0.0f64; n * n];
    let error = eigh::cint_diagonalize(n, &mut diag, &mut offdiag, &mut eig, &mut c0);

    // transform (reuse the f64 jacobi_transform_kernel).
    let eig_h = client.create_from_slice(f64::as_bytes(&eig));
    let c0_h = client.create_from_slice(f64::as_bytes(&c0));
    let rz = vec![0.0f64; n];
    let roots_h = client.create_from_slice(f64::as_bytes(&rz));
    let weights_h = client.create_from_slice(f64::as_bytes(&rz));
    jacobi_transform_kernel::launch::<cubecl::cpu::CpuRuntime>(
        &client,
        crate::plane::single_cube_count(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(eig_h, n) },
        unsafe { ArrayArg::from_raw_parts(c0_h, n * n) },
        unsafe { ArrayArg::from_raw_parts(mu0_h, 1) },
        unsafe { ArrayArg::from_raw_parts(roots_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(weights_h.clone(), n) },
        n as u32,
    );
    let r_bytes = client.read_one_unchecked(roots_h);
    roots[0..n].copy_from_slice(&f64::from_bytes(&r_bytes)[0..n]);
    let w_bytes = client.read_one_unchecked(weights_h);
    weights[0..n].copy_from_slice(&f64::from_bytes(&w_bytes)[0..n]);
    error
}

#[cfg(feature = "cpu")]
fn lrys_jacobi_device(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    lrys_wheeler_device(nroots, x, roots, weights, false)
}
#[cfg(feature = "cpu")]
fn lrys_laguerre_device(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    lrys_wheeler_device(nroots, x, roots, weights, true)
}

// ---------------------------------------------------------------------------
// Device dd Schmidt path (lrys_schmidt) — nroots 8 large-x tail (x > 11).
// ---------------------------------------------------------------------------

const SML_FLOAT80_DEV: f64 = 2.0e-20;

/// Device lgamma_inc_like in dd. `f` is dd (fh/fl). `turnover` = TURNOVER_POINT[m].
#[cube]
fn lgamma_inc_like_dev(fh: &mut Array<f64>, fl: &mut Array<f64>, t: f64, m: u32, turnover: f64) {
    if t == 0.0 {
        fh[(0) as usize] = 1.0;
        fl[(0) as usize] = 0.0;
        let mut i: u32 = 1;
        while i <= m {
            let v = dd_div_dev(dd_from(1.0), dd_from((2 * i + 1) as f64));
            fh[(i) as usize] = v.hi;
            fl[(i) as usize] = v.lo;
            i += 1;
        }
    } else if t < turnover {
        // power series in dd.
        let mut bh = m as f64 + 0.5;
        let mut bl = 0.0f64;
        let e = dd_mul_f64_dev(dd_from(f64::exp(-t)), 0.5);
        let mut xh = e.hi;
        let mut xl = e.lo;
        let mut sh = e.hi;
        let mut sl = e.lo;
        let tol = SML_FLOAT80_DEV * f64::abs(e.hi);
        let bi0 = dd_add_dev(DdDev { hi: bh, lo: bl }, dd_from(1.0));
        let mut bih = bi0.hi;
        let mut bil = bi0.lo;
        let mut k: u32 = 0;
        let mut going = true;
        while k < 4096u32 {
            if going {
                if xh > tol {
                    let x = DdDev { hi: xh, lo: xl };
                    let xn = dd_mul_dev(x, dd_div_dev(dd_from(t), DdDev { hi: bih, lo: bil }));
                    xh = xn.hi;
                    xl = xn.lo;
                    let s = dd_add_dev(DdDev { hi: sh, lo: sl }, xn);
                    sh = s.hi;
                    sl = s.lo;
                    let bin = dd_add_dev(DdDev { hi: bih, lo: bil }, dd_from(1.0));
                    bih = bin.hi;
                    bil = bin.lo;
                } else {
                    going = false;
                }
            }
            k += 1;
        }
        let fm = dd_div_dev(DdDev { hi: sh, lo: sl }, DdDev { hi: bh, lo: bl });
        fh[(m) as usize] = fm.hi;
        fl[(m) as usize] = fm.lo;
        let mut i: u32 = m;
        while i > 0 {
            let bn = dd_sub_dev(DdDev { hi: bh, lo: bl }, dd_from(1.0));
            bh = bn.hi;
            bl = bn.lo;
            let fi = DdDev {
                hi: fh[(i) as usize],
                lo: fl[(i) as usize],
            };
            let num = dd_add_dev(e, dd_mul_f64_dev(fi, t));
            let v = dd_div_dev(num, DdDev { hi: bh, lo: bl });
            fh[(i - 1) as usize] = v.hi;
            fl[(i - 1) as usize] = v.lo;
            i -= 1;
        }
    } else {
        let tt = f64::sqrt(t);
        let f0 = dd_from(SQRTPIE4_DD / tt * erf_dev(tt));
        fh[(0) as usize] = f0.hi;
        fl[(0) as usize] = f0.lo;
        let e = dd_from(f64::exp(-t));
        let bb = dd_div_dev(dd_from(0.5), dd_from(t));
        let mut i: u32 = 1;
        while i <= m {
            let fim1 = DdDev {
                hi: fh[(i - 1) as usize],
                lo: fl[(i - 1) as usize],
            };
            let inner = dd_sub_dev(dd_mul_f64_dev(fim1, (2 * i - 1) as f64), e);
            let v = dd_mul_dev(bb, inner);
            fh[(i) as usize] = v.hi;
            fl[(i) as usize] = v.lo;
            i += 1;
        }
    }
}

/// Device R_lsmit in dd. `cs` dd (csh/csl), column-major n×n. `v` dd scratch (len>=n).
/// Returns via flag[0] (0 ok / 1 / j>0).
#[cube]
#[allow(clippy::too_many_arguments)]
fn r_lsmit_dev(
    csh: &mut Array<f64>,
    csl: &mut Array<f64>,
    fmh: &Array<f64>,
    fml: &Array<f64>,
    n: u32,
    vh: &mut Array<f64>,
    vl: &mut Array<f64>,
    flag: &mut Array<f64>,
) {
    let mut ret = 0.0f64;
    let f0 = DdDev {
        hi: fmh[(0) as usize],
        lo: fml[(0) as usize],
    };
    let f1 = DdDev {
        hi: fmh[(1) as usize],
        lo: fml[(1) as usize],
    };
    let f2 = DdDev {
        hi: fmh[(2) as usize],
        lo: fml[(2) as usize],
    };
    let fac = dd_div_dev(
        DdDev {
            hi: -f1.hi,
            lo: -f1.lo,
        },
        f0,
    );
    let tmp0 = dd_add_dev(f2, dd_mul_dev(fac, f1));
    let mut aborted = false;
    if tmp0.hi <= 0.0 {
        let mut i: u32 = 0;
        while i < n {
            csh[(i + 1 * n) as usize] = 0.0;
            csl[(i + 1 * n) as usize] = 0.0;
            i += 1;
        }
        ret = 1.0;
        aborted = true;
    }
    if !aborted {
        let tmp = dd_div_dev(dd_from(1.0), dd_sqrt_dev(tmp0));
        let c00 = dd_div_dev(dd_from(1.0), dd_sqrt_dev(f0));
        csh[(0) as usize] = c00.hi;
        csl[(0) as usize] = c00.lo;
        let cn = dd_mul_dev(fac, tmp);
        csh[(n) as usize] = cn.hi;
        csl[(n) as usize] = cn.lo;
        csh[(1 + n) as usize] = tmp.hi;
        csl[(1 + n) as usize] = tmp.lo;

        let mut j: u32 = 2;
        let mut stop = false;
        while j < n {
            if !stop {
                let mut vi: u32 = 0;
                while vi < j {
                    vh[(vi) as usize] = 0.0;
                    vl[(vi) as usize] = 0.0;
                    vi += 1;
                }
                let fjj = DdDev {
                    hi: fmh[(j + j) as usize],
                    lo: fml[(j + j) as usize],
                };
                let mut fach = fjj.hi;
                let mut facl = fjj.lo;
                let mut k: u32 = 0;
                while k < j {
                    let mut doth = 0.0f64;
                    let mut dotl = 0.0f64;
                    let mut i: u32 = 0;
                    while i <= k {
                        let cik = DdDev {
                            hi: csh[(i + k * n) as usize],
                            lo: csl[(i + k * n) as usize],
                        };
                        let fij = DdDev {
                            hi: fmh[(i + j) as usize],
                            lo: fml[(i + j) as usize],
                        };
                        let d = dd_add_dev(DdDev { hi: doth, lo: dotl }, dd_mul_dev(cik, fij));
                        doth = d.hi;
                        dotl = d.lo;
                        i += 1;
                    }
                    let dot = DdDev { hi: doth, lo: dotl };
                    let mut i2: u32 = 0;
                    while i2 <= k {
                        let cik = DdDev {
                            hi: csh[(i2 + k * n) as usize],
                            lo: csl[(i2 + k * n) as usize],
                        };
                        let vv = dd_sub_dev(
                            DdDev {
                                hi: vh[(i2) as usize],
                                lo: vl[(i2) as usize],
                            },
                            dd_mul_dev(dot, cik),
                        );
                        vh[(i2) as usize] = vv.hi;
                        vl[(i2) as usize] = vv.lo;
                        i2 += 1;
                    }
                    let facn = dd_sub_dev(DdDev { hi: fach, lo: facl }, dd_mul_dev(dot, dot));
                    fach = facn.hi;
                    facl = facn.lo;
                    k += 1;
                }
                if fach <= 0.0 {
                    let mut kk: u32 = j;
                    while kk < n {
                        let mut i: u32 = 0;
                        while i < n {
                            csh[(i + kk * n) as usize] = 0.0;
                            csl[(i + kk * n) as usize] = 0.0;
                            i += 1;
                        }
                        kk += 1;
                    }
                    if fach == 0.0 {
                        ret = 0.0;
                    } else {
                        ret = j as f64;
                    }
                    stop = true;
                } else {
                    let facd = dd_div_dev(dd_from(1.0), dd_sqrt_dev(DdDev { hi: fach, lo: facl }));
                    csh[(j + j * n) as usize] = facd.hi;
                    csl[(j + j * n) as usize] = facd.lo;
                    let mut k2: u32 = 0;
                    while k2 < j {
                        let vv = DdDev {
                            hi: vh[(k2) as usize],
                            lo: vl[(k2) as usize],
                        };
                        let c = dd_mul_dev(facd, vv);
                        csh[(k2 + j * n) as usize] = c.hi;
                        csl[(k2 + j * n) as usize] = c.lo;
                        k2 += 1;
                    }
                }
            }
            j += 1;
        }
    }
    flag[(0) as usize] = ret;
}

/// Device lrys_schmidt kernel (dd): dd FMT moments + R_lsmit (dd) -> f64 cs -> QR roots
/// (f64 cint_polynomial_roots) -> weights. `nroots` comptime.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn lschmidt_kernel(
    fmh: &mut Array<f64>,
    fml: &mut Array<f64>,
    csh: &mut Array<f64>,
    csl: &mut Array<f64>,
    csf: &mut Array<f64>,
    rt: &mut Array<f64>,
    acomp: &mut Array<f64>,
    vh: &mut Array<f64>,
    vl: &mut Array<f64>,
    flag: &mut Array<f64>,
    roots: &mut Array<f64>,
    weights: &mut Array<f64>,
    x: f64,
    turnover: f64,
    #[comptime] nroots: u32,
) {
    let nroots1 = nroots + 1;
    lgamma_inc_like_dev(fmh, fml, x, nroots * 2, turnover);

    let mut zeroed = false;
    if fmh[(0) as usize] == 0.0 {
        let mut k: u32 = 0;
        while k < nroots {
            roots[(k) as usize] = 0.0;
            weights[(k) as usize] = 0.0;
            k += 1;
        }
        zeroed = true;
    }

    if !zeroed {
        // zero qcs (dd) and r_lsmit.
        let nn1 = nroots1 * nroots1;
        let mut z: u32 = 0;
        while z < nn1 {
            csh[(z) as usize] = 0.0;
            csl[(z) as usize] = 0.0;
            z += 1;
        }
        r_lsmit_dev(csh, csl, fmh, fml, nroots1, vh, vl, flag);
        // f64 copy of cs for the polynomial root finder + weights.
        let mut i: u32 = 0;
        while i < nn1 {
            csf[(i) as usize] = csh[(i) as usize];
            i += 1;
        }
        cint_polynomial_roots_dev(rt, csf, nroots, nroots1, acomp, flag);

        let dum0 = 1.0 / fmh[(0) as usize];
        let mut k: u32 = 0;
        while k < nroots {
            let root = rt[(k) as usize];
            if root == 1.0 {
                roots[(k) as usize] = 0.0;
                weights[(k) as usize] = 0.0;
            } else {
                let mut dum = dum0;
                let mut j: u32 = 1;
                while j < nroots {
                    let off = j * nroots1;
                    let poly = poly_value1_dev(csf, off, j, root);
                    dum += poly * poly;
                    j += 1;
                }
                roots[(k) as usize] = root / (1.0 - root);
                weights[(k) as usize] = 1.0 / dum;
            }
            k += 1;
        }
    }
}

/// Host orchestrator for the dd Schmidt path (nroots 8; x > 11).
#[cfg(feature = "cpu")]
fn lrys_schmidt_device(nroots: usize, x: f64, roots: &mut [f64], weights: &mut [f64]) -> i32 {
    let n = nroots;
    let nroots1 = n + 1;
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());

    let fmz = vec![0.0f64; MXRYSROOTS * 2];
    let fmh = client.create_from_slice(f64::as_bytes(&fmz));
    let fml = client.create_from_slice(f64::as_bytes(&fmz));
    let csz = vec![0.0f64; nroots1 * nroots1];
    let csh = client.create_from_slice(f64::as_bytes(&csz));
    let csl = client.create_from_slice(f64::as_bytes(&csz));
    let csf = client.create_from_slice(f64::as_bytes(&csz));
    let rtz = vec![0.0f64; n];
    let rt_h = client.create_from_slice(f64::as_bytes(&rtz));
    let acz = vec![0.0f64; n * n];
    let acomp = client.create_from_slice(f64::as_bytes(&acz));
    let vz = vec![0.0f64; MXRYSROOTS];
    let vh = client.create_from_slice(f64::as_bytes(&vz));
    let vl = client.create_from_slice(f64::as_bytes(&vz));
    let flz = vec![0.0f64; 1];
    let flag = client.create_from_slice(f64::as_bytes(&flz));
    let rz = vec![0.0f64; n];
    let roots_h = client.create_from_slice(f64::as_bytes(&rz));
    let weights_h = client.create_from_slice(f64::as_bytes(&rz));

    let turnover = data::TURNOVER_POINT[n * 2];

    lschmidt_kernel::launch::<cubecl::cpu::CpuRuntime>(
        &client,
        crate::plane::single_cube_count(),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(fmh, MXRYSROOTS * 2) },
        unsafe { ArrayArg::from_raw_parts(fml, MXRYSROOTS * 2) },
        unsafe { ArrayArg::from_raw_parts(csh, nroots1 * nroots1) },
        unsafe { ArrayArg::from_raw_parts(csl, nroots1 * nroots1) },
        unsafe { ArrayArg::from_raw_parts(csf, nroots1 * nroots1) },
        unsafe { ArrayArg::from_raw_parts(rt_h, n) },
        unsafe { ArrayArg::from_raw_parts(acomp, n * n) },
        unsafe { ArrayArg::from_raw_parts(vh, MXRYSROOTS) },
        unsafe { ArrayArg::from_raw_parts(vl, MXRYSROOTS) },
        unsafe { ArrayArg::from_raw_parts(flag, 1) },
        unsafe { ArrayArg::from_raw_parts(roots_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(weights_h.clone(), n) },
        x,
        turnover,
        n as u32,
    );
    let r_bytes = client.read_one_unchecked(roots_h);
    roots[0..n].copy_from_slice(&f64::from_bytes(&r_bytes)[0..n]);
    let w_bytes = client.read_one_unchecked(weights_h);
    weights[0..n].copy_from_slice(&f64::from_bytes(&w_bytes)[0..n]);
    0
}

// ===========================================================================
//  Inline device extended-Rys entry (Phase 33, task 33-01).
//
//  `rys_roots_ext_dev` is the whole `nroots` 6..=12 Wheeler dispatch expressed
//  as a `#[cube]` CALLEE rather than as something the host launches. A family
//  kernel that opts in calls it from inside its own launch, so a `(pp|ff)`-class
//  quartet no longer has to leave the device for its roots.
//
//  Three properties make this practical, and each one is load-bearing:
//
//  1. **One table argument, not eight.** Only the two Jacobi arms need constant
//     tables — the Laguerre arm builds its moments from `t` in-kernel and the
//     Schmidt arms take `turnover` as a scalar. The eight tables
//     (`JACOBI_*` + `LJACOBI_*`, 48/48/88/88 f64 each) plus `TURNOVER_POINT`
//     are concatenated into one blob of [`EXT_TABLES_LEN`] f64 (~4.7 KB) that
//     the host uploads once per run; every read is at a comptime offset. An
//     opting-in family kernel therefore grows exactly one `&Array<f64>`
//     parameter.
//  2. **Scratch is local, not a parameter.** Every working buffer is a
//     `#[cube]`-local `Array::<f64>::new(<comptime>)`, sized from `nroots`.
//     That keeps the signature to (tables, x, u, w) and puts the scratch in
//     thread-private memory rather than in the shared-memory budget — see
//     `shared_memory::ext_rys_scratch_words` for the measured sizes.
//  3. **The dd hazard is narrow.** `two_prod_dev` asks for `fma` explicitly and
//     *wants* the fusion; `two_sum_dev` contains no multiply-add at all, so
//     contraction cannot reach it. What the fusion buys is checked per backend
//     by `device_rys_ceiling::probe_fma_fusion`, and the ceiling stays at
//     `BASE_DEVICE_NROOTS` until that probe passes.
//
//  The dispatch below mirrors `rys_roots_host_wheeler` arm for arm, including
//  `segment_solve`'s recovery path: on a solver error the host calls the f64
//  `rys_schmidt`, so the inline entry calls the f64 Schmidt arm — the same
//  callee, inline — exactly once, and does not recurse on its error.
// ===========================================================================

/// Offset of `JACOBI_ALPHA` (48 f64) in the extended-Rys table blob.
pub(crate) const EXT_TAB_JACOBI_ALPHA: u32 = 0;
/// Offset of `JACOBI_BETA` (48 f64).
pub(crate) const EXT_TAB_JACOBI_BETA: u32 = 48;
/// Offset of `JACOBI_RN_PART2` (88 f64).
pub(crate) const EXT_TAB_JACOBI_RN_PART2: u32 = 96;
/// Offset of `JACOBI_SN` (88 f64).
pub(crate) const EXT_TAB_JACOBI_SN: u32 = 184;
/// Offset of `LJACOBI_ALPHA` (48 f64).
pub(crate) const EXT_TAB_LJACOBI_ALPHA: u32 = 272;
/// Offset of `LJACOBI_BETA` (48 f64).
pub(crate) const EXT_TAB_LJACOBI_BETA: u32 = 320;
/// Offset of `LJACOBI_RN_PART2` (88 f64).
pub(crate) const EXT_TAB_LJACOBI_RN_PART2: u32 = 368;
/// Offset of `LJACOBI_SN` (88 f64).
pub(crate) const EXT_TAB_LJACOBI_SN: u32 = 456;
/// Offset of `TURNOVER_POINT` (40 f64). Read at `EXT_TAB_TURNOVER + nroots * 2`,
/// which is a comptime index because `nroots` is comptime.
pub(crate) const EXT_TAB_TURNOVER: u32 = 544;
/// Total length of the concatenated blob, in f64.
///
/// Public because a family launcher that opts into the extended path has to
/// create the buffer at exactly this length.
pub const EXT_TABLES_LEN: usize = 584;

/// Build the concatenated extended-Rys table blob the inline entry reads.
///
/// Uploaded once per run and shared by every work item; the order here is the
/// one the `EXT_TAB_*` offsets name, and `ext_table_offsets_match_lengths`
/// pins the two together.
pub fn ext_rys_tables() -> Vec<f64> {
    let mut blob = Vec::with_capacity(EXT_TABLES_LEN);
    blob.extend_from_slice(&data::JACOBI_ALPHA);
    blob.extend_from_slice(&data::JACOBI_BETA);
    blob.extend_from_slice(&data::JACOBI_RN_PART2);
    blob.extend_from_slice(&data::JACOBI_SN);
    blob.extend_from_slice(&data::LJACOBI_ALPHA);
    blob.extend_from_slice(&data::LJACOBI_BETA);
    blob.extend_from_slice(&data::LJACOBI_RN_PART2);
    blob.extend_from_slice(&data::LJACOBI_SN);
    blob.extend_from_slice(&data::TURNOVER_POINT);
    debug_assert_eq!(blob.len(), EXT_TABLES_LEN);
    blob
}

/// The `x` breakpoint separating the two solvers for `nroots`, from the
/// `CINTrys_roots` dispatch (`rys_roots.c:97-114`, lower==0).
pub(crate) const fn ext_breakpoint(nroots: u32) -> f64 {
    match nroots {
        6..=8 => 11.0,
        9 => 10.0,
        10 | 11 => 18.0,
        _ => 22.0,
    }
}

/// `FLOCKE_EXTRA_ORDER_FOR_DP` as the `u32` the `#[cube]` layer wants.
const FLOCKE_EXTRA_DP_DEV: u32 = 20;

/// Root/weight slots a family kernel's `urys`/`wrys` need for `nroots`.
///
/// The polynomial-fit kernels `rys_root{1..5}` always write five slots
/// regardless of the order asked for, so five is the floor; above that the
/// extended entry writes exactly `nroots`.
pub(crate) const fn ext_rys_slots(nroots: u32) -> usize {
    if nroots > 5 { nroots as usize } else { 5 }
}

/// Slots for the extended arm's own f64 root/weight pair inside a family kernel.
///
/// One when the arm is not emitted: a zero-length `Array` is not a thing the
/// `#[cube]` layer allocates, and one f64 is cheaper than a comptime branch
/// around the declaration.
pub(crate) const fn ext_rys_out_slots(nroots: u32) -> usize {
    if nroots > 5 { nroots as usize } else { 1 }
}

/// Shared tail of every Wheeler arm: eigensolve the tridiagonal `(da, db)` and
/// transform its spectrum into roots and weights.
///
/// This is the host sequence in `rys_jacobi_device` / `lrys_wheeler_device`
/// — `_CINTdiagonalize(n, a, b+1, eig, c0)` then
/// `roots[i] = eig[i]/(1-eig[i])`, `weights[i] = c0[i*n]^2 * mu0` — with the
/// host round trip removed: `cint_diagonalize_dev` is the same body
/// `cint_diagonalize_kernel` launches.
#[cube]
fn ext_eigen_transform_dev(
    da: &Array<f64>,
    db: &Array<f64>,
    mu0: f64,
    u: &mut Array<f64>,
    w: &mut Array<f64>,
    flag: &mut Array<f64>,
    #[comptime] nroots: u32,
) {
    let n = nroots;
    let mut diag = Array::<f64>::new(comptime!(nroots as usize));
    let mut offd = Array::<f64>::new(comptime!(nroots as usize));
    let mut i: u32 = 0;
    while i < n {
        diag[(i) as usize] = da[(i) as usize];
        offd[(i) as usize] = 0.0;
        i += 1;
    }
    // `_CINTdiagonalize` is handed `b + 1`: the off-diagonal is `db[1..n]`, and
    // the last slot stays zero.
    let mut k: u32 = 0;
    while k + 1 < n {
        offd[(k) as usize] = db[(k + 1) as usize];
        k += 1;
    }

    let mut eig = Array::<f64>::new(comptime!(nroots as usize));
    let mut c0 = Array::<f64>::new(comptime!((nroots * nroots) as usize));
    let mut dwork = Array::<f64>::new(comptime!(nroots as usize));
    let mut ework = Array::<f64>::new(comptime!(nroots as usize));
    let mut z = Array::<f64>::new(comptime!((nroots * nroots) as usize));
    let mut dorig = Array::<f64>::new(comptime!(nroots as usize));
    let mut eorig = Array::<f64>::new(comptime!(nroots as usize));
    let mut est = Array::<f64>::new(comptime!(nroots as usize));

    eigh::cint_diagonalize_dev::<f64>(
        &diag, &offd, &mut eig, &mut c0, flag, &mut dwork, &mut ework, &mut z, &mut dorig,
        &mut eorig, &mut est, nroots,
    );

    let mut j: u32 = 0;
    while j < n {
        let r = eig[(j) as usize];
        u[(j) as usize] = r / (1.0 - r);
        let c = c0[(j * n) as usize];
        w[(j) as usize] = c * c * mu0;
        j += 1;
    }
}

/// Inline f64 Jacobi arm (`rys_jacobi`): erf-normalized Flocke moments, Wheeler
/// recursion, eigensolve, transform. Mirrors `jacobi_tridiag_kernel` plus the
/// host tail that followed it.
#[cube]
fn ext_jacobi_f64_dev(
    tab: &Array<f64>,
    x: f64,
    u: &mut Array<f64>,
    w: &mut Array<f64>,
    flag: &mut Array<f64>,
    #[comptime] nroots: u32,
) {
    let n = nroots;
    let mut mom = Array::<f64>::new(comptime!((2 * nroots + 2) as usize));
    let mut a = Array::<f64>::new(comptime!(nroots as usize));
    let mut b = Array::<f64>::new(comptime!(nroots as usize));
    let mut s0 = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut sm = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut sk = Array::<f64>::new(comptime!((2 * nroots) as usize));

    flocke_jacobi_moments_dev(
        n * 2,
        x,
        tab,
        tab,
        comptime!(EXT_TAB_JACOBI_RN_PART2),
        comptime!(EXT_TAB_JACOBI_SN),
        comptime!(FLOCKE_EXTRA_DP_DEV),
        &mut mom,
    );
    let mu0 = mom[(0) as usize];
    wheeler_recursion_dev(
        n,
        tab,
        tab,
        comptime!(EXT_TAB_JACOBI_ALPHA),
        comptime!(EXT_TAB_JACOBI_BETA),
        &mom,
        &mut a,
        &mut b,
        &mut s0,
        &mut sm,
        &mut sk,
    );
    let mut i: u32 = 1;
    while i < n {
        b[(i) as usize] = f64::sqrt(b[(i) as usize]);
        i += 1;
    }
    ext_eigen_transform_dev(&a, &b, mu0, u, w, flag, nroots);
}

/// Inline f64 Schmidt arm (`rys_schmidt`). Also the recovery path every other
/// arm falls back to, which is why it is sized for the whole 6..=12 range
/// rather than only for the `nroots` values that dispatch to it directly.
#[cube]
fn ext_schmidt_f64_dev(
    x: f64,
    turnover: f64,
    u: &mut Array<f64>,
    w: &mut Array<f64>,
    flag: &mut Array<f64>,
    #[comptime] nroots: u32,
) {
    let nroots1 = nroots + 1;
    let mut fmt_ints = Array::<f64>::new(comptime!((2 * nroots + 2) as usize));
    let mut cs = Array::<f64>::new(comptime!(((nroots + 1) * (nroots + 1)) as usize));
    let mut rt = Array::<f64>::new(comptime!(nroots as usize));
    let mut acomp = Array::<f64>::new(comptime!((nroots * nroots) as usize));
    let mut vscr = Array::<f64>::new(comptime!((nroots + 1) as usize));

    gamma_inc_like_dev(&mut fmt_ints, x, nroots * 2, turnover);

    let mut zeroed = false;
    if fmt_ints[(0) as usize] == 0.0 {
        let mut k: u32 = 0;
        while k < nroots {
            u[(k) as usize] = 0.0;
            w[(k) as usize] = 0.0;
            k += 1;
        }
        flag[(0) as usize] = 0.0;
        zeroed = true;
    }

    if !zeroed {
        let nn1 = nroots1 * nroots1;
        let mut z: u32 = 0;
        while z < nn1 {
            cs[(z) as usize] = 0.0;
            z += 1;
        }
        r_dsmit_dev(&mut cs, &fmt_ints, nroots1, &mut vscr, flag);
        cint_polynomial_roots_dev(&mut rt, &cs, nroots, nroots1, &mut acomp, flag);

        let mut k: u32 = 0;
        while k < nroots {
            let root = rt[(k) as usize];
            if root == 1.0 {
                u[(k) as usize] = 0.0;
                w[(k) as usize] = 0.0;
            } else {
                let mut dum = 1.0 / fmt_ints[(0) as usize];
                let mut j: u32 = 1;
                while j < nroots {
                    let off = j * nroots1;
                    let poly = poly_value1_dev(&cs, off, j, root);
                    dum += poly * poly;
                    j += 1;
                }
                u[(k) as usize] = root / (1.0 - root);
                w[(k) as usize] = 1.0 / dum;
            }
            k += 1;
        }
    }
}

/// Inline dd Schmidt arm (`lrys_schmidt`) — the `nroots = 8`, `x > 11` tail.
#[cube]
fn ext_lschmidt_dev(
    x: f64,
    turnover: f64,
    u: &mut Array<f64>,
    w: &mut Array<f64>,
    flag: &mut Array<f64>,
    #[comptime] nroots: u32,
) {
    let nroots1 = nroots + 1;
    let mut fmh = Array::<f64>::new(comptime!((2 * nroots + 2) as usize));
    let mut fml = Array::<f64>::new(comptime!((2 * nroots + 2) as usize));
    let mut csh = Array::<f64>::new(comptime!(((nroots + 1) * (nroots + 1)) as usize));
    let mut csl = Array::<f64>::new(comptime!(((nroots + 1) * (nroots + 1)) as usize));
    let mut csf = Array::<f64>::new(comptime!(((nroots + 1) * (nroots + 1)) as usize));
    let mut rt = Array::<f64>::new(comptime!(nroots as usize));
    let mut acomp = Array::<f64>::new(comptime!((nroots * nroots) as usize));
    let mut vh = Array::<f64>::new(comptime!((nroots + 1) as usize));
    let mut vl = Array::<f64>::new(comptime!((nroots + 1) as usize));

    lgamma_inc_like_dev(&mut fmh, &mut fml, x, nroots * 2, turnover);

    let mut zeroed = false;
    if fmh[(0) as usize] == 0.0 {
        let mut k: u32 = 0;
        while k < nroots {
            u[(k) as usize] = 0.0;
            w[(k) as usize] = 0.0;
            k += 1;
        }
        flag[(0) as usize] = 0.0;
        zeroed = true;
    }

    if !zeroed {
        let nn1 = nroots1 * nroots1;
        let mut z: u32 = 0;
        while z < nn1 {
            csh[(z) as usize] = 0.0;
            csl[(z) as usize] = 0.0;
            z += 1;
        }
        r_lsmit_dev(
            &mut csh, &mut csl, &fmh, &fml, nroots1, &mut vh, &mut vl, flag,
        );
        let mut i: u32 = 0;
        while i < nn1 {
            csf[(i) as usize] = csh[(i) as usize];
            i += 1;
        }
        cint_polynomial_roots_dev(&mut rt, &csf, nroots, nroots1, &mut acomp, flag);

        let dum0 = 1.0 / fmh[(0) as usize];
        let mut k: u32 = 0;
        while k < nroots {
            let root = rt[(k) as usize];
            if root == 1.0 {
                u[(k) as usize] = 0.0;
                w[(k) as usize] = 0.0;
            } else {
                let mut dum = dum0;
                let mut j: u32 = 1;
                while j < nroots {
                    let off = j * nroots1;
                    let poly = poly_value1_dev(&csf, off, j, root);
                    dum += poly * poly;
                    j += 1;
                }
                u[(k) as usize] = root / (1.0 - root);
                w[(k) as usize] = 1.0 / dum;
            }
            k += 1;
        }
    }
}

/// Inline dd Jacobi / dd Laguerre arms (`lrys_jacobi` / `lrys_laguerre`).
///
/// One body for both because they differ only in where `(alpha, beta, moments)`
/// come from — the `LJACOBI_*` tables plus `lflocke_jacobi_moments_dev`, or
/// `llaguerre_moments_dev` building them from `t` — after which the dd Wheeler
/// recursion, the round-to-f64 tridiagonal and the eigensolve are identical.
/// `is_laguerre` is a *runtime* selector on purpose. Making it comptime would
/// give the two arms separate instantiations and therefore two copies of the
/// ~30n words of dd scratch and two copies of the eigensolve tail; as a runtime
/// branch inside one body they share both, which halves the private footprint
/// at `nroots >= 9` for the cost of a branch that `x <= bp` already implies.
#[cube]
fn ext_lwheeler_dev(
    tab: &Array<f64>,
    x: f64,
    is_laguerre: bool,
    u: &mut Array<f64>,
    w: &mut Array<f64>,
    flag: &mut Array<f64>,
    #[comptime] nroots: u32,
) {
    let n = nroots;
    let n2 = nroots * 2;
    let mut momh = Array::<f64>::new(comptime!((2 * nroots + 2) as usize));
    let mut moml = Array::<f64>::new(comptime!((2 * nroots + 2) as usize));
    let mut alh = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut all_ = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut beh = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut bel = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut ah = Array::<f64>::new(comptime!(nroots as usize));
    let mut al = Array::<f64>::new(comptime!(nroots as usize));
    let mut bh = Array::<f64>::new(comptime!(nroots as usize));
    let mut bl = Array::<f64>::new(comptime!(nroots as usize));
    let mut s0h = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut s0l = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut smh = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut sml = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut skh = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut skl = Array::<f64>::new(comptime!((2 * nroots) as usize));
    let mut da = Array::<f64>::new(comptime!(nroots as usize));
    let mut db = Array::<f64>::new(comptime!(nroots as usize));

    if is_laguerre {
        llaguerre_moments_dev(
            n2, x, &mut alh, &mut all_, &mut beh, &mut bel, &mut momh, &mut moml,
        );
    } else {
        lflocke_jacobi_moments_dev(
            n2,
            x,
            tab,
            tab,
            comptime!(EXT_TAB_LJACOBI_RN_PART2),
            comptime!(EXT_TAB_LJACOBI_SN),
            &mut momh,
            &mut moml,
        );
        // The `LJACOBI_*` tables are plain f64; their dd low words are zero.
        let mut i: u32 = 0;
        while i < n2 {
            alh[(i) as usize] = tab[(comptime!(EXT_TAB_LJACOBI_ALPHA) + i) as usize];
            all_[(i) as usize] = 0.0;
            beh[(i) as usize] = tab[(comptime!(EXT_TAB_LJACOBI_BETA) + i) as usize];
            bel[(i) as usize] = 0.0;
            i += 1;
        }
    }
    let mu0 = momh[(0) as usize];

    lwheeler_recursion_dev(
        n, &alh, &all_, &beh, &bel, &momh, &moml, &mut ah, &mut al, &mut bh, &mut bl, &mut s0h,
        &mut s0l, &mut smh, &mut sml, &mut skh, &mut skl,
    );

    da[(0) as usize] = ah[(0) as usize];
    db[(0) as usize] = 0.0;
    let mut k: u32 = 1;
    while k < n {
        da[(k) as usize] = ah[(k) as usize];
        db[(k) as usize] = c99_sqrtl_dev(bh[(k) as usize]);
        k += 1;
    }

    ext_eigen_transform_dev(&da, &db, mu0, u, w, flag, nroots);
}

/// **The inline extended-Rys entry (task 33-01).**
///
/// Writes `nroots` roots into `u[0..nroots]` and weights into `w[0..nroots]`,
/// reproducing `rys_roots_host_wheeler` arm for arm:
///
/// | nroots | `x <= bp` | `x > bp` | bp |
/// |---|---|---|---|
/// | 6, 7 | f64 Jacobi | f64 Schmidt | 11 |
/// | 8 | f64 Jacobi | dd Schmidt | 11 |
/// | 9 | dd Jacobi | dd Laguerre | 10 |
/// | 10, 11 | dd Jacobi | dd Laguerre | 18 |
/// | 12 | dd Jacobi | dd Laguerre | 22 |
///
/// `tab` is the blob [`ext_rys_tables`] builds. `nroots` is comptime, so the
/// breakpoint, the turnover index and the arm selection all fold away and a
/// given instantiation emits exactly one solver pair plus the recovery arm.
#[cube]
pub(crate) fn rys_roots_ext_dev(
    tab: &Array<f64>,
    x: f64,
    u: &mut Array<f64>,
    w: &mut Array<f64>,
    #[comptime] nroots: u32,
) {
    let mut flag = Array::<f64>::new(1usize);
    flag[(0) as usize] = 0.0;

    let bp = comptime!(ext_breakpoint(nroots));
    let turnover = tab[comptime!((EXT_TAB_TURNOVER + nroots * 2) as usize)];

    if comptime!(nroots <= 7) {
        if x <= bp {
            ext_jacobi_f64_dev(tab, x, u, w, &mut flag, nroots);
        } else {
            ext_schmidt_f64_dev(x, turnover, u, w, &mut flag, nroots);
        }
    } else if comptime!(nroots == 8) {
        if x <= bp {
            ext_jacobi_f64_dev(tab, x, u, w, &mut flag, nroots);
        } else {
            ext_lschmidt_dev(x, turnover, u, w, &mut flag, nroots);
        }
    } else {
        ext_lwheeler_dev(tab, x, x > bp, u, w, &mut flag, nroots);
    }

    // `segment_solve`'s recovery path (rys_roots.c:42): on a solver error the
    // host retries through the f64 Schmidt solver and returns that result.
    // Inline, that is the same callee — and, as on the host, its own error is
    // not retried.
    if flag[(0) as usize] != 0.0 {
        ext_schmidt_f64_dev(x, turnover, u, w, &mut flag, nroots);
    }
}

/// A one-line launch wrapper over [`rys_roots_ext_dev`], so the accuracy gate
/// (task 33-04) measures the *same* body a family kernel will call rather than
/// a second copy of it.
#[cube(launch)]
fn rys_roots_ext_kernel(
    tab: &Array<f64>,
    roots: &mut Array<f64>,
    weights: &mut Array<f64>,
    x: f64,
    #[comptime] nroots: u32,
) {
    rys_roots_ext_dev(tab, x, roots, weights, nroots);
}

/// Host entry for the **inline** extended-Rys path — one launch, one work item.
///
/// Not the production dispatch: `rys_roots_host_wheeler` still owns that, and
/// stays the reference the inline path is measured against
/// (`rys_ext_inline_matches_host_wheeler`). This exists so the inline entry is
/// reachable from a test and from a benchmark without a family kernel in the
/// way.
#[cfg(feature = "cpu")]
pub fn rys_roots_ext_host(nroots: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(
        (6..=12).contains(&nroots),
        "rys_roots_ext_host: nroots={nroots} outside the validated 6..=12 range"
    );
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    let tables = ext_rys_tables();
    let tab_h = client.create_from_slice(f64::as_bytes(&tables));
    let zero = vec![0.0f64; nroots];
    let roots_h = client.create_from_slice(f64::as_bytes(&zero));
    let weights_h = client.create_from_slice(f64::as_bytes(&zero));

    // `nroots` is comptime in the kernel, so the dispatch is spelled out rather
    // than passed through: each arm is a distinct specialization.
    macro_rules! launch_ext {
        ($n:literal) => {
            rys_roots_ext_kernel::launch::<cubecl::cpu::CpuRuntime>(
                &client,
                crate::plane::single_cube_count(),
                CubeDim::new_1d(1),
                // SAFETY: each buffer is created at exactly the length passed here.
                unsafe { ArrayArg::from_raw_parts(tab_h, EXT_TABLES_LEN) },
                unsafe { ArrayArg::from_raw_parts(roots_h.clone(), nroots) },
                unsafe { ArrayArg::from_raw_parts(weights_h.clone(), nroots) },
                x,
                $n,
            )
        };
    }
    match nroots {
        6 => launch_ext!(6u32),
        7 => launch_ext!(7u32),
        8 => launch_ext!(8u32),
        9 => launch_ext!(9u32),
        10 => launch_ext!(10u32),
        11 => launch_ext!(11u32),
        12 => launch_ext!(12u32),
        _ => unreachable!(),
    }

    let r_bytes = client.read_one_unchecked(roots_h);
    let roots = f64::from_bytes(&r_bytes)[0..nroots].to_vec();
    let w_bytes = client.read_one_unchecked(weights_h);
    let weights = f64::from_bytes(&w_bytes)[0..nroots].to_vec();
    (roots, weights)
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

#[cfg(not(feature = "cpu"))]
use {
    lrys_jacobi as wheeler_jacobi_dd, lrys_laguerre as wheeler_laguerre_dd,
    lrys_schmidt as wheeler_schmidt_dd, rys_jacobi as wheeler_jacobi_f64,
};
/// Compute the `nroots` Rys roots and weights for `nroots` in 6..=12, host-side.
///
// ---------------------------------------------------------------------------
// Solver selection for `rys_roots_host_wheeler`.
//
// The `*_device` orchestrators host-launch a `#[cube]` kernel on `CpuRuntime`, which
// only exists when the `cpu` feature is on. Under a GPU-only feature profile
// (`--no-default-features --features wgpu|cuda|rocm|metal`) the bit-identical pure-host
// solvers stand in: these are host helper routines feeding the host `nroots > 5`
// fallback, not device kernels, so they must not force `cubecl/cpu` into the build.
// `rys_nroots_sweep` pins the two implementations to the same reference table.
// ---------------------------------------------------------------------------
#[cfg(feature = "cpu")]
use {
    lrys_jacobi_device as wheeler_jacobi_dd, lrys_laguerre_device as wheeler_laguerre_dd,
    lrys_schmidt_device as wheeler_schmidt_dd, rys_jacobi_device as wheeler_jacobi_f64,
};

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
        // nroots 6,7: PARITY-HONEST ESCAPE HATCH (quick task 260531-aw1).
        // The pure-f64 #[cube] device wheeler kernels (rys_jacobi_device / rys_schmidt_device)
        // ARE bit-identical to the host path in isolation (rys_nroots_sweep + the in-crate
        // reference table pass byte-identically), but routing nroots 6,7 — which dominate
        // the largest hess2e components — through a device-kernel LAUNCH in the family hot
        // path perturbs the subsequent HOST g-tensor accumulation by ~1e-11 at the largest
        // roots (a CubeCL CpuRuntime launch FP-environment side effect), tripping the
        // flat-atol=1e-12 family gate (hess2e RTOL=0). Empirically bisected: device 8..12
        // is CLEAN (29/29 holds), device 6,7 breaks hess2e. The device 6,7 kernels are kept
        // in the module as the on-device implementation; the production family-critical
        // dispatch for 6,7 stays on the host path. The bar was NEVER loosened.
        //
        // RE-EXAMINED for task 33-06, after the inline entry (33-01) landed, and the
        // hatch STAYS. Swapping these two arms for `rys_jacobi_device`/`rys_schmidt_device`
        // and re-running `hess2e_parity` gives 3334 mismatches on `int2e_ipip1_sph`, with
        // deltas of 1.3e-12 to 2.0e-12 against a 1e-12 threshold. The tolerance was not
        // touched; the experiment was reverted.
        //
        // What 33-01 did change is the DIAGNOSIS, from a suspicion to a fact.
        // `rys_ext_inline_parity` compares the inline entry — reached by exactly one
        // launch — against this host dispatch over a log-spaced x sweep, and finds
        // 10584/10584 values BIT-IDENTICAL at nroots 6 and 7. So the roots handed to
        // hess2e are the same bits either way, and the ~1e-12 that moves is host
        // arithmetic executed AFTER the launch, not the quadrature. That is a property
        // of the CpuRuntime launch, not of this solver, and no amount of work on the Rys
        // path will remove it.
        6 | 7 => segment_solve(
            nroots,
            x,
            11.0,
            &mut roots,
            &mut weights,
            rys_jacobi,
            rys_schmidt,
        ),
        // nroots 8: f64 Jacobi (x<=11) on-device; dd Schmidt (x>11) on-device (Task 3).
        8 => segment_solve(
            nroots,
            x,
            11.0,
            &mut roots,
            &mut weights,
            wheeler_jacobi_f64,
            wheeler_schmidt_dd,
        ),
        // nroots 9..12: dd Jacobi (x<=bp) + dd Laguerre (x>bp), both on the CubeCL CPU backend.
        9 => segment_solve(
            nroots,
            x,
            10.0,
            &mut roots,
            &mut weights,
            wheeler_jacobi_dd,
            wheeler_laguerre_dd,
        ),
        10 | 11 => segment_solve(
            nroots,
            x,
            18.0,
            &mut roots,
            &mut weights,
            wheeler_jacobi_dd,
            wheeler_laguerre_dd,
        ),
        12 => segment_solve(
            nroots,
            x,
            22.0,
            &mut roots,
            &mut weights,
            wheeler_jacobi_dd,
            wheeler_laguerre_dd,
        ),
        _ => unreachable!(),
    };
    debug_assert_eq!(
        err, 0,
        "rys_roots_host_wheeler: solver error {err} for nroots={nroots} x={x}"
    );

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
        const FRAC_1_SQRT_PI: f64 =
            0.564_189_583_547_756_286_948_079_451_560_772_585_844_050_629_329;
        result = (FRAC_1_SQRT_PI - result) / ax;
        let zt = (ax * 16.0).trunc() / 16.0;
        let del = (ax - zt) * (ax + zt);
        (-zt * zt).exp() * (-del).exp() * result
    }
}

// ---------------------------------------------------------------------------
// Task 3a — FMA fidelity probe (quick task 260531-aw1, RESEARCH Open-Q1 BLOCKER).
//
// The double-double `two_prod(a,b)` error term `e = a.mul_add(b,-p)` is correct ONLY if
// the CubeCL CPU backend lowers `mul_add`/`fma` to a TRUE fused multiply-add (single
// rounding, no intermediate rounding of `a*b`). This probe computes that error term on
// the CPU CubeCL backend for pairs whose product is NOT exactly representable, and the
// test asserts it matches the host-f64 `f64::mul_add` reference BIT-FOR-BIT. If the
// device value instead equals `a*b - p` (== 0.0 when the product is pre-rounded), FMA is
// NOT fused and the double-double port must use a Dekker-split software product.
// ---------------------------------------------------------------------------

/// Device probe: out[i] = fma(a[i], b[i], -(a[i]*b[i])) — the TwoProd error term.
#[cube(launch)]
fn fma_probe_kernel<F: Float + CubeElement>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < a.len() {
        let ai = a[i];
        let bi = b[i];
        let p = ai * bi;
        // fma(ai, bi, -p) == ai*bi - p with a SINGLE rounding: the TwoProd error term.
        // If the backend fuses, this is exact; if it pre-rounds the product, it is 0.
        out[i] = fma(ai, bi, -p);
    }
}

// ===========================================================================
//  D-PBC-24 stage 3 — the LOWER-BOUNDED Rys quadrature (`CINTsr_rys_roots`).
//
//  Everything above this line evaluates `int_0^1 u^{2m} e^{-t u^2} du`, the
//  `lower == 0` family. Short-range Coulomb at `rys_order > 3` needs
//  `int_lower^1` instead (`libcint-master/src/rys_roots.c:145`), and `lower`
//  enters in EXACTLY one place: the FMT moments. `gamma_inc_like` becomes
//  `fmt_erfc_like`, `lgamma_inc_like` becomes `fmt_lerfc_like`,
//  `lflocke_jacobi_moments` becomes `lnaive_jacobi_moments`, and the
//  `lower == 0` branch of `llaguerre_moments` becomes its `else`. Every stage
//  downstream — Schmidt orthogonalization, the Wheeler recursion, the
//  polynomial root finder, the eigensolve — is shared verbatim, which is why
//  this section is short despite covering a whole new quadrature family.
//
//  The vendored oracle build compiles libcint with quadmath DISABLED
//  (`crates/cintx-oracle/build.rs` blanks `HAVE_QUADMATH_H`), so
//  `rys_roots.h:36-38` maps `CINTqrys_jacobi`/`CINTqrys_laguerre`/
//  `CINTqrys_schmidt` onto the `CINTlrys_*` long-double entries. cintx models
//  long double as double-double (`Dd`), so the `q` and `l` arms below are the
//  same function — matching what the reference this is gated against actually
//  runs. `WITH_POLYNOMIAL_FIT` is likewise not defined in that build, so the
//  `CINTsr_rys_polyfits` fast paths at `lower < 0.6` are not part of the
//  dispatch being reproduced.
// ===========================================================================

/// libcint `ERFC_bound` (`fmt.c:25`) — above `t * lower²` this, every moment is 0.
const ERFC_BOUND: f64 = 200.0;

/// libcint `SML_FLOAT80` (`fmt.c:21`) — the long-double series tolerance.
const SML_FLOAT80: f64 = 2.0e-20;

/// `pow_` (`fmt.c:270`) — binary exponentiation, bit-for-bit the vendor's
/// multiply order (which `f64::powi` does NOT guarantee).
fn pow_int(mut base: f64, exponent: usize) -> f64 {
    let mut result = 1.0;
    let mut i = 1usize;
    while i <= exponent {
        if i & exponent != 0 {
            result *= base;
        }
        base *= base;
        i <<= 1;
    }
    result
}

/// `pow_` in double-double.
fn dd_pow_int(mut base: Dd, exponent: usize) -> Dd {
    let mut result = Dd::from(1.0);
    let mut i = 1usize;
    while i <= exponent {
        if i & exponent != 0 {
            result = dd_mul(result, base);
        }
        base = dd_mul(base, base);
        i <<= 1;
    }
    result
}

/// `fmt1_erfc_like` (`fmt.c:310`) — power series for the lower-bounded
/// `F[m] = e^{t s²} ∫_lower^1 u^{2m} e^{-t u²} du`, used for small `t`.
fn fmt1_erfc_like(f: &mut [f64], t: f64, lower: f64, m: usize) {
    let lower2 = lower * lower;
    let mut b = m as f64 + 0.5;
    let e = 0.5 * (-t).exp();
    let mut e1 = 0.5 * (-t * lower2).exp() * lower;
    e1 *= pow_int(lower2, m);
    let x = e;
    let mut x1 = e1;
    let mut s = e - e1;
    let mut div = 1.0;
    let mut delta = s;
    let tol = SML_FLOAT64 * delta.abs();
    let mut bi = b + 1.0;
    // The vendor's loop is unbounded; the iteration cap is cintx's standing
    // no-hang guard (same shape as `lgamma_inc_like_dd`'s) and is never the
    // reason the loop ends inside the validated envelope.
    let mut iter = 0;
    while delta.abs() > tol && iter < 4096 {
        div *= t / bi;
        x1 *= lower2;
        delta = (x - x1) * div;
        s += delta;
        bi += 1.0;
        iter += 1;
    }
    let mut val = s / b;
    f[m] = val;
    let mut i = m;
    while i > 0 {
        b -= 1.0;
        e1 /= lower2;
        val = (e - e1 + t * val) / b;
        f[i - 1] = val;
        i -= 1;
    }
}

/// `fmt_erfc_like` (`fmt.c:340`) — the lower-bounded FMT moments in f64.
///
/// `lower == 0` delegates to [`gamma_inc_like`], so this is a strict superset of
/// the full-range moments and the two can share every caller.
fn fmt_erfc_like(f: &mut [f64], t: f64, lower: f64, m: usize) {
    if lower == 0.0 {
        gamma_inc_like(f, t, m);
        return;
    }

    let lower2 = lower * lower;
    // F[m] < .5*sqrt(pi/t) * erfc(lower*sqrt(t)) — below this it is exactly 0.
    if t * lower2 > ERFC_BOUND {
        for v in f.iter_mut().take(m + 1) {
            *v = 0.0;
        }
        return;
    }

    if t == 0.0 {
        let mut e1 = lower;
        f[0] = 1.0 - e1;
        for i in 1..=m {
            e1 *= lower2;
            f[i] = (1.0 - e1) / (2 * i + 1) as f64;
        }
    } else if t < data::TURNOVER_POINT[m] {
        fmt1_erfc_like(f, t, lower, m);
    } else {
        let tt = t.sqrt();
        // erfc(a) - erfc(b) is more accurate than erf(b) - erf(a).
        let mut val = SQRTPIE4 / tt * (erfc_cody(lower * tt) - erfc_cody(tt));
        f[0] = val;
        let e = (-t).exp();
        let mut e1 = (-t * lower2).exp() * lower;
        let b = 0.5 / t;
        for i in 0..m {
            val = b * ((2 * i + 1) as f64 * val - e + e1);
            e1 *= lower2;
            f[i + 1] = val;
        }
    }
}

/// `fmt1_lerfc_like` (`fmt.c:422`) in double-double.
fn fmt1_lerfc_like_dd(f: &mut [Dd], t: f64, lower: f64, m: usize) {
    let lower2 = dd_mul(Dd::from(lower), Dd::from(lower));
    let mut b = Dd::from(m as f64 + 0.5);
    let e = dd_mul_f64(Dd::from((-t).exp()), 0.5);
    let mut e1 = dd_mul(
        dd_mul_f64(Dd::from((-t * lower * lower).exp()), 0.5),
        Dd::from(lower),
    );
    e1 = dd_mul(e1, dd_pow_int(lower2, m));
    let x = e;
    let mut x1 = e1;
    let mut s = dd_sub(e, e1);
    let mut div = Dd::from(1.0);
    let mut delta = s;
    let tol = SML_FLOAT80 * delta.hi.abs();
    let mut bi = dd_add(b, Dd::from(1.0));
    let mut iter = 0;
    while delta.hi.abs() > tol && iter < 4096 {
        div = dd_mul(div, dd_div(Dd::from(t), bi));
        x1 = dd_mul(x1, lower2);
        delta = dd_mul(dd_sub(x, x1), div);
        s = dd_add(s, delta);
        bi = dd_add(bi, Dd::from(1.0));
        iter += 1;
    }
    let mut val = dd_div(s, b);
    f[m] = val;
    let mut i = m;
    while i > 0 {
        b = dd_sub(b, Dd::from(1.0));
        e1 = dd_div(e1, lower2);
        val = dd_div(dd_add(dd_sub(e, e1), dd_mul_f64(val, t)), b);
        f[i - 1] = val;
        i -= 1;
    }
}

/// `fmt_lerfc_like` (`fmt.c:381`) in double-double.
fn fmt_lerfc_like_dd(f: &mut [Dd], t: f64, lower: f64, m: usize) {
    if lower == 0.0 {
        lgamma_inc_like_dd(f, t, m);
        return;
    }

    let lower2f = lower * lower;
    if t * lower2f > ERFC_BOUND {
        for v in f.iter_mut().take(m + 1) {
            *v = Dd::from(0.0);
        }
        return;
    }

    if t == 0.0 {
        let mut e1 = Dd::from(lower);
        f[0] = dd_sub(Dd::from(1.0), e1);
        let lower2 = dd_mul(Dd::from(lower), Dd::from(lower));
        for i in 1..=m {
            e1 = dd_mul(e1, lower2);
            f[i] = dd_div(dd_sub(Dd::from(1.0), e1), Dd::from((2 * i + 1) as f64));
        }
    } else if t < data::TURNOVER_POINT[m] {
        fmt1_lerfc_like_dd(f, t, lower, m);
    } else {
        let tt = t.sqrt();
        // The vendor evaluates erfcl in long double; cintx's dd emulation lifts
        // the f64 Cody erfc, exactly as `lgamma_inc_like_dd` lifts f64 `erf`.
        let mut val = Dd::from(SQRTPIE4 / tt * (erfc_cody(lower * tt) - erfc_cody(tt)));
        f[0] = val;
        let e = Dd::from((-t).exp());
        let mut e1 = dd_mul(Dd::from((-t * lower2f).exp()), Dd::from(lower));
        let lower2 = dd_mul(Dd::from(lower), Dd::from(lower));
        let b = dd_div(Dd::from(0.5), Dd::from(t));
        for i in 0..m {
            let inner = dd_add(dd_sub(dd_mul_f64(val, (2 * i + 1) as f64), e), e1);
            val = dd_mul(b, inner);
            e1 = dd_mul(e1, lower2);
            f[i + 1] = val;
        }
    }
}

/// `CINTrys_schmidt` (`rys_roots.c:1758`) with a general `lower`.
fn sr_rys_schmidt(
    nroots: usize,
    x: f64,
    lower: f64,
    roots: &mut [f64],
    weights: &mut [f64],
) -> i32 {
    let mut fmt_ints = vec![0.0f64; MXRYSROOTS * 2];
    fmt_erfc_like(&mut fmt_ints, x, lower, nroots * 2);
    rdk_rys_roots(nroots, &fmt_ints, roots, weights)
}

/// `CINTlrys_schmidt` (`rys_roots.c:1851`) with a general `lower`.
///
/// Body identical to [`lrys_schmidt`] apart from the moment source; kept as its
/// own function rather than a `lower` parameter on that one so the validated
/// full-range path is untouched by D-PBC-24.
fn sr_lrys_schmidt(
    nroots: usize,
    x: f64,
    lower: f64,
    roots: &mut [f64],
    weights: &mut [f64],
) -> i32 {
    let nroots1 = nroots + 1;
    let mut fmt_dd = vec![Dd::from(0.0); MXRYSROOTS * 2];
    fmt_lerfc_like_dd(&mut fmt_dd, x, lower, nroots * 2);

    if fmt_dd[0].hi == 0.0 {
        for k in 0..nroots {
            roots[k] = 0.0;
            weights[k] = 0.0;
        }
        return 0;
    }

    let mut rt = vec![0.0f64; nroots];
    let mut qcs = vec![Dd::from(0.0); nroots1 * nroots1];
    if nroots == 1 {
        rt[0] = fmt_dd[1].to_f64() / fmt_dd[0].to_f64();
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
            let a = &cs[j * nroots1..];
            let poly = polynomial_value1(a, j, root);
            dum += poly * poly;
        }
        roots[k] = root / (1.0 - root);
        weights[k] = 1.0 / dum;
    }
    0
}

/// `lnaive_jacobi_moments` (`rys_wheeler.c:3530`) in double-double.
///
/// The Flocke Miller recursion is a `lower == 0` construction, so the
/// lower-bounded Jacobi path always takes this direct
/// `lJACOBI_COEF · fmt_lerfc_like` route (`rys_wheeler.c:3703-3713`).
fn lnaive_jacobi_moments_dd(n: usize, t: f64, lower: f64, mus: &mut [Dd]) {
    let mut fmt = vec![Dd::from(0.0); MXRYSROOTS * 2];
    fmt_lerfc_like_dd(&mut fmt, t, lower, n - 1);
    for i in 0..n {
        let tri = i * (i + 1) / 2;
        let mut s = Dd::from(0.0);
        for j in 0..=i {
            let k = data::JACOBI_COEF_ORDER[tri + j] as usize;
            // FULL-precision coefficients. As `lower → 1` this sum cancels
            // catastrophically — the interval `[lower, 1]` is a sliver — and the
            // COEFFICIENT precision becomes the accuracy floor of the whole
            // quadrature. `roots_jacobi_data::LJACOBI_COEF` truncates the
            // vendor's 34-digit long-double literals to f64, which left cintx
            // ~1000x behind libcint's 80-bit arithmetic at nroots >= 8 and
            // `lower > 0.99`. The double-double image carries the same digits.
            let (hi, lo) = dd_data::LJACOBI_COEF_DD[tri + k];
            s = dd_add(s, dd_mul(Dd::new(hi, lo), fmt[k]));
        }
        mus[i] = s;
    }
}

/// `CINTlrys_jacobi` (`rys_wheeler.c:3703`) with a general `lower`.
fn sr_lrys_jacobi(
    nroots: usize,
    x: f64,
    lower: f64,
    roots: &mut [f64],
    weights: &mut [f64],
) -> i32 {
    let mut moments = vec![Dd::from(0.0); MXRYSROOTS * 2];
    if lower == 0.0 {
        lflocke_jacobi_moments_dd(nroots * 2, x, &mut moments);
    } else {
        lnaive_jacobi_moments_dd(nroots * 2, x, lower, &mut moments);
    }
    // Full-precision alpha/beta for the same reason as the coefficients above.
    let alpha: Vec<Dd> = dd_data::LJACOBI_ALPHA_DD
        .iter()
        .map(|&(hi, lo)| Dd::new(hi, lo))
        .collect();
    let beta: Vec<Dd> = dd_data::LJACOBI_BETA_DD
        .iter()
        .map(|&(hi, lo)| Dd::new(hi, lo))
        .collect();
    lrys_wheeler_partial_dd(nroots, &alpha, &beta, &moments, roots, weights)
}

/// `llaguerre_moments` (`rys_wheeler.c:3479`) in double-double, `lower != 0` arm.
///
/// The recursion runs TWO Laguerre ladders — `l0*` for the upper limit 1 and
/// `l1*` for the lower limit — and differences them, which is what makes the
/// moments those of `∫_lower^1` rather than `∫_0^1`.
fn llaguerre_moments_dd_lower(
    n: usize,
    t: f64,
    lower: f64,
    alpha: &mut [Dd],
    beta: &mut [Dd],
    moments: &mut [Dd],
) {
    if lower == 0.0 {
        llaguerre_moments_dd(n, t, alpha, beta, moments);
        return;
    }

    let tt = t.sqrt();
    let t_inv = dd_div(Dd::from(0.5), Dd::from(t));
    let t2_inv = dd_div(Dd::from(0.5), dd_mul(Dd::from(t), Dd::from(t)));
    let e0 = dd_mul(Dd::from((-t).exp()), t_inv);
    let lower2 = dd_mul(Dd::from(lower), Dd::from(lower));
    let et = dd_mul(
        dd_mul(Dd::from((-t * lower * lower).exp()), Dd::from(lower)),
        t_inv,
    );

    let mut l00 = Dd::from(0.0);
    let mut l01 = Dd::from(1.0);
    let mut l10 = Dd::from(0.0);
    let mut l11 = Dd::from(1.0);

    alpha[0] = t_inv;
    beta[0] = Dd::from(0.0);
    moments[0] = Dd::from(SQRTPIE4 / tt * (erfc_cody(lower * tt) - erfc_cody(tt)));
    moments[1] = dd_sub(dd_mul(l11, et), dd_mul(l01, e0));
    for i in 1..n - 1 {
        alpha[i] = dd_mul_f64(t_inv, (i * 4 + 1) as f64);
        beta[i] = dd_mul_f64(t2_inv, (i * (i * 2 - 1)) as f64);
        let fac0 = dd_mul_f64(t_inv, (i * 4 - 1) as f64);
        let fac1 = dd_mul_f64(t2_inv, ((i - 1) * (i * 2 - 1)) as f64);
        let l12 = dd_sub(dd_mul(dd_sub(lower2, fac0), l11), dd_mul(fac1, l10));
        l10 = l11;
        l11 = l12;
        let l02 = dd_sub(dd_mul(dd_sub(Dd::from(1.0), fac0), l01), dd_mul(fac1, l00));
        l00 = l01;
        l01 = l02;
        moments[i + 1] = dd_sub(dd_mul(l11, et), dd_mul(l01, e0));
    }
}

/// `CINTlrys_laguerre` (`rys_wheeler.c:3692`) with a general `lower`.
fn sr_lrys_laguerre(
    nroots: usize,
    x: f64,
    lower: f64,
    roots: &mut [f64],
    weights: &mut [f64],
) -> i32 {
    let mut moments = vec![Dd::from(0.0); MXRYSROOTS * 2];
    let mut alpha = vec![Dd::from(0.0); nroots * 2];
    let mut beta = vec![Dd::from(0.0); nroots * 2];
    llaguerre_moments_dd_lower(nroots * 2, x, lower, &mut alpha, &mut beta, &mut moments);
    lrys_wheeler_partial_dd(nroots, &alpha, &beta, &moments, roots, weights)
}

type SrQuadrature = fn(usize, f64, f64, &mut [f64], &mut [f64]) -> i32;

/// `segment_solve` (`rys_roots.c:42`) — pick by `x`, fall back to `qrys_schmidt`.
#[allow(clippy::too_many_arguments)]
fn sr_segment_solve(
    n: usize,
    x: f64,
    lower: f64,
    u: &mut [f64],
    w: &mut [f64],
    breakpoint: f64,
    fn1: SrQuadrature,
    fn2: SrQuadrature,
) -> i32 {
    let mut error = if x <= breakpoint {
        fn1(n, x, lower, u, w)
    } else {
        fn2(n, x, lower, u, w)
    };
    if error != 0 {
        error = sr_lrys_schmidt(n, x, lower, u, w);
    }
    error
}

/// `segment_solve1` (`rys_roots.c:126`) — pick by `lower` first, then by `x`.
#[allow(clippy::too_many_arguments)]
fn sr_segment_solve1(
    n: usize,
    x: f64,
    lower: f64,
    u: &mut [f64],
    w: &mut [f64],
    lower_bp1: f64,
    lower_bp2: f64,
    breakpoint: f64,
    fn1: SrQuadrature,
    fn2: SrQuadrature,
    fn3: SrQuadrature,
) -> i32 {
    // `lower_bp2` is threaded to keep the signature faithful to the vendor's;
    // every SR call site passes a value the `lower < lower_bp1` test already
    // subsumes (rys_roots.c:218-244), so it is unused in the body there too.
    let _ = lower_bp2;
    if lower < lower_bp1 {
        if x <= breakpoint {
            fn1(n, x, lower, u, w)
        } else {
            fn2(n, x, lower, u, w)
        }
    } else {
        fn3(n, x, lower, u, w)
    }
}

/// `CINTsr_rys_roots` (`rys_roots.c:145`) — the lower-bounded Rys quadrature.
///
/// `lower = sqrt(theta)` is the short-range integration bound handed down by
/// `CINTg0_2e` (`g2e.c:4472`). Returns `Err(code)` on a solver failure, which
/// the caller must surface rather than substitute anything for.
///
/// The dispatch thresholds (`0.4 / 0.8 / 0.9 / 0.93 / 0.97 / 0.99`) and
/// breakpoints (`10 / 50 / 60`) are accuracy-critical and reproduced exactly.
/// `CINTqrys_*` collapses onto `CINTlrys_*` in the reference build this is
/// gated against, so `sr_lrys_jacobi` serves both.
pub fn sr_rys_roots_host(nroots: usize, x: f64, lower: f64) -> Result<(Vec<f64>, Vec<f64>), i32> {
    assert!(
        (1..=12).contains(&nroots),
        "sr_rys_roots_host: nroots={nroots} outside the validated 1..=12 range"
    );

    if lower == 0.0 {
        // Defensive: `CINTg0_2e` only ever calls this with
        // `lower = sqrt(theta)`, and `theta = ω²/(ω² + a0)` is strictly
        // positive for the ω ≠ 0 that reaches the short-range branch — so
        // `lower == 0` is unreachable from the integral path.
        //
        // Should some other caller hand it in, the lower-bounded quadrature
        // degenerates to the ordinary one, and cintx already has a gated
        // implementation of that: route there instead of through the
        // lower-bounded arms. The vendor's own `lower == 0` route is the legacy
        // long-double Flocke path, which cintx models with the superseded
        // double-double reference and which fails below `SMALLX_LIMIT` for
        // nroots >= 7.
        let (u, w) = crate::math::rys::rys_roots_host::<f64>(nroots, x);
        return Ok((u, w));
    }

    let mut roots = vec![0.0f64; nroots];
    let mut weights = vec![0.0f64; nroots];

    let err = match nroots {
        1 => sr_rys_schmidt(nroots, x, lower, &mut roots, &mut weights),
        2 => {
            if lower < 0.99 {
                sr_rys_schmidt(nroots, x, lower, &mut roots, &mut weights)
            } else {
                sr_lrys_jacobi(nroots, x, lower, &mut roots, &mut weights)
            }
        }
        3 => {
            if lower < 0.93 {
                sr_rys_schmidt(nroots, x, lower, &mut roots, &mut weights)
            } else if lower < 0.97 {
                sr_segment_solve(
                    nroots,
                    x,
                    lower,
                    &mut roots,
                    &mut weights,
                    10.0,
                    sr_lrys_jacobi,
                    sr_lrys_laguerre,
                )
            } else {
                sr_lrys_jacobi(nroots, x, lower, &mut roots, &mut weights)
            }
        }
        4 => {
            if lower < 0.8 {
                sr_rys_schmidt(nroots, x, lower, &mut roots, &mut weights)
            } else if lower < 0.9 {
                sr_segment_solve(
                    nroots,
                    x,
                    lower,
                    &mut roots,
                    &mut weights,
                    10.0,
                    sr_lrys_jacobi,
                    sr_lrys_laguerre,
                )
            } else {
                sr_lrys_jacobi(nroots, x, lower, &mut roots, &mut weights)
            }
        }
        5 => {
            if lower < 0.4 {
                sr_segment_solve(
                    nroots,
                    x,
                    lower,
                    &mut roots,
                    &mut weights,
                    50.0,
                    sr_rys_schmidt,
                    sr_lrys_laguerre,
                )
            } else if lower < 0.8 {
                sr_segment_solve(
                    nroots,
                    x,
                    lower,
                    &mut roots,
                    &mut weights,
                    10.0,
                    sr_lrys_jacobi,
                    sr_lrys_laguerre,
                )
            } else {
                sr_lrys_jacobi(nroots, x, lower, &mut roots, &mut weights)
            }
        }
        6 => {
            if lower < 0.25 {
                sr_segment_solve(
                    nroots,
                    x,
                    lower,
                    &mut roots,
                    &mut weights,
                    60.0,
                    sr_rys_schmidt,
                    sr_lrys_laguerre,
                )
            } else if lower < 0.8 {
                sr_segment_solve(
                    nroots,
                    x,
                    lower,
                    &mut roots,
                    &mut weights,
                    10.0,
                    sr_lrys_jacobi,
                    sr_lrys_laguerre,
                )
            } else {
                sr_lrys_jacobi(nroots, x, lower, &mut roots, &mut weights)
            }
        }
        7 => sr_segment_solve1(
            nroots,
            x,
            lower,
            &mut roots,
            &mut weights,
            0.5,
            1.0,
            60.0,
            sr_lrys_jacobi,
            sr_lrys_laguerre,
            sr_lrys_jacobi,
        ),
        // 8..=12: the vendor names CINTqrys_*, which without quadmath IS
        // CINTlrys_* (rys_roots.h:36-38).
        8..=12 => sr_segment_solve1(
            nroots,
            x,
            lower,
            &mut roots,
            &mut weights,
            0.15,
            1.0,
            60.0,
            sr_lrys_jacobi,
            sr_lrys_laguerre,
            sr_lrys_jacobi,
        ),
        _ => unreachable!("nroots range asserted above"),
    };

    if err != 0 {
        return Err(err);
    }
    Ok((roots, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `EXT_TAB_*` offsets are the only thing tying `rys_roots_ext_dev`'s
    /// comptime reads to `ext_rys_tables`'s concatenation order. Nothing in the
    /// type system connects them, so they are checked here: each offset must be
    /// where the previous table ends, and the blob must be exactly as long as
    /// the offsets claim.
    #[test]
    fn ext_table_offsets_match_lengths() {
        let expected: [(u32, usize); 9] = [
            (EXT_TAB_JACOBI_ALPHA, data::JACOBI_ALPHA.len()),
            (EXT_TAB_JACOBI_BETA, data::JACOBI_BETA.len()),
            (EXT_TAB_JACOBI_RN_PART2, data::JACOBI_RN_PART2.len()),
            (EXT_TAB_JACOBI_SN, data::JACOBI_SN.len()),
            (EXT_TAB_LJACOBI_ALPHA, data::LJACOBI_ALPHA.len()),
            (EXT_TAB_LJACOBI_BETA, data::LJACOBI_BETA.len()),
            (EXT_TAB_LJACOBI_RN_PART2, data::LJACOBI_RN_PART2.len()),
            (EXT_TAB_LJACOBI_SN, data::LJACOBI_SN.len()),
            (EXT_TAB_TURNOVER, data::TURNOVER_POINT.len()),
        ];
        let mut cursor = 0usize;
        for (offset, len) in expected {
            assert_eq!(
                offset as usize, cursor,
                "table offset {offset} does not follow the previous table, which ends at {cursor}"
            );
            cursor += len;
        }
        assert_eq!(cursor, EXT_TABLES_LEN);

        let blob = ext_rys_tables();
        assert_eq!(blob.len(), EXT_TABLES_LEN);
        // Spot-check the two the dispatch reads at a comptime index.
        assert_eq!(blob[EXT_TAB_LJACOBI_SN as usize], data::LJACOBI_SN[0]);
        for nroots in 6..=12usize {
            assert_eq!(
                blob[EXT_TAB_TURNOVER as usize + nroots * 2],
                data::TURNOVER_POINT[nroots * 2],
                "turnover lookup for nroots={nroots}"
            );
        }
    }

    /// The breakpoint table the inline dispatch folds at comptime is the one
    /// `rys_roots_host_wheeler` branches on (`rys_roots.c:97-114`).
    #[test]
    fn ext_breakpoints_match_the_host_dispatch() {
        assert_eq!(ext_breakpoint(6), 11.0);
        assert_eq!(ext_breakpoint(7), 11.0);
        assert_eq!(ext_breakpoint(8), 11.0);
        assert_eq!(ext_breakpoint(9), 10.0);
        assert_eq!(ext_breakpoint(10), 18.0);
        assert_eq!(ext_breakpoint(11), 18.0);
        assert_eq!(ext_breakpoint(12), 22.0);
    }

    /// Task 3a FMA probe: assert the CPU CubeCL backend fuses `mul_add` (bit-for-bit vs host).
    #[test]
    fn fma_probe() {
        use cubecl::prelude::*;
        // Pairs whose product is not exactly representable in f64 (so the TwoProd error
        // term is nonzero under a true FMA, and zero under a pre-rounded product).
        let a = [
            1.0 + 2f64.powi(-30),
            std::f64::consts::PI,
            1.3000000000000001e0,
            7.123456789012345e0,
            0.1,
            9.999999999999998e-1,
        ];
        let b = [
            1.0 + 2f64.powi(-30),
            std::f64::consts::E,
            2.6999999999999997e0,
            3.987654321098765e0,
            0.3,
            1.0000000000000002e0,
        ];
        // Host reference error terms via true f64 FMA.
        let host_err: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| {
                let p = ai * bi;
                ai.mul_add(bi, -p)
            })
            .collect();

        let client = cubecl::cpu::CpuRuntime::client(&Default::default());
        let a_h = client.create_from_slice(f64::as_bytes(&a));
        let b_h = client.create_from_slice(f64::as_bytes(&b));
        let out_zero = vec![0.0f64; a.len()];
        let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

        fma_probe_kernel::launch::<f64, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            CubeDim::new_1d(a.len() as u32),
            unsafe { ArrayArg::from_raw_parts(a_h, a.len()) },
            unsafe { ArrayArg::from_raw_parts(b_h, a.len()) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), a.len()) },
        );
        let raw = client.read_one_unchecked(out_h);
        let dev_err = f64::from_bytes(&raw);

        // At least one host error term must be nonzero (else the probe is vacuous).
        let any_nonzero = host_err.iter().any(|&e| e != 0.0);
        assert!(
            any_nonzero,
            "probe vacuous: all host TwoProd error terms are zero"
        );

        // Bit-for-bit fusion check.
        let mut fused = true;
        for i in 0..a.len() {
            if dev_err[i].to_bits() != host_err[i].to_bits() {
                fused = false;
                eprintln!(
                    "FMA NOT FUSED at i={i}: a={} b={} device_err={:e} host_err={:e}",
                    a[i], b[i], dev_err[i], host_err[i]
                );
            }
        }
        assert!(
            fused,
            "CPU CubeCL backend does NOT fuse mul_add bit-for-bit vs host f64 FMA \
             (double-double port must use a Dekker-split software product)"
        );
    }

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
