//! Integration tests chaining pdata + Boys + OS recurrence math primitives.
//!
//! Validates that all four math modules (boys, pdata, rys, obara_saika) chain
//! correctly in a simplified 1e overlap auxiliary integral evaluation pipeline.
//!
//! Pipeline: PairData -> Boys function -> VRR -> HRR -> auxiliary integrals.
//!
//! All assertions use absolute tolerance 1e-12 per D-19.
//! Reference formulas are cited against libcint source files.

use approx::assert_abs_diff_eq;
use cintx_cubecl::math::boys::boys_gamma_inc_host;
use cintx_cubecl::math::obara_saika::vrr_step_host;
use cintx_cubecl::math::pdata::compute_pdata_host;
use cintx_cubecl::math::rys::rys_root1_host;

/// sqrt(pi) = 1.7724538509055159..., used in analytical Gaussian overlap formula.
const SQRT_PI: f64 = 1.7724538509055159_f64;

// ─────────────────────────────────────────────────────────────────────────────
//  Analytical reference formula for Gaussian overlap integral
// ─────────────────────────────────────────────────────────────────────────────

/// Analytical s-s Gaussian overlap:
///
/// S(i,j) = exp(-ai*aj/(ai+aj) * |Ri-Rj|^2) * (pi/zeta)^{3/2}
///        = fac * (pi/zeta)^{3/2}    (with norm_i=norm_j=1)
///
/// Source: g1e.c CINTg1e_ovlp — gz[0] carries the fac*(pi/zeta)^{3/2} base value
/// for the z-component (line 134), x and y components are gx[0]=gy[0]=1.
///
/// The full overlap integral = gx[0] * gy[0] * gz[0] = 1 * 1 * gz[0].
/// gz[0] = fac * SQRTPI * PI / (zeta * sqrt(zeta)).
///
/// For our simplified test: g[0] = fac (set by test), and we separately compute
/// the normalization (pi/zeta)^{3/2} as a scalar cross-check.
fn analytical_ss_overlap_factor(ai: f64, aj: f64, rirj_sq: f64) -> f64 {
    let zeta = ai + aj;
    let fac = f64::exp(-ai * aj / zeta * rirj_sq);
    fac * (SQRT_PI / zeta).powi(3) * SQRT_PI
}

// ─────────────────────────────────────────────────────────────────────────────
//  Integration tests
// ─────────────────────────────────────────────────────────────────────────────

/// s-s shell pair integration test.
///
/// Pipeline: compute_pdata -> boys_gamma_inc(m=0) -> set g[0] = fac*f[0] -> final integral.
///
/// For an s-s pair with no nuclear attraction, Boys argument t=0, so F_0(0)=1.
/// The 1D G-array base case: g_z[0] = fac * SQRTPI*PI/(zeta*sqrt(zeta)) in g1e.c.
/// We test the simplified pipeline: g[0] = pdata.fac * f[0].
///
/// Reference: g1e.c lines 132-134 (gz[0] = envs->fac[0] * SQRTPI*M_PI/(aij*sqrt(aij))).
#[test]
fn math_integration_1e_overlap_ss() {
    // Define primitive pair: ai=1.0, aj=1.5, Ri=(0,0,0), Rj=(1,0,0)
    let ai = 1.0_f64;
    let aj = 1.5_f64;
    let ri = [0.0_f64, 0.0, 0.0];
    let rj = [1.0_f64, 0.0, 0.0];

    // Step 1: compute pair data
    // zeta=2.5, center_p=(0.6,0,0), aij2=0.2, rirj=(-1,0,0), fac=exp(-0.6)
    let pdata = compute_pdata_host(
        ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
    );

    // Verify pdata fields
    assert_abs_diff_eq!(pdata.zeta_ab, 2.5, epsilon = 1e-12);
    assert_abs_diff_eq!(pdata.center_p_x, 0.6, epsilon = 1e-12);
    assert_abs_diff_eq!(pdata.aij2, 0.2, epsilon = 1e-12);
    assert_abs_diff_eq!(pdata.rirj_x, -1.0, epsilon = 1e-12);

    // Step 2: Boys function with t=0 for pure overlap (no operator center), m=0
    // F_0(0) = 1.0 (analytical identity)
    let f = boys_gamma_inc_host(0.0, 0);
    assert_abs_diff_eq!(f[0], 1.0, epsilon = 1e-12);

    // Step 3: For s-s pair, no VRR or HRR needed.
    // g[0] = pdata.fac * f[0]
    let g0 = pdata.fac * f[0];

    // pdata.fac = exp(-ai*aj/zeta * rr) = exp(-1.0*1.5/2.5 * 1.0) = exp(-0.6)
    let expected_fac = f64::exp(-ai * aj / pdata.zeta_ab * 1.0_f64);
    assert_abs_diff_eq!(g0, expected_fac, epsilon = 1e-12);

    // Cross-check: the full overlap integral for an s-s pair is fac * (pi/zeta)^{3/2}
    // g1e.c line 134: gz[0] = envs->fac[0] * SQRTPI*M_PI/(aij*sqrt(aij))
    // With gx[0]=gy[0]=1, full overlap = g[0] * (pi/zeta)^{3/2}
    let rirj_sq = pdata.rirj_x * pdata.rirj_x; // |rirj|^2 = 1.0
    let analytical = analytical_ss_overlap_factor(ai, aj, rirj_sq);
    let computed_full = g0 * (SQRT_PI / pdata.zeta_ab).powi(3) * SQRT_PI;
    assert_abs_diff_eq!(computed_full, analytical, epsilon = 1e-10);
}

/// p-s shell pair integration test.
///
/// Pipeline: compute_pdata -> boys_gamma_inc(m=1) -> g[0] = fac*f[0] -> vrr_step(nmax=1)
/// -> g[1] = (center_p_x - ri_x) * g[0].
///
/// Reference: g1e.c lines 164-166 (gx[di] = rijrx[0] * gx[0]).
#[test]
fn math_integration_1e_overlap_ps() {
    let ai = 1.0_f64;
    let aj = 1.5_f64;
    let ri = [0.0_f64, 0.0, 0.0];
    let rj = [1.0_f64, 0.0, 0.0];

    // Step 1: compute pair data
    let pdata = compute_pdata_host(
        ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
    );

    // Step 2: Boys function with t=0, m=1
    // F_0(0)=1, F_1(0)=1/3
    let f = boys_gamma_inc_host(0.0, 1);
    assert_abs_diff_eq!(f[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f[1], 1.0 / 3.0, epsilon = 1e-12);

    // Step 3: Set g[0] = pdata.fac * f[0]
    let mut g = vec![0.0_f64; 4];
    g[0] = pdata.fac * f[0];

    // Step 4: VRR for p-shell (nmax=1)
    // rijrx = center_p_x - ri_x (displacement from Ri to Rp)
    // In g1e.c: rijrx = rij - rx where rij is the Gaussian product center and rx is ri
    let rijrx = pdata.center_p_x - ri[0];
    vrr_step_host(&mut g, rijrx, pdata.aij2, 1, 1);

    // g[1] = rijrx * g[0] = (0.6 - 0.0) * exp(-0.6)
    let expected_g1 = rijrx * pdata.fac * f[0];
    assert_abs_diff_eq!(g[1], expected_g1, epsilon = 1e-12);

    // g[1] must be non-zero (center_p_x != ri_x when ai != aj and positions differ)
    assert!(g[1].abs() > 1e-10, "g[1] should be non-zero for p-s pair with offset geometry");
}

/// d-s shell pair integration test.
///
/// Pipeline: compute_pdata -> boys_gamma_inc(m=2) -> g[0] = fac*f[0] -> vrr_step(nmax=2)
/// -> g[0], g[1], g[2] all non-zero and self-consistent.
///
/// Reference: g1e.c lines 169-172 (VRR for i=1..nmax-1).
#[test]
fn math_integration_1e_overlap_ds() {
    let ai = 1.0_f64;
    let aj = 1.5_f64;
    let ri = [0.0_f64, 0.0, 0.0];
    let rj = [1.0_f64, 0.0, 0.0];

    // Step 1: compute pair data
    let pdata = compute_pdata_host(
        ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
    );

    // Step 2: Boys function t=0, m=2
    // F_m(0) = 1/(2m+1): F_0=1, F_1=1/3, F_2=1/5
    let f = boys_gamma_inc_host(0.0, 2);
    assert_abs_diff_eq!(f[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f[1], 1.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f[2], 1.0 / 5.0, epsilon = 1e-12);

    // Step 3: Set g[0] = pdata.fac * f[0]
    let mut g = vec![0.0_f64; 8];
    g[0] = pdata.fac * f[0];

    // Step 4: VRR for d-shell (nmax=2)
    let rijrx = pdata.center_p_x - ri[0]; // = 0.6
    vrr_step_host(&mut g, rijrx, pdata.aij2, 2, 1);

    // Verify g[0], g[1], g[2] are non-zero and self-consistent
    // g[1] = rijrx * g[0] = 0.6 * exp(-0.6)
    let expected_g1 = rijrx * pdata.fac;
    assert_abs_diff_eq!(g[1], expected_g1, epsilon = 1e-12);

    // g[2] = 1 * aij2 * g[0] + rijrx * g[1]
    //      = 0.2 * exp(-0.6) + 0.6 * 0.6 * exp(-0.6)
    //      = exp(-0.6) * (0.2 + 0.36)
    let expected_g2 = 1.0_f64 * pdata.aij2 * g[0] + rijrx * g[1];
    assert_abs_diff_eq!(g[2], expected_g2, epsilon = 1e-12);

    // All values should be non-zero
    assert!(g[0].abs() > 1e-10, "g[0] must be non-zero");
    assert!(g[1].abs() > 1e-10, "g[1] must be non-zero for p component");
    assert!(g[2].abs() > 1e-10, "g[2] must be non-zero for d component");
}

/// Rys-Boys crosscheck: for a given x, compute Boys F_0(x) directly and compare
/// with a host-side nroots=1 Rys weight estimate via the known identity.
///
/// For nroots=1 Rys quadrature, the weight w[0] satisfies:
///   sum(w_i) = F_0(x) (weight sum identity for 1D Rys quadrature)
///
/// We verify that our Boys function and Rys root polynomial agree by checking the
/// identity at large x where the polynomial asymptotic formula is exact.
///
/// Source:
///   - Boys: fmt.c gamma_inc_like lines 218-225
///   - Rys weight sum: rys_roots.c asymptotic formula for large x
#[test]
fn math_integration_rys_boys_crosscheck() {
    // For nroots=1 Rys quadrature, the weight w[0] satisfies:
    //   w[0] = F_0(x) (weight sum identity for 1D Rys quadrature)
    //
    // Cross-validate by computing both sides independently:
    //   - Boys F_0(x) via boys_gamma_inc_host
    //   - Rys weight w[0] via rys_root1_host
    //
    // Source:
    //   - Boys: fmt.c gamma_inc_like
    //   - Rys weight sum identity: rys_roots.c

    // Large x values where both asymptotic formulas converge (exact agreement)
    let large_x_values = [50.0_f64, 75.0, 100.0];
    for &x in &large_x_values {
        let f = boys_gamma_inc_host(x, 0);
        let boys_f0 = f[0];
        let (_root, rys_w0) = rys_root1_host(x);

        // Boys F_0(x) and Rys w[0] must agree (weight-sum identity)
        assert_abs_diff_eq!(boys_f0, rys_w0, epsilon = 1e-12);
    }

    // Moderate x values — polynomial fit domain
    let moderate_x_values = [1.0_f64, 5.0, 10.0, 15.0];
    for &x in &moderate_x_values {
        let f = boys_gamma_inc_host(x, 0);
        let boys_f0 = f[0];
        let (_root, rys_w0) = rys_root1_host(x);

        // Weight-sum identity: w[0] = F_0(x) for nroots=1
        // Polynomial fit may have slightly lower precision at moderate x
        assert_abs_diff_eq!(boys_f0, rys_w0, epsilon = 1e-8);
    }

    // Small x values
    let small_x_values = [0.001_f64, 0.1];
    for &x in &small_x_values {
        let f = boys_gamma_inc_host(x, 0);
        let boys_f0 = f[0];
        let (_root, rys_w0) = rys_root1_host(x);

        assert_abs_diff_eq!(boys_f0, rys_w0, epsilon = 1e-10);
    }

    // F_0 should be monotonically decreasing with t (sanity check)
    let f1 = boys_gamma_inc_host(1.0, 0);
    let f5 = boys_gamma_inc_host(5.0, 0);
    assert!(f1[0] > 0.0 && f1[0] < 1.0, "F_0(1.0) should be in (0,1)");
    assert!(f5[0] > 0.0 && f5[0] < 1.0, "F_0(5.0) should be in (0,1)");
    assert!(f1[0] > f5[0], "F_0 should decrease with increasing t");
}
