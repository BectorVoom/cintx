//! D-PBC-24 stage 3 — `CINTsr_rys_roots` parity: the lower-bounded quadrature.
//!
//! Short-range Coulomb at `rys_order > 3` cannot be written as "full minus long
//! range": the doubled-root trick libcint uses below that (`g2e.c:4480-4491`) is
//! only valid while `rys_order <= 3`. Above it the nodes and weights of
//! `∫_lower^1 u^{2m} e^{-t u²} du` must be computed directly
//! (`libcint-master/src/rys_roots.c:145`), with `lower = sqrt(theta)` and
//! `theta = ω²/(ω² + a0)`.
//!
//! That dispatch is threshold-driven and accuracy-critical: it switches solver
//! at `lower ∈ {0.15, 0.25, 0.4, 0.5, 0.8, 0.9, 0.93, 0.97, 0.99}` and at
//! `x ∈ {10, 50, 60}`. This file sweeps ACROSS every one of those thresholds.
//!
//! # What is compared, and why it is not the raw nodes
//!
//! Two Gauss rules for the same measure are mathematically unique, but as
//! `lower → 1` the interval `[lower, 1]` becomes a sliver and the rule becomes
//! badly conditioned: individual nodes disagree wildly while carrying weights
//! that cannot move any integral. So the comparison is on the KERNEL-SHAPED
//! functionals `Σ_k w_k · u_k^a / (1 + u_k)^b` — the exact shape `CINTg0_2e`
//! consumes, since `tmp4 = 0.5/(u2*(aij+akl) + a1)` reduces to `0.5/(a1(1+u))`
//! and every `b00`/`b10`/`b01`/`c00`/`c0p` term is a polynomial in `u/(1+u)`
//! and `1/(1+u)` (`g2e.c:4519-4536`).
//!
//! # Where the REFERENCE stops being a reference
//!
//! Past roughly `nroots >= 8` with `lower > 0.99`, libcint's own solver breaks
//! down: it returns zero-padded weights, negative roots below the `lower²/(1−lower²)`
//! floor, and `err == 0` regardless. Cross-checked against a 60-digit mpmath
//! Gauss rule for `(nroots, x, lower) = (12, 11, 0.999)`, where the true nodes
//! are all in `[505, 4.1e4]` with weights `~1e-9` and libcint returns two
//! negative roots, seven exact zeros and nothing matching. Those points are
//! skipped by [`reference_is_a_valid_gauss_rule`] rather than silently
//! tolerated — gating cintx on them would be gating it on noise, and the
//! `assert` below pins how many are skipped so the envelope cannot quietly grow.
//!
//! Requires `CINTX_ORACLE_BUILD_VENDOR=1`; without it the file compiles to
//! nothing, so the suite still builds.

#![cfg(has_vendor_libcint)]

use cintx_cubecl::math::rys_wheeler::sr_rys_roots_host;
use cintx_oracle::vendor_ffi::vendor_CINTsr_rys_roots;

/// Every `lower` dispatch threshold in `CINTsr_rys_roots`, straddled from both
/// sides so each solver arm is entered (`rys_roots.c:145-244`).
const LOWER_SWEEP: [f64; 13] = [
    1e-8, 0.05, 0.15, 0.25, 0.4, 0.5, 0.8, 0.89, 0.9, 0.92, 0.95, 0.98, 0.999,
];

/// Straddles the `x` breakpoints (10, 50, 60) that pick Jacobi vs Laguerre vs
/// Schmidt inside each `lower` arm.
const X_SWEEP: [f64; 13] = [
    1e-6, 0.01, 0.5, 3.0, 9.0, 10.0, 11.0, 30.0, 49.0, 50.0, 51.0, 60.0, 61.0,
];

/// `Σ_k w_k · u_k^a / (1 + u_k)^b` — see the module docs.
fn kernel_functional(u: &[f64], w: &[f64], a: u32, b: u32) -> f64 {
    u.iter()
        .zip(w)
        .map(|(&uu, &ww)| ww * uu.powi(a as i32) / (1.0 + uu).powi(b as i32))
        .sum()
}

/// Whether the reference output is a valid Gauss rule for `∫_lower^1`.
///
/// Every weight of a Gauss rule with a positive measure is strictly positive,
/// and every node lies inside the support — which in libcint's `u = s/(1−s)`
/// variable means `u >= lower²/(1 − lower²)`. libcint violates both in its
/// breakdown regime while still reporting success.
fn reference_is_a_valid_gauss_rule(u: &[f64], w: &[f64], lower: f64) -> bool {
    let l2 = lower * lower;
    if l2 >= 1.0 {
        return false;
    }
    let u_min = l2 / (1.0 - l2);
    w.iter().all(|&v| v > 0.0) && u.iter().all(|&v| v >= u_min * (1.0 - 1e-9))
}

/// cintx's lower-bounded quadrature against libcint's, over every dispatch arm.
#[test]
fn sr_rys_roots_matches_vendored_libcint_across_every_dispatch_threshold() {
    let mut compared = 0usize;
    let mut skipped_invalid_reference = 0usize;
    let mut worst = 0.0_f64;
    let mut worst_at = String::new();

    for nroots in 1..=12usize {
        // nroots 11 and 12 are the top of the host Rys range and the reference
        // is already losing digits there even where it stays a valid rule.
        let tol = if nroots >= 11 { 1e-7 } else { 1e-9 };

        for &lower in LOWER_SWEEP.iter() {
            for &x in X_SWEEP.iter() {
                // THE CALL DOMAIN. `CINTg0_2e` contributes nothing at all once
                // `theta * x` passes `EXPCUTOFF_SR = 40` (`g2e.c:4459-4461`),
                // and `lower` IS `sqrt(theta)` — so `x * lower² > 40` is
                // unreachable from the integral path. libcint's own comment
                // says why: "short-range Coulomb kernel is numerically very
                // instable when the integrals are close to zero
                // (x*lower**2 > 40)" (`rys_roots.h:42-45`).
                if x * lower * lower > 40.0 {
                    continue;
                }

                let (want_u, want_w, err) = vendor_CINTsr_rys_roots(nroots as i32, x, lower);
                if err != 0 || !reference_is_a_valid_gauss_rule(&want_u, &want_w, lower) {
                    skipped_invalid_reference += 1;
                    continue;
                }
                let scale: f64 = want_w.iter().map(|v| v.abs()).sum();
                if scale == 0.0 {
                    skipped_invalid_reference += 1;
                    continue;
                }

                let (got_u, got_w) = sr_rys_roots_host(nroots, x, lower).unwrap_or_else(|code| {
                    panic!(
                        "sr_rys_roots_host failed (code {code}) at nroots={nroots} x={x} \
                         lower={lower}, where libcint returned a valid rule"
                    )
                });

                for b in 1..=(2 * nroots as u32) {
                    for a in 0..=b {
                        let want = kernel_functional(&want_u, &want_w, a, b);
                        let got = kernel_functional(&got_u, &got_w, a, b);
                        let diff = (got - want).abs() / scale;
                        if diff > worst {
                            worst = diff;
                            worst_at = format!("nroots={nroots} x={x} lower={lower} a={a} b={b}");
                        }
                        assert!(
                            diff <= tol,
                            "kernel functional mismatch nroots={nroots} x={x} lower={lower} \
                             a={a} b={b}: cintx={got:.17e} libcint={want:.17e} \
                             scaled diff={diff:.3e}"
                        );
                    }
                }
                compared += 1;
            }
        }
    }

    assert!(
        compared > 900,
        "the lower/x sweep must actually exercise the dispatch, got {compared} points"
    );
    // Pins the reference's own breakdown envelope: if a future libcint bump or a
    // sweep change moves this a lot, it should be looked at, not absorbed.
    assert!(
        skipped_invalid_reference < 600,
        "libcint returned an invalid Gauss rule at {skipped_invalid_reference} points — \
         far more than the ~420 known high-nroots/lower→1 breakdown cases"
    );
    eprintln!(
        "sr_rys_roots parity: {compared} points compared, {skipped_invalid_reference} skipped \
         (libcint itself not a valid rule); worst scaled functional diff = {worst:.3e} \
         at {worst_at}"
    );
}

/// `lower == 0` is unreachable from `CINTg0_2e` (`theta > 0` strictly whenever
/// ω ≠ 0), and `sr_rys_roots_host` documents that it delegates to the ordinary
/// full-range engine there rather than to the lower-bounded arms. Pin that
/// contract, so the delegation cannot be dropped without a test noticing.
///
/// The `x` grid stays inside cintx's validated full-range envelope: its host
/// Wheeler dispatch deliberately does not carry `CINTrys_roots`' small-`x`
/// polynomial regime (`rys_roots.c:58-66`), so `x < ~1e-5` at high `nroots` is
/// out of scope for that engine — a pre-existing limit, unrelated to D-PBC-24,
/// and unreachable through the delegation this test pins.
#[test]
fn lower_zero_delegates_to_the_full_range_engine() {
    for nroots in 1..=12usize {
        for &x in &[0.01_f64, 0.5, 3.0, 11.0, 30.0, 60.0] {
            let (got_u, got_w) = sr_rys_roots_host(nroots, x, 0.0)
                .unwrap_or_else(|code| panic!("sr_rys_roots_host(lower=0) failed: {code}"));
            let (want_u, want_w) = cintx_cubecl::math::rys::rys_roots_host::<f64>(nroots, x);
            assert_eq!(got_u, want_u, "roots at nroots={nroots} x={x}");
            assert_eq!(got_w, want_w, "weights at nroots={nroots} x={x}");
        }
    }
}
