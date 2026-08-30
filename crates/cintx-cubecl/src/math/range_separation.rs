//! The `CINTg0_2e` range-separation prologue — one implementation, three callers.
//!
//! # D-PBC-24
//!
//! `libcint-master/src/g2e.c:4443-4512` is the ONLY place upstream branches on
//! `env[PTR_RANGE_OMEGA]`, and `CINTg0_2e` is shared verbatim by `int2e`
//! (`g2e.c:171`), `int3c2e` (`g3c2e.c:131`) and `int2c2e` (`g2c2e.c:104`). cintx
//! carries three separate copies of that prologue — `two_electron.rs`'s
//! `fill_g_tensor_2e`, `center_3c2e.rs`'s `fill_g_tensor_3c2e` and
//! `center_2c2e.rs`'s `fill_g_tensor_2c2e` — so the omega branch lives HERE and
//! all three call it. A wrong ω does not fail loudly, it produces a plausible
//! 1e-6; three copies of it would be three chances to drift.
//!
//! ## The three branches (`g2e.c:4444-4512`)
//!
//! ```text
//! a0    = aij*akl/(aij+akl)
//! fac1  = sqrt(a0/(aij*akl)^3) * fac
//! x     = a0 * rr
//!
//! omega == 0   →  rys_roots(nroots, x)
//!
//! omega  > 0   →  theta = ω²/(ω² + a0)                        long range
//!                 x    *= theta
//!                 fac1 *= sqrt(theta)
//!                 rys_roots(nroots, x)
//!                 for every root:  ut = u*theta;  u = ut/(u + 1 − ut)
//!
//! omega  < 0   →  theta = ω²/(ω² + a0)                        short range
//!                 if theta*x > EXPCUTOFF_SR(=40): contribute nothing
//!                 if rys_order == nroots:  sr_rys_roots(nroots, x, sqrt(theta))
//!                 else (rys_order <= 3, nroots = 2*rys_order):
//!                     rys_roots(rorder, x)        → u[0..rorder],  w[0..rorder]
//!                     rys_roots(rorder, theta*x)  → u[rorder..],   w[rorder..]
//!                     for irys in rorder..nroots:
//!                         ut = u*theta;  u = ut/(u + 1 − ut)
//!                         w *= −sqrt(theta)
//! ```
//!
//! The short-range doubled-root arm is "full minus long range" done at the ROOT
//! level under one shared `fac1`, so the cancellation at small `ω·r` is between
//! quadrature weights rather than between two separately-rounded integrals. It
//! is why short range on every system Phase 14 gates needs no new quadrature at
//! all — see [`super::super::math`] callers and D-PBC-24 §3.4.
//!
//! ## What is deliberately NOT ported (D-PBC-24 §3.5)
//!
//! * The `theta * x > cutoff` half of the early return. `cutoff` is
//!   `expcutoff - pdata_ij->cceij` (`cint2e.c:239`, `cint3c2e.c:186`), the
//!   per-primitive-pair screening budget — and cintx's 2e/3c2e/2c2e kernels
//!   implement no `cceij` screening at all, for the full-range operator either.
//!   Skipping it keeps MORE primitives than upstream, which is the conservative
//!   direction. `EXPCUTOFF_SR` itself IS ported: it is a numerical-stability
//!   guard, not an optimisation.
//! * `cint3c2e.c:108-124` / `optimizer.c:306-315`, which LOOSEN `expcutoff`
//!   when `omega < 0`. Same reasoning: they only matter to a screener cintx
//!   does not have.
//!
//! ## `CINTsr_rys_roots` — the `rys_order > 3` arm (stage 3)
//!
//! Above `rys_order = 3` the doubled-root trick does not apply and short range
//! needs the genuinely lower-bounded quadrature `∫_lower^1`
//! (`rys_roots.c:145`), with `lower = sqrt(theta)`. That family lives in
//! [`crate::math::rys_wheeler::sr_rys_roots_host`]. A solver failure there is
//! surfaced as a typed stop; it NEVER falls through to the full-range kernel.

use cintx_core::cintxRsError;
use cintx_runtime::range_omega::EXPCUTOFF_SR;

use crate::math::rys::rys_roots_host;

/// Rys roots and weights for one primitive tuple, after the omega branch.
///
/// `fac1` is returned alongside because the long-range arm folds
/// `sqrt(theta)` into it (`g2e.c:4494`) rather than into the weights.
/// Callers apply it exactly as they did before: `gz[irys] = w[irys] * fac1`.
#[derive(Clone, Debug)]
pub struct RangeSeparatedRoots {
    pub u: Vec<f64>,
    pub w: Vec<f64>,
    pub fac1: f64,
}

/// The `CINTg0_2e` omega prologue (`g2e.c:4444-4512`).
///
/// * `rys_order` — `(Σ l_ceil)/2 + 1`, BEFORE any short-range doubling.
/// * `nroots` — the doubled count where it applies, i.e.
///   [`cintx_runtime::range_omega::nrys_roots_for`]`(rys_order, range_omega)`.
///   Callers must size their G-tensor strides from this same value.
/// * `x` — `a0 * rr`, the plain full-range Rys argument.
/// * `a0` — `aij*akl/(aij+akl)`.
/// * `fac1` — `sqrt(a0/(aij*akl)^3) * fac`, the plain full-range prefactor.
///
/// Returns `Ok(None)` when the short-range integrand is below
/// `EXPCUTOFF_SR` and libcint would return 0 for this primitive (`g2e.c:4460`)
/// — the caller must contribute nothing, not zeros-times-something.
pub fn rys_roots_range_separated(
    rys_order: usize,
    nroots: usize,
    x: f64,
    a0: f64,
    fac1: f64,
    range_omega: Option<f64>,
) -> Result<Option<RangeSeparatedRoots>, cintxRsError> {
    let omega = range_omega.unwrap_or(0.0);

    if omega == 0.0 {
        let (u, w) = rys_roots_host::<f64>(nroots, x);
        return Ok(Some(RangeSeparatedRoots { u, w, fac1 }));
    }

    let theta = omega * omega / (omega * omega + a0);

    if omega > 0.0 {
        // Long range, erf(ω r)/r (g2e.c:4493-4512).
        let x = x * theta;
        let fac1 = fac1 * theta.sqrt();
        let (mut u, w) = rys_roots_host::<f64>(nroots, x);
        // u[:] = tau^2/(1 - tau^2); transform to theta^-1 tau^2/(theta^-1 - tau^2)
        // so the rest of the recurrence is reused unchanged.
        for value in u.iter_mut().take(nroots) {
            let ut = *value * theta;
            *value = ut / (*value + 1.0 - ut);
        }
        return Ok(Some(RangeSeparatedRoots { u, w, fac1 }));
    }

    // Short range, erfc(|ω| r)/r (g2e.c:4468-4492).
    //
    // "very small erfc() leads to ~0 weights. They can cause numerical issue in
    // sr_rys_roots" — g2e.c:4457-4458.
    if theta * x > EXPCUTOFF_SR {
        return Ok(None);
    }

    if rys_order == nroots {
        // rys_order > 3: the doubled-root trick does not apply and the integral
        // needs the genuinely lower-bounded quadrature ∫_{sqrt(theta)}^1
        // (`CINTsr_rys_roots`, rys_roots.c:145). D-PBC-24 stage 3.
        //
        // A solver failure is surfaced as a typed stop, NEVER as a fall-through
        // to the full-range kernel: that would run, converge, and be silently a
        // different method.
        let (u, w) = crate::math::rys_wheeler::sr_rys_roots_host(nroots, x, theta.sqrt()).map_err(
            |code| cintxRsError::ChunkPlanFailed {
                from: "sr_rys_roots",
                detail: format!(
                    "CINTsr_rys_roots failed (code {code}) for nroots={nroots} x={x} \
                     lower={lower}; refusing rather than substituting the full-range kernel",
                    lower = theta.sqrt()
                ),
            },
        )?;
        return Ok(Some(RangeSeparatedRoots { u, w, fac1 }));
    }

    debug_assert_eq!(
        nroots,
        rys_order * 2,
        "short range below rys_order 4 must have doubled roots"
    );

    let sqrt_theta = -theta.sqrt();
    let (mut u, mut w) = rys_roots_host::<f64>(rys_order, x);
    let (u_hi, w_hi) = rys_roots_host::<f64>(rys_order, theta * x);
    u.extend_from_slice(&u_hi[..rys_order]);
    w.extend_from_slice(&w_hi[..rys_order]);

    for irys in rys_order..nroots {
        let ut = u[irys] * theta;
        u[irys] = ut / (u[irys] + 1.0 - ut);
        w[irys] *= sqrt_theta;
    }

    Ok(Some(RangeSeparatedRoots { u, w, fac1 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(rys_order: usize, x: f64, a0: f64, omega: Option<f64>) -> RangeSeparatedRoots {
        let nroots = cintx_runtime::range_omega::nrys_roots_for(rys_order, omega);
        rys_roots_range_separated(rys_order, nroots, x, a0, 1.0, omega)
            .expect("supported")
            .expect("not screened out")
    }

    /// `w` here is the raw Rys weight; `Σ w` at `x = 0` is the Boys-function
    /// normalisation. The identity that actually matters is that the doubled
    /// short-range block equals full minus long range term by term, which is
    /// what the caller's `Σ_r gx*gy*gz` reduces to for an (s,s|s,s) tuple.
    #[test]
    fn sr_weights_are_full_minus_lr_for_an_ssss_tuple() {
        let a0 = 0.7_f64;
        let x = 1.3_f64;
        let omega = 0.9_f64;

        let full = roots(1, x, a0, None);
        let lr = roots(1, x, a0, Some(omega));
        let sr = roots(1, x, a0, Some(-omega));

        // An (s,s|s,s) integral is fac1 * Σ_r w[r] (gx = gy = 1).
        let full_sum: f64 = full.w.iter().sum::<f64>() * full.fac1;
        let lr_sum: f64 = lr.w.iter().sum::<f64>() * lr.fac1;
        let sr_sum: f64 = sr.w.iter().sum::<f64>() * sr.fac1;

        let residual = (sr_sum + lr_sum - full_sum).abs();
        assert!(
            residual <= 1e-14 * full_sum.abs().max(1.0),
            "SR + LR must reproduce the full Coulomb integral: \
             sr={sr_sum:.17e} lr={lr_sum:.17e} full={full_sum:.17e} residual={residual:.3e}"
        );
    }

    #[test]
    fn sr_doubles_the_roots_and_negates_the_long_range_half() {
        let sr = roots(2, 0.9, 0.6, Some(-0.7));
        assert_eq!(sr.u.len(), 4);
        assert_eq!(sr.w.len(), 4);
        assert!(
            sr.w[0] > 0.0 && sr.w[1] > 0.0,
            "full-range half stays positive"
        );
        assert!(
            sr.w[2] < 0.0 && sr.w[3] < 0.0,
            "long-range half is subtracted"
        );
    }

    #[test]
    fn long_range_never_changes_the_root_count() {
        let lr = roots(3, 2.0, 0.5, Some(0.4));
        assert_eq!(lr.u.len(), 3);
        assert_eq!(lr.w.len(), 3);
    }

    /// Above `rys_order = 3` short range takes the lower-bounded quadrature
    /// arm. The identity `SR + LR == full` still has to hold there — it is the
    /// only check that does not depend on trusting `CINTsr_rys_roots` itself.
    #[test]
    fn sr_rys_roots_arm_still_satisfies_sr_plus_lr_equals_full() {
        let a0 = 0.7_f64;
        let omega = 0.9_f64;

        for rys_order in 4..=6 {
            for &x in &[0.05_f64, 1.3, 9.0, 25.0] {
                let full = roots(rys_order, x, a0, None);
                let lr = roots(rys_order, x, a0, Some(omega));
                let sr = roots(rys_order, x, a0, Some(-omega));
                assert_eq!(
                    sr.u.len(),
                    rys_order,
                    "the SR arm must not double the roots"
                );

                // ∫_0^1 w du = ∫_lower^1 + ∫_0^lower, and the Rys weights are
                // exactly those integrals under the shared fac1.
                let full_sum: f64 = full.w.iter().sum::<f64>() * full.fac1;
                let lr_sum: f64 = lr.w.iter().sum::<f64>() * lr.fac1;
                let sr_sum: f64 = sr.w.iter().sum::<f64>() * sr.fac1;
                let residual = (sr_sum + lr_sum - full_sum).abs();
                // Unlike the doubled-root arm — which computes SR as "full minus
                // long range" over one shared `fac1` and so closes this identity
                // to round-off — the `sr_rys_roots` arm builds an INDEPENDENT
                // lower-bounded Gauss rule, so the identity closes to that
                // rule's accuracy. Measured at ~3e-12 relative; the end-to-end
                // vendor gate is `cintx-oracle/tests/range_omega_parity.rs`.
                assert!(
                    residual <= 1e-10 * full_sum.abs().max(1e-12),
                    "rys_order={rys_order} x={x}: sr={sr_sum:.17e} lr={lr_sum:.17e} \
                     full={full_sum:.17e} residual={residual:.3e}"
                );
            }
        }
    }

    #[test]
    fn expcutoff_sr_screens_out_a_negligible_primitive() {
        // theta -> 1 for |omega| >> a0, so theta*x ~ x; pick x well past 40.
        let out =
            rys_roots_range_separated(1, 2, 1.0e3, 1.0e-6, 1.0, Some(-1.0e3)).expect("supported");
        assert!(
            out.is_none(),
            "theta*x > EXPCUTOFF_SR must contribute nothing (g2e.c:4460)"
        );
    }
}
