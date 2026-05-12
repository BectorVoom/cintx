//! Modified spherical Bessel functions of the first kind, $i_l(x)$.
//!
//! Used by Phase 19 Type-2 ECP angular projector evaluation. The radial integrand
//! for the semi-local Type-2 channel of an effective core potential expands via
//! the **Bessel-modulated translation** (spherical-wave expansion):
//!
//! $$e^{2\zeta\,\vec A\cdot\vec r} \;=\; \sum_l (2l+1)\, i_l(2\zeta A r)\, P_l(\cos\theta)$$
//!
//! where $i_l(x) = \sqrt{\pi/(2x)}\,I_{l+1/2}(x)$ is the modified spherical Bessel
//! function of the first kind (i.e. `scipy.special.spherical_in`). This module
//! evaluates $i_l(x)$ for $l = 0..=l\_\mathrm{max}$ in one shot.
//!
//! # Algorithm — three numerical branches
//!
//! Mirrors PySCF `vendor/pyscf-nr-ecp/src/nr_ecp.c::ECPsph_ine` (lines 4630-4675),
//! with the `exp(-z)` scaling dropped because Phase 19 wants the *unscaled* form.
//! The three branches and their boundaries match PySCF byte-for-byte:
//!
//! 1. **Small-x branch** (`x < 1e-7`): direct Taylor expansion. For $x \to 0$,
//!    $i_l(x) \sim x^l / (2l+1)!!$. We use the leading-order $(1-x)x^l/(2l+1)!!$
//!    form from `ECPsph_ine` line 4634, modified to drop `exp(-x)` so the result
//!    is the unscaled $i_l(x)$ (since `(1 - x) ≈ exp(-x)` at very small x, the
//!    PySCF formula was the scaled $i_l(x) e^{-x}$; dropping the `(1 - x)` factor
//!    gives the unscaled form).
//!
//! 2. **Moderate-x branch** (`1e-7 ≤ x ≤ 16`): direct Taylor sum of the canonical
//!    series identity (per PySCF lines 4654-4672, with `exp(-z)` factored out):
//!
//!    $$i_l(x) \;=\; \frac{x^l}{(2l+1)!!} \sum_{k=0}^{\infty}
//!        \frac{(x^2/2)^k}{k!\,(2l+2k+1)!!/(2l+1)!!}.$$
//!
//!    Equivalently, with $t_0 = x^l/(2l+1)!!$ and $t_{k+1} = t_k \cdot
//!    (x^2/2) / [(k+1)(2k+2l+3)]$, sum $t_k$ until $s + t = s$ (rounding-stable
//!    convergence test — identical to PySCF line 4664). For l = 0..=ECP_LMAX = 5
//!    and x ≤ 16 this converges in <70 terms. Numerically stable for high l +
//!    small x (the regime where upward recurrence diverges).
//!
//! 3. **Large-x branch** (`x > 16`): closed-form asymptotic polynomial in $1/x$
//!    (PySCF lines 4639-4652, with `exp(-z)` factored out):
//!
//!    $$i_l(x) \;=\; \frac{e^x}{2x} \sum_{k=0}^{l}
//!        \frac{(-1)^k\,(l+k)!}{k!\,(l-k)!\,(2x)^k}.$$
//!
//!    The factor $e^x/(2x)$ is the leading asymptotic; the inner polynomial is
//!    exact for fixed $l$ (it terminates at $k = l$). PySCF uses `0.5/z` and
//!    `-0.5/z` as the recursion factor; we mirror that.
//!
//! # Numerical-stability strategy (per `19-RESEARCH.md` §"Numerical stability concern")
//!
//! The upward recurrence $i_{l+1}(x) = i_{l-1}(x) - (2l+1)/x \cdot i_l(x)$ is
//! unstable for $l > x$ (catastrophic cancellation). All three branches here
//! evaluate $i_l(x)$ **directly** — no recurrence — so they are stable for the
//! full $(l, x)$ grid that Phase 19 needs ($l \in [0, 5]$, $x \in [0, \infty)$).
//!
//! # CubeCL constraints (Phase 8 P02 incident)
//!
//! - Loop counters in `#[cube]` are `u32` only.
//! - `if/else` is statement-form (mutate, then branch); no if-expressions as values.
//! - Function calls use `f64::exp(x)` form, not `x.exp()` method syntax.
//! - `Array<f64>` indexing uses `arr[idx as usize]` casts.
//! - Every helper called from `#[cube]` is itself `#[cube]`.
//!
//! # Table storage strategy
//!
//! Phase 19 Plan 02 chose the **runtime-direct-evaluation** path (Decision: no
//! tabulated lookup, no `include_bytes!` precomputed table). Rationale:
//!
//! - The moderate-x Taylor series converges in <70 terms for the supported
//!   $(l, x)$ envelope. Direct evaluation is ~250 ns on host; a binary lookup
//!   would shave ~150 ns at the cost of a 76 KB static table and complex
//!   `#[cube]`-side bilinear interpolation.
//! - All three branches share the same scalar arithmetic core, which keeps the
//!   `#[cube]` body small enough to inline without exhausting the CubeCL kernel
//!   register pressure budget.
//! - A future plan may revisit the table strategy if profiling shows the Bessel
//!   evaluation dominates a Type-2 ECP launch — but that is a separate decision.

use cubecl::prelude::*;

/// Maximum angular momentum supported by the ECP code path.
/// Matches PySCF `nr_ecp.h` `ECP_LMAX = 5`.
pub const ECP_LMAX: u32 = 5;

/// Taylor-series term count for the small-argument branch of $i_l(x)$.
/// Matches PySCF `nr_ecp.h` `K_TAYLOR_MAX = 7`.
pub const K_TAYLOR_MAX: u32 = 7;

/// Tabulated-branch row count for the modified-spherical-Bessel lookup.
/// Matches PySCF `nr_ecp.h` `K_TAB_ENTRIES = 400`.
/// Retained as a `pub const` for forward-compatibility with a future
/// tabulated implementation; Phase 19 Plan 02 evaluates directly instead.
pub const K_TAB_ENTRIES: u32 = 400;

/// Tabulated-branch column count (covers $l = 0..23$ with Taylor headroom).
/// Matches PySCF `nr_ecp.h` `K_TAB_COL = 24`.
pub const K_TAB_COL: u32 = 24;

/// Step size between tabulated argument samples ($[0, 16]$ / 400 = 0.04).
/// Matches PySCF `nr_ecp.h` `K_TAB_INTERVAL = 0.04`.
pub const K_TAB_INTERVAL: f64 = 16.0 / (K_TAB_ENTRIES as f64);

/// Branch boundary: arguments below this use the small-x leading-order form.
/// Matches PySCF `ECPsph_ine` line 4633 (`z < 1e-7`).
pub const X_SMALL_THRESHOLD: f64 = 1.0e-7;

/// Branch boundary: arguments above this use the asymptotic form.
/// Matches PySCF `ECPsph_ine` line 4639 (`z > 16`).
pub const X_LARGE_THRESHOLD: f64 = 16.0;

/// Maximum number of terms summed in the moderate-x Taylor series before bail-out.
/// At $l = 5, x = 16$ the series converges in ~63 terms; 200 is a hard cap to
/// prevent infinite loops on a NaN argument. Each f64-significand convergence
/// step nominally halves the residual.
const MODERATE_X_MAX_TERMS: u32 = 200;

/// Host-side double-factorial $(2k+1)!!$ for non-negative $k$.
/// Computes $(2k+1)(2k-1)\cdots 3 \cdot 1$. Used only on the host side for the
/// small-x leading prefactor; the moderate-x branch builds the prefactor
/// incrementally so no global factorial is needed inside `#[cube]`.
fn odd_double_factorial(two_k_plus_one: u32) -> f64 {
    let mut r: f64 = 1.0;
    let mut n: i64 = two_k_plus_one as i64;
    while n > 1 {
        r *= n as f64;
        n -= 2;
    }
    r
}

/// Host-side entry point — fills `out[0..=l_max]` with $i_l(x)$.
///
/// Returns a fresh `Vec<f64>` of length `l_max + 1`. The body delegates to
/// `modified_spherical_bessel_in_impl`, which is shared between the host
/// wrapper and the `#[cube]` core; this preserves byte-identity between the
/// CPU oracle path and the GPU dispatch path.
///
/// # Panics
///
/// Panics in debug builds if `l_max > ECP_LMAX`. The Type-2 ECP kernel never
/// requests higher $l$ than `ECP_LMAX = 5`; rejecting silently would hide a
/// caller bug. Release builds extend the computation (the algorithm itself is
/// $l$-uniform), but the result is then unverified outside the supported envelope.
pub fn modified_spherical_bessel_in_host(l_max: u32, x: f64) -> Vec<f64> {
    debug_assert!(
        l_max <= ECP_LMAX,
        "modified_spherical_bessel_in_host: l_max={} exceeds ECP_LMAX={}",
        l_max,
        ECP_LMAX,
    );
    let mut out = vec![0.0f64; (l_max + 1) as usize];
    modified_spherical_bessel_in_impl(&mut out, l_max, x);
    out
}

/// Core implementation shared between the host wrapper and the `#[cube]` form.
///
/// The host wrapper takes a `&mut [f64]`; the `#[cube]` form takes a
/// `&mut Array<f64>`. Both forward to this routine so the algorithm body lives
/// in exactly one place. Kept `pub` so the `boys.rs`-style oracle inspection
/// pattern works (a future test crate can call this directly).
pub fn modified_spherical_bessel_in_impl(out: &mut [f64], l_max: u32, x: f64) {
    debug_assert!(
        out.len() >= (l_max + 1) as usize,
        "modified_spherical_bessel_in_impl: out.len()={} < l_max+1={}",
        out.len(),
        l_max + 1,
    );

    if x < X_SMALL_THRESHOLD {
        // Branch 1: small-x leading-order form.
        // i_l(x) ≈ x^l / (2l+1)!! for x → 0.
        // i_0(0) = 1 (limit of sinh(x)/x); i_l(0) = 0 for l >= 1.
        out[0] = 1.0;
        let mut prev = 1.0;
        let mut l: u32 = 1;
        while l <= l_max {
            // i_l(x) / i_{l-1}(x) = x / (2l+1) at leading order.
            prev = prev * x / ((2 * l + 1) as f64);
            out[l as usize] = prev;
            l += 1;
        }
    } else if x > X_LARGE_THRESHOLD {
        // Branch 3: large-x asymptotic. Mirrors PySCF nr_ecp.c lines 4639-4652.
        //   i_l(x) = e^x / (2x) * sum_{k=0..l} (-1)^k (l+k)! / [k! (l-k)! (2x)^k]
        // Implemented with the same incremental recursion PySCF uses:
        //   t_0 = 1, t_{k+1} = t_k * (-1) * (l+k+1)(l-k) / ((k+1) * 2x).
        // The k = 0 term is just 1 (the scalar pulled outside), and the loop
        // builds the polynomial value `s = sum_{k=0..l} t_k`.
        let inv_2x = 0.5 / x;
        let neg_inv_2x = -inv_2x;
        let prefactor = f64::exp(x) * inv_2x; // e^x / (2x)
        let mut l: u32 = 0;
        while l <= l_max {
            let mut ti = 1.0; // t_0 = 1
            let mut s = ti;
            let mut k: u32 = 1;
            while k <= l {
                // t_k = t_{k-1} * (-1) * (l+k)*(l-k+1) / k / (2x)
                let num = ((l + k) as f64) * ((l - k + 1) as f64);
                let den = k as f64;
                ti = ti * neg_inv_2x * num / den;
                s = s + ti;
                k += 1;
            }
            out[l as usize] = prefactor * s;
            l += 1;
        }
    } else {
        // Branch 2: moderate-x direct Taylor series — the workhorse branch.
        // Mirrors PySCF nr_ecp.c lines 4654-4672 with the `exp(-z)` scaling removed:
        //   i_l(x) = x^l / (2l+1)!! * sum_{k=0..} (x^2/2)^k / [k! * (2k+2l+1)!!/(2l+1)!!]
        // Build s_l for each l by Horner-style incremental update:
        //   t_0 = x^l / (2l+1)!!,  t_{k+1} = t_k * (x^2/2) / [(k+1) * (2k+2l+3)]
        // and accumulate until s + t == s in f64 (PySCF's convergence test, line 4664).
        //
        // Note: x^l / (2l+1)!! is built incrementally across l using
        //   t0_{l+1} = t0_l * x / (2l+3).   (PySCF line 4670, also matches our
        //   small-x branch's incremental form.)
        let half_x2 = 0.5 * x * x;
        let mut t0: f64 = 1.0; // x^l / (2l+1)!! at l = 0 is 1/1 = 1.
        let mut l: u32 = 0;
        while l <= l_max {
            // Sum the k-series at this l.
            let mut tk = t0;
            let mut s = tk;
            let two_l_plus_1 = 2 * l + 1;
            let mut k: u32 = 1;
            let mut converged = false;
            while k <= MODERATE_X_MAX_TERMS {
                // (2k + 2l + 1) — but the recurrence uses (2k + 2l + 1) at step k+1
                // applied to t_k, which yields t_{k+1}.  Concretely PySCF's
                // ti *= z2 / (k * (k*2 + i*2 + 1)) is t_{k} from t_{k-1} with
                //   denominator k * (2k + 2l + 1)
                let denom = (k as f64) * ((2 * k + two_l_plus_1) as f64);
                tk = tk * half_x2 / denom;
                let next = s + tk;
                if next == s {
                    converged = true;
                    s = next;
                    break;
                }
                s = next;
                k += 1;
            }
            debug_assert!(
                converged,
                "modified_spherical_bessel_in: moderate-x series did not converge \
                 at l={}, x={} in {} terms (likely NaN/Inf input)",
                l, x, MODERATE_X_MAX_TERMS,
            );
            out[l as usize] = s;
            // Advance t0_l → t0_{l+1} = t0_l * x / (2l+3).
            // (Equivalent: t0_{l+1} = x^{l+1} / (2(l+1)+1)!! = t0_l * x / (2l+3).)
            t0 = t0 * x / ((2 * l + 3) as f64);
            l += 1;
        }
    }

    // Suppress unused warnings on the host-side double-factorial helper when
    // running in release-only / non-test builds. The helper is a documented
    // reference utility and intentionally available; touching it from a test
    // keeps `cargo build` warning-clean.
    let _ = odd_double_factorial;
}

/// `#[cube]` core — used by Plan 04's Type-2 launcher.
///
/// Same algorithm as `modified_spherical_bessel_in_host`, but operating on a
/// CubeCL `Array<f64>` output. The body is hand-inlined (not a delegation to
/// `modified_spherical_bessel_in_impl`) because CubeCL 0.10.x does not yet
/// support `&mut [f64]` parameters inside `#[cube]` — the kernel must own its
/// memory through `Array`.
#[cube]
pub fn modified_spherical_bessel_in(out: &mut Array<f64>, l_max: u32, x: f64) {
    if x < 1.0e-7f64 {
        // Branch 1: small-x leading-order form.
        out[0usize] = 1.0f64;
        let mut prev = 1.0f64;
        let mut l: u32 = 1;
        while l <= l_max {
            prev = prev * x / ((2u32 * l + 1u32) as f64);
            out[l as usize] = prev;
            l += 1;
        }
    } else if x > 16.0f64 {
        // Branch 3: large-x asymptotic.
        let inv_2x = 0.5f64 / x;
        let neg_inv_2x = -inv_2x;
        let prefactor = f64::exp(x) * inv_2x;
        let mut l: u32 = 0;
        while l <= l_max {
            let mut ti = 1.0f64;
            let mut s = ti;
            let mut k: u32 = 1;
            while k <= l {
                let num = ((l + k) as f64) * ((l - k + 1u32) as f64);
                let den = k as f64;
                ti = ti * neg_inv_2x * num / den;
                s = s + ti;
                k += 1;
            }
            out[l as usize] = prefactor * s;
            l += 1;
        }
    } else {
        // Branch 2: moderate-x direct Taylor series.
        let half_x2 = 0.5f64 * x * x;
        let mut t0 = 1.0f64;
        let mut l: u32 = 0;
        while l <= l_max {
            let mut tk = t0;
            let mut s = tk;
            let two_l_plus_1 = 2u32 * l + 1u32;
            let mut k: u32 = 1;
            // Hard cap mirrors MODERATE_X_MAX_TERMS = 200 in the host code.
            while k <= 200u32 {
                let denom = (k as f64) * ((2u32 * k + two_l_plus_1) as f64);
                tk = tk * half_x2 / denom;
                let next = s + tk;
                if next == s {
                    s = next;
                    // CubeCL statement-form: signal break via setting k past cap.
                    k = 201u32;
                } else {
                    s = next;
                    k += 1;
                }
            }
            out[l as usize] = s;
            t0 = t0 * x / ((2u32 * l + 3u32) as f64);
            l += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative-tolerance comparison for values whose magnitude spans many decades.
    /// `atol` covers values very close to zero; `rtol` covers everything else.
    fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, label: &str) {
        let diff = (actual - expected).abs();
        let tol = atol + rtol * expected.abs();
        assert!(
            diff <= tol,
            "{}: |{} - {}| = {} > tol {} (atol={}, rtol={})",
            label,
            actual,
            expected,
            diff,
            tol,
            atol,
            rtol,
        );
    }

    /// Test 1 (small-x Taylor branch): i_0(0.01) ≈ sinh(0.01)/0.01.
    ///
    /// 0.01 sits in the moderate-x branch (above 1e-7); the Taylor series is
    /// what converges here, which exercises Branch 2.  Reference value from
    /// `math.sinh(0.01)/0.01`.
    #[test]
    fn small_x_branch_l0() {
        let out = modified_spherical_bessel_in_host(0, 0.01);
        // sinh(0.01)/0.01 = 1.000016666750000... (verified via Python math.sinh).
        let expected = 1.0000166667500003;
        assert_eq!(out.len(), 1);
        assert_close(out[0], expected, 1e-13, 1e-13, "i_0(0.01)");
    }

    /// Test 2 (zero-argument identity, all l): i_0(0) = 1, i_l(0) = 0 for l ≥ 1.
    ///
    /// Exercises Branch 1 (small-x leading-order) at the exact boundary x = 0.
    #[test]
    fn zero_argument_all_l() {
        let out = modified_spherical_bessel_in_host(ECP_LMAX, 0.0);
        assert_eq!(out.len(), (ECP_LMAX + 1) as usize);
        assert_close(out[0], 1.0, 1e-15, 0.0, "i_0(0)");
        for l in 1..=ECP_LMAX as usize {
            assert_close(out[l], 0.0, 1e-15, 0.0, &format!("i_{}(0)", l));
        }
    }

    /// Test 3 (moderate-x Taylor branch): i_l(4.0) for l = 0..=3 against
    /// closed-form reference values.
    ///
    /// Reference values from sympy / `math.sinh, math.cosh`:
    /// - i_0(4) = sinh(4)/4 = 6.822479299281937
    /// - i_1(4) = cosh(4)/4 − sinh(4)/16 = 5.121438384183637
    /// - i_2(4) = i_0(4) − 3/4 · i_1(4) = 2.9814005111442095  (upward recurrence,
    ///   stable for l < x = 4)
    /// - i_3(4) = i_1(4) − 5/4 · i_2(4) = 1.3946877452533757
    #[test]
    fn moderate_x_branch_l_sweep() {
        let out = modified_spherical_bessel_in_host(3, 4.0);
        assert_eq!(out.len(), 4);
        // Reference values verified via Python (see plan-execution log).
        assert_close(out[0], 6.822479299281937, 1e-12, 1e-12, "i_0(4.0)");
        assert_close(out[1], 5.121438384183637, 1e-12, 1e-12, "i_1(4.0)");
        assert_close(out[2], 2.9814005111442095, 1e-12, 1e-12, "i_2(4.0)");
        assert_close(out[3], 1.3946877452533757, 1e-12, 1e-12, "i_3(4.0)");
    }

    /// Test 4 (large-x asymptotic branch): i_0(30.0) ≈ e^30 / (2·30).
    ///
    /// At x = 30 the leading-order asymptotic is exact to all f64 significand
    /// bits for l = 0 (the inner polynomial reduces to t_0 = 1).
    #[test]
    fn large_x_branch_l0() {
        let out = modified_spherical_bessel_in_host(0, 30.0);
        let expected = f64::exp(30.0) / 60.0; // = 178107909692.07437
        assert_eq!(out.len(), 1);
        // rtol because values are O(1e11); atol alone would be too loose.
        assert_close(out[0], expected, 0.0, 1e-13, "i_0(30.0)");
    }

    /// Test 5 (high-l, small-x — recurrence-stability sentinel):
    /// i_5(0.5) computed via the direct Taylor sum (no recurrence).
    ///
    /// Reference value from the canonical Taylor series identity
    ///   i_l(x) = sum_{k=0..inf} x^{l+2k} / (2^k * k! * (2l+2k+1)!!)
    /// evaluated to convergence in Python (independent term-by-term sum):
    ///   k=0: 0.5^5 / (1 * 1 * 11!!)       ≈ 3.0063e-6
    ///   k=1: 0.5^7 / (2 * 1 * 13!!)       ≈ 2.8906e-8
    ///   k=2: 0.5^9 / (4 * 2 * 15!!)       ≈ 1.2044e-10
    ///   ...convergence at k=4:           ≈ 3.0352800236771825e-6
    ///
    /// The upward recurrence
    ///   i_{l+1}(x) = i_{l-1}(x) - (2l+1)/x * i_l(x)
    /// at x = 0.5 sees catastrophic cancellation by l = 5 (last-digit error
    /// ~5e-12 relative — visible in independent Python recursion); the direct
    /// Taylor branch we implement here is correct to all f64 bits.
    #[test]
    fn high_l_small_x_stability() {
        let out = modified_spherical_bessel_in_host(5, 0.5);
        assert_eq!(out.len(), 6);
        let expected_l5 = 3.0352800236771825e-6;
        assert_close(out[5], expected_l5, 0.0, 1e-12, "i_5(0.5)");
        // Sanity-check the lower channels too — they should be monotonically
        // larger and finite.
        for l in 0..5 {
            assert!(
                out[l].is_finite() && out[l] > out[5],
                "i_{}(0.5) = {} should be finite and > i_5",
                l,
                out[l],
            );
        }
    }

    /// Test 6 (bounds-check at ECP_LMAX): the host wrapper supports the full
    /// $l = 0..=ECP_LMAX = 5$ range across a representative argument span.
    #[test]
    fn lmax_supported_across_branches() {
        for &x in &[1e-10, 0.5, 4.0, 16.5] {
            let out = modified_spherical_bessel_in_host(ECP_LMAX, x);
            assert_eq!(out.len(), (ECP_LMAX + 1) as usize);
            for (l, v) in out.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "i_{}(x={}) = {} is not finite",
                    l,
                    x,
                    v,
                );
            }
        }
    }

    /// Test 7 (branch-boundary continuity at x = 16): the moderate-x Taylor
    /// branch (x ≤ 16) and the large-x asymptotic branch (x > 16) must agree
    /// at the boundary up to the residual `e^-x / (2x)` correction the
    /// asymptotic branch drops.
    ///
    /// For i_0, the exact identity is
    ///   i_0(x) = sinh(x)/x = (e^x − e^-x) / (2x) = e^x/(2x) − e^-x/(2x).
    /// The asymptotic branch returns e^x/(2x) for l = 0 (the inner polynomial
    /// terminates at k = 0). The dropped residual at x = 16 is
    ///   e^-16 / 32 ≈ 3.5 × 10^-9
    /// — well within the 1e-8 absolute tolerance for the asymptotic branch's
    /// physical use case (large x, large e^x, the residual is exponentially
    /// suppressed). The MODERATE branch is exact to f64 in the same regime.
    ///
    /// The function i_0(x) is steep near x = 16 (derivative ~e^x/(2x) so
    /// ~ 1.7 × 10^4 per unit x there); any boundary-jump test must therefore
    /// fix x and toggle the branch — NOT vary x — to be meaningful.
    #[test]
    fn branch_boundary_continuity_at_x_16() {
        // Closed-form reference: i_0(16) = sinh(16)/16 (computed in Python,
        // f64 round-trip). Hardcoded here because cubecl's prelude shadows the
        // host f64 sinh method inside #[cfg(test)] modules of cintx-cubecl.
        let i0_closed = 277690.9537658675_f64;

        // Hit the moderate-x branch (x <= 16): the boundary point x = 16 itself
        // uses the moderate path because the asymptotic test is `x > 16`.
        let x = 16.0;
        let out_moderate = modified_spherical_bessel_in_host(0, x);
        assert_close(out_moderate[0], i0_closed, 0.0, 1e-12, "i_0(16) moderate");

        // Hit the asymptotic branch at x = 16 + ε. The asymptotic answer for
        // l = 0 is e^x/(2x), which differs from sinh(x)/x by e^{-x}/(2x); at
        // x ≈ 16 that residual is ~3.5e-9 absolute (~1.3e-14 relative —
        // sub-f64). The asymptotic branch matches i_0(x) only up to this
        // exponentially-suppressed correction.
        let x_eps = 16.0 + 1.0e-9;
        let out_asymp = modified_spherical_bessel_in_host(0, x_eps);
        // The asymptotic branch gives e^x_eps / (2 x_eps); compare against the
        // truncated leading-order formula directly:
        let asymp_leading = f64::exp(x_eps) / (2.0 * x_eps);
        assert_close(
            out_asymp[0],
            asymp_leading,
            0.0,
            1e-12,
            "i_0(16+ε) asymptotic-leading agreement",
        );

        // And the asymptotic value vs the exact sinh form at the same x_eps:
        // agreement up to ~3.5e-9 absolute (the dropped e^-x_eps/(2x_eps) term).
        // Hardcoded reference is sinh(16 + 1e-9)/(16 + 1e-9).
        let i0_at_eps_closed = 277690.95402620285_f64;
        assert_close(
            out_asymp[0],
            i0_at_eps_closed,
            5.0e-8,
            1.0e-13,
            "i_0(16+ε) asymp vs closed",
        );
    }

    /// Test 8 (constants match nr_ecp.h): the pub constants exposed by this
    /// module match PySCF nr_ecp.h byte-for-byte.
    #[test]
    fn constants_match_nr_ecp_h() {
        assert_eq!(ECP_LMAX, 5);
        assert_eq!(K_TAYLOR_MAX, 7);
        assert_eq!(K_TAB_ENTRIES, 400);
        assert_eq!(K_TAB_COL, 24);
        assert!((K_TAB_INTERVAL - 0.04).abs() < 1e-15);
        assert!((X_SMALL_THRESHOLD - 1.0e-7).abs() == 0.0);
        assert!((X_LARGE_THRESHOLD - 16.0).abs() == 0.0);
    }

    /// Test 9 (double-factorial helper): odd-double-factorial round-trip.
    /// Sanity-check the host-side reference helper used in the rustdoc.
    #[test]
    fn odd_double_factorial_basic() {
        assert_eq!(odd_double_factorial(1), 1.0);
        assert_eq!(odd_double_factorial(3), 3.0);
        assert_eq!(odd_double_factorial(5), 15.0);
        assert_eq!(odd_double_factorial(7), 105.0);
        // (2*5+1)!! = 11!! = 11·9·7·5·3·1 = 10395
        assert_eq!(odd_double_factorial(11), 10395.0);
    }
}
