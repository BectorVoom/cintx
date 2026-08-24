//! GTO normalization, matching libcint 6.1.3 / PySCF conventions exactly.
//!
//! Two distinct normalizations are applied, in this order, and both are
//! required for `env` coefficient parity with what PySCF hands libcint:
//!
//! 1. **Per-primitive radial norm** — `CINTgto_norm(l, a)`
//!    (`libcint-master/src/misc.c:86`):
//!    ```text
//!    CINTgto_norm(l, a) = 1 / sqrt(gaussian_int(2l + 2, 2a))
//!    gaussian_int(n, alpha) = Gamma((n + 1) / 2) / (2 * alpha^((n + 1) / 2))
//!    ```
//!
//! 2. **Per-contraction self-overlap renorm** — PySCF
//!    `gto.mole._nomalize_contracted_ao`: each contraction column `c` is scaled
//!    by `1 / sqrt(c^T S c)` with `S_pq = gaussian_int(2l + 2, e_p + e_q)`.
//!
//! Getting either wrong produces a plausible-looking constant factor in every
//! integral rather than an obvious failure, which is why `normalization.rs`
//! is gated by a dedicated parity test before any integral is evaluated.

/// `gaussian_int(n, alpha) = Gamma((n+1)/2) / (2 * alpha^((n+1)/2))`.
///
/// Verbatim port of `_gaussian_int` (`libcint-master/src/misc.c:72`).
#[must_use]
pub fn gaussian_int(n: i32, alpha: f64) -> f64 {
    let n1 = f64::from(n + 1) * 0.5;
    ln_gamma(n1).exp() / (2.0 * alpha.powf(n1))
}

/// `CINTgto_norm(l, a)` — the per-primitive radial normalization constant for
/// `g = r^l exp(-a r^2)`.
///
/// Verbatim port of `CINTgto_norm` (`libcint-master/src/misc.c:86`).
#[must_use]
pub fn gto_norm(l: i32, alpha: f64) -> f64 {
    debug_assert!(l >= 0, "gto_norm: angular momentum must be non-negative");
    1.0 / gaussian_int(l * 2 + 2, 2.0 * alpha).sqrt()
}

/// Scale a contraction block by the per-primitive `gto_norm`, in place.
///
/// `coefficients` is contraction-major (`coeff[ic * nprim + ip]`), the layout
/// libcint's `env` uses and the layout [`cintx_core::Shell`] documents.
pub fn apply_primitive_norm(l: i32, exponents: &[f64], coefficients: &mut [f64], nctr: usize) {
    let nprim = exponents.len();
    debug_assert_eq!(coefficients.len(), nprim * nctr);
    for ip in 0..nprim {
        let norm = gto_norm(l, exponents[ip]);
        for ic in 0..nctr {
            coefficients[ic * nprim + ip] *= norm;
        }
    }
}

/// Renormalize every contraction column so its self-overlap is exactly 1.
///
/// Port of PySCF `gto.mole._nomalize_contracted_ao`. Must run *after*
/// [`apply_primitive_norm`].
///
/// A column whose self-overlap is non-finite or non-positive (only reachable
/// from an all-zero column) is left untouched rather than producing NaN.
pub fn normalize_contracted(l: i32, exponents: &[f64], coefficients: &mut [f64], nctr: usize) {
    let nprim = exponents.len();
    debug_assert_eq!(coefficients.len(), nprim * nctr);

    // S_pq = gaussian_int(2l + 2, e_p + e_q); symmetric, so build once.
    let mut overlap = vec![0.0_f64; nprim * nprim];
    for p in 0..nprim {
        for q in p..nprim {
            let value = gaussian_int(l * 2 + 2, exponents[p] + exponents[q]);
            overlap[p * nprim + q] = value;
            overlap[q * nprim + p] = value;
        }
    }

    for ic in 0..nctr {
        let column = &coefficients[ic * nprim..(ic + 1) * nprim];
        let mut self_overlap = 0.0_f64;
        for p in 0..nprim {
            for q in 0..nprim {
                self_overlap += column[p] * overlap[p * nprim + q] * column[q];
            }
        }
        if !self_overlap.is_finite() || self_overlap <= 0.0 {
            continue;
        }
        let scale = 1.0 / self_overlap.sqrt();
        for value in &mut coefficients[ic * nprim..(ic + 1) * nprim] {
            *value *= scale;
        }
    }
}

/// Apply the full libcint/PySCF normalization chain to a contraction block.
pub fn normalize_block(l: i32, exponents: &[f64], coefficients: &mut [f64], nctr: usize) {
    apply_primitive_norm(l, exponents, coefficients, nctr);
    normalize_contracted(l, exponents, coefficients, nctr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lanczos log-gamma.
//
// `gaussian_int` needs `lgamma` at half-integer arguments only (n is even, so
// n1 = (n+1)/2 is a half-integer >= 0.5). Rust's std has no `lgamma`, and the
// value feeds a normalization constant that must match C's `lgamma` to within
// f64 rounding, so a high-accuracy Lanczos approximation (g = 7, n = 9) is used
// rather than a Stirling series.
// ─────────────────────────────────────────────────────────────────────────────

const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural log of the Gamma function for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    debug_assert!(x > 0.0, "ln_gamma is only used for positive arguments here");

    // Reflection is unnecessary: every call site passes x >= 0.5.
    let x = x - 1.0;
    let mut series = LANCZOS_COEFFICIENTS[0];
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (x + index as f64);
    }
    let t = x + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Gamma(n) == (n-1)!` for small integers, and `Gamma(1/2) == sqrt(pi)`.
    #[test]
    fn ln_gamma_matches_known_values() {
        let cases = [
            (1.0_f64, 1.0_f64),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 6.0),
            (5.0, 24.0),
            (6.0, 120.0),
            (0.5, std::f64::consts::PI.sqrt()),
            (1.5, std::f64::consts::PI.sqrt() / 2.0),
            (2.5, 0.75 * std::f64::consts::PI.sqrt()),
            (3.5, 1.875 * std::f64::consts::PI.sqrt()),
        ];
        for (x, expected) in cases {
            let actual = ln_gamma(x).exp();
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "ln_gamma({x}) -> {actual}, expected {expected}"
            );
        }
    }

    /// The commented-out closed form in `libcint-master/src/misc.c:88` is an
    /// independent expression of the same constant; both must agree.
    #[test]
    fn gto_norm_matches_libcint_closed_form() {
        fn factorial(n: u32) -> f64 {
            (1..=n).map(f64::from).product::<f64>().max(1.0)
        }
        for l in 0..=6_i32 {
            for alpha in [0.05_f64, 0.31, 1.0, 7.5, 130.7] {
                let closed = (2.0_f64.powi(2 * l + 3)
                    * factorial(l as u32 + 1)
                    * (2.0 * alpha).powf(f64::from(l) + 1.5)
                    / (factorial(2 * l as u32 + 2) * std::f64::consts::PI.sqrt()))
                .sqrt();
                let actual = gto_norm(l, alpha);
                assert!(
                    (actual - closed).abs() <= 1e-11 * closed.abs(),
                    "gto_norm({l}, {alpha}) -> {actual}, closed form {closed}"
                );
            }
        }
    }

    /// After the full chain, every contraction column has unit self-overlap.
    #[test]
    fn normalize_block_yields_unit_self_overlap() {
        let exponents = [130.709_32_f64, 23.808_861, 6.443_608_3];
        let mut coefficients = [0.154_328_97_f64, 0.535_328_14, 0.444_634_54];
        normalize_block(0, &exponents, &mut coefficients, 1);

        let mut self_overlap = 0.0_f64;
        for (p, &cp) in coefficients.iter().enumerate() {
            for (q, &cq) in coefficients.iter().enumerate() {
                self_overlap += cp * cq * gaussian_int(2, exponents[p] + exponents[q]);
            }
        }
        assert!(
            (self_overlap - 1.0).abs() < 1e-13,
            "self overlap {self_overlap} should be 1"
        );
    }

    /// The primitive norm must scale every contraction column of an nctr>1
    /// block, not just the first — the def2 general-contraction case.
    #[test]
    fn primitive_norm_applies_to_every_contraction_column() {
        let exponents = [5.0_f64, 1.2];
        let mut coefficients = [1.0_f64, 1.0, 1.0, 1.0];
        apply_primitive_norm(1, &exponents, &mut coefficients, 2);
        assert_eq!(coefficients[0], gto_norm(1, 5.0));
        assert_eq!(coefficients[1], gto_norm(1, 1.2));
        assert_eq!(coefficients[2], gto_norm(1, 5.0));
        assert_eq!(coefficients[3], gto_norm(1, 1.2));
    }
}
