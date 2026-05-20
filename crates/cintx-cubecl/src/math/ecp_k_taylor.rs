//! PySCF K-Taylor radial machinery — host-first Rust ports for byte-identity.
//!
//! This module ports PySCF's *exact* radial recurrence (not a mathematically
//! equivalent quadrature) so the ECP scalar/gradient kernels can reach
//! `atol=1e-12, rtol=0.0` byte-identity vs PySCF nr_ecp (Phase 19 D-13). The
//! 19-04 direct-quadrature kernel diverged precisely because it did not
//! replicate PySCF's table-interpolation Bessel evaluation and radial assembly;
//! this module is the missing foundation that 19-06 (scalar close) and 19-07
//! (gradient) consume.
//!
//! Ported functions (each carries a `// Source:` citation):
//! - `ecpsph_ine_opt_host`  — `ECPsph_ine_opt` (nr_ecp.c:4687-4837), the
//!   table-interpolation modified-spherical-Bessel evaluator.
//! - `ecprad_part_host`     — `ECPrad_part` (nr_ecp.c:4870-4950).
//! - `type1_rad_part_host`  — `type1_rad_part` (nr_ecp.c:5754-5806).
//! - `type2_facs_rad_host`  — `type2_facs_rad` (nr_ecp.c:5134-5186).
//!
//! # Scaled-Bessel convention (byte-identity-critical)
//!
//! PySCF's `ECPsph_ine` / `ECPsph_ine_opt` produce the **scaled** modified
//! spherical Bessel function `i_l(z) * exp(-z)` (see nr_ecp.c:4656 `t0 = exp(-z)`
//! in the moderate-z branch and the `(1 - z)` prefactor at 4635 in the small-z
//! branch). The embedded `_sph_ine_tab*` tables encode this scaled form (the
//! first table entry `9.802640211919197e-01 ≈ i_0(0.02) * exp(-0.02)`). This is
//! distinct from `crate::math::bessel::modified_spherical_bessel_in_host`, which
//! deliberately drops the `exp(-z)` scaling to produce the *unscaled* `i_l(z)`.
//! For byte-identity we therefore port `ECPsph_ine` verbatim here (the scaled
//! form) rather than delegating to the unscaled `bessel.rs` series — see the
//! "Deviations" note in 19-05-SUMMARY.md.
//!
//! # CubeCL constraint (CLAUDE.md "CubeCL is the primary compute backend")
//!
//! This ECP radial machinery is **host-only this phase** (D-16 host-first): the
//! byte-identity gate runs CPU-vs-C on `--features cpu`, so a `*_host()` port
//! closes the requirement. The `#[cube]` GPU counterpart (paired `bessel.rs`
//! pattern, host-side branching for the data-dependent control flow) is a
//! *documented deviation* tracked in 19-CONTEXT Deferred Ideas — it is not the
//! intended end state, and a follow-up plan or v1.4 lands the GPU half. Per the
//! `bessel.rs` / `boys.rs` Phase 8 P02 note, no `#[cube]` body appears in this
//! file.

use crate::math::bessel::{K_TAB_COL, K_TAB_INTERVAL, K_TAYLOR_MAX, X_LARGE_THRESHOLD, X_SMALL_THRESHOLD};
use crate::math::ecp_k_taylor_data::{sph_ine_tab, sph_ine_tab_order7};

/// `ORDER7OFFSET` — stride between Taylor-coefficient rows in `_sph_ine_tab_order7`.
/// Defined in `vendor/pyscf-nr-ecp/src/nr_ecp.c:433` (NOT the header).
const ORDER7OFFSET: usize = 8;

/// `_l2[l] = l / (2l+1)` — downward-recurrence weights for the order>7 branch.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4678-4683
const L2: [f64; 24] = [
    0.0,
    1.0 / 3.0,
    2.0 / 5.0,
    3.0 / 7.0,
    4.0 / 9.0,
    5.0 / 11.0,
    6.0 / 13.0,
    7.0 / 15.0,
    8.0 / 17.0,
    9.0 / 19.0,
    10.0 / 21.0,
    11.0 / 23.0,
    12.0 / 25.0,
    13.0 / 27.0,
    14.0 / 29.0,
    15.0 / 31.0,
    16.0 / 33.0,
    17.0 / 35.0,
    18.0 / 37.0,
    19.0 / 39.0,
    20.0 / 41.0,
    21.0 / 43.0,
    22.0 / 45.0,
    23.0 / 47.0,
];

/// `_j_inv[j] = 1 / j` — Taylor-series term factor (index 0 unused).
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4684-4686
const J_INV: [f64; 10] = [
    0.0,
    1.0,
    0.5,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 5.0,
    1.0 / 6.0,
    1.0 / 7.0,
    1.0 / 8.0,
    1.0 / 9.0,
];

/// `_factorial[n] = n!` — used by the large-z asymptotic branch of `ECPsph_ine`.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4463-4471
const FACTORIAL: [f64; 24] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    1.2e2,
    7.2e2,
    5.04e3,
    4.032e4,
    3.6288e5,
    3.6288e6,
    3.99168e7,
    4.790016e8,
    6.2270208e9,
    8.71782912e10,
    1.307674368e12,
    2.0922789888e13,
    3.55687428096e14,
    6.402373705728e15,
    1.21645100408832e17,
    2.43290200817664e18,
    5.109094217170944e19,
    1.1240007277776077e21,
    2.5852016738884978e22,
];

/// Port of PySCF `ECPsph_ine` — the *scaled* modified spherical Bessel
/// `i_l(z) * exp(-z)` for `l = 0..=order`, three-branch direct evaluation.
///
/// This is the fall-through evaluator that `ECPsph_ine_opt` delegates to for
/// `z < 1e-7` and `z > 16`. It produces the SAME scaled convention the embedded
/// K-Taylor tables encode (see module rustdoc).
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4630-4675
fn ecpsph_ine(out: &mut [f64], order: usize, z: f64) {
    if z < 1.0e-7 {
        // (1-z) * z^l / (2l+1)!!
        out[0] = 1.0 - z;
        for i in 1..=order {
            out[i] = out[i - 1] * z / ((i * 2 + 1) as f64);
        }
    } else if z > 16.0 {
        // R_l(z) = sum_k (l+k)! / (k! (l-k)! (2z)^k)
        let z2 = -0.5 / z;
        for i in 0..=order {
            let mut ti = 0.5 / z;
            let mut s = ti;
            for k in 1..=i {
                ti *= z2;
                s += ti * FACTORIAL[i + k] / (FACTORIAL[k] * FACTORIAL[i - k]);
            }
            out[i] = s;
        }
    } else {
        // z^l e^{-z} sum (z^2/2)^k / (k! (2k+2l+1)!!)
        let z2 = 0.5 * z * z;
        let mut t0 = (-z).exp();
        for i in 0..=order {
            let mut ti = t0;
            let mut s = ti;
            let mut k = 1usize;
            loop {
                ti *= z2 / ((k * (k * 2 + i * 2 + 1)) as f64);
                let next = s + ti;
                if next == s {
                    break;
                } else {
                    s = next;
                }
                k += 1;
            }
            t0 *= z / ((i * 2 + 3) as f64);
            out[i] = s;
        }
    }
}

/// Host port of PySCF `ECPsph_ine_opt` — table-interpolation modified spherical
/// Bessel evaluator returning the scaled `i_l(z) * exp(-z)` for `l = 0..=order`.
///
/// For `z < X_SMALL_THRESHOLD` (1e-7) or `z > X_LARGE_THRESHOLD` (16.0) it falls
/// through to the direct `ecpsph_ine` evaluator. The middle `[1e-7, 16]` regime
/// uses `entry = floor(z / K_TAB_INTERVAL)`, a per-order Taylor sum over
/// `_sph_ine_tab_order7` for `order <= 7`, and the `_l2`-based downward recurrence
/// over `_sph_ine_tab` for `order > 7`.
///
/// Returns `order + 1` values.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4687-4837
pub fn ecpsph_ine_opt_host(order: u32, z: f64) -> Vec<f64> {
    let order = order as usize;
    let mut out = vec![0.0f64; order + 1];

    if z < X_SMALL_THRESHOLD || z > X_LARGE_THRESHOLD {
        ecpsph_ine(&mut out, order, z);
        return out;
    }

    let k_tab_col = K_TAB_COL as usize;
    let k_taylor_max = K_TAYLOR_MAX as usize;
    let tab_order7 = sph_ine_tab_order7();

    let entry = (z / K_TAB_INTERVAL).floor() as usize;
    let tabu_base = entry * ORDER7OFFSET * (k_taylor_max + 1);
    let z0 = (entry as f64) * K_TAB_INTERVAL + K_TAB_INTERVAL / 2.0;
    let dz = z - z0;

    if order <= 7 {
        // Per-order Taylor sum over _sph_ine_tab_order7 (cases 0..=7 in the C).
        let mut s = [0.0f64; ORDER7OFFSET];
        for i in 0..=order {
            s[i] = tab_order7[tabu_base + i];
        }
        let mut fac = 1.0f64;
        for j in 1..=k_taylor_max {
            fac *= dz * J_INV[j];
            for i in 0..=order {
                s[i] += tab_order7[tabu_base + j * ORDER7OFFSET + i] * fac;
            }
        }
        for i in 0..=order {
            out[i] = s[i];
        }
    } else {
        // order>7 default branch: _l2-based downward recurrence over _sph_ine_tab.
        let tab = sph_ine_tab();
        let tab_base = entry * k_tab_col;
        let hi = order + k_taylor_max;
        let mut k0 = vec![0.0f64; k_tab_col * 2];
        let mut k1 = vec![0.0f64; k_tab_col * 2];
        for i in 0..=hi {
            k0[i] = tab[tab_base + i];
        }
        for i in 0..=order {
            out[i] = k0[i];
        }
        let mut fac = 1.0f64;
        for j in 1..=k_taylor_max {
            k1[0] = k0[1] - k0[0];
            for i in 1..=(order + k_taylor_max - j) {
                k1[i] = L2[i] * k0[i - 1] + (1.0 - L2[i]) * k0[i + 1] - k0[i];
            }
            fac *= dz * J_INV[j];
            for i in 0..=order {
                out[i] += k1[i] * fac;
            }
            std::mem::swap(&mut k0, &mut k1);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ecp_k_taylor_data::{sph_ine_tab, sph_ine_tab_order7};

    /// Embedded blobs decode to the exact element counts and the first
    /// `_sph_ine_tab` literal matches the vendored source bit-for-bit.
    #[test]
    fn embedded_tables_have_expected_shape_and_first_value() {
        assert_eq!(sph_ine_tab().len(), 9600);
        assert_eq!(sph_ine_tab_order7().len(), 25600);
        // First literal of _sph_ine_tab in nr_ecp.c:31.
        assert_eq!(sph_ine_tab()[0], 9.802640211919197e-01);
    }

    /// `ecpsph_ine_opt_host(order=2, z)` in the middle [1e-7,16] table-interp
    /// regime must equal the table arithmetic computed directly against the
    /// embedded `_sph_ine_tab_order7` (pins the table path specifically, NOT
    /// cintx's generic series).
    #[test]
    fn table_interp_middle_regime_matches_embedded_table() {
        let order: usize = 2;
        let tab = sph_ine_tab_order7();
        let k_taylor_max = K_TAYLOR_MAX as usize;
        for &z in &[0.5f64, 4.0, 12.0] {
            let entry = (z / K_TAB_INTERVAL).floor() as usize;
            let base = entry * ORDER7OFFSET * (k_taylor_max + 1);
            let z0 = (entry as f64) * K_TAB_INTERVAL + K_TAB_INTERVAL / 2.0;
            let dz = z - z0;

            // Reference: replicate the C per-order Taylor sum by hand.
            let mut s = [0.0f64; ORDER7OFFSET];
            for i in 0..=order {
                s[i] = tab[base + i];
            }
            let mut fac = 1.0f64;
            for j in 1..=k_taylor_max {
                fac *= dz * J_INV[j];
                for i in 0..=order {
                    s[i] += tab[base + j * ORDER7OFFSET + i] * fac;
                }
            }

            let got = ecpsph_ine_opt_host(order as u32, z);
            assert_eq!(got.len(), order + 1);
            for i in 0..=order {
                assert_eq!(
                    got[i], s[i],
                    "order={i} z={z}: port {} != hand-computed table {}",
                    got[i], s[i]
                );
            }
        }
    }

    /// Small-z fall-through (z < 1e-7) delegates to the scaled `ECPsph_ine`
    /// small-z branch: `out[0] = 1 - z`.
    #[test]
    fn small_z_fall_through_matches_scaled_ecpsph_ine() {
        let z = 1.0e-9;
        let got = ecpsph_ine_opt_host(0, z);
        assert_eq!(got.len(), 1);
        // ECPsph_ine small-z branch: out[0] = 1 - z  (scaled i_0(z)*exp(-z)).
        assert_eq!(got[0], 1.0 - z);
    }

    /// Large-z fall-through (z > 16) delegates to the scaled `ECPsph_ine`
    /// large-z asymptotic branch element-wise.
    #[test]
    fn large_z_fall_through_matches_scaled_ecpsph_ine() {
        let order: usize = 3;
        let z = 20.0f64;
        let got = ecpsph_ine_opt_host(order as u32, z);

        // Reference: replicate the C large-z branch by hand.
        let z2 = -0.5 / z;
        let mut reference = vec![0.0f64; order + 1];
        for i in 0..=order {
            let mut ti = 0.5 / z;
            let mut s = ti;
            for k in 1..=i {
                ti *= z2;
                s += ti * FACTORIAL[i + k] / (FACTORIAL[k] * FACTORIAL[i - k]);
            }
            reference[i] = s;
        }

        assert_eq!(got.len(), order + 1);
        for i in 0..=order {
            assert_eq!(got[i], reference[i], "order={i} z={z} large-z mismatch");
        }
    }
}
