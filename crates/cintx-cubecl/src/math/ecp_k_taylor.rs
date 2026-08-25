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

// Transcribed verbatim from vendored libcint 6.1.3 (and, in `cintx-basis`, from the
// Lanczos reference these normalization constants come from). Result compatibility
// is decided by the exact bits these literals carry, so none is truncated to the
// shortest form that round-trips.
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
// Element loops rather than `copy_from_slice`: the source index carries a table
// base offset (`tabu_base + i`, `j * ORDER7OFFSET + i`) or a recursion bound that
// the slice form would have to restate as two range expressions.
#![allow(clippy::manual_memcpy)]

use crate::math::bessel::{
    K_TAB_COL, K_TAB_INTERVAL, K_TAYLOR_MAX, X_LARGE_THRESHOLD, X_SMALL_THRESHOLD,
};
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

    if !(X_SMALL_THRESHOLD..=X_LARGE_THRESHOLD).contains(&z) {
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

/// `SIM_ZERO` — radial Gaussian underflow early-break threshold.
/// Source: vendor/pyscf-nr-ecp/include/gto/nr_ecp.h:13 (`SIM_ZERO 1e-50`).
pub const SIM_ZERO: f64 = 1.0e-50;

/// `EXPCUTOFF` — exponent-argument cutoff used by the radial guards.
/// Source: vendor/pyscf-nr-ecp/include/gto/nr_ecp.h:14 (`EXPCUTOFF 39`).
pub const EXPCUTOFF: f64 = 39.0;

/// `CUTOFF` — upper exp-argument guard (`~ 1e200`).
/// Source: vendor/pyscf-nr-ecp/include/gto/nr_ecp.h:15 (`CUTOFF 460`).
pub const CUTOFF: f64 = 460.0;

/// One radial-shell group fed to [`ecprad_part_host`].
///
/// Bundles the per-ECP-shell data the C reads out of the `ecpbas`/`env` slabs
/// (`NPRIM_OF`, `PTR_EXP`, `PTR_COEFF`, `RADI_POWER`). ECP basis rows that share
/// the same `(atom, l, so_type)` are grouped into one shell and summed together,
/// matching `_loc_ecpbas` (nr_ecp.c:5261).
#[derive(Clone, Debug)]
pub struct EcpRadShell {
    /// Primitive Gaussian exponents `a_k` (length = nprim).
    pub exponents: Vec<f64>,
    /// Primitive Gaussian coefficients `c_k` (length = nprim).
    pub coefficients: Vec<f64>,
    /// `RADI_POWER` for this shell (slot 3 of the ecpbas row).
    pub radial_power: i32,
}

/// Host port of PySCF `ECPrad_part` — sum the per-ECP-shell radial Gaussian
/// contributions onto `ur`, applying the `RADI_POWER` switch and the `SIM_ZERO`
/// early-break.
///
/// `rs` is the radial grid (strided by `inc`, offset by `rs_off`), `nrs` the
/// grid length. `ur` is the accumulator (length `>= nrs`); it is zeroed for
/// `0..nrs` first. Returns `nrs_max` — the largest effective grid length any
/// shell reached before its `SIM_ZERO` early-break (the C "number of effective
/// grids" return value).
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4870-4950
pub fn ecprad_part_host(
    ur: &mut [f64],
    rs: &[f64],
    rs_off: usize,
    nrs: usize,
    inc: usize,
    shells: &[EcpRadShell],
) -> usize {
    let mut ubuf = vec![0.0f64; nrs];
    let mut r2 = vec![0.0f64; nrs];

    // rs += rs_off;  r2[i] = rs[i*inc]^2;  ur[i] = 0;
    for i in 0..nrs {
        let r = rs[rs_off + i * inc];
        r2[i] = r * r;
        ur[i] = 0.0;
    }

    let mut nrs_max = 0usize;

    for shell in shells {
        let npk = shell.exponents.len();
        let ak = &shell.exponents;
        let ck = &shell.coefficients;

        // Per-grid Gaussian sum with SIM_ZERO early-break.
        let mut nrs_now = nrs;
        for i in 0..nrs {
            let mut s = ck[0] * (-ak[0] * r2[i]).exp();
            for kp in 1..npk {
                s += ck[kp] * (-ak[kp] * r2[i]).exp();
            }
            ubuf[i] = s;
            if i > 2 && ubuf[i].abs() < SIM_ZERO && ubuf[i - 1].abs() < SIM_ZERO {
                nrs_now = i;
                break;
            }
        }
        nrs_max = nrs_max.max(nrs_now);

        // RADI_POWER switch (cases 1/2/3/default).
        match shell.radial_power {
            1 => {
                for i in 0..nrs_now {
                    ubuf[i] *= rs[rs_off + i * inc];
                }
            }
            2 => {
                for i in 0..nrs_now {
                    ubuf[i] *= r2[i];
                }
            }
            3 => {
                for i in 0..nrs_now {
                    ubuf[i] *= r2[i] * rs[rs_off + i * inc];
                }
            }
            other => {
                for i in 0..nrs_now {
                    for _ in 0..other {
                        ubuf[i] *= rs[rs_off + i * inc];
                    }
                }
            }
        }

        for i in 0..nrs_now {
            ur[i] += ubuf[i];
        }
    }

    nrs_max
}

/// Host port of PySCF `type1_rad_part` — assemble the Type-1 radial table
/// `rad_all[lab*lmax1 + i]`.
///
/// `rad_all` is `(lmax+1)*(lmax+1)` long and is accumulated into (not zeroed).
/// `k` and `aij` are the Type-1 geometry; `ur` is the radial-part accumulator
/// from [`ecprad_part_host`]; `rs` the radial grid (strided by `inc`). Early
/// returns on `nrs == 0`. The `CUTOFF` / `EXPCUTOFF+6+30` guards zero a grid
/// point's `rur`/`bval`; otherwise `rur[n] = ur[n]*exp(tmp)` and
/// `bval[n*lmax1..] = ecpsph_ine_opt(lmax, k*rs[n])`. The `lab` loop multiplies
/// `rur` by `rs` for `lab>0` and accumulates with the `i = lab%2; i+=2` parity
/// stride.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5754-5806
#[allow(clippy::too_many_arguments)]
pub fn type1_rad_part_host(
    rad_all: &mut [f64],
    lmax: usize,
    k: f64,
    aij: f64,
    ur: &[f64],
    rs: &[f64],
    nrs: usize,
    inc: usize,
) {
    if nrs == 0 {
        return;
    }

    let lmax1 = lmax + 1;
    let mut rur = vec![0.0f64; nrs];
    let mut bval = vec![0.0f64; nrs * lmax1];

    let kaij = k / (2.0 * aij);
    let fac = kaij * kaij * aij;
    for n in 0..nrs {
        let mut tmp = rs[n * inc] - kaij;
        tmp = fac - aij * tmp * tmp;
        if ur[n] == 0.0 || !(-(EXPCUTOFF + 6.0 + 30.0)..=CUTOFF).contains(&tmp) {
            rur[n] = 0.0;
            for i in 0..lmax1 {
                bval[n * lmax1 + i] = 0.0;
            }
        } else {
            rur[n] = ur[n] * tmp.exp();
            let b = ecpsph_ine_opt_host(lmax as u32, k * rs[n * inc]);
            bval[n * lmax1..n * lmax1 + lmax1].copy_from_slice(&b);
        }
    }

    for lab in 0..=lmax {
        if lab > 0 {
            for n in 0..nrs {
                rur[n] *= rs[n * inc];
            }
        }
        let prad = &mut rad_all[lab * lmax1..lab * lmax1 + lmax1];
        let mut i = lab % 2;
        while i <= lmax {
            let mut s = prad[i];
            for n in 0..nrs {
                s += rur[n] * bval[n * lmax1 + i];
            }
            prad[i] = s;
            i += 2;
        }
    }
}

/// Host port of PySCF `type2_facs_rad` — build the Type-2 radial factors for one
/// `(ish, lc, rca)` triple.
///
/// `facs` is the output radial-factor buffer of length `nrs * lilc1 * nc`
/// (`lilc1 = li + lc + 1`, `nc` = number of contractions). For each primitive
/// `ip` and grid point `i` the buffer holds `exp(-a_i (r-rca)^2) *
/// ecpsph_ine_opt(li+lc, 2 a_i rca r)` (zeroed when `a_i (r-rca)^2 > EXPCUTOFF+6`),
/// then the primitive axis is contracted by the coefficient matrix `ci`
/// (`np x nc`) — the `dgemm_(N,N, m=nrs*lilc1, nc, np, ...)` in the C, replicated
/// here as a plain Rust matmul. Early-returns on `nrs == 0`.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5134-5186
#[allow(clippy::too_many_arguments)]
pub fn type2_facs_rad_host(
    facs: &mut [f64],
    li: usize,
    lc: usize,
    rca: f64,
    exponents_i: &[f64],
    coeffs_i: &[f64],
    nc: usize,
    rs: &[f64],
    nrs: usize,
    inc: usize,
) {
    if nrs == 0 {
        return;
    }

    let np = exponents_i.len();
    let lilc1 = li + lc + 1;
    let mut r2 = vec![0.0f64; nrs];
    // buf is column-major (m = nrs*lilc1) x np, matching the C MALLOC_INSTACK
    // layout `buf[np*nrs*lilc1]` filled primitive-major.
    let m = nrs * lilc1;
    let mut buf = vec![0.0f64; m * np];

    for i in 0..nrs {
        let t1 = rs[i * inc] - rca;
        r2[i] = t1 * t1;
    }

    for ip in 0..np {
        let ai = exponents_i[ip];
        let ka = 2.0 * ai * rca;
        for i in 0..nrs {
            let pbuf_off = ip * (nrs * lilc1) + i * lilc1;
            let ar2 = ai * r2[i];
            if ar2 > EXPCUTOFF + 6.0 {
                for j in 0..=(li + lc) {
                    buf[pbuf_off + j] = 0.0;
                }
            } else {
                let t1 = (-ar2).exp();
                let pb = ecpsph_ine_opt_host((li + lc) as u32, ka * rs[i * inc]);
                for j in 0..=(li + lc) {
                    buf[pbuf_off + j] = pb[j] * t1;
                }
            }
        }
    }

    // facs[m x nc] = buf[m x np] * ci[np x nc]  (dgemm_ N,N), column-major.
    for col in 0..nc {
        for row in 0..m {
            let mut s = 0.0f64;
            for p in 0..np {
                s += buf[p * m + row] * coeffs_i[col * np + p];
            }
            facs[col * m + row] = s;
        }
    }
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

    /// `ecprad_part_host` over a single Local ECP shell (radial_power=2, one
    /// Gaussian primitive) reproduces `sum_k c_k exp(-a_k r^2) * r^2`
    /// element-wise (pins the RADI_POWER=2 case + the Gaussian sum).
    #[test]
    fn ecprad_part_radi_power_2_single_gaussian() {
        let rs = vec![0.1, 0.5, 1.0, 1.5, 2.0];
        let nrs = rs.len();
        let ck = 1.7f64;
        let ak = 0.8f64;
        let shells = vec![EcpRadShell {
            exponents: vec![ak],
            coefficients: vec![ck],
            radial_power: 2,
        }];
        let mut ur = vec![0.0f64; nrs];
        let nrs_max = ecprad_part_host(&mut ur, &rs, 0, nrs, 1, &shells);
        assert_eq!(nrs_max, nrs);
        for i in 0..nrs {
            let r2 = rs[i] * rs[i];
            let expected = ck * (-ak * r2).exp() * r2;
            assert!(
                (ur[i] - expected).abs() <= 1e-14,
                "i={i}: got {} expected {}",
                ur[i],
                expected
            );
        }
    }

    /// `ecprad_part_host` honors the SIM_ZERO early-break: with a steep exponent
    /// the Gaussian underflows below SIM_ZERO after a few points, the break
    /// fires (nrs_max < nrs), and the trailing ur entries stay 0.
    #[test]
    fn ecprad_part_sim_zero_early_break() {
        // Steep, well-separated grid so exp(-a r^2) drops below 1e-50 fast.
        let rs = vec![0.1, 0.2, 0.3, 12.0, 13.0, 14.0, 15.0];
        let nrs = rs.len();
        let shells = vec![EcpRadShell {
            exponents: vec![3.0],
            coefficients: vec![1.0],
            radial_power: 1,
        }];
        let mut ur = vec![0.0f64; nrs];
        let nrs_max = ecprad_part_host(&mut ur, &rs, 0, nrs, 1, &shells);
        // The break fired before consuming the whole grid.
        assert!(nrs_max < nrs, "expected early break, got nrs_max={nrs_max}");
        // Trailing ur entries (beyond the effective length) remain 0.
        for i in nrs_max..nrs {
            assert_eq!(ur[i], 0.0, "trailing ur[{i}] should stay 0");
        }
    }

    /// `type1_rad_part_host` for a single primitive pair with k/aij chosen so the
    /// CUTOFF guard does NOT fire produces rad_all consistent with
    /// rur[n] = ur[n]*exp(tmp) and bval = ecpsph_ine_opt_host(lmax, k*rs[n]).
    #[test]
    fn type1_rad_part_internal_consistency() {
        let lmax = 2usize;
        let lmax1 = lmax + 1;
        let k = 1.3f64;
        let aij = 0.9f64;
        let rs = vec![0.2, 0.6, 1.1, 1.7];
        let nrs = rs.len();
        let ur = vec![0.7, 0.5, 0.3, 0.15];

        let mut rad_all = vec![0.0f64; lmax1 * lmax1];
        type1_rad_part_host(&mut rad_all, lmax, k, aij, &ur, &rs, nrs, 1);

        // Recompute the reference from the ported pieces.
        let kaij = k / (2.0 * aij);
        let fac = kaij * kaij * aij;
        let mut rur = vec![0.0f64; nrs];
        let mut bval = vec![0.0f64; nrs * lmax1];
        for n in 0..nrs {
            let mut tmp = rs[n] - kaij;
            tmp = fac - aij * tmp * tmp;
            if ur[n] == 0.0 || !(-(EXPCUTOFF + 6.0 + 30.0)..=CUTOFF).contains(&tmp) {
                rur[n] = 0.0;
            } else {
                rur[n] = ur[n] * tmp.exp();
                let b = ecpsph_ine_opt_host(lmax as u32, k * rs[n]);
                bval[n * lmax1..n * lmax1 + lmax1].copy_from_slice(&b);
            }
        }
        let mut expected = vec![0.0f64; lmax1 * lmax1];
        for lab in 0..=lmax {
            if lab > 0 {
                for n in 0..nrs {
                    rur[n] *= rs[n];
                }
            }
            let mut i = lab % 2;
            while i <= lmax {
                let mut s = 0.0f64;
                for n in 0..nrs {
                    s += rur[n] * bval[n * lmax1 + i];
                }
                expected[lab * lmax1 + i] = s;
                i += 2;
            }
        }

        for idx in 0..(lmax1 * lmax1) {
            assert_eq!(
                rad_all[idx], expected[idx],
                "rad_all[{idx}] {} != expected {}",
                rad_all[idx], expected[idx]
            );
        }
    }

    /// `type1_rad_part_host` leaves rad_all all-zero when nrs == 0 (early return).
    #[test]
    fn type1_rad_part_zero_grid_early_return() {
        let lmax = 3usize;
        let lmax1 = lmax + 1;
        let mut rad_all = vec![0.0f64; lmax1 * lmax1];
        let rs: Vec<f64> = Vec::new();
        let ur: Vec<f64> = Vec::new();
        type1_rad_part_host(&mut rad_all, lmax, 1.0, 1.0, &ur, &rs, 0, 1);
        assert!(
            rad_all.iter().all(|&v| v == 0.0),
            "nrs==0 must leave rad_all zero"
        );
    }

    /// `type2_facs_rad_host` early-returns (no panic, no writes) when nrs == 0,
    /// and for a single contraction reproduces the buf*ci matmul internally.
    #[test]
    fn type2_facs_rad_consistency_and_zero_grid() {
        // nrs == 0 early return: facs untouched.
        let mut facs_empty = vec![0.0f64; 4];
        let rs_empty: Vec<f64> = Vec::new();
        type2_facs_rad_host(
            &mut facs_empty,
            1,
            1,
            0.5,
            &[1.0],
            &[1.0],
            1,
            &rs_empty,
            0,
            1,
        );
        assert!(facs_empty.iter().all(|&v| v == 0.0));

        // Internal consistency for li=1, lc=1, single primitive, single contraction.
        let li = 1usize;
        let lc = 1usize;
        let lilc1 = li + lc + 1;
        let rca = 0.4f64;
        let rs = vec![0.3, 0.9, 1.6];
        let nrs = rs.len();
        let exponents = vec![0.75f64];
        let coeffs = vec![1.25f64]; // np=1, nc=1
        let nc = 1usize;
        let m = nrs * lilc1;
        let mut facs = vec![0.0f64; m * nc];
        type2_facs_rad_host(&mut facs, li, lc, rca, &exponents, &coeffs, nc, &rs, nrs, 1);

        // Reference: same buf fill then trivial 1x1 contraction.
        let ai = exponents[0];
        let ka = 2.0 * ai * rca;
        let mut expected = vec![0.0f64; m * nc];
        for i in 0..nrs {
            let t = rs[i] - rca;
            let ar2 = ai * t * t;
            let off = i * lilc1;
            if ar2 > EXPCUTOFF + 6.0 {
                for j in 0..=(li + lc) {
                    expected[off + j] = 0.0;
                }
            } else {
                let scal = (-ar2).exp();
                let pb = ecpsph_ine_opt_host((li + lc) as u32, ka * rs[i]);
                for j in 0..=(li + lc) {
                    expected[off + j] = pb[j] * scal * coeffs[0];
                }
            }
        }
        for idx in 0..(m * nc) {
            assert_eq!(facs[idx], expected[idx], "facs[{idx}] mismatch");
        }
    }
}
