//! Host-side Effective Core Potential (ECP) integral kernel — Type-1 (local,
//! Coulomb-like) and Type-2 (semi-local, projector-based) scalar integrals.
//!
//! # K-Taylor port (Phase 19 replan)
//!
//! This module ports PySCF nr_ecp's *exact* scalar ECP evaluation so the
//! `int1e_ecp_{cart,sph}` operators reach `atol=1e-12, rtol=0.0` byte-identity
//! vs the vendored C (`ECPscalar_cart` / `ECPscalar_sph`). It replaces the
//! 19-04 numerical-quadrature approximation (Gauss-Hermite for Type-1,
//! Gauss-Chebyshev + the generic unscaled `i_l` for Type-2) with the 19-05
//! K-Taylor radial machinery plus PySCF's per-`(li, lj, l_c)` angular splice.
//!
//! Ported C drivers (each helper carries a `// Source:` citation):
//! - `ECPtype1_cart`  — vendor/pyscf-nr-ecp/src/nr_ecp.c:5808-5991. Calls
//!   `ECPrad_part` (5910) then `type1_rad_part` (5931), the level-adaptive
//!   Gauss-Chebyshev convergence loop (5909-5952), `type1_rad_ang` (5960),
//!   `type1_static_facs` (5971-5972), and the `LOOP_CART`/`LOOP_XYZ` angular
//!   accumulation into `gctr` (5976-5988).
//! - `ECPtype2_cart`  — vendor/pyscf-nr-ecp/src/nr_ecp.c:5337-5515. Calls
//!   `ECPrad_part`, `type2_facs_rad` (5450/5452), the same level-adaptive loop,
//!   `type2_facs_ang` (5498/5499), and the two-`dgemm` angular splice
//!   (5505-5510).
//! - `ECPscalar_{cart,sph}` — vendor/pyscf-nr-ecp/src/nr_ecp.c:4687-4837 (the
//!   K-Taylor `ECPsph_ine_opt`, ported in 19-05) feeds the radial assemblers;
//!   the combined wrapper sums Type-1 + Type-2 across the ecpbas slab.
//!
//! The radial primitives (`ecprad_part_host`, `type1_rad_part_host`,
//! `type2_facs_rad_host`, `ecpsph_ine_opt_host`) live in
//! `crate::math::ecp_k_taylor` (19-05 output). The angular helpers
//! (`int_unit_xyz`, `ang_nuc_in_cart`, `cache_3dfac`, `type1_static_facs`,
//! `type1_rad_ang`, `type2_facs_ang`, `scale_coeff`) are ported here.
//!
//! ## CubeCL device/host split (CLAUDE.md "CubeCL is the primary backend")
//!
//! As of quick-260529-gbf the Type-1 **angular splice** (the bounded
//! triple-product accumulation `acc += ifac*jfac*rad_ang` at the tail of
//! `ecp_type1_cart`) runs on-device as a real `#[cube(launch)]` kernel
//! ([`ecp_angular_kernel`]) **generic over `F: Float`**, dispatched onto the
//! resolved backend's `ComputeClient` (CPU `CpuRuntime`, ROCm `HipRuntime`, …)
//! via [`run_ecp_angular_device`] / [`run_ecp_angular_splice_on_backend`],
//! mirroring center_2c2e / center_4c1e. It launches at f64 with the SAME nested
//! summation order as the prior host loop, so the byte-identity gate
//! (atol=1e-12/rtol=0.0 CPU-vs-C) is preserved.
//!
//! The **adaptive radial machinery** (the level-adaptive Gauss-Chebyshev
//! convergence loop, the special-function modified-spherical-Bessel /
//! K-Taylor recurrences in `ecp_k_taylor.rs`, the `type1_static_facs` /
//! `type2_facs_ang` angular-factor tables) STAYS host-side as planning /
//! validation / marshaling — it has data-dependent termination, dynamic loop
//! bounds and special functions that the Phase 8 P02 `cond_br` MLIR limitation
//! forbids under `#[cube]`. Those helpers feed flat f64 buffers into the device
//! kernel. As of quick-260529-hin the Type-2 **angular splice** (the two-`dgemm`
//! matmul at the tail of `ecp_type2_cart`) ALSO runs on-device via
//! [`run_ecp_type2_splice_on_backend`] / [`ecp_type2_angular_kernel`] — generic
//! over `F: Float`, launched at f64 with the same dgemm summation order
//! (recomputing each intermediate `buf` entry inline, no device-local scratch) so
//! byte-identity is preserved. Only the Phase-A adaptive radial machinery stays
//! host-side as marshaling. The Cu/LANL2DZ oracle drives BOTH the Local (Type-1)
//! and Projected (Type-2) device kernels on every case.
//!
//! ## Normalization & coordinate convention
//!
//! PySCF's primitive loop reads AO contraction coefficients VERBATIM from
//! `env[PTR_COEFF + p]`; the per-primitive Gaussian normalization `N(a, l)` is
//! embedded in the coefficients at basis-set build time. `scale_coeff`
//! (nr_ecp.c:5740) additionally folds in `exp(-a |r_CA|^2) * CINTcommon_fac_sp(l)
//! * 4 pi` per AO primitive. The kernel reads `shell.coefficients` directly
//! (no extra `N` factor) matching `launch_one_electron`.
//!
//! The ECP center `R_C` (stored on its atom) enters the radial integral via the
//! AO-center displacements `rca = R_C - R_A`, `rcb = R_C - R_B` exactly as the
//! C reference computes them.
//!
//! Both Type-1 and Type-2 accumulate into a Cartesian buffer `[ao_i, ao_j]`
//! (F-order, ao_i fastest-varying within a contraction block, matching
//! `launch_one_electron` line 434). `cart_to_sph_1e` is applied when
//! `plan.representation == Representation::Spheric`.
//!
//! # Gradient operator (`operator_name == "ecp_ipnuc"`) — Plan 19-07
//!
//! The gradient arm ports PySCF nr_ecp_deriv.c's `_deriv1_cart`
//! (vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:201-286) — the 3-component
//! `∂/∂A_i^{x,y,z}` derivative of the ECP integral w.r.t. the *bra* AO center
//! `A_i` (matching libcint/PySCF `ipnuc` = derivative on the first index;
//! `ECPscalar_ipnuc_cart` line 366 / `ECPscalar_ipnuc_sph` line 453). The
//! derivative recurrence is the standard GTO identity
//!   `∂/∂A χ_l = 2 a_i · χ_{l+1} − l · χ_{l-1}`
//! implemented by `_l_down` (the `2 a_i χ_{l+1}` term, nr_ecp_deriv.c:148-172)
//! and `_l_up` (the `l χ_{l-1}` term, nr_ecp_deriv.c:174-199). Both reuse the
//! *same* scalar `ecp_type1_cart` / `ecp_type2_cart` drivers (no new radial
//! machinery — `_deriv1_cart` just re-evaluates them at `l ± 1` on
//! per-primitive uncontracted "fake" shells, dividing out `expi·expj` and
//! re-applying the AO contraction coefficients afterward). See
//! `compute_type1_pair_grad` / `compute_type2_pair_grad` (both fold into the
//! unified `deriv1_cart_pair` driver since `_deriv1_cart` sums Type-1+Type-2
//! into one buffer per fake shell).
//!
//! ## Component layout (D-11)
//!
//! PySCF writes the 3 components as `[comp, dij]` with `comp` slowest-varying
//! (`gctrx`, `gctry`, `gctrz` are three consecutive `dij`-blocks, line 240-245),
//! and inside each block the `dij` layout is F-order `n + j*di + i` = `[ao_j,
//! ao_i]` with `ao_i` fastest (line 278-280). cintx's required gradient layout
//! is F-order `[axis ∈ {x,y,z}, ao_j, ao_i]` (axis slowest) — the int3c2e_ip1_*
//! precedent (CONTEXT D-11). These are IDENTICAL: PySCF's comp == cintx's axis
//! (same {x,y,z} order), and PySCF's per-block `[ao_j, ao_i]` == cintx's
//! `[ao_j, ao_i]`. No transpose is required in either the kernel or the
//! collector (resolved by reading nr_ecp_deriv.c, not assumed).

use crate::backend::ResolvedBackend;
use crate::math::ecp_k_taylor::{
    EcpRadShell, ecprad_part_host, type1_rad_part_host, type2_facs_rad_host,
};
use crate::math::radial_quadrature::{LEVEL_MAX, LEVEL0, gauss_chebyshev_nodes_weights_host};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::ecp::{EcpChannel, EcpShell};
use cintx_core::shell::Shell;
use cintx_core::{Atom, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::sync::Arc;

/// `ECP_LMAX` — maximum ECP projector angular momentum (Phase 19 envelope).
/// Source: vendor/pyscf-nr-ecp/include/gto/nr_ecp.h:10 (`ECP_LMAX 5`).
const ECP_LMAX: usize = 5;

/// `CLOSE_ENOUGH(x, y)` — PySCF radial convergence predicate.
/// Source: vendor/pyscf-nr-ecp/include/gto/nr_ecp.h:16.
#[inline]
fn close_enough(x: f64, y: f64) -> bool {
    (x - y).abs() < 1e-12 * y.abs() || (x - y).abs() < 1e-12
}

/// `CINTcommon_fac_sp(l)` — libcint s/p common factor.
/// Source: libcint-master/src/g1e.c:566-573.
#[inline]
fn cint_common_fac_sp(l: usize) -> f64 {
    match l {
        0 => 0.282094791773878143,
        1 => 0.488602511902919921,
        _ => 1.0,
    }
}

/// `_factorial2[n] = n!!` — used by `int_unit_xyz`. `factorial2(n<0) = 1`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4473-4486 + 4488-4496.
const FACTORIAL2: [f64; 40] = [
    1.,
    1.,
    2.,
    3.,
    8.,
    15.,
    48.,
    105.,
    384.,
    945.,
    3840.,
    10395.,
    46080.,
    135135.,
    645120.,
    2027025.,
    10321920.,
    34459425.,
    185794560.,
    654729075.,
    3715891200.,
    13749310575.,
    81749606400.,
    316234143225.,
    1961990553600.,
    7905853580625.,
    51011754393600.,
    213458046676875.,
    1428329123020800.,
    6190283353629376.,
    42849873690624000.,
    1.9189878396251069e+17,
    1.371195958099968e+18,
    6.3326598707628524e+18,
    4.6620662575398912e+19,
    2.2164309547669976e+20,
    1.6783438527143608e+21,
    8.2007945326378929e+21,
    6.3777066403145712e+22,
    3.1983098677287775e+23,
];

/// `factorial2(n)` with the `n < 0 => 1` guard (PySCF nr_ecp.c:4488-4496).
#[inline]
fn factorial2(n: i32) -> f64 {
    if n < 0 { 1.0 } else { FACTORIAL2[n as usize] }
}

/// `_binom[n][m]` for `n < 10`; PySCF `binom(n, m)`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4498-4517.
const BINOM: [f64; 55] = [
    1., 1., 1., 1., 2., 1., 1., 3., 3., 1., 1., 4., 6., 4., 1., 1., 5., 10., 10., 5., 1., 1., 6.,
    15., 20., 15., 6., 1., 1., 7., 21., 35., 35., 21., 7., 1., 1., 8., 28., 56., 70., 56., 28., 8.,
    1., 1., 9., 36., 84., 126., 126., 84., 36., 9., 1.,
];

#[inline]
fn binom(n: usize, m: usize) -> f64 {
    // Phase-19 envelope keeps n < 10 (li + lj <= 4); the n >= 10 factorial form
    // is unreachable here, so the table is sufficient.
    BINOM[n * (n + 1) / 2 + m]
}

/// `_offset_cart[l]` — cumulative cartesian-component offset.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4543-4544.
const OFFSET_CART: [usize; 15] = [
    0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560,
];

// ─────────────────────────────────────────────────────────────────────────────
// Cartesian-component enumeration (libcint LOOP_CART order).
// Verified bit-equal to PySCF `_cart_pow_y` / `_cart_pow_z` for l <= 4.
// ─────────────────────────────────────────────────────────────────────────────

fn cart_comps(l: u8) -> Vec<(u8, u8, u8)> {
    let mut comps = Vec::new();
    let l = l as i32;
    let mut lx = l;
    while lx >= 0 {
        let mut ly = l - lx;
        while ly >= 0 {
            let lz = l - lx - ly;
            comps.push((lx as u8, ly as u8, lz as u8));
            ly -= 1;
        }
        lx -= 1;
    }
    comps
}

// ─────────────────────────────────────────────────────────────────────────────
// Angular helpers (ported verbatim from nr_ecp.c).
// ─────────────────────────────────────────────────────────────────────────────

/// `int_unit_xyz(i, j, k)` — analytic angular integral of `x^i y^j z^k` over
/// the unit sphere (returns 0 for any odd power).
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4952-4960.
fn int_unit_xyz(i: i32, j: i32, k: i32) -> f64 {
    if i % 2 != 0 || j % 2 != 0 || k % 2 != 0 {
        0.0
    } else {
        factorial2(i - 1) * factorial2(j - 1) * factorial2(k - 1) / factorial2(i + j + k + 1)
    }
}

/// Project a cartesian-monomial coefficient vector `omega` (length `ncart(l)`)
/// onto its pure spherical-harmonic component — `omega <- M^T (M omega)` where
/// `M = g_c2s[l].cart2sph` (cintx `C2S_L*`). For `l < 2` the libcint transform
/// is the identity (s, p) so `omega` is unchanged.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4998-4999 (CINTc2s_bra_sph +
/// CINTs2c_bra_sph round-trip); libcint cart2sph.c:6476/6965 + fblas.c
/// (CINTdgemm_TN / CINTdgemm_NN1).
fn purify_cart_to_sph(omega: &mut [f64], l: usize) {
    if l < 2 {
        return; // s_bra/p_bra cart2spheric are identity (no PYPZPX)
    }
    let nf = ncart(l as u8);
    let nd = nsph(l as u8);
    // buf = M . omega
    let mut buf = vec![0.0f64; nd];
    for (m_row, b) in buf.iter_mut().enumerate() {
        let mut s = 0.0;
        for (c, &w) in omega.iter().enumerate().take(nf) {
            s += c2s_coeff(l, m_row, c) * w;
        }
        *b = s;
    }
    // omega = M^T . buf
    for (c, ow) in omega.iter_mut().enumerate().take(nf) {
        let mut s = 0.0;
        for (m_row, &bv) in buf.iter().enumerate() {
            s += c2s_coeff(l, m_row, c) * bv;
        }
        *ow = s;
    }
}

/// `ang_nuc_in_cart(omega, l, r)` — cartesian angular nuclear factors with the
/// hard-coded l=0/l=1 fast paths and the cart->sph->cart purification for l>=2.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:4966-5001.
fn ang_nuc_in_cart(omega: &mut [f64], l: usize, r: &[f64; 3]) {
    match l {
        0 => {
            omega[0] = 0.07957747154594767;
        }
        1 => {
            omega[0] = r[0] * 0.2387324146378430;
            omega[1] = r[1] * 0.2387324146378430;
            omega[2] = r[2] * 0.2387324146378430;
        }
        _ => {
            let mut xx = [0.0f64; 16];
            let mut yy = [0.0f64; 16];
            let mut zz = [0.0f64; 16];
            xx[0] = 1.0;
            yy[0] = 1.0;
            zz[0] = 1.0;
            for i in 1..=l {
                xx[i] = xx[i - 1] * r[0];
                yy[i] = yy[i - 1] * r[1];
                zz[i] = zz[i - 1] * r[2];
            }
            let mut n = 0usize;
            let li = l as i32;
            let mut i = li;
            while i >= 0 {
                let mut j = li - i;
                while j >= 0 {
                    let k = li - i - j;
                    omega[n] = xx[i as usize] * yy[j as usize] * zz[k as usize];
                    n += 1;
                    j -= 1;
                }
                i -= 1;
            }
            purify_cart_to_sph(omega, l);
        }
    }
}

/// `cache_3dfac(facs, l, r)` — per-axis binomial expansion factors
/// `fac[axis][i*l1 + j] = binom(i, j) * r[axis]^(i-j)`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5003-5031.
fn cache_3dfac(facs: &mut [f64], l: usize, r: &[f64; 3]) {
    let l1 = l + 1;
    let (facx, rest) = facs.split_at_mut(l1 * l1);
    let (facy, facz) = rest.split_at_mut(l1 * l1);
    let mut xx = [0.0f64; 16];
    let mut yy = [0.0f64; 16];
    let mut zz = [0.0f64; 16];
    xx[0] = 1.0;
    yy[0] = 1.0;
    zz[0] = 1.0;
    for i in 1..=l {
        xx[i] = xx[i - 1] * r[0];
        yy[i] = yy[i - 1] * r[1];
        zz[i] = zz[i - 1] * r[2];
    }
    for i in 0..=l {
        for j in 0..=i {
            let bfac = binom(i, j);
            let off = i * l1 + j;
            facx[off] = bfac * xx[i - j];
            facy[off] = bfac * yy[i - j];
            facz[off] = bfac * zz[i - j];
        }
    }
}

/// `type1_static_facs(facs, li, ri)` — static cartesian factors
/// `facs[mi*d3 + i*d2 + j*d1 + k]` for the Type-1 angular accumulation.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5188-5209.
fn type1_static_facs(facs: &mut [f64], li: usize, ri: &[f64; 3]) {
    let d1 = li + 1;
    let d2 = d1 * d1;
    let d3 = d2 * d1;
    let mut fac3d = vec![0.0f64; 3 * d1 * d1];
    cache_3dfac(&mut fac3d, li, ri);
    let (fac3dx, rest) = fac3d.split_at(d1 * d1);
    let (fac3dy, fac3dz) = rest.split_at(d1 * d1);

    for (mi, (px, py, pz)) in cart_comps(li as u8).into_iter().enumerate() {
        let (px, py, pz) = (px as usize, py as usize, pz as usize);
        let pfacs = &mut facs[mi * d3..mi * d3 + d3];
        for i in 0..=px {
            for j in 0..=py {
                for k in 0..=pz {
                    pfacs[i * d2 + j * d1 + k] =
                        fac3dx[px * d1 + i] * fac3dy[py * d1 + j] * fac3dz[pz * d1 + k];
                }
            }
        }
    }
}

/// `type1_rad_ang(rad_ang, lmax, r, rad_all)` — combine the Type-1 radial
/// moments with the angular nuclear factors.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5211-5257.
fn type1_rad_ang(rad_ang: &mut [f64], lmax: usize, r: &[f64; 3], rad_all: &[f64]) {
    let mut unitr = [0.0f64; 3];
    if !(r[0] == 0.0 && r[1] == 0.0 && r[2] == 0.0) {
        let norm_r = -1.0 / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        unitr[0] = r[0] * norm_r;
        unitr[1] = r[1] * norm_r;
        unitr[2] = r[2] * norm_r;
    }

    // omega_nuc holds ang_nuc_in_cart for each l = 0..=lmax, packed at OFFSET_CART.
    let mut omega_nuc = vec![0.0f64; OFFSET_CART[lmax + 1]];
    for i in 0..=lmax {
        let off = OFFSET_CART[i];
        let nf = ncart(i as u8);
        ang_nuc_in_cart(&mut omega_nuc[off..off + nf], i, &unitr);
    }

    let d1 = lmax + 1;
    let d2 = d1 * d1;
    let d3 = d2 * d1;
    for v in rad_ang.iter_mut().take(d3) {
        *v = 0.0;
    }
    for i in 0..=lmax {
        for j in 0..=(lmax - i) {
            for k in 0..=(lmax - i - j) {
                let pout_idx = i * d2 + j * d1 + k;
                let prad = &rad_all[(i + j + k) * d1..(i + j + k) * d1 + d1];
                let need_even = (i + j + k) % 2;
                let mut lmb = need_even;
                let mut acc = 0.0;
                while lmb <= lmax {
                    let mut tmp = 0.0;
                    let pnuc = &omega_nuc[OFFSET_CART[lmb]..];
                    for (n, (pr, ps, pt)) in cart_comps(lmb as u8).into_iter().enumerate() {
                        tmp += pnuc[n]
                            * int_unit_xyz(
                                (i + pr as usize) as i32,
                                (j + ps as usize) as i32,
                                (k + pt as usize) as i32,
                            );
                    }
                    acc += prad[lmb] * tmp;
                    lmb += 2;
                }
                rad_ang[pout_idx] += acc;
            }
        }
    }
}

/// `type2_facs_ang(facs, li, lc, ri)` — the Type-2 angular factor tensor
/// `facs[(a+b+c)*nfi*dlclmb + mi*dlclmb + m*dlambda + n]`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5033-5132.
fn type2_facs_ang(facs: &mut [f64], li: usize, lc: usize, ri: &[f64; 3]) {
    let mut unitr = [0.0f64; 3];
    if !(ri[0] == 0.0 && ri[1] == 0.0 && ri[2] == 0.0) {
        let norm_ri = -1.0 / (ri[0] * ri[0] + ri[1] * ri[1] + ri[2] * ri[2]).sqrt();
        unitr[0] = ri[0] * norm_ri;
        unitr[1] = ri[1] * norm_ri;
        unitr[2] = ri[2] * norm_ri;
    }

    let li1 = li + 1;
    let dlc = lc * 2 + 1;
    let dlambda = li + lc + 1;

    let mut omega_nuc = vec![0.0f64; OFFSET_CART[li + lc + 1]];
    for i in 0..=(li + lc) {
        let off = OFFSET_CART[i];
        let nf = ncart(i as u8);
        ang_nuc_in_cart(&mut omega_nuc[off..off + nf], i, &unitr);
    }
    for v in omega_nuc.iter_mut().take(OFFSET_CART[li + lc + 1]) {
        *v *= 4.0 * std::f64::consts::PI;
    }

    let dlclmb = dlambda * dlc;
    // omega[(i*li1*li1 + j*li1 + k)*dlclmb + lmb*dlc + m]
    let mut omega = vec![0.0f64; li1 * li1 * li1 * dlclmb];
    let mut buf = vec![0.0f64; dlc.max(1)];

    for i in 0..=li {
        for j in 0..=(li - i) {
            for k in 0..=(li - i - j) {
                let need_even = (lc + i + j + k) % 2;
                let base = (i * li1 * li1 + j * li1 + k) * dlclmb;
                let mut lmb = need_even;
                // even-parity lambda block
                while lmb <= li + lc {
                    let pnuc = &omega_nuc[OFFSET_CART[lmb]..];
                    for (m, (pu, pv, pw)) in cart_comps(lc as u8).into_iter().enumerate() {
                        let mut s = 0.0;
                        for (n, (pr, ps, pt)) in cart_comps(lmb as u8).into_iter().enumerate() {
                            s += pnuc[n]
                                * int_unit_xyz(
                                    (i + pu as usize + pr as usize) as i32,
                                    (j + pv as usize + ps as usize) as i32,
                                    (k + pw as usize + pt as usize) as i32,
                                );
                        }
                        buf[m] = s;
                    }
                    let pomega_off = base + lmb * dlc;
                    match lc {
                        0 => {
                            omega[pomega_off] = buf[0] * 0.282094791773878143;
                        }
                        1 => {
                            omega[pomega_off] = buf[0] * 0.488602511902919921;
                            omega[pomega_off + 1] = buf[1] * 0.488602511902919921;
                            omega[pomega_off + 2] = buf[2] * 0.488602511902919921;
                        }
                        _ => {
                            // CINTc2s_bra_sph(pomega, 1, buf, lc): pomega[m_sph] =
                            // sum_cart M[m_sph][cart] * buf[cart].
                            for m_row in 0..dlc {
                                let mut s = 0.0;
                                for (c, &bv) in buf.iter().enumerate().take(ncart(lc as u8)) {
                                    s += c2s_coeff(lc, m_row, c) * bv;
                                }
                                omega[pomega_off + m_row] = s;
                            }
                        }
                    }
                    lmb += 2;
                }
                // odd-parity lambda block is zeroed (omega already 0-init).
            }
        }
    }

    let nfi = ncart(li as u8);
    let mut fac3d = vec![0.0f64; 3 * li1 * li1];
    cache_3dfac(&mut fac3d, li, ri);
    let (fac3dx, rest) = fac3d.split_at(li1 * li1);
    let (fac3dy, fac3dz) = rest.split_at(li1 * li1);

    for v in facs.iter_mut().take(li1 * nfi * dlclmb) {
        *v = 0.0;
    }
    for (mi, (pr, ps, pt)) in cart_comps(li as u8).into_iter().enumerate() {
        let (pr, ps, pt) = (pr as usize, ps as usize, pt as usize);
        for i in 0..=pr {
            for j in 0..=ps {
                for k in 0..=pt {
                    let need_even = (lc + i + j + k) % 2;
                    let fac = fac3dx[pr * li1 + i] * fac3dy[ps * li1 + j] * fac3dz[pt * li1 + k];
                    let pomega_off = (i * li1 * li1 + j * li1 + k) * dlclmb;
                    let pfacs_off = ((i + j + k) * nfi + mi) * dlclmb;
                    for m in 0..dlc {
                        let mut n = need_even;
                        while n < dlambda {
                            facs[pfacs_off + m * dlambda + n] +=
                                fac * omega[pomega_off + n * dlc + m];
                            n += 2;
                        }
                    }
                }
            }
        }
    }
}

/// `scale_coeff(cei, ci, ai, r2ca, npi, nci, li)` — fold the AO-center radial
/// Gaussian prefactor and `CINTcommon_fac_sp(li) * 4 pi` into the contraction
/// coefficients. `cei[ic*npi + ip] = ci[ic*npi + ip] * exp(-ai[ip]*r2ca) * fac`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5740-5752.
fn scale_coeff(
    cei: &mut [f64],
    ci: &[f64],
    ai: &[f64],
    r2ca: f64,
    npi: usize,
    nci: usize,
    li: usize,
) {
    let common_fac = cint_common_fac_sp(li) * 4.0 * std::f64::consts::PI;
    for ip in 0..npi {
        let tmp = (-ai[ip] * r2ca).exp() * common_fac;
        for ic in 0..nci {
            cei[ic * npi + ip] = ci[ic * npi + ip] * tmp;
        }
    }
}

/// Transpose canonical `Shell.coefficients` from the workspace's row-major
/// `[ip * nctr + ic]` layout into the column-major `[ic * nprim + ip]` layout
/// that this ECP kernel's internal indexing assumes (`scale_coeff`, the
/// contiguous per-contraction slices in `ecp_type2_cart`, and the
/// `ci[ic*npi+ip]` reads in `deriv1_cart_pair`). All other cintx kernels
/// (`one_electron`, `two_electron`, `center_*`) read `coefficients[ip*nctr+ic]`
/// directly, so that row-major form is canonical; the ECP port was written
/// against libcint's per-contraction-contiguous convention. For `nctr == 1`
/// this is the identity (so single-contraction output is byte-unchanged); the
/// difference only matters for generally-contracted shells (e.g. LANL2DZ Cu).
fn coeffs_col_major(shell: &Shell) -> Vec<f64> {
    let np = shell.nprim as usize;
    let nc = shell.nctr as usize;
    let src = &shell.coefficients;
    let mut out = vec![0.0f64; np * nc];
    for ip in 0..np {
        for ic in 0..nc {
            out[ic * np + ip] = src[ip * nc + ic];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-shell data marshaling (typed EcpShell -> radial-grid inputs).
// ─────────────────────────────────────────────────────────────────────────────

/// One ECP "slot" — the group of `EcpShell`s on the same atom sharing
/// `(channel, so_type)`, mirroring PySCF's `_loc_ecpbas` (nr_ecp.c:5261).
struct EcpSlot<'a> {
    atom_index: u32,
    /// `-1` for Local (Type-1), `>=0` for Projected(l) (Type-2).
    lc: i32,
    /// The radial-shell groups summed inside `ECPrad_part`.
    rad_shells: Vec<EcpRadShell>,
    _marker: std::marker::PhantomData<&'a ()>,
}

/// Group `ecp_shells` into slots keyed by `(atom_index, channel, so_type)`,
/// preserving input order (`_loc_ecpbas` groups *consecutive* rows; the typed
/// fixture lists one shell per channel so each group is a single slot).
fn group_ecp_slots(ecp_shells: &[Arc<EcpShell>]) -> Vec<EcpSlot<'_>> {
    let mut slots: Vec<EcpSlot<'_>> = Vec::new();
    for ec in ecp_shells {
        let lc = match ec.channel {
            EcpChannel::Local => -1,
            EcpChannel::Projected(l) => l as i32,
        };
        let rad = EcpRadShell {
            exponents: ec.exponents.to_vec(),
            coefficients: ec.coefficients.to_vec(),
            radial_power: ec.radial_power as i32,
        };
        match slots.last_mut() {
            Some(s) if s.atom_index == ec.atom_index && s.lc == lc => {
                s.rad_shells.push(rad);
            }
            _ => slots.push(EcpSlot {
                atom_index: ec.atom_index,
                lc,
                rad_shells: vec![rad],
                _marker: std::marker::PhantomData,
            }),
        }
    }
    slots
}

/// Tolerance (bohr) for matching the `iprinv` rinv origin to an ECP-bearing
/// atom's coordinate. The safe API supplies a coordinate origin (not the
/// integer `env[AS_RINV_ORIG_ATOM]` index the vendor uses); a tight tolerance
/// keeps the selection unambiguous while absorbing float round-trip noise
/// through the env slab. Mirrors the per-nucleus selection intent of D-09.
const IPRINV_ORIGIN_MATCH_TOL: f64 = 1e-10;

/// `ECPscalar_iprinv` per-nucleus selector (D-09). Given the grouped ECP
/// `slots`, the molecule's `atoms`, and the `iprinv` rinv `origin`, return the
/// indices of the slots whose atom's `coord_bohr` matches `origin` within
/// [`IPRINV_ORIGIN_MATCH_TOL`].
///
/// This is the differentiator from `ecp_ipnuc`: `ipnuc` accumulates the
/// gradient over EVERY ECP slot (all atoms), while `iprinv` differentiates only
/// the single atom selected by the rinv origin and drops the all-slot/`-Z_C`
/// accumulation. The vendor `ECPscalar_iprinv_cart` (nr_ecp_deriv.c:333) does
/// the same via `env[AS_RINV_ORIG_ATOM]` + `_one_shell_ecpbas`; cintx matches by
/// coordinate because the safe API supplies a coordinate origin.
///
/// Returns an empty `Vec` when no atom matches the origin — mirroring the
/// vendor's `_one_shell_ecpbas` returning `shl_id < 0` (→ a zero-filled output),
/// never a panic and never selecting the wrong atom (T-21-07-02). A matched
/// atom may host MORE than one slot (Local + several Projected channels); ALL
/// of that atom's slots are selected (they all belong to the single selected
/// nucleus), which is still strictly per-nucleus.
fn select_iprinv_slots(slots: &[EcpSlot<'_>], atoms: &[Atom], origin: [f64; 3]) -> Vec<usize> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(idx, slot)| {
            let ac = slot.atom_index as usize;
            let coord = atoms.get(ac)?.coord_bohr;
            let d2 = (coord[0] - origin[0]).powi(2)
                + (coord[1] - origin[1]).powi(2)
                + (coord[2] - origin[2]).powi(2);
            (d2.sqrt() < IPRINV_ORIGIN_MATCH_TOL).then_some(idx)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-1 (Local) ECP driver — port of ECPtype1_cart.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5808-5991.
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn ecp_type1_cart(
    backend: &ResolvedBackend,
    gctr: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    rad_shells: &[EcpRadShell],
    rs_max: &[f64],
    ws_max: &[f64],
) {
    let li = shell_i.ang_momentum as usize;
    let lj = shell_j.ang_momentum as usize;
    let npi = shell_i.nprim as usize;
    let npj = shell_j.nprim as usize;
    let nci = shell_i.nctr as usize;
    let ncj = shell_j.nctr as usize;
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let ai = &shell_i.exponents;
    let aj = &shell_j.exponents;
    // Canonical Shell.coefficients are row-major [ip*nctr+ic]; this ECP kernel's
    // internal indexing is column-major [ic*nprim+ip]. Convert once (identity for
    // nctr==1). Without this, generally-contracted shells read scrambled coeffs.
    let ci_cm = coeffs_col_major(shell_i);
    let cj_cm = coeffs_col_major(shell_j);
    let ci = &ci_cm;
    let cj = &cj_cm;

    let lilj1 = li + lj + 1;
    let d1 = lilj1;
    let d2 = d1 * d1;
    let d3 = d2 * d1;
    let di1 = li + 1;
    let di2 = di1 * di1;
    let di3 = di2 * di1;
    let dj1 = lj + 1;
    let dj2 = dj1 * dj1;
    let dj3 = dj2 * dj1;

    let rca = [rc[0] - ri[0], rc[1] - ri[1], rc[2] - ri[2]];
    let rcb = [rc[0] - rj[0], rc[1] - rj[1], rc[2] - rj[2]];
    let sq_rca = rca[0] * rca[0] + rca[1] * rca[1] + rca[2] * rca[2];
    let sq_rcb = rcb[0] * rcb[0] + rcb[1] * rcb[1] + rcb[2] * rcb[2];

    let mut cei = vec![0.0f64; npi * nci];
    let mut cej = vec![0.0f64; npj * ncj];
    scale_coeff(&mut cei, ci, ai, sq_rca, npi, nci, li);
    scale_coeff(&mut cej, cj, aj, sq_rcb, npj, ncj, lj);

    // rad_all[(ip*npj+jp)*d2 + lab*d1 + i]
    let mut rad_all = vec![0.0f64; npi * npj * d2];

    // Level-adaptive Gauss-Chebyshev convergence loop (nr_ecp.c:5897-5952).
    let mut converged = vec![false; npi * npj];
    let mut nrs0 = (1usize << LEVEL0) - 1;
    let mut step = 1usize << (LEVEL_MAX - LEVEL0);
    let mut start = step - 1;
    let mut wtscale = step as f64;
    let mut plast = vec![0.0f64; d2];
    let mut ur = vec![0.0f64; 1usize << LEVEL_MAX];

    let mut level = LEVEL0;
    while level <= LEVEL_MAX {
        let nrs = ecprad_part_host(&mut ur, rs_max, start, nrs0, step, rad_shells);
        if nrs == 0 {
            break;
        }
        for n in 0..nrs {
            ur[n] *= ws_max[start + n * step] * wtscale;
        }

        let mut all_conv = true;
        for ip in 0..npi {
            for jp in 0..npj {
                if converged[ip * npj + jp] {
                    continue;
                }
                let prad = &mut rad_all[(ip * npj + jp) * d2..(ip * npj + jp) * d2 + d2];
                plast[..d2].copy_from_slice(prad);
                for v in prad.iter_mut() {
                    *v *= 0.5;
                }
                let rij = [
                    ai[ip] * rca[0] + aj[jp] * rcb[0],
                    ai[ip] * rca[1] + aj[jp] * rcb[1],
                    ai[ip] * rca[2] + aj[jp] * rcb[2],
                ];
                let k = (rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]).sqrt() * 2.0;
                type1_rad_part_host(
                    prad,
                    li + lj,
                    k,
                    ai[ip] + aj[jp],
                    &ur,
                    &rs_max[start..],
                    nrs,
                    step,
                );
                converged[ip * npj + jp] = true;
                for i in 0..d2 {
                    if !close_enough(plast[i], prad[i]) {
                        converged[ip * npj + jp] = false;
                        all_conv = false;
                        break;
                    }
                }
            }
        }

        if all_conv {
            break;
        }
        nrs0 = (1usize << level) - 1;
        step = 1usize << (LEVEL_MAX - level);
        // C uses signed int: (0 - 1) / 2 == 0. Mirror that exactly under usize
        // (saturating_sub avoids the Rust underflow panic at the finest level
        // when a non-converging far-field pair reaches start == 0; byte-identical
        // to the C arithmetic since the loop exits at level > LEVEL_MAX anyway).
        start = start.saturating_sub(1) / 2;
        wtscale *= 0.5;
        level += 1;
    }

    // rad_ang_all[(ic*ncj+jc)*d3]
    let mut rad_ang_all = vec![0.0f64; nci * ncj * d3];
    let mut rad_ang = vec![0.0f64; d3];
    for ip in 0..npi {
        for jp in 0..npj {
            let rij = [
                ai[ip] * rca[0] + aj[jp] * rcb[0],
                ai[ip] * rca[1] + aj[jp] * rcb[1],
                ai[ip] * rca[2] + aj[jp] * rcb[2],
            ];
            type1_rad_ang(
                &mut rad_ang,
                li + lj,
                &rij,
                &rad_all[(ip * npj + jp) * d2..(ip * npj + jp) * d2 + d2],
            );
            for ic in 0..nci {
                for jc in 0..ncj {
                    let fac = cei[ic * npi + ip] * cej[jc * npj + jp];
                    let prad = &mut rad_ang_all[(ic * ncj + jc) * d3..(ic * ncj + jc) * d3 + d3];
                    for n in 0..d3 {
                        prad[n] += fac * rad_ang[n];
                    }
                }
            }
        }
    }

    let mut ifac = vec![0.0f64; nfi * di3];
    let mut jfac = vec![0.0f64; nfj * dj3];
    type1_static_facs(&mut ifac, li, &rca);
    type1_static_facs(&mut jfac, lj, &rcb);

    // Phase-B angular splice — now a `#[cube(launch)]` device kernel dispatched on
    // the resolved backend (quick-260529-gbf). The host-precomputed f64 tensors
    // `rad_ang_all` (per (ic,jc): d3), `ifac` (nfi*di3), `jfac` (nfj*dj3) and the
    // u32 cart-power triples cross to the device; the kernel runs at f64 with the
    // SAME nested summation order (i1 outer .. j3 inner), so the result is
    // byte-identical to the prior host loop (atol=1e-12/rtol=0.0 vendor parity).
    let comps_i = cart_comps_flat_u32(li as u8);
    let comps_j = cart_comps_flat_u32(lj as u8);
    for ic in 0..nci {
        for jc in 0..ncj {
            let prad = &rad_ang_all[(ic * ncj + jc) * d3..(ic * ncj + jc) * d3 + d3];
            // Device-backed splice for this contraction tuple → block [ao_i, ao_j].
            let block = run_ecp_angular_splice_on_backend(
                backend, li as u32, lj as u32, prad, &ifac, &jfac, &comps_i, &comps_j,
            );
            // Scatter the (nfi*nfj) block (F-order mj*nfi+mi) into the full
            // contraction-major gctr: index = (jc*nfj+mj)*(nci*nfi) + ic*nfi+mi.
            for mj in 0..nfj {
                for mi in 0..nfi {
                    let pout_idx = (jc * nfj + mj) * (nci * nfi) + ic * nfi + mi;
                    gctr[pout_idx] += block[mj * nfi + mi];
                }
            }
        }
    }
}

/// Backend-dispatch wrapper for the Type-1 angular splice: routes the
/// `#[cube(launch)]` [`ecp_angular_kernel`] onto the resolved backend's device
/// client (Cpu => CpuRuntime, Rocm => HipRuntime, Wgpu, Cuda, Metal — each
/// `#[cfg]`-gated), mirroring center_4c1e.rs's per-backend `match`. Runs at f64
/// (ECP keeps f64 staging / byte-identity gate). Returns the `nfi*nfj` block.
#[allow(clippy::too_many_arguments)]
fn run_ecp_angular_splice_on_backend(
    backend: &ResolvedBackend,
    li: u32,
    lj: u32,
    rad_ang: &[f64],
    ifac: &[f64],
    jfac: &[f64],
    comps_i: &[u32],
    comps_j: &[u32],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_ecp_angular_device::<cubecl::cpu::CpuRuntime>(
            client, li, lj, rad_ang, ifac, jfac, comps_i, comps_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_ecp_angular_device::<cubecl_wgpu::WgpuRuntime>(
            client, li, lj, rad_ang, ifac, jfac, comps_i, comps_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_ecp_angular_device::<cubecl_cuda::CudaRuntime>(
            client, li, lj, rad_ang, ifac, jfac, comps_i, comps_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_ecp_angular_device::<cubecl_hip::HipRuntime>(
            client, li, lj, rad_ang, ifac, jfac, comps_i, comps_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_ecp_angular_device::<cubecl_wgpu::WgpuRuntime>(
            client, li, lj, rad_ang, ifac, jfac, comps_i, comps_j,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase-B angular splice — `#[cube(launch)]` device kernel, generic over F.
//
// CLAUDE.md mandates CubeCL as the primary compute backend with host CPU work
// limited to planning/validation/marshaling. The Type-1 angular accumulation
// (the inner sextuple loop at the tail of `ecp_type1_cart`, ported from
// nr_ecp.c LOOP_CART/LOOP_XYZ, lines 5976-5988) is the part that ports cleanly
// to `#[cube]`: it consumes the host-precomputed radial/angular f64 tensors and
// does bounded triple-product arithmetic with NO special functions, NO
// break/continue. The adaptive radial machinery (special-function recurrences,
// data-dependent convergence) STAYS host-side as marshaling — it cannot be
// `#[cube]`. This is the honest, CLAUDE.md-compliant host/device split (quick
// task 260529-gbf), mirroring center_2c2e / center_4c1e.
//
// Layout — per (ic,jc) contraction tuple this kernel computes ONE block:
//   cart_out[(jc*nfj+mj)*(nci*nfi) + ic*nfi + mi]
//     += sum_{i1=0..ix, i2=0..iy, i3=0..iz, j1=0..jx, j2=0..jy, j3=0..jz}
//          ifac[mi*di3 + i1*di2 + i2*di1 + i3]
//        * jfac[mj*dj3 + j1*dj2 + j2*dj1 + j3]
//        * rad_ang[(ic*ncj+jc)*d3 + (i1+j1)*d2 + (i2+j2)*d1 + (i3+j3)]
// The nested loop order (i1 outer .. j3 inner) is IDENTICAL to the host driver
// (ecp.rs `ecp_type1_cart`) to preserve f64 summation order / byte-identity.
//
// `comps_i`/`comps_j` are flat u32 cartesian-power triples (3 entries per
// component) precomputed host-side, so the kernel never re-derives `cart_comps`
// (that enumeration is marshaling). The kernel is launched once per (ic,jc)
// tuple by `run_ecp_angular_device` (the splice is per-contraction-pair).
// ─────────────────────────────────────────────────────────────────────────────

/// Type-1 angular-splice device kernel for ONE (ic,jc) contraction tuple,
/// generic over `F: Float`. Single work item (`UNIT_POS == 0`). All math uses
/// `F` arithmetic, statement-form `if`, `u32` indices, and `while` loops bounded
/// by the (runtime-`u32`) cartesian powers — no break/continue, no special
/// functions (those stay host-side).
///
/// Source of the splice arithmetic: vendor/pyscf-nr-ecp/src/nr_ecp.c:5976-5988
/// (the cintx host driver `ecp_type1_cart` tail loop, lines ~783-817).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ecp_angular_kernel<F: Float + CubeElement>(
    rad_ang: &Array<F>,
    ifac: &Array<F>,
    jfac: &Array<F>,
    comps_i: &Array<u32>,
    comps_j: &Array<u32>,
    cart_out: &mut Array<F>,
    li: u32,
    lj: u32,
    nfi: u32,
    nfj: u32,
) {
    // Strides for the radial/angular tensor (d1 = li+lj+1).
    let d1 = li + lj + 1u32;
    let d2 = d1 * d1;
    // ifac strides (di1 = li+1).
    let di1 = li + 1u32;
    let di2 = di1 * di1;
    let di3 = di2 * di1;
    // jfac strides (dj1 = lj+1).
    let dj1 = lj + 1u32;
    let dj2 = dj1 * dj1;
    let dj3 = dj2 * dj1;

    // Zero the output block cooperatively.
    let out_len = nfi * nfj;
    let mut oi = UNIT_POS as usize;
    let stride = CUBE_DIM as usize;
    while oi < (out_len as usize) {
        cart_out[oi] = F::new(0.0);
        oi += stride;
    }
    sync_cube();

    // Loop over bra cartesian components mi, then ket mj cooperatively.
    let mut mi = 0u32;
    while mi < nfi {
        let ix = comps_i[(mi * 3u32) as usize];
        let iy = comps_i[(mi * 3u32 + 1u32) as usize];
        let iz = comps_i[(mi * 3u32 + 2u32) as usize];
        let ifac_base = mi * di3;
        let mut mj = 0u32;
        while mj < nfj {
            let pout_idx = mj * nfi + mi;
            if ((pout_idx as u32) % CUBE_DIM) == UNIT_POS {
                let jx = comps_j[(mj * 3u32) as usize];
                let jy = comps_j[(mj * 3u32 + 1u32) as usize];
                let jz = comps_j[(mj * 3u32 + 2u32) as usize];
                let jfac_base = mj * dj3;

                let mut acc = F::new(0.0);
                // Sextuple accumulation — i1 outer .. j3 inner, matching the
                // host driver loop order exactly (byte-identical f64 sum order).
                let mut i1 = 0u32;
                while i1 <= ix {
                    let mut i2 = 0u32;
                    while i2 <= iy {
                        let mut i3 = 0u32;
                        while i3 <= iz {
                            let ifac_idx = ifac_base + i1 * di2 + i2 * di1 + i3;
                            let pif = ifac[ifac_idx as usize];
                            let mut j1 = 0u32;
                            while j1 <= jx {
                                let mut j2 = 0u32;
                                while j2 <= jy {
                                    let mut j3 = 0u32;
                                    while j3 <= jz {
                                        let jfac_idx = jfac_base + j1 * dj2 + j2 * dj1 + j3;
                                        let pjf = jfac[jfac_idx as usize];
                                        let rad_idx = (i1 + j1) * d2 + (i2 + j2) * d1 + (i3 + j3);
                                        let pr = rad_ang[rad_idx as usize];
                                        acc += pif * pjf * pr;
                                        j3 += 1u32;
                                    }
                                    j2 += 1u32;
                                }
                                j1 += 1u32;
                            }
                            i3 += 1u32;
                        }
                        i2 += 1u32;
                    }
                    i1 += 1u32;
                }

                // F-order block [ao_i, ao_j] within this (ic=0,jc=0) tuple.
                cart_out[pout_idx as usize] = acc;
            }
            mj += 1u32;
        }
        mi += 1u32;
    }
}

/// Dispatch [`ecp_angular_kernel`] at `f64` on a resolved backend's client for
/// ONE (ic,jc) contraction tuple, reading back the `nfi*nfj` Cartesian block.
///
/// Generic over `R: Runtime` so the same path serves CPU, ROCm, etc. Intermediate
/// device compute is `f64` (ECP keeps f64 staging; the byte-identity gate is
/// f64/CPU-vs-C). `rad_ang` is the single-tuple slice (`d3`), `ifac`/`jfac` are
/// the full `nfi*di3` / `nfj*dj3` static-factor tables, `comps_i`/`comps_j` are
/// flat u32 cartesian-power triples (3 per component).
// Wired into `ecp_type1_cart` in quick-260529-gbf Task 2; until then it is only
// exercised by the device-vs-host equivalence tests.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn run_ecp_angular_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    rad_ang: &[f64],
    ifac: &[f64],
    jfac: &[f64],
    comps_i: &[u32],
    comps_j: &[u32],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let out_len = nfi * nfj;

    // Sanity (T-gbf-01): buffer lengths derived from li/lj must match the host
    // tensor lengths exactly, or the kernel reads out of bounds on-device.
    let d1 = li_u + lj_u + 1;
    let d3 = d1 * d1 * d1;
    let di3 = (li_u + 1).pow(3);
    let dj3 = (lj_u + 1).pow(3);
    debug_assert_eq!(rad_ang.len(), d3, "ecp angular: rad_ang len != d3");
    debug_assert_eq!(ifac.len(), nfi * di3, "ecp angular: ifac len != nfi*di3");
    debug_assert_eq!(jfac.len(), nfj * dj3, "ecp angular: jfac len != nfj*dj3");
    debug_assert_eq!(comps_i.len(), nfi * 3, "ecp angular: comps_i len != nfi*3");
    debug_assert_eq!(comps_j.len(), nfj * 3, "ecp angular: comps_j len != nfj*3");

    let rad_h = client.create_from_slice(f64::as_bytes(rad_ang));
    let ifac_h = client.create_from_slice(f64::as_bytes(ifac));
    let jfac_h = client.create_from_slice(f64::as_bytes(jfac));
    let comps_i_h = client.create_from_slice(u32::as_bytes(comps_i));
    let comps_j_h = client.create_from_slice(u32::as_bytes(comps_j));

    let out_h = client.empty(out_len * std::mem::size_of::<f64>());

    // SAFETY: Input buffer lengths match exact slice lengths.
    // Out buffer is allocated to exact out_len.
    // In-kernel loops strictly bound indices to valid array ranges.
    unsafe {
        ecp_angular_kernel::launch_unchecked::<f64, R>(
            client,
            crate::plane::single_cube_count(),
            crate::plane::standard_plane_cube_dim(),
            ArrayArg::from_raw_parts(rad_h, rad_ang.len()),
            ArrayArg::from_raw_parts(ifac_h, ifac.len()),
            ArrayArg::from_raw_parts(jfac_h, jfac.len()),
            ArrayArg::from_raw_parts(comps_i_h, comps_i.len()),
            ArrayArg::from_raw_parts(comps_j_h, comps_j.len()),
            ArrayArg::from_raw_parts(out_h.clone(), out_len),
            li,
            lj,
            nfi as u32,
            nfj as u32,
        );
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Flatten `cart_comps(l)` into the `[mi*3 + axis]` u32 triple layout the device
/// kernel consumes (marshaling — keeps `cart_comps` enumeration host-side).
// Wired into `ecp_type1_cart` in quick-260529-gbf Task 2.
#[cfg_attr(not(test), allow(dead_code))]
fn cart_comps_flat_u32(l: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity(ncart(l) * 3);
    for (lx, ly, lz) in cart_comps(l) {
        out.push(lx as u32);
        out.push(ly as u32);
        out.push(lz as u32);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-2 (Projected) angular splice — `#[cube(launch)]` device kernel, generic
// over F (quick-260529-hin follow-up to -gbf, which ported only Type-1).
//
// CLAUDE.md mandates CubeCL as the primary compute backend with host work limited
// to planning/validation/marshaling. Type-2's angular contraction (the two-`dgemm`
// matmul at the tail of `ecp_type2_cart`, ported from nr_ecp.c:5505-5510) is fully
// bounded arithmetic — all loop bounds derive from comptime `li`/`lj`/`lc`; NO
// break/continue, NO special functions — and therefore belongs on-device, same as
// the Type-1 splice. The adaptive Gauss-Chebyshev radial machinery (Phase A) STAYS
// host-side (data-dependent convergence, special functions).
//
// Byte-identity: the host materializes an intermediate `buf` (dgemm-1 output) then
// consumes it in dgemm-2. Each `buf` entry is an independent inner sum, so the
// device kernel RECOMPUTES each entry inline at the moment dgemm-2 reads it — this
// is bit-exact because the dgemm-1 inner sum (over `kk in 0..lilc1`) and the
// dgemm-2 contraction (over `kk2 in 0..mq`) are evaluated in the SAME order as the
// host loops. This avoids a device-local Array scratch (no proven `Array::new`
// precedent in center_2c2e/3c2e/4c1e — those accumulate in registers). The host
// `buf[row*mq+kk2]` equals dgemm-1's `buf[col*ljlc1+row2]` with
//   col = row*dlc + kk2/ljlc1 ,  row2 = kk2 - (kk2/ljlc1)*ljlc1
// (derived: row*mq+kk2 == col*ljlc1+row2 with mq = dlc*ljlc1).
//
// Per (ic,jc) contraction tuple this kernel computes ONE `nfi*nfj` block in F-order
// `cart_out[col2*nfi + row]`, accumulating across the i in 0..=li, j in 0..=lj loop
// (mirroring the host `gctr += ...` accumulation), `common_fac` folded in-kernel so
// the host scatter is a trivial `+=` (matching the Type-1 template).
// ─────────────────────────────────────────────────────────────────────────────

/// On-device angular contraction for Type-2 semi-local ECP with lane-owned outputs.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ecp_type2_angular_kernel<F: Float + CubeElement>(
    prad: &Array<F>,
    angi: &Array<F>,
    angj: &Array<F>,
    cart_out: &mut Array<F>,
    common_fac: F,
    li: u32,
    lj: u32,
    lc: u32,
    nfi: u32,
    nfj: u32,
) {
    // Strides — recomputed from the runtime u32 args (mirror the host driver).
    let dlc = lc * 2u32 + 1u32;
    let lilc1 = li + lc + 1u32;
    let ljlc1 = lj + lc + 1u32;
    let d2 = lilc1 * ljlc1;
    let mq = dlc * ljlc1;

    // Zero the output block cooperatively.
    let out_len = nfi * nfj;
    let mut oi = UNIT_POS as usize;
    let stride = CUBE_DIM as usize;
    while oi < (out_len as usize) {
        cart_out[oi] = F::new(0.0);
        oi += stride;
    }
    sync_cube();

    // Accumulate over the angular pair (i in 0..=li, j in 0..=lj), matching the
    // host `for i / for j` order.
    let mut i = 0u32;
    while i <= li {
        let mut j = 0u32;
        while j <= lj {
            let a_base = (i + j) * d2; // prad block: lilc1 x ljlc1 (col-major)
            let b_base = i * nfi * dlc * lilc1; // angi block: lilc1 x im (col-major)
            let bj_base = j * nfj * dlc * ljlc1; // angj block: mq x nfj (col-major)

            let mut row = 0u32;
            while row < nfi {
                let mut col2 = 0u32;
                while col2 < nfj {
                    let out_idx = col2 * nfi + row;
                    if ((out_idx as u32) % CUBE_DIM) == UNIT_POS {
                        let mut acc = F::new(0.0);
                        let mut kk2 = 0u32;
                        while kk2 < mq {
                            // buf[row*mq+kk2] == dgemm-1 buf[col*ljlc1+row2] with
                            //   col  = row*dlc + kk2/ljlc1   (∈ 0..im)
                            //   row2 = kk2 - (kk2/ljlc1)*ljlc1  (∈ 0..ljlc1)
                            let qd = kk2 / ljlc1;
                            let col = row * dlc + qd;
                            let row2 = kk2 - qd * ljlc1;
                            // dgemm-1: bufv = Σ_kk(0..lilc1) a[kk*ljlc1+row2] * b[col*lilc1+kk]
                            let mut bufv = F::new(0.0);
                            let mut kk = 0u32;
                            while kk < lilc1 {
                                let a_idx = a_base + kk * ljlc1 + row2;
                                let b_idx = b_base + col * lilc1 + kk;
                                bufv += prad[a_idx as usize] * angi[b_idx as usize];
                                kk += 1u32;
                            }
                            let bj_idx = bj_base + col2 * mq + kk2;
                            acc += bufv * angj[bj_idx as usize];
                            kk2 += 1u32;
                        }
                        // F-order [ao_i, ao_j] within this (ic=0,jc=0) tuple;
                        // accumulate across (i,j) and fold common_fac in-kernel.
                        let out_idx = col2 * nfi + row;
                        cart_out[out_idx as usize] += common_fac * acc;
                    }
                    col2 += 1u32;
                }
                row += 1u32;
            }
            j += 1u32;
        }
        i += 1u32;
    }
}

/// Dispatch [`ecp_type2_angular_kernel`] at `f64` on a resolved backend's client
/// for ONE (ic=0,jc=0) contraction tuple, reading back the `nfi*nfj` block
/// (already `common_fac`-scaled in-kernel, F-order `col2*nfi+row`).
///
/// Generic over `R: Runtime` (CPU, ROCm, …). Intermediate device compute is `f64`
/// (ECP keeps f64 staging; the byte-identity gate is f64/CPU-vs-C). `prad` is the
/// single-tuple slice (`d3 = (li+lj+1)*lilc1*ljlc1`); `angi`/`angj` are the full
/// `type2_facs_ang` tables.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn run_ecp_type2_angular_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    lc: u32,
    prad: &[f64],
    angi: &[f64],
    angj: &[f64],
    common_fac: f64,
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let lc_u = lc as usize;
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let out_len = nfi * nfj;

    // Sanity (T-hin-01): buffer lengths derived from li/lj/lc must match the host
    // tensor lengths exactly, or the kernel reads out of bounds on-device. Same
    // formulas as the host driver (`ecp_type2_cart`).
    let dlc = lc_u * 2 + 1;
    let lilc1 = li_u + lc_u + 1;
    let ljlc1 = lj_u + lc_u + 1;
    let d2 = lilc1 * ljlc1;
    let d3 = (li_u + lj_u + 1) * d2;
    debug_assert_eq!(prad.len(), d3, "ecp type2: prad len != d3");
    debug_assert_eq!(
        angi.len(),
        (li_u + 1) * nfi * dlc * lilc1,
        "ecp type2: angi len != (li+1)*nfi*dlc*lilc1"
    );
    debug_assert_eq!(
        angj.len(),
        (lj_u + 1) * nfj * dlc * ljlc1,
        "ecp type2: angj len != (lj+1)*nfj*dlc*ljlc1"
    );

    let prad_h = client.create_from_slice(f64::as_bytes(prad));
    let angi_h = client.create_from_slice(f64::as_bytes(angi));
    let angj_h = client.create_from_slice(f64::as_bytes(angj));

    let out_h = client.empty(out_len * std::mem::size_of::<f64>());

    // SAFETY: Input buffer lengths match exact slice lengths.
    // Out buffer is allocated to exact out_len.
    // In-kernel loops strictly bound indices to valid array ranges.
    unsafe {
        ecp_type2_angular_kernel::launch_unchecked::<f64, R>(
            client,
            crate::plane::single_cube_count(),
            crate::plane::standard_plane_cube_dim(),
            ArrayArg::from_raw_parts(prad_h, prad.len()),
            ArrayArg::from_raw_parts(angi_h, angi.len()),
            ArrayArg::from_raw_parts(angj_h, angj.len()),
            ArrayArg::from_raw_parts(out_h.clone(), out_len),
            common_fac,
            li,
            lj,
            lc,
            nfi as u32,
            nfj as u32,
        );
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Backend-dispatch wrapper for the Type-2 angular splice: routes the
/// `#[cube(launch)]` [`ecp_type2_angular_kernel`] onto the resolved backend's
/// device client (Cpu => CpuRuntime, Rocm => HipRuntime, Wgpu, Cuda, Metal — each
/// `#[cfg]`-gated), mirroring [`run_ecp_angular_splice_on_backend`]. Runs at f64
/// (ECP keeps f64 staging / byte-identity gate). Returns the `nfi*nfj` block
/// (`common_fac`-scaled in-kernel, F-order `col2*nfi+row`).
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn run_ecp_type2_splice_on_backend(
    backend: &ResolvedBackend,
    li: u32,
    lj: u32,
    lc: u32,
    prad: &[f64],
    angi: &[f64],
    angj: &[f64],
    common_fac: f64,
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_ecp_type2_angular_device::<cubecl::cpu::CpuRuntime>(
            client, li, lj, lc, prad, angi, angj, common_fac,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_ecp_type2_angular_device::<cubecl_wgpu::WgpuRuntime>(
                client, li, lj, lc, prad, angi, angj, common_fac,
            )
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_ecp_type2_angular_device::<cubecl_cuda::CudaRuntime>(
            client, li, lj, lc, prad, angi, angj, common_fac,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_ecp_type2_angular_device::<cubecl_hip::HipRuntime>(
            client, li, lj, lc, prad, angi, angj, common_fac,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_ecp_type2_angular_device::<cubecl_wgpu::WgpuRuntime>(
                client, li, lj, lc, prad, angi, angj, common_fac,
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-2 (Projected) ECP driver — port of ECPtype2_cart.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:5337-5515.
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn ecp_type2_cart(
    backend: &ResolvedBackend,
    gctr: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    lc: usize,
    rad_shells: &[EcpRadShell],
    rs_max: &[f64],
    ws_max: &[f64],
) {
    // Type-2's angular splice (the two-`dgemm` matmul at the tail of this driver)
    // runs on-device as `ecp_type2_angular_kernel` via `run_ecp_type2_splice_on_backend`
    // (quick-260529-hin), generic over F, f64-internal, byte-identity preserved.
    // The `backend` is dispatched per (ic,jc) contraction tuple below; only the
    // Phase-A adaptive Gauss-Chebyshev radial machinery stays host as marshaling.
    let li = shell_i.ang_momentum as usize;
    let lj = shell_j.ang_momentum as usize;
    let npi = shell_i.nprim as usize;
    let npj = shell_j.nprim as usize;
    let nci = shell_i.nctr as usize;
    let ncj = shell_j.nctr as usize;
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let di = nfi * nci;
    let ai = &shell_i.exponents;
    let aj = &shell_j.exponents;
    // Canonical Shell.coefficients are row-major [ip*nctr+ic]; this ECP kernel's
    // internal indexing is column-major [ic*nprim+ip]. Convert once (identity for
    // nctr==1). Without this, generally-contracted shells read scrambled coeffs.
    let ci_cm = coeffs_col_major(shell_i);
    let cj_cm = coeffs_col_major(shell_j);
    let ci = &ci_cm;
    let cj = &cj_cm;

    let common_fac = cint_common_fac_sp(li)
        * cint_common_fac_sp(lj)
        * 16.0
        * std::f64::consts::PI
        * std::f64::consts::PI;

    let rca = [rc[0] - ri[0], rc[1] - ri[1], rc[2] - ri[2]];
    let rcb = [rc[0] - rj[0], rc[1] - rj[1], rc[2] - rj[2]];
    let dca = (rca[0] * rca[0] + rca[1] * rca[1] + rca[2] * rca[2]).sqrt();
    let dcb = (rcb[0] * rcb[0] + rcb[1] * rcb[1] + rcb[2] * rcb[2]).sqrt();

    let lilj1 = li + lj + 1;
    let dlc = lc * 2 + 1;
    let lilc1 = li + lc + 1;
    let ljlc1 = lj + lc + 1;
    let d2 = lilc1 * ljlc1;
    let d3 = lilj1 * d2;
    // `im = nfi*dlc` / `mq = dlc*ljlc1` (the dgemm dims) are now recomputed inside
    // `run_ecp_type2_angular_device`; the host driver no longer materializes them.

    let nrs_alloc = 1usize << LEVEL_MAX;
    // rad_all[(ic*ncj+jc)*d3 + lab*d2 + i*ljlc1 + j]
    let mut rad_all = vec![0.0f64; nci * ncj * d3];
    // rur[lab*nrs + n]
    let mut rur = vec![0.0f64; nrs_alloc * lilj1];
    // radi[ic*nrs*lilc1 + n*lilc1 + i], radj likewise
    let mut radi = vec![0.0f64; nci * lilc1 * nrs_alloc];
    let mut radj = vec![0.0f64; ncj * ljlc1 * nrs_alloc];

    let mut converged = vec![false; nci * ncj * lilj1];
    let mut nrs0 = (1usize << LEVEL0) - 1;
    let mut step = 1usize << (LEVEL_MAX - LEVEL0);
    let mut start = step - 1;
    let mut wtscale = step as f64;
    let mut plast = vec![0.0f64; d2];

    let mut level = LEVEL0;
    while level <= LEVEL_MAX {
        let nrs = {
            // rur[0..nrs] = ECPrad_part radial sum * ws * wtscale.
            let mut ur_lab0 = vec![0.0f64; nrs0];
            let n = ecprad_part_host(&mut ur_lab0, rs_max, start, nrs0, step, rad_shells);
            for i in 0..n {
                rur[i] = ur_lab0[i] * ws_max[start + i * step] * wtscale;
            }
            n
        };
        // rur[lab*nrs + i] = rur[(lab-1)*nrs + i] * rs[start + i*step]
        for i in 0..nrs {
            for lab in 1..=(li + lj) {
                rur[nrs * lab + i] = rur[nrs * (lab - 1) + i] * rs_max[start + i * step];
            }
        }

        // type2_facs_rad for both centers, contraction layout.
        for ic in 0..nci {
            let off = ic * nrs * lilc1;
            type2_facs_rad_host(
                &mut radi[off..off + nrs * lilc1],
                li,
                lc,
                dca,
                ai,
                &ci[ic * npi..ic * npi + npi],
                1,
                &rs_max[start..],
                nrs,
                step,
            );
        }
        for jc in 0..ncj {
            let off = jc * nrs * ljlc1;
            type2_facs_rad_host(
                &mut radj[off..off + nrs * ljlc1],
                lj,
                lc,
                dcb,
                aj,
                &cj[jc * npj..jc * npj + npj],
                1,
                &rs_max[start..],
                nrs,
                step,
            );
        }

        let mut all_conv = true;
        let mut ijl = 0usize;
        for ic in 0..nci {
            for jc in 0..ncj {
                let pradi_off = ic * nrs_alloc * lilc1; // note: radi laid out with nrs_alloc stride
                let _ = pradi_off;
                for lab in 0..=(li + lj) {
                    if !converged[ijl] {
                        // prur = rur + lab*nrs
                        let prad = &mut rad_all[ijl * d2..ijl * d2 + d2];
                        plast[..d2].copy_from_slice(prad);
                        for v in prad.iter_mut() {
                            *v *= 0.5;
                        }
                        for i in 0..lilc1 {
                            for j in 0..ljlc1 {
                                let mut s = prad[i * ljlc1 + j];
                                for n in 0..nrs {
                                    s += rur[lab * nrs + n]
                                        * radi[ic * nrs * lilc1 + n * lilc1 + i]
                                        * radj[jc * nrs * ljlc1 + n * ljlc1 + j];
                                }
                                prad[i * ljlc1 + j] = s;
                            }
                        }
                        converged[ijl] = true;
                        for i in 0..d2 {
                            if !close_enough(plast[i], prad[i]) {
                                converged[ijl] = false;
                                all_conv = false;
                                break;
                            }
                        }
                    }
                    ijl += 1;
                }
            }
        }

        if all_conv {
            break;
        }
        nrs0 = (1usize << level) - 1;
        step = 1usize << (LEVEL_MAX - level);
        // C uses signed int: (0 - 1) / 2 == 0. Mirror that exactly under usize
        // (saturating_sub avoids the Rust underflow panic at the finest level
        // when a non-converging far-field pair reaches start == 0; byte-identical
        // to the C arithmetic since the loop exits at level > LEVEL_MAX anyway).
        start = start.saturating_sub(1) / 2;
        wtscale *= 0.5;
        level += 1;
    }

    // Angular splice: type2_facs_ang + two dgemms (nr_ecp.c:5498-5512).
    let dlclmb = (li + lc + 1) * dlc;
    let _ = dlclmb;
    let mut angi = vec![0.0f64; (li + 1) * nfi * dlc * lilc1];
    let mut angj = vec![0.0f64; (lj + 1) * nfj * dlc * ljlc1];
    type2_facs_ang(&mut angi, li, lc, &rca);
    type2_facs_ang(&mut angj, lj, lc, &rcb);

    // angi laid out [(a)*nfi*dlc*lilc1 + mi*dlc*lilc1 + m*lilc1 + n]
    // (type2_facs_ang writes facs[(a+b+c)*nfi*dlclmb + mi*dlclmb + m*dlambda + n]
    //  where dlclmb = dlambda*dlc; we slice per-`a` block below).
    // The two-`dgemm` angular splice now runs on-device via the
    // `ecp_type2_angular_kernel` #[cube(launch)] kernel (quick-260529-hin), once
    // per (ic,jc) contraction tuple. The device kernel folds `common_fac` in and
    // accumulates across the (i,j) angular pair, returning an `nfi*nfj` block in
    // F-order `block[col2*nfi + row]`; the host scatters it into the
    // contraction-major `gctr` with a plain `+=` (NO extra multiply — `common_fac`
    // is already applied in-kernel, matching the prior host scatter target).
    for ic in 0..nci {
        for jc in 0..ncj {
            let prad = &rad_all[(ic * ncj + jc) * d3..(ic * ncj + jc) * d3 + d3];
            let block = run_ecp_type2_splice_on_backend(
                backend, li as u32, lj as u32, lc as u32, prad, &angi, &angj, common_fac,
            );
            // Scatter into gctr at c_off = jc*nfj*di + ic*nfi (di = nci*nfi), the
            // exact target the host two-dgemm wrote (gctr[c_off + col2*di + row]).
            let c_off = jc * nfj * di + ic * nfi;
            for col2 in 0..nfj {
                for row in 0..nfi {
                    gctr[c_off + col2 * di + row] += block[col2 * nfi + row];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient (ipnuc) driver — port of nr_ecp_deriv.c::_deriv1_cart.
// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:201-286 (_deriv1_cart),
//         :148-172 (_l_down), :174-199 (_l_up).
//
// `_deriv1_cart` computes ∂/∂A_i^{x,y,z} of the (Type-1 + Type-2) ECP integral
// w.r.t. the bra AO center, via the GTO derivative identity
//   ∂/∂A χ_l = 2 a_i χ_{l+1} − l χ_{l-1}.
// It uncontracts the primitives (each primitive becomes a single-primitive
// "fake" shell with coefficient == exponent), re-evaluates the SAME scalar
// `ECPtype1_cart` / `ECPtype2_cart` drivers at l+1 and l-1, applies `_l_down`
// (the 2·a_i·χ_{l+1} term) and `_l_up` (the −l·χ_{l-1} term), divides out
// expi·expj, then re-applies the AO contraction coefficients.
// ─────────────────────────────────────────────────────────────────────────────

/// Local index in `cart_comps(l+1)` reached by raising the `axis` power of
/// `cart_comps(l)[i]` by one. `axis`: 0=x, 1=y, 2=z. The x-raise is the
/// identity prefix in cintx's `cart_comps` ordering (verified: matches
/// libcint's `_l_down` `outx[...i] = buf1[...i]` convention), so `raise_idx`
/// is the cintx-ordering analog of nr_ecp_deriv.c's global `_y_addr` / `_z_addr`
/// tables (nr_ecp_deriv.c:56-75) — derived from `cart_comps` rather than
/// transcribed, since cintx's cart enumeration order (not libcint's global
/// addr layout) is what the cintx scalar drivers produce.
#[inline]
fn raise_idx(l: usize, i: usize, axis: usize) -> usize {
    let comps_l = cart_comps(l as u8);
    let (lx, ly, lz) = comps_l[i];
    let target = match axis {
        0 => (lx + 1, ly, lz),
        1 => (lx, ly + 1, lz),
        _ => (lx, ly, lz + 1),
    };
    cart_comps((l + 1) as u8)
        .iter()
        .position(|&c| c == target)
        .expect("raised cartesian component must exist in cart_comps(l+1)")
}

/// `_l_down(out, buf1, fac, ai, li, nfj)` — the `2·a_i·χ_{li+1}` derivative
/// term. Writes the 3 axis blocks `out = [outx | outy | outz]` (each `nfi*nfj`,
/// F-order `j*nfi+i`) by down-mapping the `nfi1 = ncart(li+1)` rows of `buf1`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:148-172.
fn l_down(out: &mut [f64], buf1: &[f64], fac: f64, ai: f64, li: usize, nfj: usize) {
    let nfi = ncart(li as u8);
    let nfi1 = ncart((li + 1) as u8);
    // Cart→sph common-factor compensation between the (li+1) and li shells.
    let fac = match li {
        0 => fac * (-2.0 / 3.0_f64.sqrt()) * ai,
        1 => fac * (-2.0 * 0.488602511902919921) * ai,
        _ => fac * (-2.0) * ai,
    };
    let (outx, rest) = out.split_at_mut(nfi * nfj);
    let (outy, outz) = rest.split_at_mut(nfi * nfj);
    for j in 0..nfj {
        for i in 0..nfi {
            // x-raise is the identity prefix; y/z raise via raise_idx.
            outx[j * nfi + i] = fac * buf1[j * nfi1 + i];
            outy[j * nfi + i] = fac * buf1[j * nfi1 + raise_idx(li, i, 1)];
            outz[j * nfi + i] = fac * buf1[j * nfi1 + raise_idx(li, i, 2)];
        }
    }
}

/// `_l_up(out, buf1, fac, li, nfj)` — the `−li·χ_{li-1}` derivative term (the
/// minus sign is folded into the `_l_down` `-2` so this `_l_up` is the libcint
/// `+= (pow+1)` accumulation). Adds the power-weighted `nfi0 = ncart(li-1)`
/// rows of `buf1` into the 3 axis blocks.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:174-199.
fn l_up(out: &mut [f64], buf1: &[f64], fac: f64, li: usize, nfj: usize) {
    let nfi = ncart(li as u8);
    let nfi0 = ncart((li - 1) as u8);
    let fac = match li {
        1 => fac * 3.0_f64.sqrt(),
        2 => fac * (1.0 / 0.488602511902919921),
        _ => fac,
    };
    let comps0 = cart_comps((li - 1) as u8);
    let (outx, rest) = out.split_at_mut(nfi * nfj);
    let (outy, outz) = rest.split_at_mut(nfi * nfj);
    for (i, &(px, py, pz)) in comps0.iter().enumerate() {
        let xfac = fac * (px as f64 + 1.0);
        let yfac = fac * (py as f64 + 1.0);
        let zfac = fac * (pz as f64 + 1.0);
        let xi = i; // x-raise identity prefix
        let yi = raise_idx(li - 1, i, 1);
        let zi = raise_idx(li - 1, i, 2);
        for j in 0..nfj {
            outx[j * nfi + xi] += xfac * buf1[j * nfi0 + i];
            outy[j * nfi + yi] += yfac * buf1[j * nfi0 + i];
            outz[j * nfi + zi] += zfac * buf1[j * nfi0 + i];
        }
    }
}

/// Evaluate the (Type-1 + Type-2) scalar ECP cart integral for a single
/// uncontracted *primitive* pair with the bra shell forced to angular momentum
/// `li_eff` (the `l ± 1` shifts `_deriv1_cart` needs). Mirrors PySCF's
/// per-primitive `ECPtype1_cart(buf1,...) | ECPtype2_cart(buf1,...)` on the
/// `_uncontract_bas` fake shells: each primitive becomes a 1-prim/1-ctr shell
/// whose coefficient equals its exponent (nr_ecp_deriv.c:129-145), so the
/// returned buffer carries the `expi·expj` factor that the caller divides out.
/// Output `buf` is F-order `[ao_j, ao_i]` (`j*nfi_eff + i`), length
/// `ncart(li_eff) * ncart(lj)`.
#[allow(clippy::too_many_arguments)]
fn ecp_scalar_prim_pair_cart(
    backend: &ResolvedBackend,
    buf: &mut [f64],
    li_eff: usize,
    lj: usize,
    ai: f64,
    aj: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    lc: i32,
    rad_shells: &[EcpRadShell],
    rs_max: &[f64],
    ws_max: &[f64],
) {
    // Single-primitive, single-contraction fake shells (coeff == exponent,
    // matching _uncontract_bas which sets PTR_COEFF = PTR_EXP).
    let shell_i = Shell::try_new(
        0,
        li_eff as u8,
        1,
        1,
        0,
        Representation::Cart,
        Arc::from(vec![ai].into_boxed_slice()),
        Arc::from(vec![ai].into_boxed_slice()),
    )
    .expect("fake bra primitive shell");
    let shell_j = Shell::try_new(
        0,
        lj as u8,
        1,
        1,
        0,
        Representation::Cart,
        Arc::from(vec![aj].into_boxed_slice()),
        Arc::from(vec![aj].into_boxed_slice()),
    )
    .expect("fake ket primitive shell");
    if lc < 0 {
        ecp_type1_cart(
            backend, buf, &shell_i, &shell_j, ri, rj, rc, rad_shells, rs_max, ws_max,
        );
    } else {
        ecp_type2_cart(
            backend,
            buf,
            &shell_i,
            &shell_j,
            ri,
            rj,
            rc,
            lc as usize,
            rad_shells,
            rs_max,
            ws_max,
        );
    }
}

/// Port of `_deriv1_cart` for one ECP slot (Type-1 if `lc < 0`, else Type-2
/// projector channel `lc`). Accumulates the 3-component gradient into `gctr`,
/// laid out F-order `[axis, ao_j, ao_i]` (axis slowest): `gctr` is three
/// consecutive `dij = nci*nfi * ncj*nfj` blocks (x, y, z), each block F-order
/// `n + j*di + i` with `di = nci*nfi`. For the Cu/LANL2DZ fixture nci=ncj=1 so
/// `dij = nfi*nfj` and the per-axis block is exactly `[ao_j, ao_i]`.
/// Both `compute_type1_pair_grad` (lc<0) and `compute_type2_pair_grad` (lc>=0)
/// route through this single driver because nr_ecp_deriv.c's `_deriv1_cart`
/// sums Type-1+Type-2 into the same buffer (line 256-259 / 265-268).
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:201-286.
#[allow(clippy::too_many_arguments)]
fn deriv1_cart_pair(
    backend: &ResolvedBackend,
    gctr: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    lc: i32,
    rad_shells: &[EcpRadShell],
    rs_max: &[f64],
    ws_max: &[f64],
) {
    let li = shell_i.ang_momentum as usize;
    let lj = shell_j.ang_momentum as usize;
    let npi = shell_i.nprim as usize;
    let npj = shell_j.nprim as usize;
    let nci = shell_i.nctr as usize;
    let ncj = shell_j.nctr as usize;
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfi0 = if li > 0 { ncart((li - 1) as u8) } else { 0 };
    let nfi1 = ncart((li + 1) as u8);
    let di = nfi * nci;
    let dj = nfj * ncj;
    let dij = di * dj;
    let expi = &shell_i.exponents;
    let expj = &shell_j.exponents;
    // Canonical Shell.coefficients are row-major [ip*nctr+ic]; this ECP kernel's
    // internal indexing is column-major [ic*nprim+ip]. Convert once (identity for
    // nctr==1). Without this, generally-contracted shells read scrambled coeffs.
    let ci_cm = coeffs_col_major(shell_i);
    let cj_cm = coeffs_col_major(shell_j);
    let ci = &ci_cm;
    let cj = &cj_cm;

    // gprim = [gpx | gpy | gpz], each nfi*nfj, F-order j*nfi+i.
    let mut gprim = vec![0.0f64; 3 * nfi * nfj];
    let mut buf1 = vec![0.0f64; nfi1 * nfj];

    for jp in 0..npj {
        for ip in 0..npi {
            // divide expi*expj — exponents were used as coefficients on the
            // uncontracted fake shells (nr_ecp_deriv.c:251-253).
            let fac = 1.0 / (expi[ip] * expj[jp]);

            // l+1 term → _l_down.
            for v in buf1[..nfi1 * nfj].iter_mut() {
                *v = 0.0;
            }
            ecp_scalar_prim_pair_cart(
                backend,
                &mut buf1[..nfi1 * nfj],
                li + 1,
                lj,
                expi[ip],
                expj[jp],
                ri,
                rj,
                rc,
                lc,
                rad_shells,
                rs_max,
                ws_max,
            );
            l_down(&mut gprim, &buf1[..nfi1 * nfj], fac, expi[ip], li, nfj);

            // l-1 term → _l_up (only if li > 0).
            if li > 0 {
                for v in buf1[..nfi0 * nfj].iter_mut() {
                    *v = 0.0;
                }
                ecp_scalar_prim_pair_cart(
                    backend,
                    &mut buf1[..nfi0 * nfj],
                    li - 1,
                    lj,
                    expi[ip],
                    expj[jp],
                    ri,
                    rj,
                    rc,
                    lc,
                    rad_shells,
                    rs_max,
                    ws_max,
                );
                l_up(&mut gprim, &buf1[..nfi0 * nfj], fac, li, nfj);
            }

            // Re-apply AO contraction coefficients into the 3 axis blocks.
            let (gpx, rest) = gprim.split_at(nfi * nfj);
            let (gpy, gpz) = rest.split_at(nfi * nfj);
            for jc in 0..ncj {
                for ic in 0..nci {
                    let cfac = ci[ic * npi + ip] * cj[jc * npj + jp];
                    let n = jc * nfj * di + ic * nfi;
                    for j in 0..nfj {
                        for i in 0..nfi {
                            let g = n + j * di + i;
                            gctr[g] += cfac * gpx[j * nfi + i];
                            gctr[dij + g] += cfac * gpy[j * nfi + i];
                            gctr[2 * dij + g] += cfac * gpz[j * nfi + i];
                        }
                    }
                }
            }
            // gprim is reset per primitive pair (matches PySCF's _l_down which
            // overwrites and _l_up which accumulates within one (ip,jp)).
            for v in gprim.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

/// `compute_type1_pair_grad` — the Local (Type-1) gradient channel of
/// `_deriv1_cart`. Thin alias over `deriv1_cart_pair` with `lc = -1`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:256-257 (ECPtype1_cart at l±1).
#[allow(clippy::too_many_arguments)]
fn compute_type1_pair_grad(
    backend: &ResolvedBackend,
    gctr: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    rad_shells: &[EcpRadShell],
    rs_max: &[f64],
    ws_max: &[f64],
) {
    deriv1_cart_pair(
        backend, gctr, shell_i, shell_j, ri, rj, rc, -1, rad_shells, rs_max, ws_max,
    );
}

/// `compute_type2_pair_grad` — the Projected (Type-2) gradient channel of
/// `_deriv1_cart` for projector angular momentum `lc`. Thin alias over
/// `deriv1_cart_pair`.
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:258-259 (ECPtype2_cart at l±1).
#[allow(clippy::too_many_arguments)]
fn compute_type2_pair_grad(
    backend: &ResolvedBackend,
    gctr: &mut [f64],
    shell_i: &Shell,
    shell_j: &Shell,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    lc: usize,
    rad_shells: &[EcpRadShell],
    rs_max: &[f64],
    ws_max: &[f64],
) {
    deriv1_cart_pair(
        backend, gctr, shell_i, shell_j, ri, rj, rc, lc as i32, rad_shells, rs_max, ws_max,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher
// ─────────────────────────────────────────────────────────────────────────────

/// ECP integral host-side launcher for `canonical_family = "ecp"`.
///
/// Dispatches Type-1 (channel == Local) and Type-2 (channel == Projected(l))
/// ECP shells. For the scalar `int1e_ecp_*` operator, accumulates a Cartesian
/// `[ao_i, ao_j]` buffer and applies `cart_to_sph_1e` when
/// `plan.representation == Spheric`. For the gradient `int1e_ecp_ipnuc_*`
/// operator (`operator_name == "ecp_ipnuc"`, component_rank=3) it accumulates
/// the 3-component `∂/∂A_i^{x,y,z}` gradient via `_deriv1_cart`
/// (`compute_type1_pair_grad` / `compute_type2_pair_grad`), writing F-order
/// `[axis, ao_j, ao_i]` (axis slowest) and applying `cart_to_sph_1e` per axis
/// for the spherical representation.
///
/// For the per-nucleus `int1e_ecp_iprinv_*` operator
/// (`operator_name == "ecp_iprinv"`, component_rank=3, 21-07/D-09) it reuses the
/// SAME comp=3 `deriv1_cart_pair` driver but selects ONLY the single ECP shell
/// on the atom whose coordinate matches the rinv origin
/// (`plan.operator_env_params.rinv_orig`) — dropping the `ipnuc` all-slot/`-Z_C`
/// accumulation. An origin matching no atom yields a zero-filled output
/// (mirroring the vendor `_one_shell_ecpbas` shl_id<0 → 0); the spinor iprinv
/// representation fails closed with `UnsupportedApi` (R5).
pub fn launch_ecp(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "ecp" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail: format!(
                "canonical_family mismatch: expected ecp, got {}",
                specialization.canonical_family()
            ),
        });
    }

    // `backend` is now threaded into the Type-1 angular-splice device dispatch
    // (`run_ecp_angular_splice_on_backend`) — no longer a host-only pipeline
    // (quick-260529-gbf).

    let operator_name = plan.descriptor.operator_name();
    let is_gradient = match operator_name {
        "ecp" => false,
        // ecp_ipnuc: ∂/∂A_i gradient summed over ALL ECP slots (Phase 19-07).
        // ecp_iprinv: ∂/∂A_i gradient on the SINGLE atom matching the rinv
        //   origin, no all-slot/-Z_C accumulation (21-07, D-09).
        "ecp_ipnuc" | "ecp_iprinv" => true,
        other => {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("unknown ecp operator name: {other}"),
            });
        }
    };
    // iprinv differs from ipnuc only in single-atom selection (D-09); both reuse
    // the same comp=3 deriv1_cart_pair driver below.
    let is_iprinv = operator_name == "ecp_iprinv";

    let shells = plan.shells.as_slice();
    if shells.len() != 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail: format!("ecp kernel requires exactly 2 shells, got {}", shells.len()),
        });
    }

    let ecp_shells: &[Arc<EcpShell>] = plan.basis.ecp_shells();
    if ecp_shells.is_empty() {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail:
                "ecp_shells empty at kernel layer — query_workspace preflight should have caught this"
                    .to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);
    // Contraction counts: generally-contracted shells (nctr>1) produce
    // ao_per_shell = nctr * (ncart|nsph) AOs per axis, laid out contraction-major.
    let nctr_i = shell_i.nctr as usize;
    let nctr_j = shell_j.nctr as usize;
    let di_cart = nctr_i * nci;
    let dj_cart = nctr_j * ncj;
    let di_sph = nctr_i * nsi;
    let dj_sph = nctr_j * nsj;

    let atoms: &[Atom] = plan.basis.atoms();
    let atom_count = atoms.len();
    // CLAUDE.md fail-closed: a malformed BasisSet must surface a typed error, not
    // an array-index panic in this public-API-reachable path.
    let ai = shell_i.atom_index as usize;
    if ai >= atom_count {
        return Err(cintxRsError::InvalidShellAtomIndex {
            index: ai,
            atom_count,
        });
    }
    let aj = shell_j.atom_index as usize;
    if aj >= atom_count {
        return Err(cintxRsError::InvalidShellAtomIndex {
            index: aj,
            atom_count,
        });
    }
    let ri = atoms[ai].coord_bohr;
    let rj = atoms[aj].coord_bohr;

    // R5 (T-21-07-04): the spinor ECP gradient is NOT byte-identity-gated this
    // phase. The scalar/ipnuc spinor path is the D-12 zero-write escape hatch,
    // but `ecp_iprinv` is a fresh per-nucleus force surface added here — fail it
    // CLOSED with a typed UnsupportedApi rather than emit unverified zeros, so a
    // caller never silently trusts an iprinv spinor result (21-07 behavior spec).
    if is_iprinv && matches!(plan.representation, Representation::Spinor) {
        return Err(cintxRsError::UnsupportedApi {
            requested: "int1e_ecp_iprinv_spinor (spinor ECP gradient, R5)".to_owned(),
        });
    }

    // Number of output components: 1 for scalar `ecp`, 3 for gradient
    // `ecp_ipnuc` (∂/∂A_i^{x,y,z}, F-order [axis, ao_j, ao_i] axis slowest).
    let n_comp = if is_gradient { 3 } else { 1 };

    // Output buffer-size invariant (T-19-23, extended to the scalar path per the
    // CLAUDE.md fail-closed / no-partial-write contract): the staging slice MUST
    // hold n_comp * (cart-or-sph product) for BOTH scalar (n_comp=1) and gradient
    // (n_comp=3). Reject an undersized buffer with a typed BufferTooSmall error
    // rather than silently truncating / under-filling and returning Ok.
    // Spinor is the D-12 zero-write path; sized by the caller.
    let needed = match plan.representation {
        Representation::Spheric => n_comp * di_sph * dj_sph,
        Representation::Cart => n_comp * di_cart * dj_cart,
        Representation::Spinor => 0,
    };
    if !matches!(plan.representation, Representation::Spinor) && staging.len() < needed {
        return Err(cintxRsError::BufferTooSmall {
            required: needed,
            provided: staging.len(),
        });
    }

    // gctr is the cartesian shell-pair output. For scalar this is F-order
    // [ao_i, ao_j] (length nci*ncj). For gradient it is 3 consecutive
    // dij = (nci*nfi)*(ncj*nfj) blocks (x, y, z) — F-order [axis, ao_j, ao_i],
    // axis slowest (D-11, int3c2e_ip1_* precedent). PySCF's _deriv1_cart writes
    // the SAME [comp, j*di+i] layout (nr_ecp_deriv.c:240-280) — no transpose.
    let mut gctr = vec![0.0_f64; n_comp * di_cart * dj_cart];

    // Build the 2047-point Gauss-Chebyshev grid PySCF samples adaptively
    // (rs/ws_gauss_chebyshev2047). gauss_chebyshev_nodes_weights_host(LEVEL_MAX)
    // reproduces ECPgauss_chebyshev(.., 2047) to f64; the adaptive loop indexes
    // rs[start + n*step] into it.
    let (rs_max, ws_max) = gauss_chebyshev_nodes_weights_host(LEVEL_MAX);

    let slots = group_ecp_slots(ecp_shells);

    // D-09: `ecp_iprinv` is a per-nucleus selector. Resolve the rinv origin up
    // front (the validator should reject a None origin before the kernel, but we
    // fail typed — never panic, never read a garbage origin; mirrors the
    // one_electron.rs iprinv defensive gate, T-21-07-02). Then select ONLY the
    // slot(s) on the atom whose coordinate matches the origin. An origin matching
    // no atom yields an empty selection → a zero-filled output (mirroring the
    // vendor `_one_shell_ecpbas` shl_id<0 → 0 path). `ecp`/`ecp_ipnuc` are
    // unaffected: they process every slot (None = "no restriction").
    let iprinv_selected: Option<Vec<usize>> = if is_iprinv {
        let origin = plan
            .operator_env_params
            .rinv_orig
            .ok_or(cintxRsError::InvalidEnvParam {
                param: "PTR_RINV_ORIG",
                reason: "ecp_iprinv kernel reached with no rinv origin".to_owned(),
            })?;
        Some(select_iprinv_slots(&slots, atoms, origin))
    } else {
        None
    };

    for (slot_idx, slot) in slots.iter().enumerate() {
        // iprinv: skip every slot NOT on the selected nucleus (no all-slot
        // accumulation, no -Z_C). For an unmatched origin `iprinv_selected` is
        // an empty Vec, so EVERY slot is skipped → zero output (T-21-07-03).
        if let Some(selected) = &iprinv_selected
            && !selected.contains(&slot_idx)
        {
            continue;
        }
        let ac = slot.atom_index as usize;
        if ac >= atom_count {
            return Err(cintxRsError::InvalidShellAtomIndex {
                index: ac,
                atom_count,
            });
        }
        let rc = atoms[ac].coord_bohr;
        if slot.lc >= 0 {
            let lc = slot.lc as usize;
            if lc > ECP_LMAX {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("ecp projector l={lc} exceeds ECP_LMAX={ECP_LMAX}"),
                });
            }
        }
        match (is_gradient, slot.lc < 0) {
            (false, true) => ecp_type1_cart(
                backend,
                &mut gctr,
                shell_i,
                shell_j,
                ri,
                rj,
                rc,
                &slot.rad_shells,
                &rs_max,
                &ws_max,
            ),
            (false, false) => ecp_type2_cart(
                backend,
                &mut gctr,
                shell_i,
                shell_j,
                ri,
                rj,
                rc,
                slot.lc as usize,
                &slot.rad_shells,
                &rs_max,
                &ws_max,
            ),
            // Gradient: route Local → compute_type1_pair_grad,
            // Projected(l) → compute_type2_pair_grad (both via deriv1_cart_pair).
            (true, true) => compute_type1_pair_grad(
                backend,
                &mut gctr,
                shell_i,
                shell_j,
                ri,
                rj,
                rc,
                &slot.rad_shells,
                &rs_max,
                &ws_max,
            ),
            (true, false) => compute_type2_pair_grad(
                backend,
                &mut gctr,
                shell_i,
                shell_j,
                ri,
                rj,
                rc,
                slot.lc as usize,
                &slot.rad_shells,
                &rs_max,
                &ws_max,
            ),
        }
    }

    // Apply representation transform and write into staging.
    match plan.representation {
        Representation::Spheric => {
            // gctr is the cartesian [di_cart, dj_cart] block per axis, column-major
            // (bra fastest) and contraction-major within each axis (AO = ctr*ncart+m,
            // matching ecp_type1/2_cart's `(jc*nfj+mj)*(nctr_i*ncart_i)+ic*nfi+mi`).
            // Transform each (ci,cj) contraction sub-block to spherical and scatter
            // it into the contraction-major sph grid staging[ii + jj*di_sph]
            // (ii = ci*nsi+msi, jj = cj*nsj+msj). For nctr==1 this is byte-identical
            // to the prior single-block cart_to_sph_1e linear write.
            let cart_block = di_cart * dj_cart;
            let sph_block = di_sph * dj_sph;
            let mut sub = vec![0.0_f64; nci * ncj];
            let mut sph_tmp = vec![0.0_f64; nsi * nsj];
            for axis in 0..n_comp {
                let axis_cart = &gctr[axis * cart_block..(axis + 1) * cart_block];
                let axis_off = axis * sph_block;
                for ci in 0..nctr_i {
                    for cj in 0..nctr_j {
                        // Extract the (ci,cj) cartesian sub-block in cart_to_sph_1e's
                        // expected layout: sub[mj*nci + mi] (ket cart outer, bra inner).
                        for mj in 0..ncj {
                            for mi in 0..nci {
                                let g = (ci * nci + mi) + (cj * ncj + mj) * di_cart;
                                sub[mj * nci + mi] = axis_cart[g];
                            }
                        }
                        cart_to_sph_1e(&sub, &mut sph_tmp, li, lj);
                        for msj in 0..nsj {
                            for msi in 0..nsi {
                                let dst = axis_off + (ci * nsi + msi) + (cj * nsj + msj) * di_sph;
                                staging[dst] = sph_tmp[msj * nsi + msi];
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            // gctr.len() == needed (Cart) and staging.len() >= needed by the
            // invariant above — full copy, no truncation (CLAUDE.md no-partial-write).
            staging[..gctr.len()].copy_from_slice(&gctr);
        }
        Representation::Spinor => {
            // D-12: spinor accepted by resolver but NOT byte-identity-gated this
            // phase. Write zeros (compiled-but-unverified per Phase 18 precedent).
            for slot in staging.iter_mut() {
                *slot = 0.0;
            }
        }
    }

    let not0 = staging.iter().filter(|&&v| v.abs() > 1e-18).count() as i32;
    let staging_bytes = staging.len() * std::mem::size_of::<f64>();
    Ok(ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: staging_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: staging_bytes,
        not0,
        fallback_reason: plan.workspace.fallback_reason,
    })
}

/// Single Condon-Shortley cart2sph coefficient (mirrors `c2s.rs::c2s_coeff`,
/// kept local because that helper is private to the transform module).
/// `C2S_L*[m_row][cart_col] == libcint g_trans_cart2sph` rows (l <= 4).
#[inline]
fn c2s_coeff(l: usize, m_row: usize, cart_col: usize) -> f64 {
    use crate::transform::c2s::{C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4};
    match l {
        0 => C2S_L0[m_row][cart_col],
        1 => C2S_L1[m_row][cart_col],
        2 => C2S_L2[m_row][cart_col],
        3 => C2S_L3[m_row][cart_col],
        4 => C2S_L4[m_row][cart_col],
        _ => 0.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-internal unit tests.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Module-internal unit tests for the ECP kernel helpers and dispatch.
    //! End-to-end byte-identity coverage of `launch_ecp` lives in
    //! `crates/cintx-oracle/tests/safe_api_ecp_parity.rs`.

    use super::*;

    #[test]
    fn launch_ecp_registered_under_canonical_family_ecp() {
        let resolved = crate::kernels::resolve_family_name_for_tests("ecp");
        assert!(
            resolved.is_some(),
            "kernels::resolve_family_name(\"ecp\") must return Some(launch_ecp)"
        );
        assert!(
            crate::kernels::supports_canonical_family("ecp"),
            "supports_canonical_family(\"ecp\") must return true (D-09 stable, unconditional)"
        );
    }

    #[test]
    fn cart_comps_returns_expected_count() {
        assert_eq!(cart_comps(0).len(), 1); // s
        assert_eq!(cart_comps(1).len(), 3); // p
        assert_eq!(cart_comps(2).len(), 6); // d
        assert_eq!(cart_comps(3).len(), 10); // f
    }

    /// `int_unit_xyz` matches PySCF: 1/(4π)·(unit-sphere monomial integral).
    /// `int_unit_xyz(0,0,0) = factorial2(-1)^3 / factorial2(1) = 1/1 = 1`.
    #[test]
    fn int_unit_xyz_basic_values() {
        assert_eq!(int_unit_xyz(0, 0, 0), 1.0);
        // odd power => 0
        assert_eq!(int_unit_xyz(1, 0, 0), 0.0);
        // x^2: factorial2(1)*factorial2(-1)*factorial2(-1)/factorial2(3) = 1/3
        assert!((int_unit_xyz(2, 0, 0) - 1.0 / 3.0).abs() < 1e-15);
    }

    /// `binom` agrees with the closed-form for the small-n table.
    #[test]
    fn binom_table_values() {
        assert_eq!(binom(0, 0), 1.0);
        assert_eq!(binom(4, 2), 6.0);
        assert_eq!(binom(5, 2), 10.0);
    }

    /// `cint_common_fac_sp` matches libcint g1e.c:566.
    #[test]
    fn common_fac_sp_values() {
        assert_eq!(cint_common_fac_sp(0), 0.282094791773878143);
        assert_eq!(cint_common_fac_sp(1), 0.488602511902919921);
        assert_eq!(cint_common_fac_sp(2), 1.0);
    }

    // ── Gradient (ipnuc / _deriv1_cart) helpers ───────────────────────────────

    /// The gradient operator `ecp_ipnuc` resolves to the same `launch_ecp`
    /// launcher under canonical_family "ecp" (positive assertion — replaces the
    /// 19-04 "rejects gradient operator name" stub).
    #[test]
    fn gradient_operator_routes_through_launch_ecp() {
        // Both scalar and gradient operators live under canonical_family "ecp".
        assert!(
            crate::kernels::resolve_family_name_for_tests("ecp").is_some(),
            "ecp family (scalar + ecp_ipnuc gradient) must resolve to launch_ecp"
        );
        assert!(crate::kernels::supports_canonical_family("ecp"));
    }

    /// `raise_idx`: x-raise is the identity prefix; y/z raise land on the
    /// expected cart_comps(l+1) component.
    #[test]
    fn raise_idx_matches_cart_comps() {
        for l in 0..4usize {
            for (i, &(lx, ly, lz)) in cart_comps(l as u8).iter().enumerate() {
                assert_eq!(raise_idx(l, i, 0), i, "x-raise must be identity prefix");
                let yc = cart_comps((l + 1) as u8);
                assert_eq!(yc[raise_idx(l, i, 1)], (lx, ly + 1, lz));
                assert_eq!(yc[raise_idx(l, i, 2)], (lx, ly, lz + 1));
            }
        }
    }

    /// Zero-overlap sanity: an AO shell pair placed far from the ECP center
    /// yields a near-zero 3-component gradient buffer (the radial Gaussian
    /// `exp(-a r_CA^2)` damps it to machine epsilon). Exercises the full
    /// `deriv1_cart_pair` driver and asserts the buffer is exactly 3*nfi*nfj.
    #[cfg(feature = "cpu")]
    #[test]
    fn gradient_zero_overlap_is_negligible() {
        let backend = ResolvedBackend::Cpu(cubecl::cpu::CpuRuntime::client(&Default::default()));
        let (rs_max, ws_max) = gauss_chebyshev_nodes_weights_host(LEVEL_MAX);
        // s-shell bra/ket, single tight primitive.
        let mk = |l: u8| {
            Shell::try_new(
                0,
                l,
                1,
                1,
                0,
                Representation::Cart,
                Arc::from(vec![3.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap()
        };
        let si = mk(0);
        let sj = mk(0);
        let nfi = ncart(0);
        let nfj = ncart(0);
        let mut gctr = vec![0.0_f64; 3 * nfi * nfj];
        // ECP center 50 bohr away from the AO centers at origin.
        let rad = EcpRadShell {
            exponents: vec![2.0],
            coefficients: vec![1.0],
            radial_power: 0,
        };
        deriv1_cart_pair(
            &backend,
            &mut gctr,
            &si,
            &sj,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            -1, // Local (Type-1)
            std::slice::from_ref(&rad),
            &rs_max,
            &ws_max,
        );
        assert_eq!(gctr.len(), 3 * nfi * nfj);
        for (k, &v) in gctr.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "far-separated ECP gradient component {k} should vanish, got {v:e}"
            );
        }
    }

    /// The `s`-shell (li=0) gradient takes ONLY the `_l_down` path (no `_l_up`,
    /// since li=0). A finite, non-zero ECP-center-coincident s-s gradient must
    /// produce a 3-component buffer where the symmetry of the configuration is
    /// reflected. Here we assert the buffer is sized and that an on-center
    /// (rca=0) configuration yields a finite (not NaN/Inf) result.
    #[cfg(feature = "cpu")]
    #[test]
    fn gradient_on_center_is_finite() {
        let backend = ResolvedBackend::Cpu(cubecl::cpu::CpuRuntime::client(&Default::default()));
        let (rs_max, ws_max) = gauss_chebyshev_nodes_weights_host(LEVEL_MAX);
        let si = Shell::try_new(
            0,
            1,
            1,
            1,
            0,
            Representation::Cart,
            Arc::from(vec![1.5_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice()),
        )
        .unwrap();
        let sj = Shell::try_new(
            0,
            0,
            1,
            1,
            0,
            Representation::Cart,
            Arc::from(vec![1.2_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice()),
        )
        .unwrap();
        let nfi = ncart(1);
        let nfj = ncart(0);
        let mut gctr = vec![0.0_f64; 3 * nfi * nfj];
        let rad = EcpRadShell {
            exponents: vec![1.0],
            coefficients: vec![1.0],
            radial_power: 0,
        };
        deriv1_cart_pair(
            &backend,
            &mut gctr,
            &si,
            &sj,
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0],
            -1,
            std::slice::from_ref(&rad),
            &rs_max,
            &ws_max,
        );
        assert_eq!(gctr.len(), 3 * nfi * nfj);
        assert!(
            gctr.iter().all(|v| v.is_finite()),
            "gradient must be finite"
        );
    }

    // ── ecp_iprinv per-nucleus selector (21-07, D-09) ─────────────────────────

    /// Build a small set of `Atom`s for selector tests (atomic_number is
    /// irrelevant to iprinv — no `-Z_C` factor — so any positive value works).
    fn mk_atoms(coords: &[[f64; 3]]) -> Vec<Atom> {
        use cintx_core::atom::NuclearModel;
        coords
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                Atom::try_new((i as u16) + 1, c, NuclearModel::Point, None, None).unwrap()
            })
            .collect()
    }

    /// Build a single-primitive `EcpShell` on `atom_index` for slot grouping.
    fn mk_ecp_shell_channel(atom_index: u32, channel: EcpChannel) -> Arc<EcpShell> {
        EcpShell::try_new(
            atom_index,
            channel,
            0, // radial_power
            1, // nprim
            1, // nctr
            0, // so_type
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice()),
        )
        .map(Arc::new)
        .unwrap()
    }

    /// Build a single Local-channel `EcpShell` on `atom_index` for slot grouping.
    fn mk_ecp_shell(atom_index: u32) -> Arc<EcpShell> {
        mk_ecp_shell_channel(atom_index, EcpChannel::Local)
    }

    /// The single-atom selector returns ONLY the slot(s) on the atom whose
    /// coordinate matches the rinv origin — the key proof that `iprinv` is NOT
    /// the all-slot `ipnuc` accumulation (D-09).
    #[test]
    fn iprinv_selects_only_the_origin_atom() {
        // Two ECP-bearing atoms at distinct coordinates.
        let atoms = mk_atoms(&[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]);
        let ecp_shells = [mk_ecp_shell(0), mk_ecp_shell(1)];
        let slots = group_ecp_slots(&ecp_shells);
        assert_eq!(slots.len(), 2, "two atoms → two Local slots");

        // Origin == atom 1's coordinate → ONLY slot 1 selected.
        let sel = select_iprinv_slots(&slots, &atoms, [0.0, 0.0, 2.5]);
        assert_eq!(sel, vec![1], "iprinv at atom 1 must select only slot 1");

        // Origin == atom 0's coordinate → ONLY slot 0 selected.
        let sel0 = select_iprinv_slots(&slots, &atoms, [0.0, 0.0, 0.0]);
        assert_eq!(sel0, vec![0], "iprinv at atom 0 must select only slot 0");

        // A single nucleus with multiple channels: all its slots are selected
        // (still strictly per-nucleus), proving we never silently drop channels.
        let multi = [
            mk_ecp_shell(0),
            mk_ecp_shell_channel(0, EcpChannel::Projected(1)),
        ];
        let multi_slots = group_ecp_slots(&multi);
        let multi_sel = select_iprinv_slots(&multi_slots, &atoms, [0.0, 0.0, 0.0]);
        assert_eq!(
            multi_sel.len(),
            multi_slots.len(),
            "all channels on the selected nucleus are kept"
        );
    }

    /// An `iprinv` origin matching NO atom returns an empty selection (→ the
    /// launcher writes a zero-filled output), mirroring the vendor
    /// `_one_shell_ecpbas` shl_id<0 → 0 path — never a panic, never a
    /// wrong-atom selection (T-21-07-02).
    #[test]
    fn iprinv_origin_matching_no_atom_selects_nothing() {
        let atoms = mk_atoms(&[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]);
        let ecp_shells = [mk_ecp_shell(0), mk_ecp_shell(1)];
        let slots = group_ecp_slots(&ecp_shells);
        let sel = select_iprinv_slots(&slots, &atoms, [10.0, 10.0, 10.0]);
        assert!(
            sel.is_empty(),
            "no-match origin selects no slot (zero output)"
        );
    }

    /// The selector tolerance is tight: an origin off by more than
    /// `IPRINV_ORIGIN_MATCH_TOL` does NOT match, while one within it does.
    #[test]
    fn iprinv_origin_match_is_tight() {
        let atoms = mk_atoms(&[[1.0, 2.0, 3.0]]);
        let ecp_shells = [mk_ecp_shell(0)];
        let slots = group_ecp_slots(&ecp_shells);

        // Within tolerance → match.
        let near = select_iprinv_slots(&slots, &atoms, [1.0, 2.0, 3.0 + 1e-11]);
        assert_eq!(near, vec![0], "origin within 1e-10 must match");

        // Outside tolerance → no match.
        let far = select_iprinv_slots(&slots, &atoms, [1.0, 2.0, 3.0 + 1e-6]);
        assert!(far.is_empty(), "origin off by 1e-6 must NOT match");
    }

    /// `ecp_iprinv` is routed under the same canonical_family "ecp" launcher as
    /// the scalar/ipnuc operators (the dispatcher resolves by family, then
    /// branches on operator_name internally).
    #[test]
    fn iprinv_operator_routes_through_launch_ecp() {
        assert!(
            crate::kernels::resolve_family_name_for_tests("ecp").is_some(),
            "ecp family (scalar + ipnuc + iprinv) must resolve to launch_ecp"
        );
        assert!(crate::kernels::supports_canonical_family("ecp"));
    }

    // ── Phase-B angular splice: device-vs-host equivalence (quick 260529-gbf) ──
    #[cfg(feature = "cpu")]
    mod ecp_angular_device_cross_check {
        use super::super::*;

        fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
            cubecl::cpu::CpuRuntime::client(&Default::default())
        }

        /// Tiny LCG so the random inputs are deterministic and reproducible.
        struct Lcg(u64);
        impl Lcg {
            fn new(seed: u64) -> Self {
                Lcg(seed)
            }
            fn next_f64(&mut self) -> f64 {
                // Numerical-Recipes LCG; map to [-1, 1).
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
                2.0 * u - 1.0
            }
        }

        /// Host reference splice for a SINGLE (ic=0,jc=0) tuple — the exact
        /// arithmetic + loop order of `ecp_type1_cart`'s tail (ecp.rs ~783-817),
        /// computed in f64. The device kernel must reproduce this byte-for-byte.
        fn host_angular_splice(
            li: usize,
            lj: usize,
            rad_ang: &[f64],
            ifac: &[f64],
            jfac: &[f64],
        ) -> Vec<f64> {
            let nfi = ncart(li as u8);
            let nfj = ncart(lj as u8);
            let d1 = li + lj + 1;
            let d2 = d1 * d1;
            let di1 = li + 1;
            let di2 = di1 * di1;
            let di3 = di2 * di1;
            let dj1 = lj + 1;
            let dj2 = dj1 * dj1;
            let dj3 = dj2 * dj1;
            let comps_i = cart_comps(li as u8);
            let comps_j = cart_comps(lj as u8);
            let mut out = vec![0.0_f64; nfi * nfj];
            for (mi, &(ix, iy, iz)) in comps_i.iter().enumerate() {
                let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
                for (mj, &(jx, jy, jz)) in comps_j.iter().enumerate() {
                    let (jx, jy, jz) = (jx as usize, jy as usize, jz as usize);
                    let pifac = &ifac[mi * di3..mi * di3 + di3];
                    let pjfac = &jfac[mj * dj3..mj * dj3 + dj3];
                    let mut acc = 0.0;
                    for i1 in 0..=ix {
                        for i2 in 0..=iy {
                            for i3 in 0..=iz {
                                for j1 in 0..=jx {
                                    for j2 in 0..=jy {
                                        for j3 in 0..=jz {
                                            acc += pifac[i1 * di2 + i2 * di1 + i3]
                                                * pjfac[j1 * dj2 + j2 * dj1 + j3]
                                                * rad_ang
                                                    [(i1 + j1) * d2 + (i2 + j2) * d1 + i3 + j3];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out[mj * nfi + mi] = acc;
                }
            }
            out
        }

        fn random_inputs(li: usize, lj: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let mut rng = Lcg::new(seed);
            let d1 = li + lj + 1;
            let d3 = d1 * d1 * d1;
            let nfi = ncart(li as u8);
            let nfj = ncart(lj as u8);
            let di3 = (li + 1).pow(3);
            let dj3 = (lj + 1).pow(3);
            let rad_ang: Vec<f64> = (0..d3).map(|_| rng.next_f64()).collect();
            let ifac: Vec<f64> = (0..nfi * di3).map(|_| rng.next_f64()).collect();
            let jfac: Vec<f64> = (0..nfj * dj3).map(|_| rng.next_f64()).collect();
            (rad_ang, ifac, jfac)
        }

        /// f64 device-vs-host equivalence: max-abs-diff MUST be exactly 0.0
        /// (identical f64 op order ⇒ byte-identity).
        fn assert_f64_byte_identity(li: usize, lj: usize, seed: u64) {
            let (rad_ang, ifac, jfac) = random_inputs(li, lj, seed);
            let host = host_angular_splice(li, lj, &rad_ang, &ifac, &jfac);
            let comps_i = cart_comps_flat_u32(li as u8);
            let comps_j = cart_comps_flat_u32(lj as u8);
            let dev = run_ecp_angular_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client(),
                li as u32,
                lj as u32,
                &rad_ang,
                &ifac,
                &jfac,
                &comps_i,
                &comps_j,
            );
            assert_eq!(host.len(), dev.len(), "len mismatch li={li} lj={lj}");
            let mut max_diff = 0.0_f64;
            let mut any_nonzero = false;
            for (&h, &d) in host.iter().zip(dev.iter()) {
                if h.abs() > 1e-18 {
                    any_nonzero = true;
                }
                max_diff = max_diff.max((h - d).abs());
            }
            assert_eq!(
                max_diff, 0.0,
                "ecp angular device/host f64 max-abs-diff must be 0.0 (li={li} lj={lj}), got {max_diff:e}"
            );
            assert!(any_nonzero, "host reference all zeros (li={li} lj={lj})");
        }

        #[test]
        fn ecp_angular_device_matches_host_f64() {
            // Several (li,lj) up to (2,2).
            for (li, lj) in [
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (0, 2),
                (2, 1),
                (1, 2),
                (2, 2),
            ] {
                assert_f64_byte_identity(li, lj, 0x9E3779B97F4A7C15 ^ ((li * 7 + lj) as u64));
            }
        }

        /// Generic-F: running the SAME kernel at F=f32 on CpuRuntime reproduces
        /// the f64 result cast to f32 within f32 eps. Proves `ecp_angular_kernel`
        /// is genuinely generic over `F: Float`.
        #[test]
        fn ecp_angular_device_generic_f32_within_eps() {
            let (li, lj) = (2usize, 1usize);
            let (rad_ang, ifac, jfac) = random_inputs(li, lj, 0xDEADBEEFCAFEF00D);
            let host = host_angular_splice(li, lj, &rad_ang, &ifac, &jfac);

            let nfi = ncart(li as u8);
            let nfj = ncart(lj as u8);
            let out_len = nfi * nfj;
            let comps_i = cart_comps_flat_u32(li as u8);
            let comps_j = cart_comps_flat_u32(lj as u8);

            let client = cpu_client();
            let rad_f32: Vec<f32> = rad_ang.iter().map(|&v| v as f32).collect();
            let ifac_f32: Vec<f32> = ifac.iter().map(|&v| v as f32).collect();
            let jfac_f32: Vec<f32> = jfac.iter().map(|&v| v as f32).collect();
            let rad_h = client.create_from_slice(f32::as_bytes(&rad_f32));
            let ifac_h = client.create_from_slice(f32::as_bytes(&ifac_f32));
            let jfac_h = client.create_from_slice(f32::as_bytes(&jfac_f32));
            let comps_i_h = client.create_from_slice(u32::as_bytes(&comps_i));
            let comps_j_h = client.create_from_slice(u32::as_bytes(&comps_j));
            let out_zero = vec![0.0_f32; out_len];
            let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

            ecp_angular_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
                &client,
                crate::plane::single_cube_count(),
                crate::plane::standard_plane_cube_dim(),
                unsafe { ArrayArg::from_raw_parts(rad_h, rad_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(ifac_h, ifac_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(jfac_h, jfac_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(comps_i_h, comps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(comps_j_h, comps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                li as u32,
                lj as u32,
                nfi as u32,
                nfj as u32,
            );
            let raw = client.read_one_unchecked(out_h);
            let dev_f32 = &f32::from_bytes(&raw)[0..out_len];

            for (&h, &d) in host.iter().zip(dev_f32.iter()) {
                let diff = (h as f32 - d).abs();
                let thr = 1e-4_f32 + 1e-4_f32 * (h.abs() as f32);
                assert!(
                    diff <= thr,
                    "f32 device result not within eps: host={h:e} dev={d:e} diff={diff:e}"
                );
            }
        }

        // ── Type-2 (two-dgemm) angular splice: device-vs-host (quick 260529-hin) ──

        /// Host reference for the Type-2 two-`dgemm` splice for a SINGLE (ic=0,jc=0)
        /// tuple — the EXACT arithmetic + loop order of `ecp_type2_cart`'s tail
        /// (ecp.rs:1272-1309), including the materialized `buf` intermediate,
        /// computed in f64. The device kernel must reproduce this byte-for-byte.
        #[allow(clippy::too_many_arguments)]
        fn host_type2_splice(
            li: usize,
            lj: usize,
            lc: usize,
            prad: &[f64],
            angi: &[f64],
            angj: &[f64],
            common_fac: f64,
        ) -> Vec<f64> {
            let nfi = ncart(li as u8);
            let nfj = ncart(lj as u8);
            let dlc = lc * 2 + 1;
            let lilc1 = li + lc + 1;
            let ljlc1 = lj + lc + 1;
            let d2 = lilc1 * ljlc1;
            let im = nfi * dlc;
            let mq = dlc * ljlc1;
            // Single (ic=0,jc=0) tuple: di = nfi, c_off = 0.
            let di = nfi;
            let mut gctr = vec![0.0_f64; nfi * nfj];
            let mut buf = vec![0.0_f64; nfi * dlc * ljlc1];
            for i in 0..=li {
                for j in 0..=lj {
                    let a = &prad[(i + j) * d2..(i + j) * d2 + d2];
                    let b_off = i * nfi * dlc * lilc1;
                    let b = &angi[b_off..b_off + lilc1 * im];
                    for col in 0..im {
                        for row in 0..ljlc1 {
                            let mut s = 0.0;
                            for kk in 0..lilc1 {
                                s += a[kk * ljlc1 + row] * b[col * lilc1 + kk];
                            }
                            buf[col * ljlc1 + row] = s;
                        }
                    }
                    let bj_off = j * nfj * dlc * ljlc1;
                    let bj = &angj[bj_off..bj_off + mq * nfj];
                    for col in 0..nfj {
                        for row in 0..nfi {
                            let mut s = 0.0;
                            for kk in 0..mq {
                                s += buf[row * mq + kk] * bj[col * mq + kk];
                            }
                            gctr[col * di + row] += common_fac * s;
                        }
                    }
                }
            }
            gctr
        }

        fn random_type2_inputs(
            li: usize,
            lj: usize,
            lc: usize,
            seed: u64,
        ) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
            let mut rng = Lcg::new(seed);
            let nfi = ncart(li as u8);
            let nfj = ncart(lj as u8);
            let dlc = lc * 2 + 1;
            let lilc1 = li + lc + 1;
            let ljlc1 = lj + lc + 1;
            let d2 = lilc1 * ljlc1;
            let d3 = (li + lj + 1) * d2;
            let prad: Vec<f64> = (0..d3).map(|_| rng.next_f64()).collect();
            let angi: Vec<f64> = (0..(li + 1) * nfi * dlc * lilc1)
                .map(|_| rng.next_f64())
                .collect();
            let angj: Vec<f64> = (0..(lj + 1) * nfj * dlc * ljlc1)
                .map(|_| rng.next_f64())
                .collect();
            // common_fac: a positive-ish nonzero scalar in (0, 2).
            let common_fac = rng.next_f64() + 1.0;
            (prad, angi, angj, common_fac)
        }

        /// f64 device-vs-host equivalence for the Type-2 splice: max-abs-diff MUST
        /// be exactly 0.0 (identical f64 op order ⇒ byte-identity).
        fn assert_type2_f64_byte_identity(li: usize, lj: usize, lc: usize, seed: u64) {
            let (prad, angi, angj, common_fac) = random_type2_inputs(li, lj, lc, seed);
            let host = host_type2_splice(li, lj, lc, &prad, &angi, &angj, common_fac);
            let dev = run_ecp_type2_angular_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client(),
                li as u32,
                lj as u32,
                lc as u32,
                &prad,
                &angi,
                &angj,
                common_fac,
            );
            assert_eq!(
                host.len(),
                dev.len(),
                "len mismatch li={li} lj={lj} lc={lc}"
            );
            let mut max_diff = 0.0_f64;
            let mut any_nonzero = false;
            for (&h, &d) in host.iter().zip(dev.iter()) {
                if h.abs() > 1e-18 {
                    any_nonzero = true;
                }
                max_diff = max_diff.max((h - d).abs());
            }
            assert_eq!(
                max_diff, 0.0,
                "ecp type2 device/host f64 max-abs-diff must be 0.0 (li={li} lj={lj} lc={lc}), got {max_diff:e}"
            );
            assert!(
                any_nonzero,
                "host type2 reference all zeros (li={li} lj={lj} lc={lc})"
            );
        }

        #[test]
        fn ecp_type2_angular_device_matches_host_f64() {
            // (li,lj) up to (2,2) × lc in {0,1,2}.
            for (li, lj) in [
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (0, 2),
                (2, 1),
                (1, 2),
                (2, 2),
            ] {
                for lc in [0usize, 1, 2] {
                    let seed = 0xA5A5_5A5A_C3C3_3C3C ^ ((li * 31 + lj * 7 + lc) as u64);
                    assert_type2_f64_byte_identity(li, lj, lc, seed);
                }
            }
        }

        /// Generic-F: running the SAME Type-2 kernel at F=f32 on CpuRuntime
        /// reproduces the f64 result cast to f32 within f32 eps. Proves
        /// `ecp_type2_angular_kernel` is genuinely generic over `F: Float`.
        #[test]
        fn ecp_type2_angular_device_generic_f32_within_eps() {
            let (li, lj, lc) = (2usize, 1usize, 1usize);
            let (prad, angi, angj, common_fac) =
                random_type2_inputs(li, lj, lc, 0xF00DCAFE_DEADBEEF);
            let host = host_type2_splice(li, lj, lc, &prad, &angi, &angj, common_fac);

            let nfi = ncart(li as u8);
            let nfj = ncart(lj as u8);
            let out_len = nfi * nfj;

            let client = cpu_client();
            let prad_f32: Vec<f32> = prad.iter().map(|&v| v as f32).collect();
            let angi_f32: Vec<f32> = angi.iter().map(|&v| v as f32).collect();
            let angj_f32: Vec<f32> = angj.iter().map(|&v| v as f32).collect();
            let prad_h = client.create_from_slice(f32::as_bytes(&prad_f32));
            let angi_h = client.create_from_slice(f32::as_bytes(&angi_f32));
            let angj_h = client.create_from_slice(f32::as_bytes(&angj_f32));
            let out_zero = vec![0.0_f32; out_len];
            let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

            ecp_type2_angular_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
                &client,
                crate::plane::single_cube_count(),
                crate::plane::standard_plane_cube_dim(),
                unsafe { ArrayArg::from_raw_parts(prad_h, prad_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(angi_h, angi_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(angj_h, angj_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                common_fac as f32,
                li as u32,
                lj as u32,
                lc as u32,
                nfi as u32,
                nfj as u32,
            );
            let raw = client.read_one_unchecked(out_h);
            let dev_f32 = &f32::from_bytes(&raw)[0..out_len];

            for (&h, &d) in host.iter().zip(dev_f32.iter()) {
                let diff = (h as f32 - d).abs();
                let thr = 1e-3_f32 + 1e-3_f32 * (h.abs() as f32);
                assert!(
                    diff <= thr,
                    "type2 f32 device result not within eps: host={h:e} dev={d:e} diff={diff:e}"
                );
            }
        }
    }
}
