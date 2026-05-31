//! F12/STG/YP kernel entry points.
//!
//! Implements the 10 F12 integral entry points (5 STG variants + 5 YP variants).
//! STG and YP share the 2e VRR/HRR/contraction pipeline but use different weight
//! post-processing after `stg_roots_host` (per g2e_f12.c lines 290-296 vs 197-200).
//!
//! nroots formula for F12: `ceil((li_ceil + lj_ceil + lk_ceil + ll_ceil + 1) / 2)`
//! equivalent to `(L_tot + 3) / 2` in integer arithmetic, matching libcint g2e_f12.c line 75.
//!
//! Derivative variant angular momentum increments from cint2e_f12.c ng arrays:
//!   base:    ng = [0, 0, 0, 0, ...] → ncomp = 1
//!   ip1:     ng = [1, 0, 0, 0, ...] → ncomp = 3
//!   ipip1:   ng = [2, 0, 0, 0, ...] → ncomp = 9
//!   ipvip1:  ng = [1, 1, 0, 0, ...] → ncomp = 9
//!   ip1ip2:  ng = [1, 0, 1, 0, ...] → ncomp = 9

use crate::backend::ResolvedBackend;
use crate::math::pdata::compute_pdata_host;
use crate::math::stg::stg_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_4d;
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats, validator::validate_f12_env_params};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
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

/// Derivative variant angular momentum increments and component count.
///
/// From cint2e_f12.c ng arrays (IINC, JINC, KINC, LINC, ncomp):
///   base:    [0, 0, 0, 0, 1]
///   ip1:     [1, 0, 0, 0, 3]
///   ipip1:   [2, 0, 0, 0, 9]
///   ipvip1:  [1, 1, 0, 0, 9]
///   ip1ip2:  [1, 0, 1, 0, 9]
#[derive(Clone, Copy, Debug)]
struct F12Variant {
    i_inc: usize,
    j_inc: usize,
    k_inc: usize,
    l_inc: usize,
    ncomp: usize,
}

const F12_BASE: F12Variant = F12Variant { i_inc: 0, j_inc: 0, k_inc: 0, l_inc: 0, ncomp: 1 };
const F12_IP1: F12Variant = F12Variant { i_inc: 1, j_inc: 0, k_inc: 0, l_inc: 0, ncomp: 3 };
const F12_IPIP1: F12Variant = F12Variant { i_inc: 2, j_inc: 0, k_inc: 0, l_inc: 0, ncomp: 9 };
const F12_IPVIP1: F12Variant = F12Variant { i_inc: 1, j_inc: 1, k_inc: 0, l_inc: 0, ncomp: 9 };
const F12_IP1IP2: F12Variant = F12Variant { i_inc: 1, j_inc: 0, k_inc: 1, l_inc: 0, ncomp: 9 };

/// Stride/layout metadata for F12 (identical structure to two_electron's TwoEShape).
///
/// `pub(crate)` so the plain-Coulomb gradient launchers (`two_electron.rs::int2e_ip1`,
/// `center_3c2e.rs::int3c2e_ip1`) can construct one from their own `build_2e_shape`
/// result and feed it to [`gout_ip1`] / [`nabla1i_2e`]. The first-derivative math
/// in those two functions is F12-free (it implements the standard libcint
/// `∂/∂A χ_l = -2α·χ_{l+1} + l·χ_{l-1}` / `CINTnabla1i_2e`); only the G-tensor
/// fill (`fill_g_tensor_f12` via `stg_roots_host`) is F12-specific. Sharing these
/// symbols lets the plain-Coulomb gradients reuse the exact verbatim derivative
/// math (Phase 21 D-04) instead of re-deriving it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct F12Shape {
    pub(crate) nroots: usize,
    pub(crate) nmax: usize,
    pub(crate) mmax: usize,
    pub(crate) li: usize,
    pub(crate) lj: usize,
    pub(crate) lk: usize,
    pub(crate) ll: usize,
    pub(crate) ibase: bool,
    pub(crate) kbase: bool,
    pub(crate) di: usize,
    pub(crate) dk: usize,
    pub(crate) dl: usize,
    pub(crate) dj: usize,
    pub(crate) g2d_ijmax: usize,
    pub(crate) g2d_klmax: usize,
    pub(crate) g_size: usize,
}

/// Build F12 shape using ceiling nroots formula from g2e_f12.c line 75:
///   `nroots = ceil((L_tot + 1) / 2)` where L_tot = li_ceil + lj_ceil + lk_ceil + ll_ceil
///   in integer arithmetic: `(L_tot + 3) / 2`
fn build_f12_shape(li: usize, lj: usize, lk: usize, ll: usize) -> F12Shape {
    let l_tot = li + lj + lk + ll;
    // F12 uses ceil((L_tot + 1) / 2) = (L_tot + 3) / 2 for integer nroots.
    // This differs from the plain 2e formula (L_tot / 2 + 1) at odd L_tot values.
    let nroots = (l_tot + 3) / 2;
    let nmax = li + lj;
    let mmax = lk + ll;

    let ibase = li > lj;
    let kbase = lk > ll;

    let (dli, dlj) = if ibase {
        (li + lj + 1, lj + 1)
    } else {
        (li + 1, li + lj + 1)
    };
    let (dlk, dll) = if kbase {
        (lk + ll + 1, ll + 1)
    } else {
        (lk + 1, lk + ll + 1)
    };

    let di = nroots;
    let dk = nroots * dli;
    let dl = nroots * dli * dlk;
    let dj = nroots * dli * dlk * dll;
    let g_size = nroots * dli * dlk * dll * dlj;

    let g2d_ijmax = if ibase { di } else { dj };
    let g2d_klmax = if kbase { dk } else { dl };

    F12Shape {
        nroots,
        nmax,
        mmax,
        li,
        lj,
        lk,
        ll,
        ibase,
        kbase,
        di,
        dk,
        dl,
        dj,
        g2d_ijmax,
        g2d_klmax,
        g_size,
    }
}

#[inline]
fn vrr_fill_axis_f12(
    g_axis: &mut [f64],
    root: usize,
    nmax: usize,
    mmax: usize,
    dn: usize,
    dm: usize,
    c00: f64,
    c0p: f64,
    b10: f64,
    b01: f64,
    b00: f64,
) {
    if nmax > 0 {
        let mut s0 = g_axis[root];
        let mut s1 = c00 * s0;
        g_axis[root + dn] = s1;
        for n in 1..nmax {
            let s2 = c00 * s1 + n as f64 * b10 * s0;
            g_axis[root + (n + 1) * dn] = s2;
            s0 = s1;
            s1 = s2;
        }
    }

    if mmax > 0 {
        let mut s0 = g_axis[root];
        let mut s1 = c0p * s0;
        g_axis[root + dm] = s1;
        for m in 1..mmax {
            let s2 = c0p * s1 + m as f64 * b01 * s0;
            g_axis[root + (m + 1) * dm] = s2;
            s0 = s1;
            s1 = s2;
        }

        if nmax > 0 {
            let mut s0n = g_axis[root + dn];
            let mut s1n = c0p * s0n + b00 * g_axis[root];
            g_axis[root + dn + dm] = s1n;
            for m in 1..mmax {
                let s2n = c0p * s1n + m as f64 * b01 * s0n + b00 * g_axis[root + m * dm];
                g_axis[root + dn + (m + 1) * dm] = s2n;
                s0n = s1n;
                s1n = s2n;
            }
        }
    }

    if nmax > 0 {
        for m in 1..=mmax {
            let off = m * dm;
            let j = off + root;
            let mut s0 = g_axis[j];
            let mut s1 = g_axis[j + dn];
            for n in 1..nmax {
                let s2 = c00 * s1 + n as f64 * b10 * s0 + m as f64 * b00 * g_axis[j + n * dn - dm];
                g_axis[j + (n + 1) * dn] = s2;
                s0 = s1;
                s1 = s2;
            }
        }
    }
}

fn hrr_lj2d_4d_f12(g: &mut [f64], shape: F12Shape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.li == 0 && shape.lk == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rirj[axis];

        for i in 1..=shape.li {
            for j in 0..=(shape.nmax - i) {
                for l in 0..=shape.mmax {
                    let ptr = j * shape.dj + l * shape.dl + i * shape.di;
                    for r in 0..nroots {
                        let idx = ptr + r;
                        g[off + idx] =
                            rx * g[off + idx - shape.di] + g[off + idx - shape.di + shape.dj];
                    }
                }
            }
        }

        let rx = rkrl[axis];
        for j in 0..=shape.lj {
            for k in 1..=shape.lk {
                for l in 0..=(shape.mmax - k) {
                    let ptr = j * shape.dj + l * shape.dl + k * shape.dk;
                    for n in 0..shape.dk {
                        let idx = ptr + n;
                        g[off + idx] =
                            rx * g[off + idx - shape.dk] + g[off + idx - shape.dk + shape.dl];
                    }
                }
            }
        }
    }
}

fn hrr_kj2d_4d_f12(g: &mut [f64], shape: F12Shape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.li == 0 && shape.ll == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rirj[axis];

        for i in 1..=shape.li {
            for j in 0..=(shape.nmax - i) {
                for k in 0..=shape.mmax {
                    let ptr = j * shape.dj + k * shape.dk + i * shape.di;
                    for r in 0..nroots {
                        let idx = ptr + r;
                        g[off + idx] =
                            rx * g[off + idx - shape.di] + g[off + idx - shape.di + shape.dj];
                    }
                }
            }
        }

        let rx = rkrl[axis];
        for j in 0..=shape.lj {
            for l in 1..=shape.ll {
                for k in 0..=(shape.mmax - l) {
                    let ptr = j * shape.dj + l * shape.dl + k * shape.dk;
                    for n in 0..shape.dk {
                        let idx = ptr + n;
                        g[off + idx] =
                            rx * g[off + idx - shape.dl] + g[off + idx - shape.dl + shape.dk];
                    }
                }
            }
        }
    }
}

fn hrr_il2d_4d_f12(g: &mut [f64], shape: F12Shape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.lj == 0 && shape.lk == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rkrl[axis];

        for k in 1..=shape.lk {
            for l in 0..=(shape.mmax - k) {
                for i in 0..=shape.nmax {
                    let ptr = l * shape.dl + k * shape.dk + i * shape.di;
                    for r in 0..nroots {
                        let idx = ptr + r;
                        g[off + idx] =
                            rx * g[off + idx - shape.dk] + g[off + idx - shape.dk + shape.dl];
                    }
                }
            }
        }

        let rx = rirj[axis];
        for j in 1..=shape.lj {
            for l in 0..=shape.ll {
                for k in 0..=shape.lk {
                    let ptr = j * shape.dj + l * shape.dl + k * shape.dk;
                    for i in 0..=(shape.nmax - j) {
                        let base = ptr + i * shape.di;
                        for r in 0..nroots {
                            let idx = base + r;
                            g[off + idx] =
                                rx * g[off + idx - shape.dj] + g[off + idx - shape.dj + shape.di];
                        }
                    }
                }
            }
        }
    }
}

fn hrr_ik2d_4d_f12(g: &mut [f64], shape: F12Shape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.lj == 0 && shape.ll == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rkrl[axis];

        for l in 1..=shape.ll {
            for k in 0..=(shape.mmax - l) {
                for i in 0..=shape.nmax {
                    let ptr = l * shape.dl + k * shape.dk + i * shape.di;
                    for r in 0..nroots {
                        let idx = ptr + r;
                        g[off + idx] =
                            rx * g[off + idx - shape.dl] + g[off + idx - shape.dl + shape.dk];
                    }
                }
            }
        }

        let rx = rirj[axis];
        for j in 1..=shape.lj {
            for l in 0..=shape.ll {
                for k in 0..=shape.lk {
                    let ptr = j * shape.dj + l * shape.dl + k * shape.dk;
                    for i in 0..=(shape.nmax - j) {
                        let base = ptr + i * shape.di;
                        for r in 0..nroots {
                            let idx = base + r;
                            g[off + idx] =
                                rx * g[off + idx - shape.dj] + g[off + idx - shape.dj + shape.di];
                        }
                    }
                }
            }
        }
    }
}

/// Fill the 3D [gx|gy|gz] tensor for one F12 primitive quartet.
///
/// Differs from the plain 2e version in the root computation step:
/// - Calls `stg_roots_host` instead of `rys_roots_host`.
/// - Applies STG or YP specific weight post-processing (per g2e_f12.c).
/// - Everything after weight post-processing is identical to the plain 2e VRR.
fn fill_g_tensor_f12(
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
    ri: &[f64; 3],
    rj: &[f64; 3],
    rk: &[f64; 3],
    rl: &[f64; 3],
    shape: F12Shape,
    fac_env: f64,
    zeta: f64,
    is_stg: bool,
) -> Vec<f64> {
    let aij = ai + aj;
    let akl = ak + al;

    let rij = [
        (ai * ri[0] + aj * rj[0]) / aij,
        (ai * ri[1] + aj * rj[1]) / aij,
        (ai * ri[2] + aj * rj[2]) / aij,
    ];
    let rkl = [
        (ak * rk[0] + al * rl[0]) / akl,
        (ak * rk[1] + al * rl[1]) / akl,
        (ak * rk[2] + al * rl[2]) / akl,
    ];

    let xij_kl = rij[0] - rkl[0];
    let yij_kl = rij[1] - rkl[1];
    let zij_kl = rij[2] - rkl[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

    let a1 = aij * akl;
    let a0 = a1 / (aij + akl);

    // F12 fac1 formula from g2e_f12.c: fac1 = envs->fac[0] / (sqrt(aij+akl) * a1)
    // envs->fac[0] is computed from pdata, which already includes the Gaussian product exponent
    // factor. Here we match compute_pdata_host output where pdata.fac includes exp factor.
    // The plain 2e formula is: fac1 = sqrt(a0 / (a1 * a1 * a1)) * fac_env
    // which equals: fac_env * (a0/(a1^3))^0.5 = fac_env / (sqrt(aij+akl) * a1)
    // since a0 = a1/(aij+akl) => a0/a1^3 = 1/((aij+akl)*a1^2) => sqrt(a0/a1^3) = 1/(sqrt(aij+akl)*a1)
    let fac1 = fac_env / ((aij + akl).sqrt() * a1);

    // ua = zeta^2 / (4*a0) per g2e_f12.c line 276
    let ua = 0.25 * zeta * zeta / a0;
    let ta = a0 * rr;

    let (u_roots, mut w_weights) = stg_roots_host::<f64>(shape.nroots, ta, ua);

    if is_stg {
        // STG weight post-processing (g2e_f12.c lines 292-297):
        //   w[irys] *= (1 - u[irys]) * 2*ua/zeta;
        //   u[irys] = u[irys] / (1 - u[irys]);
        let ua2 = 2.0 * ua / zeta;
        let mut u_transformed = u_roots.clone();
        for irys in 0..shape.nroots {
            w_weights[irys] *= (1.0 - u_roots[irys]) * ua2;
            u_transformed[irys] = u_roots[irys] / (1.0 - u_roots[irys]);
        }
        let u_roots = u_transformed;

        // Apply fac1 scaling to weights
        for w in &mut w_weights {
            *w *= fac1;
        }

        fill_g_tensor_inner(shape, &u_roots, &w_weights, ri, rj, rk, rl, rij, rkl, xij_kl, yij_kl, zij_kl, a0, a1, aij, akl)
    } else {
        // YP weight post-processing (g2e_f12.c lines 197-200):
        //   w[irys] *= u[irys];
        //   u[irys] = u[irys] / (1 - u[irys]);
        let mut u_transformed = u_roots.clone();
        for irys in 0..shape.nroots {
            w_weights[irys] *= u_roots[irys];
            u_transformed[irys] = u_roots[irys] / (1.0 - u_roots[irys]);
        }
        let u_roots = u_transformed;

        // Apply fac1 scaling to weights
        for w in &mut w_weights {
            *w *= fac1;
        }

        fill_g_tensor_inner(shape, &u_roots, &w_weights, ri, rj, rk, rl, rij, rkl, xij_kl, yij_kl, zij_kl, a0, a1, aij, akl)
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_g_tensor_inner(
    shape: F12Shape,
    u_roots: &[f64],
    w_weights: &[f64],
    ri: &[f64; 3],
    rj: &[f64; 3],
    rk: &[f64; 3],
    rl: &[f64; 3],
    rij: [f64; 3],
    rkl: [f64; 3],
    xij_kl: f64,
    yij_kl: f64,
    zij_kl: f64,
    a0: f64,
    a1: f64,
    aij: f64,
    akl: f64,
) -> Vec<f64> {
    let (rx_in_rijrx, rirj) = if shape.ibase {
        (*ri, [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]])
    } else {
        (*rj, [rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]])
    };
    let (rx_in_rklrx, rkrl) = if shape.kbase {
        (*rk, [rk[0] - rl[0], rk[1] - rl[1], rk[2] - rl[2]])
    } else {
        (*rl, [rl[0] - rk[0], rl[1] - rk[1], rl[2] - rk[2]])
    };

    let rijrx = [
        rij[0] - rx_in_rijrx[0],
        rij[1] - rx_in_rijrx[1],
        rij[2] - rx_in_rijrx[2],
    ];
    let rklrx = [
        rkl[0] - rx_in_rklrx[0],
        rkl[1] - rx_in_rklrx[1],
        rkl[2] - rx_in_rklrx[2],
    ];

    let mut g = vec![0.0_f64; 3 * shape.g_size];
    let gy_off = shape.g_size;
    let gz_off = 2 * shape.g_size;

    for irys in 0..shape.nroots {
        g[irys] = 1.0;
        g[gy_off + irys] = 1.0;
        g[gz_off + irys] = w_weights[irys];
    }

    for irys in 0..shape.nroots {
        // After post-processing, u_roots[irys] = t/(1-t) where t is the original Rys root
        // This is the "u2" variable in libcint (u2 = a0 * u[irys])
        let u2 = a0 * u_roots[irys];
        let tmp4 = 0.5 / (u2 * (aij + akl) + a1);
        let tmp5 = u2 * tmp4;
        let tmp1 = 2.0 * tmp5;
        let tmp2 = tmp1 * akl;
        let tmp3 = tmp1 * aij;

        let b00 = tmp5;
        let b10 = tmp5 + tmp4 * akl;
        let b01 = tmp5 + tmp4 * aij;

        let c00 = [
            rijrx[0] - tmp2 * xij_kl,
            rijrx[1] - tmp2 * yij_kl,
            rijrx[2] - tmp2 * zij_kl,
        ];
        let c0p = [
            rklrx[0] + tmp3 * xij_kl,
            rklrx[1] + tmp3 * yij_kl,
            rklrx[2] + tmp3 * zij_kl,
        ];

        let (gx, rest) = g.split_at_mut(shape.g_size);
        let (gy, gz) = rest.split_at_mut(shape.g_size);
        vrr_fill_axis_f12(gx, irys, shape.nmax, shape.mmax, shape.g2d_ijmax, shape.g2d_klmax, c00[0], c0p[0], b10, b01, b00);
        vrr_fill_axis_f12(gy, irys, shape.nmax, shape.mmax, shape.g2d_ijmax, shape.g2d_klmax, c00[1], c0p[1], b10, b01, b00);
        vrr_fill_axis_f12(gz, irys, shape.nmax, shape.mmax, shape.g2d_ijmax, shape.g2d_klmax, c00[2], c0p[2], b10, b01, b00);
    }

    // HRR transfer
    if shape.kbase {
        if shape.ibase {
            hrr_ik2d_4d_f12(&mut g, shape, rirj, rkrl);
        } else {
            hrr_kj2d_4d_f12(&mut g, shape, rirj, rkrl);
        }
    } else if shape.ibase {
        hrr_il2d_4d_f12(&mut g, shape, rirj, rkrl);
    } else {
        hrr_lj2d_4d_f12(&mut g, shape, rirj, rkrl);
    }

    g
}

// ─────────────────────────────────────────────────────────────────────────────
// Nabla derivative operators (ported from libcint g2e.c CINTnabla1{i,j,k}_2e)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the `\nabla_i` operator to the G tensor.
///
/// Corresponds to `CINTnabla1i_2e` in libcint/g2e.c.
///
/// Both `f` and `g` have layout `[gx | gy | gz]` with each axis of size `g_size`.
/// The operator reads up to index `i = li` in g (which requires `li+1` levels to be
/// present in the G tensor), matching the headroom built by using `li_ceil = li + 1`.
///
/// Formula (per axis):
///   f[n @ i=0] = -2*ai * g[n+di]
///   f[n @ i>=1] = i * g[n-di] + (-2*ai) * g[n+di]
pub(crate) fn nabla1i_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, ai: f64, shape: &F12Shape) {
    let ai2 = -2.0 * ai;
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        for j in 0..=lj {
            for l in 0..=ll {
                for k in 0..=lk {
                    // i=0: f[n] = ai2 * g[n+di]
                    let ptr = dj * j + dl * l + dk * k;
                    for n in ptr..ptr + nroots {
                        f[off + n] = ai2 * g[off + n + di];
                    }
                    // i>=1: f[n] = i*g[n-di] + ai2*g[n+di]
                    for i in 1..=li {
                        let ptr = dj * j + dl * l + dk * k + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = i as f64 * g[off + n - di] + ai2 * g[off + n + di];
                        }
                    }
                }
            }
        }
    }
}

/// Apply the `\nabla_j` operator to the G tensor.
///
/// Corresponds to `CINTnabla1j_2e` in libcint/g2e.c.
///
/// Formula (per axis):
///   f[n @ j=0] = -2*aj * g[n+dj]
///   f[n @ j>=1] = j * g[n-dj] + (-2*aj) * g[n+dj]
///
/// `pub(crate)` (Phase 23 plan 01): shared with sibling kernel launchers
/// (`two_electron.rs`, `center_2c2e.rs`, `center_3c2e.rs`) for ket/remaining-center
/// derivative families (int2e_ip2, int2c2e_ip1/ip2). The math is F12-free.
pub(crate) fn nabla1j_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, aj: f64, shape: &F12Shape) {
    let aj2 = -2.0 * aj;
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        // j=0: outer loop over l,k,i
        for l in 0..=ll {
            for k in 0..=lk {
                let base = dl * l + dk * k;
                for i in 0..=li {
                    let ptr = base + di * i;
                    for n in ptr..ptr + nroots {
                        f[off + n] = aj2 * g[off + n + dj];
                    }
                }
            }
        }
        // j>=1
        for j in 1..=lj {
            for l in 0..=ll {
                for k in 0..=lk {
                    let base = dj * j + dl * l + dk * k;
                    for i in 0..=li {
                        let ptr = base + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = j as f64 * g[off + n - dj] + aj2 * g[off + n + dj];
                        }
                    }
                }
            }
        }
    }
}

/// Apply the `\nabla_k` operator to the G tensor.
///
/// Corresponds to `CINTnabla1k_2e` in libcint/g2e.c.
///
/// Formula (per axis):
///   f[n @ k=0] = -2*ak * g[n+dk]
///   f[n @ k>=1] = k * g[n-dk] + (-2*ak) * g[n+dk]
///
/// `pub(crate)` (Phase 23 plan 01): shared with sibling kernel launchers
/// (`two_electron.rs`, `center_2c2e.rs`, `center_3c2e.rs`) for remaining-center
/// derivative families. The math is F12-free.
pub(crate) fn nabla1k_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, ak: f64, shape: &F12Shape) {
    let ak2 = -2.0 * ak;
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        for j in 0..=lj {
            for l in 0..=ll {
                // k=0: all i
                let base = dj * j + dl * l;
                for i in 0..=li {
                    let ptr = base + di * i;
                    for n in ptr..ptr + nroots {
                        f[off + n] = ak2 * g[off + n + dk];
                    }
                }
                // k>=1
                for k in 1..=lk {
                    let base = dj * j + dl * l + dk * k;
                    for i in 0..=li {
                        let ptr = base + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = k as f64 * g[off + n - dk] + ak2 * g[off + n + dk];
                        }
                    }
                }
            }
        }
    }
}

/// Apply the `\nabla_l` operator to the G tensor.
///
/// Corresponds to `CINTnabla1l_2e` in libcint/g2e.c (the `G2E_D_L` macro).
///
/// `pub(crate)` (Phase 23 plan 01): added because cintx's 3c2e g-tensor builder maps
/// the real auxiliary `k` center into the 2e `ll` slot (`build_2e_shape(li+1, lj, 0, lk)`,
/// phantom 2e `lk`=0), so int3c2e_ip2's auxiliary derivative must nabla the `ll` slot —
/// `nabla1k_2e` would touch the phantom slot (RESEARCH Pitfall 2). The G2E_D_L recurrence
/// is the structural mirror of `nabla1k_2e`, operating on the `ll` loop bound and `dl`
/// stride; `nabla1l_breit` (breit.rs:1206) is the in-tree reference (Don't Hand-Roll).
///
/// Formula (per axis):
///   f[n @ l=0] = -2*al * g[n+dl]
///   f[n @ l>=1] = l * g[n-dl] + (-2*al) * g[n+dl]
pub(crate) fn nabla1l_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, al: f64, shape: &F12Shape) {
    let al2 = -2.0 * al;
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        for j in 0..=lj {
            // l=0: all k, i
            for k in 0..=lk {
                let base = dj * j + dk * k;
                for i in 0..=li {
                    let ptr = base + di * i;
                    for n in ptr..ptr + nroots {
                        f[off + n] = al2 * g[off + n + dl];
                    }
                }
            }
            // l>=1
            for l in 1..=ll {
                for k in 0..=lk {
                    let base = dj * j + dl * l + dk * k;
                    for i in 0..=li {
                        let ptr = base + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = l as f64 * g[off + n - dl] + al2 * g[off + n + dl];
                        }
                    }
                }
            }
        }
    }
}

/// Apply the `r0i` position operator (multiply by `r - ri` in the i-index) to the
/// G tensor. Corresponds to `CINTx1i_2e` / the `G2E_R0I` macro in libcint/g2e.h.
///
/// Formula (per axis), reading the i+1-elevated index from the headroom tensor:
///   f[n @ i] = g[n + di] + ri[axis] * g[n]
///
/// Used by the Phase-26 GIAO-02 2e families (`int2e_g1`/`ig1`/`gg1`/`g1g2`). The
/// final post-HRR G tensor carries the true i-center in the i-index, so the
/// position multiply uses the actual `ri` coordinate exactly as libcint does
/// (gout runs after the HRR transfer). `ri` is the i-center coordinate `[x,y,z]`.
pub(crate) fn r0i_2e(
    f: &mut [f64],
    g: &[f64],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ri: &[f64; 3],
    shape: &F12Shape,
) {
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        let ri_a = ri[axis];
        for j in 0..=lj {
            for l in 0..=ll {
                for k in 0..=lk {
                    for i in 0..=li {
                        let ptr = dj * j + dl * l + dk * k + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = g[off + n + di] + ri_a * g[off + n];
                        }
                    }
                }
            }
        }
    }
}

/// Apply the `r0k` position operator (multiply by `r - rk` in the k-index) to the
/// G tensor. Corresponds to `CINTx1k_2e` / the `G2E_R0K` macro in libcint/g2e.h.
///
/// Formula (per axis): f[n @ k] = g[n + dk] + rk[axis] * g[n].
///
/// Used by the Phase-26 GIAO-02 `int2e_g1g2` family (gauge factor on electron 2).
pub(crate) fn r0k_2e(
    f: &mut [f64],
    g: &[f64],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    rk: &[f64; 3],
    shape: &F12Shape,
) {
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        let rk_a = rk[axis];
        for j in 0..=lj {
            for l in 0..=ll {
                for k in 0..=lk {
                    for i in 0..=li {
                        let ptr = dj * j + dl * l + dk * k + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = g[off + n + dk] + rk_a * g[off + n];
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 26 GIAO-02 (D-16): spin-free 2e GIAO gout contractions.
//
// Each gout is transcribed VERBATIM from libcint autocode (intor4.c:1255 g1,
// intor2.c:19 ig1, intor2.c:148 gg1, intor2.c:283 g1g2). The `c[]` gauge factor
// uses rirj = ri - rj (and rkrl = rk - rl for g1g2); the derivative tensors are
// built with the r0i_2e / r0k_2e position operators above. The C gout writes
// gout[n*rank + comp] DIRECTLY (no column-major reorder — unlike ipip1).
// ─────────────────────────────────────────────────────────────────────────────

/// gout for `int2e_g1` (rank 3) — intor4.c:1255 `CINTgout2e_int2e_g1`.
///
/// g1 = R0I(g0, i_l+0); c = ri - rj;
///   gout[0] = + c1*s2 - c2*s1, gout[1] = + c2*s0 - c0*s2, gout[2] = + c0*s1 - c1*s0
/// where s0 = g1x*g0y*g0z, s1 = g0x*g1y*g0z, s2 = g0x*g0y*g1z.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_g1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ri: &[f64; 3],
    rj: &[f64; 3],
) -> Vec<f64> {
    gout_g1_signed(g, shape, li, lj, lk, ll, ri, rj, 1.0)
}

/// gout for `int2e_ig1` (rank 3) — intor2.c:19 `CINTgout2e_int2e_ig1`.
///
/// Identical s[] triple products as g1, but the gout combos are sign-flipped:
///   gout[0] = - c1*s2 + c2*s1, gout[1] = - c2*s0 + c0*s2, gout[2] = - c0*s1 + c1*s0
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_ig1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ri: &[f64; 3],
    rj: &[f64; 3],
) -> Vec<f64> {
    gout_g1_signed(g, shape, li, lj, lk, ll, ri, rj, -1.0)
}

/// Shared rank-3 GIAO cross-product body for g1 (`sign=+1`) and ig1 (`sign=-1`).
#[allow(clippy::too_many_arguments)]
fn gout_g1_signed(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ri: &[f64; 3],
    rj: &[f64; 3],
    sign: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let c = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // g1 = R0I(g0, i_l+0); built with i_l+1 headroom in the G tensor.
    let mut g1 = vec![0.0_f64; 3 * g_size];
    r0i_2e(&mut g1, g, li, lj, lk, ll, ri, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 3 * nf];
    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 3];
                    for r in 0..shape.nroots {
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        s[0] += g1x * g0y * g0z;
                        s[1] += g0x * g1y * g0z;
                        s[2] += g0x * g0y * g1z;
                    }
                    // g1:  gout = + (c1*s2 - c2*s1, c2*s0 - c0*s2, c0*s1 - c1*s0)
                    // ig1: gout = - (the same)  → `sign` carries the overall flip.
                    out[n * 3 + 0] = sign * (c[1] * s[2] - c[2] * s[1]);
                    out[n * 3 + 1] = sign * (c[2] * s[0] - c[0] * s[2]);
                    out[n * 3 + 2] = sign * (c[0] * s[1] - c[1] * s[0]);
                    n += 1;
                }
            }
        }
    }
    out
}

/// gout for `int2e_gg1` (rank 9) — intor2.c:148 `CINTgout2e_int2e_gg1`.
///
/// 2nd-order gauge tensor on electron 1. c = (ri-rj)⊗(ri-rj) (9 components).
/// g1 = R0I(g0, i_l+1); g2 = R0I(g0, i_l+0); g3 = R0I(g1, i_l+0).
/// s[] and the 9-component gout combos transcribed verbatim from the source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_gg1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ri: &[f64; 3],
    rj: &[f64; 3],
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let c = [
        rirj[0] * rirj[0], rirj[0] * rirj[1], rirj[0] * rirj[2],
        rirj[1] * rirj[0], rirj[1] * rirj[1], rirj[1] * rirj[2],
        rirj[2] * rirj[0], rirj[2] * rirj[1], rirj[2] * rirj[2],
    ];

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // g1 = R0I at li+1 (elevated); g2 = R0I at li+0; g3 = R0I(g1) at li+0.
    r0i_2e(&mut g1, g, li + 1, lj, lk, ll, ri, shape);
    r0i_2e(&mut g2, g, li, lj, lk, ll, ri, shape);
    r0i_2e(&mut g3, &g1, li, lj, lk, ll, ri, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 9 * nf];
    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for r in 0..shape.nroots {
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // Verbatim from intor2.c:192-200 (direct gout[n*9+..] write).
                    out[n * 9 + 0] = -c[4] * s[8] + 2.0 * c[5] * s[7] - c[8] * s[4];
                    out[n * 9 + 1] = -c[7] * s[2] + c[1] * s[8] + c[8] * s[1] - c[2] * s[7];
                    out[n * 9 + 2] = -c[1] * s[5] + c[4] * s[2] + c[2] * s[4] - c[5] * s[1];
                    out[n * 9 + 3] = -c[5] * s[6] + c[8] * s[3] + c[3] * s[8] - c[6] * s[5];
                    out[n * 9 + 4] = -c[8] * s[0] + 2.0 * c[6] * s[2] - c[0] * s[8];
                    out[n * 9 + 5] = -c[2] * s[3] + c[5] * s[0] + c[0] * s[5] - c[3] * s[2];
                    out[n * 9 + 6] = -c[3] * s[7] + c[6] * s[4] + c[4] * s[6] - c[7] * s[3];
                    out[n * 9 + 7] = -c[6] * s[1] + c[0] * s[7] + c[7] * s[0] - c[1] * s[6];
                    out[n * 9 + 8] = -c[0] * s[4] + 2.0 * c[1] * s[3] - c[4] * s[0];
                    n += 1;
                }
            }
        }
    }
    out
}

/// gout for `int2e_g1g2` (rank 9) — intor2.c:283 `CINTgout2e_int2e_g1g2` (D-16).
///
/// Gauge factor on BOTH electrons: c = (ri-rj)⊗(rk-rl) (9 components).
/// g1 = R0K(g0, i_l+1); g2 = R0I(g0, i_l+0); g3 = R0I(g1, i_l+0).
/// (Note: g1 raises i by +1 then R0K; g3 = R0I(g1).)
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_g1g2(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ri: &[f64; 3],
    rj: &[f64; 3],
    rk: &[f64; 3],
    rl: &[f64; 3],
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let rkrl = [rk[0] - rl[0], rk[1] - rl[1], rk[2] - rl[2]];
    let c = [
        rirj[0] * rkrl[0], rirj[0] * rkrl[1], rirj[0] * rkrl[2],
        rirj[1] * rkrl[0], rirj[1] * rkrl[1], rirj[1] * rkrl[2],
        rirj[2] * rkrl[0], rirj[2] * rkrl[1], rirj[2] * rkrl[2],
    ];

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // intor2.c:310-312: g1=R0K(g0,i_l+1); g2=R0I(g0,i_l+0); g3=R0I(g1,i_l+0).
    r0k_2e(&mut g1, g, li + 1, lj, lk, ll, rk, shape);
    r0i_2e(&mut g2, g, li, lj, lk, ll, ri, shape);
    r0i_2e(&mut g3, &g1, li, lj, lk, ll, ri, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 9 * nf];
    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for r in 0..shape.nroots {
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // Verbatim from intor2.c:331-339 (direct gout[n*9+..] write).
                    out[n * 9 + 0] = c[4] * s[8] - c[7] * s[5] - c[5] * s[7] + c[8] * s[4];
                    out[n * 9 + 1] = c[5] * s[6] - c[8] * s[3] - c[3] * s[8] + c[6] * s[5];
                    out[n * 9 + 2] = c[3] * s[7] - c[6] * s[4] - c[4] * s[6] + c[7] * s[3];
                    out[n * 9 + 3] = c[7] * s[2] - c[1] * s[8] - c[8] * s[1] + c[2] * s[7];
                    out[n * 9 + 4] = c[8] * s[0] - c[2] * s[6] - c[6] * s[2] + c[0] * s[8];
                    out[n * 9 + 5] = c[6] * s[1] - c[0] * s[7] - c[7] * s[0] + c[1] * s[6];
                    out[n * 9 + 6] = c[1] * s[5] - c[4] * s[2] - c[2] * s[4] + c[5] * s[1];
                    out[n * 9 + 7] = c[2] * s[3] - c[5] * s[0] - c[0] * s[5] + c[3] * s[2];
                    out[n * 9 + 8] = c[0] * s[4] - c[3] * s[1] - c[1] * s[3] + c[4] * s[0];
                    n += 1;
                }
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-variant gout contraction functions
//
// Each function takes the G tensor (per-primitive, layout [gx|gy|gz]) and produces
// ncomp * nf Cartesian values where nf = ncart(li)*ncart(lj)*ncart(lk)*ncart(ll).
// The BASE angular momenta (li, lj, lk, ll) are used for loop bounds; the G tensor
// was built with ceiling angular momenta providing the nabla headroom.
//
// These match the libcint autocode patterns in autocode/grad2.c and autocode/hess.c.
// ─────────────────────────────────────────────────────────────────────────────

/// Which center the single-side `\nabla` acts on inside [`gout_ipn`].
///
/// Phase 23 plan 01: the s[0..2] mixing body of the single-side contraction is
/// IDENTICAL for every center — only the nabla operator and its exponent change.
/// `I` reproduces the original `gout_ip1` (int2e_ip1, bra-i); `J`/`K`/`L` cover the
/// ket / remaining-center / auxiliary-center derivative families (int2e_ip2,
/// int2c2e_ip1/ip2, int3c2e_ip2 — the last via the 2e `ll` slot, RESEARCH Pitfall 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Nabla1Center {
    I,
    J,
    K,
    L,
}

/// Apply the matching single-side `\nabla` operator at base angular momenta.
#[inline]
fn apply_nabla1_center(
    center: Nabla1Center,
    g1: &mut [f64],
    g: &[f64],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    exponent: f64,
    shape: &F12Shape,
) {
    match center {
        Nabla1Center::I => nabla1i_2e(g1, g, li, lj, lk, ll, exponent, shape),
        Nabla1Center::J => nabla1j_2e(g1, g, li, lj, lk, ll, exponent, shape),
        Nabla1Center::K => nabla1k_2e(g1, g, li, lj, lk, ll, exponent, shape),
        Nabla1Center::L => nabla1l_2e(g1, g, li, lj, lk, ll, exponent, shape),
    }
}

/// Parameterized single-side gradient contraction (ncomp=3).
///
/// Phase 23 plan 01: generalizes the original `gout_ip1` s[0..2] mixing body over
/// which center the `\nabla` acts on (`center`) and its exponent (`exponent`). The
/// mixing math (`s[0]=g1x·g0y·g0z`, `s[1]=g0x·g1y·g0z`, `s[2]=g0x·g0y·g1z`) is the
/// single source of truth; [`gout_ip1`] is now a thin `Nabla1Center::I` wrapper so
/// the int2e_ip1 (bra-i) path stays byte-identical (Phase 21 D-04, no regression).
///
/// Output layout: gout[n*3+comp] for comp in 0..3 (x, y, z), n walking the
/// `[ll][lk][lj][li]` Cartesian product (outer→inner), matching `gout_ip1`.
pub(crate) fn gout_ipn(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    center: Nabla1Center,
    exponent: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let mut g1 = vec![0.0_f64; 3 * g_size];
    // nabla at base angular momenta; g was built with the matching ceiling headroom.
    apply_nabla1_center(center, &mut g1, g, li, lj, lk, ll, exponent, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 3 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 3];
                    for irys in 0..shape.nroots {
                        // s[0] = g1x * g0y * g0z
                        s[0] += g1[gx_off + ix_base + irys] * g[gy_off + iy_base + irys] * g[gz_off + iz_base + irys];
                        // s[1] = g0x * g1y * g0z
                        s[1] += g[gx_off + ix_base + irys] * g1[gy_off + iy_base + irys] * g[gz_off + iz_base + irys];
                        // s[2] = g0x * g0y * g1z
                        s[2] += g[gx_off + ix_base + irys] * g[gy_off + iy_base + irys] * g1[gz_off + iz_base + irys];
                    }
                    out[n * 3 + 0] = s[0];
                    out[n * 3 + 1] = s[1];
                    out[n * 3 + 2] = s[2];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute gout for the ip1 variant (ncomp=3): `\nabla_i` on electron 1.
///
/// Matches `CINTgout2e_int2e_ip1` in autocode/grad2.c.
/// Output layout: gout[n*3+comp] for comp in 0..3 (x, y, z).
///
/// `pub(crate)` (Phase 21 D-04): shared verbatim with `two_electron.rs::int2e_ip1`
/// and `center_3c2e.rs::int3c2e_ip1` — the plain-Coulomb gradients feed it the
/// plain `fill_g_tensor_2e` G-tensor (rys roots) instead of the F12 stg-roots
/// tensor; the contraction math here is identical for both.
///
/// Phase 23 plan 01: now a thin `Nabla1Center::I` wrapper over [`gout_ipn`]; the
/// signature and numeric output are byte-identical to the pre-change implementation.
pub(crate) fn gout_ip1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
) -> Vec<f64> {
    gout_ipn(g, shape, li, lj, lk, ll, Nabla1Center::I, ai)
}

/// Compute gout for the ipip1 variant (ncomp=9): `\nabla_i \nabla_i` on electron 1.
///
/// Matches `CINTgout2e_int2e_ipip1` in autocode/hess.c.
/// CRITICAL: output has column-major reordering of the 3×3 Hessian.
/// Output layout: gout[n*9+{0,1,2,3,4,5,6,7,8}] = {s0,s3,s6,s1,s4,s7,s2,s5,s8}
///
/// `pub(crate)` so the plain-Coulomb 2e Hessian launcher
/// (`two_electron.rs::launch_two_electron_hess2e`) can reuse it verbatim with a
/// plain Coulomb G-tensor (Phase 25 HESS-02 / D-07).
pub(crate) fn gout_ipip1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // g1 = nabla1i at li+1 (elevated)
    nabla1i_2e(&mut g1, g, li + 1, lj, lk, ll, ai, shape);
    // g2 = nabla1i at li+0 (base)
    nabla1i_2e(&mut g2, g, li, lj, lk, ll, ai, shape);
    // g3 = nabla1i(g1) at li+0
    nabla1i_2e(&mut g3, &g1, li, lj, lk, ll, ai, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 9 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for irys in 0..shape.nroots {
                        let r = irys;
                        // g0 = original G tensor; g1 = nabla(g0,li+1); g2 = nabla(g0,li+0); g3 = nabla(g1,li+0)
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        // Matches libcint CINTgout2e_int2e_ipip1 exactly
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // Column-major reordering: gout[n*9+{0..8}] = {s0,s3,s6,s1,s4,s7,s2,s5,s8}
                    out[n * 9 + 0] = s[0];
                    out[n * 9 + 1] = s[3];
                    out[n * 9 + 2] = s[6];
                    out[n * 9 + 3] = s[1];
                    out[n * 9 + 4] = s[4];
                    out[n * 9 + 5] = s[7];
                    out[n * 9 + 6] = s[2];
                    out[n * 9 + 7] = s[5];
                    out[n * 9 + 8] = s[8];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute gout for the ket-side `ipip2` variant (ncomp=9): `\nabla_k \nabla_k`
/// on electron 2 (the real auxiliary center), applied to the 2e `ll` slot.
///
/// Matches `CINTgout2e_int3c2e_ipip2` in autocode/int3c2e.c — IDENTICAL `s[]`
/// triple product and column-major 3×3 reorder as `gout_ipip1`, but the second
/// derivative is taken on the KET (`G2E_D_K` in the C source). In cintx's 3c2e
/// layout the real aux k is mapped to the 2e `ll` slot (`int3c2e_ip2` Pitfall 2),
/// so the ket double-nabla is applied via `nabla1l_2e` with `ll+1`/`ll+0`
/// headroom (the G-tensor is built with `build_2e_shape(li, lj, 0, lk+2)`).
///
/// `pub(crate)` for the multi-center 3c2e Hessian launcher (Phase 25 HESS-03).
pub(crate) fn gout_ipip2_l(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    al: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // G2E_D_K(g1, g0, ..., ll+1); G2E_D_K(g2, g0, ..., ll+0); G2E_D_K(g3, g1, ..., ll+0).
    // The real aux k lives in the 2e `ll` slot, so nabla1l_2e is the ket derivative.
    nabla1l_2e(&mut g1, g, li, lj, lk, ll + 1, al, shape);
    nabla1l_2e(&mut g2, g, li, lj, lk, ll, al, shape);
    nabla1l_2e(&mut g3, &g1, li, lj, lk, ll, al, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 9 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for irys in 0..shape.nroots {
                        let r = irys;
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        // Matches libcint CINTgout2e_int3c2e_ipip2 exactly.
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // Column-major reorder (same as ipip1): {s0,s3,s6,s1,s4,s7,s2,s5,s8}.
                    out[n * 9 + 0] = s[0];
                    out[n * 9 + 1] = s[3];
                    out[n * 9 + 2] = s[6];
                    out[n * 9 + 3] = s[1];
                    out[n * 9 + 4] = s[4];
                    out[n * 9 + 5] = s[7];
                    out[n * 9 + 6] = s[2];
                    out[n * 9 + 7] = s[5];
                    out[n * 9 + 8] = s[8];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute gout for the ipvip1 variant (ncomp=9): `\nabla_i \nabla_j` on electron 1.
///
/// Matches `CINTgout2e_int2e_ipvip1` in autocode/hess.c.
/// No column-major reordering (unlike ipip1).
///
/// `pub(crate)` for the plain-Coulomb 2e Hessian launcher (Phase 25 HESS-02).
pub(crate) fn gout_ipvip1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    aj: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // g1 = nabla1j at (li+1, lj+0): j-derivative at elevated i
    nabla1j_2e(&mut g1, g, li + 1, lj, lk, ll, aj, shape);
    // g2 = nabla1i at (li+0): i-derivative at base
    nabla1i_2e(&mut g2, g, li, lj, lk, ll, ai, shape);
    // g3 = nabla1i(g1) at (li+0): mixed i,j second derivative
    nabla1i_2e(&mut g3, &g1, li, lj, lk, ll, ai, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 9 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for irys in 0..shape.nroots {
                        let r = irys;
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // No reordering for ipvip1
                    out[n * 9 + 0] = s[0];
                    out[n * 9 + 1] = s[1];
                    out[n * 9 + 2] = s[2];
                    out[n * 9 + 3] = s[3];
                    out[n * 9 + 4] = s[4];
                    out[n * 9 + 5] = s[5];
                    out[n * 9 + 6] = s[6];
                    out[n * 9 + 7] = s[7];
                    out[n * 9 + 8] = s[8];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute the σ·p₁ G-tensor for `int2e_spsp1` (component_rank=1 → 4 cart blocks).
///
/// Matches `CINTgout2e_int2e_spsp1` in autocode/intor4.c:19-58 VERBATIM. The σ·p₁
/// operator `(σ·∇_i)(σ·∇_j)` on electron 1 (bra i, ket j) uses the IDENTICAL `s[0..8]`
/// triple-product tensor as `gout_ipvip1` (∇_i∇_j: `g1=nabla1j(g,li+1)`,
/// `g2=nabla1i(g)`, `g3=nabla1i(g1)`), then folds it into the four σ-tensor cart
/// blocks the `c2s_si_2e1` transform consumes (gc_x, gc_y, gc_z, gc_1):
///
/// ```text
/// gc_x = + s[5] - s[7]   (σ_x)
/// gc_y = + s[6] - s[2]   (σ_y)
/// gc_z = + s[1] - s[3]   (σ_z)
/// gc_1 = + s[0] + s[4] + s[8]   (scalar)
/// ```
///
/// Returns interleaved `out[n*4 + comp]` (`comp` in {x,y,z,1}); `n` walks `[cl,ck,cj,ci]`
/// i-fastest. The hess2e-style launcher TRANSPOSES this into the four contiguous
/// component-leading cart blocks `cart[comp*block + n]` that `cart_to_spinor_si_2e1`
/// reads as gc_x/gc_y/gc_z/gc_1.
///
/// Headroom matches `Hess2eKind::Ipvip1` = `(i_inc, j_inc, k_inc) = (1, 1, 0)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_spsp1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    aj: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // Identical derivative setup to gout_ipvip1 (= libcint spsp1 g1/g2/g3):
    //   g1 = G2E_D_J(g0, i_l+1)  → nabla1j at (li+1, lj+0)
    //   g2 = G2E_D_I(g0)         → nabla1i at (li+0)
    //   g3 = G2E_D_I(g1)         → nabla1i(g1) (mixed i,j second derivative)
    nabla1j_2e(&mut g1, g, li + 1, lj, lk, ll, aj, shape);
    nabla1i_2e(&mut g2, g, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g3, &g1, li, lj, lk, ll, ai, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 4 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for irys in 0..shape.nroots {
                        let r = irys;
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        // Identical s[] triple products to CINTgout2e_int2e_spsp1.
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // σ·p₁ fold (intor4.c:49-52). gc order = (x, y, z, scalar).
                    out[n * 4 + 0] = s[5] - s[7];
                    out[n * 4 + 1] = s[6] - s[2];
                    out[n * 4 + 2] = s[1] - s[3];
                    out[n * 4 + 3] = s[0] + s[4] + s[8];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute the σ·r₁ G-tensor for `int2e_srsr1` (component_rank=1 → 4 cart blocks).
///
/// Matches `CINTgout2e_int2e_srsr1` (autocode/intor4.c:283-322) VERBATIM. The σ·r₁
/// operator on electron 1 uses the IDENTICAL `s[0..8]` triple-product tensor and the
/// IDENTICAL σ fold as [`gout_spsp1`] — the ONLY difference is the derivative setup:
/// libcint `G2E_R_J`/`G2E_R_I` are pure G-tensor POINTER SHIFTS by `g_stride_j`/`g_stride_i`
/// (g2e.h:104-107: `f = g + envs->g_stride_*`), NOT nabla derivatives. In cintx terms
/// `g_stride_i == shape.di`, `g_stride_j == shape.dj`, so:
/// ```text
/// g1 = G2E_R_J(g0)        → g1[idx] = g0[idx + dj]   (index-shift in j)
/// g2 = G2E_R_I(g0)        → g2[idx] = g0[idx + di]   (index-shift in i)
/// g3 = G2E_R_I(g1)        → g3[idx] = g0[idx + di + dj]
/// ```
/// The G-tensor MUST be built with `(li+1, lj+1)` headroom (same as spsp1) so the
/// `+di`/`+dj` reads land in valid storage. Returns interleaved `out[n*4 + comp]`
/// (`comp` in {x,y,z,1}); `n` walks `[cl,ck,cj,ci]` i-fastest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_srsr1(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;
    let di = shape.di;
    let dj = shape.dj;

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 4 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for r in 0..shape.nroots {
                        // R-shifts: g1=+dj, g2=+di, g3=+di+dj (per-axis bases).
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g[gx_off + ix_base + dj + r];
                        let g1y = g[gy_off + iy_base + dj + r];
                        let g1z = g[gz_off + iz_base + dj + r];
                        let g2x = g[gx_off + ix_base + di + r];
                        let g2y = g[gy_off + iy_base + di + r];
                        let g2z = g[gz_off + iz_base + di + r];
                        let g3x = g[gx_off + ix_base + di + dj + r];
                        let g3y = g[gy_off + iy_base + di + dj + r];
                        let g3z = g[gz_off + iz_base + di + dj + r];
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    out[n * 4 + 0] = s[5] - s[7];
                    out[n * 4 + 1] = s[6] - s[2];
                    out[n * 4 + 2] = s[1] - s[3];
                    out[n * 4 + 3] = s[0] + s[4] + s[8];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Shared rank-16 fold for the 2-sided σ families (`spsp1spsp2`, `srsr1srsr2`).
///
/// `g0..g15` are the 16 derivative/shift compositions (built by the caller per
/// family). For each cartesian quad index `n` the 81-term `s[]` triple-product
/// tensor and the 16-component σ⊗σ fold are transcribed VERBATIM from
/// `CINTgout2e_int2e_spsp1spsp2` (intor4.c:127-250) — identical to
/// `srsr1srsr2` (intor4.c:391-514).
///
/// gout layout: `out[n*16 + comp]`, `comp = e2_sigma*4 + e1_sigma` with e1∈{x,y,z,1}
/// fastest, e2∈{x,y,z,1} slowest — exactly what `c2s_si_2e1` (consumes the inner 4)
/// then `c2s_si_2e2` (consumes the outer 4) expect. `n` walks `[cl,ck,cj,ci]`.
#[allow(clippy::too_many_arguments)]
fn fold_2sided_sigma16(
    g_blocks: &[&[f64]; 16],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;
    let gb = g_blocks;

    let mut out = vec![0.0_f64; 16 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 81];
                    for r in 0..shape.nroots {
                        // Per-axis accessors into g0..g15 (x, y, z slices).
                        let gx = |m: usize| gb[m][gx_off + ix_base + r];
                        let gy = |m: usize| gb[m][gy_off + iy_base + r];
                        let gz = |m: usize| gb[m][gz_off + iz_base + r];
                        s[0] += gx(15) * gy(0) * gz(0);
                        s[1] += gx(14) * gy(1) * gz(0);
                        s[2] += gx(14) * gy(0) * gz(1);
                        s[3] += gx(13) * gy(2) * gz(0);
                        s[4] += gx(12) * gy(3) * gz(0);
                        s[5] += gx(12) * gy(2) * gz(1);
                        s[6] += gx(13) * gy(0) * gz(2);
                        s[7] += gx(12) * gy(1) * gz(2);
                        s[8] += gx(12) * gy(0) * gz(3);
                        s[9] += gx(11) * gy(4) * gz(0);
                        s[10] += gx(10) * gy(5) * gz(0);
                        s[11] += gx(10) * gy(4) * gz(1);
                        s[12] += gx(9) * gy(6) * gz(0);
                        s[13] += gx(8) * gy(7) * gz(0);
                        s[14] += gx(8) * gy(6) * gz(1);
                        s[15] += gx(9) * gy(4) * gz(2);
                        s[16] += gx(8) * gy(5) * gz(2);
                        s[17] += gx(8) * gy(4) * gz(3);
                        s[18] += gx(11) * gy(0) * gz(4);
                        s[19] += gx(10) * gy(1) * gz(4);
                        s[20] += gx(10) * gy(0) * gz(5);
                        s[21] += gx(9) * gy(2) * gz(4);
                        s[22] += gx(8) * gy(3) * gz(4);
                        s[23] += gx(8) * gy(2) * gz(5);
                        s[24] += gx(9) * gy(0) * gz(6);
                        s[25] += gx(8) * gy(1) * gz(6);
                        s[26] += gx(8) * gy(0) * gz(7);
                        s[27] += gx(7) * gy(8) * gz(0);
                        s[28] += gx(6) * gy(9) * gz(0);
                        s[29] += gx(6) * gy(8) * gz(1);
                        s[30] += gx(5) * gy(10) * gz(0);
                        s[31] += gx(4) * gy(11) * gz(0);
                        s[32] += gx(4) * gy(10) * gz(1);
                        s[33] += gx(5) * gy(8) * gz(2);
                        s[34] += gx(4) * gy(9) * gz(2);
                        s[35] += gx(4) * gy(8) * gz(3);
                        s[36] += gx(3) * gy(12) * gz(0);
                        s[37] += gx(2) * gy(13) * gz(0);
                        s[38] += gx(2) * gy(12) * gz(1);
                        s[39] += gx(1) * gy(14) * gz(0);
                        s[40] += gx(0) * gy(15) * gz(0);
                        s[41] += gx(0) * gy(14) * gz(1);
                        s[42] += gx(1) * gy(12) * gz(2);
                        s[43] += gx(0) * gy(13) * gz(2);
                        s[44] += gx(0) * gy(12) * gz(3);
                        s[45] += gx(3) * gy(8) * gz(4);
                        s[46] += gx(2) * gy(9) * gz(4);
                        s[47] += gx(2) * gy(8) * gz(5);
                        s[48] += gx(1) * gy(10) * gz(4);
                        s[49] += gx(0) * gy(11) * gz(4);
                        s[50] += gx(0) * gy(10) * gz(5);
                        s[51] += gx(1) * gy(8) * gz(6);
                        s[52] += gx(0) * gy(9) * gz(6);
                        s[53] += gx(0) * gy(8) * gz(7);
                        s[54] += gx(7) * gy(0) * gz(8);
                        s[55] += gx(6) * gy(1) * gz(8);
                        s[56] += gx(6) * gy(0) * gz(9);
                        s[57] += gx(5) * gy(2) * gz(8);
                        s[58] += gx(4) * gy(3) * gz(8);
                        s[59] += gx(4) * gy(2) * gz(9);
                        s[60] += gx(5) * gy(0) * gz(10);
                        s[61] += gx(4) * gy(1) * gz(10);
                        s[62] += gx(4) * gy(0) * gz(11);
                        s[63] += gx(3) * gy(4) * gz(8);
                        s[64] += gx(2) * gy(5) * gz(8);
                        s[65] += gx(2) * gy(4) * gz(9);
                        s[66] += gx(1) * gy(6) * gz(8);
                        s[67] += gx(0) * gy(7) * gz(8);
                        s[68] += gx(0) * gy(6) * gz(9);
                        s[69] += gx(1) * gy(4) * gz(10);
                        s[70] += gx(0) * gy(5) * gz(10);
                        s[71] += gx(0) * gy(4) * gz(11);
                        s[72] += gx(3) * gy(0) * gz(12);
                        s[73] += gx(2) * gy(1) * gz(12);
                        s[74] += gx(2) * gy(0) * gz(13);
                        s[75] += gx(1) * gy(2) * gz(12);
                        s[76] += gx(0) * gy(3) * gz(12);
                        s[77] += gx(0) * gy(2) * gz(13);
                        s[78] += gx(1) * gy(0) * gz(14);
                        s[79] += gx(0) * gy(1) * gz(14);
                        s[80] += gx(0) * gy(0) * gz(15);
                    }
                    // 16-component σ⊗σ fold (intor4.c:216-232) — comp = e2*4 + e1.
                    out[n * 16 + 0] = s[50] - s[68] - s[52] + s[70];
                    out[n * 16 + 1] = s[59] - s[23] - s[61] + s[25];
                    out[n * 16 + 2] = s[14] - s[32] - s[16] + s[34];
                    out[n * 16 + 3] = s[5] + s[41] + s[77] - s[7] - s[43] - s[79];
                    out[n * 16 + 4] = s[51] - s[69] - s[47] + s[65];
                    out[n * 16 + 5] = s[60] - s[24] - s[56] + s[20];
                    out[n * 16 + 6] = s[15] - s[33] - s[11] + s[29];
                    out[n * 16 + 7] = s[6] + s[42] + s[78] - s[2] - s[38] - s[74];
                    out[n * 16 + 8] = s[46] - s[64] - s[48] + s[66];
                    out[n * 16 + 9] = s[55] - s[19] - s[57] + s[21];
                    out[n * 16 + 10] = s[10] - s[28] - s[12] + s[30];
                    out[n * 16 + 11] = s[1] + s[37] + s[73] - s[3] - s[39] - s[75];
                    out[n * 16 + 12] = s[45] - s[63] + s[49] - s[67] + s[53] - s[71];
                    out[n * 16 + 13] = s[54] - s[18] + s[58] - s[22] + s[62] - s[26];
                    out[n * 16 + 14] = s[9] - s[27] + s[13] - s[31] + s[17] - s[35];
                    out[n * 16 + 15] = s[0] + s[36] + s[72] + s[4] + s[40] + s[76] + s[8] + s[44] + s[80];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute the (σ·p₁)(σ·p₂) G-tensor for `int2e_spsp1spsp2` (rank-16 gout).
///
/// Matches `CINTgout2e_int2e_spsp1spsp2` (intor4.c:91-250) VERBATIM. Builds the 16
/// nabla compositions `g1..g15` (the D_L/D_K/D_J/D_I cascade, lines 112-126) on a
/// `(li+1, lj+1, lk+1, ll+0)` headroom G-tensor, then folds via [`fold_2sided_sigma16`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_spsp1spsp2(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
) -> Vec<f64> {
    let g_size = shape.g_size;
    let mut gv: Vec<Vec<f64>> = (0..16).map(|_| vec![0.0_f64; 3 * g_size]).collect();
    // g0 = identity (the base G-tensor).
    gv[0].copy_from_slice(&g[..3 * g_size]);
    // Build g1..g15 via the D_L/D_K/D_J/D_I cascade (intor4.c:112-126). Each is a
    // nabla on the lower-index intermediate; the elevated `(i+1,j+1,k+1)` headroom
    // lets each composition read the raised indices.
    //
    // (dst, src, op, l-bounds, exponent) — op selects the nabla axis. The cascade
    // is strictly increasing in dst and reads only already-built lower-index src,
    // so we compute each into a scratch then move it into gv[dst].
    #[derive(Clone, Copy)]
    enum Op { L, K, J, I }
    let plan: [(usize, usize, Op, usize, usize, usize, usize, f64); 15] = [
        (1, 0, Op::L, li + 1, lj + 1, lk + 1, ll, al),
        (2, 0, Op::K, li + 1, lj + 1, lk, ll, ak),
        (3, 1, Op::K, li + 1, lj + 1, lk, ll, ak),
        (4, 0, Op::J, li + 1, lj, lk, ll, aj),
        (5, 1, Op::J, li + 1, lj, lk, ll, aj),
        (6, 2, Op::J, li + 1, lj, lk, ll, aj),
        (7, 3, Op::J, li + 1, lj, lk, ll, aj),
        (8, 0, Op::I, li, lj, lk, ll, ai),
        (9, 1, Op::I, li, lj, lk, ll, ai),
        (10, 2, Op::I, li, lj, lk, ll, ai),
        (11, 3, Op::I, li, lj, lk, ll, ai),
        (12, 4, Op::I, li, lj, lk, ll, ai),
        (13, 5, Op::I, li, lj, lk, ll, ai),
        (14, 6, Op::I, li, lj, lk, ll, ai),
        (15, 7, Op::I, li, lj, lk, ll, ai),
    ];
    let mut scratch = vec![0.0_f64; 3 * g_size];
    for &(dst, src, op, pli, plj, plk, pll, a) in &plan {
        for v in scratch.iter_mut() {
            *v = 0.0;
        }
        match op {
            Op::L => nabla1l_2e(&mut scratch, &gv[src], pli, plj, plk, pll, a, shape),
            Op::K => nabla1k_2e(&mut scratch, &gv[src], pli, plj, plk, pll, a, shape),
            Op::J => nabla1j_2e(&mut scratch, &gv[src], pli, plj, plk, pll, a, shape),
            Op::I => nabla1i_2e(&mut scratch, &gv[src], pli, plj, plk, pll, a, shape),
        }
        gv[dst].copy_from_slice(&scratch);
    }

    let blocks: [&[f64]; 16] = std::array::from_fn(|m| gv[m].as_slice());
    fold_2sided_sigma16(&blocks, shape, li, lj, lk, ll)
}

/// Compute the (σ·r₁)(σ·r₂) G-tensor for `int2e_srsr1srsr2` (rank-16 gout).
///
/// Matches `CINTgout2e_int2e_srsr1srsr2` (intor4.c:355-514) VERBATIM. The `g1..g15`
/// compositions are pure G-tensor POINTER SHIFTS (`G2E_R_L/R_K/R_J/R_I` = `+g_stride_*`,
/// g2e.h:104-107), NOT nablas — the 81-term `s[]` and 16-component fold are byte-identical
/// to `spsp1spsp2`. Built on a `(li+1, lj+1, lk+1, ll+0)` headroom G-tensor so the
/// shifted reads land in valid storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_srsr1srsr2(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
) -> Vec<f64> {
    let g_size = shape.g_size;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;
    // R-shift offsets for the D_L/D_K/D_J/D_I cascade structure (intor4.c:376-390):
    //   g1 = R_L(g0)        = +dl
    //   g2 = R_K(g0)        = +dk
    //   g3 = R_K(g1)        = +dk+dl
    //   g4 = R_J(g0)        = +dj
    //   g5 = R_J(g1)        = +dj+dl
    //   g6 = R_J(g2)        = +dj+dk
    //   g7 = R_J(g3)        = +dj+dk+dl
    //   g8 = R_I(g0)        = +di
    //   g9 = R_I(g1)        = +di+dl
    //   g10 = R_I(g2)       = +di+dk
    //   g11 = R_I(g3)       = +di+dk+dl
    //   g12 = R_I(g4)       = +di+dj
    //   g13 = R_I(g5)       = +di+dj+dl
    //   g14 = R_I(g6)       = +di+dj+dk
    //   g15 = R_I(g7)       = +di+dj+dk+dl
    let shifts: [usize; 16] = [
        0,
        dl,
        dk,
        dk + dl,
        dj,
        dj + dl,
        dj + dk,
        dj + dk + dl,
        di,
        di + dl,
        di + dk,
        di + dk + dl,
        di + dj,
        di + dj + dl,
        di + dj + dk,
        di + dj + dk + dl,
    ];
    // Materialize each shifted block (3 axes × g_size) so fold_2sided_sigma16 can read
    // the same per-axis ix_base+r offsets as the nabla path.
    let mut gv: Vec<Vec<f64>> = Vec::with_capacity(16);
    for &sh in &shifts {
        let mut blk = vec![0.0_f64; 3 * g_size];
        for axis in 0..3 {
            let off = axis * g_size;
            for idx in 0..g_size {
                let src = off + idx + sh;
                blk[off + idx] = if src < off + g_size { g[src] } else { 0.0 };
            }
        }
        gv.push(blk);
    }
    let blocks: [&[f64]; 16] = std::array::from_fn(|m| gv[m].as_slice());
    fold_2sided_sigma16(&blocks, shape, li, lj, lk, ll)
}

/// Compute gout for the ip1ip2 variant (ncomp=9): `\nabla_i` on e1 and `\nabla_k` on e2.
///
/// Matches `CINTgout2e_int2e_ip1ip2` in autocode/hess.c.
/// No column-major reordering.
///
/// `pub(crate)` for the plain-Coulomb 2e Hessian launcher (Phase 25 HESS-02).
pub(crate) fn gout_ip1ip2(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    ak: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    // g1 = nabla1k at (li+1, lj+0, lk+0): k-derivative at elevated i
    nabla1k_2e(&mut g1, g, li + 1, lj, lk, ll, ak, shape);
    // g2 = nabla1i at (li+0): i-derivative at base
    nabla1i_2e(&mut g2, g, li, lj, lk, ll, ai, shape);
    // g3 = nabla1i(g1) at (li+0): mixed i,k second derivative
    nabla1i_2e(&mut g3, &g1, li, lj, lk, ll, ai, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 9 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 9];
                    for irys in 0..shape.nroots {
                        let r = irys;
                        let g0x = g[gx_off + ix_base + r];
                        let g0y = g[gy_off + iy_base + r];
                        let g0z = g[gz_off + iz_base + r];
                        let g1x = g1[gx_off + ix_base + r];
                        let g1y = g1[gy_off + iy_base + r];
                        let g1z = g1[gz_off + iz_base + r];
                        let g2x = g2[gx_off + ix_base + r];
                        let g2y = g2[gy_off + iy_base + r];
                        let g2z = g2[gz_off + iz_base + r];
                        let g3x = g3[gx_off + ix_base + r];
                        let g3y = g3[gy_off + iy_base + r];
                        let g3z = g3[gz_off + iz_base + r];
                        s[0] += g3x * g0y * g0z;
                        s[1] += g2x * g1y * g0z;
                        s[2] += g2x * g0y * g1z;
                        s[3] += g1x * g2y * g0z;
                        s[4] += g0x * g3y * g0z;
                        s[5] += g0x * g2y * g1z;
                        s[6] += g1x * g0y * g2z;
                        s[7] += g0x * g1y * g2z;
                        s[8] += g0x * g0y * g3z;
                    }
                    // No reordering for ip1ip2
                    out[n * 9 + 0] = s[0];
                    out[n * 9 + 1] = s[1];
                    out[n * 9 + 2] = s[2];
                    out[n * 9 + 3] = s[3];
                    out[n * 9 + 4] = s[4];
                    out[n * 9 + 5] = s[5];
                    out[n * 9 + 6] = s[6];
                    out[n * 9 + 7] = s[7];
                    out[n * 9 + 8] = s[8];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Compute gout for the ipip1ipip2 variant (ncomp=81, 4th-order 2e):
/// `\nabla_i \nabla_i` on electron 1 AND `\nabla_k \nabla_k` on electron 2.
///
/// Matches `CINTgout2e_int2e_ipip1ipip2` in autocode/hess.c VERBATIM — the 16
/// G2E_D_K/G2E_D_I composition (g0..g15), the 81-term `s[]` triple product, and
/// the column-major 9×9 reorder permutation are copied 1:1 from the C source
/// (D-09/D-10: rank-81, transpose-sensitive, must emit all 81 components).
///
/// G-tensor headroom (D-09): built with `build_2e_shape(li+2, lj, lk+2, ll)` so
/// the D_K composition reads up to k_l+2 and the D_I composition reads up to i_l+2.
///
/// `pub(crate)` for the plain-Coulomb 2e Hessian launcher (Phase 25 HESS-02).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gout_ipip1ipip2(
    g: &[f64],
    shape: &F12Shape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    ak: f64,
) -> Vec<f64> {
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);
    let nf = nfi * nfj * nfk * nfl;
    let g_size = shape.g_size;

    // 16 work buffers (g0 is the input G tensor; g1..g15 are derived).
    // Composition copied VERBATIM from hess.c CINTgout2e_int2e_ipip1ipip2:
    //   G2E_D_K(g1,  g0, i+2, j, k+1, l);  G2E_D_K(g2,  g0, i+2, j, k,   l);
    //   G2E_D_K(g3,  g1, i+2, j, k,   l);  G2E_D_I(g4,  g0, i+1, j, k,   l);
    //   G2E_D_I(g5,  g1, i+1, j, k, l);    G2E_D_I(g6,  g2, i+1, j, k, l);
    //   G2E_D_I(g7,  g3, i+1, j, k, l);    G2E_D_I(g8,  g0, i+0, j, k, l);
    //   G2E_D_I(g9,  g1, i+0, j, k, l);    G2E_D_I(g10, g2, i+0, j, k, l);
    //   G2E_D_I(g11, g3, i+0, j, k, l);    G2E_D_I(g12, g4, i+0, j, k, l);
    //   G2E_D_I(g13, g5, i+0, j, k, l);    G2E_D_I(g14, g6, i+0, j, k, l);
    //   G2E_D_I(g15, g7, i+0, j, k, l);
    let mut g1 = vec![0.0_f64; 3 * g_size];
    let mut g2 = vec![0.0_f64; 3 * g_size];
    let mut g3 = vec![0.0_f64; 3 * g_size];
    let mut g4 = vec![0.0_f64; 3 * g_size];
    let mut g5 = vec![0.0_f64; 3 * g_size];
    let mut g6 = vec![0.0_f64; 3 * g_size];
    let mut g7 = vec![0.0_f64; 3 * g_size];
    let mut g8 = vec![0.0_f64; 3 * g_size];
    let mut g9 = vec![0.0_f64; 3 * g_size];
    let mut g10 = vec![0.0_f64; 3 * g_size];
    let mut g11 = vec![0.0_f64; 3 * g_size];
    let mut g12 = vec![0.0_f64; 3 * g_size];
    let mut g13 = vec![0.0_f64; 3 * g_size];
    let mut g14 = vec![0.0_f64; 3 * g_size];
    let mut g15 = vec![0.0_f64; 3 * g_size];

    nabla1k_2e(&mut g1, g, li + 2, lj, lk + 1, ll, ak, shape);
    nabla1k_2e(&mut g2, g, li + 2, lj, lk, ll, ak, shape);
    nabla1k_2e(&mut g3, &g1, li + 2, lj, lk, ll, ak, shape);
    nabla1i_2e(&mut g4, g, li + 1, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g5, &g1, li + 1, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g6, &g2, li + 1, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g7, &g3, li + 1, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g8, g, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g9, &g1, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g10, &g2, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g11, &g3, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g12, &g4, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g13, &g5, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g14, &g6, li, lj, lk, ll, ai, shape);
    nabla1i_2e(&mut g15, &g7, li, lj, lk, ll, ai, shape);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; 81 * nf];

    let mut n = 0usize;
    for &(lx, ly, lz) in &cl_comps {
        for &(kx, ky, kz) in &ck_comps {
            for &(jx, jy, jz) in &cj_comps {
                for &(ix, iy, iz) in &ci_comps {
                    let ix_base = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let iy_base = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let iz_base = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = [0.0_f64; 81];
                    for irys in 0..shape.nroots {
                        let r = irys;
                        // closures to fetch the x/y/z component of buffer b at the
                        // (ix_base/iy_base/iz_base + r) index.
                        macro_rules! gx { ($b:expr) => { $b[gx_off + ix_base + r] }; }
                        macro_rules! gy { ($b:expr) => { $b[gy_off + iy_base + r] }; }
                        macro_rules! gz { ($b:expr) => { $b[gz_off + iz_base + r] }; }
                        // s[] triple products — copied VERBATIM from hess.c.
                        s[0] += gx!(g15) * gy!(g) * gz!(g);
                        s[1] += gx!(g14) * gy!(g1) * gz!(g);
                        s[2] += gx!(g14) * gy!(g) * gz!(g1);
                        s[3] += gx!(g13) * gy!(g2) * gz!(g);
                        s[4] += gx!(g12) * gy!(g3) * gz!(g);
                        s[5] += gx!(g12) * gy!(g2) * gz!(g1);
                        s[6] += gx!(g13) * gy!(g) * gz!(g2);
                        s[7] += gx!(g12) * gy!(g1) * gz!(g2);
                        s[8] += gx!(g12) * gy!(g) * gz!(g3);
                        s[9] += gx!(g11) * gy!(g4) * gz!(g);
                        s[10] += gx!(g10) * gy!(g5) * gz!(g);
                        s[11] += gx!(g10) * gy!(g4) * gz!(g1);
                        s[12] += gx!(g9) * gy!(g6) * gz!(g);
                        s[13] += gx!(g8) * gy!(g7) * gz!(g);
                        s[14] += gx!(g8) * gy!(g6) * gz!(g1);
                        s[15] += gx!(g9) * gy!(g4) * gz!(g2);
                        s[16] += gx!(g8) * gy!(g5) * gz!(g2);
                        s[17] += gx!(g8) * gy!(g4) * gz!(g3);
                        s[18] += gx!(g11) * gy!(g) * gz!(g4);
                        s[19] += gx!(g10) * gy!(g1) * gz!(g4);
                        s[20] += gx!(g10) * gy!(g) * gz!(g5);
                        s[21] += gx!(g9) * gy!(g2) * gz!(g4);
                        s[22] += gx!(g8) * gy!(g3) * gz!(g4);
                        s[23] += gx!(g8) * gy!(g2) * gz!(g5);
                        s[24] += gx!(g9) * gy!(g) * gz!(g6);
                        s[25] += gx!(g8) * gy!(g1) * gz!(g6);
                        s[26] += gx!(g8) * gy!(g) * gz!(g7);
                        s[27] += gx!(g7) * gy!(g8) * gz!(g);
                        s[28] += gx!(g6) * gy!(g9) * gz!(g);
                        s[29] += gx!(g6) * gy!(g8) * gz!(g1);
                        s[30] += gx!(g5) * gy!(g10) * gz!(g);
                        s[31] += gx!(g4) * gy!(g11) * gz!(g);
                        s[32] += gx!(g4) * gy!(g10) * gz!(g1);
                        s[33] += gx!(g5) * gy!(g8) * gz!(g2);
                        s[34] += gx!(g4) * gy!(g9) * gz!(g2);
                        s[35] += gx!(g4) * gy!(g8) * gz!(g3);
                        s[36] += gx!(g3) * gy!(g12) * gz!(g);
                        s[37] += gx!(g2) * gy!(g13) * gz!(g);
                        s[38] += gx!(g2) * gy!(g12) * gz!(g1);
                        s[39] += gx!(g1) * gy!(g14) * gz!(g);
                        s[40] += gx!(g) * gy!(g15) * gz!(g);
                        s[41] += gx!(g) * gy!(g14) * gz!(g1);
                        s[42] += gx!(g1) * gy!(g12) * gz!(g2);
                        s[43] += gx!(g) * gy!(g13) * gz!(g2);
                        s[44] += gx!(g) * gy!(g12) * gz!(g3);
                        s[45] += gx!(g3) * gy!(g8) * gz!(g4);
                        s[46] += gx!(g2) * gy!(g9) * gz!(g4);
                        s[47] += gx!(g2) * gy!(g8) * gz!(g5);
                        s[48] += gx!(g1) * gy!(g10) * gz!(g4);
                        s[49] += gx!(g) * gy!(g11) * gz!(g4);
                        s[50] += gx!(g) * gy!(g10) * gz!(g5);
                        s[51] += gx!(g1) * gy!(g8) * gz!(g6);
                        s[52] += gx!(g) * gy!(g9) * gz!(g6);
                        s[53] += gx!(g) * gy!(g8) * gz!(g7);
                        s[54] += gx!(g7) * gy!(g) * gz!(g8);
                        s[55] += gx!(g6) * gy!(g1) * gz!(g8);
                        s[56] += gx!(g6) * gy!(g) * gz!(g9);
                        s[57] += gx!(g5) * gy!(g2) * gz!(g8);
                        s[58] += gx!(g4) * gy!(g3) * gz!(g8);
                        s[59] += gx!(g4) * gy!(g2) * gz!(g9);
                        s[60] += gx!(g5) * gy!(g) * gz!(g10);
                        s[61] += gx!(g4) * gy!(g1) * gz!(g10);
                        s[62] += gx!(g4) * gy!(g) * gz!(g11);
                        s[63] += gx!(g3) * gy!(g4) * gz!(g8);
                        s[64] += gx!(g2) * gy!(g5) * gz!(g8);
                        s[65] += gx!(g2) * gy!(g4) * gz!(g9);
                        s[66] += gx!(g1) * gy!(g6) * gz!(g8);
                        s[67] += gx!(g) * gy!(g7) * gz!(g8);
                        s[68] += gx!(g) * gy!(g6) * gz!(g9);
                        s[69] += gx!(g1) * gy!(g4) * gz!(g10);
                        s[70] += gx!(g) * gy!(g5) * gz!(g10);
                        s[71] += gx!(g) * gy!(g4) * gz!(g11);
                        s[72] += gx!(g3) * gy!(g) * gz!(g12);
                        s[73] += gx!(g2) * gy!(g1) * gz!(g12);
                        s[74] += gx!(g2) * gy!(g) * gz!(g13);
                        s[75] += gx!(g1) * gy!(g2) * gz!(g12);
                        s[76] += gx!(g) * gy!(g3) * gz!(g12);
                        s[77] += gx!(g) * gy!(g2) * gz!(g13);
                        s[78] += gx!(g1) * gy!(g) * gz!(g14);
                        s[79] += gx!(g) * gy!(g1) * gz!(g14);
                        s[80] += gx!(g) * gy!(g) * gz!(g15);
                    }
                    // Column-major reorder — copied VERBATIM from hess.c gout_empty.
                    let base = n * 81;
                    out[base + 0] = s[0];
                    out[base + 1] = s[3];
                    out[base + 2] = s[6];
                    out[base + 3] = s[1];
                    out[base + 4] = s[4];
                    out[base + 5] = s[7];
                    out[base + 6] = s[2];
                    out[base + 7] = s[5];
                    out[base + 8] = s[8];
                    out[base + 9] = s[27];
                    out[base + 10] = s[30];
                    out[base + 11] = s[33];
                    out[base + 12] = s[28];
                    out[base + 13] = s[31];
                    out[base + 14] = s[34];
                    out[base + 15] = s[29];
                    out[base + 16] = s[32];
                    out[base + 17] = s[35];
                    out[base + 18] = s[54];
                    out[base + 19] = s[57];
                    out[base + 20] = s[60];
                    out[base + 21] = s[55];
                    out[base + 22] = s[58];
                    out[base + 23] = s[61];
                    out[base + 24] = s[56];
                    out[base + 25] = s[59];
                    out[base + 26] = s[62];
                    out[base + 27] = s[9];
                    out[base + 28] = s[12];
                    out[base + 29] = s[15];
                    out[base + 30] = s[10];
                    out[base + 31] = s[13];
                    out[base + 32] = s[16];
                    out[base + 33] = s[11];
                    out[base + 34] = s[14];
                    out[base + 35] = s[17];
                    out[base + 36] = s[36];
                    out[base + 37] = s[39];
                    out[base + 38] = s[42];
                    out[base + 39] = s[37];
                    out[base + 40] = s[40];
                    out[base + 41] = s[43];
                    out[base + 42] = s[38];
                    out[base + 43] = s[41];
                    out[base + 44] = s[44];
                    out[base + 45] = s[63];
                    out[base + 46] = s[66];
                    out[base + 47] = s[69];
                    out[base + 48] = s[64];
                    out[base + 49] = s[67];
                    out[base + 50] = s[70];
                    out[base + 51] = s[65];
                    out[base + 52] = s[68];
                    out[base + 53] = s[71];
                    out[base + 54] = s[18];
                    out[base + 55] = s[21];
                    out[base + 56] = s[24];
                    out[base + 57] = s[19];
                    out[base + 58] = s[22];
                    out[base + 59] = s[25];
                    out[base + 60] = s[20];
                    out[base + 61] = s[23];
                    out[base + 62] = s[26];
                    out[base + 63] = s[45];
                    out[base + 64] = s[48];
                    out[base + 65] = s[51];
                    out[base + 66] = s[46];
                    out[base + 67] = s[49];
                    out[base + 68] = s[52];
                    out[base + 69] = s[47];
                    out[base + 70] = s[50];
                    out[base + 71] = s[53];
                    out[base + 72] = s[72];
                    out[base + 73] = s[75];
                    out[base + 74] = s[78];
                    out[base + 75] = s[73];
                    out[base + 76] = s[76];
                    out[base + 77] = s[79];
                    out[base + 78] = s[74];
                    out[base + 79] = s[77];
                    out[base + 80] = s[80];
                    n += 1;
                }
            }
        }
    }
    out
}

/// Contract [gx|gy|gz] into Cartesian 2e tensor for F12 (identical to two_electron version).
///
/// As of quick-260529-i2q the base / `ncomp == 1` path runs on-device via
/// [`run_f12_cart_contraction_on_backend`]; this host reference is retained as the
/// byte-identity oracle for the device-vs-host equivalence tests (cfg(test)).
#[cfg_attr(not(test), allow(dead_code))]
fn contract_f12_cart(g: &[f64], shape: F12Shape, li: u8, lj: u8, lk: u8, ll: u8) -> Vec<f64> {
    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);
    let cl_comps = cart_comps(ll);

    let gx_off = 0usize;
    let gy_off = shape.g_size;
    let gz_off = 2 * shape.g_size;

    let mut out = vec![0.0_f64; nfi * nfj * nfk * nfl];

    for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
        for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let mut sum = 0.0_f64;
                    for irys in 0..shape.nroots {
                        let x_idx = irys
                            + ix as usize * shape.di
                            + kx as usize * shape.dk
                            + lx as usize * shape.dl
                            + jx as usize * shape.dj;
                        let y_idx = irys
                            + iy as usize * shape.di
                            + ky as usize * shape.dk
                            + ly as usize * shape.dl
                            + jy as usize * shape.dj;
                        let z_idx = irys
                            + iz as usize * shape.di
                            + kz as usize * shape.dk
                            + lz as usize * shape.dl
                            + jz as usize * shape.dj;
                        sum += g[gx_off + x_idx] * g[gy_off + y_idx] * g[gz_off + z_idx];
                    }
                    let out_idx = i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                    out[out_idx] = sum;
                }
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// F12 base Cartesian-contraction splice — `#[cube(launch)]` device kernel,
// generic over F (quick task 260529-i2q). Mirrors the ECP Type-1 angular-splice
// port (ecp.rs `ecp_angular_kernel` / `run_ecp_angular_device` /
// `run_ecp_angular_splice_on_backend`, quick-260529-gbf).
//
// CLAUDE.md mandates CubeCL as the primary compute backend with host CPU work
// limited to planning/validation/marshaling. The F12 base variant's Cartesian
// contraction (`contract_f12_cart`: the bounded `[gx|gy|gz] -> cart tensor`
// triple-product accumulation, the base / `ncomp == 1` analog of the ECP angular
// splice) is the part that ports cleanly to `#[cube]`: it consumes the
// host-precomputed flat f64 `g` tensor and does bounded triple-product arithmetic
// over `irys` with NO special functions, NO break/continue. The G-tensor fill
// (`fill_g_tensor_f12` via `stg_roots_host`), the adaptive root machinery and the
// sph/spinor transforms STAY host-side as marshaling.
//
// `cart_comps` enumeration stays host-side (marshaling); the kernel consumes flat
// u32 power triples (`[mi*3 + axis]`, like ECP's `cart_comps_flat_u32`). The
// kernel computes the FULL `nfi*nfj*nfk*nfl` Cartesian block in ONE launch (single
// work item, `if UNIT_POS == 0`), preserving the host nested loop order
// (l outer, then k, j, i, irys inner) so the f64 result is byte-identical.
// ─────────────────────────────────────────────────────────────────────────────

/// Flatten `cart_comps(l)` into the `[mi*3 + axis]` u32 triple layout the device
/// kernel consumes (marshaling — keeps `cart_comps` enumeration host-side).
/// Mirrors `ecp.rs::cart_comps_flat_u32`.
// Wired into `f12_kernel_core` in quick-260529-i2q Task 2; until then it is only
// exercised by the device-vs-host equivalence tests.
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

/// F12 base Cartesian-contraction device kernel, generic over `F: Float`.
///
/// Computes the full `nfi*nfj*nfk*nfl` Cartesian block (the base / `ncomp == 1`
/// variant of `contract_f12_cart`) in ONE launch (single work item,
/// `UNIT_POS == 0`). All math uses `F` arithmetic, statement-form `if`, `u32`
/// indices, and `while` loops bounded by runtime `u32` component counts — no
/// for/break/continue, no special functions, no device-local `Array` scratch
/// (inline-recompute, like ECP Type-2). The nested loop order (l outer, then k,
/// j, i, with `irys` 0..nroots innermost) is IDENTICAL to the host driver
/// `contract_f12_cart` so the f64 summation order — and hence byte-identity — is
/// preserved.
///
/// Args mirror `contract_f12_cart`:
/// - `g`: the flat `[gx|gy|gz]` buffer; `gx_off=0`, `gy_off=g_size`, `gz_off=2*g_size`.
/// - `comps_i/j/k/l`: flat u32 cartesian-power triples (3 entries per component).
/// - `out`: the `nfi*nfj*nfk*nfl` Cartesian block.
/// - scalars (u32): `nfi, nfj, nfk, nfl, nroots, di, dk, dl, dj, g_size`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn f12_cart_contraction_kernel<F: Float + CubeElement>(
    g: &Array<F>,
    comps_i: &Array<u32>,
    comps_j: &Array<u32>,
    comps_k: &Array<u32>,
    comps_l: &Array<u32>,
    out: &mut Array<F>,
    nfi: u32,
    nfj: u32,
    nfk: u32,
    nfl: u32,
    nroots: u32,
    di: u32,
    dk: u32,
    dl: u32,
    dj: u32,
    g_size: u32,
) {
    if UNIT_POS == 0u32 {
        let gx_off = 0u32;
        let gy_off = g_size;
        let gz_off = 2u32 * g_size;

        // Zero the output block.
        let out_len = nfi * nfj * nfk * nfl;
        let mut oz = 0u32;
        while oz < out_len {
            out[oz as usize] = F::new(0.0);
            oz += 1u32;
        }

        // 4-nested component loop (l outer, then k, j, i) × inner irys
        // accumulation — IDENTICAL order to the host `contract_f12_cart`.
        let mut l_idx = 0u32;
        while l_idx < nfl {
            let lx = comps_l[(l_idx * 3u32) as usize];
            let ly = comps_l[(l_idx * 3u32 + 1u32) as usize];
            let lz = comps_l[(l_idx * 3u32 + 2u32) as usize];
            let mut k_idx = 0u32;
            while k_idx < nfk {
                let kx = comps_k[(k_idx * 3u32) as usize];
                let ky = comps_k[(k_idx * 3u32 + 1u32) as usize];
                let kz = comps_k[(k_idx * 3u32 + 2u32) as usize];
                let mut j_idx = 0u32;
                while j_idx < nfj {
                    let jx = comps_j[(j_idx * 3u32) as usize];
                    let jy = comps_j[(j_idx * 3u32 + 1u32) as usize];
                    let jz = comps_j[(j_idx * 3u32 + 2u32) as usize];
                    let mut i_idx = 0u32;
                    while i_idx < nfi {
                        let ix = comps_i[(i_idx * 3u32) as usize];
                        let iy = comps_i[(i_idx * 3u32 + 1u32) as usize];
                        let iz = comps_i[(i_idx * 3u32 + 2u32) as usize];

                        let mut sum = F::new(0.0);
                        let mut irys = 0u32;
                        while irys < nroots {
                            let x_idx = irys + ix * di + kx * dk + lx * dl + jx * dj;
                            let y_idx = irys + iy * di + ky * dk + ly * dl + jy * dj;
                            let z_idx = irys + iz * di + kz * dk + lz * dl + jz * dj;
                            let gx = g[(gx_off + x_idx) as usize];
                            let gy = g[(gy_off + y_idx) as usize];
                            let gz = g[(gz_off + z_idx) as usize];
                            sum += gx * gy * gz;
                            irys += 1u32;
                        }

                        let out_idx =
                            i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                        out[out_idx as usize] = sum;
                        i_idx += 1u32;
                    }
                    j_idx += 1u32;
                }
                k_idx += 1u32;
            }
            l_idx += 1u32;
        }
    }
}

/// Dispatch [`f12_cart_contraction_kernel`] at `f64` on a resolved backend's
/// client, reading back the `nfi*nfj*nfk*nfl` Cartesian block.
///
/// Generic over `R: Runtime` so the same path serves CPU, ROCm, etc. Intermediate
/// device compute is `f64` (F12 keeps f64 staging; the byte-identity gate is
/// f64/CPU-vs-C). Mirrors `ecp.rs::run_ecp_angular_device`.
// Wired into `f12_kernel_core` in quick-260529-i2q Task 2; until then it is only
// exercised by the device-vs-host equivalence tests.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn run_f12_cart_contraction_device<R: Runtime>(
    client: &ComputeClient<R>,
    g: &[f64],
    shape: F12Shape,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
) -> Vec<f64> {
    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let out_len = nfi * nfj * nfk * nfl;

    let comps_i = cart_comps_flat_u32(li);
    let comps_j = cart_comps_flat_u32(lj);
    let comps_k = cart_comps_flat_u32(lk);
    let comps_l = cart_comps_flat_u32(ll);

    // Sanity (T-i2q-01): buffer lengths must match what the kernel indexes, or
    // the device kernel reads out of bounds.
    debug_assert_eq!(g.len(), 3 * shape.g_size, "f12 cart: g len != 3*g_size");
    debug_assert_eq!(comps_i.len(), nfi * 3, "f12 cart: comps_i len != nfi*3");
    debug_assert_eq!(comps_j.len(), nfj * 3, "f12 cart: comps_j len != nfj*3");
    debug_assert_eq!(comps_k.len(), nfk * 3, "f12 cart: comps_k len != nfk*3");
    debug_assert_eq!(comps_l.len(), nfl * 3, "f12 cart: comps_l len != nfl*3");

    let g_h = client.create_from_slice(f64::as_bytes(g));
    let comps_i_h = client.create_from_slice(u32::as_bytes(&comps_i));
    let comps_j_h = client.create_from_slice(u32::as_bytes(&comps_j));
    let comps_k_h = client.create_from_slice(u32::as_bytes(&comps_k));
    let comps_l_h = client.create_from_slice(u32::as_bytes(&comps_l));

    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    f12_cart_contraction_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(g_h, g.len()) },
        unsafe { ArrayArg::from_raw_parts(comps_i_h, comps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(comps_j_h, comps_j.len()) },
        unsafe { ArrayArg::from_raw_parts(comps_k_h, comps_k.len()) },
        unsafe { ArrayArg::from_raw_parts(comps_l_h, comps_l.len()) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
        nfi as u32,
        nfj as u32,
        nfk as u32,
        nfl as u32,
        shape.nroots as u32,
        shape.di as u32,
        shape.dk as u32,
        shape.dl as u32,
        shape.dj as u32,
        shape.g_size as u32,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Backend-dispatch wrapper for the F12 base Cartesian contraction: routes the
/// `#[cube(launch)]` [`f12_cart_contraction_kernel`] onto the resolved backend's
/// device client (Cpu => CpuRuntime, Rocm => HipRuntime, Wgpu, Cuda, Metal — each
/// `#[cfg]`-gated), mirroring `ecp.rs::run_ecp_angular_splice_on_backend`. Runs at
/// f64 (F12 keeps f64 staging / byte-identity gate). Returns the
/// `nfi*nfj*nfk*nfl` block.
// Wired into `f12_kernel_core` in quick-260529-i2q Task 2.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn run_f12_cart_contraction_on_backend(
    backend: &ResolvedBackend,
    g: &[f64],
    shape: F12Shape,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_f12_cart_contraction_device::<cubecl::cpu::CpuRuntime>(
            client, g, shape, li, lj, lk, ll,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_f12_cart_contraction_device::<cubecl_wgpu::WgpuRuntime>(
                client, g, shape, li, lj, lk, ll,
            )
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_f12_cart_contraction_device::<cubecl_cuda::CudaRuntime>(
            client, g, shape, li, lj, lk, ll,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_f12_cart_contraction_device::<cubecl_hip::HipRuntime>(
                client, g, shape, li, lj, lk, ll,
            )
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_f12_cart_contraction_device::<cubecl_wgpu::WgpuRuntime>(
                client, g, shape, li, lj, lk, ll,
            )
        }
    }
}

/// Shared F12 kernel core called by all 10 entry points.
///
/// Follows the same structure as `launch_two_electron` in `two_electron.rs` with the
/// following changes:
/// - Uses derivative-adjusted angular momenta (li_ceil = li + variant.i_inc, etc.)
/// - Uses F12 nroots formula: `(L_tot + 3) / 2`
/// - Calls `stg_roots_host` for root computation
/// - Applies STG or YP specific weight post-processing
///
/// # Parameters
/// - `is_stg`: true for STG post-processing, false for YP post-processing
fn f12_kernel_core(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    _specialization: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
    variant: &F12Variant,
    is_stg: bool,
) -> Result<ExecutionStats, cintxRsError> {
    let shells = plan.shells.as_slice();
    if shells.len() < 4 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_f12",
            detail: "f12 kernel requires exactly 4 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let li = shell_i.ang_momentum as usize;
    let lj = shell_j.ang_momentum as usize;
    let lk = shell_k.ang_momentum as usize;
    let ll = shell_l.ang_momentum as usize;

    // Apply derivative angular momentum increments
    let li_ceil = li + variant.i_inc;
    let lj_ceil = lj + variant.j_inc;
    let lk_ceil = lk + variant.k_inc;
    let ll_ceil = ll + variant.l_inc;

    let shape = build_f12_shape(li_ceil, lj_ceil, lk_ceil, ll_ceil);

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    let rl = atoms[shell_l.atom_index as usize].coord_bohr;

    // Angular momenta for the sph transform use BASE (not ceil) values.
    let li_base_u8 = li as u8;
    let lj_base_u8 = lj as u8;
    let lk_base_u8 = lk as u8;
    let ll_base_u8 = ll as u8;

    // Ceil values needed for the G tensor shape and gout internal loops.
    let li_u8 = li_ceil as u8;
    let lj_u8 = lj_ceil as u8;
    let lk_u8 = lk_ceil as u8;
    let ll_u8 = ll_ceil as u8;

    // ncart at CEIL angular momenta — used for contract_f12_cart (base variant only)
    let nfi_ceil = ncart(li_u8);
    let nfj_ceil = ncart(lj_u8);
    let nfk_ceil = ncart(lk_u8);
    let nfl_ceil = ncart(ll_u8);

    // ncart/nsph at BASE angular momenta — used for gout and sph transforms
    let nfi_base = ncart(li_base_u8);
    let nfj_base = ncart(lj_base_u8);
    let nfk_base = ncart(lk_base_u8);
    let nfl_base = ncart(ll_base_u8);
    let nf_base = nfi_base * nfj_base * nfk_base * nfl_base;

    let nsi = nsph(li_base_u8);
    let nsj = nsph(lj_base_u8);
    let nsk = nsph(lk_base_u8);
    let nsl = nsph(ll_base_u8);

    let ncomp = variant.ncomp;

    // Common factor: same as two_electron (fac_sp for all four shells)
    let sp_factor = common_fac_sp(li_base_u8)
        * common_fac_sp(lj_base_u8)
        * common_fac_sp(lk_base_u8)
        * common_fac_sp(ll_base_u8);
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * sp_factor;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    if ncomp == 1 {
        // ── Base variant: single Cartesian contraction and one sph transform ──
        // The cart_buf uses CEIL angular momenta (matching the G tensor shape).
        let mut cart_buf = vec![0.0_f64; nfi_ceil * nfj_ceil * nfk_ceil * nfl_ceil];

        for pi in 0..n_prim_i {
            let ai = shell_i.exponents[pi];
            for pj in 0..n_prim_j {
                let aj = shell_j.exponents[pj];
                let pdata_ij = compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );
                for pk in 0..n_prim_k {
                    let ak = shell_k.exponents[pk];
                    for pl in 0..n_prim_l {
                        let al = shell_l.exponents[pl];
                        let pdata_kl = compute_pdata_host(
                            ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                        );
                        let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                        let g = fill_g_tensor_f12(
                            ai, aj, ak, al, &ri, &rj, &rk, &rl,
                            shape, quartet_fac, zeta, is_stg,
                        );
                        // Base Cartesian contraction now runs on-device as a
                        // #[cube(launch)] kernel dispatched onto the resolved
                        // backend's ComputeClient (quick-260529-i2q). Launches at
                        // f64 with the SAME nested summation order as the host
                        // `contract_f12_cart`, so byte-identity is preserved.
                        let prim_cart = run_f12_cart_contraction_on_backend(
                            backend, &g, shape, li_u8, lj_u8, lk_u8, ll_u8,
                        );

                        for ci in 0..n_ctr_i {
                            let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                            for cj in 0..n_ctr_j {
                                let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                                for ck in 0..n_ctr_k {
                                    let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                                    for cl in 0..n_ctr_l {
                                        let coeff_l = shell_l.coefficients[pl * n_ctr_l + cl];
                                        let weight = coeff_i * coeff_j * coeff_k * coeff_l;
                                        for idx in 0..cart_buf.len() {
                                            cart_buf[idx] += weight * prim_cart[idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        match plan.representation {
            Representation::Spheric => {
                // Use base angular momenta for sph transform
                let sph = cart_to_sph_2e(&cart_buf, li_base_u8, lj_base_u8, lk_base_u8, ll_base_u8);
                let sph_size = nsi * nsj * nsk * nsl;
                let copy_len = staging.len().min(sph.len()).min(sph_size);
                staging[..copy_len].copy_from_slice(&sph[..copy_len]);
            }
            Representation::Spinor => {
                let kappa_i = shell_i.kappa;
                let kappa_j = shell_j.kappa;
                let kappa_k = shell_k.kappa;
                let kappa_l = shell_l.kappa;
                cart_to_spinor_sf_4d(
                    staging, &cart_buf,
                    li_base_u8, kappa_i, lj_base_u8, kappa_j,
                    lk_base_u8, kappa_k, ll_base_u8, kappa_l,
                )?;
            }
            Representation::Cart => {
                let copy_len = staging.len().min(cart_buf.len());
                staging[..copy_len].copy_from_slice(&cart_buf[..copy_len]);
            }
        }
    } else {
        // ── Derivative variant: per-primitive gout contraction, then per-component sph ──
        //
        // The gout functions produce ncomp * nf_base values per primitive.
        // These are accumulated (contracted) across primitives, then sph-transformed
        // per component. The nabla operators read into the ceil headroom of the G tensor.
        let mut gout_contracted = vec![0.0_f64; ncomp * nf_base];

        for pi in 0..n_prim_i {
            let ai = shell_i.exponents[pi];
            for pj in 0..n_prim_j {
                let aj = shell_j.exponents[pj];
                let pdata_ij = compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );
                for pk in 0..n_prim_k {
                    let ak = shell_k.exponents[pk];
                    for pl in 0..n_prim_l {
                        let al = shell_l.exponents[pl];
                        let pdata_kl = compute_pdata_host(
                            ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                        );
                        let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                        let g = fill_g_tensor_f12(
                            ai, aj, ak, al, &ri, &rj, &rk, &rl,
                            shape, quartet_fac, zeta, is_stg,
                        );

                        // Apply the variant-specific gout function to get ncomp * nf_base values.
                        // The gout functions use BASE angular momenta for the loop bounds.
                        let prim_gout = match ncomp {
                            3 => gout_ip1(&g, &shape, li, lj, lk, ll, ai),
                            9 => match (variant.j_inc, variant.k_inc) {
                                (0, 0) => gout_ipip1(&g, &shape, li, lj, lk, ll, ai),
                                (1, 0) => gout_ipvip1(&g, &shape, li, lj, lk, ll, ai, aj),
                                (0, 1) => gout_ip1ip2(&g, &shape, li, lj, lk, ll, ai, ak),
                                _ => return Err(cintxRsError::UnsupportedApi {
                                    requested: format!("f12 derivative: unknown 9-component variant j_inc={} k_inc={}", variant.j_inc, variant.k_inc),
                                }),
                            },
                            _ => return Err(cintxRsError::UnsupportedApi {
                                requested: format!("f12 derivative: unsupported ncomp={ncomp}"),
                            }),
                        };

                        // Accumulate with contraction weights
                        for ci in 0..n_ctr_i {
                            let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                            for cj in 0..n_ctr_j {
                                let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                                for ck in 0..n_ctr_k {
                                    let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                                    for cl in 0..n_ctr_l {
                                        let coeff_l = shell_l.coefficients[pl * n_ctr_l + cl];
                                        let weight = coeff_i * coeff_j * coeff_k * coeff_l;
                                        for idx in 0..gout_contracted.len() {
                                            gout_contracted[idx] += weight * prim_gout[idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply sph transform per component and write to staging.
        // staging layout: [comp0_sph | comp1_sph | ... | comp{ncomp-1}_sph]
        match plan.representation {
            Representation::Spheric => {
                let sph_size = nsi * nsj * nsk * nsl;
                for comp in 0..ncomp {
                    let cart_slice = &gout_contracted[comp * nf_base..(comp + 1) * nf_base];
                    let sph = cart_to_sph_2e(cart_slice, li_base_u8, lj_base_u8, lk_base_u8, ll_base_u8);
                    let stage_off = comp * sph_size;
                    // FND-06 (D-04): the upfront planner assertion proves staging is
                    // large enough for all `ncomp` components; copy unconditionally
                    // (no staging.len() clamp, no silent truncation of trailing comps).
                    let copy_len = sph.len().min(sph_size);
                    staging[stage_off..stage_off + copy_len].copy_from_slice(&sph[..copy_len]);
                }
            }
            Representation::Cart => {
                let copy_len = staging.len().min(gout_contracted.len());
                staging[..copy_len].copy_from_slice(&gout_contracted[..copy_len]);
            }
            Representation::Spinor => {
                // Derivative F12 spinor not implemented; return empty
                return Err(cintxRsError::UnsupportedApi {
                    requested: "F12 derivative spinor representation not supported".to_owned(),
                });
            }
        }
    }

    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > 1e-18)
        .count() as i32;

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

// --- 10 individual entry point functions ---

fn launch_stg_base(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_BASE, true)
}

fn launch_stg_ip1(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IP1, true)
}

fn launch_stg_ipip1(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IPIP1, true)
}

fn launch_stg_ipvip1(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IPVIP1, true)
}

fn launch_stg_ip1ip2(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IP1IP2, true)
}

fn launch_yp_base(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_BASE, false)
}

fn launch_yp_ip1(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IP1, false)
}

fn launch_yp_ipip1(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IPIP1, false)
}

fn launch_yp_ipvip1(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IPVIP1, false)
}

fn launch_yp_ip1ip2(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    spec: &SpecializationKey,
    staging: &mut [f64],
    zeta: f64,
) -> Result<ExecutionStats, cintxRsError> {
    f12_kernel_core(backend, plan, spec, staging, zeta, &F12_IP1IP2, false)
}

/// Generic inner for the f12 launcher.
///
/// Contains the full routing logic of `launch_f12` parameterized over the output
/// float type `F: CintFloat`. The staging buffer is typed `&mut [F]` so the
/// bytemuck-cast pattern at the outer boundary is sound (Plan 01 A5 proven).
///
/// `f12_zeta` STAYS `Option<f64>` on `ExecutionOptions`/`OperatorEnvParams` (env-side
/// f64, D-06 / Open Question 3). The `zeta: f64` value is used in `f64` arithmetic
/// throughout the entire computation pipeline (G-tensor, root computation, weight
/// post-processing). If the kernel were to use `F`-typed internal math in the future,
/// the cast would be `F::from_f64_lossy(zeta)` at the point zeta enters the `F` math.
/// For now, all intermediates stay `f64` and only the final staging write uses
/// `F::from_f64_lossy`.
fn launch_f12_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Validate and extract f12_zeta. The field stays Option<f64> (env-side, D-06).
    // cast to F only when/if it enters F-typed kernel math (currently stays f64 throughout).
    validate_f12_env_params("f12", &plan.operator_env_params)?;

    // f12_zeta: Option<f64> — env parameter stays f64 per D-06 / Open Question 3.
    // If we ever move kernel math to F, the cast point is: F::from_f64_lossy(zeta)
    let zeta: f64 = plan
        .operator_env_params
        .f12_zeta
        .expect("validate_f12_env_params guarantees Some non-zero zeta");

    let operator_name = plan.descriptor.operator_name();

    let (is_stg, variant_suffix) = if let Some(suffix) = operator_name.strip_prefix("stg") {
        (true, suffix)
    } else if let Some(suffix) = operator_name.strip_prefix("yp") {
        (false, suffix)
    } else {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("f12 launch: unrecognized operator_name: {operator_name}"),
        });
    };

    // CR-02: the outer F32 arm already slices staging to out_elems, so staging.len()
    // is the TRUE output element count for both f64 and f32. Size the temporary f64
    // buffer to out_elems (not staging.len() from a doubled f32 view) to prevent
    // sub-kernel length-contract violations and stale-lane corruption.
    let out_elems = staging.len(); // true output element count (outer arm sliced for F32; == chunk_len for F64)

    // OOM-safe / no-partial-write guard: assert the inner slice covers out_elems.
    // With CR-01 the outer arm already enforces this; the inner guard documents the invariant.
    if staging.len() < out_elems {
        return Err(cintxRsError::BufferTooSmall {
            required: out_elems,
            provided: staging.len(),
        });
    }

    let mut staging_f64 = vec![0.0_f64; out_elems];

    let stats = if is_stg {
        match variant_suffix {
            "" => launch_stg_base(backend, plan, specialization, &mut staging_f64, zeta),
            "_ip1" => launch_stg_ip1(backend, plan, specialization, &mut staging_f64, zeta),
            "_ipip1" => launch_stg_ipip1(backend, plan, specialization, &mut staging_f64, zeta),
            "_ipvip1" => launch_stg_ipvip1(backend, plan, specialization, &mut staging_f64, zeta),
            "_ip1ip2" => launch_stg_ip1ip2(backend, plan, specialization, &mut staging_f64, zeta),
            other => Err(cintxRsError::UnsupportedApi {
                requested: format!("f12 launch: unknown stg variant suffix: {other}"),
            }),
        }
    } else {
        match variant_suffix {
            "" => launch_yp_base(backend, plan, specialization, &mut staging_f64, zeta),
            "_ip1" => launch_yp_ip1(backend, plan, specialization, &mut staging_f64, zeta),
            "_ipip1" => launch_yp_ipip1(backend, plan, specialization, &mut staging_f64, zeta),
            "_ipvip1" => launch_yp_ipvip1(backend, plan, specialization, &mut staging_f64, zeta),
            "_ip1ip2" => launch_yp_ip1ip2(backend, plan, specialization, &mut staging_f64, zeta),
            other => Err(cintxRsError::UnsupportedApi {
                requested: format!("f12 launch: unknown yp variant suffix: {other}"),
            }),
        }
    }?;

    // CR-02: readback bounded to out_elems.
    // Cast f64 results to F at the output boundary.
    // For f64 this is a zero-cost identity; for f32 it truncates.
    for (dst, &src) in staging[..out_elems].iter_mut().zip(staging_f64.iter()) {
        *dst = F::from_f64_lossy(src);
    }

    // Per-symbol nonzero sentinel
    // WR-06: precision-aware sentinel so f32 stale lanes (< f32 noise floor ~1e-7)
    // are not counted. Bounded to out_elems so stale upper-half lanes cannot register.
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging[..out_elems]
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    // WR-01: true-output bytes (out_elems, not the doubled f32 lane count).
    let staging_bytes = out_elems * std::mem::size_of::<F>();
    Ok(ExecutionStats {
        not0,
        peak_workspace_bytes: staging_bytes,
        transfer_bytes: staging_bytes,
        ..stats
    })
}

/// Dispatch to the correct STG or YP entry point based on operator_name.
///
/// Outer precision dispatcher: keeps the registered `FamilyLaunchFn` signature and
/// `#[cfg(feature = "with-f12")]` gate unchanged. Internally matches on `plan.precision`
/// and delegates to `launch_f12_typed::<F>`, reinterpreting staging via
/// `bytemuck::cast_slice_mut` for the F32 arm (A5 proven sound).
/// CR-01: captures the true output element count BEFORE the bytemuck cast and bounds
/// the typed inner to that count, returning `BufferTooSmall` if the view cannot hold it.
///
/// `f12_zeta` STAYS `Option<f64>` on `ExecutionOptions`/`OperatorEnvParams` (D-06).
/// The cast to `F` is `F::from_f64_lossy(zeta)` inside the typed inner, at the
/// kernel output boundary (see `launch_f12_typed` documentation).
pub fn launch_f12(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_f12_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            // CR-01: capture the true output element count BEFORE the bytemuck cast.
            // api.rs sizes Vec<f64> to chunk_len == the TRUE output element count;
            // after cast staging_f32.len() == chunk_len*2, so out_elems = staging.len() pre-cast.
            let out_elems = staging.len(); // f64 slice length == TRUE output element count
            let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
            if staging_f32.len() < out_elems {
                return Err(cintxRsError::BufferTooSmall {
                    required: out_elems,
                    provided: staging_f32.len(),
                });
            }
            launch_f12_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-2c: launch_f12_typed::<f64> byte-identical to launch_f12 at f64.
    // RED: compile fails until launch_f12_typed is defined.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_f12_parity_f64() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Spheric,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a0 = make_s_shell(0);
        let shell_a1 = make_s_shell(0);
        let shell_b0 = make_s_shell(1);
        let shell_b1 = make_s_shell(1);
        let all_shells = Arc::from(vec![shell_a0.clone(), shell_a1.clone(), shell_b0.clone(), shell_b1.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a0, shell_a1, shell_b0, shell_b1]).unwrap();

        use cintx_ops::resolver::Resolver;
        let desc = Resolver::descriptor_by_symbol("int2e_stg_sph").expect("int2e_stg_sph must exist");
        let op_id = desc.id;

        let opts = ExecutionOptions::default();
        let query = query_workspace(op_id, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op_id, Representation::Spheric, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;
        // f12_zeta stays Option<f64> on plan.operator_env_params (env-side, D-06 / Open Q3)
        plan.operator_env_params.f12_zeta = Some(1.0);

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging_outer = vec![0.0_f64; 1];
        let mut staging_typed = vec![0.0_f64; 1];

        let result_outer = launch_f12(&backend, &plan, &spec, &mut staging_outer);
        assert!(result_outer.is_ok(), "outer f64 f12 should succeed: {:?}", result_outer);

        // RED: compile fails until launch_f12_typed is defined
        let result_typed = launch_f12_typed::<f64>(&backend, &plan, &spec, &mut staging_typed);
        assert!(result_typed.is_ok(), "typed f64 f12 should succeed: {:?}", result_typed);

        assert_eq!(staging_outer[0].to_bits(), staging_typed[0].to_bits(),
            "f64 outer and typed f12 should be byte-identical: outer={} typed={}", staging_outer[0], staging_typed[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-2d: F32 path runs without panic; zeta=0 rejection still fires.
    // RED: compile fails until launch_f12 dispatches on plan.precision.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_f12_f32_smoke() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Spheric,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a0 = make_s_shell(0);
        let shell_a1 = make_s_shell(0);
        let shell_b0 = make_s_shell(1);
        let shell_b1 = make_s_shell(1);
        let all_shells = Arc::from(vec![shell_a0.clone(), shell_a1.clone(), shell_b0.clone(), shell_b1.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a0, shell_a1, shell_b0, shell_b1]).unwrap();

        use cintx_ops::resolver::Resolver;
        let desc = Resolver::descriptor_by_symbol("int2e_stg_sph").expect("int2e_stg_sph must exist");
        let op_id = desc.id;

        let opts = ExecutionOptions::default();
        let query = query_workspace(op_id, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op_id, Representation::Spheric, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;
        plan.operator_env_params.f12_zeta = Some(1.0);

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_f12(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "F32 f12 should succeed: {:?}", result);

        let staging_f32 = bytemuck::cast_slice::<f64, f32>(&staging);
        assert!(staging_f32[0].is_finite(), "F32 f12 result should be finite: {}", staging_f32[0]);
        assert!(staging_f32[0].abs() > 0.0, "F32 f12 result should be nonzero: {}", staging_f32[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-2e: zeta=0 rejection still fires (validate_f12_env_params unchanged).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_f12_zeta_zero_rejection() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Spheric,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a0 = make_s_shell(0);
        let shell_a1 = make_s_shell(0);
        let shell_b0 = make_s_shell(1);
        let shell_b1 = make_s_shell(1);
        let all_shells = Arc::from(vec![shell_a0.clone(), shell_a1.clone(), shell_b0.clone(), shell_b1.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a0, shell_a1, shell_b0, shell_b1]).unwrap();

        use cintx_ops::resolver::Resolver;
        let desc = Resolver::descriptor_by_symbol("int2e_stg_sph").expect("int2e_stg_sph must exist");
        let op_id = desc.id;

        // No zeta set — should be rejected
        let opts = ExecutionOptions::default();
        let query = query_workspace(op_id, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op_id, Representation::Spheric, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_f12(&backend, &plan, &spec, &mut staging);
        assert!(result.is_err(), "f12 without zeta should fail closed: got {:?}", result);
    }

    /// Smoke test: STG weight post-processing produces non-zero values and differs from YP.
    ///
    /// Uses a known (ta, ua) pair and verifies that STG and YP post-processing produce
    /// distinct weight values.
    #[test]
    fn stg_vs_yp_weight_post_processing_diverge() {
        let nroots = 1;
        let ta = 1.0_f64;  // Moderate t value
        let zeta = 1.2_f64;
        let aij = 1.0_f64;
        let akl = 1.0_f64;
        let a0 = aij * akl / (aij + akl);
        let ua = 0.25 * zeta * zeta / a0;

        let (roots_stg, weights_raw_stg) = stg_roots_host::<f64>(nroots, ta, ua);
        let (roots_yp, weights_raw_yp) = stg_roots_host::<f64>(nroots, ta, ua);

        assert!(!roots_stg.is_empty(), "stg_roots_host should return non-empty roots");
        assert!(!weights_raw_stg.is_empty(), "stg_roots_host should return non-empty weights");

        // Apply STG weight post-processing
        let ua2 = 2.0 * ua / zeta;
        let mut stg_weights = weights_raw_stg.clone();
        let mut stg_u = roots_stg.clone();
        for irys in 0..nroots {
            stg_weights[irys] *= (1.0 - roots_stg[irys]) * ua2;
            stg_u[irys] = roots_stg[irys] / (1.0 - roots_stg[irys]);
        }

        // Apply YP weight post-processing
        let mut yp_weights = weights_raw_yp.clone();
        let mut yp_u = roots_yp.clone();
        for irys in 0..nroots {
            yp_weights[irys] *= roots_yp[irys];
            yp_u[irys] = roots_yp[irys] / (1.0 - roots_yp[irys]);
        }

        // STG and YP weights should differ
        assert!(
            (stg_weights[0] - yp_weights[0]).abs() > 1e-15,
            "STG and YP weight post-processing should produce different weights: stg={}, yp={}",
            stg_weights[0],
            yp_weights[0]
        );

        // Both should be finite and non-zero
        assert!(stg_weights[0].is_finite() && stg_weights[0].abs() > 1e-50,
            "STG weight should be finite and non-zero, got {}", stg_weights[0]);
        assert!(yp_weights[0].is_finite() && yp_weights[0].abs() > 1e-50,
            "YP weight should be finite and non-zero, got {}", yp_weights[0]);

        // Transformed u values should be equal (same formula applied to same inputs)
        assert!(
            (stg_u[0] - yp_u[0]).abs() < 1e-14,
            "Both should transform u the same way: stg_u={}, yp_u={}",
            stg_u[0],
            yp_u[0]
        );
    }

    /// Verify F12 nroots formula matches libcint g2e_f12.c line 75: (L_tot + 3) / 2.
    #[test]
    fn f12_nroots_formula_matches_libcint() {
        // For ss|ss: L_tot = 0, nroots = (0+3)/2 = 1
        let s = build_f12_shape(0, 0, 0, 0);
        assert_eq!(s.nroots, 1, "ss|ss nroots should be 1, got {}", s.nroots);

        // For sp|ss: L_tot = 1, nroots = (1+3)/2 = 2
        let sp = build_f12_shape(0, 1, 0, 0);
        assert_eq!(sp.nroots, 2, "sp|ss nroots should be 2, got {}", sp.nroots);

        // For pp|ss: L_tot = 2, nroots = (2+3)/2 = 2
        let pp = build_f12_shape(1, 1, 0, 0);
        assert_eq!(pp.nroots, 2, "pp|ss nroots should be 2, got {}", pp.nroots);

        // For pp|pp: L_tot = 4, nroots = (4+3)/2 = 3
        let pppp = build_f12_shape(1, 1, 1, 1);
        assert_eq!(pppp.nroots, 3, "pp|pp nroots should be 3, got {}", pppp.nroots);
    }

    /// Verify F12Variant constants match cint2e_f12.c ng arrays.
    #[test]
    fn f12_variant_constants_match_cint2e_f12_ng_arrays() {
        // base: ng = [0, 0, 0, 0, ...]
        assert_eq!(F12_BASE.i_inc, 0);
        assert_eq!(F12_BASE.j_inc, 0);
        assert_eq!(F12_BASE.k_inc, 0);
        assert_eq!(F12_BASE.l_inc, 0);
        assert_eq!(F12_BASE.ncomp, 1);

        // ip1: ng = [1, 0, 0, 0, ..., 3]
        assert_eq!(F12_IP1.i_inc, 1);
        assert_eq!(F12_IP1.j_inc, 0);
        assert_eq!(F12_IP1.k_inc, 0);
        assert_eq!(F12_IP1.l_inc, 0);
        assert_eq!(F12_IP1.ncomp, 3);

        // ipip1: ng = [2, 0, 0, 0, ..., 9]
        assert_eq!(F12_IPIP1.i_inc, 2);
        assert_eq!(F12_IPIP1.j_inc, 0);
        assert_eq!(F12_IPIP1.k_inc, 0);
        assert_eq!(F12_IPIP1.l_inc, 0);
        assert_eq!(F12_IPIP1.ncomp, 9);

        // ipvip1: ng = [1, 1, 0, 0, ..., 9]
        assert_eq!(F12_IPVIP1.i_inc, 1);
        assert_eq!(F12_IPVIP1.j_inc, 1);
        assert_eq!(F12_IPVIP1.k_inc, 0);
        assert_eq!(F12_IPVIP1.l_inc, 0);
        assert_eq!(F12_IPVIP1.ncomp, 9);

        // ip1ip2: ng = [1, 0, 1, 0, ..., 9]
        assert_eq!(F12_IP1IP2.i_inc, 1);
        assert_eq!(F12_IP1IP2.j_inc, 0);
        assert_eq!(F12_IP1IP2.k_inc, 1);
        assert_eq!(F12_IP1IP2.l_inc, 0);
        assert_eq!(F12_IP1IP2.ncomp, 9);
    }

    // ── F12 base Cartesian-contraction splice: device-vs-host equivalence ──
    // (quick task 260529-i2q). The device kernel must reproduce the host
    // `contract_f12_cart` byte-for-byte (identical f64 summation order).
    #[cfg(feature = "cpu")]
    mod f12_cart_device_cross_check {
        use super::super::*;

        fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
            cubecl::cpu::CpuRuntime::client(&Default::default())
        }

        /// Tiny LCG so the synthetic `g` buffer is deterministic and reproducible.
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

        /// f64 device-vs-host equivalence: max-abs-diff MUST be exactly 0.0
        /// (identical f64 op order ⇒ byte-identity).
        fn assert_f64_byte_identity(li: u8, lj: u8, lk: u8, ll: u8, seed: u64) {
            let shape = build_f12_shape(li as usize, lj as usize, lk as usize, ll as usize);
            // Synthetic flat [gx|gy|gz] buffer (3 * g_size entries).
            let mut rng = Lcg::new(seed);
            let g: Vec<f64> = (0..3 * shape.g_size).map(|_| rng.next_f64()).collect();

            let host = contract_f12_cart(&g, shape, li, lj, lk, ll);
            let dev = run_f12_cart_contraction_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client(),
                &g,
                shape,
                li,
                lj,
                lk,
                ll,
            );
            assert_eq!(
                host.len(),
                dev.len(),
                "len mismatch li={li} lj={lj} lk={lk} ll={ll}"
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
                "f12 cart device/host f64 max-abs-diff must be 0.0 (li={li} lj={lj} lk={lk} ll={ll}), got {max_diff:e}"
            );
            assert!(
                any_nonzero,
                "host reference all zeros (li={li} lj={lj} lk={lk} ll={ll})"
            );
        }

        #[test]
        fn f12_cart_device_matches_host_f64() {
            // All-s quartet plus quartets exercising p-shell cart_comps / strides.
            let quartets: [(u8, u8, u8, u8); 5] = [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (1, 1, 0, 0),
                (1, 0, 1, 0),
            ];
            for (idx, &(li, lj, lk, ll)) in quartets.iter().enumerate() {
                assert_f64_byte_identity(
                    li,
                    lj,
                    lk,
                    ll,
                    0x9E3779B97F4A7C15 ^ (idx as u64).wrapping_mul(0x100000001B3),
                );
            }
        }

        /// Generic-F: launching the SAME kernel at F=f32 on CpuRuntime reproduces
        /// the f32-rounded host result within f32 eps. Proves
        /// `f12_cart_contraction_kernel` is genuinely generic over `F: Float`.
        #[test]
        fn f12_cart_device_generic_f32_within_eps() {
            let (li, lj, lk, ll) = (1u8, 1u8, 0u8, 0u8);
            let shape = build_f12_shape(li as usize, lj as usize, lk as usize, ll as usize);
            let mut rng = Lcg::new(0xDEADBEEFCAFEF00D);
            let g: Vec<f64> = (0..3 * shape.g_size).map(|_| rng.next_f64()).collect();
            let host = contract_f12_cart(&g, shape, li, lj, lk, ll);

            let nfi = ncart(li);
            let nfj = ncart(lj);
            let nfk = ncart(lk);
            let nfl = ncart(ll);
            let out_len = nfi * nfj * nfk * nfl;
            let comps_i = cart_comps_flat_u32(li);
            let comps_j = cart_comps_flat_u32(lj);
            let comps_k = cart_comps_flat_u32(lk);
            let comps_l = cart_comps_flat_u32(ll);

            let client = cpu_client();
            let g_f32: Vec<f32> = g.iter().map(|&v| v as f32).collect();
            let g_h = client.create_from_slice(f32::as_bytes(&g_f32));
            let comps_i_h = client.create_from_slice(u32::as_bytes(&comps_i));
            let comps_j_h = client.create_from_slice(u32::as_bytes(&comps_j));
            let comps_k_h = client.create_from_slice(u32::as_bytes(&comps_k));
            let comps_l_h = client.create_from_slice(u32::as_bytes(&comps_l));
            let out_zero = vec![0.0_f32; out_len];
            let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

            f12_cart_contraction_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(g_h, g_f32.len()) },
                unsafe { ArrayArg::from_raw_parts(comps_i_h, comps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(comps_j_h, comps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(comps_k_h, comps_k.len()) },
                unsafe { ArrayArg::from_raw_parts(comps_l_h, comps_l.len()) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                nfi as u32,
                nfj as u32,
                nfk as u32,
                nfl as u32,
                shape.nroots as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
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
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 23 plan 01: nabla1l_2e (G2E_D_L) unit tests.
    //
    // nabla1l_2e is the mirror of nabla1k_2e operating on the `ll` loop bound and
    // the `dl` stride. It must reproduce the analytic ∂χ_l recurrence on the l axis:
    //   f[l=0]   = -2*al * g[l+1]
    //   f[l>=1]  =  l * g[l-1] + (-2*al) * g[l+1]
    // ─────────────────────────────────────────────────────────────────────────

    /// Fill a single-axis g block (g_size elements) with a distinct, easily-checkable
    /// value per element so stride/offset errors surface immediately.
    fn fill_distinct(g: &mut [f64]) {
        for (idx, v) in g.iter_mut().enumerate() {
            *v = (idx as f64) + 1.0; // 1.0, 2.0, 3.0, ...
        }
    }

    #[test]
    fn nabla1l_2e_ssp_first_order_term() {
        // s-s-s-p base: nabla on a base ll=1. Mirror the real launchers, which build the
        // G-tensor with CEILING angular momenta (ll_ceil = ll+1) so g[+dl] at the top base
        // l-level stays in bounds. Here ll_base=1, so build the shape with ll_ceil=2.
        let ll_base = 1usize;
        let shape = build_f12_shape(0, 0, 0, ll_base + 1);
        let al = 0.75_f64;
        let three_g = 3 * shape.g_size;

        let mut g = vec![0.0_f64; three_g];
        fill_distinct(&mut g);
        let mut f = vec![0.0_f64; three_g];
        nabla1l_2e(&mut f, &g, 0, 0, 0, ll_base, al, &shape);

        let dl = shape.dl;
        let nroots = shape.nroots;
        for axis in 0..3 {
            let off = axis * shape.g_size;
            // l=0 first-order term: f = -2*al * g[+dl]
            for n in 0..nroots {
                let expected = -2.0 * al * g[off + n + dl];
                assert_eq!(
                    f[off + n], expected,
                    "axis {axis} l=0 n={n}: nabla1l first-order term mismatch"
                );
            }
            // l=1: f = 1*g[-dl] + (-2*al)*g[+dl]
            for n in 0..nroots {
                let ptr = dl + n;
                let expected = 1.0 * g[off + ptr - dl] + (-2.0 * al) * g[off + ptr + dl];
                assert_eq!(f[off + ptr], expected, "axis {axis} l=1 n={n}");
            }
        }
    }

    #[test]
    fn nabla1l_2e_matches_analytic_l_recurrence() {
        // base ll=2 with one extra ceiling level (ll_ceil=3) so the full
        // f[l>=1] = l*g[-dl] + (-2*al)*g[+dl] recurrence stays in bounds up to l=2.
        // i/j/k slots held at 0 so only the l axis recurrence is exercised.
        let ll_base = 2usize;
        let shape = build_f12_shape(0, 0, 0, ll_base + 1);
        let al = 1.3_f64;
        let three_g = 3 * shape.g_size;

        let mut g = vec![0.0_f64; three_g];
        fill_distinct(&mut g);
        let mut f = vec![0.0_f64; three_g];
        nabla1l_2e(&mut f, &g, 0, 0, 0, ll_base, al, &shape);

        let dl = shape.dl;
        let nroots = shape.nroots;
        for axis in 0..3 {
            let off = axis * shape.g_size;
            // l=0: f = -2*al*g[+dl]
            for n in 0..nroots {
                let expected = -2.0 * al * g[off + n + dl];
                assert_eq!(f[off + n], expected, "axis {axis} l=0 n={n}");
            }
            // l=1..=2: f = l*g[-dl] + (-2*al)*g[+dl]
            for l in 1..=ll_base {
                for n in 0..nroots {
                    let ptr = dl * l + n;
                    let expected = l as f64 * g[off + ptr - dl] + (-2.0 * al) * g[off + ptr + dl];
                    assert_eq!(f[off + ptr], expected, "axis {axis} l={l} n={n}");
                }
            }
        }
    }

    #[test]
    fn nabla1l_2e_structural_mirror_of_nabla1k_2e() {
        // Structural cross-check: nabla1l on the l axis must mirror nabla1k on the k axis
        // once the strides align. Build both shapes with one extra ceiling level on the
        // active center (lk_ceil=ll_ceil=2 for a base AM of 1) so neither operator reads
        // out of bounds, and the dk (k-shape) and dl (l-shape) strides coincide.
        let base = 1usize;
        let shape_k = build_f12_shape(0, 0, base + 1, 0);
        let shape_l = build_f12_shape(0, 0, 0, base + 1);
        let a = 0.9_f64;

        // For these structurally-equivalent shapes the stride that the active center
        // walks must be identical: dk in the k-shape == dl in the l-shape.
        assert_eq!(shape_k.g_size, shape_l.g_size);
        assert_eq!(shape_k.nroots, shape_l.nroots);
        assert_eq!(shape_k.dk, shape_l.dl, "dk and dl strides must align for the mirror check");

        let mut gk = vec![0.0_f64; 3 * shape_k.g_size];
        let mut gl = vec![0.0_f64; 3 * shape_l.g_size];
        fill_distinct(&mut gk);
        fill_distinct(&mut gl);

        let mut fk = vec![0.0_f64; 3 * shape_k.g_size];
        let mut fl = vec![0.0_f64; 3 * shape_l.g_size];
        nabla1k_2e(&mut fk, &gk, 0, 0, base, 0, a, &shape_k);
        nabla1l_2e(&mut fl, &gl, 0, 0, 0, base, a, &shape_l);

        for (idx, (&vk, &vl)) in fk.iter().zip(fl.iter()).enumerate() {
            assert_eq!(vk, vl, "nabla1l must mirror nabla1k at element {idx}");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 23 plan 01: parameterized single-side contraction (gout_ipn).
    //
    // Regression guard: gout_ipn with center=I must reproduce gout_ip1 bit-for-bit
    // (int2e_ip1, bra-i, Phase 21 D-04 must not regress). Also assert the J/K/L
    // centers run and produce finite output on the same small tensor.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn gout_ipn_center_i_matches_gout_ip1() {
        // s-p-s-s: base li=0, lj=1, lk=0, ll=0. The i center (nabla1i) needs li_ceil=li+1
        // headroom, so build the G-tensor shape with li raised by one.
        let (li, lj, lk, ll) = (0usize, 1usize, 0usize, 0usize);
        let shape = build_f12_shape(li + 1, lj, lk, ll);
        let ai = 0.65_f64;

        let mut g = vec![0.0_f64; 3 * shape.g_size];
        fill_distinct(&mut g);

        let baseline = gout_ip1(&g, &shape, li, lj, lk, ll, ai);
        let viaparam = gout_ipn(&g, &shape, li, lj, lk, ll, Nabla1Center::I, ai);

        assert_eq!(baseline.len(), viaparam.len(), "gout length mismatch");
        for (idx, (&b, &p)) in baseline.iter().zip(viaparam.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                p.to_bits(),
                "gout_ipn(I) must be byte-identical to gout_ip1 at element {idx}: {b} vs {p}"
            );
        }
    }

    #[test]
    fn gout_ipn_other_centers_run() {
        // Exercise each non-I center with the matching ceiling headroom so the nabla
        // stays in bounds, and assert the contraction produces finite output and is
        // not a verbatim copy of the center-I result (different center ⇒ different mix).
        let exponent = 0.8_f64;

        // center J: ket-i derivative needs lj_ceil = lj+1.
        let (li, lj, lk, ll) = (1usize, 0usize, 0usize, 0usize);
        let shape_j = build_f12_shape(li, lj + 1, lk, ll);
        let mut gj = vec![0.0_f64; 3 * shape_j.g_size];
        fill_distinct(&mut gj);
        let out_j = gout_ipn(&gj, &shape_j, li, lj, lk, ll, Nabla1Center::J, exponent);
        assert!(out_j.iter().all(|v| v.is_finite()), "center J output must be finite");

        // center K: needs lk_ceil = lk+1.
        let (li, lj, lk, ll) = (0usize, 0usize, 1usize, 0usize);
        let shape_k = build_f12_shape(li, lj, lk + 1, ll);
        let mut gk = vec![0.0_f64; 3 * shape_k.g_size];
        fill_distinct(&mut gk);
        let out_k = gout_ipn(&gk, &shape_k, li, lj, lk, ll, Nabla1Center::K, exponent);
        assert!(out_k.iter().all(|v| v.is_finite()), "center K output must be finite");

        // center L: needs ll_ceil = ll+1.
        let (li, lj, lk, ll) = (0usize, 0usize, 0usize, 1usize);
        let shape_l = build_f12_shape(li, lj, lk, ll + 1);
        let mut gl = vec![0.0_f64; 3 * shape_l.g_size];
        fill_distinct(&mut gl);
        let out_l = gout_ipn(&gl, &shape_l, li, lj, lk, ll, Nabla1Center::L, exponent);
        assert!(out_l.iter().all(|v| v.is_finite()), "center L output must be finite");

        // At least one component must be non-zero for each (the synthetic tensor is dense).
        assert!(out_j.iter().any(|&v| v != 0.0), "center J output should be non-trivial");
        assert!(out_k.iter().any(|&v| v != 0.0), "center K output should be non-trivial");
        assert!(out_l.iter().any(|&v| v != 0.0), "center L output should be non-trivial");
    }
}
