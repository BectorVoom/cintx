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
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats, validator::validate_f12_env_params};
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
#[derive(Clone, Copy, Debug)]
struct F12Shape {
    nroots: usize,
    nmax: usize,
    mmax: usize,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ibase: bool,
    kbase: bool,
    di: usize,
    dk: usize,
    dl: usize,
    dj: usize,
    g2d_ijmax: usize,
    g2d_klmax: usize,
    g_size: usize,
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

    let (u_roots, mut w_weights) = stg_roots_host(shape.nroots, ta, ua);

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
fn nabla1i_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, ai: f64, shape: &F12Shape) {
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
fn nabla1j_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, aj: f64, shape: &F12Shape) {
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
fn nabla1k_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, ak: f64, shape: &F12Shape) {
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

/// Compute gout for the ip1 variant (ncomp=3): `\nabla_i` on electron 1.
///
/// Matches `CINTgout2e_int2e_ip1` in autocode/grad2.c.
/// Output layout: gout[n*3+comp] for comp in 0..3 (x, y, z).
fn gout_ip1(
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
    // nabla1i at li+0 (base li); g was built with li_ceil = li+1
    nabla1i_2e(&mut g1, g, li, lj, lk, ll, ai, shape);

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

/// Compute gout for the ipip1 variant (ncomp=9): `\nabla_i \nabla_i` on electron 1.
///
/// Matches `CINTgout2e_int2e_ipip1` in autocode/hess.c.
/// CRITICAL: output has column-major reordering of the 3×3 Hessian.
/// Output layout: gout[n*9+{0,1,2,3,4,5,6,7,8}] = {s0,s3,s6,s1,s4,s7,s2,s5,s8}
fn gout_ipip1(
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

/// Compute gout for the ipvip1 variant (ncomp=9): `\nabla_i \nabla_j` on electron 1.
///
/// Matches `CINTgout2e_int2e_ipvip1` in autocode/hess.c.
/// No column-major reordering (unlike ipip1).
fn gout_ipvip1(
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

/// Compute gout for the ip1ip2 variant (ncomp=9): `\nabla_i` on e1 and `\nabla_k` on e2.
///
/// Matches `CINTgout2e_int2e_ip1ip2` in autocode/hess.c.
/// No column-major reordering.
fn gout_ip1ip2(
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

/// Contract [gx|gy|gz] into Cartesian 2e tensor for F12 (identical to two_electron version).
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
    let _ = backend;

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
                        let prim_cart = contract_f12_cart(&g, shape, li_u8, lj_u8, lk_u8, ll_u8);

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
                    let copy_len = (staging.len() - stage_off).min(sph.len()).min(sph_size);
                    if stage_off < staging.len() {
                        staging[stage_off..stage_off + copy_len].copy_from_slice(&sph[..copy_len]);
                    }
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

/// Dispatch to the correct STG or YP entry point based on operator_name.
///
/// Reads `plan.operator_env_params.f12_zeta` (pre-validated by validate_f12_env_params).
/// Routes by operator_name prefix ("stg" or "yp") and suffix ("", "_ip1", "_ipip1", "_ipvip1", "_ip1ip2").
pub fn launch_f12(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    // Validate and extract f12_zeta.
    // canonical_family for STG/YP operators in the manifest is "f12"; we pass
    // "f12" explicitly so validate_f12_env_params performs the zeta gate unconditionally.
    validate_f12_env_params("f12", &plan.operator_env_params)?;

    let zeta = plan
        .operator_env_params
        .f12_zeta
        .expect("validate_f12_env_params guarantees Some non-zero zeta");

    // operator_name() returns "stg", "stg_ip1", "yp", "yp_ip1", etc. (no "int2e_" prefix).
    let operator_name = plan.descriptor.operator_name();

    // Determine STG vs YP and variant suffix from operator_name.
    let (is_stg, variant_suffix) = if let Some(suffix) = operator_name.strip_prefix("stg") {
        (true, suffix)
    } else if let Some(suffix) = operator_name.strip_prefix("yp") {
        (false, suffix)
    } else {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("f12 launch: unrecognized operator_name: {operator_name}"),
        });
    };

    if is_stg {
        match variant_suffix {
            "" => launch_stg_base(backend, plan, specialization, staging, zeta),
            "_ip1" => launch_stg_ip1(backend, plan, specialization, staging, zeta),
            "_ipip1" => launch_stg_ipip1(backend, plan, specialization, staging, zeta),
            "_ipvip1" => launch_stg_ipvip1(backend, plan, specialization, staging, zeta),
            "_ip1ip2" => launch_stg_ip1ip2(backend, plan, specialization, staging, zeta),
            other => Err(cintxRsError::UnsupportedApi {
                requested: format!("f12 launch: unknown stg variant suffix: {other}"),
            }),
        }
    } else {
        match variant_suffix {
            "" => launch_yp_base(backend, plan, specialization, staging, zeta),
            "_ip1" => launch_yp_ip1(backend, plan, specialization, staging, zeta),
            "_ipip1" => launch_yp_ipip1(backend, plan, specialization, staging, zeta),
            "_ipvip1" => launch_yp_ipvip1(backend, plan, specialization, staging, zeta),
            "_ip1ip2" => launch_yp_ip1ip2(backend, plan, specialization, staging, zeta),
            other => Err(cintxRsError::UnsupportedApi {
                requested: format!("f12 launch: unknown yp variant suffix: {other}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let (roots_stg, weights_raw_stg) = stg_roots_host(nroots, ta, ua);
        let (roots_yp, weights_raw_yp) = stg_roots_host(nroots, ta, ua);

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
}
