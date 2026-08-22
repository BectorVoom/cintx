//! `breit` family: Breit spinor-only 2e integrals (`breit.c`).
//!
//! HOST/DEVICE SPLIT (quick-260529-twi, Phase 21 D-04 honest split):
//!
//! ON DEVICE (`breit_g_kernel<F>`, generic over `F: Float`): the per-quartet
//! G-tensor BUILD — `fill_g_tensor_breit` = the 3-axis Rys VRR fill
//! (`vrr_fill_axis_breit`) plus the ibase/kbase-selected 4-branch HRR transfer
//! (`hrr_{lj2d,kj2d,il2d,ik2d}_4d_breit`). The kernel produces the raw `3*g_size`
//! G-tensor block for ONE primitive quartet, exactly the buffer the host
//! `fill_g_tensor_breit` returned. ALL strides (di,dk,dl,dj,g_size,nmax,mmax,
//! g2d_ijmax,g2d_klmax) and the ibase/kbase flags are computed host-side by
//! `build_breit_shape` and passed as runtime u32; the adaptive dli/dlj/dlk/dll
//! branch logic is NOT recomputed on-device (avoids if-expressions). The HRR
//! branch is selected at runtime by `if kbase==1u32` / `if ibase==1u32`
//! STATEMENTS (model: `two_electron_scalar_kernel`'s 4-branch device HRR).
//! `#[comptime] nroots` selects `rys_root{1..5}`. `fac_env` (the per-quartet
//! `common_factor * pdata_ij.fac * pdata_kl.fac`, with the Gaussian-product
//! overlap exponentials) is computed host-side via `compute_pdata_host` and
//! passed in as a scalar `F`, matching the host part of the split.
//!
//! DEFERRED TO HOST (documented per the 1e/2e spinor-deferral precedent; these
//! are large separable ports left for a follow-up):
//!   (a) the Breit-specific gout operator ladder applied AFTER the G-tensor —
//!       `gout_breit_r1p2` / `gout_breit_r2p2` and their `nabla1{i,j,l}_breit` +
//!       `x1{j,l}_breit` derivative/position operators. These walk the elevated
//!       G-tensor with many intermediate `3*g_size` buffers and a 9-term Rys
//!       contraction; porting them is a self-contained follow-up.
//!   (b) the `cart_to_spinor_sf_4d` transform — Breit is spinor-only (D-07); the
//!       spinor coefficient table is host-only and the documented KET-major →
//!       BRA-major transpose gotcha stays host-side, exactly as the 1e spinor
//!       ports kept their c2spinor host (project memory: 1e spinor orientation).
//!
//! Split out of the original single-file `unstable.rs`; the host helper bodies
//! below are move-only (unchanged).

use super::shared::{SQRTPI, cart_comps, common_fac_sp};
use crate::backend::ResolvedBackend;
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::rys_roots_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::ncart;
use crate::transform::c2spinor::cart_to_spinor_sf_4d;
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate.
/// Breit `nroots = (li_e+lj_e+lk_e+ll_e)/2 + 1` over the ELEVATED momenta, so this
/// fail-closes any quartet whose elevated l-sum would exceed the rys_root1..5 ceiling.
const MAX_DEVICE_NROOTS: usize = 5;

/// Shape parameters for the Breit g-tensor, built from elevated angular momenta.
///
/// Breit integrals use elevated dims for g-tensor construction (like derivative 2e integrals)
/// but contract at the base (final) angular momenta.
#[derive(Clone, Copy, Debug)]
struct BreitShape {
    nroots: usize,
    nmax: usize,
    mmax: usize,
    /// Elevated li for g-tensor construction (li_base + IINC)
    li_elev: usize,
    /// Elevated lj for g-tensor construction (lj_base + JINC)
    lj_elev: usize,
    /// Elevated lk for g-tensor construction (lk_base + KINC)
    lk_elev: usize,
    /// Elevated ll for g-tensor construction (ll_base + LINC)
    ll_elev: usize,
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

/// Build the Breit shape from elevated angular momenta.
///
/// For breit_r1p2: ng = {2, 2, 0, 1, ...}
///   li_elev = li + 2, lj_elev = lj + 2, lk_elev = lk + 0, ll_elev = ll + 1
/// For breit_r2p2: ng = {2, 1, 0, 2, ...}
///   li_elev = li + 2, lj_elev = lj + 1, lk_elev = lk + 0, ll_elev = ll + 2
fn build_breit_shape(li_e: usize, lj_e: usize, lk_e: usize, ll_e: usize) -> BreitShape {
    let nroots = (li_e + lj_e + lk_e + ll_e) / 2 + 1;
    let nmax = li_e + lj_e;
    let mmax = lk_e + ll_e;

    let ibase = li_e > lj_e;
    let kbase = lk_e > ll_e;

    let (dli, dlj) = if ibase {
        (li_e + lj_e + 1, lj_e + 1)
    } else {
        (li_e + 1, li_e + lj_e + 1)
    };
    let (dlk, dll) = if kbase {
        (lk_e + ll_e + 1, ll_e + 1)
    } else {
        (lk_e + 1, lk_e + ll_e + 1)
    };

    let di = nroots;
    let dk = nroots * dli;
    let dl = nroots * dli * dlk;
    let dj = nroots * dli * dlk * dll;
    let g_size = nroots * dli * dlk * dll * dlj;

    let g2d_ijmax = if ibase { di } else { dj };
    let g2d_klmax = if kbase { dk } else { dl };

    BreitShape {
        nroots,
        nmax,
        mmax,
        li_elev: li_e,
        lj_elev: lj_e,
        lk_elev: lk_e,
        ll_elev: ll_e,
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

// ─────────────────────────────────────────────────────────────────────────────
// VRR fill and HRR transfer (same as two_electron.rs but operating on BreitShape)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn vrr_fill_axis_breit(
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

fn hrr_lj2d_4d_breit(g: &mut [f64], shape: BreitShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.li_elev == 0 && shape.lk_elev == 0 {
        return;
    }
    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rirj[axis];
        for i in 1..=shape.li_elev {
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
        for j in 0..=shape.lj_elev {
            for k in 1..=shape.lk_elev {
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

fn hrr_kj2d_4d_breit(g: &mut [f64], shape: BreitShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.li_elev == 0 && shape.ll_elev == 0 {
        return;
    }
    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rirj[axis];
        for i in 1..=shape.li_elev {
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
        for j in 0..=shape.lj_elev {
            for l in 1..=shape.ll_elev {
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

fn hrr_il2d_4d_breit(g: &mut [f64], shape: BreitShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.lj_elev == 0 && shape.lk_elev == 0 {
        return;
    }
    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rkrl[axis];
        for k in 1..=shape.lk_elev {
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
        for j in 1..=shape.lj_elev {
            for l in 0..=shape.ll_elev {
                for k in 0..=shape.lk_elev {
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

fn hrr_ik2d_4d_breit(g: &mut [f64], shape: BreitShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.lj_elev == 0 && shape.ll_elev == 0 {
        return;
    }
    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rkrl[axis];
        for l in 1..=shape.ll_elev {
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
        for j in 1..=shape.lj_elev {
            for l in 0..=shape.ll_elev {
                for k in 0..=shape.lk_elev {
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

/// Fill the full [gx|gy|gz] tensor for one Breit primitive quartet.
///
/// Uses Rys quadrature (same as 2e), but with elevated angular momenta to
/// provide headroom for the derivative operators in the Breit gout functions.
#[allow(clippy::too_many_arguments)]
fn fill_g_tensor_breit(
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
    ri: &[f64; 3],
    rj: &[f64; 3],
    rk: &[f64; 3],
    rl: &[f64; 3],
    shape: BreitShape,
    fac_env: f64,
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
    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_env;
    let x_rys = a0 * rr;

    let (u_roots, mut w_weights) = rys_roots_host(shape.nroots, x_rys);
    for w in &mut w_weights {
        *w *= fac1;
    }

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
        vrr_fill_axis_breit(
            gx,
            irys,
            shape.nmax,
            shape.mmax,
            shape.g2d_ijmax,
            shape.g2d_klmax,
            c00[0],
            c0p[0],
            b10,
            b01,
            b00,
        );
        vrr_fill_axis_breit(
            gy,
            irys,
            shape.nmax,
            shape.mmax,
            shape.g2d_ijmax,
            shape.g2d_klmax,
            c00[1],
            c0p[1],
            b10,
            b01,
            b00,
        );
        vrr_fill_axis_breit(
            gz,
            irys,
            shape.nmax,
            shape.mmax,
            shape.g2d_ijmax,
            shape.g2d_klmax,
            c00[2],
            c0p[2],
            b10,
            b01,
            b00,
        );
    }

    if shape.kbase {
        if shape.ibase {
            hrr_ik2d_4d_breit(&mut g, shape, rirj, rkrl);
        } else {
            hrr_kj2d_4d_breit(&mut g, shape, rirj, rkrl);
        }
    } else if shape.ibase {
        hrr_il2d_4d_breit(&mut g, shape, rirj, rkrl);
    } else {
        hrr_lj2d_4d_breit(&mut g, shape, rirj, rkrl);
    }

    g
}

// ─────────────────────────────────────────────────────────────────────────────
//  Breit G-tensor device kernel — `#[cube(launch)]`, generic over `F: Float`
//
//  Faithful inline port of the host `fill_g_tensor_breit` for ONE primitive
//  quartet: the 3-axis Rys VRR fill (`vrr_fill_axis_breit`) + the ibase/kbase
//  4-branch HRR transfer (`hrr_{ik2d,kj2d,il2d,lj2d}_4d_breit`). Output is the
//  raw `3*g_size` G-tensor block (gx | gy | gz), identical to what the host
//  `fill_g_tensor_breit` returns; the host gout/nabla/x1 + spinor ladder consumes
//  it unchanged. No primitive/contraction loop on-device (breit's gout runs
//  per-primitive on host, so the device builds exactly one quartet's G-tensor).
//
//  All BreitShape strides + ibase/kbase come in as runtime u32. `fac_env` (the
//  per-quartet `common_factor * pdata_ij.fac * pdata_kl.fac`) is host-computed
//  (compute_pdata_host) and passed as a scalar F. `#[comptime] nroots` selects
//  rys_root{1..5}. The Rys weight scaling `fac1 = sqrt(a0/a1^3) * fac_env` and the
//  c00/c0p/b00/b10/b01 recurrence coefficients reproduce the host code exactly.
// ─────────────────────────────────────────────────────────────────────────────

/// Single-work-item Breit G-tensor kernel. See module note above.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn breit_g_kernel<F: Float + CubeElement>(
    g: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    ai: F,
    aj: F,
    ak: F,
    al: F,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    rkx: F,
    rky: F,
    rkz: F,
    rlx: F,
    rly: F,
    rlz: F,
    fac_env: F,
    pie4: F,
    li_e: u32,
    lj_e: u32,
    lk_e: u32,
    ll_e: u32,
    di: u32,
    dk: u32,
    dl: u32,
    dj: u32,
    g_size: u32,
    nmax: u32,
    mmax: u32,
    g2d_ijmax: u32,
    g2d_klmax: u32,
    ibase: u32,
    kbase: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let nrys = nroots;
        let gy_off = g_size;
        let gz_off = 2u32 * g_size;
        let total_g = 3u32 * g_size;

        let aij = ai + aj;
        let akl = ak + al;

        // Gaussian-product centers (host: rij / rkl).
        let rijx = (ai * rix + aj * rjx) / aij;
        let rijy = (ai * riy + aj * rjy) / aij;
        let rijz = (ai * riz + aj * rjz) / aij;
        let rklx = (ak * rkx + al * rlx) / akl;
        let rkly = (ak * rky + al * rly) / akl;
        let rklz = (ak * rkz + al * rlz) / akl;

        let xij_kl = rijx - rklx;
        let yij_kl = rijy - rkly;
        let zij_kl = rijz - rklz;
        let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

        let a1 = aij * akl;
        let a0 = a1 / (aij + akl);
        let x_rys = a0 * rr;

        // Rys roots/weights (comptime nroots branch).
        if comptime!(nroots == 1u32) {
            rys_root1::<F>(x_rys, urys, wrys, pie4);
        } else if comptime!(nroots == 2u32) {
            rys_root2::<F>(x_rys, urys, wrys, pie4);
        } else if comptime!(nroots == 3u32) {
            rys_root3::<F>(x_rys, urys, wrys, pie4);
        } else if comptime!(nroots == 4u32) {
            rys_root4::<F>(x_rys, urys, wrys, pie4);
        } else {
            rys_root5::<F>(x_rys, urys, wrys, pie4);
        }

        // fac1 = sqrt(a0/(a1^3)) * fac_env (host: fac1 then w_weights *= fac1).
        let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * fac_env;

        // ibase/kbase-selected reference centers (rijrx / rklrx) and the HRR
        // displacement vectors rirj / rkrl (host: the if shape.ibase / if shape.kbase).
        let mut rx_ij_x = rjx;
        let mut rx_ij_y = rjy;
        let mut rx_ij_z = rjz;
        let mut rirjx = rjx - rix;
        let mut rirjy = rjy - riy;
        let mut rirjz = rjz - riz;
        if ibase == 1u32 {
            rx_ij_x = rix;
            rx_ij_y = riy;
            rx_ij_z = riz;
            rirjx = rix - rjx;
            rirjy = riy - rjy;
            rirjz = riz - rjz;
        }
        let mut rx_kl_x = rlx;
        let mut rx_kl_y = rly;
        let mut rx_kl_z = rlz;
        let mut rkrlx = rlx - rkx;
        let mut rkrly = rly - rky;
        let mut rkrlz = rlz - rkz;
        if kbase == 1u32 {
            rx_kl_x = rkx;
            rx_kl_y = rky;
            rx_kl_z = rkz;
            rkrlx = rkx - rlx;
            rkrly = rky - rly;
            rkrlz = rkz - rlz;
        }

        let rijrxx = rijx - rx_ij_x;
        let rijrxy = rijy - rx_ij_y;
        let rijrxz = rijz - rx_ij_z;
        let rklrxx = rklx - rx_kl_x;
        let rklrxy = rkly - rx_kl_y;
        let rklrxz = rklz - rx_kl_z;

        // ── Initialize the [gx|gy|gz] tensor (host: zero then seed). ──────────
        let mut gi = 0u32;
        while gi < total_g {
            g[gi as usize] = F::new(0.0);
            gi += 1u32;
        }
        let mut irys = 0u32;
        while irys < nrys {
            g[irys as usize] = F::new(1.0);
            g[(gy_off + irys) as usize] = F::new(1.0);
            g[(gz_off + irys) as usize] = wrys[irys as usize] * fac1;
            irys += 1u32;
        }

        // ── VRR per-axis fill (inline vrr_fill_axis_breit). ───────────────────
        let mut irys2 = 0u32;
        while irys2 < nrys {
            let u2 = a0 * urys[irys2 as usize];
            let tmp4 = F::new(0.5) / (u2 * (aij + akl) + a1);
            let tmp5 = u2 * tmp4;
            let tmp1 = F::new(2.0) * tmp5;
            let tmp2 = tmp1 * akl;
            let tmp3 = tmp1 * aij;
            let b00 = tmp5;
            let b10 = tmp5 + tmp4 * akl;
            let b01 = tmp5 + tmp4 * aij;

            let mut axis = 0u32;
            while axis < 3u32 {
                let off = axis * g_size;
                let mut xkl = xij_kl;
                let mut rijrx = rijrxx;
                let mut rklrx = rklrxx;
                if axis == 1u32 {
                    xkl = yij_kl;
                    rijrx = rijrxy;
                    rklrx = rklrxy;
                }
                if axis == 2u32 {
                    xkl = zij_kl;
                    rijrx = rijrxz;
                    rklrx = rklrxz;
                }
                let c00 = rijrx - tmp2 * xkl;
                let c0p = rklrx + tmp3 * xkl;

                // Inline vrr_fill_axis_breit(g[off..], irys2, nmax, mmax,
                //   dn=g2d_ijmax, dm=g2d_klmax, c00, c0p, b10, b01, b00).
                let root = irys2;
                let dn = g2d_ijmax;
                let dm = g2d_klmax;

                if nmax > 0u32 {
                    let mut s0 = g[(off + root) as usize];
                    let mut s1 = c00 * s0;
                    g[(off + root + dn) as usize] = s1;
                    let mut n = 1u32;
                    while n < nmax {
                        let s2 = c00 * s1 + F::cast_from(n) * b10 * s0;
                        g[(off + root + (n + 1u32) * dn) as usize] = s2;
                        s0 = s1;
                        s1 = s2;
                        n += 1u32;
                    }
                }

                if mmax > 0u32 {
                    let mut s0 = g[(off + root) as usize];
                    let mut s1 = c0p * s0;
                    g[(off + root + dm) as usize] = s1;
                    let mut m = 1u32;
                    while m < mmax {
                        let s2 = c0p * s1 + F::cast_from(m) * b01 * s0;
                        g[(off + root + (m + 1u32) * dm) as usize] = s2;
                        s0 = s1;
                        s1 = s2;
                        m += 1u32;
                    }

                    if nmax > 0u32 {
                        let mut s0n = g[(off + root + dn) as usize];
                        let mut s1n = c0p * s0n + b00 * g[(off + root) as usize];
                        g[(off + root + dn + dm) as usize] = s1n;
                        let mut m2 = 1u32;
                        while m2 < mmax {
                            let s2n = c0p * s1n
                                + F::cast_from(m2) * b01 * s0n
                                + b00 * g[(off + root + m2 * dm) as usize];
                            g[(off + root + dn + (m2 + 1u32) * dm) as usize] = s2n;
                            s0n = s1n;
                            s1n = s2n;
                            m2 += 1u32;
                        }
                    }
                }

                if nmax > 0u32 {
                    let mut m3 = 1u32;
                    while m3 <= mmax {
                        let offm = m3 * dm;
                        let jbase = offm + root;
                        let mut s0 = g[(off + jbase) as usize];
                        let mut s1 = g[(off + jbase + dn) as usize];
                        let mut n2 = 1u32;
                        while n2 < nmax {
                            let s2 = c00 * s1
                                + F::cast_from(n2) * b10 * s0
                                + F::cast_from(m3) * b00 * g[(off + jbase + n2 * dn - dm) as usize];
                            g[(off + jbase + (n2 + 1u32) * dn) as usize] = s2;
                            s0 = s1;
                            s1 = s2;
                            n2 += 1u32;
                        }
                        m3 += 1u32;
                    }
                }

                axis += 1u32;
            }
            irys2 += 1u32;
        }

        // ── HRR transfer (4 branches by kbase/ibase). ─────────────────────────
        // Mirrors host fill_g_tensor_breit: kbase ? (ibase ? ik2d : kj2d)
        //                                          : (ibase ? il2d : lj2d).
        let mut axis2 = 0u32;
        while axis2 < 3u32 {
            let off = axis2 * g_size;
            let mut rirj = rirjx;
            let mut rkrl = rkrlx;
            if axis2 == 1u32 {
                rirj = rirjy;
                rkrl = rkrly;
            }
            if axis2 == 2u32 {
                rirj = rirjz;
                rkrl = rkrlz;
            }

            if kbase == 1u32 {
                if ibase == 1u32 {
                    // hrr_ik2d_4d_breit: ll-loop (dl←dk), then j-loop (dj←di).
                    let mut l = 1u32;
                    while l <= ll_e {
                        let mut k = 0u32;
                        while k <= (mmax - l) {
                            let mut i = 0u32;
                            while i <= nmax {
                                let ptr = l * dl + k * dk + i * di;
                                let mut r = 0u32;
                                while r < nrys {
                                    let idx = ptr + r;
                                    g[(off + idx) as usize] = rkrl * g[(off + idx - dl) as usize]
                                        + g[(off + idx - dl + dk) as usize];
                                    r += 1u32;
                                }
                                i += 1u32;
                            }
                            k += 1u32;
                        }
                        l += 1u32;
                    }
                    let mut j = 1u32;
                    while j <= lj_e {
                        let mut l2 = 0u32;
                        while l2 <= ll_e {
                            let mut k2 = 0u32;
                            while k2 <= lk_e {
                                let ptr = j * dj + l2 * dl + k2 * dk;
                                let mut i2 = 0u32;
                                while i2 <= (nmax - j) {
                                    let pbase = ptr + i2 * di;
                                    let mut r = 0u32;
                                    while r < nrys {
                                        let idx = pbase + r;
                                        g[(off + idx) as usize] = rirj
                                            * g[(off + idx - dj) as usize]
                                            + g[(off + idx - dj + di) as usize];
                                        r += 1u32;
                                    }
                                    i2 += 1u32;
                                }
                                k2 += 1u32;
                            }
                            l2 += 1u32;
                        }
                        j += 1u32;
                    }
                } else {
                    // hrr_kj2d_4d_breit: i-loop (dj←di), then l-loop (dl←dk).
                    let mut i = 1u32;
                    while i <= li_e {
                        let mut j = 0u32;
                        while j <= (nmax - i) {
                            let mut k = 0u32;
                            while k <= mmax {
                                let ptr = j * dj + k * dk + i * di;
                                let mut r = 0u32;
                                while r < nrys {
                                    let idx = ptr + r;
                                    g[(off + idx) as usize] = rirj * g[(off + idx - di) as usize]
                                        + g[(off + idx - di + dj) as usize];
                                    r += 1u32;
                                }
                                k += 1u32;
                            }
                            j += 1u32;
                        }
                        i += 1u32;
                    }
                    let mut j2 = 0u32;
                    while j2 <= lj_e {
                        let mut l = 1u32;
                        while l <= ll_e {
                            let mut k = 0u32;
                            while k <= (mmax - l) {
                                let ptr = j2 * dj + l * dl + k * dk;
                                let mut n = 0u32;
                                while n < dk {
                                    let idx = ptr + n;
                                    g[(off + idx) as usize] = rkrl * g[(off + idx - dl) as usize]
                                        + g[(off + idx - dl + dk) as usize];
                                    n += 1u32;
                                }
                                k += 1u32;
                            }
                            l += 1u32;
                        }
                        j2 += 1u32;
                    }
                }
            } else if ibase == 1u32 {
                // hrr_il2d_4d_breit: k-loop (dk←dl), then j-loop (dj←di).
                let mut k = 1u32;
                while k <= lk_e {
                    let mut l = 0u32;
                    while l <= (mmax - k) {
                        let mut i = 0u32;
                        while i <= nmax {
                            let ptr = l * dl + k * dk + i * di;
                            let mut r = 0u32;
                            while r < nrys {
                                let idx = ptr + r;
                                g[(off + idx) as usize] = rkrl * g[(off + idx - dk) as usize]
                                    + g[(off + idx - dk + dl) as usize];
                                r += 1u32;
                            }
                            i += 1u32;
                        }
                        l += 1u32;
                    }
                    k += 1u32;
                }
                let mut j = 1u32;
                while j <= lj_e {
                    let mut l = 0u32;
                    while l <= ll_e {
                        let mut k2 = 0u32;
                        while k2 <= lk_e {
                            let ptr = j * dj + l * dl + k2 * dk;
                            let mut i2 = 0u32;
                            while i2 <= (nmax - j) {
                                let pbase = ptr + i2 * di;
                                let mut r = 0u32;
                                while r < nrys {
                                    let idx = pbase + r;
                                    g[(off + idx) as usize] = rirj * g[(off + idx - dj) as usize]
                                        + g[(off + idx - dj + di) as usize];
                                    r += 1u32;
                                }
                                i2 += 1u32;
                            }
                            k2 += 1u32;
                        }
                        l += 1u32;
                    }
                    j += 1u32;
                }
            } else {
                // hrr_lj2d_4d_breit: i-loop (dj←di), then k-loop (dl←dk).
                let mut i = 1u32;
                while i <= li_e {
                    let mut j = 0u32;
                    while j <= (nmax - i) {
                        let mut l = 0u32;
                        while l <= mmax {
                            let ptr = j * dj + l * dl + i * di;
                            let mut r = 0u32;
                            while r < nrys {
                                let idx = ptr + r;
                                g[(off + idx) as usize] = rirj * g[(off + idx - di) as usize]
                                    + g[(off + idx - di + dj) as usize];
                                r += 1u32;
                            }
                            l += 1u32;
                        }
                        j += 1u32;
                    }
                    i += 1u32;
                }
                let mut j2 = 0u32;
                while j2 <= lj_e {
                    let mut k = 1u32;
                    while k <= lk_e {
                        let mut l = 0u32;
                        while l <= (mmax - k) {
                            let ptr = j2 * dj + l * dl + k * dk;
                            let mut n = 0u32;
                            while n < dk {
                                let idx = ptr + n;
                                g[(off + idx) as usize] = rkrl * g[(off + idx - dk) as usize]
                                    + g[(off + idx - dk + dl) as usize];
                                n += 1u32;
                            }
                            l += 1u32;
                        }
                        k += 1u32;
                    }
                    j2 += 1u32;
                }
            }

            axis2 += 1u32;
        }
    }
}

/// Dispatch [`breit_g_kernel`] at `f64` on a resolved backend client and read back
/// the per-quartet `3*g_size` G-tensor block.
#[allow(clippy::too_many_arguments)]
fn run_breit_g_device<R: Runtime>(
    client: &ComputeClient<R>,
    li_e: u32,
    lj_e: u32,
    lk_e: u32,
    ll_e: u32,
    di: u32,
    dk: u32,
    dl: u32,
    dj: u32,
    g_size: u32,
    nmax: u32,
    mmax: u32,
    g2d_ijmax: u32,
    g2d_klmax: u32,
    ibase: u32,
    kbase: u32,
    nroots: u32,
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    fac_env: f64,
) -> Vec<f64> {
    let nroots_u = nroots as usize;
    let g_size_u = g_size as usize;

    let g_zero = vec![0.0_f64; 3 * g_size_u];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));

    breit_g_kernel::launch::<f64, R>(
        client,
        crate::plane::single_cube_count(),
        crate::plane::standard_plane_cube_dim(),
        unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3 * g_size_u) },
        unsafe { ArrayArg::from_raw_parts(u_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(w_h, nroots_u) },
        ai,
        aj,
        ak,
        al,
        ri[0],
        ri[1],
        ri[2],
        rj[0],
        rj[1],
        rj[2],
        rk[0],
        rk[1],
        rk[2],
        rl[0],
        rl[1],
        rl[2],
        fac_env,
        PIE4,
        li_e,
        lj_e,
        lk_e,
        ll_e,
        di,
        dk,
        dl,
        dj,
        g_size,
        nmax,
        mmax,
        g2d_ijmax,
        g2d_klmax,
        ibase,
        kbase,
        nroots,
    );

    let raw = client.read_one_unchecked(g_h);
    f64::from_bytes(&raw)[0..3 * g_size_u].to_vec()
}

/// 5-arm backend dispatch for the Breit G-tensor device kernel
/// (Cpu / Wgpu / Cuda / ROCm-HIP / Metal). Returns the `3*g_size` f64 G-tensor
/// block for one primitive quartet (Rocm → `cubecl_hip::HipRuntime`).
#[allow(clippy::too_many_arguments)]
fn run_breit_g_on_backend(
    backend: &ResolvedBackend,
    shape: BreitShape,
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    fac_env: f64,
) -> Vec<f64> {
    let li_e = shape.li_elev as u32;
    let lj_e = shape.lj_elev as u32;
    let lk_e = shape.lk_elev as u32;
    let ll_e = shape.ll_elev as u32;
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_breit_g_device::<cubecl::cpu::CpuRuntime>(
            client,
            li_e,
            lj_e,
            lk_e,
            ll_e,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
            ai,
            aj,
            ak,
            al,
            ri,
            rj,
            rk,
            rl,
            fac_env,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_breit_g_device::<cubecl_wgpu::WgpuRuntime>(
            client,
            li_e,
            lj_e,
            lk_e,
            ll_e,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
            ai,
            aj,
            ak,
            al,
            ri,
            rj,
            rk,
            rl,
            fac_env,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_breit_g_device::<cubecl_cuda::CudaRuntime>(
            client,
            li_e,
            lj_e,
            lk_e,
            ll_e,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
            ai,
            aj,
            ak,
            al,
            ri,
            rj,
            rk,
            rl,
            fac_env,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_breit_g_device::<cubecl_hip::HipRuntime>(
            client,
            li_e,
            lj_e,
            lk_e,
            ll_e,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
            ai,
            aj,
            ak,
            al,
            ri,
            rj,
            rk,
            rl,
            fac_env,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_breit_g_device::<cubecl_wgpu::WgpuRuntime>(
            client,
            li_e,
            lj_e,
            lk_e,
            ll_e,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
            ai,
            aj,
            ak,
            al,
            ri,
            rj,
            rk,
            rl,
            fac_env,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// G-tensor operator functions for Breit gout
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `\nabla_i` to the g-tensor.
/// Corresponds to `CINTnabla1i_2e` in libcint/g2e.c (G2E_D_I macro).
///
/// Formula (per axis):
///   f[n @ i=0] = -2*ai * g[n+di]
///   f[n @ i>=1] = i * g[n-di] + (-2*ai) * g[n+di]
fn nabla1i_breit(
    f: &mut [f64],
    g: &[f64],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    shape: &BreitShape,
) {
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
                    let ptr = dj * j + dl * l + dk * k;
                    for n in ptr..ptr + nroots {
                        f[off + n] = ai2 * g[off + n + di];
                    }
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

/// Apply `\nabla_j` to the g-tensor.
/// Corresponds to `CINTnabla1j_2e` in libcint/g2e.c (G2E_D_J macro).
///
/// Formula (per axis):
///   f[n @ j=0] = -2*aj * g[n+dj]
///   f[n @ j>=1] = j * g[n-dj] + (-2*aj) * g[n+dj]
fn nabla1j_breit(
    f: &mut [f64],
    g: &[f64],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    aj: f64,
    shape: &BreitShape,
) {
    let aj2 = -2.0 * aj;
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
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

/// Apply `\nabla_l` to the g-tensor.
/// Corresponds to `CINTnabla1l_2e` in libcint/g2e.c (G2E_D_L macro).
///
/// Formula (per axis):
///   f[n @ l=0] = -2*al * g[n+dl]
///   f[n @ l>=1] = l * g[n-dl] + (-2*al) * g[n+dl]
fn nabla1l_breit(
    f: &mut [f64],
    g: &[f64],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    al: f64,
    shape: &BreitShape,
) {
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
            // l=0
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

/// Apply the position-r_j operator: `f = g[n+dj] + rj[axis]*g[n]`.
/// Corresponds to `CINTx1j_2e` in libcint/g2e.c (G2E_R0J macro).
///
/// Formula (per axis):
///   f[n @ j=0..lj] = g[n+dj] + rj[axis]*g[n]
fn x1j_breit(
    f: &mut [f64],
    g: &[f64],
    rj: &[f64; 3],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    shape: &BreitShape,
) {
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        let rja = rj[axis];
        for j in 0..=lj {
            for l in 0..=ll {
                for k in 0..=lk {
                    let base = dj * j + dl * l + dk * k;
                    for i in 0..=li {
                        let ptr = base + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = g[off + n + dj] + rja * g[off + n];
                        }
                    }
                }
            }
        }
    }
}

/// Apply the position-r_l operator: `f = g[n+dl] + rl[axis]*g[n]`.
/// Corresponds to `CINTx1l_2e` in libcint/g2e.c (G2E_R0L macro).
///
/// Formula (per axis):
///   f[n @ l=0..ll] = g[n+dl] + rl[axis]*g[n]
fn x1l_breit(
    f: &mut [f64],
    g: &[f64],
    rl: &[f64; 3],
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    shape: &BreitShape,
) {
    let g_size = shape.g_size;
    let nroots = shape.nroots;
    let di = shape.di;
    let dj = shape.dj;
    let dk = shape.dk;
    let dl = shape.dl;

    for axis in 0..3 {
        let off = axis * g_size;
        let rla = rl[axis];
        for j in 0..=lj {
            for l in 0..=ll {
                for k in 0..=lk {
                    let base = dj * j + dl * l + dk * k;
                    for i in 0..=li {
                        let ptr = base + di * i;
                        for n in ptr..ptr + nroots {
                            f[off + n] = g[off + n + dl] + rla * g[off + n];
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Breit gout contractions
// ─────────────────────────────────────────────────────────────────────────────

/// Contract the g-tensor for `int2e_breit_r1p2_spinor` (breit.c CINTgout2e_int2e_breit_r1p2).
///
/// This is a 1-component gout using ng = {2, 2, 0, 1, 4, 1, 1, 1}.
/// The g-tensor is built with li+2, lj+2, lk+0, ll+1 angular momenta.
///
/// Operator sequence (per breit.c lines ~155–185):
///   g1  = G2E_D_L(g0, li+2, lj+2, lk, ll)       → nabla_l at elevated dims
///   g3  = G2E_R0J(g1, li+1, lj, lk, ll)          → r_j position on g1
///   g4  = G2E_D_J(g0, li+1, lj+1, lk, ll)        → nabla_j on g0
///        + G2E_D_I(g0, li+1, lj+1, lk, ll)        → nabla_i on g0 (sum)
///   g5  = G2E_D_J(g1, li+1, lj+1, lk, ll)        → nabla_j on g1
///        + G2E_D_I(g1, li+1, lj+1, lk, ll)        → nabla_i on g1 (sum)
///   g7  = G2E_R0J(g5, li+1, lj, lk, ll)          → r_j position on g5
///   g12 = G2E_D_I(g4, li, lj, lk, ll)            → nabla_i on g4
///   g15 = G2E_D_I(g7, li, lj, lk, ll)            → nabla_i on g7
///
/// Contraction sum (9 terms): g15*g0*g0 + g12*g3*g0 + g12*g0*g3
///                           + g3*g12*g0 + g0*g15*g0 + g0*g12*g3
///                           + g3*g0*g12 + g0*g3*g12 + g0*g0*g15
fn gout_breit_r1p2(
    g: &[f64],
    shape: &BreitShape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    aj: f64,
    al: f64,
    rj: &[f64; 3],
) -> Vec<f64> {
    let g_size = shape.g_size;
    let nroots = shape.nroots;

    // g1 = nabla_l(g0) at (li+2, lj+2, lk, ll)
    let mut g1 = vec![0.0_f64; 3 * g_size];
    nabla1l_breit(&mut g1, g, li + 2, lj + 2, lk, ll, al, shape);

    // g3 = x1j(g1, rj) at (li+1, lj+0, lk, ll)
    let mut g3 = vec![0.0_f64; 3 * g_size];
    x1j_breit(&mut g3, &g1, rj, li + 1, lj, lk, ll, shape);

    // g4 = nabla_j(g0) at (li+1, lj+1, lk, ll) + nabla_i(g0) at (li+1, lj+1, lk, ll)
    let mut g4 = vec![0.0_f64; 3 * g_size];
    let mut g5_tmp = vec![0.0_f64; 3 * g_size];
    nabla1j_breit(&mut g4, g, li + 1, lj + 1, lk, ll, aj, shape);
    nabla1i_breit(&mut g5_tmp, g, li + 1, lj + 1, lk, ll, ai, shape);
    for ix in 0..3 * g_size {
        g4[ix] += g5_tmp[ix];
    }

    // g5 = nabla_j(g1) at (li+1, lj+1, lk, ll) + nabla_i(g1) at (li+1, lj+1, lk, ll)
    let mut g5 = vec![0.0_f64; 3 * g_size];
    let mut g6 = vec![0.0_f64; 3 * g_size];
    nabla1j_breit(&mut g5, &g1, li + 1, lj + 1, lk, ll, aj, shape);
    nabla1i_breit(&mut g6, &g1, li + 1, lj + 1, lk, ll, ai, shape);
    for ix in 0..3 * g_size {
        g5[ix] += g6[ix];
    }

    // g7 = x1j(g5, rj) at (li+1, lj+0, lk, ll)
    let mut g7 = vec![0.0_f64; 3 * g_size];
    x1j_breit(&mut g7, &g5, rj, li + 1, lj, lk, ll, shape);

    // g12 = nabla_i(g4) at (li, lj, lk, ll)
    let mut g12 = vec![0.0_f64; 3 * g_size];
    nabla1i_breit(&mut g12, &g4, li, lj, lk, ll, ai, shape);

    // g15 = nabla_i(g7) at (li, lj, lk, ll)
    let mut g15 = vec![0.0_f64; 3 * g_size];
    nabla1i_breit(&mut g15, &g7, li, lj, lk, ll, ai, shape);

    // Contract: output has nfi * nfj * nfk * nfl elements (1-component)
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; nfi * nfj * nfk * nfl];

    for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
        for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let x_idx = ix as usize * shape.di
                        + kx as usize * shape.dk
                        + lx as usize * shape.dl
                        + jx as usize * shape.dj;
                    let y_idx = iy as usize * shape.di
                        + ky as usize * shape.dk
                        + ly as usize * shape.dl
                        + jy as usize * shape.dj;
                    let z_idx = iz as usize * shape.di
                        + kz as usize * shape.dk
                        + lz as usize * shape.dl
                        + jz as usize * shape.dj;

                    let mut s = 0.0_f64;
                    for irys in 0..nroots {
                        // 9-term contraction per breit.c CINTgout2e_int2e_breit_r1p2
                        s += g15[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys]
                            * g3[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys]
                            * g12[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g15[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g12[gy_off + y_idx + irys]
                            * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g3[gy_off + y_idx + irys]
                            * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g15[gz_off + z_idx + irys];
                    }
                    let out_idx = i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                    out[out_idx] = s;
                }
            }
        }
    }

    out
}

/// Contract the g-tensor for `int2e_breit_r2p2_spinor` (breit.c CINTgout2e_int2e_breit_r2p2).
///
/// This is a 1-component gout using ng = {2, 1, 0, 2, 4, 1, 1, 1}.
/// The g-tensor is built with li+2, lj+1, lk+0, ll+2 angular momenta.
///
/// Operator sequence (per breit.c lines ~233–265):
///   g2  = G2E_R0L(g0, li+2, lj+1, lk, ll+1)     → r_l position on g0
///   g3  = G2E_D_L(g2, li+2, lj+1, lk, ll)        → nabla_l on g2
///   g4  = G2E_D_J(g0, li+1, lj+0, lk, ll)        → nabla_j on g0
///        + G2E_D_I(g0, li+1, lj+0, lk, ll)        → nabla_i on g0 (sum)
///   g7  = G2E_D_J(g3, li+1, lj+0, lk, ll)        → nabla_j on g3
///        + G2E_D_I(g3, li+1, lj+0, lk, ll)        → nabla_i on g3 (sum)
///   g12 = G2E_D_I(g4, li, lj, lk, ll)            → nabla_i on g4
///   g15 = G2E_D_I(g7, li, lj, lk, ll)            → nabla_i on g7
///
/// Contraction sum (same 9 terms as r1p2, using g3 not g1):
fn gout_breit_r2p2(
    g: &[f64],
    shape: &BreitShape,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ai: f64,
    aj: f64,
    al: f64,
    rl: &[f64; 3],
) -> Vec<f64> {
    let g_size = shape.g_size;
    let nroots = shape.nroots;

    // g2 = x1l(g0, rl) at (li+2, lj+1, lk, ll+1)
    let mut g2 = vec![0.0_f64; 3 * g_size];
    x1l_breit(&mut g2, g, rl, li + 2, lj + 1, lk, ll + 1, shape);

    // g3 = nabla_l(g2) at (li+2, lj+1, lk, ll)
    let mut g3 = vec![0.0_f64; 3 * g_size];
    nabla1l_breit(&mut g3, &g2, li + 2, lj + 1, lk, ll, al, shape);

    // g4 = nabla_j(g0) at (li+1, lj+0, lk, ll) + nabla_i(g0) at (li+1, lj+0, lk, ll)
    let mut g4 = vec![0.0_f64; 3 * g_size];
    let mut g5_tmp = vec![0.0_f64; 3 * g_size];
    nabla1j_breit(&mut g4, g, li + 1, lj, lk, ll, aj, shape);
    nabla1i_breit(&mut g5_tmp, g, li + 1, lj, lk, ll, ai, shape);
    for ix in 0..3 * g_size {
        g4[ix] += g5_tmp[ix];
    }

    // g7 = nabla_j(g3) at (li+1, lj+0, lk, ll) + nabla_i(g3) at (li+1, lj+0, lk, ll)
    let mut g7 = vec![0.0_f64; 3 * g_size];
    let mut g8 = vec![0.0_f64; 3 * g_size];
    nabla1j_breit(&mut g7, &g3, li + 1, lj, lk, ll, aj, shape);
    nabla1i_breit(&mut g8, &g3, li + 1, lj, lk, ll, ai, shape);
    for ix in 0..3 * g_size {
        g7[ix] += g8[ix];
    }

    // g12 = nabla_i(g4) at (li, lj, lk, ll)
    let mut g12 = vec![0.0_f64; 3 * g_size];
    nabla1i_breit(&mut g12, &g4, li, lj, lk, ll, ai, shape);

    // g15 = nabla_i(g7) at (li, lj, lk, ll)
    let mut g15 = vec![0.0_f64; 3 * g_size];
    nabla1i_breit(&mut g15, &g7, li, lj, lk, ll, ai, shape);

    // Contract
    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);

    let ci_comps = cart_comps(li as u8);
    let cj_comps = cart_comps(lj as u8);
    let ck_comps = cart_comps(lk as u8);
    let cl_comps = cart_comps(ll as u8);

    let gx_off = 0usize;
    let gy_off = g_size;
    let gz_off = 2 * g_size;

    let mut out = vec![0.0_f64; nfi * nfj * nfk * nfl];

    for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
        for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let x_idx = ix as usize * shape.di
                        + kx as usize * shape.dk
                        + lx as usize * shape.dl
                        + jx as usize * shape.dj;
                    let y_idx = iy as usize * shape.di
                        + ky as usize * shape.dk
                        + ly as usize * shape.dl
                        + jy as usize * shape.dj;
                    let z_idx = iz as usize * shape.di
                        + kz as usize * shape.dk
                        + lz as usize * shape.dl
                        + jz as usize * shape.dj;

                    let mut s = 0.0_f64;
                    for irys in 0..nroots {
                        // Same 9-term contraction as r1p2 (same pattern, different g3/g12/g15)
                        s += g15[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys]
                            * g3[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys]
                            * g12[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g15[gy_off + y_idx + irys]
                            * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g12[gy_off + y_idx + irys]
                            * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g3[gy_off + y_idx + irys]
                            * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys]
                            * g[gy_off + y_idx + irys]
                            * g15[gz_off + z_idx + irys];
                    }
                    let out_idx = i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                    out[out_idx] = s;
                }
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// launch_breit: Breit spinor-only 2e integrals
// ─────────────────────────────────────────────────────────────────────────────

/// Launch the Breit two-electron spinor integral.
///
/// Implements `int2e_breit_r1p2_spinor` and `int2e_breit_r2p2_spinor` from libcint breit.c.
///
/// Both variants use the same single-pass computation (not the three-stage composite
/// `_int2e_breit_drv` used by the ssp/sps spinor variants). They use a specific gout
/// function that applies gradient and position operators to the g-tensor.
///
/// Per D-07: Breit family supports only spinor representation. Cart and spherical
/// are rejected before this function is called via `resolve_family` (manifest forms guard).
///
/// Angular momentum increments (GSHIFT=4, ncomp_tensor=1):
///   breit_r1p2: ng = {2, 2, 0, 1, 4, 1, 1, 1}  → li+2, lj+2, lk+0, ll+1
///   breit_r2p2: ng = {2, 1, 0, 2, 4, 1, 1, 1}  → li+2, lj+1, lk+0, ll+2
pub fn launch_breit(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    // Per-quartet G-tensor BUILD runs on-device (run_breit_g_on_backend); the
    // gout/nabla/x1 operator ladder + cart_to_spinor transform stay host (split
    // documented in the module header).

    // D-07: Breit is spinor-only. Cart/sph are rejected by manifest forms guard in
    // resolve_family before we reach here, but add a defensive check.
    if plan.representation != Representation::Spinor {
        let rep = plan.representation.to_string();
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("breit requires spinor representation, got: {rep}"),
        });
    }

    let operator_name = plan.descriptor.entry.operator_name;
    let is_r1p2 = match operator_name {
        "breit_r1p2" => true,
        "breit_r2p2" => false,
        other => {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("launch_breit: unknown operator_name: {other}"),
            });
        }
    };

    let shells = plan.shells.as_slice();
    if shells.len() < 4 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_breit",
            detail: "breit kernel requires exactly 4 shells".to_owned(),
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

    // Angular momentum increments per breit.c ng arrays:
    //   breit_r1p2: ng = {2, 2, 0, 1, 4, 1, 1, 1} → IINC=2, JINC=2, KINC=0, LINC=1
    //   breit_r2p2: ng = {2, 1, 0, 2, 4, 1, 1, 1} → IINC=2, JINC=1, KINC=0, LINC=2
    let (iinc, jinc, kinc, linc) = if is_r1p2 { (2, 2, 0, 1) } else { (2, 1, 0, 2) };
    let li_e = li + iinc;
    let lj_e = lj + jinc;
    let lk_e = lk + kinc;
    let ll_e = ll + linc;

    let shape = build_breit_shape(li_e, lj_e, lk_e, ll_e);

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    let rl = atoms[shell_l.atom_index as usize].coord_bohr;

    let nfi = ncart(li as u8);
    let nfj = ncart(lj as u8);
    let nfk = ncart(lk as u8);
    let nfl = ncart(ll as u8);

    let mut cart_buf = vec![0.0_f64; nfi * nfj * nfk * nfl];

    // Common factor: same as 2e (Pitfall 2: all four sp factors required)
    let sp_factor = common_fac_sp(li as u8)
        * common_fac_sp(lj as u8)
        * common_fac_sp(lk as u8)
        * common_fac_sp(ll as u8);
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * sp_factor;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    // Build g-tensor with elevated angular momenta for derivative
                    // headroom — ON DEVICE (CubeCL breit_g_kernel, generic over F).
                    let g = run_breit_g_on_backend(
                        backend,
                        shape,
                        ai,
                        aj,
                        ak,
                        al,
                        ri,
                        rj,
                        rk,
                        rl,
                        quartet_fac,
                    );

                    // Apply the Breit-specific gout contraction
                    let prim_cart = if is_r1p2 {
                        gout_breit_r1p2(&g, &shape, li, lj, lk, ll, ai, aj, al, &rj)
                    } else {
                        gout_breit_r2p2(&g, &shape, li, lj, lk, ll, ai, aj, al, &rl)
                    };

                    // Accumulate with contraction coefficients
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

    // Apply spinor transform (Breit is spinor-only per D-07)
    //
    // libcint int2e_breit_r1p2_spinor and int2e_breit_r2p2_spinor use c2s_sf_2e1i + c2s_sf_2e2i
    // (iket variants), which apply a phase of i to both the j-ket (step 1) and l-ket (step 2).
    // The combined phase is i_j * i_l = i^2 = -1 relative to the regular c2s_sf_2e1 + c2s_sf_2e2
    // transform used by ordinary 2e integrals. We apply cart_to_spinor_sf_4d (regular) and then
    // negate, matching the iket phase convention.
    cart_to_spinor_sf_4d(
        staging,
        &cart_buf,
        li as u8,
        shell_i.kappa,
        lj as u8,
        shell_j.kappa,
        lk as u8,
        shell_k.kappa,
        ll as u8,
        shell_l.kappa,
    )?;
    // Negate to account for c2s_sf_2e1i + c2s_sf_2e2i phase convention.
    for v in staging.iter_mut() {
        *v = -*v;
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

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;
    use cubecl::Runtime;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    fn assert_close(host: &[f64], dev: &[f64], tag: &str) {
        assert_eq!(host.len(), dev.len(), "length mismatch ({tag})");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "breit device/host G-tensor mismatch ({tag}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    /// Device-vs-host cross-check (CpuRuntime, f64): the on-device
    /// `breit_g_kernel` must reproduce the host `fill_g_tensor_breit` G-tensor
    /// (the device deliverable for breit) within atol=1e-12 / rtol=1e-10, for
    /// elevated-momentum quartets covering all FOUR HRR branches
    /// (ibase = li_e>lj_e, kbase = lk_e>ll_e).
    #[test]
    fn test_device_matches_host_breit() {
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        let ak = 0.7_f64;
        let al = 1.1_f64;
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let rk = [0.2_f64, 0.8, 0.3];
        let rl = [0.4_f64, 0.1, 0.9];
        let fac_env = 1.0_f64;
        // (li_e, lj_e, lk_e, ll_e): FF, TF, FT, TT branch coverage.
        for &(lie, lje, lke, lle) in &[
            (0usize, 1usize, 0usize, 1usize),
            (1, 0, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 1, 0),
        ] {
            let shape = build_breit_shape(lie, lje, lke, lle);
            let host = fill_g_tensor_breit(ai, aj, ak, al, &ri, &rj, &rk, &rl, shape, fac_env);
            let dev = run_breit_g_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client(),
                shape.li_elev as u32,
                shape.lj_elev as u32,
                shape.lk_elev as u32,
                shape.ll_elev as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ai,
                aj,
                ak,
                al,
                ri,
                rj,
                rk,
                rl,
                fac_env,
            );
            assert_close(
                &host,
                &dev,
                &format!("breit g-tensor lie={lie} lje={lje} lke={lke} lle={lle}"),
            );
        }
    }

    /// Genericity smoke test: the `breit_g_kernel` monomorphizes and launches for
    /// `F = f32` (not just f64), producing finite output for a small quartet.
    #[test]
    fn test_breit_g_kernel_generic_f32() {
        let shape = build_breit_shape(0, 1, 0, 1);
        let client = cpu_client();
        let g_size_u = shape.g_size;
        let nroots_u = shape.nroots;
        let g_zero = vec![0.0_f32; 3 * g_size_u];
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let rys_zero = vec![0.0_f32; nroots_u];
        let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        breit_g_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            crate::plane::standard_plane_cube_dim(),
            unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3 * g_size_u) },
            unsafe { ArrayArg::from_raw_parts(u_h, nroots_u) },
            unsafe { ArrayArg::from_raw_parts(w_h, nroots_u) },
            0.9_f32,
            1.3_f32,
            0.7_f32,
            1.1_f32,
            0.0_f32,
            0.0_f32,
            0.0_f32,
            0.6_f32,
            0.5_f32,
            0.7_f32,
            0.2_f32,
            0.8_f32,
            0.3_f32,
            0.4_f32,
            0.1_f32,
            0.9_f32,
            1.0_f32,
            PIE4 as f32,
            shape.li_elev as u32,
            shape.lj_elev as u32,
            shape.lk_elev as u32,
            shape.ll_elev as u32,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
        );
        let raw = client.read_one_unchecked(g_h);
        let out = f32::from_bytes(&raw);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "breit f32 G-tensor produced non-finite output"
        );
    }
}
