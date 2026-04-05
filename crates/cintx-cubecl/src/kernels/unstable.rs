//! Launch functions for unstable-source API families.
//!
//! Phase 14 Wave 2: `launch_breit` is fully implemented.
//! Other families (origi, grids, origk, ssc) remain stubbed — pending future plans.
//!
//! Families covered:
//!   - origi: origin-displaced r^n one-electron integrals (cint1e_a.c)      [stub]
//!   - grids: grid-point integrals with NGRIDS env parameter (cint1e_grids.c) [stub]
//!   - breit: Breit spinor-only 2e integrals (breit.c)                        [implemented]
//!   - origk: origin-k-displaced 3c1e integrals (cint3c1e_a.c)               [stub]
//!   - ssc: spin-spin contact 3c2e integral (cint3c2e.c)                     [stub]

use crate::backend::ResolvedBackend;
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::ncart;
use crate::transform::c2spinor::cart_to_spinor_sf_4d;
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
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
        vrr_fill_axis_breit(gx, irys, shape.nmax, shape.mmax, shape.g2d_ijmax, shape.g2d_klmax, c00[0], c0p[0], b10, b01, b00);
        vrr_fill_axis_breit(gy, irys, shape.nmax, shape.mmax, shape.g2d_ijmax, shape.g2d_klmax, c00[1], c0p[1], b10, b01, b00);
        vrr_fill_axis_breit(gz, irys, shape.nmax, shape.mmax, shape.g2d_ijmax, shape.g2d_klmax, c00[2], c0p[2], b10, b01, b00);
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
                    let x_idx = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let y_idx = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let z_idx = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = 0.0_f64;
                    for irys in 0..nroots {
                        // 9-term contraction per breit.c CINTgout2e_int2e_breit_r1p2
                        s += g15[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys] * g3[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys] * g12[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g15[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g12[gy_off + y_idx + irys] * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g3[gy_off + y_idx + irys] * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g15[gz_off + z_idx + irys];
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
                    let x_idx = ix as usize * shape.di + kx as usize * shape.dk + lx as usize * shape.dl + jx as usize * shape.dj;
                    let y_idx = iy as usize * shape.di + ky as usize * shape.dk + ly as usize * shape.dl + jy as usize * shape.dj;
                    let z_idx = iz as usize * shape.di + kz as usize * shape.dk + lz as usize * shape.dl + jz as usize * shape.dj;

                    let mut s = 0.0_f64;
                    for irys in 0..nroots {
                        // Same 9-term contraction as r1p2 (same pattern, different g3/g12/g15)
                        s += g15[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys] * g3[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g12[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys] * g12[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g15[gy_off + y_idx + irys] * g[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g12[gy_off + y_idx + irys] * g3[gz_off + z_idx + irys];
                        s += g3[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g3[gy_off + y_idx + irys] * g12[gz_off + z_idx + irys];
                        s += g[gx_off + x_idx + irys] * g[gy_off + y_idx + irys] * g15[gz_off + z_idx + irys];
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
    // Host-side implementation — no CubeCL dispatch needed.
    let _ = backend;

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
    let sp_factor = common_fac_sp(li as u8) * common_fac_sp(lj as u8) * common_fac_sp(lk as u8) * common_fac_sp(ll as u8);
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

                    // Build g-tensor with elevated angular momenta for derivative headroom
                    let g = fill_g_tensor_breit(ai, aj, ak, al, &ri, &rj, &rk, &rl, shape, quartet_fac);

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

/// Stub for origi family (int1e_r2_origi, int1e_r4_origi, ip2 derivatives).
/// Implementation pending in a future plan.
pub fn launch_origi(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "origi: stub — implementation pending".to_owned(),
    })
}

/// Stub for grids family (int1e_grids and derivative variants).
/// Implementation pending in a future plan.
pub fn launch_grids(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "grids: stub — implementation pending".to_owned(),
    })
}

/// Stub for origk family (int3c1e_r2/r4/r6_origk and ip1 derivatives).
/// Implementation pending in a future plan.
pub fn launch_origk(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "origk: stub — implementation pending".to_owned(),
    })
}

/// Stub for ssc family (int3c2e_sph_ssc).
/// Implementation pending in a future plan.
pub fn launch_ssc(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "ssc: stub — implementation pending".to_owned(),
    })
}
