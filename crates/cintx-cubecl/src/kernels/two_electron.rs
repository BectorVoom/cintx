//! Host-side 2e (four-center electron-repulsion) integral kernel.
//!
//! Implements the libcint `g2e.c` recurrence pipeline:
//! 1. Rys roots/weights per primitive quartet (`rys_roots_host`).
//! 2. 2D VRR fill (`CINTg0_2e_2d` equivalent).
//! 3. Branch-specific 4D HRR transfer (ibase/kbase adaptive stride choice).
//! 4. Cartesian contraction + optional `cart_to_sph_2e` transform.

use crate::backend::ResolvedBackend;
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::rys_roots_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_4d;
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate.
/// Scalar 2e `nroots = (li+lj+lk+ll)/2 + 1`, so this covers `l-sum <= 8`.
const MAX_DEVICE_NROOTS: usize = 5;

/// Maximum `nroots` the HOST Rys engine (`rys_roots_host` → `rys_wheeler`) evaluates
/// (Phase 25 FND-02). The host gradient/Hessian path uses the Wheeler nroots 6..12
/// engine; the vendor build caps at 12 (quadmath disabled), so nroots>12 stays
/// fail-closed (T-25-03). Device kernels keep their own `MAX_DEVICE_NROOTS=5` cap.
const HOST_RYS_NROOTS_CEILING: usize = 12;

/// Spherical harmonic normalization prefactor for s and p shells.
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Test-only since quick-260529-q4k: the scalar 2e contraction now runs on-device
/// (`two_electron_scalar_kernel`); `contract_2e_cart` / `cart_comps` remain as the
/// host cross-check reference used by `device_tests`.
#[cfg(test)]
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

/// `pub(crate)` (Phase 21 D-04): shared with `center_3c2e.rs::int3c2e_ip1`, which
/// builds its derivative G-tensor through the SAME 2e recurrence machinery using the
/// 3c2e Pitfall-4 kl mapping (real `k` → 2e `ll` slot; 2e `lk` slot is a phantom
/// s-function). The struct carries the identical field set as
/// [`crate::kernels::f12::F12Shape`] so `gout_ip1` can be reused verbatim.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TwoEShape {
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

/// Initialize stride/layout metadata following `CINTinit_int2e_EnvVars`.
///
/// `pub(crate)` (Phase 21 D-04): `center_3c2e.rs::int3c2e_ip1` calls this with the
/// 3c2e kl mapping `build_2e_shape(li+1, lj, 0, lk)` (phantom `lk=0`, real k in the
/// `ll` slot, bra `i` raised to `li+1` for the `∇_i` headroom).
pub(crate) fn build_2e_shape(li: usize, lj: usize, lk: usize, ll: usize) -> TwoEShape {
    let nroots = (li + lj + lk + ll) / 2 + 1;
    let nmax = li + lj;
    let mmax = lk + ll;

    // Adaptive branch selection from libcint (strict >).
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

    TwoEShape {
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
fn vrr_fill_axis(
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

/// HRR branch for `ibase=false && kbase=false` (`CINTg0_lj2d_4d`).
fn hrr_lj2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=false && kbase=true` (`CINTg0_kj2d_4d`).
fn hrr_kj2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=true && kbase=false` (`CINTg0_il2d_4d`).
fn hrr_il2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=true && kbase=true` (`CINTg0_ik2d_4d`).
fn hrr_ik2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// Fill the full `[gx|gy|gz]` tensor for one primitive quartet.
///
/// `pub(crate)` (Phase 21 D-04): `center_3c2e.rs::int3c2e_ip1` fills its derivative
/// G-tensor through this exact recurrence using the 3c2e kl mapping (phantom 2e `lk`
/// shell with exponent `ak=0` at the real-k center, real k in the 2e `ll` slot).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_g_tensor_2e(
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
    ri: &[f64; 3],
    rj: &[f64; 3],
    rk: &[f64; 3],
    rl: &[f64; 3],
    shape: TwoEShape,
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
        vrr_fill_axis(
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
        vrr_fill_axis(
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
        vrr_fill_axis(
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

    // HRR transfer to final (i,k,l,j) layout with branch-specific ordering.
    if shape.kbase {
        if shape.ibase {
            hrr_ik2d_4d(&mut g, shape, rirj, rkrl);
        } else {
            hrr_kj2d_4d(&mut g, shape, rirj, rkrl);
        }
    } else if shape.ibase {
        hrr_il2d_4d(&mut g, shape, rirj, rkrl);
    } else {
        hrr_lj2d_4d(&mut g, shape, rirj, rkrl);
    }

    g
}

/// Contract `[gx|gy|gz]` into Cartesian 2e tensor with output order:
/// `out[i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk]` (i fastest, l slowest).
///
/// Test-only since quick-260529-q4k (see `cart_comps` note): the production scalar
/// 2e path runs `two_electron_scalar_kernel` on-device; this host reference is the
/// `device_tests` cross-check oracle.
#[cfg(test)]
fn contract_2e_cart(g: &[f64], shape: TwoEShape, li: u8, lj: u8, lk: u8, ll: u8) -> Vec<f64> {
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

/// Bridge a plain-2e [`TwoEShape`] into the [`crate::kernels::f12::F12Shape`] that
/// [`crate::kernels::f12::gout_ip1`] / `nabla1i_2e` consume.
///
/// The two structs carry the IDENTICAL field set (di/dj/dk/dl/nroots/nmax/mmax/
/// li/lj/lk/ll/ibase/kbase/g2d_ijmax/g2d_klmax/g_size). The gradient math is
/// F12-free, so this 1:1 field copy lets the plain-Coulomb gradient reuse the
/// exact verbatim derivative code (Phase 21 D-04).
///
/// `pub(crate)` (Phase 21 D-04): `center_3c2e.rs::int3c2e_ip1` reuses this bridge so
/// its 3c2e derivative G-tensor can be fed to the verbatim `gout_ip1` contraction.
#[inline]
pub(crate) fn two_e_shape_as_f12(shape: &TwoEShape) -> crate::kernels::f12::F12Shape {
    crate::kernels::f12::F12Shape {
        nroots: shape.nroots,
        nmax: shape.nmax,
        mmax: shape.mmax,
        li: shape.li,
        lj: shape.lj,
        lk: shape.lk,
        ll: shape.ll,
        ibase: shape.ibase,
        kbase: shape.kbase,
        di: shape.di,
        dk: shape.dk,
        dl: shape.dl,
        dj: shape.dj,
        g2d_ijmax: shape.g2d_ijmax,
        g2d_klmax: shape.g2d_klmax,
        g_size: shape.g_size,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scalar 2e device kernel — `#[cube(launch)]`, generic over `F: Float`
//
//  Faithful inline port of the host SCALAR pipeline
//  `fill_g_tensor_2e` → `contract_2e_cart`, accumulated over all primitive
//  quartets (pi,pj,pk,pl) and contraction quads (ci,cj,ck,cl) into a per-quad
//  i-fastest Cartesian block buffer. Intermediate arithmetic in `F` (run at f64
//  by the launcher), output written to `cart_out` in `F`.
//
//  ALL strides (di,dk,dl,dj,g_size,nmax,mmax,g2d_ijmax,g2d_klmax) plus the
//  ibase/kbase flags are computed host-side via `build_2e_shape` and passed in as
//  runtime u32 — the adaptive dli/dlj/dlk/dll branch logic is NOT recomputed
//  on-device (avoids if-expressions). `#[comptime] nroots` selects rys_root{1..5}.
// ─────────────────────────────────────────────────────────────────────────────

/// Single-work-item scalar 2e kernel. See module note above.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn two_electron_scalar_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    exps_k: &Array<F>,
    exps_l: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    coeff_k: &Array<F>,
    coeff_l: &Array<F>,
    g: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    cart_out: &mut Array<F>,
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
    common_factor: F,
    pie4: F,
    li: u32,
    lj: u32,
    lk: u32,
    ll: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nprim_l: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    nctr_l: u32,
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

        let nfi = (li + 1u32) * (li + 2u32) / 2u32;
        let nfj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let nfk = (lk + 1u32) * (lk + 2u32) / 2u32;
        let nfl = (ll + 1u32) * (ll + 2u32) / 2u32;
        let block_len = nfi * nfj * nfk * nfl;
        let out_len = nctr_i * nctr_j * nctr_k * nctr_l * block_len;

        // Zero the per-quad accumulation buffer.
        let mut oi = 0u32;
        while oi < out_len {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // Primitive quartet loop.
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];
                let aij = ai + aj;
                // Gaussian product center for ij and the bra overlap exponential.
                let rijx = (ai * rix + aj * rjx) / aij;
                let rijy = (ai * riy + aj * rjy) / aij;
                let rijz = (ai * riz + aj * rjz) / aij;
                let dxij = rix - rjx;
                let dyij = riy - rjy;
                let dzij = riz - rjz;
                let rr_ij = dxij * dxij + dyij * dyij + dzij * dzij;
                let fac_ij = F::exp(-ai * aj / aij * rr_ij);

                let mut pk = 0u32;
                while pk < nprim_k {
                    let ak = exps_k[pk as usize];
                    let mut pl = 0u32;
                    while pl < nprim_l {
                        let al = exps_l[pl as usize];
                        let akl = ak + al;
                        let rklx = (ak * rkx + al * rlx) / akl;
                        let rkly = (ak * rky + al * rly) / akl;
                        let rklz = (ak * rkz + al * rlz) / akl;
                        let dxkl = rkx - rlx;
                        let dykl = rky - rly;
                        let dzkl = rkz - rlz;
                        let rr_kl = dxkl * dxkl + dykl * dykl + dzkl * dzkl;
                        let fac_kl = F::exp(-ak * al / akl * rr_kl);

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

                        // ibase/kbase-selected reference centers (rijrx / rklrx)
                        // and the HRR displacement vectors rirj / rkrl.
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

                        let fac1 = F::sqrt(a0 / (a1 * a1 * a1))
                            * common_factor
                            * fac_ij
                            * fac_kl;

                        // ── Build the [gx|gy|gz] tensor ───────────────────────
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

                            // Per-axis c00/c0p then inline vrr_fill_axis.
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

                                // Inline vrr_fill_axis(g[off..], irys2, nmax, mmax,
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
                                                + F::cast_from(m3)
                                                    * b00
                                                    * g[(off + jbase + n2 * dn - dm) as usize];
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

                        // ── HRR transfer (branch by kbase/ibase) ──────────────
                        // Each axis block is at off = axis*g_size.
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
                                    // ik2d: i then k done; transfer dl←dk (ll), dj←di (lj).
                                    // ll loop
                                    let mut l = 1u32;
                                    while l <= ll {
                                        let mut k = 0u32;
                                        while k <= (mmax - l) {
                                            let mut i = 0u32;
                                            while i <= nmax {
                                                let ptr = l * dl + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nrys {
                                                    let idx = ptr + r;
                                                    g[(off + idx) as usize] = rkrl
                                                        * g[(off + idx - dl) as usize]
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
                                    while j <= lj {
                                        let mut l2 = 0u32;
                                        while l2 <= ll {
                                            let mut k2 = 0u32;
                                            while k2 <= lk {
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
                                    // kj2d: i raise (dj←di), then l raise (dl←dk).
                                    let mut i = 1u32;
                                    while i <= li {
                                        let mut j = 0u32;
                                        while j <= (nmax - i) {
                                            let mut k = 0u32;
                                            while k <= mmax {
                                                let ptr = j * dj + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nrys {
                                                    let idx = ptr + r;
                                                    g[(off + idx) as usize] = rirj
                                                        * g[(off + idx - di) as usize]
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
                                    while j2 <= lj {
                                        let mut l = 1u32;
                                        while l <= ll {
                                            let mut k = 0u32;
                                            while k <= (mmax - l) {
                                                let ptr = j2 * dj + l * dl + k * dk;
                                                let mut n = 0u32;
                                                while n < dk {
                                                    let idx = ptr + n;
                                                    g[(off + idx) as usize] = rkrl
                                                        * g[(off + idx - dl) as usize]
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
                                // il2d: k raise (dk←... using dl), then j raise (dj←di).
                                let mut k = 1u32;
                                while k <= lk {
                                    let mut l = 0u32;
                                    while l <= (mmax - k) {
                                        let mut i = 0u32;
                                        while i <= nmax {
                                            let ptr = l * dl + k * dk + i * di;
                                            let mut r = 0u32;
                                            while r < nrys {
                                                let idx = ptr + r;
                                                g[(off + idx) as usize] = rkrl
                                                    * g[(off + idx - dk) as usize]
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
                                while j <= lj {
                                    let mut l = 0u32;
                                    while l <= ll {
                                        let mut k2 = 0u32;
                                        while k2 <= lk {
                                            let ptr = j * dj + l * dl + k2 * dk;
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
                                        l += 1u32;
                                    }
                                    j += 1u32;
                                }
                            } else {
                                // lj2d: i raise (dj←di), then k raise (dl←dk).
                                let mut i = 1u32;
                                while i <= li {
                                    let mut j = 0u32;
                                    while j <= (nmax - i) {
                                        let mut l = 0u32;
                                        while l <= mmax {
                                            let ptr = j * dj + l * dl + i * di;
                                            let mut r = 0u32;
                                            while r < nrys {
                                                let idx = ptr + r;
                                                g[(off + idx) as usize] = rirj
                                                    * g[(off + idx - di) as usize]
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
                                while j2 <= lj {
                                    let mut k = 1u32;
                                    while k <= lk {
                                        let mut l = 0u32;
                                        while l <= (mmax - k) {
                                            let ptr = j2 * dj + l * dl + k * dk;
                                            let mut n = 0u32;
                                            while n < dk {
                                                let idx = ptr + n;
                                                g[(off + idx) as usize] = rkrl
                                                    * g[(off + idx - dk) as usize]
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

                        // ── Contract into per-quad Cartesian blocks ───────────
                        // Descending cart_comps over (l,k,j,i); i fastest.
                        let mut l_idx = 0u32;
                        let mut la = 0u32;
                        while la <= ll {
                            let lx = ll - la;
                            let ll_minus = ll - lx;
                            let mut lb = 0u32;
                            while lb <= ll_minus {
                                let ly = ll_minus - lb;
                                let lz = ll - lx - ly;

                                let mut k_idx = 0u32;
                                let mut ka = 0u32;
                                while ka <= lk {
                                    let kx = lk - ka;
                                    let lk_minus = lk - kx;
                                    let mut kb = 0u32;
                                    while kb <= lk_minus {
                                        let ky = lk_minus - kb;
                                        let kz = lk - kx - ky;

                                        let mut j_idx = 0u32;
                                        let mut ja = 0u32;
                                        while ja <= lj {
                                            let jx = lj - ja;
                                            let lj_minus = lj - jx;
                                            let mut jb = 0u32;
                                            while jb <= lj_minus {
                                                let jy = lj_minus - jb;
                                                let jz = lj - jx - jy;

                                                let mut i_idx = 0u32;
                                                let mut ia = 0u32;
                                                while ia <= li {
                                                    let ix = li - ia;
                                                    let li_minus = li - ix;
                                                    let mut ib = 0u32;
                                                    while ib <= li_minus {
                                                        let iy = li_minus - ib;
                                                        let iz = li - ix - iy;

                                                        let mut sum = F::new(0.0);
                                                        let mut r = 0u32;
                                                        while r < nrys {
                                                            let xi = r
                                                                + ix * di
                                                                + kx * dk
                                                                + lx * dl
                                                                + jx * dj;
                                                            let yi = r
                                                                + iy * di
                                                                + ky * dk
                                                                + ly * dl
                                                                + jy * dj;
                                                            let zi = r
                                                                + iz * di
                                                                + kz * dk
                                                                + lz * dl
                                                                + jz * dj;
                                                            sum += g[xi as usize]
                                                                * g[(gy_off + yi) as usize]
                                                                * g[(gz_off + zi) as usize];
                                                            r += 1u32;
                                                        }

                                                        // Accumulate into every
                                                        // contraction quad block.
                                                        let mut ci = 0u32;
                                                        while ci < nctr_i {
                                                            let cvi =
                                                                coeff_i[(pi * nctr_i + ci) as usize];
                                                            let mut cj = 0u32;
                                                            while cj < nctr_j {
                                                                let cvj = coeff_j
                                                                    [(pj * nctr_j + cj) as usize];
                                                                let mut ck = 0u32;
                                                                while ck < nctr_k {
                                                                    let cvk = coeff_k[(pk
                                                                        * nctr_k
                                                                        + ck)
                                                                        as usize];
                                                                    let mut cl = 0u32;
                                                                    while cl < nctr_l {
                                                                        let cvl = coeff_l[(pl
                                                                            * nctr_l
                                                                            + cl)
                                                                            as usize];
                                                                        let weight =
                                                                            cvi * cvj * cvk * cvl;
                                                                        let qbase = (((ci
                                                                            * nctr_j
                                                                            + cj)
                                                                            * nctr_k
                                                                            + ck)
                                                                            * nctr_l
                                                                            + cl)
                                                                            * block_len;
                                                                        let oidx = qbase
                                                                            + i_idx
                                                                            + j_idx * nfi
                                                                            + k_idx * nfi * nfj
                                                                            + l_idx
                                                                                * nfi
                                                                                * nfj
                                                                                * nfk;
                                                                        cart_out[oidx as usize] +=
                                                                            weight * sum;
                                                                        cl += 1u32;
                                                                    }
                                                                    ck += 1u32;
                                                                }
                                                                cj += 1u32;
                                                            }
                                                            ci += 1u32;
                                                        }

                                                        i_idx += 1u32;
                                                        ib += 1u32;
                                                    }
                                                    ia += 1u32;
                                                }
                                                j_idx += 1u32;
                                                jb += 1u32;
                                            }
                                            ja += 1u32;
                                        }
                                        k_idx += 1u32;
                                        kb += 1u32;
                                    }
                                    ka += 1u32;
                                }
                                l_idx += 1u32;
                                lb += 1u32;
                            }
                            la += 1u32;
                        }

                        pl += 1u32;
                    }
                    pk += 1u32;
                }
                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`two_electron_scalar_kernel`] at `f64` on a resolved backend client
/// and read back the per-quad Cartesian accumulation buffer (`out_len`).
#[allow(clippy::too_many_arguments)]
fn run_2e_scalar_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    lk: u32,
    ll: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nprim_l: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    nctr_l: u32,
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
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    exps_l: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
    coeff_l: &[f64],
    out_len: usize,
) -> Vec<f64> {
    let nroots_u = nroots as usize;
    let g_size_u = g_size as usize;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let exps_k_h = client.create_from_slice(f64::as_bytes(exps_k));
    let exps_l_h = client.create_from_slice(f64::as_bytes(exps_l));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let coeff_k_h = client.create_from_slice(f64::as_bytes(coeff_k));
    let coeff_l_h = client.create_from_slice(f64::as_bytes(coeff_l));

    let g_zero = vec![0.0_f64; 3 * g_size_u];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    two_electron_scalar_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(exps_i_h, exps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_j_h, exps_j.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_k_h, exps_k.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_l_h, exps_l.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_j_h, coeff_j.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_k_h, coeff_k.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_l_h, coeff_l.len()) },
        unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size_u) },
        unsafe { ArrayArg::from_raw_parts(u_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(w_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
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
        common_factor,
        PIE4,
        li,
        lj,
        lk,
        ll,
        nprim_i,
        nprim_j,
        nprim_k,
        nprim_l,
        nctr_i,
        nctr_j,
        nctr_k,
        nctr_l,
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

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// `int2e_ip1` gradient launch — the ∇_A <ij|kl> two-electron force (GRAD-07).
///
/// Builds the plain Coulomb G-tensor with `li_ceil = li+1` headroom
/// ([`fill_g_tensor_2e`] via `rys_roots_host`), reuses
/// [`crate::kernels::f12::gout_ip1`] verbatim on it, and emits component-leading
/// `[3, nl, nk, nj, ni]` F-order matching pyscf-gto `layout_table.rs` (Risk R3).
///
/// Guards (fail-closed):
///   - `grad_shape.nroots > 12` → `UnsupportedApi` (Phase 25 FND-02 / T-25-03): the
///     host Rys engine (`rys_roots_host` → `rys_wheeler`) supports nroots 6..12, so the
///     li→li+1 raise routes Hessian-elevated quartets to the host `fill_g_tensor_2e`
///     path; only nroots>12 (vendor quadmath ceiling) is rejected.
///   - `Representation::Spinor` → `UnsupportedApi` (R5 / T-21-05-04).
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_ip1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Spinor gradient: not supported (R5 / D-03). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int2e_ip1 gradient".to_owned(),
        });
    }

    // li → li+1 headroom shape (D-06). gout_ip1's nabla1i_2e reads up to index li+1.
    let grad_shape = build_2e_shape(li as usize + 1, lj as usize, lk as usize, ll as usize);

    // Phase 25 FND-02: this is the HOST gradient path (the loop below calls
    // `fill_g_tensor_2e` → `rys_roots_host`, NOT the device comptime kernel). The host
    // Rys engine now supports nroots 6..12 (rys_wheeler.rs), so the elevated-li Hessian
    // d-quartets that push nroots to 6 route here instead of returning UnsupportedApi.
    // The ceiling is the vendor-validated 12 (quadmath disabled); nroots>12 stays
    // fail-closed (T-25-03: typed error, never a panic).
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = 3 * block_len; // 3 components × Cartesian AO product

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    // Per-contraction-quad component-leading Cartesian accumulator: one
    // `3 * nfi*nfj*nfk*nfl` block per (ci,cj,ck,cl) quad (mirrors the scalar 2e
    // general-contraction layout). For all-nctr==1 this is a single block.
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

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

                    // Plain Coulomb G-tensor at the elevated li (li+1 headroom).
                    let g = fill_g_tensor_2e(
                        ai, aj, ak, al, &ri, &rj, &rk, &rl, grad_shape, quartet_fac,
                    );

                    // Reuse gout_ip1 verbatim (f12.rs). It returns interleaved
                    // out[n*3+comp]; n walks [cl, ck, cj, ci] (ll slowest, li fastest).
                    // gout_ip1 is called at BASE li (the G-tensor carries li+1 headroom).
                    let gout = crate::kernels::f12::gout_ip1(
                        &g,
                        &grad_f12_shape,
                        li as usize,
                        lj as usize,
                        lk as usize,
                        ll as usize,
                        ai,
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
                                    let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l
                                        + cl)
                                        * total_len;
                                    // TRANSPOSE interleaved gout[n*3+comp] into the
                                    // component-leading block: cart[comp*block + n].
                                    for n in 0..block_len {
                                        for comp in 0..3usize {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * 3 + comp];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write component-leading `[3, nl, nk, nj, ni]` F-order to staging.
    // For each component, the per-quad block is the i-fastest `[nl][nk][nj][ni]`
    // Cartesian tensor — run the cart→sph transform per component for the sph rep.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
                                                staging[dst] = F::from_f64_lossy(sph[src]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let block = &cart_blocks[base..base + block_len];
                                for lc in 0..nfl {
                                    let lidx = cl * nfl + lc;
                                    for kc in 0..nfk {
                                        let kidx = ck * nfk + kc;
                                        for jc in 0..nfj {
                                            let jidx = cj * nfj + jc;
                                            for ic in 0..nfi {
                                                let iidx = ci * nfi + ic;
                                                let src = ic + nfi * (jc + nfj * (kc + nfk * lc));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
                                                staging[dst] = F::from_f64_lossy(block[src]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => unreachable!("spinor int2e_ip1 rejected above"),
    }

    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;

    let staging_bytes = staging.len() * std::mem::size_of::<F>();
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

/// int2e_ip2 gradient launcher (Phase 23 DRV1-01).
///
/// Sibling of [`launch_two_electron_ip1`] for the **ket** bra-center `k`
/// (`G2E_D_K`, libcint `CINTgout2e_int2e_ip2`, grad2.c:101). The only differences
/// vs ip1 are:
///   - headroom raised on `lk` (`build_2e_shape(li, lj, lk+1, ll)`) so
///     `nabla1k_2e` can read up to index `lk+1`;
///   - the single-side contraction uses [`crate::kernels::f12::gout_ipn`] with
///     `Nabla1Center::K` and the per-primitive **k-shell** exponent `ak`.
/// The s[0..2] mixing, the component-leading transpose, and the cart/sph output
/// path are identical to ip1.
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_ip2<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Spinor gradient: not supported (R5 / D-06). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int2e_ip2 gradient".to_owned(),
        });
    }

    // lk → lk+1 headroom shape (D-06). gout_ipn's nabla1k_2e reads up to index lk+1.
    let grad_shape = build_2e_shape(li as usize, lj as usize, lk as usize + 1, ll as usize);

    // Phase 25 FND-02: HOST gradient path (fill_g_tensor_2e → rys_roots_host). The host
    // Rys engine supports nroots 6..12 (rys_wheeler.rs); route Hessian-elevated quartets
    // here instead of UnsupportedApi. Ceiling = vendor-validated 12; nroots>12 fail-closed.
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = 3 * block_len; // 3 components × Cartesian AO product

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

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

                    // Plain Coulomb G-tensor at the elevated lk (lk+1 headroom).
                    let g = fill_g_tensor_2e(
                        ai, aj, ak, al, &ri, &rj, &rk, &rl, grad_shape, quartet_fac,
                    );

                    // ∇ on the ket bra-center k (Nabla1Center::K, exponent ak).
                    // gout_ipn is called at BASE lk (the G-tensor carries lk+1 headroom).
                    let gout = crate::kernels::f12::gout_ipn(
                        &g,
                        &grad_f12_shape,
                        li as usize,
                        lj as usize,
                        lk as usize,
                        ll as usize,
                        crate::kernels::f12::Nabla1Center::K,
                        ak,
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
                                    let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l
                                        + cl)
                                        * total_len;
                                    for n in 0..block_len {
                                        for comp in 0..3usize {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * 3 + comp];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Component-leading `[3, nl, nk, nj, ni]` F-order write (identical to ip1).
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
                                                staging[dst] = F::from_f64_lossy(sph[src]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let block = &cart_blocks[base..base + block_len];
                                for lc in 0..nfl {
                                    let lidx = cl * nfl + lc;
                                    for kc in 0..nfk {
                                        let kidx = ck * nfk + kc;
                                        for jc in 0..nfj {
                                            let jidx = cj * nfj + jc;
                                            for ic in 0..nfi {
                                                let iidx = ci * nfi + ic;
                                                let src = ic + nfi * (jc + nfj * (kc + nfk * lc));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
                                                staging[dst] = F::from_f64_lossy(block[src]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => unreachable!("spinor int2e_ip2 rejected above"),
    }

    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;

    let staging_bytes = staging.len() * std::mem::size_of::<F>();
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

/// Generic inner for the 2e launcher. See `launch_two_electron` for the dispatch rationale.
///
/// Intermediate computations (G-tensor, cart_buf) remain `f64`; output staging
/// is written via `F::from_f64_lossy`. The `f64` monomorphization is byte-identical
/// to the pre-generic code.
fn launch_two_electron_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: format!(
                "canonical_family mismatch for 2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    let shells = plan.shells.as_slice();
    if shells.len() < 4 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: "2e kernel requires exactly 4 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;
    let ll = shell_l.ang_momentum;

    let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);

    // Keep branch logic explicit for auditability against libcint Pitfall 1.
    let _ibase_kbase_used = (shape.ibase, shape.kbase);

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    let rl = atoms[shell_l.atom_index as usize].coord_bohr;

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let block_len = nfi * nfj * nfk * nfl;

    // Pitfall 2: all four common_fac_sp factors are required for 2e.
    let sp_factor = common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk) * common_fac_sp(ll);
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * sp_factor;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    // ─────────────────────────────────────────────────────────────────────────
    // int2e_ip1 gradient path (Plan 21-05 / GRAD-07).
    //
    // The two-electron force ∇_A <ij|kl> — the highest-impact term in every
    // analytical HF/DFT/MP2/CCSD gradient. The first-derivative math is the
    // standard libcint `∂/∂A χ_l = -2α·χ_{l+1} + l·χ_{l-1}` (`CINTnabla1i_2e`),
    // reused VERBATIM from f12.rs (`gout_ip1` / `nabla1i_2e`, made pub(crate) in
    // Task 0). The only difference vs the F12 gradient is the G-tensor source:
    // here we feed `gout_ip1` the PLAIN Coulomb G-tensor from `fill_g_tensor_2e`
    // (`rys_roots_host`) instead of the F12 stg-roots tensor (D-04).
    //
    // The G-tensor is built with `li_ceil = li + 1` headroom so `nabla1i_2e` can
    // read up to index li+1; `gout_ip1` is then called at BASE li (the documented
    // headroom recipe from f12.rs:584-585 + 1163-1168).
    //
    // Output is component-leading `[3, nl, nk, nj, ni]` F-order: `gout_ip1`
    // returns interleaved `out[n*3+comp]` where `n` walks `[cl, ck, cj, ci]`
    // (ll slowest, li fastest); we TRANSPOSE to `staging[comp*block + n]`
    // matching pyscf-gto `layout_table.rs` (Risk R3, validated vs vendor in the
    // oracle test).
    if plan.descriptor.operator_name() == "ip1" {
        return launch_two_electron_ip1::<F>(
            plan,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // int2e_ip2 gradient path (Phase 23 DRV1-01): ∇ on the ket bra-center k.
    if plan.descriptor.operator_name() == "ip2" {
        return launch_two_electron_ip2::<F>(
            plan,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar 2e device dispatch (quick task 260529-q4k).
    //
    // The per-(ci,cj,ck,cl) i-fastest Cartesian block accumulation
    // (`fill_g_tensor_2e` → `contract_2e_cart`, weighted + summed over every
    // primitive and contraction quad) now runs in the `two_electron_scalar_kernel`
    // `#[cube(launch)]` device kernel via `run_2e_scalar_device`, dispatched on
    // the resolved backend (CPU / Wgpu / Cuda / ROCm / Metal). The returned
    // `cart_blocks` has the IDENTICAL layout the host loop produced (block
    // (ci,cj,ck,cl) at offset (((ci*n_ctr_j+cj)*n_ctr_k+ck)*n_ctr_l+cl)*block_len,
    // i fastest within each block), so the representation-dispatch host scatter
    // below (cart_to_sph_2e / cart_to_spinor_sf_4d + contraction-major AO scatter)
    // consumes it unchanged — the host part of the honest host/device split.
    // ─────────────────────────────────────────────────────────────────────────

    // Fail-closed nroots guard BEFORE any dispatch (mirrors 2c2e's MAX_DEVICE_NROOTS):
    // scalar nroots = (li+lj+lk+ll)/2+1; >5 means l-sum>8, outside rys_root1..5.
    if shape.nroots > MAX_DEVICE_NROOTS {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: format!(
                "device 2e kernel supports nroots<={MAX_DEVICE_NROOTS} (l_i+l_j+l_k+l_l<=8); \
                 got nroots={} for l=({li},{lj},{lk},{ll})",
                shape.nroots
            ),
        });
    }

    // Flatten the f64 per-shell exps/coeffs the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let exps_l: Vec<f64> = shell_l.exponents[..n_prim_l].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();
    let coeff_l: Vec<f64> = shell_l.coefficients[..n_prim_l * n_ctr_l].to_vec();

    let out_len = n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * block_len;

    let cart_blocks: Vec<f64> = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_2e_scalar_device::<cubecl::cpu::CpuRuntime>(
            client, li as u32, lj as u32, lk as u32, ll as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_prim_l as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32,
            n_ctr_l as u32, shape.di as u32, shape.dk as u32, shape.dl as u32, shape.dj as u32,
            shape.g_size as u32, shape.nmax as u32, shape.mmax as u32, shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32, shape.ibase as u32, shape.kbase as u32, shape.nroots as u32,
            ri, rj, rk, rl, common_factor, &exps_i, &exps_j, &exps_k, &exps_l, &coeff_i, &coeff_j,
            &coeff_k, &coeff_l, out_len,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_2e_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lj as u32, lk as u32, ll as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_prim_l as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32,
            n_ctr_l as u32, shape.di as u32, shape.dk as u32, shape.dl as u32, shape.dj as u32,
            shape.g_size as u32, shape.nmax as u32, shape.mmax as u32, shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32, shape.ibase as u32, shape.kbase as u32, shape.nroots as u32,
            ri, rj, rk, rl, common_factor, &exps_i, &exps_j, &exps_k, &exps_l, &coeff_i, &coeff_j,
            &coeff_k, &coeff_l, out_len,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_2e_scalar_device::<cubecl_cuda::CudaRuntime>(
            client, li as u32, lj as u32, lk as u32, ll as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_prim_l as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32,
            n_ctr_l as u32, shape.di as u32, shape.dk as u32, shape.dl as u32, shape.dj as u32,
            shape.g_size as u32, shape.nmax as u32, shape.mmax as u32, shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32, shape.ibase as u32, shape.kbase as u32, shape.nroots as u32,
            ri, rj, rk, rl, common_factor, &exps_i, &exps_j, &exps_k, &exps_l, &coeff_i, &coeff_j,
            &coeff_k, &coeff_l, out_len,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_2e_scalar_device::<cubecl_hip::HipRuntime>(
            client, li as u32, lj as u32, lk as u32, ll as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_prim_l as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32,
            n_ctr_l as u32, shape.di as u32, shape.dk as u32, shape.dl as u32, shape.dj as u32,
            shape.g_size as u32, shape.nmax as u32, shape.mmax as u32, shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32, shape.ibase as u32, shape.kbase as u32, shape.nroots as u32,
            ri, rj, rk, rl, common_factor, &exps_i, &exps_j, &exps_k, &exps_l, &coeff_i, &coeff_j,
            &coeff_k, &coeff_l, out_len,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_2e_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lj as u32, lk as u32, ll as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_prim_l as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32,
            n_ctr_l as u32, shape.di as u32, shape.dk as u32, shape.dl as u32, shape.dj as u32,
            shape.g_size as u32, shape.nmax as u32, shape.mmax as u32, shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32, shape.ibase as u32, shape.kbase as u32, shape.nroots as u32,
            ri, rj, rk, rl, common_factor, &exps_i, &exps_j, &exps_k, &exps_l, &coeff_i, &coeff_j,
            &coeff_k, &coeff_l, out_len,
        ),
    };

    // Representation dispatch: intermediate transforms use f64 temp buffers;
    // final values cast to F via F::from_f64_lossy.
    match plan.representation {
        Representation::Spheric => {
            // Per-contraction-quad cart→sph, scattered into the contraction-major
            // AO grid. di = n_ctr_i * nsi = shell_i.ao_per_shell(). cart_to_sph_2e
            // emits an i-fastest [nsl][nsk][nsj][nsi] block
            // (sph[mi + nsi*(mj + nsj*(mk + nsk*ml))]); the downstream stitch
            // (pyscf-gto evaluate_arity4) reads F-order i-fastest with AO index
            // ci*nsi+mi, so dst = ii + di*(jj + dj*(kk + dk*ll)). For all-nctr==1
            // (di==nsi, …) this is byte-identical to the prior linear copy.
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    for ck in 0..n_ctr_k {
                        for cl in 0..n_ctr_l {
                            let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                * block_len;
                            let sph =
                                cart_to_sph_2e(&cart_blocks[base..base + block_len], li, lj, lk, ll);
                            for ml in 0..nsl {
                                let lidx = cl * nsl + ml;
                                for mk in 0..nsk {
                                    let kidx = ck * nsk + mk;
                                    for mj in 0..nsj {
                                        let jidx = cj * nsj + mj;
                                        for mi in 0..nsi {
                                            let iidx = ci * nsi + mi;
                                            let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                            let dst = iidx + di * (jidx + dj * (kidx + dk * lidx));
                                            staging[dst] = F::from_f64_lossy(sph[src]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            // Single-contraction spinor preserves the exact prior behavior. General
            // contraction (nctr>1) is not wired for the 4D spinor transform (and is
            // not exercised by the non-relativistic callers); return an explicit
            // error rather than silently truncating to the (0,0,0,0) block.
            if n_ctr_i != 1 || n_ctr_j != 1 || n_ctr_k != 1 || n_ctr_l != 1 {
                return Err(cintxRsError::UnsupportedApi {
                    requested: "spinor 2e with general contraction (nctr>1)".to_owned(),
                });
            }
            // cart_blocks is exactly one nfi*nfj*nfk*nfl block here — identical to
            // the old cart_buf.
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            let kappa_k = shell_k.kappa;
            let kappa_l = shell_l.kappa;
            cart_to_spinor_sf_4d::<F>(
                staging, &cart_blocks,
                li, kappa_i, lj, kappa_j,
                lk, kappa_k, ll, kappa_l,
            )?;
        }
        Representation::Cart => {
            // Each contraction block is i-fastest [nfl][nfk][nfj][nfi]; scatter it
            // into the contraction-major AO grid (di = n_ctr_i*nfi) i-fastest to
            // match the Cart stitch. For all-nctr==1 this is the prior linear copy.
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    for ck in 0..n_ctr_k {
                        for cl in 0..n_ctr_l {
                            let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for lc in 0..nfl {
                                let lidx = cl * nfl + lc;
                                for kc in 0..nfk {
                                    let kidx = ck * nfk + kc;
                                    for jc in 0..nfj {
                                        let jidx = cj * nfj + jc;
                                        for ic in 0..nfi {
                                            let iidx = ci * nfi + ic;
                                            let src = ic + nfi * (jc + nfj * (kc + nfk * lc));
                                            let dst = iidx + di * (jidx + dj * (kidx + dk * lidx));
                                            staging[dst] = F::from_f64_lossy(block[src]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // WR-06: precision-aware sentinel so f32 stale lanes (< f32 noise floor ~1e-7)
    // are not counted. The outer F32 arm already bounds staging to out_elems, so this
    // scan cannot touch stale upper-half lanes.
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = staging.len() * std::mem::size_of::<F>();
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

/// Host-side 2e launcher — outer precision dispatcher.
///
/// Keeps the registered `FamilyLaunchFn` signature unchanged so the `as FamilyLaunchFn`
/// cast in `kernels/mod.rs` compiles. Dispatches on `plan.precision` to the generic inner.
pub fn launch_two_electron(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_two_electron_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            // F32 arm: capture the true output element count BEFORE the bytemuck cast.
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
            launch_two_electron_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar 2e device-vs-host cross-check + f32 genericity (quick task 260529-q4k)
//
// The device kernel must reproduce the host
// `contract_2e_cart(fill_g_tensor_2e(...))` Cartesian buffer for a single
// primitive/contraction pair across all four HRR branches (ibase/kbase ∈
// {true,false} via li>lj / lk>ll), and must compile+launch for F = f32.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod device_tests {
    use super::*;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host single-pair Cartesian reference: fill_g_tensor_2e → contract_2e_cart
    /// for one primitive and one contraction (coeff weights applied).
    #[allow(clippy::too_many_arguments)]
    fn host_cart_2e(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
        ai: f64,
        aj: f64,
        ak: f64,
        al: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rk: [f64; 3],
        rl: [f64; 3],
        common_factor: f64,
        ci: f64,
        cj: f64,
        ck: f64,
        cl: f64,
    ) -> Vec<f64> {
        let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
        let pdata_ij =
            compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let pdata_kl =
            compute_pdata_host(ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0);
        let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;
        let g = fill_g_tensor_2e(ai, aj, ak, al, &ri, &rj, &rk, &rl, shape, quartet_fac);
        let prim_cart = contract_2e_cart(&g, shape, li, lj, lk, ll);
        let weight = ci * cj * ck * cl;
        prim_cart.iter().map(|&v| v * weight).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_device_matches_host(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
        ai: f64,
        aj: f64,
        ak: f64,
        al: f64,
    ) {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 1.1];
        let rk = [0.7_f64, 0.0, 0.0];
        let rl = [0.0_f64, 0.9, 0.0];
        let ci = 0.9_f64;
        let cj = 1.1_f64;
        let ck = 0.8_f64;
        let cl = 1.2_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk)
            * common_fac_sp(ll);
        let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
        let nfi = ncart(li);
        let nfj = ncart(lj);
        let nfk = ncart(lk);
        let nfl = ncart(ll);
        let out_len = nfi * nfj * nfk * nfl;

        let host = host_cart_2e(
            li, lj, lk, ll, ai, aj, ak, al, ri, rj, rk, rl, common_factor, ci, cj, ck, cl,
        );
        let dev = run_2e_scalar_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
            li as u32,
            lj as u32,
            lk as u32,
            ll as u32,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
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
            ri,
            rj,
            rk,
            rl,
            common_factor,
            &[ai],
            &[aj],
            &[ak],
            &[al],
            &[ci],
            &[cj],
            &[ck],
            &[cl],
            out_len,
        );

        assert_eq!(
            host.len(),
            dev.len(),
            "length mismatch for ({li},{lj},{lk},{ll})"
        );
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch ({li},{lj},{lk},{ll}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    // (s,s,s,s): nroots=1, ibase=false, kbase=false → lj2d branch.
    #[test]
    fn test_2e_device_ssss() {
        assert_device_matches_host(0, 0, 0, 0, 1.0, 0.8, 0.9, 1.1);
    }

    // (p,s,s,s): li>lj → ibase=true; lk==ll → kbase=false → il2d branch.
    #[test]
    fn test_2e_device_psss() {
        assert_device_matches_host(1, 0, 0, 0, 0.8, 1.0, 0.9, 1.1);
    }

    // (s,p,s,s): li<lj → ibase=false; kbase=false → lj2d branch.
    #[test]
    fn test_2e_device_spss() {
        assert_device_matches_host(0, 1, 0, 0, 1.0, 0.8, 0.9, 1.1);
    }

    // (p,p,s,s): li==lj → ibase=false, lj2d branch with i+k mixing.
    #[test]
    fn test_2e_device_ppss() {
        assert_device_matches_host(1, 1, 0, 0, 0.8, 0.9, 1.0, 1.1);
    }

    // (d,s,s,s): higher li, ibase=true → il2d branch, nroots=2.
    #[test]
    fn test_2e_device_dsss() {
        assert_device_matches_host(2, 0, 0, 0, 0.7, 1.0, 0.9, 1.1);
    }

    // (s,s,p,s): lk>ll → kbase=true, ibase=false → kj2d branch.
    #[test]
    fn test_2e_device_sspsk() {
        assert_device_matches_host(0, 0, 1, 0, 1.0, 0.8, 0.9, 1.1);
    }

    // (p,s,p,s): ibase=true, kbase=true → ik2d branch (the 4th HRR branch).
    #[test]
    fn test_2e_device_psps() {
        assert_device_matches_host(1, 0, 1, 0, 0.8, 1.0, 0.9, 1.1);
    }

    // (p,p,p,p): nroots=3, full b00 cross-coupling, lj2d branch.
    #[test]
    fn test_2e_device_pppp() {
        assert_device_matches_host(1, 1, 1, 1, 0.8, 0.9, 1.0, 1.1);
    }

    /// Genericity: the kernel compiles AND launches for F = f32. An s-s-s-s f32
    /// launch yields a finite result.
    #[test]
    fn test_two_electron_scalar_kernel_generic_f32() {
        let client = cpu_client();
        let shape = build_2e_shape(0, 0, 0, 0);
        let g_size = shape.g_size;
        let g_zero = vec![0.0_f32; 3 * g_size];
        let rys_zero = vec![0.0_f32; shape.nroots];
        let out_zero = [0.0_f32; 1];

        let one = [1.0_f32];
        let exps_i_h = client.create_from_slice(f32::as_bytes(&one));
        let exps_j_h = client.create_from_slice(f32::as_bytes(&one));
        let exps_k_h = client.create_from_slice(f32::as_bytes(&one));
        let exps_l_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_j_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_k_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_l_h = client.create_from_slice(f32::as_bytes(&one));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let common_factor =
            ((PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(0) * common_fac_sp(0)
                * common_fac_sp(0) * common_fac_sp(0)) as f32;

        two_electron_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_l_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_l_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size) },
            unsafe { ArrayArg::from_raw_parts(u_h, shape.nroots) },
            unsafe { ArrayArg::from_raw_parts(w_h, shape.nroots) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32,
            0.0,
            0.0,
            0.0,
            0.0,
            1.1,
            0.7,
            0.0,
            0.0,
            0.0,
            0.9,
            0.0,
            common_factor,
            PIE4 as f32,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
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
            1u32,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 scalar 2e kernel result must be finite");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// int2e_ip1 gradient tests (Plan 21-05)
//
// The behavior contract (per the plan):
//   - nroots guard: a quartet whose gradient nroots = (li+1+lj+lk+ll)/2+1 > 5
//     (e.g. an all-f quartet) returns UnsupportedApi; an s/p/d quartet does not.
//   - component count: an (s,s,s,s) quartet produces 3 outputs; a (p,s,s,s)
//     quartet produces 3 * 3*1*1*1 = 9.
//   - determinism: repeated evaluation is bit-identical (ordered reduction, D-10).
//   - spinor: int2e_ip1 with Representation::Spinor returns UnsupportedApi (R5).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod ip1_tests {
    use super::*;
    use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
    use crate::specialization::SpecializationKey;
    use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell, ShellTuple};
    use cintx_ops::resolver::Resolver;
    use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
    use std::sync::Arc;

    /// Build a 4-shell same-l quartet plan for the int2e_ip1 sph operator.
    ///
    /// Returns the plan plus a correctly-sized f64 staging buffer (the runtime
    /// planner already multiplies the AO product by the manifest `component_rank=3`).
    fn build_ip1_plan(
        l: u8,
        rep: Representation,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        // Two atoms so the four shells are not all on the same center (a same-center
        // s,s,s,s ERI gradient is nonzero only off-center; off-center keeps the math
        // exercised, but for the unit contract we only need shape/guard behavior).
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());

        let mk = |atom_index: u32| {
            Arc::new(
                Shell::try_new(
                    atom_index,
                    l,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.8_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let s0 = mk(0);
        let s1 = mk(1);
        let s2 = mk(0);
        let s3 = mk(1);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();

        let op = Resolver::descriptor_by_symbol("int2e_ip1_sph")
            .expect("int2e_ip1_sph must be in manifest")
            .id;
        let _ = rep;
        (basis, shells, op)
    }

    /// Build a (li, lj, lk, ll) quartet plan with explicit per-shell angular momenta.
    fn build_ip1_plan_lll(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());

        let mk = |atom_index: u32, l: u8| {
            Arc::new(
                Shell::try_new(
                    atom_index,
                    l,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.8_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let s0 = mk(0, li);
        let s1 = mk(1, lj);
        let s2 = mk(0, lk);
        let s3 = mk(1, ll);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();

        let op = Resolver::descriptor_by_symbol("int2e_ip1_sph")
            .expect("int2e_ip1_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    fn run_ip1(
        basis: &BasisSet,
        shells: ShellTuple,
        op: cintx_core::OperatorId,
        rep: Representation,
    ) -> Result<(Vec<f64>, ExecutionStats), cintxRsError> {
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, rep, basis, shells.clone(), &opts)?;
        let mut plan = ExecutionPlan::new(op, rep, basis, shells, &q)?;
        plan.precision = cintx_core::PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        // Size staging to the planner-declared output element count (includes the
        // 3-component axis via component_rank=3).
        let out_elems = plan.output_layout.staging_elements;
        let mut staging = vec![0.0_f64; out_elems];
        let stats = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok((staging, stats))
    }

    // nroots ceiling (Phase 25 FND-02): the HOST gradient path now supports nroots 6..12
    // via the Wheeler engine. An all-f quartet has gradient nroots = (3+1+3+3+3)/2+1 = 7,
    // which previously returned UnsupportedApi but now routes to the host fill_g_tensor_2e
    // path. The fail-closed ceiling moves to nroots>12 (HOST_RYS_NROOTS_CEILING), e.g. an
    // all-i (l=6) quartet → gradient nroots = (6+1+6+6+6)/2+1 = 13 > 12 → UnsupportedApi.
    #[test]
    fn test_int2e_ip1_nroots_guard() {
        // (f,f,f,f) gradient nroots = 7 ∈ 6..=12 → now ALLOWED via the host Wheeler path.
        let (basis, shells, op) = build_ip1_plan_lll(3, 3, 3, 3);
        let ok = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "all-f int2e_ip1 quartet (nroots=7) must route to the host path (FND-02), got: {:?}",
            ok.err()
        );

        // (i,i,i,i) gradient nroots = (6+1+6+6+6)/2 + 1 = 13 > 12 → fail-closed (T-25-03).
        let (basis, shells, op) = build_ip1_plan_lll(6, 6, 6, 6);
        let result = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "all-i int2e_ip1 quartet (nroots=13 > 12) must return UnsupportedApi, got: {:?}",
            result.map(|(s, _)| s.len())
        );

        // (d,d,d,d) gradient nroots = (2+1+2+2+2)/2 + 1 = 5 → allowed.
        let (basis, shells, op) = build_ip1_plan_lll(2, 2, 2, 2);
        let ok = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "(d,d,d,d) int2e_ip1 quartet (nroots=5) must be allowed, got: {:?}",
            ok.err()
        );
    }

    // Component count: (s,s,s,s) → 3 nonzero-capable outputs; (p,s,s,s) → 9 (sph p = 3).
    #[test]
    fn test_int2e_ip1_component_count() {
        let (basis, shells, op) = build_ip1_plan(0, Representation::Spheric);
        let (staging, _stats) = run_ip1(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(
            staging.len(),
            3,
            "(s,s,s,s) int2e_ip1 should produce 3 components, got {}",
            staging.len()
        );

        let (basis, shells, op) = build_ip1_plan_lll(1, 0, 0, 0);
        let (staging, _stats) = run_ip1(&basis, shells, op, Representation::Spheric).unwrap();
        // sph p = 3 AOs; 3 components × 3×1×1×1 = 9.
        assert_eq!(
            staging.len(),
            9,
            "(p,s,s,s) int2e_ip1 should produce 9 outputs, got {}",
            staging.len()
        );
    }

    // Determinism (D-10): repeated evaluation is bit-identical.
    #[test]
    fn test_int2e_ip1_determinism() {
        let (basis, shells, op) = build_ip1_plan_lll(1, 1, 0, 0);
        let (out1, _) = run_ip1(&basis, shells.clone(), op, Representation::Spheric).unwrap();
        let (out2, _) = run_ip1(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "int2e_ip1 output not bit-identical across two evaluations"
            );
        }
    }

    // Spinor (R5): int2e_ip1 with Representation::Spinor returns UnsupportedApi.
    #[test]
    fn test_int2e_ip1_spinor_unsupported() {
        // Build with sph for a valid workspace query, then force Spinor on the plan.
        let (basis, shells, op) = build_ip1_plan(0, Representation::Spheric);
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = cintx_core::PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 6];
        let result = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "spinor int2e_ip1 should return UnsupportedApi, got: {:?}",
            result
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// int2e_ip2 gradient tests (Phase 23 DRV1-01)
//
// ip2 is the ket-side (∇ on k) sibling of ip1. Same behavior contract:
//   - nroots guard: an all-f quartet (gradient nroots (li+lj+(lk+1)+ll)/2+1=7>5)
//     returns UnsupportedApi; an s/p/d quartet does not.
//   - component count: an (s,s,s,s) quartet → 3 outputs; (s,s,p,s) → 9.
//   - determinism: repeated evaluation is bit-identical.
//   - spinor: int2e_ip2 with Representation::Spinor returns UnsupportedApi.
//   - non-square sanity: an explicitly NON-SQUARE quartet (p on i, p on k in
//     different slots) is evaluated without panic and is nonzero (D-05 discipline).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod ip2_tests {
    use super::*;
    use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
    use crate::specialization::SpecializationKey;
    use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell, ShellTuple};
    use cintx_ops::resolver::Resolver;
    use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
    use std::sync::Arc;

    fn build_ip2_plan_lll(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());

        let mk = |atom_index: u32, l: u8| {
            Arc::new(
                Shell::try_new(
                    atom_index,
                    l,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.8_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let s0 = mk(0, li);
        let s1 = mk(1, lj);
        let s2 = mk(0, lk);
        let s3 = mk(1, ll);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();

        let op = Resolver::descriptor_by_symbol("int2e_ip2_sph")
            .expect("int2e_ip2_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    fn run_ip2(
        basis: &BasisSet,
        shells: ShellTuple,
        op: cintx_core::OperatorId,
        rep: Representation,
    ) -> Result<(Vec<f64>, ExecutionStats), cintxRsError> {
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, rep, basis, shells.clone(), &opts)?;
        let mut plan = ExecutionPlan::new(op, rep, basis, shells, &q)?;
        plan.precision = cintx_core::PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let out_elems = plan.output_layout.staging_elements;
        let mut staging = vec![0.0_f64; out_elems];
        let stats = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok((staging, stats))
    }

    // nroots ceiling (Phase 25 FND-02): the HOST ip2 gradient path now supports nroots
    // 6..12 via the Wheeler engine. An all-f quartet (gradient nroots = (3+3+(3+1)+3)/2+1
    // = 7) routes to the host path; an all-i quartet ((6+6+(6+1)+6)/2+1 = 13 > 12) stays
    // fail-closed (T-25-03).
    #[test]
    fn test_int2e_ip2_nroots_guard() {
        // (f,f,f,f) gradient nroots = 7 ∈ 6..=12 → now ALLOWED via the host Wheeler path.
        let (basis, shells, op) = build_ip2_plan_lll(3, 3, 3, 3);
        let ok = run_ip2(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "all-f int2e_ip2 quartet (nroots=7) must route to the host path (FND-02), got: {:?}",
            ok.err()
        );

        // (i,i,i,i) gradient nroots = (6+6+(6+1)+6)/2 + 1 = 13 > 12 → fail-closed.
        let (basis, shells, op) = build_ip2_plan_lll(6, 6, 6, 6);
        let result = run_ip2(&basis, shells, op, Representation::Spheric);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "all-i int2e_ip2 quartet (nroots=13 > 12) must return UnsupportedApi, got: {:?}",
            result.map(|(s, _)| s.len())
        );

        let (basis, shells, op) = build_ip2_plan_lll(2, 2, 2, 2);
        let ok = run_ip2(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "(d,d,d,d) int2e_ip2 quartet (nroots=5) must be allowed, got: {:?}",
            ok.err()
        );
    }

    // Component count: (s,s,s,s) → 3; (s,s,p,s) → 3 * 1*1*3*1 = 9 (sph p on k).
    #[test]
    fn test_int2e_ip2_component_count() {
        let (basis, shells, op) = build_ip2_plan_lll(0, 0, 0, 0);
        let (staging, _stats) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(staging.len(), 3, "(s,s,s,s) int2e_ip2 should produce 3 components");

        let (basis, shells, op) = build_ip2_plan_lll(0, 0, 1, 0);
        let (staging, _stats) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(staging.len(), 9, "(s,s,p,s) int2e_ip2 should produce 9 outputs");
    }

    // Determinism: repeated evaluation is bit-identical on a NON-SQUARE quartet.
    #[test]
    fn test_int2e_ip2_determinism_nonsquare() {
        // p on i, p on k in different slots → non-square (ni=3, nk=3 but distinct
        // axes) and nonzero off-center.
        let (basis, shells, op) = build_ip2_plan_lll(1, 0, 1, 0);
        let (out1, _) = run_ip2(&basis, shells.clone(), op, Representation::Spheric).unwrap();
        let (out2, _) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out1.len(), out2.len());
        let any_nonzero = out1.iter().any(|v| v.abs() > 1e-14);
        assert!(any_nonzero, "int2e_ip2 (p,s,p,s) output is all-zero (regression)");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "int2e_ip2 output not bit-identical");
        }
    }

    // Electron-exchange symmetry: int2e_ip2(i,j,k,l) must equal int2e_ip1(k,l,i,j)
    // (value multiset; the element ORDER differs because the AO indices permute).
    // This is the kernel-level guard that ip2's ∇_k reproduces ip1's proven ∇_i.
    #[test]
    fn test_int2e_ip2_matches_ip1_electron_swap() {
        // (p,s | s,p) on two atoms — distinct l's so a layout bug would show.
        let (li, lj, lk, ll) = (1u8, 0u8, 0u8, 1u8);
        let (basis, shells, op) = build_ip2_plan_lll(li, lj, lk, ll);
        let (ip2, _) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();

        // int2e_ip1 of the swapped quartet (k,l,i,j).
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());
        let mk = |ai: u32, l: u8| {
            Arc::new(
                Shell::try_new(
                    ai,
                    l,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.8_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        // swapped: i<-k(atom0), j<-l(atom1), k<-i(atom0), l<-j(atom1)
        let s0 = mk(0, lk);
        let s1 = mk(1, ll);
        let s2 = mk(0, li);
        let s3 = mk(1, lj);
        let all: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let b2 = BasisSet::try_new(atoms, all).unwrap();
        let s2t = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();
        let op1 = Resolver::descriptor_by_symbol("int2e_ip1_sph").unwrap().id;

        let opts = ExecutionOptions::default();
        let q = query_workspace(op1, Representation::Spheric, &b2, s2t.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op1, Representation::Spheric, &b2, s2t, &q).unwrap();
        plan.precision = cintx_core::PrecisionKind::F64;
        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut ip1 = vec![0.0_f64; plan.output_layout.staging_elements];
        launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut ip1).unwrap();

        assert_eq!(ip2.len(), ip1.len());
        assert!(ip2.iter().any(|v| v.abs() > 1e-14), "ip2 swap-check is all-zero");
        let round = |v: &f64| (v * 1e10).round() / 1e10;
        let mut a: Vec<f64> = ip2.iter().map(round).collect();
        let mut b: Vec<f64> = ip1.iter().map(round).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b, "int2e_ip2 vs electron-swapped int2e_ip1 value multiset differs");
    }

    // Spinor: int2e_ip2 with Representation::Spinor returns UnsupportedApi.
    #[test]
    fn test_int2e_ip2_spinor_unsupported() {
        let (basis, shells, op) = build_ip2_plan_lll(0, 0, 0, 0);
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = cintx_core::PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 6];
        let result = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "spinor int2e_ip2 should return UnsupportedApi, got: {:?}",
            result
        );
    }
}
