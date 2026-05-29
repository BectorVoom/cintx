//! Host-side 3c2e (three-center two-electron Coulomb) integral kernel.
//!
//! Implements the G-tensor fill + contraction + c2s pipeline following
//! libcint `g3c2e.c` / `cint3c2e.c` with shared 2e recurrence machinery from `g2e.c`.
//!
//! # Pitfall 4 mapping (critical)
//! 3c2e has real shells `(i, j, k)` but libcint reuses 2e machinery by mapping:
//! - 2e "ij side"  <- real `(i, j)`
//! - 2e "kl side"  <- real `k` mapped into the 2e `ll` slot
//! - 2e `lk` slot is a phantom s-function (`lk_ceil = 0`, `ak = 0`)
//! This file follows that mapping explicitly: the third center `k` is treated as
//! the 2e `ll` angular channel, with only one real "ket-side" angular axis.

use crate::backend::ResolvedBackend;
use crate::kernels::two_electron::{build_2e_shape, fill_g_tensor_2e, two_e_shape_as_f12};
use crate::math::pdata::compute_pdata_host;
#[cfg(test)]
use crate::math::pdata::PairData;
#[cfg(test)]
use crate::math::rys::rys_roots_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c2e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_3c2e;
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate.
const MAX_DEVICE_NROOTS: usize = 5;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Matches libcint `CINTcommon_fac_sp(l)`:
///   l=0: 1/(2*sqrt(pi))
///   l=1: sqrt(3/(4*pi))
///   l>=2: 1.0
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Follows libcint `CINTcart_comp` ordering. Host reference (the device kernels
/// reproduce this ordering inline) kept for the host-vs-device cross-checks.
#[cfg(test)]
fn cart_comps(l: u8) -> Vec<(usize, usize, usize)> {
    let mut comps = Vec::new();
    let l = l as i32;
    let mut lx = l;
    while lx >= 0 {
        let mut ly = l - lx;
        while ly >= 0 {
            let lz = l - lx - ly;
            comps.push((lx as usize, ly as usize, lz as usize));
            ly -= 1;
        }
        lx -= 1;
    }
    comps
}

/// Fill the 2d 3c2e G-tensor for one primitive triple (ip, jp, kp).
///
/// The returned tensor is `[gx | gy | gz]` where each axis block has shape:
/// `[m=0..lk][n=0..(li+lj)][root=0..nrys-1]`, root-fastest.
///
/// This is the shared 2e recurrence stage before ij-HRR splitting:
/// - `n` corresponds to combined `(i+j)` angular order
/// - `m` corresponds to real third-center `k` angular order (2e ll-slot)
///
/// Host f64 reference of the exact device algorithm — kept under `#[cfg(test)]`
/// as the device-vs-host cross-check reference (the device kernel inlines it).
#[cfg(test)]
fn fill_g_tensor_3c2e(
    pair: &PairData,
    ak: f64,
    ri: [f64; 3],
    rk: [f64; 3],
    li: u8,
    lj: u8,
    lk: u8,
    nrys_roots: usize,
    fac_env: f64,
) -> Vec<f64> {
    let nmax = li as usize + lj as usize;
    let mmax = lk as usize;
    let dn = nrys_roots;
    let dm = nrys_roots * (nmax + 1);
    let g_size = nrys_roots * (nmax + 1) * (mmax + 1);

    let mut g = vec![0.0_f64; 3 * g_size];

    let aij = pair.zeta_ab;
    let akl = ak; // 3c2e mapping: 2e "kl" pair uses only the real k shell (l-slot), phantom k-slot has exponent 0.
    let p = [pair.center_p_x, pair.center_p_y, pair.center_p_z];

    // 2e-style pair displacement: rij - rkl with rij=P and rkl=Rk (mapped ll slot).
    let xij_kl = p[0] - rk[0];
    let yij_kl = p[1] - rk[1];
    let zij_kl = p[2] - rk[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

    let a1 = aij * akl;
    let a0 = a1 / (aij + akl);
    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_env;
    let x_rys = a0 * rr;
    let (u_roots, w_weights) = rys_roots_host(nrys_roots, x_rys);

    // 3c2e uses 2e recurrence with rx_in_rijrx = Ri and rx_in_rklrx = Rk.
    let rijrx = [p[0] - ri[0], p[1] - ri[1], p[2] - ri[2]];

    for irys in 0..nrys_roots {
        let u2 = a0 * u_roots[irys];
        let tmp4 = 0.5 / (u2 * (aij + akl) + a1);
        let tmp5 = u2 * tmp4;
        let b00 = tmp5;
        let b10 = tmp5 + tmp4 * akl;
        let b01 = tmp5 + tmp4 * aij;

        let tmp2 = 2.0 * tmp5 * akl;
        let tmp3 = 2.0 * tmp5 * aij;
        let c00 = [
            rijrx[0] - tmp2 * xij_kl,
            rijrx[1] - tmp2 * yij_kl,
            rijrx[2] - tmp2 * zij_kl,
        ];
        // rklrx = rkl - rk = 0 for the mapped ll-slot center, so c0p is only the coupling term.
        let c0p = [tmp3 * xij_kl, tmp3 * yij_kl, tmp3 * zij_kl];

        // Base for this Rys root.
        g[irys] = 1.0;
        g[g_size + irys] = 1.0;
        g[2 * g_size + irys] = w_weights[irys] * fac1;

        for axis in 0..3 {
            let axis_off = axis * g_size;
            let c00_axis = c00[axis];
            let c0p_axis = c0p[axis];

            // VRR in combined ij direction (n-axis).
            if nmax > 0 {
                let mut s_prev = g[axis_off + irys];
                let mut s1 = c00_axis * s_prev;
                g[axis_off + irys + dn] = s1;
                for n in 1..nmax {
                    let s2 = c00_axis * s1 + n as f64 * b10 * s_prev;
                    g[axis_off + irys + (n + 1) * dn] = s2;
                    s_prev = s1;
                    s1 = s2;
                }
            }

            // VRR in mapped k(ll)-direction (m-axis), including b00 cross-coupling.
            if mmax > 0 {
                // n=0 ladder over m
                let mut s_prev = g[axis_off + irys];
                let mut s1 = c0p_axis * s_prev;
                g[axis_off + irys + dm] = s1;
                for m in 1..mmax {
                    let s2 = c0p_axis * s1 + m as f64 * b01 * s_prev;
                    g[axis_off + irys + (m + 1) * dm] = s2;
                    s_prev = s1;
                    s1 = s2;
                }

                // n>0 ladders over m with b00 cross term.
                if nmax > 0 {
                    for n in 1..=nmax {
                        let i_off = irys + n * dn;
                        let s0_k0 = g[axis_off + i_off];
                        let prev_i_k0 = g[axis_off + irys + (n - 1) * dn];
                        let mut s1 = c0p_axis * s0_k0 + b00 * prev_i_k0;
                        g[axis_off + i_off + dm] = s1;
                        let mut s_prev = s0_k0;
                        for m in 1..mmax {
                            let prev_i_km = g[axis_off + irys + (n - 1) * dn + m * dm];
                            let s2 = c0p_axis * s1 + m as f64 * b01 * s_prev + b00 * prev_i_km;
                            g[axis_off + i_off + (m + 1) * dm] = s2;
                            s_prev = s1;
                            s1 = s2;
                        }
                    }
                }
            }
        }
    }

    g
}

/// Split ij angular momentum for ibase=true layout.
///
/// Input `n` channel is the ij-base ladder (i-like axis) from 2e-style VRR.
/// We recover explicit `(i,j)` channels via HRR transfer along j:
/// `g(i,j,...) = (Ri-Rj) * g(i,j-1,...) + g(i+1,j-1,...)`.
///
/// Input:  `[axis][m][n][root]` from `fill_g_tensor_3c2e`
/// Output: `[axis][root][k][j][i]` (i fastest inside each root block).
///
/// Host f64 reference — kept under `#[cfg(test)]` (the device kernel inlines it).
#[cfg(test)]
fn split_ij_hrr(
    g2d: &[f64],
    li: u8,
    lj: u8,
    lk: u8,
    nrys_roots: usize,
    rirj: [f64; 3],
) -> Vec<f64> {
    let nmax = li as usize + lj as usize;
    let mmax = lk as usize;
    let dn = nrys_roots;
    let dm = nrys_roots * (nmax + 1);
    let g2d_size = nrys_roots * (nmax + 1) * (mmax + 1);

    let ni = li as usize + 1;
    let nj = lj as usize + 1;
    let nk = lk as usize + 1;
    let axis_size = nrys_roots * nk * nj * ni;
    let mut out = vec![0.0_f64; 3 * axis_size];

    let work_stride = nmax + 1;
    for axis in 0..3 {
        let axis_in_off = axis * g2d_size;
        let axis_out_off = axis * axis_size;

        for k in 0..=mmax {
            for root in 0..nrys_roots {
                // Work rows are j (0..lj), columns are i-base index (0..li+lj).
                let mut work = vec![0.0_f64; nj * work_stride];
                for i in 0..=nmax {
                    work[i] = g2d[axis_in_off + root + i * dn + k * dm];
                }

                for j in 1..=lj as usize {
                    let prev = (j - 1) * work_stride;
                    let cur = j * work_stride;
                    let i_max = nmax - j;
                    for i in 0..=i_max {
                        work[cur + i] = rirj[axis] * work[prev + i] + work[prev + i + 1];
                    }
                }

                for j in 0..=lj as usize {
                    for i in 0..=li as usize {
                        let out_idx = ((root * nk + k) * nj + j) * ni + i;
                        out[axis_out_off + out_idx] = work[j * work_stride + i];
                    }
                }
            }
        }
    }

    out
}

/// Contract HRR-split G-tensor into Cartesian integral buffer.
///
/// Output layout: i fastest, j middle, k slowest:
/// `out[(k * ncj + j) * nci + i]`.
///
/// Host f64 reference — kept under `#[cfg(test)]` (the device kernel inlines it).
#[cfg(test)]
fn contract_3c2e(g: &[f64], li: u8, lj: u8, lk: u8, nrys_roots: usize) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);

    let ni = li as usize + 1;
    let nj = lj as usize + 1;
    let nk = lk as usize + 1;
    let axis_size = nrys_roots * nk * nj * ni;

    let gx_off = 0usize;
    let gy_off = axis_size;
    let gz_off = 2 * axis_size;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);

    let mut out = vec![0.0_f64; nci * ncj * nck];

    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let mut val = 0.0_f64;
                for root in 0..nrys_roots {
                    let idx_x = ((root * nk + kx) * nj + jx) * ni + ix;
                    let idx_y = ((root * nk + ky) * nj + jy) * ni + iy;
                    let idx_z = ((root * nk + kz) * nj + jz) * ni + iz;
                    val += g[gx_off + idx_x] * g[gy_off + idx_y] * g[gz_off + idx_z];
                }
                out[(k_idx * ncj + j_idx) * nci + i_idx] += val;
            }
        }
    }

    out
}

/// Transpose a flat 3-index buffer from `(i,j,k)` to `(j,i,k)` ordering.
///
/// Input/output are both i-fastest, then j, then k slowest:
/// `idx = (k * nj + j) * ni + i`.
fn transpose_ij_3idx(buf: &[f64], ni: usize, nj: usize, nk: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; buf.len()];
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let src = (k * nj + j) * ni + i;
                let dst = (k * ni + i) * nj + j;
                out[dst] = buf[src];
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scalar 3c2e device kernel — `#[cube(launch)]`, generic over `F: Float`
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar 3c2e G-tensor fill (2e recurrence) + ij-HRR split + Cartesian
/// contraction for one shell triple, on-device.
///
/// Single work item (`UNIT_POS == 0`) — a faithful, correctness-first port of the
/// host pipeline (`fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`) inlined
/// per the `#[cube]` authoring rules (no plain-fn calls; the device `rys_root{1..5}`
/// are the only callees). The kernel runs in canonical `li>=lj` order — the host
/// launcher decides the `swap_ij` and transposes the read-back buffer.
///
/// `#[comptime] nroots` selects the `rys_root{1..5}` device function at JIT time.
///
/// Layout of `g` (size `3 * g_size`, the 2e-style 2D fill, root-fastest):
/// `g[axis*g_size + m*dm + n*dn + root]` with `dn = nrys`,
/// `dm = nrys*(nmax+1)`, `g_size = nrys*(nmax+1)*(mmax+1)`, `nmax = li+lj`,
/// `mmax = lk`.
///
/// Layout of `g_split` (size `3 * split_size`, after the j-HRR transfer):
/// `g_split[axis*split_size + ((root*nk + k)*nj + j)*ni + i]` with `ni = li+1`,
/// `nj = lj+1`, `nk = lk+1`, `split_size = nrys*nk*nj*ni`.
///
/// `cart_out` (size `nci*ncj*nck`, i fastest, k slowest) is zeroed in-kernel and
/// accumulated over all primitive and contraction triples.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn center_3c2e_scalar_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    exps_k: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    coeff_k: &Array<F>,
    g: &mut Array<F>,
    g_split: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    work: &mut Array<F>,
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
    common_factor: F,
    pie4: F,
    li: u32,
    lj: u32,
    lk: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let nrys = nroots;
        let nmax = li + lj;
        let mmax = lk;
        let dn = nrys;
        let dm = nrys * (nmax + 1u32);
        let g_size = nrys * (nmax + 1u32) * (mmax + 1u32);
        let total_g = 3u32 * g_size;

        let ni = li + 1u32;
        let nj = lj + 1u32;
        let nk = lk + 1u32;
        let split_size = nrys * nk * nj * ni;
        let total_split = 3u32 * split_size;
        let work_stride = nmax + 1u32;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
        let out_len = nci * ncj * nck;

        // Zero the accumulation buffer.
        let mut oi = 0u32;
        while oi < out_len {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // rirj = ri - rj (for the j-HRR transfer).
        let rirj_x = rix - rjx;
        let rirj_y = riy - rjy;
        let rirj_z = riz - rjz;

        let mut kp = 0u32;
        while kp < nprim_k {
            let ak = exps_k[kp as usize];
            let mut jp = 0u32;
            while jp < nprim_j {
                let aj = exps_j[jp as usize];
                let mut ip = 0u32;
                while ip < nprim_i {
                    let ai = exps_i[ip as usize];

                    // ── Inlined Gaussian-product pdata (compute_pdata_host) ──
                    // zeta_ab = ai+aj; center_p = (ai*ri+aj*rj)/zeta_ab;
                    // fac = exp(-ai*aj/zeta_ab * |ri-rj|^2).
                    let zeta_ab = ai + aj;
                    let px = (ai * rix + aj * rjx) / zeta_ab;
                    let py = (ai * riy + aj * rjy) / zeta_ab;
                    let pz = (ai * riz + aj * rjz) / zeta_ab;
                    let rij_x = rix - rjx;
                    let rij_y = riy - rjy;
                    let rij_z = riz - rjz;
                    let rr_ij = rij_x * rij_x + rij_y * rij_y + rij_z * rij_z;
                    let pair_fac = F::exp(-ai * aj / zeta_ab * rr_ij);

                    // 2e-style pair: aij = zeta_ab, akl = ak.
                    let aij = zeta_ab;
                    let akl = ak;
                    // Displacement P - Rk.
                    let xij_kl = px - rkx;
                    let yij_kl = py - rky;
                    let zij_kl = pz - rkz;
                    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

                    let a1 = aij * akl;
                    let a0 = a1 / (aij + akl);
                    let fac_env = common_factor * pair_fac;
                    let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * fac_env;
                    let x_rys = a0 * rr;

                    // rijrx = P - Ri (the bra-side reference displacement).
                    let rijrx_x = px - rix;
                    let rijrx_y = py - riy;
                    let rijrx_z = pz - riz;

                    // Rys roots/weights for this primitive triple.
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

                    // ── Fill the 2D G-tensor (fill_g_tensor_3c2e) ──────────────
                    let mut gi = 0u32;
                    while gi < total_g {
                        g[gi as usize] = F::new(0.0);
                        gi += 1u32;
                    }

                    let mut irys = 0u32;
                    while irys < nrys {
                        let u2 = a0 * urys[irys as usize];
                        let tmp4 = F::new(0.5) / (u2 * (aij + akl) + a1);
                        let tmp5 = u2 * tmp4;
                        let b00 = tmp5;
                        let b10 = tmp5 + tmp4 * akl;
                        let b01 = tmp5 + tmp4 * aij;
                        let tmp2 = F::new(2.0) * tmp5 * akl;
                        let tmp3 = F::new(2.0) * tmp5 * aij;

                        // Base case: gx=gy=1, gz=w*fac1.
                        g[irys as usize] = F::new(1.0);
                        g[(g_size + irys) as usize] = F::new(1.0);
                        g[(2u32 * g_size + irys) as usize] = wrys[irys as usize] * fac1;

                        let mut axis = 0u32;
                        while axis < 3u32 {
                            let base = axis * g_size;
                            // Per-axis displacement components.
                            let mut d = xij_kl;
                            let mut rx = rijrx_x;
                            if axis == 1u32 {
                                d = yij_kl;
                                rx = rijrx_y;
                            }
                            if axis == 2u32 {
                                d = zij_kl;
                                rx = rijrx_z;
                            }
                            let c00a = rx - tmp2 * d;
                            let c0pa = tmp3 * d;

                            // VRR in combined ij direction (n-axis), nmax = li+lj.
                            if nmax >= 1u32 {
                                let mut s_prev = g[(base + irys) as usize];
                                let mut s1 = c00a * s_prev;
                                g[(base + irys + dn) as usize] = s1;
                                let mut n = 1u32;
                                while n < nmax {
                                    let s2 = c00a * s1 + F::cast_from(n) * b10 * s_prev;
                                    g[(base + irys + (n + 1u32) * dn) as usize] = s2;
                                    s_prev = s1;
                                    s1 = s2;
                                    n += 1u32;
                                }
                            }

                            // VRR in mapped k(ll)-direction (m-axis), mmax = lk.
                            if mmax >= 1u32 {
                                let mut s_prev = g[(base + irys) as usize];
                                let mut s1 = c0pa * s_prev;
                                g[(base + irys + dm) as usize] = s1;
                                let mut m = 1u32;
                                while m < mmax {
                                    let s2 = c0pa * s1 + F::cast_from(m) * b01 * s_prev;
                                    g[(base + irys + (m + 1u32) * dm) as usize] = s2;
                                    s_prev = s1;
                                    s1 = s2;
                                    m += 1u32;
                                }

                                // n>0 ladders over m with b00 cross term.
                                if nmax >= 1u32 {
                                    let mut n = 1u32;
                                    while n <= nmax {
                                        let i_off = irys + n * dn;
                                        let s0_k0 = g[(base + i_off) as usize];
                                        let prev_i_k0 =
                                            g[(base + irys + (n - 1u32) * dn) as usize];
                                        let mut s1 = c0pa * s0_k0 + b00 * prev_i_k0;
                                        g[(base + i_off + dm) as usize] = s1;
                                        let mut s_prev = s0_k0;
                                        let mut m = 1u32;
                                        while m < mmax {
                                            let prev_i_km = g[(base
                                                + irys
                                                + (n - 1u32) * dn
                                                + m * dm)
                                                as usize];
                                            let s2 = c0pa * s1
                                                + F::cast_from(m) * b01 * s_prev
                                                + b00 * prev_i_km;
                                            g[(base + i_off + (m + 1u32) * dm) as usize] = s2;
                                            s_prev = s1;
                                            s1 = s2;
                                            m += 1u32;
                                        }
                                        n += 1u32;
                                    }
                                }
                            }

                            axis += 1u32;
                        }

                        irys += 1u32;
                    }

                    // ── split_ij_hrr: recover (i,j) channels via j-HRR ─────────
                    let mut gsi = 0u32;
                    while gsi < total_split {
                        g_split[gsi as usize] = F::new(0.0);
                        gsi += 1u32;
                    }

                    let mut axis2 = 0u32;
                    while axis2 < 3u32 {
                        let axis_in_off = axis2 * g_size;
                        let axis_out_off = axis2 * split_size;
                        let mut rirj = rirj_x;
                        if axis2 == 1u32 {
                            rirj = rirj_y;
                        }
                        if axis2 == 2u32 {
                            rirj = rirj_z;
                        }

                        let mut k = 0u32;
                        while k <= mmax {
                            let mut root = 0u32;
                            while root < nrys {
                                // Load the i-base ladder into `work` (rows = j, cols = i-base).
                                let mut i = 0u32;
                                while i <= nmax {
                                    work[i as usize] =
                                        g[(axis_in_off + root + i * dn + k * dm) as usize];
                                    i += 1u32;
                                }

                                // HRR transfer along j.
                                let mut j = 1u32;
                                while j <= lj {
                                    let prev = (j - 1u32) * work_stride;
                                    let cur = j * work_stride;
                                    let i_max = nmax - j;
                                    let mut i2 = 0u32;
                                    while i2 <= i_max {
                                        work[(cur + i2) as usize] = rirj
                                            * work[(prev + i2) as usize]
                                            + work[(prev + i2 + 1u32) as usize];
                                        i2 += 1u32;
                                    }
                                    j += 1u32;
                                }

                                // Scatter (i in 0..=li, j in 0..=lj) into g_split.
                                let mut jj = 0u32;
                                while jj <= lj {
                                    let mut ii = 0u32;
                                    while ii <= li {
                                        let out_idx = ((root * nk + k) * nj + jj) * ni + ii;
                                        g_split[(axis_out_off + out_idx) as usize] =
                                            work[(jj * work_stride + ii) as usize];
                                        ii += 1u32;
                                    }
                                    jj += 1u32;
                                }

                                root += 1u32;
                            }
                            k += 1u32;
                        }
                        axis2 += 1u32;
                    }

                    // ── contract_3c2e: triple cart_comps contraction ───────────
                    // Output i fastest, j middle, k slowest.
                    let gx_off = 0u32;
                    let gy_off = split_size;
                    let gz_off = 2u32 * split_size;

                    // Contraction coefficients (canonical nctr handling: one block).
                    let mut cci = 0u32;
                    while cci < nctr_i {
                        let coeff_i_val = coeff_i[(ip * nctr_i + cci) as usize];
                        let mut ccj = 0u32;
                        while ccj < nctr_j {
                            let coeff_j_val = coeff_j[(jp * nctr_j + ccj) as usize];
                            let mut cck = 0u32;
                            while cck < nctr_k {
                                let coeff_k_val = coeff_k[(kp * nctr_k + cck) as usize];
                                let weight = coeff_i_val * coeff_j_val * coeff_k_val;

                                // k cart triples (descending nested-while), k slowest.
                                let mut k_idx = 0u32;
                                let mut ka = 0u32;
                                while ka <= lk {
                                    let kx = lk - ka;
                                    let lk_minus_kx = lk - kx;
                                    let mut kb = 0u32;
                                    while kb <= lk_minus_kx {
                                        let ky = lk_minus_kx - kb;
                                        let kz = lk - kx - ky;

                                        // j cart triples.
                                        let mut j_idx = 0u32;
                                        let mut ja = 0u32;
                                        while ja <= lj {
                                            let jx = lj - ja;
                                            let lj_minus_jx = lj - jx;
                                            let mut jb = 0u32;
                                            while jb <= lj_minus_jx {
                                                let jy = lj_minus_jx - jb;
                                                let jz = lj - jx - jy;

                                                // i cart triples (i fastest).
                                                let mut i_idx = 0u32;
                                                let mut ia = 0u32;
                                                while ia <= li {
                                                    let ix = li - ia;
                                                    let li_minus_ix = li - ix;
                                                    let mut ib = 0u32;
                                                    while ib <= li_minus_ix {
                                                        let iy = li_minus_ix - ib;
                                                        let iz = li - ix - iy;

                                                        let mut val = F::new(0.0);
                                                        let mut root2 = 0u32;
                                                        while root2 < nrys {
                                                            let idx_x = ((root2 * nk + kx) * nj
                                                                + jx)
                                                                * ni
                                                                + ix;
                                                            let idx_y = ((root2 * nk + ky) * nj
                                                                + jy)
                                                                * ni
                                                                + iy;
                                                            let idx_z = ((root2 * nk + kz) * nj
                                                                + jz)
                                                                * ni
                                                                + iz;
                                                            val += g_split
                                                                [(gx_off + idx_x) as usize]
                                                                * g_split
                                                                    [(gy_off + idx_y) as usize]
                                                                * g_split
                                                                    [(gz_off + idx_z) as usize];
                                                            root2 += 1u32;
                                                        }
                                                        let out_idx =
                                                            (k_idx * ncj + j_idx) * nci + i_idx;
                                                        cart_out[out_idx as usize] +=
                                                            weight * val;

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

                                cck += 1u32;
                            }
                            ccj += 1u32;
                        }
                        cci += 1u32;
                    }

                    ip += 1u32;
                }
                jp += 1u32;
            }
            kp += 1u32;
        }
    }
}

/// Dispatch [`center_3c2e_scalar_kernel`] at `f64` on a resolved backend's client
/// and read back the Cartesian accumulation buffer (`nci*ncj*nck`, i fastest).
///
/// Generic over `R: Runtime` so the same path serves CPU, ROCm, etc. Intermediate
/// device compute is `f64` (module precision policy). Runs in canonical `li>=lj`
/// order — the caller decides the `swap_ij` and transposes the read-back.
#[allow(clippy::too_many_arguments)]
fn run_3c2e_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    lk: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    nroots: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let lk_u = lk as usize;
    let nroots_u = nroots as usize;
    let nmax = li_u + lj_u;
    let mmax = lk_u;
    let g_size = nroots_u * (nmax + 1) * (mmax + 1);
    let split_size = nroots_u * (lk_u + 1) * (lj_u + 1) * (li_u + 1);
    let work_len = (lj_u + 1) * (nmax + 1);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let nck = (lk_u + 1) * (lk_u + 2) / 2;
    let out_len = nci * ncj * nck;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let exps_k_h = client.create_from_slice(f64::as_bytes(exps_k));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let coeff_k_h = client.create_from_slice(f64::as_bytes(coeff_k));

    let g_zero = vec![0.0_f64; 3 * g_size];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let gs_zero = vec![0.0_f64; 3 * split_size];
    let gs_h = client.create_from_slice(f64::as_bytes(&gs_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let work_zero = vec![0.0_f64; work_len];
    let work_h = client.create_from_slice(f64::as_bytes(&work_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    center_3c2e_scalar_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(exps_i_h, exps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_j_h, exps_j.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_k_h, exps_k.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_j_h, coeff_j.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_k_h, coeff_k.len()) },
        unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size) },
        unsafe { ArrayArg::from_raw_parts(gs_h, 3 * split_size) },
        unsafe { ArrayArg::from_raw_parts(u_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(w_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(work_h, work_len) },
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
        common_factor,
        PIE4,
        li,
        lj,
        lk,
        nprim_i,
        nprim_j,
        nprim_k,
        nctr_i,
        nctr_j,
        nctr_k,
        nroots,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Generic inner for the 3c2e launcher.
///
/// Contains the full algorithm of `launch_center_3c2e` parameterized over the
/// output float type `F: CintFloat`. Intermediate computations (G-tensor, cart_buf)
/// remain `f64`; precision conversion happens only at the final staging write via
/// `F::from_f64_lossy`. Preserves the li>=lj canonicalization + transpose-back.
/// `int3c2e_ip1` gradient launch — the `∇_A` first-center derivative of the
/// three-center two-electron Coulomb integral (GRAD-08 / Risk R1).
///
/// Mirrors `two_electron.rs::launch_two_electron_ip1`, applying the 3c2e Pitfall-4
/// kl mapping (file header lines 6-12):
///   - 2e "ij side"  ← real `(i, j)` (the bra; `i` raised to `li+1` for `∇_i`)
///   - 2e `ll` slot   ← real `k` (the only real ket-side angular axis)
///   - 2e `lk` slot   ← phantom s-function (`lk_ceil = 0`, exponent `ak = 0`)
///
/// Builds the plain Coulomb G-tensor through the SHARED 2e recurrence
/// ([`fill_g_tensor_2e`]) with `li_ceil = li+1` headroom, reuses
/// [`crate::kernels::f12::gout_ip1`] verbatim, and emits 3-component component-leading
/// `[3, nk, nj, ni]` F-order (same convention as `int2e_ip1`).
///
/// Guards (fail-closed):
///   - `Representation::Spinor` → `UnsupportedApi` (R5 / T-21-06-04).
///   - `grad_shape.nroots > 5` → `UnsupportedApi` (R2 / T-21-06-04): the `li→li+1`
///     raise can push high-l triples past the rys_root1..5 ceiling; reject BEFORE
///     any rys dispatch.
///
/// No `swap_ij` canonicalization: the derivative acts on the first (`i`) shell, and
/// the output keeps the caller's `(i, j, k)` shell order.
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_ip1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // R5 / T-21-06-04: spinor gradient is not supported. Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int3c2e_ip1 gradient".to_owned(),
        });
    }

    // 3c2e kl mapping into the 2e shape (Pitfall-4): real k → 2e `ll` slot, phantom
    // 2e `lk` slot = 0; bra `i` raised to `li+1` so `nabla1i_2e` can read index li+1.
    let grad_shape = build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize);

    // R2 / T-21-06-04: the elevated li can push nroots past the rys_root1..5 ceiling.
    // Reject fail-closed BEFORE any rys_roots_host call (which would otherwise panic).
    if grad_shape.nroots > 5 {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    // From CINTinit_int3c2e_EnvVars (same prefactor as the scalar 3c2e path).
    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let block_len = nci * ncj * nck; // per-component Cartesian AO product
    let total_len = 3 * block_len;

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // Per-contraction-triple component-leading Cartesian accumulator.
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    for kp in 0..n_prim_k {
        let ak = shell_k.exponents[kp];
        for jp in 0..n_prim_j {
            let aj = shell_j.exponents[jp];
            for ip in 0..n_prim_i {
                let ai = shell_i.exponents[ip];

                // 3c2e mapping for the 2e G-tensor fill:
                //   bra pair (i, j): ai, aj at ri, rj.
                //   ket pair: phantom 2e `lk` shell (exponent 0, at the real-k center)
                //   in the lk-slot; real k in the 2e `ll` slot (exponent ak at rk).
                //
                // `fill_g_tensor_2e` computes the bra-ket Rys prefactor but NOT the
                // intra-pair Gaussian product factors — those come from the pdata
                // `fac` (g1e.c:134) and must be folded into `fac_env`, exactly as
                // `launch_two_electron_ip1` does (`quartet_fac = common_factor *
                // pdata_ij.fac * pdata_kl.fac`). For the phantom-real_k ket pair the
                // product factor is `exp(-0) = 1`, but we compute it for fidelity.
                let pdata_ij = compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );
                let pdata_kl = compute_pdata_host(
                    0.0, ak, rk[0], rk[1], rk[2], rk[0], rk[1], rk[2], 1.0, 1.0,
                );
                let fac_env = common_factor * pdata_ij.fac * pdata_kl.fac;
                let g = fill_g_tensor_2e(
                    ai, aj, 0.0, ak, &ri, &rj, &rk, &rk, grad_shape, fac_env,
                );

                // Reuse gout_ip1 verbatim (f12.rs). Called at BASE li (the G-tensor
                // carries the li+1 headroom). With the phantom 2e lk=0, the gout
                // n-walk [ll, lk, lj, li] collapses to [real_k, (phantom size 1),
                // j, i] → effectively [k][j][i] (i fastest), matching the scalar
                // 3c2e cart layout the cart_to_sph_3c2e transform expects.
                let gout = crate::kernels::f12::gout_ip1(
                    &g,
                    &grad_f12_shape,
                    li as usize,
                    lj as usize,
                    0, // phantom 2e lk slot
                    lk as usize, // real k in the 2e ll slot
                    ai,
                );

                for ci in 0..n_ctr_i {
                    let coeff_i = shell_i.coefficients[ip * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let coeff_j = shell_j.coefficients[jp * n_ctr_j + cj];
                        for ck in 0..n_ctr_k {
                            let coeff_k = shell_k.coefficients[kp * n_ctr_k + ck];
                            let weight = coeff_i * coeff_j * coeff_k;
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len;
                            // TRANSPOSE interleaved gout[n*3+comp] into component-leading.
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

    // Write component-leading `[3, nk, nj, ni]` F-order to staging. Per component,
    // the per-triple block is the i-fastest `[nk][nj][ni]` Cartesian tensor — run
    // the cart→sph 3c2e transform per component for the sph rep.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let sph_block = di * dj * dk;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len
                                + comp * block_len;
                            let sph = cart_to_sph_3c2e(
                                &cart_blocks[base..base + block_len],
                                li,
                                lj,
                                lk,
                            );
                            for mk in 0..nsk {
                                let kidx = ck * nsk + mk;
                                for mj in 0..nsj {
                                    let jidx = cj * nsj + mj;
                                    for mi in 0..nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * mk);
                                        let dst = staging_comp_base
                                            + iidx
                                            + di * (jidx + dj * kidx);
                                        if dst < staging.len() {
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
        Representation::Cart => {
            let di = n_ctr_i * nci;
            let dj = n_ctr_j * ncj;
            let dk = n_ctr_k * nck;
            let cart_block = di * dj * dk;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len
                                + comp * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for kc in 0..nck {
                                let kidx = ck * nck + kc;
                                for jc in 0..ncj {
                                    let jidx = cj * ncj + jc;
                                    for ic in 0..nci {
                                        let iidx = ci * nci + ic;
                                        let src = ic + nci * (jc + ncj * kc);
                                        let dst = staging_comp_base
                                            + iidx
                                            + di * (jidx + dj * kidx);
                                        if dst < staging.len() {
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
        Representation::Spinor => unreachable!("spinor int3c2e_ip1 rejected above"),
    }

    // Per-symbol nonzero sentinel (precision-aware; matches the scalar path).
    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
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

fn launch_center_3c2e_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "3c2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c2e",
            detail: format!(
                "canonical_family mismatch for 3c2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    let shells = plan.shells.as_slice();
    if shells.len() < 3 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c2e",
            detail: format!(
                "3c2e kernel requires exactly 3 shells, got {}",
                shells.len()
            ),
        });
    }

    let shell_i_in = &shells[0];
    let shell_j_in = &shells[1];
    let shell_k = &shells[2];

    let li_in = shell_i_in.ang_momentum;
    let lj_in = shell_j_in.ang_momentum;
    let lk = shell_k.ang_momentum;

    // int3c2e_ip1 gradient path (Plan 21-06 / GRAD-08 / Risk R1).
    //
    // The scalar path below is operator-blind and silently returns the PLAIN 3c2e
    // integral; pyscf-grad's DF-gradient runtime consumes `int3c2e_ip1` as the
    // `∇_A` first-center DERIVATIVE. This branch ships the real derivative, reusing
    // `gout_ip1` (f12.rs, made pub(crate) by 21-05) verbatim — the same `∇_i` math
    // as `int2e_ip1` (two_electron.rs). It preserves the 3c2e Pitfall-4 kl mapping
    // (file header lines 6-12): real k is mapped to the 2e `ll` slot and the 2e `lk`
    // slot is a phantom s-function (`lk_ceil = 0`, exponent 0). The G-tensor is built
    // through the SAME 2e recurrence (`fill_g_tensor_2e`) the contraction expects.
    //
    // After 21-02's manifest change `id.operator` for int3c2e_ip1 is "ip1".
    if plan.descriptor.operator_name() == "ip1" {
        return launch_center_3c2e_ip1::<F>(
            plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }

    let swap_ij = li_in < lj_in;
    let (shell_i, shell_j, li, lj) = if swap_ij {
        (shell_j_in, shell_i_in, lj_in, li_in)
    } else {
        (shell_i_in, shell_j_in, li_in, lj_in)
    };
    let nrys_roots = (li as usize + lj as usize + lk as usize) / 2 + 1;
    if nrys_roots > MAX_DEVICE_NROOTS {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nrys_roots}"),
        });
    }

    // Coordinates
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // From CINTinit_int3c2e_EnvVars:
    // common_factor = pi^3 * 2 / sqrt(pi) * fac_sp(i) * fac_sp(j) * fac_sp(k)
    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    // `rirj` is recomputed inside the device kernel from ri/rj; keep the host-side
    // value only as documentation of the canonical-order displacement.
    let _ = rirj;

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsi_in = nsph(li_in);
    let nsj_in = nsph(lj_in);
    let nsk = nsph(lk);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // Flatten the f64 primitive data the kernel reads (canonical li>=lj order).
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();

    // Dispatch onto the resolved backend's device client (compute in f64).
    let cart_buf: Vec<f64> = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_3c2e_device::<cubecl::cpu::CpuRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, nrys_roots as u32,
            ri, rj, rk, common_factor, &exps_i, &exps_j, &exps_k, &coeff_i, &coeff_j, &coeff_k,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_3c2e_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, nrys_roots as u32,
            ri, rj, rk, common_factor, &exps_i, &exps_j, &exps_k, &coeff_i, &coeff_j, &coeff_k,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_3c2e_device::<cubecl_cuda::CudaRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, nrys_roots as u32,
            ri, rj, rk, common_factor, &exps_i, &exps_j, &exps_k, &coeff_i, &coeff_j, &coeff_k,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_3c2e_device::<cubecl_hip::HipRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, nrys_roots as u32,
            ri, rj, rk, common_factor, &exps_i, &exps_j, &exps_k, &coeff_i, &coeff_j, &coeff_k,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_3c2e_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, nrys_roots as u32,
            ri, rj, rk, common_factor, &exps_i, &exps_j, &exps_k, &coeff_i, &coeff_j, &coeff_k,
        ),
    };

    let cart_out = if swap_ij {
        // libcint's 3c2e recurrence chooses ibase adaptively (li > lj).
        // We evaluate in canonical order li>=lj and transpose back when input had li<lj.
        transpose_ij_3idx(&cart_buf, nci, ncj, nck)
    } else {
        cart_buf
    };

    // Apply cart-to-sph/spinor or copy Cartesian, casting to F at the staging write.
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_3c2e(&cart_out, li_in, lj_in, lk);
            let sph_size = nsi_in * nsj_in * nsk;
            let copy_len = staging.len().min(sph_size);
            for (dst, &src) in staging[..copy_len].iter_mut().zip(sph[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
        Representation::Spinor => {
            // cart_to_spinor_sf_3c2e is generic over F: CintFloat (Plan 04).
            let kappa_i = shell_i_in.kappa;
            let kappa_j = shell_j_in.kappa;
            cart_to_spinor_sf_3c2e::<F>(
                staging, &cart_out,
                li_in, kappa_i, lj_in, kappa_j, lk,
            )?;
        }
        Representation::Cart => {
            let copy_len = staging.len().min(cart_out.len());
            for (dst, &src) in staging[..copy_len].iter_mut().zip(cart_out[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
    }

    // Per-symbol nonzero sentinel
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

/// Outer precision dispatcher for the 3c2e kernel.
///
/// Keeps the registered `FamilyLaunchFn` signature unchanged. Internally matches on
/// `plan.precision` and delegates to `launch_center_3c2e_typed::<F>`, reinterpreting
/// staging via `bytemuck::cast_slice_mut` for the F32 arm (A5 proven sound).
/// CR-01: captures the true output element count BEFORE the bytemuck cast and bounds
/// the typed inner to that count, returning `BufferTooSmall` if the view cannot hold it.
pub fn launch_center_3c2e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_center_3c2e_typed::<f64>(backend, plan, specialization, staging)
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
            launch_center_3c2e_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1c: launch_center_3c2e_typed::<f64> is byte-identical to the
    // existing launch_center_3c2e at f64 (center_3c2e_parity).
    // RED: compile fails until launch_center_3c2e_typed is defined.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c2e_parity_f64() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(22), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(22), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging_outer = vec![0.0_f64; 1];
        let mut staging_typed = vec![0.0_f64; 1];

        // Call outer dispatcher
        let result_outer = launch_center_3c2e(&backend, &plan, &spec, &mut staging_outer);
        assert!(result_outer.is_ok(), "outer f64 3c2e should succeed: {:?}", result_outer);

        // Call typed inner directly (RED: compile fails until launch_center_3c2e_typed defined)
        let result_typed = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging_typed);
        assert!(result_typed.is_ok(), "typed f64 3c2e should succeed: {:?}", result_typed);

        // Byte-identical check
        assert_eq!(staging_outer[0].to_bits(), staging_typed[0].to_bits(),
            "f64 outer and typed 3c2e should be byte-identical: outer={} typed={}", staging_outer[0], staging_typed[0]);
        assert!(staging_outer[0].is_finite() && staging_outer[0].abs() > 1e-30,
            "3c2e s-s-s value should be finite and nonzero: {}", staging_outer[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1d: launch_center_3c2e F32 path runs without panic.
    // RED: compile fails until launch_center_3c2e dispatches on plan.precision.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c2e_f32_smoke() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(22), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(22), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_center_3c2e(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "F32 3c2e should succeed without panic: {:?}", result);

        let staging_f32 = bytemuck::cast_slice::<f64, f32>(&staging);
        assert!(staging_f32[0].is_finite(), "F32 3c2e result should be finite: {}", staging_f32[0]);
        assert!(staging_f32[0] > 0.0, "F32 3c2e result should be positive: {}", staging_f32[0]);
    }

    #[test]
    fn test_fill_g_tensor_3c2e_sss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.1, 0.2];
        let pair = compute_pdata_host(
            1.0, 1.0, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
        );

        let g = fill_g_tensor_3c2e(&pair, 1.0, ri, rk, 0, 0, 0, 1, 1.0);
        assert_eq!(g.len(), 3, "s-s-s should produce one root x one n x one m");
        assert!(g[2].abs() > 1e-20, "gz root must be non-zero for s-s-s primitive");
    }

    #[test]
    fn test_contract_3c2e_sss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.1, 0.2];
        let pair = compute_pdata_host(
            1.0, 1.0, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
        );

        let g2d = fill_g_tensor_3c2e(&pair, 1.0, ri, rk, 0, 0, 0, 1, 1.0);
        let g_split = split_ij_hrr(&g2d, 0, 0, 0, 1, [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]]);
        let out = contract_3c2e(&g_split, 0, 0, 0, 1);
        assert_eq!(out.len(), 1);
        assert!(out[0].abs() > 1e-20, "contracted s-s-s 3c2e value must be non-zero");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar 3c2e device-vs-host cross-check (CpuRuntime, f64). The CubeCL kernel
// (`center_3c2e_scalar_kernel`) must reproduce the host pipeline
// (`fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`) exactly.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod scalar_device_tests {
    use super::*;

    /// Host reference: single-primitive single-contraction shell triple the same
    /// way the device kernel does, in canonical li>=lj order.
    fn host_cart_3c2e(
        ai: f64,
        aj: f64,
        ak: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rk: [f64; 3],
        li: u8,
        lj: u8,
        lk: u8,
        common_factor: f64,
        coeff_i: f64,
        coeff_j: f64,
        coeff_k: f64,
    ) -> Vec<f64> {
        let nrys = (li as usize + lj as usize + lk as usize) / 2 + 1;
        let pair = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let fac_env = common_factor * pair.fac;
        let g2d = fill_g_tensor_3c2e(&pair, ak, ri, rk, li, lj, lk, nrys, fac_env);
        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        let g_split = split_ij_hrr(&g2d, li, lj, lk, nrys, rirj);
        let prim = contract_3c2e(&g_split, li, lj, lk, nrys);
        let weight = coeff_i * coeff_j * coeff_k;
        prim.iter().map(|&v| weight * v).collect()
    }

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Cross-check: requires li>=lj (the device kernel runs in canonical order).
    fn assert_device_matches_host_3c2e(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64) {
        assert!(li >= lj, "cross-check requires canonical li>=lj order");
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.3_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.7, 0.2];
        let coeff_i = 0.9_f64;
        let coeff_j = 1.1_f64;
        let coeff_k = 0.8_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk);
        let nrys = (li as usize + lj as usize + lk as usize) / 2 + 1;

        let host = host_cart_3c2e(
            ai, aj, ak, ri, rj, rk, li, lj, lk, common_factor, coeff_i, coeff_j, coeff_k,
        );
        let dev = run_3c2e_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
            li as u32,
            lj as u32,
            lk as u32,
            1,
            1,
            1,
            1,
            1,
            1,
            nrys as u32,
            ri,
            rj,
            rk,
            common_factor,
            &[ai],
            &[aj],
            &[ak],
            &[coeff_i],
            &[coeff_j],
            &[coeff_k],
        );

        assert_eq!(host.len(), dev.len(), "length mismatch li={li} lj={lj} lk={lk}");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host 3c2e mismatch li={li} lj={lj} lk={lk} idx={idx}: \
                 host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    /// Cross-check the inlined Gaussian-product pdata against `compute_pdata_host`.
    #[test]
    fn test_3c2e_inlined_pdata_matches_host() {
        // s-s-s device run reduces to the pdata fac × Rys weight; the
        // device-vs-host cross-check below already validates the full chain,
        // but pin the pdata math directly too.
        let ai = 0.8_f64;
        let aj = 1.3_f64;
        let ri = [0.1_f64, -0.2, 0.3];
        let rj = [0.4_f64, 0.5, -0.6];
        let pair = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let zeta_ab = ai + aj;
        let px = (ai * ri[0] + aj * rj[0]) / zeta_ab;
        let rij = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        let rr = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
        let fac = (-ai * aj / zeta_ab * rr).exp();
        assert!((pair.zeta_ab - zeta_ab).abs() < 1e-15, "zeta_ab mismatch");
        assert!((pair.center_p_x - px).abs() < 1e-15, "center_p mismatch");
        assert!((pair.fac - fac).abs() < 1e-15, "fac mismatch");
    }

    #[test]
    fn test_device_matches_host_sss() {
        assert_device_matches_host_3c2e(0, 0, 0, 0.8, 1.0, 1.2);
    }

    #[test]
    fn test_device_matches_host_ssp() {
        assert_device_matches_host_3c2e(0, 0, 1, 0.8, 1.0, 1.2);
    }

    #[test]
    fn test_device_matches_host_pss() {
        assert_device_matches_host_3c2e(1, 0, 0, 1.3, 0.8, 0.9);
    }

    #[test]
    fn test_device_matches_host_sps_canonical() {
        // s-p-s in the caller maps to canonical li>=lj as p(i)-s(j)-s(k).
        assert_device_matches_host_3c2e(1, 0, 0, 0.7, 1.1, 0.6);
    }

    #[test]
    fn test_device_matches_host_psp() {
        assert_device_matches_host_3c2e(1, 0, 1, 0.9, 0.5, 0.7);
    }

    #[test]
    fn test_device_matches_host_pps() {
        // li>0 and lj>0: exercises the j-HRR transfer.
        assert_device_matches_host_3c2e(1, 1, 0, 0.6, 0.9, 0.8);
    }

    /// Genericity: the scalar kernel compiles and runs at `F = f32`. Launch an
    /// s-s-s triple at f32 on the CPU runtime and assert a finite result.
    #[test]
    fn test_center_3c2e_kernel_generic_f32() {
        let client = cpu_client();
        // nroots=1, g_size=1, split_size=1, work_len=1, out_len=1.
        let exps = [1.0_f32];
        let coeff = [1.0_f32];
        let g_zero = [0.0_f32; 3];
        let gs_zero = [0.0_f32; 3];
        let rys_zero = [0.0_f32; 1];
        let work_zero = [0.0_f32; 1];
        let out_zero = [0.0_f32; 1];

        let mk = |s: &[f32]| client.create_from_slice(f32::as_bytes(s));
        let exps_i_h = mk(&exps);
        let exps_j_h = mk(&exps);
        let exps_k_h = mk(&exps);
        let coeff_i_h = mk(&coeff);
        let coeff_j_h = mk(&coeff);
        let coeff_k_h = mk(&coeff);
        let g_h = mk(&g_zero);
        let gs_h = mk(&gs_zero);
        let u_h = mk(&rys_zero);
        let w_h = mk(&rys_zero);
        let work_h = mk(&work_zero);
        let out_h = mk(&out_zero);

        let common_factor =
            ((PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(0) * common_fac_sp(0) * common_fac_sp(0))
                as f32;

        center_3c2e_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(gs_h, 3) },
            unsafe { ArrayArg::from_raw_parts(u_h, 1) },
            unsafe { ArrayArg::from_raw_parts(w_h, 1) },
            unsafe { ArrayArg::from_raw_parts(work_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32,
            0.0,
            0.0,
            0.3,
            0.0,
            0.5,
            0.0,
            0.7,
            0.2,
            common_factor,
            PIE4 as f32,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1u32,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 3c2e kernel result must be finite");
        assert!(out > 0.0, "s-s-s 3c2e f32 result should be positive: {out}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// int3c2e_ip1 gradient tests (Plan 21-06 / GRAD-08 / Risk R1).
//
// The latent bug: `launch_center_3c2e_typed` was operator-blind and returned the
// PLAIN (scalar) 3c2e integral for `int3c2e_ip1`. These tests pin the REAL
// 3-component derivative:
//   - component count: (s,s,s) → 3; (p,s,s) → 3×3 = 9 (the 3× multiplier is the
//     proof the scalar stub is gone — the stub returned 1×nci*ncj*nck).
//   - NOT-equal-to-plain: the (p,s,s) ip1 output is NOT element-wise equal to the
//     plain int3c2e output broadcast across components (regression-proof for R1).
//   - determinism (D-10): repeated evaluation is bit-identical.
//   - spinor (R5): int3c2e_ip1 with Representation::Spinor returns UnsupportedApi.
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

    /// Build a 3-shell (li, lj, lk) triple plan for the int3c2e_ip1 sph operator.
    fn build_ip1_plan(
        li: u8,
        lj: u8,
        lk: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom2 = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1, atom2].into_boxed_slice());

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
        let s2 = mk(2, lk);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2]).unwrap();

        let op = Resolver::descriptor_by_symbol("int3c2e_ip1_sph")
            .expect("int3c2e_ip1_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    /// Build a plain int3c2e_sph triple plan (same shells/centers) for the
    /// NOT-equal-to-plain regression comparison.
    fn build_plain_plan(
        li: u8,
        lj: u8,
        lk: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom2 = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1, atom2].into_boxed_slice());

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
        let s2 = mk(2, lk);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2]).unwrap();

        let op = Resolver::descriptor_by_symbol("int3c2e_sph")
            .expect("int3c2e_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    fn run(
        basis: &BasisSet,
        shells: ShellTuple,
        op: cintx_core::OperatorId,
        rep: Representation,
    ) -> Result<(Vec<f64>, ExecutionStats), cintxRsError> {
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, rep, basis, shells.clone(), &opts)?;
        let mut plan = ExecutionPlan::new(op, rep, basis, shells, &q)?;
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let out_elems = plan.output_layout.staging_elements;
        let mut staging = vec![0.0_f64; out_elems];
        let stats = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok((staging, stats))
    }

    // Component count (the 3× multiplier proves the scalar stub is gone). The scalar
    // stub only WROTE `nci*ncj*nck` elements (3 for (p,s,s)) and left the remaining
    // 6 lanes at zero. The real derivative must FILL all 3 component blocks with
    // genuinely-derived values. We assert the kernel writes a nonzero value into
    // each of the 3 component blocks (not just into the first block).
    #[test]
    fn test_int3c2e_ip1_component_count() {
        let (basis, shells, op) = build_ip1_plan(0, 0, 0);
        let (staging, _) = run(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(
            staging.len(),
            3,
            "(s,s,s) int3c2e_ip1 should produce 3 components, got {}",
            staging.len()
        );
        // s-s-s ∇_i with the test geometry (all atoms in the z=0 plane): the x and y
        // gradient components are nonzero; the z component vanishes by symmetry. A
        // genuine 3-component derivative therefore fills ≥2 lanes with distinct
        // nonzero values — the scalar stub (which broadcast a single scalar) would
        // give 3 identical lanes. We assert at least 2 nonzero AND not all equal.
        let nonzero = staging.iter().filter(|v| v.abs() > 1e-12).count();
        assert!(
            nonzero >= 2,
            "(s,s,s) int3c2e_ip1 must fill ≥2 component lanes (in-plane ∇_i): {staging:?}"
        );
        let all_equal = staging
            .iter()
            .all(|v| (v - staging[0]).abs() <= 1e-15);
        assert!(
            !all_equal,
            "(s,s,s) int3c2e_ip1 lanes are all identical — the scalar stub (R1) is \
             NOT closed: {staging:?}"
        );

        let (basis, shells, op) = build_ip1_plan(1, 0, 0);
        let (staging, _) = run(&basis, shells, op, Representation::Spheric).unwrap();
        // sph p = 3 AOs; 3 components × 3×1×1 = 9. The scalar stub gave 3 written.
        assert_eq!(
            staging.len(),
            9,
            "(p,s,s) int3c2e_ip1 should produce 9 outputs (3 comps × 3 AO), got {}",
            staging.len()
        );
        // PROOF the scalar stub is gone: every one of the 3 component blocks (each
        // of length nci*ncj*nck = 3) must carry a nonzero derived value. The stub
        // wrote only the first block (lanes 0..3) and left lanes 3..9 at zero, so
        // blocks 1 and 2 would be all-zero under the stub.
        let block_len = 3usize; // nci(p)*ncj(s)*nck(s) = 3*1*1
        for comp in 0..3usize {
            let block = &staging[comp * block_len..(comp + 1) * block_len];
            assert!(
                block.iter().any(|v| v.abs() > 1e-12),
                "(p,s,s) int3c2e_ip1 component block {comp} is all-zero — the scalar \
                 stub (R1) is NOT closed. staging={staging:?}"
            );
        }
    }

    // NOT-equal-to-plain (R1 regression proof): the (p,s,s) ip1 output must NOT be
    // element-wise equal to the plain int3c2e output broadcast across the 3 comps.
    // The scalar stub WROTE `plain` into the first component block (lanes 0..3) and
    // zeros into the rest, so the regression-proof here is the first block: a real
    // derivative ∇_i is NOT equal to the plain integral value itself.
    #[test]
    fn test_int3c2e_ip1_not_equal_to_plain() {
        let (basis_ip1, shells_ip1, op_ip1) = build_ip1_plan(1, 0, 0);
        let (ip1, _) = run(&basis_ip1, shells_ip1, op_ip1, Representation::Spheric).unwrap();

        let (basis_plain, shells_plain, op_plain) = build_plain_plan(1, 0, 0);
        let (plain, _) = run(&basis_plain, shells_plain, op_plain, Representation::Spheric).unwrap();

        // plain has 3 AOs (p,s,s); ip1 has 9 (3 comps × 3 AO).
        assert_eq!(plain.len(), 3, "plain (p,s,s) 3c2e should be 3 AOs");
        assert_eq!(ip1.len(), 9, "ip1 (p,s,s) 3c2e should be 9 (3 comps × 3 AO)");

        // The FIRST component block (lanes 0..3) is what the scalar stub wrote `plain`
        // into. A real ∇_i derivative differs from the plain integral value. If the
        // stub were still in place, ip1[0..3] would equal `plain` exactly.
        let first_block = &ip1[0..3];
        let first_block_equals_plain = first_block
            .iter()
            .zip(plain.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-12);
        assert!(
            !first_block_equals_plain,
            "int3c2e_ip1 first component block is byte-equal to the plain integral — \
             the scalar stub (R1) is NOT closed. ip1[0..3]={first_block:?} plain={plain:?}"
        );
        // The derivative must produce nonzero values across all 3 component blocks.
        assert!(
            ip1.iter().any(|v| v.abs() > 1e-12),
            "int3c2e_ip1 (p,s,s) output is all-zero: {ip1:?}"
        );
    }

    // Determinism (D-10): repeated evaluation is bit-identical.
    #[test]
    fn test_int3c2e_ip1_determinism() {
        let (basis, shells, op) = build_ip1_plan(1, 0, 0);
        let (out1, _) = run(&basis, shells.clone(), op, Representation::Spheric).unwrap();
        let (out2, _) = run(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "int3c2e_ip1 output not bit-identical across two evaluations"
            );
        }
    }

    // Spinor (R5): int3c2e_ip1 with Representation::Spinor returns UnsupportedApi.
    #[test]
    fn test_int3c2e_ip1_spinor_unsupported() {
        let (basis, shells, op) = build_ip1_plan(0, 0, 0);
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 6];
        let result = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "spinor int3c2e_ip1 should return UnsupportedApi, got: {result:?}"
        );
    }
}
