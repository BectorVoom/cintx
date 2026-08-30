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
//!   This file follows that mapping explicitly: the third center `k` is treated as
//!   the 2e `ll` angular channel, with only one real "ket-side" angular axis.

// Transcribed verbatim from vendored libcint 6.1.3 (and, in `cintx-basis`, from the
// Lanczos reference these normalization constants come from). Result compatibility
// is decided by the exact bits these literals carry, so none is truncated to the
// shortest form that round-trips.
#![allow(clippy::excessive_precision)]
// Index arithmetic here is written in full — `base + 0 * stride`, `base + 1 * stride`,
// `out[n * 3 + 0]` — so that a slot or component index lines up column-wise with its
// neighbours and with the libcint layout being mirrored. Folding the `0 *` and `1 *`
// away would shorten the line and hide the stride.
#![allow(clippy::identity_op)]
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

use crate::backend::ResolvedBackend;
use crate::kernels::two_electron::fill_g_tensor_2e;
use crate::kernels::two_electron::{BatchOptions, ResidentBasis};
use crate::kernels::two_electron::{build_2e_shape, two_e_shape_as_f12};
// Phase 25 HESS-03: verbatim Hessian gout helpers (bra-i ∇² + ket-k ∇²).
use crate::kernels::f12::{gout_ip1ip2_l, gout_ipip1, gout_ipip2_l, gout_ipvip1};
use crate::math::pdata::PairData;
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::math::rys_wheeler::{
    EXT_TABLES_LEN, ext_rys_out_slots, ext_rys_slots, rys_roots_ext_dev,
};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c2e, cart_to_sph_3c2e_into, ncart, nsph};
use crate::transform::c2spinor::{
    cart_to_spinor_sf_3c2e, cart_to_spinor_sf_derivative_3c2e, cart_to_spinor_si_3c2e1, spinor_len,
};
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Host Rys nroots ceiling (FND-02 Wheeler engine supports nroots 6..12). The
/// multi-center Hessian families (ipip1/ipip2) route through the HOST
/// `fill_g_tensor_2e` path so the `+2` headroom raise can reach nroots 6..12;
/// nroots>12 stays fail-closed.
const HOST_RYS_NROOTS_CEILING: usize = 12;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
// Verbatim libcint literal, not `std::f64::consts::FRAC_PI_4`: result compatibility
// with upstream is decided by the exact bits this file feeds the Rys kernels, so
// the constant is transcribed from `rys_roots.c` rather than recomputed.
#[allow(clippy::approx_constant)]
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate.
const MAX_DEVICE_NROOTS: usize = 5;

/// Largest `nroots` this crate has launch arms compiled for in the 3c2e batch
/// dispatch.
///
/// Not the same thing as [`crate::device_rys_ceiling::device_nroots_ceiling`],
/// which additionally requires the backend's FMA probe to have passed. This is
/// only "was the code compiled in", and it is what the launcher's
/// `debug_assert` checks so that a class arriving without an arm is a loud
/// failure rather than a silent evaluation at `nroots = 5`.
const fn three_c2e_launch_nroots_ceiling() -> u32 {
    if cfg!(feature = "extended-device-rys") {
        crate::device_rys_ceiling::EXTENDED_DEVICE_NROOTS as u32
    } else {
        MAX_DEVICE_NROOTS as u32
    }
}

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
/// D-PBC-24: no longer `#[cfg(test)]` — the range-separated host arm
/// (`host_3c2e_cart_blocks`) is production code and needs it.
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
/// Host f64 reference of the exact device algorithm, and — since D-PBC-24 — the
/// PRODUCTION path whenever `range_omega` is set, because the device kernel has
/// no omega branch yet (stage 4). It is therefore no longer `#[cfg(test)]`.
///
/// `nrys_roots` is the caller's, already doubled where short range demands it
/// (`g3c2e.c:70-77`); `range_omega` selects the `CINTg0_2e` arm. Returns
/// `Ok(None)` when the short-range integrand is past `EXPCUTOFF_SR`.
#[allow(clippy::too_many_arguments)]
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
    range_omega: Option<f64>,
) -> Result<Option<Vec<f64>>, cintxRsError> {
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

    // D-PBC-24: the shared `CINTg0_2e` omega branch (g2e.c:4443-4512). 3c2e
    // reaches it through `g3c2e.c:131`'s `envs->f_g0_2e = &CINTg0_2e`, with
    // `lk_ceil = 0` and the real auxiliary shell in the `ll` slot — so its
    // `rys_order` is `(li + lj + lk)/2 + 1` (g3c2e.c:70). The `omega == 0` arm is
    // the plain `rys_roots_host(nrys_roots, x_rys)` this used to call.
    let rys_order = (li as usize + lj as usize + lk as usize) / 2 + 1;
    let Some(roots) = crate::math::range_separation::rys_roots_range_separated(
        rys_order,
        nrys_roots,
        x_rys,
        a0,
        fac1,
        range_omega,
    )?
    else {
        return Ok(None);
    };
    let (u_roots, w_weights, fac1) = (roots.u, roots.w, roots.fac1);

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
                // g(n,m+1) = c0p*g(n,m) + m*b01*g(n,m-1) + n*b00*g(n-1,m): the b00
                // cross term carries the n (combined-ij) index factor (libcint
                // g2e.c CINTg0_2e). Omitting it under-counts d+ terms (n>=2) — the
                // int3c2e analog of the int2c2e d-shell bug; s/p (n<=1) unaffected.
                if nmax > 0 {
                    for n in 1..=nmax {
                        let i_off = irys + n * dn;
                        let s0_k0 = g[axis_off + i_off];
                        let prev_i_k0 = g[axis_off + irys + (n - 1) * dn];
                        let mut s1 = c0p_axis * s0_k0 + n as f64 * b00 * prev_i_k0;
                        g[axis_off + i_off + dm] = s1;
                        let mut s_prev = s0_k0;
                        for m in 1..mmax {
                            let prev_i_km = g[axis_off + irys + (n - 1) * dn + m * dm];
                            let s2 = c0p_axis * s1
                                + m as f64 * b01 * s_prev
                                + n as f64 * b00 * prev_i_km;
                            g[axis_off + i_off + (m + 1) * dm] = s2;
                            s_prev = s1;
                            s1 = s2;
                        }
                    }
                }
            }
        }
    }

    Ok(Some(g))
}

/// Cartesian contraction blocks for one 3c2e shell triple, on the HOST.
///
/// # D-PBC-24 stage 4
///
/// The scalar 3c2e production path is the `#[cube]` device kernel, which has no
/// omega branch: its comptime `nroots` arms evaluate `rys_root{1..5}` at a
/// single argument, and short range needs two evaluations plus a root
/// rescaling. Until the device arms land, range separation routes here.
///
/// The output layout is EXACTLY `run_3c2e_device`'s, so the caller's
/// canonical-order restore, cart→sph/spinor transform and AO scatter consume it
/// unchanged: `n_ctr_i * n_ctr_j * n_ctr_k` Cartesian blocks, contraction-major
/// as `((ci * n_ctr_j + cj) * n_ctr_k + ck) * nci*ncj*nck`, with `i` fastest
/// inside each block. Shells arrive in canonical `li >= lj` order.
///
/// The chain is the one the `scalar_device_tests` cross-check already proves
/// device-equivalent: `fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`.
/// Coefficients are PRIMITIVE-major (`coeff[p * nctr + c]`, WR-03).
#[allow(clippy::too_many_arguments)]
fn host_3c2e_cart_blocks(
    li: u8,
    lj: u8,
    lk: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    n_ctr_k: usize,
    nrys_roots: usize,
    common_factor: f64,
    range_omega: Option<f64>,
) -> Result<Vec<f64>, cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let block_len = nci * ncj * nck;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    let mut cart_accum = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * block_len];

    for (pi, &ai) in exps_i.iter().enumerate() {
        for (pj, &aj) in exps_j.iter().enumerate() {
            let pair =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for (pk, &ak) in exps_k.iter().enumerate() {
                let fac_env = common_factor * pair.fac;
                let Some(g2d) = fill_g_tensor_3c2e(
                    &pair,
                    ak,
                    ri,
                    rk,
                    li,
                    lj,
                    lk,
                    nrys_roots,
                    fac_env,
                    range_omega,
                )?
                else {
                    // Short-range integrand past EXPCUTOFF_SR: this primitive
                    // triple contributes nothing (g2e.c:4460).
                    continue;
                };
                let g_split = split_ij_hrr(&g2d, li, lj, lk, nrys_roots, rirj);
                let prim = contract_3c2e(&g_split, li, lj, lk, nrys_roots);

                for ci in 0..n_ctr_i {
                    let ci_coeff = coeff_i[pi * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let cj_coeff = coeff_j[pj * n_ctr_j + cj];
                        for ck in 0..n_ctr_k {
                            let weight = ci_coeff * cj_coeff * coeff_k[pk * n_ctr_k + ck];
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * block_len;
                            for idx in 0..block_len {
                                cart_accum[base + idx] += weight * prim[idx];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(cart_accum)
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
/// Host f64 reference, and the production path under `range_omega` (D-PBC-24).
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
/// Host f64 reference, and the production path under `range_omega` (D-PBC-24).
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
    transpose_ij_3idx_into(buf, ni, nj, nk, &mut out);
    out
}

/// [`transpose_ij_3idx`] writing into a caller-owned buffer.
///
/// Every element of `out` is written, so a reused (non-zeroed) buffer gives
/// bit-identical output (Task 36-T0/36-T1). `out` must be `buf.len()` long.
fn transpose_ij_3idx_into(buf: &[f64], ni: usize, nj: usize, nk: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), buf.len());
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let src = (k * nj + j) * ni + i;
                let dst = (k * ni + i) * nj + j;
                out[dst] = buf[src];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scalar 3c2e device kernel — `#[cube(launch)]`, generic over `F: Float`
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar 3c2e G-tensor fill (2e recurrence) + ij-HRR split + Cartesian
/// contraction for one shell triple, on-device.
///
/// A faithful, correctness-first port of the host pipeline
/// (`fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`) inlined per the
/// `#[cube]` authoring rules (no plain-fn calls; the device `rys_root{1..5}` are
/// the only callees). The kernel runs in canonical `li>=lj` order — the host
/// launcher decides the `swap_ij` and transposes the read-back buffer.
///
/// Batched (Phase 35): one shell triple per work slot. Every triple in the list
/// shares `(li, lj, lk)` and therefore the G-tensor shape and Rys order; what
/// varies is only the shell data, read through a flattened basis plus an index
/// table:
///
/// - `exps` / `coeffs` — every shell's primitives concatenated;
/// - `centers` — 3 floats per shell;
/// - `shell_meta` — 4 `u32` per shell: `[exp_off, coeff_off, nprim, nctr]`;
/// - `triples` — 4 `u32` per triple: `[si, sj, sk, out_off]`.
///
/// This kernel's arithmetic is not split across a cube: one slot evaluates a
/// whole triple. `per_unit == 1` therefore maps a triple to each *unit* (the
/// CubeCL CPU shape, where a unit is an OS thread); `per_unit == 0` maps a
/// triple to each *cube*, which is what the pre-batching kernel did with its
/// `UNIT_POS == 0` guard.
///
/// `#[comptime] nroots` selects the `rys_root{1..5}` device function at JIT time.
///
/// Layout of one slot's `g` slab (`3 * g_size` elements at `slot * g_stride`,
/// the 2e-style 2D fill, root-fastest):
/// `g[gbase + axis*g_size + m*dm + n*dn + root]` with `dn = nrys`,
/// `dm = nrys*(nmax+1)`, `g_size = nrys*(nmax+1)*(mmax+1)`, `nmax = li+lj`,
/// `mmax = lk`.
///
/// Layout of one slot's `g_split` slab (`3 * split_size` elements at
/// `slot * split_stride`, after the j-HRR transfer):
/// `g_split[sbase + axis*split_size + ((root*nk + k)*nj + j)*ni + i]` with
/// `ni = li+1`, `nj = lj+1`, `nk = lk+1`, `split_size = nrys*nk*nj*ni`.
///
/// `cart_out` (size `nctr_i*nctr_j*nctr_k * nci*ncj*nck`) is zeroed in-kernel
/// and accumulated over all primitive triples. Contraction block
/// `(ci, cj, ck)` lives at `((ci*nctr_j + cj)*nctr_k + ck) * nci*ncj*nck`, and
/// within a block the Cartesian index is i-fastest, k-slowest — the same
/// contraction-major layout `center_3c2e_ip1_kernel` and the 2e kernel use.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn center_3c2e_scalar_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    triples: &Array<u32>,
    class_shape: &Array<u32>,
    class_factor: &Array<F>,
    rys_tab: &Array<f64>,
    g: &mut Array<F>,
    g_split: &mut Array<F>,
    work: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    prim_tol: F,
    n_triples: u32,
    n_cubes: u32,
    g_stride: u32,
    split_stride: u32,
    work_slab: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    // Slot decomposition — see the doc comment above, and the identical block in
    // `two_electron.rs` for why this is arithmetic on comptime-folded flags.
    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Rys roots/weights are read only inside the `lane == 0` region below, so
    // they are per-unit private storage rather than buffers.
    let mut urys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    let mut wrys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    // The extended (`nroots >= 6`) entry is f64-only — its double-double arms
    // are what buy the accuracy, and there is no meaningful `f32` version. It
    // therefore lands in its own pair of buffers and is cast into `urys`/`wrys`,
    // which stay `F`. Both collapse to one element when the arm is not emitted.
    let mut uext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));
    let mut wext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));

    if lane == 0u32 {
        let nrys = nroots;

        let gbase = slot * g_stride;
        let sbase = slot * split_stride;
        let wbase = slot * work_slab;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
        // `u32::div_ceil` has no `#[cube]` expansion, so the blocked-walk
        // chunk size is written out.
        #[allow(clippy::manual_div_ceil)]
        let chunk = (n_triples + n_slots - 1u32) / n_slots;
        let qi_start = slot * (chunk * punit + coop);
        let mut qi_stop = (qi_start + chunk) * punit + n_triples * coop;
        if qi_stop > n_triples {
            qi_stop = n_triples;
        }
        let qi_step = n_slots * coop + punit;

        let mut qi = qi_start;
        while qi < qi_stop {
            let trow = qi * 5u32;
            let si = triples[trow as usize];
            let sj = triples[(trow + 1u32) as usize];
            let sk = triples[(trow + 2u32) as usize];
            let out_off = triples[(trow + 3u32) as usize];

            // ── Per-class shape (Task 35-M2) ──────────────────────────────
            //
            // `nroots` is this kernel's only comptime parameter, so one
            // dispatch carries every `(li,lj,lk)` class of the same Rys order.
            // The G, split and work slabs are sized to the widest class in the
            // dispatch and each class indexes only the leading elements it
            // owns, so a narrow class touches exactly what it did alone —
            // which is what keeps the merge bit-identical.
            let cls = triples[(trow + 4u32) as usize];
            let srow = cls * comptime!(THREE_C2E_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];
            let lk = class_shape[(srow + 2u32) as usize];
            let common_factor = class_factor[cls as usize];

            let nmax = li + lj;
            let mmax = lk;
            let dn = nrys;
            let dm = nrys * (nmax + 1u32);
            let g_size = nrys * (nmax + 1u32) * (mmax + 1u32);

            let ni = li + 1u32;
            let nj = lj + 1u32;
            let nk = lk + 1u32;
            let split_size = nrys * nk * nj * ni;
            let work_stride = nmax + 1u32;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
            let block_len = nci * ncj * nck;

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];
            let mk = sk * 4u32;
            let eoff_k = shell_meta[mk as usize];
            let coff_k = shell_meta[(mk + 1u32) as usize];
            let nprim_k = shell_meta[(mk + 2u32) as usize];
            let nctr_k = shell_meta[(mk + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];
            let ck3 = sk * 3u32;
            let rkx = centers[ck3 as usize];
            let rky = centers[(ck3 + 1u32) as usize];
            let rkz = centers[(ck3 + 2u32) as usize];

            let out_len = nctr_i * nctr_j * nctr_k * block_len;

            // Zero the accumulation buffer.
            let mut oi = 0u32;
            while oi < out_len {
                cart_out[(out_off + oi) as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // rirj = ri - rj (for the j-HRR transfer).
            let rirj_x = rix - rjx;
            let rirj_y = riy - rjy;
            let rirj_z = riz - rjz;

            let mut kp = 0u32;
            while kp < nprim_k {
                let ak = exps[(eoff_k + kp) as usize];
                let mut jp = 0u32;
                while jp < nprim_j {
                    let aj = exps[(eoff_j + jp) as usize];
                    let mut ip = 0u32;
                    while ip < nprim_i {
                        let ai = exps[(eoff_i + ip) as usize];

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

                        // Primitive-triple screening (Task 34-D2) — the same
                        // test the 2e path applies, on the same quantity.
                        //
                        // `fac1` is the scalar every element of this primitive
                        // triple's G tensor is built from: `gz` starts at
                        // `wrys[irys] * fac1` and `gx`/`gy` at 1, so the whole
                        // contribution scales with it. Screening here rather
                        // than on `pair_fac` alone keeps the
                        // `sqrt(a0 / a1^3)` factor in the bound, which is not
                        // O(1): for diffuse primitives `a1` is small and that
                        // square root is large.
                        //
                        // At `prim_tol == 0` (the default) the only triples
                        // dropped are those whose `fac1` underflowed to exactly
                        // zero, whose contribution is exactly zero — which is
                        // why the tolerance-zero identity gate holds bit for
                        // bit. The Rys weights and the VRR/HRR coefficients are
                        // *not* bounded by one, so a non-zero tolerance is a
                        // proxy, not a certificate.
                        if fac1 > prim_tol {
                            let x_rys = a0 * rr;

                            // rijrx = P - Ri (the bra-side reference displacement).
                            let rijrx_x = px - rix;
                            let rijrx_y = py - riy;
                            let rijrx_z = pz - riz;

                            // Rys roots/weights for this primitive triple.
                            if comptime!(nroots == 1u32) {
                                rys_root1::<F>(x_rys, &mut urys, &mut wrys, pie4);
                            } else if comptime!(nroots == 2u32) {
                                rys_root2::<F>(x_rys, &mut urys, &mut wrys, pie4);
                            } else if comptime!(nroots == 3u32) {
                                rys_root3::<F>(x_rys, &mut urys, &mut wrys, pie4);
                            } else if comptime!(nroots == 4u32) {
                                rys_root4::<F>(x_rys, &mut urys, &mut wrys, pie4);
                            } else if comptime!(nroots == 5u32) {
                                rys_root5::<F>(x_rys, &mut urys, &mut wrys, pie4);
                            } else {
                                // nroots 6..=12: the inline Wheeler/Jacobi entry
                                // (task 33-01). Reachable only when
                                // `device_nroots_ceiling` raised past 5, which
                                // needs both the `extended-device-rys` feature
                                // and a passing per-backend FMA probe.
                                rys_roots_ext_dev(
                                    rys_tab,
                                    f64::cast_from(x_rys),
                                    &mut uext,
                                    &mut wext,
                                    nroots,
                                );
                                #[unroll]
                                for iext in 0..nroots {
                                    urys[iext as usize] = F::cast_from(uext[iext as usize]);
                                    wrys[iext as usize] = F::cast_from(wext[iext as usize]);
                                }
                            }

                            // ── Fill the 2D G-tensor (fill_g_tensor_3c2e) ──────────────
                            #[unroll]
                            for irys in 0..nroots {
                                let u2 = a0 * urys[irys as usize];
                                let tmp4 = F::new(0.5_f32) / (u2 * (aij + akl) + a1);
                                let tmp5 = u2 * tmp4;
                                let b00 = tmp5;
                                let b10 = tmp5 + tmp4 * akl;
                                let b01 = tmp5 + tmp4 * aij;
                                let tmp2 = F::new(2.0_f32) * tmp5 * akl;
                                let tmp3 = F::new(2.0_f32) * tmp5 * aij;

                                // Base case: gx=gy=1, gz=w*fac1.
                                g[(gbase + irys) as usize] = F::new(1.0_f32);
                                g[(gbase + g_size + irys) as usize] = F::new(1.0_f32);
                                g[(gbase + 2u32 * g_size + irys) as usize] =
                                    wrys[irys as usize] * fac1;

                                #[unroll]
                                for axis in 0..3u32 {
                                    let base = gbase + axis * g_size;
                                    // Per-axis displacement components.
                                    let mut d = xij_kl;
                                    let mut rx = rijrx_x;
                                    if axis == 1u32 {
                                        d = yij_kl;
                                        rx = rijrx_y;
                                    } else if axis == 2u32 {
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
                                                let mut s1 = c0pa * s0_k0
                                                    + F::cast_from(n) * b00 * prev_i_k0;
                                                g[(base + i_off + dm) as usize] = s1;
                                                let mut s_prev = s0_k0;
                                                let mut m = 1u32;
                                                while m < mmax {
                                                    let prev_i_km =
                                                        g[(base + irys + (n - 1u32) * dn + m * dm)
                                                            as usize];
                                                    let s2 = c0pa * s1
                                                        + F::cast_from(m) * b01 * s_prev
                                                        + F::cast_from(n) * b00 * prev_i_km;
                                                    g[(base + i_off + (m + 1u32) * dm) as usize] =
                                                        s2;
                                                    s_prev = s1;
                                                    s1 = s2;
                                                    m += 1u32;
                                                }
                                                n += 1u32;
                                            }
                                        }
                                    }
                                }
                            }

                            // ── split_ij_hrr: recover (i,j) channels via j-HRR ─────────
                            #[unroll]
                            for axis2 in 0..3u32 {
                                let axis_in_off = gbase + axis2 * g_size;
                                let axis_out_off = sbase + axis2 * split_size;
                                let mut rirj = rirj_x;
                                if axis2 == 1u32 {
                                    rirj = rirj_y;
                                } else if axis2 == 2u32 {
                                    rirj = rirj_z;
                                }

                                let mut k = 0u32;
                                while k <= mmax {
                                    #[unroll]
                                    for root in 0..nroots {
                                        // Load the i-base ladder into `work` (rows = j, cols = i-base).
                                        let mut i = 0u32;
                                        while i <= nmax {
                                            work[(wbase + i) as usize] =
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
                                                work[(wbase + cur + i2) as usize] = rirj
                                                    * work[(wbase + prev + i2) as usize]
                                                    + work[(wbase + prev + i2 + 1u32) as usize];
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
                                                    work[(wbase + jj * work_stride + ii) as usize];
                                                ii += 1u32;
                                            }
                                            jj += 1u32;
                                        }
                                    }
                                    k += 1u32;
                                }
                            }

                            // ── contract_3c2e: triple cart_comps contraction ───────────
                            // Output i fastest, j middle, k slowest.
                            let gx_off = sbase;
                            let gy_off = sbase + split_size;
                            let gz_off = sbase + 2u32 * split_size;

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

                                                    let mut val = F::new(0.0_f32);
                                                    #[unroll]
                                                    for root2 in 0..nroots {
                                                        let idx_x =
                                                            ((root2 * nk + kx) * nj + jx) * ni + ix;
                                                        let idx_y =
                                                            ((root2 * nk + ky) * nj + jy) * ni + iy;
                                                        let idx_z =
                                                            ((root2 * nk + kz) * nj + jz) * ni + iz;
                                                        val += g_split[(gx_off + idx_x) as usize]
                                                            * g_split[(gy_off + idx_y) as usize]
                                                            * g_split[(gz_off + idx_z) as usize];
                                                    }
                                                    // The Cartesian value is the same
                                                    // for every contraction triple; only
                                                    // the weight differs. Summing the
                                                    // weights into one scalar and writing
                                                    // a single block (as this kernel did
                                                    // before) is correct only when every
                                                    // `nctr` is 1.
                                                    let elem = (k_idx * ncj + j_idx) * nci + i_idx;
                                                    let mut cci = 0u32;
                                                    while cci < nctr_i {
                                                        let coeff_i_val = coeffs
                                                            [(coff_i + ip * nctr_i + cci) as usize];
                                                        let mut ccj = 0u32;
                                                        while ccj < nctr_j {
                                                            let coeff_j_val =
                                                                coeffs[(coff_j + jp * nctr_j + ccj)
                                                                    as usize];
                                                            let mut cck = 0u32;
                                                            while cck < nctr_k {
                                                                let coeff_k_val = coeffs[(coff_k
                                                                    + kp * nctr_k
                                                                    + cck)
                                                                    as usize];
                                                                let ctr_base =
                                                                    ((cci * nctr_j + ccj) * nctr_k
                                                                        + cck)
                                                                        * block_len;
                                                                cart_out[(out_off + ctr_base + elem)
                                                                    as usize] += val
                                                                    * coeff_i_val
                                                                    * coeff_j_val
                                                                    * coeff_k_val;
                                                                cck += 1u32;
                                                            }
                                                            ccj += 1u32;
                                                        }
                                                        cci += 1u32;
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
                        }

                        ip += 1u32;
                    }
                    jp += 1u32;
                }
                kp += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// `u32` shape scalars per class row of the device shape table: `li, lj, lk`,
/// in the kernel's canonical `li >= lj` order.
const THREE_C2E_SHAPE_STRIDE: usize = 3;

/// `u32` shape scalars per class row of the **derivative** shape table:
/// `li, lj, lk, di, dk, dl, dj, g_size, nmax, mmax, ibase`.
///
/// Wider than the scalar table because the derivative kernels do not rederive
/// the G-tensor extents in-kernel: they come from `build_2e_shape` on the host,
/// under the 3c2e slot mapping (real `k` in the 2e `ll` slot, phantom `lk = 0`)
/// with the family's headroom already applied — `li + 1` for ip1, `lk + 1` for
/// ip2. Recomputing that in-kernel would duplicate the mapping in two places.
const THREE_C2E_DERIV_SHAPE_STRIDE: usize = 11;

/// One dispatch of a 3c2e **derivative** family: every triple of one Rys order
/// (Task 35-D).
///
/// Same construction as [`ThreeC2eLaunchGroup`], but the shape row is the wider
/// [`THREE_C2E_DERIV_SHAPE_STRIDE`] one and there is a single `g_size` governing
/// both the `g` and `g1` slabs — the contraction reads them at the same relative
/// offsets, so they must share a layout.
#[derive(Clone, Debug)]
pub struct ThreeC2eDerivLaunchGroup {
    /// Rys order — the kernel's only shape-bearing comptime parameter.
    pub nroots: u32,
    /// [`THREE_C2E_DERIV_SHAPE_STRIDE`] `u32` per merged class.
    pub class_shape: Vec<u32>,
    /// One libcint `common_factor` per merged class.
    pub class_factor: Vec<f64>,
    /// `[si, sj, sk, out_off, class]` per triple.
    pub triples: Vec<u32>,
    /// Total Cartesian output elements across this group's triples.
    pub out_len: usize,
    /// Widest `g_size` in the group — what each per-slot slab is sized to.
    pub max_g_size: usize,
}

impl ThreeC2eDerivLaunchGroup {
    /// An empty group of Rys order `nroots`.
    #[must_use]
    pub fn new(nroots: u32) -> Self {
        Self {
            nroots,
            class_shape: Vec::new(),
            class_factor: Vec::new(),
            triples: Vec::new(),
            out_len: 0,
            max_g_size: 0,
        }
    }

    /// Append a class and return the index its triple rows carry.
    ///
    /// `pub(crate)` because it takes `TwoEShape`, which is a crate-internal
    /// description of the G-tensor extents. Exposing it would make the 2e
    /// recurrence machinery part of the public contract.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_class(
        &mut self,
        li: u32,
        lj: u32,
        lk: u32,
        shape: &crate::kernels::two_electron::TwoEShape,
        common_factor: f64,
    ) -> u32 {
        let index = self.class_factor.len() as u32;
        self.class_shape.extend_from_slice(&[
            li,
            lj,
            lk,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            u32::from(shape.ibase),
        ]);
        self.class_factor.push(common_factor);
        self.max_g_size = self.max_g_size.max(shape.g_size);
        index
    }

    /// Number of triples in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len() / 5
    }

    /// Is this group empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Number of angular-momentum classes merged into this dispatch.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_factor.len()
    }

    /// Bytes this group's triple and class tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        (self.triples.len() + self.class_shape.len()) * std::mem::size_of::<u32>()
            + self.class_factor.len() * std::mem::size_of::<f64>()
    }
}

/// One dispatch: every shell triple of the same Rys order (Task 35-M2).
///
/// `center_3c2e_scalar_kernel` specializes on `nroots` alone — `li`, `lj` and
/// `lk` are runtime scalars it already derives every extent from — so a launch
/// class is a Rys order, not an `(li,lj,lk)` tuple. Each triple names its class
/// in the fifth column of its table row; the scratch slabs are sized to the
/// widest class in the dispatch.
#[derive(Clone, Debug)]
pub struct ThreeC2eLaunchGroup {
    /// Rys order — the kernel's only comptime parameter.
    pub nroots: u32,
    /// [`THREE_C2E_SHAPE_STRIDE`] `u32` per merged class: canonical `li, lj, lk`.
    pub class_shape: Vec<u32>,
    /// One libcint `common_factor` per merged class.
    pub class_factor: Vec<f64>,
    /// `[si, sj, sk, out_off, class]` per triple, in canonical `(i, j)` order.
    pub triples: Vec<u32>,
    /// Total Cartesian output elements across this group's triples.
    pub out_len: usize,
    /// Widest per-slot G, split and work slab lengths in the group.
    pub max_g_size: usize,
    pub max_split_size: usize,
    pub max_work_len: usize,
}

impl ThreeC2eLaunchGroup {
    /// An empty group of Rys order `nroots`.
    #[must_use]
    pub fn new(nroots: u32) -> Self {
        Self {
            nroots,
            class_shape: Vec::new(),
            class_factor: Vec::new(),
            triples: Vec::new(),
            out_len: 0,
            max_g_size: 0,
            max_split_size: 0,
            max_work_len: 0,
        }
    }

    /// Append a class and return the index its triple rows carry.
    ///
    /// `li`/`lj` must already be canonical (`li >= lj`), as the kernel assumes.
    pub fn push_class(&mut self, li: u32, lj: u32, lk: u32, common_factor: f64) -> u32 {
        let index = self.class_factor.len() as u32;
        self.class_shape.extend_from_slice(&[li, lj, lk]);
        self.class_factor.push(common_factor);

        let (li_u, lj_u, lk_u) = (li as usize, lj as usize, lk as usize);
        let nroots_u = self.nroots as usize;
        let nmax = li_u + lj_u;
        self.max_g_size = self.max_g_size.max(nroots_u * (nmax + 1) * (lk_u + 1));
        self.max_split_size = self
            .max_split_size
            .max(nroots_u * (lk_u + 1) * (lj_u + 1) * (li_u + 1));
        self.max_work_len = self.max_work_len.max((lj_u + 1) * (nmax + 1));
        index
    }

    /// Number of triples in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len() / 5
    }

    /// Is this group empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Number of angular-momentum classes merged into this dispatch.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_factor.len()
    }

    /// Bytes this group's triple and class tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        (self.triples.len() + self.class_shape.len()) * std::mem::size_of::<u32>()
            + self.class_factor.len() * std::mem::size_of::<f64>()
    }
}

/// Round a slab length up to a 64-byte cache line so concurrent slots never
/// share a line.
fn three_c2e_slab_stride(elements: usize) -> usize {
    const LINE: usize = 8;
    elements.div_ceil(LINE) * LINE
}

/// Does this backend want the one-triple-per-unit decomposition? Same reasoning
/// and override knob as `two_electron::two_e_per_unit`.
fn three_c2e_per_unit<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let pinned = *OVERRIDE.get_or_init(|| {
        std::env::var("CINTX_3C2E_PER_UNIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    });
    match pinned {
        Some(value) => value != 0,
        None => !crate::plane::has_planes(client),
    }
}

/// Launch geometry for one 3c2e class: `(cube_count, cube_dim, n_slots)`.
fn three_c2e_launch_geometry<R: Runtime>(
    client: &ComputeClient<R>,
    n_triples: usize,
    bytes_per_slot: usize,
) -> (u32, CubeDim, usize) {
    /// Ceiling on the per-launch scratch (G + split + work) across all slots.
    const MAX_BATCH_SCRATCH_BYTES: usize = 128 * 1024 * 1024;

    let by_memory = (MAX_BATCH_SCRATCH_BYTES / bytes_per_slot.max(1)).max(1);

    if three_c2e_per_unit::<R>(client) {
        let units = crate::plane::per_unit_width(
            client,
            n_triples,
            crate::plane::MIN_ITEMS_PER_UNIT_PAIR,
            by_memory,
        );
        return (1, CubeDim::new_1d(units), units as usize);
    }
    // The kernel's arithmetic is not split across a cube, so a wider cube would
    // only add idle lanes.
    let cubes = crate::plane::grid_cube_count(client, n_triples.min(by_memory));
    (cubes, CubeDim::new_1d(1), cubes as usize)
}

/// Evaluate every launch group of a batched 3c2e run: one dispatch and one
/// readback per group, one basis upload for the whole run.
fn run_3c2e_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[ThreeC2eLaunchGroup],
    prim_tol: f64,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

    // The basis buffers are already on the device — uploaded either by this
    // call's throwaway residency or by a caller-held one (Task 34-C2).
    let crate::kernels::two_electron::TwoEBasisHandles {
        exps: exps_h,
        coeffs: coeffs_h,
        centers: centers_h,
        shell_meta: meta_h,
        exps_len,
        coeffs_len,
        centers_len,
        shell_meta_len,
    } = basis;

    let rys_tables = crate::math::rys_wheeler::ext_rys_tables();

    let mut results = Vec::with_capacity(groups.len());
    for class in groups {
        let n_triples = class.len();
        if n_triples == 0 {
            results.push(Vec::new());
            continue;
        }
        // Sized to the widest class merged into this dispatch.
        let g_size = class.max_g_size;
        let split_size = class.max_split_size;
        let work_len = class.max_work_len;

        let g_stride = three_c2e_slab_stride(3 * g_size);
        let split_stride = three_c2e_slab_stride(3 * split_size);
        let work_slab = three_c2e_slab_stride(work_len);
        let bytes_per_slot = (g_stride + split_stride + work_slab) * std::mem::size_of::<f64>();

        let (n_cubes, cube_dim, n_slots) =
            three_c2e_launch_geometry::<R>(client, n_triples, bytes_per_slot);

        let triples_h = client.create_from_slice(u32::as_bytes(&class.triples));
        let shape_h = client.create_from_slice(u32::as_bytes(&class.class_shape));
        let factor_h = client.create_from_slice(f64::as_bytes(&class.class_factor));
        // The extended-Rys constant tables (~4.7 KB) — one buffer, uploaded per
        // dispatch whatever the class's order, because the kernel signature is
        // the same for every `nroots`. Only a class with `nroots >= 6` reads it.
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));
        let g_h = client.empty(n_slots * g_stride * std::mem::size_of::<f64>());
        let gs_h = client.empty(n_slots * split_stride * std::mem::size_of::<f64>());
        let work_h = client.empty(n_slots * work_slab * std::mem::size_of::<f64>());
        let out_h = client.empty(class.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(three_c2e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_triples`, by the per-shell `nprim`/`nctr` read from `shell_meta`,
        // and by the class-uniform G-tensor extents.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    center_3c2e_scalar_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(triples_h.clone(), class.triples.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), class.class_shape.len()),
                        ArrayArg::from_raw_parts(factor_h.clone(), class.class_factor.len()),
                        ArrayArg::from_raw_parts(rys_tab_h.clone(), EXT_TABLES_LEN),
                        ArrayArg::from_raw_parts(g_h.clone(), n_slots * g_stride),
                        ArrayArg::from_raw_parts(gs_h.clone(), n_slots * split_stride),
                        ArrayArg::from_raw_parts(work_h.clone(), n_slots * work_slab),
                        ArrayArg::from_raw_parts(out_h.clone(), class.out_len),
                        PIE4,
                        prim_tol,
                        n_triples as u32,
                        n_cubes,
                        g_stride as u32,
                        split_stride as u32,
                        work_slab as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        // Every reachable order gets its own arm. The upstream ceiling check in
        // `evaluate_3c2e_triple_batch_with` already refused anything above
        // `device_nroots_ceiling`, which is 5 unless `extended-device-rys` is
        // compiled in *and* this backend's FMA probe passed — so the 6..=12 arms
        // are both feature-gated and unreachable without that evidence.
        debug_assert!(
            class.nroots <= three_c2e_launch_nroots_ceiling(),
            "3c2e launch class nroots={} above the compiled ceiling {}",
            class.nroots,
            three_c2e_launch_nroots_ceiling()
        );
        match class.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            #[cfg(feature = "extended-device-rys")]
            6 => launch_with!(6u32),
            #[cfg(feature = "extended-device-rys")]
            7 => launch_with!(7u32),
            #[cfg(feature = "extended-device-rys")]
            8 => launch_with!(8u32),
            #[cfg(feature = "extended-device-rys")]
            9 => launch_with!(9u32),
            #[cfg(feature = "extended-device-rys")]
            10 => launch_with!(10u32),
            #[cfg(feature = "extended-device-rys")]
            11 => launch_with!(11u32),
            #[cfg(feature = "extended-device-rys")]
            12 => launch_with!(12u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..class.out_len].to_vec());
    }
    results
}

/// Spherical AO blocks for a 3c2e triple batch, plus the offsets locating each triple.
#[derive(Clone, Debug, Default)]
pub struct ThreeC2eBatchOutput {
    /// Concatenated spherical AO blocks, in the caller's triple order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where triple `n`'s block starts in [`Self::values`].
    pub offsets: Vec<usize>,
    /// Execution statistics.
    pub stats: crate::kernels::two_electron::BatchExecutionStats,
}

/// Evaluate a list of shell triples as `int3c2e_sph`, one dispatch per launch
/// class (Phase 35).
///
/// This is the family RI-J actually spends its time in: the work list is
/// `nbas^2 * naux` and buckets almost perfectly, because a class is fixed by
/// `(li, lj, lk)` alone.
///
/// The kernel evaluates in canonical `li >= lj` order and the host transposes
/// the `(i, j)` axes back when the caller's order had `li < lj`. That decision
/// is class-uniform — it depends only on the class's `(li, lj)` — so the swap is
/// resolved once per class rather than per triple.
///
/// `triples` are `[i, j, k]` indices into `shells`.
/// Where one `(li,lj,lk)` class landed after launch-group merging (Task 35-M2).
struct ThreeC2eClassPlacement {
    /// Canonical `li >= lj` angular momenta, plus the auxiliary `lk`.
    li: u32,
    lj: u32,
    lk: u32,
    /// Index into the group list — which dispatch's buffer holds these blocks.
    group: usize,
    /// Caller-order indices of this class's triples.
    members: Vec<usize>,
    /// Each member's offset into the group's Cartesian buffer.
    cart_offsets: Vec<usize>,
    /// `(swap_ij, li_in, lj_in)` — the caller's order before canonicalization.
    swap: (bool, u8, u8),
}

pub fn evaluate_3c2e_triple_batch(
    backend: &ResolvedBackend,
    shells: &[crate::kernels::two_electron::BatchShell],
    triples: &[[u32; 3]],
) -> Result<ThreeC2eBatchOutput, cintxRsError> {
    evaluate_3c2e_triple_batch_with(backend, shells, triples, BatchOptions::default())
}

/// [`evaluate_3c2e_triple_batch`] with primitive screening (Task 34-D2).
///
/// `options.primitive_tolerance` is exact at its default `0.0`: the only
/// primitive triples dropped there are the ones whose scale factor underflowed
/// to exactly zero, so the result is bit-identical to no screening at all.
///
/// # Errors
/// As [`evaluate_3c2e_triple_batch`].
pub fn evaluate_3c2e_triple_batch_with(
    backend: &ResolvedBackend,
    shells: &[crate::kernels::two_electron::BatchShell],
    triples: &[[u32; 3]],
    options: BatchOptions,
) -> Result<ThreeC2eBatchOutput, cintxRsError> {
    let resident = ResidentBasis::new(backend, shells)?;
    evaluate_3c2e_triple_batch_resident_with(backend, &resident, triples, options)
}

/// [`evaluate_3c2e_triple_batch`] against a basis already on the device.
///
/// Task 34-C2. Identical results; the difference is that the flattened basis is
/// the caller's [`ResidentBasis`] rather than a throwaway one, so
/// [`BatchExecutionStats::basis_upload_bytes`] is zero on every call after the
/// first. This is the shape RI-J wants: a Fock build evaluates the same
/// `nbas^2 x naux` list every SCF iteration, and re-uploading the exponents,
/// coefficients and centres each time is the waste Task 34-C already removed
/// for `int2e`.
///
/// # Errors
/// Returns [`cintxRsError::UnsupportedApi`] if `resident` was uploaded through a
/// different backend, or if a class needs more Rys roots than the device serves.
pub fn evaluate_3c2e_triple_batch_resident(
    backend: &ResolvedBackend,
    resident: &ResidentBasis,
    triples: &[[u32; 3]],
) -> Result<ThreeC2eBatchOutput, cintxRsError> {
    evaluate_3c2e_triple_batch_resident_with(backend, resident, triples, BatchOptions::default())
}

/// [`evaluate_3c2e_triple_batch_resident`] with primitive screening
/// (Task 34-D2).
///
/// # Errors
/// As [`evaluate_3c2e_triple_batch_resident`].
pub fn evaluate_3c2e_triple_batch_resident_with(
    backend: &ResolvedBackend,
    resident: &ResidentBasis,
    triples: &[[u32; 3]],
    options: BatchOptions,
) -> Result<ThreeC2eBatchOutput, cintxRsError> {
    resident.check_for("3c2e-batch", backend)?;
    let shells = resident.shells();
    evaluate_3c2e_batch_inner(backend, resident, shells, triples, options)
}

fn evaluate_3c2e_batch_inner(
    backend: &ResolvedBackend,
    resident: &ResidentBasis,
    shells: &[crate::kernels::two_electron::BatchShell],
    triples: &[[u32; 3]],
    options: BatchOptions,
) -> Result<ThreeC2eBatchOutput, cintxRsError> {
    let mut offsets = Vec::with_capacity(triples.len());
    let mut total = 0_usize;
    for triple in triples {
        for &s in triple {
            if s as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("3c2e-batch:shell-index-out-of-range:{s}"),
                });
            }
        }
        offsets.push(total);
        total += triple
            .iter()
            .map(|&s| shells[s as usize].ao_len())
            .product::<usize>();
    }

    let mut output = ThreeC2eBatchOutput {
        values: vec![0.0; total],
        offsets,
        stats: crate::kernels::two_electron::BatchExecutionStats {
            quartets: triples.len(),
            ..Default::default()
        },
    };
    if triples.is_empty() {
        return Ok(output);
    }

    // Group by the *caller's* (li, lj, lk); the canonical order is derived from
    // the class key, so every triple in a class swaps or does not swap together.
    let ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int3c2e,
    );

    let mut grouped: std::collections::BTreeMap<[u8; 3], Vec<usize>> = Default::default();
    for (index, triple) in triples.iter().enumerate() {
        let key = [
            shells[triple[0] as usize].l,
            shells[triple[1] as usize].l,
            shells[triple[2] as usize].l,
        ];
        grouped.entry(key).or_default().push(index);
    }

    // Classes are merged into dispatch groups keyed on the kernel's only
    // comptime parameter, the Rys order (Task 35-M2). The `(li,lj,lk)` grouping
    // survives as the sub-grouping the host cart-to-sph and the canonical-order
    // swap are expressed in; only the *launch* is merged.
    let mut groups: Vec<ThreeC2eLaunchGroup> = Vec::new();
    let mut group_of: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut classes: Vec<ThreeC2eClassPlacement> = Vec::with_capacity(grouped.len());
    for (class, members) in grouped {
        let [li_in, lj_in, lk] = class;
        let swap_ij = li_in < lj_in;
        let (li, lj) = if swap_ij {
            (lj_in, li_in)
        } else {
            (li_in, lj_in)
        };
        let nroots = (li as usize + lj as usize + lk as usize) / 2 + 1;
        // Per-backend ceiling (task 33-05): the base value everywhere, raised
        // only on a backend whose FMA-fusion probe passed and only with the
        // `extended-device-rys` opt-in. See `crate::device_rys_ceiling`.
        if nroots > ceiling {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "3c2e-batch:nroots={nroots} exceeds device ceiling {ceiling} \
                     for l=({li_in},{lj_in},{lk})"
                ),
            });
        }

        let nroots = nroots as u32;
        let group_index = match group_of.get(&nroots) {
            Some(&index) => index,
            None => {
                groups.push(ThreeC2eLaunchGroup::new(nroots));
                let index = groups.len() - 1;
                group_of.insert(nroots, index);
                index
            }
        };
        let group = &mut groups[group_index];
        let class_index = group.push_class(
            u32::from(li),
            u32::from(lj),
            u32::from(lk),
            // `CINTinit_int3c2e_EnvVars`: pi^3 * 2/sqrt(pi) * the three fac_sp.
            (PI * PI * PI) * 2.0 / SQRTPI
                * common_fac_sp(li)
                * common_fac_sp(lj)
                * common_fac_sp(lk),
        );

        let cart_block = ncart(li) * ncart(lj) * ncart(lk);
        group.triples.reserve(members.len() * 5);
        let mut cart_offsets = Vec::with_capacity(members.len());
        for &index in &members {
            let t = triples[index];
            let (si, sj) = if swap_ij { (t[1], t[0]) } else { (t[0], t[1]) };
            let nctr_product: usize = t
                .iter()
                .map(|&s| shells[s as usize].nctr as usize)
                .product();
            cart_offsets.push(group.out_len);
            group
                .triples
                .extend_from_slice(&[si, sj, t[2], group.out_len as u32, class_index]);
            group.out_len += nctr_product * cart_block;
        }

        classes.push(ThreeC2eClassPlacement {
            li: u32::from(li),
            lj: u32::from(lj),
            lk: u32::from(lk),
            group: group_index,
            members,
            cart_offsets,
            swap: (swap_ij, li_in, lj_in),
        });
    }

    let dispatch_start = std::time::Instant::now();
    let carts = dispatch_3c2e_batches(
        backend,
        resident.handles(),
        &groups,
        options.primitive_tolerance,
    )?;
    output.stats.dispatch_ns = dispatch_start.elapsed().as_nanos() as u64;

    // Charged to the first evaluation only, so a repeated RI-J build shows the
    // triple tables alone and the amortization is visible rather than asserted.
    output.stats.basis_upload_bytes = if resident.take_first_use() {
        resident.upload_bytes()
    } else {
        0
    };
    output.stats.kernel_launch_count = groups.len();
    output.stats.launch_classes = classes.len();
    output.stats.readback_count = groups.len();
    output.stats.max_g_slab_bytes = groups
        .iter()
        .map(|group| {
            (three_c2e_slab_stride(3 * group.max_g_size)
                + three_c2e_slab_stride(3 * group.max_split_size)
                + three_c2e_slab_stride(group.max_work_len))
                * std::mem::size_of::<f64>()
        })
        .max()
        .unwrap_or(0);
    output.stats.transfer_bytes = output.stats.basis_upload_bytes
        + groups
            .iter()
            .map(ThreeC2eLaunchGroup::upload_bytes)
            .sum::<usize>();

    let transform_start = std::time::Instant::now();
    // Task 36-T1: one output block, one c2s scratch and one transpose buffer
    // per worker, not three per contraction block. Each is fully written before
    // being read on every call, so reuse does not change a single bit.
    //
    // Task 36-T2: one job per triple, in the caller's order, each writing a
    // disjoint output block. Each output element is produced by exactly one
    // triple, so the split reorders no summation.
    let carts = &carts;
    let mut placement = vec![(0_usize, 0_usize); triples.len()];
    for (class_index, class) in classes.iter().enumerate() {
        for (slot, &index) in class.members.iter().enumerate() {
            placement[index] = (class_index, slot);
        }
    }
    let lens: Vec<usize> = triples
        .iter()
        .map(|triple| {
            triple
                .iter()
                .map(|&shell| shells[shell as usize].ao_len())
                .product::<usize>()
        })
        .collect();
    let jobs: Vec<(usize, &mut [f64])> =
        crate::transform::host_batch::split_output_blocks(&mut output.values, &lens)
            .into_iter()
            .enumerate()
            .collect();

    let states = crate::transform::host_batch::for_each_block(
        jobs,
        || {
            (
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                crate::transform::profile::HostTransformProfile::new(),
            )
        },
        |(sph, c2s_scratch, transposed, profile), (index, block)| {
            let (class_index, slot) = placement[index];
            let class = &classes[class_index];
            let (li, lj, lk) = (class.li as u8, class.lj as u8, class.lk as u8);
            let (swap_ij, li_in, lj_in) = class.swap;
            let (nci, ncj, nck) = (ncart(li), ncart(lj), ncart(lk));
            let cart_block = nci * ncj * nck;
            let (nsi_in, nsj_in, nsk) = (nsph(li_in), nsph(lj_in), nsph(lk));

            profile.start();
            sph.clear();
            sph.resize(nsk * nsj_in * nsi_in, 0.0);
            if swap_ij && transposed.len() < cart_block {
                transposed.resize(cart_block, 0.0);
            }
            profile.charge_alloc();

            let cart = &carts[class.group];
            let t = triples[index];
            let (nci_ctr, ncj_ctr, nck_ctr) = (
                shells[t[0] as usize].nctr as usize,
                shells[t[1] as usize].nctr as usize,
                shells[t[2] as usize].nctr as usize,
            );
            // The kernel wrote the class's contraction blocks in *canonical*
            // (i, j) order — block `(ca, cb, ck)` at
            // `((ca * nctr_canonical_j + cb) * nctr_k + ck)` — so the block index
            // has to be swapped exactly the way the shell indices were.
            let ctr_stride_b = if swap_ij { nci_ctr } else { ncj_ctr };
            let di = nci_ctr * nsi_in;
            let dj = ncj_ctr * nsj_in;
            let src_base = class.cart_offsets[slot];

            for ci in 0..nci_ctr {
                for cj in 0..ncj_ctr {
                    for ck in 0..nck_ctr {
                        let (ca, cb) = if swap_ij { (cj, ci) } else { (ci, cj) };
                        let base =
                            src_base + ((ca * ctr_stride_b + cb) * nck_ctr + ck) * cart_block;
                        let cart_slice = &cart[base..base + cart_block];
                        if swap_ij {
                            transpose_ij_3idx_into(
                                cart_slice,
                                nci,
                                ncj,
                                nck,
                                &mut transposed[..cart_block],
                            );
                            cart_to_sph_3c2e_into(
                                &transposed[..cart_block],
                                li_in,
                                lj_in,
                                lk,
                                sph,
                                c2s_scratch,
                            );
                        } else {
                            cart_to_sph_3c2e_into(cart_slice, li_in, lj_in, lk, sph, c2s_scratch);
                        }
                        profile.charge_transform();
                        for mk in 0..nsk {
                            let kidx = ck * nsk + mk;
                            for mj in 0..nsj_in {
                                let jidx = cj * nsj_in + mj;
                                for mi in 0..nsi_in {
                                    let iidx = ci * nsi_in + mi;
                                    block[iidx + di * (jidx + dj * kidx)] =
                                        sph[mi + nsi_in * (mj + nsj_in * mk)];
                                }
                            }
                        }
                        profile.charge_scatter();
                    }
                }
            }
            profile.pause();
        },
    );

    let mut profile = crate::transform::profile::HostTransformProfile::new();
    for (_, _, _, worker) in &states {
        profile.merge(worker);
    }
    output.stats.host_transform_ns = transform_start.elapsed().as_nanos() as u64;
    profile.store_into(&mut output.stats);

    Ok(output)
}

/// Backend dispatch for a whole batched 3c2e run.
fn dispatch_3c2e_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[ThreeC2eLaunchGroup],
    prim_tol: f64,
) -> Result<Vec<Vec<f64>>, cintxRsError> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => Ok(run_3c2e_batches::<cubecl::cpu::CpuRuntime>(
            client, basis, groups, prim_tol,
        )),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => Ok(run_3c2e_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups, prim_tol,
        )),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => Ok(run_3c2e_batches::<cubecl_cuda::CudaRuntime>(
            client, basis, groups, prim_tol,
        )),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => Ok(run_3c2e_batches::<cubecl_hip::HipRuntime>(
            client, basis, groups, prim_tol,
        )),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => Ok(run_3c2e_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups, prim_tol,
        )),
    }
}

/// Single-triple dispatch — a one-class, one-triple batch.
///
/// Kept as its own entry point because the per-tuple compatibility API evaluates
/// exactly one shell triple. It marshals the three shells into the flattened
/// form [`run_3c2e_batches`] consumes, so both paths execute the *same* kernel
/// and every existing parity test covers the batched code at `n_triples == 1`.
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
    let (li_u, lj_u, lk_u) = (li as usize, lj as usize, lk as usize);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let nck = (lk_u + 1) * (lk_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * (nctr_k as usize) * nci * ncj * nck;

    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (exps, coeffs, center, nprim, nctr) in [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_j, coeff_j, rj, nprim_j, nctr_j),
        (exps_k, coeff_k, rk, nprim_k, nctr_k),
    ] {
        basis.shell_meta.extend_from_slice(&[
            basis.exps.len() as u32,
            basis.coeffs.len() as u32,
            nprim,
            nctr,
        ]);
        basis.exps.extend_from_slice(exps);
        basis.coeffs.extend_from_slice(coeffs);
        basis.centers.extend_from_slice(&center);
    }

    let mut group = ThreeC2eLaunchGroup::new(nroots);
    let class_index = group.push_class(li, lj, lk, common_factor);
    group.triples.extend_from_slice(&[0, 1, 2, 0, class_index]);
    group.out_len = out_len;

    let handles = crate::kernels::two_electron::upload_2e_basis::<R>(client, &basis);
    // The per-tuple compatibility path never screens: it has no options.
    run_3c2e_batches::<R>(client, &handles, std::slice::from_ref(&group), 0.0)
        .pop()
        .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  int3c2e_ip1 device kernel — `#[cube(launch)]`, generic over `F: Float`
// ─────────────────────────────────────────────────────────────────────────────

/// int3c2e_ip1 (∇_A first-center derivative, 3 components) for one shell triple,
/// on-device. Single work item (`UNIT_POS == 0`).
///
/// Faithful inline port of the host pipeline `fill_g_tensor_2e` (kbase=false only;
/// `hrr_lj2d_4d` for ibase==0, `hrr_il2d_4d` for ibase==1) → `nabla1i_2e` → the
/// `gout_ip1` contraction, applying the 3c2e Pitfall-4 kl mapping
/// (`build_2e_shape(li+1, lj, 0, lk)`: real k → 2e `ll` slot; phantom 2e `lk` slot
/// size 1 with exponent 0; bra `i` raised to `li+1` for the ∇_i headroom).
///
/// `kbase` is ALWAYS false for this mapping (`0 > lk` is never true), so only the
/// kbase==false HRR branches are reachable; the kbase==true branches
/// (`hrr_ik2d_4d`/`hrr_kj2d_4d`) are dead and not inlined.
///
/// Strides (`di,dk,dl,dj,g_size`) and `nmax`/`mmax` are computed host-side from
/// `build_2e_shape(li+1, lj, 0, lk)` and passed as runtime `u32`. `ibase` is a
/// runtime `u32` 0/1. `#[comptime] nroots` selects `rys_root{1..5}`.
///
/// Output `cart_out` (size `3*nci*ncj*nck`, component-leading `[3][nk][nj][ni]`,
/// i fastest within each component) is zeroed in-kernel and accumulated.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn center_3c2e_ip1_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    triples: &Array<u32>,
    class_shape: &Array<u32>,
    class_factor: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    n_triples: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — identical to the scalar 3c2e kernel, and to
    // `two_electron.rs` for why this is arithmetic on comptime-folded flags
    // rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Read and written entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;

        // `g` and `g1` share a slot index and a stride, so one base serves both.
        // The contraction below reads them at the *same* relative offsets, which
        // is only sound because their layouts are identical.
        let gbase = slot * g_stride;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
        #[allow(clippy::manual_div_ceil)]
        let chunk = (n_triples + n_slots - 1u32) / n_slots;
        let qi_start = slot * (chunk * punit + coop);
        let mut qi_stop = (qi_start + chunk) * punit + n_triples * coop;
        if qi_stop > n_triples {
            qi_stop = n_triples;
        }
        let qi_step = n_slots * coop + punit;

        let mut qi = qi_start;
        while qi < qi_stop {
            let trow = qi * 5u32;
            let si = triples[trow as usize];
            let sj = triples[(trow + 1u32) as usize];
            let sk = triples[(trow + 2u32) as usize];
            let out_off = triples[(trow + 3u32) as usize];

            // ── Per-class shape (Task 35-D) ───────────────────────────────
            //
            // `nroots` is this kernel's only shape-bearing comptime parameter,
            // so one dispatch carries every `(li,lj,lk)` class of the same Rys
            // order. The G slabs are sized to the widest class in the dispatch
            // and each class indexes only what it owns.
            let cls = triples[(trow + 4u32) as usize];
            let srow = cls * comptime!(THREE_C2E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];
            let lk = class_shape[(srow + 2u32) as usize];
            let di = class_shape[(srow + 3u32) as usize];
            let dk = class_shape[(srow + 4u32) as usize];
            let dl = class_shape[(srow + 5u32) as usize];
            let dj = class_shape[(srow + 6u32) as usize];
            let g_size = class_shape[(srow + 7u32) as usize];
            let nmax = class_shape[(srow + 8u32) as usize];
            let mmax = class_shape[(srow + 9u32) as usize];
            let ibase = class_shape[(srow + 10u32) as usize];
            let common_factor = class_factor[cls as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];
            let mk = sk * 4u32;
            let eoff_k = shell_meta[mk as usize];
            let coff_k = shell_meta[(mk + 1u32) as usize];
            let nprim_k = shell_meta[(mk + 2u32) as usize];
            let nctr_k = shell_meta[(mk + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];
            let ck3 = sk * 3u32;
            let rkx = centers[ck3 as usize];
            let rky = centers[(ck3 + 1u32) as usize];
            let rkz = centers[(ck3 + 2u32) as usize];

            let total_g = 3u32 * g_size;
            let gy_off = gbase + g_size;
            let gz_off = gbase + 2u32 * g_size;

            // Elevated bra angular momentum used for the VRR/HRR loop bounds (li+1).
            // The 2e `lk` slot is the phantom (size 1, base lk=0); real k lives in `ll`.
            let li_e = li + 1u32;
            let ll = lk; // real k mapped into the 2e ll-slot
            let lk2e = 0u32; // phantom 2e lk slot

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
            let block_len = nci * ncj * nck;
            let total_len = 3u32 * block_len; // per-(ci,cj,ck) component-leading block
            let out_len = nctr_i * nctr_j * nctr_k * total_len;

            let mut oi = out_off;
            while oi < out_off + out_len {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut kp = 0u32;
            while kp < nprim_k {
                let ak = exps[(eoff_k + kp) as usize];
                let mut jp = 0u32;
                while jp < nprim_j {
                    let aj = exps[(eoff_j + jp) as usize];
                    let mut ip = 0u32;
                    while ip < nprim_i {
                        let ai = exps[(eoff_i + ip) as usize];

                        // ── Inlined pdata_ij (bra) and pdata_kl (phantom-real_k) ──
                        let zeta_ab = ai + aj;
                        let rij_dx = rix - rjx;
                        let rij_dy = riy - rjy;
                        let rij_dz = riz - rjz;
                        let rr_ij = rij_dx * rij_dx + rij_dy * rij_dy + rij_dz * rij_dz;
                        let pdata_ij_fac = F::exp(-ai * aj / zeta_ab * rr_ij);
                        // pdata_kl: zeta=ak, center=rk, rr=0 → fac=1.
                        let pdata_kl_fac = F::new(1.0_f32);
                        let fac_env = common_factor * pdata_ij_fac * pdata_kl_fac;

                        // ── fill_g_tensor_2e math (ai,aj | 0,ak at ri,rj | rk,rk) ──
                        let aij = zeta_ab;
                        let akl = ak; // ak + al with al=ak? No: 2e lk slot exp=0, ll slot exp=ak.
                        // aij = ai+aj; akl = ak(lk slot=0) + ak(ll slot)? In the host call
                        // fill_g_tensor_2e(ai,aj,0.0,ak,..) → akl = 0.0 + ak = ak.
                        let rij_x = (ai * rix + aj * rjx) / aij;
                        let rij_y = (ai * riy + aj * rjy) / aij;
                        let rij_z = (ai * riz + aj * rjz) / aij;
                        // rkl = (0*rk + ak*rk)/ak = rk.
                        let rkl_x = rkx;
                        let rkl_y = rky;
                        let rkl_z = rkz;

                        let xij_kl = rij_x - rkl_x;
                        let yij_kl = rij_y - rkl_y;
                        let zij_kl = rij_z - rkl_z;
                        let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

                        let a1 = aij * akl;
                        let a0 = a1 / (aij + akl);
                        let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * fac_env;
                        let x_rys = a0 * rr;

                        // ibase selects rx_in_rijrx; kbase is always false → rklrx = 0.
                        // rirj used in the HRR transfer.
                        let mut rx_rij_x = rjx;
                        let mut rx_rij_y = rjy;
                        let mut rx_rij_z = rjz;
                        let mut rirj_x = rjx - rix;
                        let mut rirj_y = rjy - riy;
                        let mut rirj_z = rjz - riz;
                        if ibase == 1u32 {
                            rx_rij_x = rix;
                            rx_rij_y = riy;
                            rx_rij_z = riz;
                            rirj_x = rix - rjx;
                            rirj_y = riy - rjy;
                            rirj_z = riz - rjz;
                        }
                        let rijrx_x = rij_x - rx_rij_x;
                        let rijrx_y = rij_y - rx_rij_y;
                        let rijrx_z = rij_z - rx_rij_z;
                        // rklrx = rkl - rl(=rk) = 0; rkrl = rl - rk = 0.

                        // Rys roots/weights.
                        if comptime!(nroots == 1u32) {
                            rys_root1::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 2u32) {
                            rys_root2::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 3u32) {
                            rys_root3::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 4u32) {
                            rys_root4::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else {
                            rys_root5::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        }

                        // Zero g.
                        let mut gi = gbase;
                        while gi < gbase + total_g {
                            g[gi as usize] = F::new(0.0_f32);
                            gi += 1u32;
                        }

                        // g2d strides for VRR: g2d_ijmax = ibase? di : dj;
                        //                      g2d_klmax = kbase? dk : dl  (kbase=false → dl).
                        let mut g2d_ijmax = dj;
                        if ibase == 1u32 {
                            g2d_ijmax = di;
                        }
                        let g2d_klmax = dl;

                        // Base + VRR per axis (vrr_fill_axis inlined for 3 axes).
                        let mut irys = 0u32;
                        while irys < nrys {
                            g[(gbase + irys) as usize] = F::new(1.0_f32);
                            g[(gy_off + irys) as usize] = F::new(1.0_f32);
                            g[(gz_off + irys) as usize] = wrys[irys as usize] * fac1;
                            irys += 1u32;
                        }

                        let mut irys2 = 0u32;
                        while irys2 < nrys {
                            let u2 = a0 * urys[irys2 as usize];
                            let tmp4 = F::new(0.5_f32) / (u2 * (aij + akl) + a1);
                            let tmp5 = u2 * tmp4;
                            let tmp1 = F::new(2.0_f32) * tmp5;
                            let tmp2 = tmp1 * akl;
                            let tmp3 = tmp1 * aij;
                            let b00 = tmp5;
                            let b10 = tmp5 + tmp4 * akl;
                            let b01 = tmp5 + tmp4 * aij;

                            let mut axis = 0u32;
                            while axis < 3u32 {
                                let base = gbase + axis * g_size;
                                let mut d = xij_kl;
                                let mut rijrx = rijrx_x;
                                if axis == 1u32 {
                                    d = yij_kl;
                                    rijrx = rijrx_y;
                                }
                                if axis == 2u32 {
                                    d = zij_kl;
                                    rijrx = rijrx_z;
                                }
                                let c00 = rijrx - tmp2 * d;
                                // c0p = rklrx + tmp3*d; rklrx = 0.
                                let c0p = tmp3 * d;

                                // vrr_fill_axis with dn=g2d_ijmax, dm=g2d_klmax.
                                let dn = g2d_ijmax;
                                let dm = g2d_klmax;
                                let root = base + irys2;

                                // n-ladder (nmax).
                                if nmax >= 1u32 {
                                    let mut s0 = g[root as usize];
                                    let mut s1 = c00 * s0;
                                    g[(root + dn) as usize] = s1;
                                    let mut n = 1u32;
                                    while n < nmax {
                                        let s2 = c00 * s1 + F::cast_from(n) * b10 * s0;
                                        g[(root + (n + 1u32) * dn) as usize] = s2;
                                        s0 = s1;
                                        s1 = s2;
                                        n += 1u32;
                                    }
                                }

                                // m-ladder (mmax) at n=0, then n=1 cross term, then n>0 fill.
                                if mmax >= 1u32 {
                                    let mut s0 = g[root as usize];
                                    let mut s1 = c0p * s0;
                                    g[(root + dm) as usize] = s1;
                                    let mut m = 1u32;
                                    while m < mmax {
                                        let s2 = c0p * s1 + F::cast_from(m) * b01 * s0;
                                        g[(root + (m + 1u32) * dm) as usize] = s2;
                                        s0 = s1;
                                        s1 = s2;
                                        m += 1u32;
                                    }

                                    if nmax >= 1u32 {
                                        let mut s0n = g[(root + dn) as usize];
                                        let mut s1n = c0p * s0n + b00 * g[root as usize];
                                        g[(root + dn + dm) as usize] = s1n;
                                        let mut m2 = 1u32;
                                        while m2 < mmax {
                                            let s2n = c0p * s1n
                                                + F::cast_from(m2) * b01 * s0n
                                                + b00 * g[(root + m2 * dm) as usize];
                                            g[(root + dn + (m2 + 1u32) * dm) as usize] = s2n;
                                            s0n = s1n;
                                            s1n = s2n;
                                            m2 += 1u32;
                                        }
                                    }
                                }

                                if nmax >= 1u32 {
                                    let mut m3 = 1u32;
                                    while m3 <= mmax {
                                        let off = m3 * dm;
                                        let j = off + root;
                                        let mut s0 = g[j as usize];
                                        let mut s1 = g[(j + dn) as usize];
                                        let mut n2 = 1u32;
                                        while n2 < nmax {
                                            let s2 = c00 * s1
                                                + F::cast_from(n2) * b10 * s0
                                                + F::cast_from(m3)
                                                    * b00
                                                    * g[(j + n2 * dn - dm) as usize];
                                            g[(j + (n2 + 1u32) * dn) as usize] = s2;
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

                        // ── HRR transfer (kbase==false): ibase selects branch. ─────
                        if ibase == 0u32 {
                            // hrr_lj2d_4d (li-then-k transfer).
                            if li_e != 0u32 || lk2e != 0u32 {
                                let mut axis = 0u32;
                                while axis < 3u32 {
                                    let off = gbase + axis * g_size;
                                    let mut rx = rirj_x;
                                    if axis == 1u32 {
                                        rx = rirj_y;
                                    }
                                    if axis == 2u32 {
                                        rx = rirj_z;
                                    }

                                    // i-transfer.
                                    let mut i = 1u32;
                                    while i <= li_e {
                                        let jmax = nmax - i;
                                        let mut jjj = 0u32;
                                        while jjj <= jmax {
                                            let mut l = 0u32;
                                            while l <= mmax {
                                                let ptr = jjj * dj + l * dl + i * di;
                                                let mut r = 0u32;
                                                while r < nrys {
                                                    let idx = ptr + r;
                                                    g[(off + idx) as usize] = rx
                                                        * g[(off + idx - di) as usize]
                                                        + g[(off + idx - di + dj) as usize];
                                                    r += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            jjj += 1u32;
                                        }
                                        i += 1u32;
                                    }

                                    // k-transfer (rkrl = 0).
                                    let mut rxk = F::new(0.0_f32);
                                    let _ = &mut rxk;
                                    let mut jjj2 = 0u32;
                                    while jjj2 <= lj {
                                        let mut k = 1u32;
                                        while k <= lk2e {
                                            let lmax = mmax - k;
                                            let mut l = 0u32;
                                            while l <= lmax {
                                                let ptr = jjj2 * dj + l * dl + k * dk;
                                                let mut n = 0u32;
                                                while n < dk {
                                                    let idx = ptr + n;
                                                    g[(off + idx) as usize] = rxk
                                                        * g[(off + idx - dk) as usize]
                                                        + g[(off + idx - dk + dl) as usize];
                                                    n += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            k += 1u32;
                                        }
                                        jjj2 += 1u32;
                                    }

                                    axis += 1u32;
                                }
                            }
                        } else {
                            // hrr_il2d_4d (k-then-j transfer).
                            if lj != 0u32 || lk2e != 0u32 {
                                let mut axis = 0u32;
                                while axis < 3u32 {
                                    let off = gbase + axis * g_size;
                                    // k-transfer (rkrl = 0).
                                    let rxk = F::new(0.0_f32);
                                    let mut k = 1u32;
                                    while k <= lk2e {
                                        let lmax = mmax - k;
                                        let mut l = 0u32;
                                        while l <= lmax {
                                            let mut i = 0u32;
                                            while i <= nmax {
                                                let ptr = l * dl + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nrys {
                                                    let idx = ptr + r;
                                                    g[(off + idx) as usize] = rxk
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

                                    // j-transfer (rirj).
                                    let mut rx = rirj_x;
                                    if axis == 1u32 {
                                        rx = rirj_y;
                                    }
                                    if axis == 2u32 {
                                        rx = rirj_z;
                                    }
                                    let mut jjj = 1u32;
                                    while jjj <= lj {
                                        let mut l = 0u32;
                                        while l <= ll {
                                            let mut k2 = 0u32;
                                            while k2 <= lk2e {
                                                let ptr = jjj * dj + l * dl + k2 * dk;
                                                let imax = nmax - jjj;
                                                let mut i = 0u32;
                                                while i <= imax {
                                                    let base2 = ptr + i * di;
                                                    let mut r = 0u32;
                                                    while r < nrys {
                                                        let idx = base2 + r;
                                                        g[(off + idx) as usize] = rx
                                                            * g[(off + idx - dj) as usize]
                                                            + g[(off + idx - dj + di) as usize];
                                                        r += 1u32;
                                                    }
                                                    i += 1u32;
                                                }
                                                k2 += 1u32;
                                            }
                                            l += 1u32;
                                        }
                                        jjj += 1u32;
                                    }

                                    axis += 1u32;
                                }
                            }
                        }

                        // ── nabla1i_2e → g1 (at base li; g has li+1 headroom). ─────
                        let mut g1i = gbase;
                        while g1i < gbase + total_g {
                            g1[g1i as usize] = F::new(0.0_f32);
                            g1i += 1u32;
                        }
                        let ai2 = F::new(-2.0_f32) * ai;
                        let mut axisn = 0u32;
                        while axisn < 3u32 {
                            let off = gbase + axisn * g_size;
                            let mut jn = 0u32;
                            while jn <= lj {
                                let mut ln = 0u32;
                                while ln <= ll {
                                    let mut kn = 0u32;
                                    while kn <= lk2e {
                                        let ptr = dj * jn + dl * ln + dk * kn;
                                        // i=0.
                                        let mut n = ptr;
                                        while n < ptr + nrys {
                                            g1[(off + n) as usize] =
                                                ai2 * g[(off + n + di) as usize];
                                            n += 1u32;
                                        }
                                        // i>=1 (base li, not li_e).
                                        let mut i = 1u32;
                                        while i <= li {
                                            let ptr2 = dj * jn + dl * ln + dk * kn + di * i;
                                            let mut n2 = ptr2;
                                            while n2 < ptr2 + nrys {
                                                g1[(off + n2) as usize] = F::cast_from(i)
                                                    * g[(off + n2 - di) as usize]
                                                    + ai2 * g[(off + n2 + di) as usize];
                                                n2 += 1u32;
                                            }
                                            i += 1u32;
                                        }
                                        kn += 1u32;
                                    }
                                    ln += 1u32;
                                }
                                jn += 1u32;
                            }
                            axisn += 1u32;
                        }

                        // ── gout_ip1 contraction (n walks [ll=k][lk2e=phantom][j][i]) ──
                        // s[0]=g1x*g0y*g0z, s[1]=g0x*g1y*g0z, s[2]=g0x*g0y*g1z.
                        // n increments i-fastest; transpose to component-leading.
                        let gx_off = gbase;

                        let mut cci = 0u32;
                        while cci < nctr_i {
                            let coeff_i_val = coeffs[(coff_i + ip * nctr_i + cci) as usize];
                            let mut ccj = 0u32;
                            while ccj < nctr_j {
                                let coeff_j_val = coeffs[(coff_j + jp * nctr_j + ccj) as usize];
                                let mut cck = 0u32;
                                while cck < nctr_k {
                                    let coeff_k_val = coeffs[(coff_k + kp * nctr_k + cck) as usize];
                                    let weight = coeff_i_val * coeff_j_val * coeff_k_val;
                                    let ctr_base =
                                        out_off + ((cci * nctr_j + ccj) * nctr_k + cck) * total_len;

                                    // n walk: cl(real k, ll) slowest → ck(phantom) →
                                    // cj → ci fastest. Reproduce cart_comps ordering.
                                    let mut n = 0u32;
                                    // l = real k (ll slot).
                                    let mut la = 0u32;
                                    while la <= ll {
                                        let lx = ll - la;
                                        let ll_minus_lx = ll - lx;
                                        let mut lb = 0u32;
                                        while lb <= ll_minus_lx {
                                            let ly = ll_minus_lx - lb;
                                            let lz = ll - lx - ly;

                                            // k = phantom (lk2e, size 1): only (0,0,0).
                                            let kx = 0u32;
                                            let ky = 0u32;
                                            let kz = 0u32;

                                            // j.
                                            let mut ja = 0u32;
                                            while ja <= lj {
                                                let jx = lj - ja;
                                                let lj_minus_jx = lj - jx;
                                                let mut jb = 0u32;
                                                while jb <= lj_minus_jx {
                                                    let jy = lj_minus_jx - jb;
                                                    let jz = lj - jx - jy;

                                                    // i (base li).
                                                    let mut ia = 0u32;
                                                    while ia <= li {
                                                        let ix = li - ia;
                                                        let li_minus_ix = li - ix;
                                                        let mut ib = 0u32;
                                                        while ib <= li_minus_ix {
                                                            let iy = li_minus_ix - ib;
                                                            let iz = li - ix - iy;

                                                            let ix_base = ix * di
                                                                + kx * dk
                                                                + lx * dl
                                                                + jx * dj;
                                                            let iy_base = iy * di
                                                                + ky * dk
                                                                + ly * dl
                                                                + jy * dj;
                                                            let iz_base = iz * di
                                                                + kz * dk
                                                                + lz * dl
                                                                + jz * dj;

                                                            let mut s0 = F::new(0.0_f32);
                                                            let mut s1 = F::new(0.0_f32);
                                                            let mut s2 = F::new(0.0_f32);
                                                            let mut r = 0u32;
                                                            while r < nrys {
                                                                let g1x = g1[(gx_off + ix_base + r)
                                                                    as usize];
                                                                let g0x = g[(gx_off + ix_base + r)
                                                                    as usize];
                                                                let g1y = g1[(gy_off + iy_base + r)
                                                                    as usize];
                                                                let g0y = g[(gy_off + iy_base + r)
                                                                    as usize];
                                                                let g1z = g1[(gz_off + iz_base + r)
                                                                    as usize];
                                                                let g0z = g[(gz_off + iz_base + r)
                                                                    as usize];
                                                                s0 += g1x * g0y * g0z;
                                                                s1 += g0x * g1y * g0z;
                                                                s2 += g0x * g0y * g1z;
                                                                r += 1u32;
                                                            }

                                                            // Component-leading accumulate.
                                                            cart_out[(ctr_base
                                                                + 0u32 * block_len
                                                                + n)
                                                                as usize] += weight * s0;
                                                            cart_out[(ctr_base
                                                                + 1u32 * block_len
                                                                + n)
                                                                as usize] += weight * s1;
                                                            cart_out[(ctr_base
                                                                + 2u32 * block_len
                                                                + n)
                                                                as usize] += weight * s2;

                                                            n += 1u32;
                                                            ib += 1u32;
                                                        }
                                                        ia += 1u32;
                                                    }

                                                    jb += 1u32;
                                                }
                                                ja += 1u32;
                                            }

                                            lb += 1u32;
                                        }
                                        la += 1u32;
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

            qi += qi_step;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Batched 3c2e derivative dispatch (Task 35-D)
// ─────────────────────────────────────────────────────────────────────────────

/// Which 3c2e derivative kernel a launch group dispatches.
///
/// The two kernels take the same arguments and differ only in which centre the
/// `\nabla` acts on and how much G-tensor headroom the host sized for them, so
/// one runner serves both and this selects the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreeC2eDerivFamily {
    /// `int3c2e_ip1` — `\nabla` on the bra `i` centre.
    Ip1,
    /// `int3c2e_ip2` — `\nabla` on the auxiliary `k` centre.
    Ip2,
}

impl ThreeC2eDerivFamily {
    fn label(self) -> &'static str {
        match self {
            Self::Ip1 => "3c2e-ip1",
            Self::Ip2 => "3c2e-ip2",
        }
    }
}

/// Evaluate every launch group of a batched 3c2e derivative run.
///
/// One dispatch and one readback per group, and the basis is already on the
/// device — the same contract the scalar path has had since Task 34-C.
fn run_3c2e_deriv_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[ThreeC2eDerivLaunchGroup],
    family: ThreeC2eDerivFamily,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

    let crate::kernels::two_electron::TwoEBasisHandles {
        exps: exps_h,
        coeffs: coeffs_h,
        centers: centers_h,
        shell_meta: meta_h,
        exps_len,
        coeffs_len,
        centers_len,
        shell_meta_len,
    } = basis;

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_triples = group.len();
        if n_triples == 0 {
            results.push(Vec::new());
            continue;
        }
        // `g` and `g1` are separate buffers with the same per-slot stride: the
        // contraction reads them at identical relative offsets, so their layouts
        // must match. Sized to the widest class merged into this dispatch.
        let g_stride = three_c2e_slab_stride(3 * group.max_g_size);
        // Two slabs per slot.
        let bytes_per_slot = 2 * g_stride * std::mem::size_of::<f64>();
        let (n_cubes, cube_dim, n_slots) =
            three_c2e_launch_geometry::<R>(client, n_triples, bytes_per_slot);
        let g_len = n_slots * g_stride;

        let triples_h = client.create_from_slice(u32::as_bytes(&group.triples));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let factor_h = client.create_from_slice(f64::as_bytes(&group.class_factor));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(three_c2e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_triples`, by the class index in each triple row, by the per-shell
        // `nprim`/`nctr` read from `shell_meta`, and by the per-class G extents.
        macro_rules! launch_with {
            ($kernel:path, $nr:expr) => {
                unsafe {
                    $kernel(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(triples_h.clone(), group.triples.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(factor_h.clone(), group.class_factor.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        PIE4,
                        n_triples as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        macro_rules! launch_family {
            ($nr:expr) => {
                match family {
                    ThreeC2eDerivFamily::Ip1 => {
                        launch_with!(center_3c2e_ip1_kernel::launch_unchecked::<f64, R>, $nr)
                    }
                    ThreeC2eDerivFamily::Ip2 => {
                        launch_with!(center_3c2e_ip2_kernel::launch_unchecked::<f64, R>, $nr)
                    }
                }
            };
        }

        match group.nroots {
            1 => launch_family!(1u32),
            2 => launch_family!(2u32),
            3 => launch_family!(3u32),
            4 => launch_family!(4u32),
            _ => launch_family!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched 3c2e derivative run.
fn dispatch_3c2e_deriv_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[ThreeC2eDerivLaunchGroup],
    family: ThreeC2eDerivFamily,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_3c2e_deriv_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, family)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_3c2e_deriv_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, family)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_3c2e_deriv_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, family)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_3c2e_deriv_batches::<cubecl_hip::HipRuntime>(client, basis, groups, family)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_3c2e_deriv_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, family)
        }
    }
}

/// One shell triple through the batched derivative path.
///
/// The per-tuple compatibility API evaluates exactly one triple and must keep
/// doing so, but it goes through the *same* kernel as a wide batch — a
/// one-triple group. That is what makes every existing `int3c2e_ip1` /
/// `int3c2e_ip2` parity test a test of the batched kernel too (Task 35-D's
/// acceptance bar), rather than of a second code path that merely ought to agree.
#[allow(clippy::too_many_arguments)]
fn run_3c2e_deriv_single(
    backend: &ResolvedBackend,
    family: ThreeC2eDerivFamily,
    li: u32,
    lj: u32,
    lk: u32,
    shape: &crate::kernels::two_electron::TwoEShape,
    common_factor: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    nprim: [u32; 3],
    nctr: [u32; 3],
    exps: [&[f64]; 3],
    coeffs: [&[f64]; 3],
) -> Vec<f64> {
    let _ = family.label();
    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (index, center) in [ri, rj, rk].into_iter().enumerate() {
        basis.shell_meta.extend_from_slice(&[
            basis.exps.len() as u32,
            basis.coeffs.len() as u32,
            nprim[index],
            nctr[index],
        ]);
        basis.exps.extend_from_slice(exps[index]);
        basis.coeffs.extend_from_slice(coeffs[index]);
        basis.centers.extend_from_slice(&center);
    }

    let nci = ((li + 1) * (li + 2) / 2) as usize;
    let ncj = ((lj + 1) * (lj + 2) / 2) as usize;
    let nck = ((lk + 1) * (lk + 2) / 2) as usize;
    let out_len = nctr[0] as usize * nctr[1] as usize * nctr[2] as usize * 3 * nci * ncj * nck;

    let mut group = ThreeC2eDerivLaunchGroup::new(shape.nroots as u32);
    let class_index = group.push_class(li, lj, lk, shape, common_factor);
    group.triples.extend_from_slice(&[0, 1, 2, 0, class_index]);
    group.out_len = out_len;

    let handles = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            crate::kernels::two_electron::upload_2e_basis::<cubecl::cpu::CpuRuntime>(client, &basis)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => crate::kernels::two_electron::upload_2e_basis::<
            cubecl_wgpu::WgpuRuntime,
        >(client, &basis),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => crate::kernels::two_electron::upload_2e_basis::<
            cubecl_cuda::CudaRuntime,
        >(client, &basis),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            crate::kernels::two_electron::upload_2e_basis::<cubecl_hip::HipRuntime>(client, &basis)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => crate::kernels::two_electron::upload_2e_basis::<
            cubecl_wgpu::WgpuRuntime,
        >(client, &basis),
    };

    dispatch_3c2e_deriv_batches(backend, &handles, std::slice::from_ref(&group), family)
        .pop()
        .unwrap_or_default()
}

/// Spherical AO blocks for a batched 3c2e derivative run, plus the offsets that
/// locate each triple.
///
/// Each triple's block is **component-leading**: three consecutive
/// `[nk][nj][ni]` spherical tensors, `i` fastest, contraction-major on all three
/// axes — the same layout the per-triple path writes to staging.
#[derive(Clone, Debug, Default)]
pub struct ThreeC2eDerivBatchOutput {
    /// Concatenated spherical AO blocks, in the caller's triple order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where triple `n`'s 3-component block starts.
    pub offsets: Vec<usize>,
    /// Execution statistics.
    pub stats: crate::kernels::two_electron::BatchExecutionStats,
}

/// Where one `(li,lj,lk)` class landed after launch-group merging.
struct ThreeC2eDerivPlacement {
    li: u8,
    lj: u8,
    lk: u8,
    /// Index into the group list — which dispatch's buffer holds these blocks.
    group: usize,
    /// Caller-order indices of this class's triples.
    members: Vec<usize>,
    /// Each member's offset into the group's Cartesian buffer.
    cart_offsets: Vec<usize>,
}

/// Evaluate a list of shell triples as `int3c2e_ip1` or `int3c2e_ip2`, one
/// dispatch per Rys order (Task 35-D).
///
/// This is what makes the derivative families *batched* rather than merely
/// batch-capable: before this, every triple cost its own launch, and on an RI-J
/// gradient list that is `nbas^2 x naux` launches.
///
/// Output blocks are spherical and component-leading — byte-identical to what
/// the per-triple path writes for the same triple.
///
/// # Errors
/// Returns [`cintxRsError::UnsupportedApi`] when a class needs more Rys roots
/// than the device serves, or when a shell index is out of range. The batch is
/// rejected as a whole rather than partly evaluated.
pub fn evaluate_3c2e_deriv_triple_batch(
    backend: &ResolvedBackend,
    family: ThreeC2eDerivFamily,
    shells: &[crate::kernels::two_electron::BatchShell],
    triples: &[[u32; 3]],
) -> Result<ThreeC2eDerivBatchOutput, cintxRsError> {
    let resident = ResidentBasis::new(backend, shells)?;
    evaluate_3c2e_deriv_triple_batch_resident(backend, family, &resident, triples)
}

/// [`evaluate_3c2e_deriv_triple_batch`] against a basis already on the device.
///
/// # Errors
/// As [`evaluate_3c2e_deriv_triple_batch`], plus a backend mismatch on `resident`.
pub fn evaluate_3c2e_deriv_triple_batch_resident(
    backend: &ResolvedBackend,
    family: ThreeC2eDerivFamily,
    resident: &ResidentBasis,
    triples: &[[u32; 3]],
) -> Result<ThreeC2eDerivBatchOutput, cintxRsError> {
    resident.check_for(family.label(), backend)?;
    let shells = resident.shells();

    // Output offsets in the caller's order, computed before any dispatch so a
    // failure cannot leave a partially-sized buffer behind.
    let mut offsets = Vec::with_capacity(triples.len());
    let mut total = 0_usize;
    for triple in triples {
        for &shell in triple {
            if shell as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("{}:shell-index-out-of-range:{shell}", family.label()),
                });
            }
        }
        offsets.push(total);
        total += 3 * triple
            .iter()
            .map(|&shell| shells[shell as usize].ao_len())
            .product::<usize>();
    }

    let mut output = ThreeC2eDerivBatchOutput {
        values: vec![0.0; total],
        offsets,
        stats: crate::kernels::two_electron::BatchExecutionStats {
            quartets: triples.len(),
            ..Default::default()
        },
    };
    if triples.is_empty() {
        return Ok(output);
    }

    // Group by the caller's `(li, lj, lk)`; unlike the scalar path there is no
    // canonical swap here, because the derivative is taken on a named centre and
    // swapping `i` with `j` would move it.
    let mut grouped: std::collections::BTreeMap<[u8; 3], Vec<usize>> = Default::default();
    for (index, triple) in triples.iter().enumerate() {
        grouped
            .entry([
                shells[triple[0] as usize].l,
                shells[triple[1] as usize].l,
                shells[triple[2] as usize].l,
            ])
            .or_default()
            .push(index);
    }

    let ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int3c2eDeriv,
    );
    let mut groups: Vec<ThreeC2eDerivLaunchGroup> = Vec::new();
    let mut group_of: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut classes: Vec<ThreeC2eDerivPlacement> = Vec::with_capacity(grouped.len());

    for (class, members) in grouped {
        let [li, lj, lk] = class;
        // The 3c2e slot mapping with the family's headroom: ip1 raises the bra
        // `i`, ip2 raises the real aux `k` (which lives in the 2e `ll` slot).
        let shape = match family {
            ThreeC2eDerivFamily::Ip1 => {
                build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize)
            }
            ThreeC2eDerivFamily::Ip2 => {
                build_2e_shape(li as usize, lj as usize, 0, lk as usize + 1)
            }
        };
        if shape.nroots > ceiling {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "{}:nroots={} exceeds device ceiling {ceiling} for l=({li},{lj},{lk})",
                    family.label(),
                    shape.nroots
                ),
            });
        }

        let nroots = shape.nroots as u32;
        let group_index = match group_of.get(&nroots) {
            Some(&index) => index,
            None => {
                groups.push(ThreeC2eDerivLaunchGroup::new(nroots));
                let index = groups.len() - 1;
                group_of.insert(nroots, index);
                index
            }
        };
        let group = &mut groups[group_index];
        let class_index = group.push_class(
            u32::from(li),
            u32::from(lj),
            u32::from(lk),
            &shape,
            // `CINTinit_int3c2e_EnvVars`: pi^3 * 2/sqrt(pi) * the three fac_sp,
            // at the *base* angular momenta — the headroom above is a G-tensor
            // extent, not a normalization.
            (PI * PI * PI) * 2.0 / SQRTPI
                * common_fac_sp(li)
                * common_fac_sp(lj)
                * common_fac_sp(lk),
        );

        // Three components per contraction block.
        let cart_block = 3 * ncart(li) * ncart(lj) * ncart(lk);
        group.triples.reserve(members.len() * 5);
        let mut cart_offsets = Vec::with_capacity(members.len());
        for &index in &members {
            let triple = triples[index];
            let nctr_product: usize = triple
                .iter()
                .map(|&shell| shells[shell as usize].nctr as usize)
                .product();
            cart_offsets.push(group.out_len);
            group.triples.extend_from_slice(&[
                triple[0],
                triple[1],
                triple[2],
                group.out_len as u32,
                class_index,
            ]);
            group.out_len += nctr_product * cart_block;
        }

        classes.push(ThreeC2eDerivPlacement {
            li,
            lj,
            lk,
            group: group_index,
            members,
            cart_offsets,
        });
    }

    let dispatch_start = std::time::Instant::now();
    let carts = dispatch_3c2e_deriv_batches(backend, resident.handles(), &groups, family);
    output.stats.dispatch_ns = dispatch_start.elapsed().as_nanos() as u64;

    output.stats.basis_upload_bytes = if resident.take_first_use() {
        resident.upload_bytes()
    } else {
        0
    };
    output.stats.kernel_launch_count = groups.len();
    output.stats.launch_classes = classes.len();
    output.stats.readback_count = groups.len();
    output.stats.max_g_slab_bytes = groups
        .iter()
        .map(|group| {
            // Two slabs per slot: `g` and `g1`.
            2 * three_c2e_slab_stride(3 * group.max_g_size) * std::mem::size_of::<f64>()
        })
        .max()
        .unwrap_or(0);
    output.stats.transfer_bytes = output.stats.basis_upload_bytes
        + groups
            .iter()
            .map(ThreeC2eDerivLaunchGroup::upload_bytes)
            .sum::<usize>();

    let transform_start = std::time::Instant::now();
    // Task 36-T1: one output block and one c2s scratch per worker.
    // Task 36-T2: one job per triple, in the caller's order, each writing a
    // disjoint output block. Each output element is produced by exactly one
    // triple, so the split reorders no summation.
    let carts = &carts;
    let mut placement = vec![(0_usize, 0_usize); triples.len()];
    for (class_index, class) in classes.iter().enumerate() {
        for (slot, &index) in class.members.iter().enumerate() {
            placement[index] = (class_index, slot);
        }
    }
    let lens: Vec<usize> = triples
        .iter()
        .map(|triple| {
            3 * triple
                .iter()
                .map(|&shell| shells[shell as usize].ao_len())
                .product::<usize>()
        })
        .collect();
    let jobs: Vec<(usize, &mut [f64])> =
        crate::transform::host_batch::split_output_blocks(&mut output.values, &lens)
            .into_iter()
            .enumerate()
            .collect();

    let states = crate::transform::host_batch::for_each_block(
        jobs,
        || {
            (
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                crate::transform::profile::HostTransformProfile::new(),
            )
        },
        |(sph, c2s_scratch, profile), (index, block)| {
            let (class_index, slot) = placement[index];
            let class = &classes[class_index];
            let (li, lj, lk) = (class.li, class.lj, class.lk);
            let (nci, ncj, nck) = (ncart(li), ncart(lj), ncart(lk));
            let (nsi, nsj, nsk) = (nsph(li), nsph(lj), nsph(lk));
            let block_len = nci * ncj * nck;
            let total_len = 3 * block_len;

            profile.start();
            sph.clear();
            sph.resize(nsk * nsj * nsi, 0.0);
            profile.charge_alloc();

            let cart = &carts[class.group];
            let triple = triples[index];
            let (nctr_i, nctr_j, nctr_k) = (
                shells[triple[0] as usize].nctr as usize,
                shells[triple[1] as usize].nctr as usize,
                shells[triple[2] as usize].nctr as usize,
            );
            let (di, dj, dk) = (nctr_i * nsi, nctr_j * nsj, nctr_k * nsk);
            let sph_block = di * dj * dk;
            let src_base = class.cart_offsets[slot];

            // Component-leading, exactly as the per-triple path writes it.
            for comp in 0..3usize {
                let dst_comp = comp * sph_block;
                for ci in 0..nctr_i {
                    for cj in 0..nctr_j {
                        for ck in 0..nctr_k {
                            let base = src_base
                                + ((ci * nctr_j + cj) * nctr_k + ck) * total_len
                                + comp * block_len;
                            cart_to_sph_3c2e_into(
                                &cart[base..base + block_len],
                                li,
                                lj,
                                lk,
                                sph,
                                c2s_scratch,
                            );
                            profile.charge_transform();
                            for mk in 0..nsk {
                                let kidx = ck * nsk + mk;
                                for mj in 0..nsj {
                                    let jidx = cj * nsj + mj;
                                    for mi in 0..nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * mk);
                                        block[dst_comp + iidx + di * (jidx + dj * kidx)] = sph[src];
                                    }
                                }
                            }
                            profile.charge_scatter();
                        }
                    }
                }
            }
            profile.pause();
        },
    );

    let mut profile = crate::transform::profile::HostTransformProfile::new();
    for (_, _, worker) in &states {
        profile.merge(worker);
    }
    output.stats.host_transform_ns = transform_start.elapsed().as_nanos() as u64;
    profile.store_into(&mut output.stats);

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
//  int3c2e_ip2 device kernel — `#[cube(launch)]`, generic over `F: Float`
// ─────────────────────────────────────────────────────────────────────────────

/// int3c2e_ip2 (∇ on the auxiliary `k` center, 3 components) for one shell triple,
/// on-device. Single work item (`UNIT_POS == 0`).
///
/// Faithful inline port of the host pipeline `fill_g_tensor_2e` (kbase=false only)
/// → `nabla1l_2e` → the `gout_ipn(Nabla1Center::L)` contraction, applying the 3c2e
/// Pitfall-2 slot mapping: cintx maps the real auxiliary `k` into the 2e `ll` slot
/// (the 2e `lk` slot is a phantom s-function). The ip2 derivative is therefore taken
/// on the `ll` slot via the `G2E_D_L` recurrence — NOT `nabla1k_2e`, which would
/// touch the phantom slot (RESEARCH Pitfall 2).
///
/// Headroom: `build_2e_shape(li, lj, 0, lk + 1)` — bra `i` is NOT raised; the real
/// aux `k` (the `ll` slot) is raised to `lk+1` so `nabla1l_2e` can read index `lk+1`.
/// `kbase` is ALWAYS false for this mapping (`0 > lk+1` is never true), so only the
/// kbase==false HRR branches are reachable.
///
/// Strides (`di,dk,dl,dj,g_size`) and `nmax`/`mmax` are computed host-side from
/// `build_2e_shape(li, lj, 0, lk+1)` and passed as runtime `u32`. `ibase` is a
/// runtime `u32` 0/1. `#[comptime] nroots` selects `rys_root{1..5}`. The nabla
/// exponent is `ak` (the real-k exponent).
///
/// Output `cart_out` (size `3*nci*ncj*nck`, component-leading `[3][nk][nj][ni]`,
/// i fastest within each component) is zeroed in-kernel and accumulated.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn center_3c2e_ip2_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    triples: &Array<u32>,
    class_shape: &Array<u32>,
    class_factor: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    n_triples: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — identical to the scalar 3c2e kernel, and to
    // `two_electron.rs` for why this is arithmetic on comptime-folded flags
    // rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Read and written entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;

        // `g` and `g1` share a slot index and a stride, so one base serves both.
        // The contraction below reads them at the *same* relative offsets, which
        // is only sound because their layouts are identical.
        let gbase = slot * g_stride;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
        #[allow(clippy::manual_div_ceil)]
        let chunk = (n_triples + n_slots - 1u32) / n_slots;
        let qi_start = slot * (chunk * punit + coop);
        let mut qi_stop = (qi_start + chunk) * punit + n_triples * coop;
        if qi_stop > n_triples {
            qi_stop = n_triples;
        }
        let qi_step = n_slots * coop + punit;

        let mut qi = qi_start;
        while qi < qi_stop {
            let trow = qi * 5u32;
            let si = triples[trow as usize];
            let sj = triples[(trow + 1u32) as usize];
            let sk = triples[(trow + 2u32) as usize];
            let out_off = triples[(trow + 3u32) as usize];

            // ── Per-class shape (Task 35-D) ───────────────────────────────
            //
            // `nroots` is this kernel's only shape-bearing comptime parameter,
            // so one dispatch carries every `(li,lj,lk)` class of the same Rys
            // order. The G slabs are sized to the widest class in the dispatch
            // and each class indexes only what it owns.
            let cls = triples[(trow + 4u32) as usize];
            let srow = cls * comptime!(THREE_C2E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];
            let lk = class_shape[(srow + 2u32) as usize];
            let di = class_shape[(srow + 3u32) as usize];
            let dk = class_shape[(srow + 4u32) as usize];
            let dl = class_shape[(srow + 5u32) as usize];
            let dj = class_shape[(srow + 6u32) as usize];
            let g_size = class_shape[(srow + 7u32) as usize];
            let nmax = class_shape[(srow + 8u32) as usize];
            let mmax = class_shape[(srow + 9u32) as usize];
            let ibase = class_shape[(srow + 10u32) as usize];
            let common_factor = class_factor[cls as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];
            let mk = sk * 4u32;
            let eoff_k = shell_meta[mk as usize];
            let coff_k = shell_meta[(mk + 1u32) as usize];
            let nprim_k = shell_meta[(mk + 2u32) as usize];
            let nctr_k = shell_meta[(mk + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];
            let ck3 = sk * 3u32;
            let rkx = centers[ck3 as usize];
            let rky = centers[(ck3 + 1u32) as usize];
            let rkz = centers[(ck3 + 2u32) as usize];

            let total_g = 3u32 * g_size;
            let gy_off = gbase + g_size;
            let gz_off = gbase + 2u32 * g_size;

            // ip2 headroom: bra `i` NOT raised (li_e = li); the real aux k (`ll` slot)
            // raised to lk+1 for the ∇_k headroom. The 2e `lk` slot is phantom (base 0).
            let li_e = li;
            let ll = lk + 1u32; // real k mapped into the 2e ll-slot, raised by +1
            let lk2e = 0u32; // phantom 2e lk slot

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
            let block_len = nci * ncj * nck;
            let total_len = 3u32 * block_len; // per-(ci,cj,ck) component-leading block
            let out_len = nctr_i * nctr_j * nctr_k * total_len;

            let mut oi = out_off;
            while oi < out_off + out_len {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut kp = 0u32;
            while kp < nprim_k {
                let ak = exps[(eoff_k + kp) as usize];
                let mut jp = 0u32;
                while jp < nprim_j {
                    let aj = exps[(eoff_j + jp) as usize];
                    let mut ip = 0u32;
                    while ip < nprim_i {
                        let ai = exps[(eoff_i + ip) as usize];

                        // ── Inlined pdata_ij (bra) and pdata_kl (phantom-real_k) ──
                        let zeta_ab = ai + aj;
                        let rij_dx = rix - rjx;
                        let rij_dy = riy - rjy;
                        let rij_dz = riz - rjz;
                        let rr_ij = rij_dx * rij_dx + rij_dy * rij_dy + rij_dz * rij_dz;
                        let pdata_ij_fac = F::exp(-ai * aj / zeta_ab * rr_ij);
                        // pdata_kl: zeta=ak, center=rk, rr=0 → fac=1.
                        let pdata_kl_fac = F::new(1.0_f32);
                        let fac_env = common_factor * pdata_ij_fac * pdata_kl_fac;

                        // ── fill_g_tensor_2e math (ai,aj | 0,ak at ri,rj | rk,rk) ──
                        let aij = zeta_ab;
                        let akl = ak; // 2e lk slot exp=0, ll slot exp=ak → akl = 0 + ak.
                        let rij_x = (ai * rix + aj * rjx) / aij;
                        let rij_y = (ai * riy + aj * rjy) / aij;
                        let rij_z = (ai * riz + aj * rjz) / aij;
                        // rkl = (0*rk + ak*rk)/ak = rk.
                        let rkl_x = rkx;
                        let rkl_y = rky;
                        let rkl_z = rkz;

                        let xij_kl = rij_x - rkl_x;
                        let yij_kl = rij_y - rkl_y;
                        let zij_kl = rij_z - rkl_z;
                        let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

                        let a1 = aij * akl;
                        let a0 = a1 / (aij + akl);
                        let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * fac_env;
                        let x_rys = a0 * rr;

                        let mut rx_rij_x = rjx;
                        let mut rx_rij_y = rjy;
                        let mut rx_rij_z = rjz;
                        let mut rirj_x = rjx - rix;
                        let mut rirj_y = rjy - riy;
                        let mut rirj_z = rjz - riz;
                        if ibase == 1u32 {
                            rx_rij_x = rix;
                            rx_rij_y = riy;
                            rx_rij_z = riz;
                            rirj_x = rix - rjx;
                            rirj_y = riy - rjy;
                            rirj_z = riz - rjz;
                        }
                        let rijrx_x = rij_x - rx_rij_x;
                        let rijrx_y = rij_y - rx_rij_y;
                        let rijrx_z = rij_z - rx_rij_z;
                        // rklrx = rkl - rl(=rk) = 0; rkrl = rl - rk = 0.

                        // Rys roots/weights.
                        if comptime!(nroots == 1u32) {
                            rys_root1::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 2u32) {
                            rys_root2::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 3u32) {
                            rys_root3::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 4u32) {
                            rys_root4::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        } else {
                            rys_root5::<F>(x_rys, &mut urys, &mut wrys, pie4);
                        }

                        // Zero g.
                        let mut gi = gbase;
                        while gi < gbase + total_g {
                            g[gi as usize] = F::new(0.0_f32);
                            gi += 1u32;
                        }

                        // g2d strides for VRR: g2d_ijmax = ibase? di : dj;
                        //                      g2d_klmax = kbase? dk : dl  (kbase=false → dl).
                        let mut g2d_ijmax = dj;
                        if ibase == 1u32 {
                            g2d_ijmax = di;
                        }
                        let g2d_klmax = dl;

                        // Base + VRR per axis.
                        let mut irys = 0u32;
                        while irys < nrys {
                            g[(gbase + irys) as usize] = F::new(1.0_f32);
                            g[(gy_off + irys) as usize] = F::new(1.0_f32);
                            g[(gz_off + irys) as usize] = wrys[irys as usize] * fac1;
                            irys += 1u32;
                        }

                        let mut irys2 = 0u32;
                        while irys2 < nrys {
                            let u2 = a0 * urys[irys2 as usize];
                            let tmp4 = F::new(0.5_f32) / (u2 * (aij + akl) + a1);
                            let tmp5 = u2 * tmp4;
                            let tmp1 = F::new(2.0_f32) * tmp5;
                            let tmp2 = tmp1 * akl;
                            let tmp3 = tmp1 * aij;
                            let b00 = tmp5;
                            let b10 = tmp5 + tmp4 * akl;
                            let b01 = tmp5 + tmp4 * aij;

                            let mut axis = 0u32;
                            while axis < 3u32 {
                                let base = gbase + axis * g_size;
                                let mut d = xij_kl;
                                let mut rijrx = rijrx_x;
                                if axis == 1u32 {
                                    d = yij_kl;
                                    rijrx = rijrx_y;
                                }
                                if axis == 2u32 {
                                    d = zij_kl;
                                    rijrx = rijrx_z;
                                }
                                let c00 = rijrx - tmp2 * d;
                                let c0p = tmp3 * d;

                                let dn = g2d_ijmax;
                                let dm = g2d_klmax;
                                let root = base + irys2;

                                // n-ladder (nmax).
                                if nmax >= 1u32 {
                                    let mut s0 = g[root as usize];
                                    let mut s1 = c00 * s0;
                                    g[(root + dn) as usize] = s1;
                                    let mut n = 1u32;
                                    while n < nmax {
                                        let s2 = c00 * s1 + F::cast_from(n) * b10 * s0;
                                        g[(root + (n + 1u32) * dn) as usize] = s2;
                                        s0 = s1;
                                        s1 = s2;
                                        n += 1u32;
                                    }
                                }

                                // m-ladder (mmax).
                                if mmax >= 1u32 {
                                    let mut s0 = g[root as usize];
                                    let mut s1 = c0p * s0;
                                    g[(root + dm) as usize] = s1;
                                    let mut m = 1u32;
                                    while m < mmax {
                                        let s2 = c0p * s1 + F::cast_from(m) * b01 * s0;
                                        g[(root + (m + 1u32) * dm) as usize] = s2;
                                        s0 = s1;
                                        s1 = s2;
                                        m += 1u32;
                                    }

                                    if nmax >= 1u32 {
                                        let mut s0n = g[(root + dn) as usize];
                                        let mut s1n = c0p * s0n + b00 * g[root as usize];
                                        g[(root + dn + dm) as usize] = s1n;
                                        let mut m2 = 1u32;
                                        while m2 < mmax {
                                            let s2n = c0p * s1n
                                                + F::cast_from(m2) * b01 * s0n
                                                + b00 * g[(root + m2 * dm) as usize];
                                            g[(root + dn + (m2 + 1u32) * dm) as usize] = s2n;
                                            s0n = s1n;
                                            s1n = s2n;
                                            m2 += 1u32;
                                        }
                                    }
                                }

                                if nmax >= 1u32 {
                                    let mut m3 = 1u32;
                                    while m3 <= mmax {
                                        let off = m3 * dm;
                                        let j = off + root;
                                        let mut s0 = g[j as usize];
                                        let mut s1 = g[(j + dn) as usize];
                                        let mut n2 = 1u32;
                                        while n2 < nmax {
                                            let s2 = c00 * s1
                                                + F::cast_from(n2) * b10 * s0
                                                + F::cast_from(m3)
                                                    * b00
                                                    * g[(j + n2 * dn - dm) as usize];
                                            g[(j + (n2 + 1u32) * dn) as usize] = s2;
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

                        // ── HRR transfer (kbase==false): ibase selects branch. ─────
                        if ibase == 0u32 {
                            // hrr_lj2d_4d (li-then-k transfer). li_e=li (no bra raise),
                            // lk2e=0 (phantom) → k-transfer is a no-op; ll headroom from VRR.
                            if li_e != 0u32 || lk2e != 0u32 {
                                let mut axis = 0u32;
                                while axis < 3u32 {
                                    let off = gbase + axis * g_size;
                                    let mut rx = rirj_x;
                                    if axis == 1u32 {
                                        rx = rirj_y;
                                    }
                                    if axis == 2u32 {
                                        rx = rirj_z;
                                    }

                                    // i-transfer (up to li_e = li).
                                    let mut i = 1u32;
                                    while i <= li_e {
                                        let jmax = nmax - i;
                                        let mut jjj = 0u32;
                                        while jjj <= jmax {
                                            let mut l = 0u32;
                                            while l <= mmax {
                                                let ptr = jjj * dj + l * dl + i * di;
                                                let mut r = 0u32;
                                                while r < nrys {
                                                    let idx = ptr + r;
                                                    g[(off + idx) as usize] = rx
                                                        * g[(off + idx - di) as usize]
                                                        + g[(off + idx - di + dj) as usize];
                                                    r += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            jjj += 1u32;
                                        }
                                        i += 1u32;
                                    }

                                    // k-transfer (rkrl = 0; lk2e=0 → no-op).
                                    let rxk = F::new(0.0_f32);
                                    let mut jjj2 = 0u32;
                                    while jjj2 <= lj {
                                        let mut k = 1u32;
                                        while k <= lk2e {
                                            let lmax = mmax - k;
                                            let mut l = 0u32;
                                            while l <= lmax {
                                                let ptr = jjj2 * dj + l * dl + k * dk;
                                                let mut n = 0u32;
                                                while n < dk {
                                                    let idx = ptr + n;
                                                    g[(off + idx) as usize] = rxk
                                                        * g[(off + idx - dk) as usize]
                                                        + g[(off + idx - dk + dl) as usize];
                                                    n += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            k += 1u32;
                                        }
                                        jjj2 += 1u32;
                                    }

                                    axis += 1u32;
                                }
                            }
                        } else {
                            // hrr_il2d_4d (k-then-j transfer).
                            if lj != 0u32 || lk2e != 0u32 {
                                let mut axis = 0u32;
                                while axis < 3u32 {
                                    let off = gbase + axis * g_size;
                                    // k-transfer (rkrl = 0; lk2e=0 → no-op).
                                    let rxk = F::new(0.0_f32);
                                    let mut k = 1u32;
                                    while k <= lk2e {
                                        let lmax = mmax - k;
                                        let mut l = 0u32;
                                        while l <= lmax {
                                            let mut i = 0u32;
                                            while i <= nmax {
                                                let ptr = l * dl + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nrys {
                                                    let idx = ptr + r;
                                                    g[(off + idx) as usize] = rxk
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

                                    // j-transfer (rirj).
                                    let mut rx = rirj_x;
                                    if axis == 1u32 {
                                        rx = rirj_y;
                                    }
                                    if axis == 2u32 {
                                        rx = rirj_z;
                                    }
                                    let mut jjj = 1u32;
                                    while jjj <= lj {
                                        let mut l = 0u32;
                                        while l <= ll {
                                            let mut k2 = 0u32;
                                            while k2 <= lk2e {
                                                let ptr = jjj * dj + l * dl + k2 * dk;
                                                let imax = nmax - jjj;
                                                let mut i = 0u32;
                                                while i <= imax {
                                                    let base2 = ptr + i * di;
                                                    let mut r = 0u32;
                                                    while r < nrys {
                                                        let idx = base2 + r;
                                                        g[(off + idx) as usize] = rx
                                                            * g[(off + idx - dj) as usize]
                                                            + g[(off + idx - dj + di) as usize];
                                                        r += 1u32;
                                                    }
                                                    i += 1u32;
                                                }
                                                k2 += 1u32;
                                            }
                                            l += 1u32;
                                        }
                                        jjj += 1u32;
                                    }

                                    axis += 1u32;
                                }
                            }
                        }

                        // ── nabla1l_2e → g1 (G2E_D_L on the ll slot; real-k derivative).
                        // ll has lk+1 headroom; write g1 at base ll=lk reading ±dl.
                        let mut g1i = gbase;
                        while g1i < gbase + total_g {
                            g1[g1i as usize] = F::new(0.0_f32);
                            g1i += 1u32;
                        }
                        let ak2 = F::new(-2.0_f32) * ak;
                        let mut axisn = 0u32;
                        while axisn < 3u32 {
                            let off = gbase + axisn * g_size;
                            let mut jn = 0u32;
                            while jn <= lj {
                                // l=0 block: all k(phantom), i.
                                let mut kn0 = 0u32;
                                while kn0 <= lk2e {
                                    let base = dj * jn + dk * kn0;
                                    let mut i0 = 0u32;
                                    while i0 <= li {
                                        let ptr = base + di * i0;
                                        let mut n = ptr;
                                        while n < ptr + nrys {
                                            g1[(off + n) as usize] =
                                                ak2 * g[(off + n + dl) as usize];
                                            n += 1u32;
                                        }
                                        i0 += 1u32;
                                    }
                                    kn0 += 1u32;
                                }
                                // l>=1 (base ll = lk).
                                let mut ln = 1u32;
                                while ln <= lk {
                                    let mut kn = 0u32;
                                    while kn <= lk2e {
                                        let base = dj * jn + dl * ln + dk * kn;
                                        let mut i = 0u32;
                                        while i <= li {
                                            let ptr = base + di * i;
                                            let mut n2 = ptr;
                                            while n2 < ptr + nrys {
                                                g1[(off + n2) as usize] = F::cast_from(ln)
                                                    * g[(off + n2 - dl) as usize]
                                                    + ak2 * g[(off + n2 + dl) as usize];
                                                n2 += 1u32;
                                            }
                                            i += 1u32;
                                        }
                                        kn += 1u32;
                                    }
                                    ln += 1u32;
                                }
                                jn += 1u32;
                            }
                            axisn += 1u32;
                        }

                        // ── gout_ipn(Nabla1Center::L) contraction. n walks
                        // [ll=real_k][lk2e=phantom][j][i] i-fastest; transpose to
                        // component-leading. s[0]=g1x*g0y*g0z, etc. ─────────────────
                        let gx_off = gbase;

                        let mut cci = 0u32;
                        while cci < nctr_i {
                            let coeff_i_val = coeffs[(coff_i + ip * nctr_i + cci) as usize];
                            let mut ccj = 0u32;
                            while ccj < nctr_j {
                                let coeff_j_val = coeffs[(coff_j + jp * nctr_j + ccj) as usize];
                                let mut cck = 0u32;
                                while cck < nctr_k {
                                    let coeff_k_val = coeffs[(coff_k + kp * nctr_k + cck) as usize];
                                    let weight = coeff_i_val * coeff_j_val * coeff_k_val;
                                    let ctr_base =
                                        out_off + ((cci * nctr_j + ccj) * nctr_k + cck) * total_len;

                                    let mut n = 0u32;
                                    // l = real k (ll slot), BASE lk Cartesian comps.
                                    let mut la = 0u32;
                                    while la <= lk {
                                        let lx = lk - la;
                                        let lk_minus_lx = lk - lx;
                                        let mut lb = 0u32;
                                        while lb <= lk_minus_lx {
                                            let ly = lk_minus_lx - lb;
                                            let lz = lk - lx - ly;

                                            // k = phantom (lk2e, size 1): only (0,0,0).
                                            let kx = 0u32;
                                            let ky = 0u32;
                                            let kz = 0u32;

                                            // j.
                                            let mut ja = 0u32;
                                            while ja <= lj {
                                                let jx = lj - ja;
                                                let lj_minus_jx = lj - jx;
                                                let mut jb = 0u32;
                                                while jb <= lj_minus_jx {
                                                    let jy = lj_minus_jx - jb;
                                                    let jz = lj - jx - jy;

                                                    // i (base li).
                                                    let mut ia = 0u32;
                                                    while ia <= li {
                                                        let ix = li - ia;
                                                        let li_minus_ix = li - ix;
                                                        let mut ib = 0u32;
                                                        while ib <= li_minus_ix {
                                                            let iy = li_minus_ix - ib;
                                                            let iz = li - ix - iy;

                                                            let ix_base = ix * di
                                                                + kx * dk
                                                                + lx * dl
                                                                + jx * dj;
                                                            let iy_base = iy * di
                                                                + ky * dk
                                                                + ly * dl
                                                                + jy * dj;
                                                            let iz_base = iz * di
                                                                + kz * dk
                                                                + lz * dl
                                                                + jz * dj;

                                                            let mut s0 = F::new(0.0_f32);
                                                            let mut s1 = F::new(0.0_f32);
                                                            let mut s2 = F::new(0.0_f32);
                                                            let mut r = 0u32;
                                                            while r < nrys {
                                                                let g1x = g1[(gx_off + ix_base + r)
                                                                    as usize];
                                                                let g0x = g[(gx_off + ix_base + r)
                                                                    as usize];
                                                                let g1y = g1[(gy_off + iy_base + r)
                                                                    as usize];
                                                                let g0y = g[(gy_off + iy_base + r)
                                                                    as usize];
                                                                let g1z = g1[(gz_off + iz_base + r)
                                                                    as usize];
                                                                let g0z = g[(gz_off + iz_base + r)
                                                                    as usize];
                                                                s0 += g1x * g0y * g0z;
                                                                s1 += g0x * g1y * g0z;
                                                                s2 += g0x * g0y * g1z;
                                                                r += 1u32;
                                                            }

                                                            cart_out[(ctr_base
                                                                + 0u32 * block_len
                                                                + n)
                                                                as usize] += weight * s0;
                                                            cart_out[(ctr_base
                                                                + 1u32 * block_len
                                                                + n)
                                                                as usize] += weight * s1;
                                                            cart_out[(ctr_base
                                                                + 2u32 * block_len
                                                                + n)
                                                                as usize] += weight * s2;

                                                            n += 1u32;
                                                            ib += 1u32;
                                                        }
                                                        ia += 1u32;
                                                    }

                                                    jb += 1u32;
                                                }
                                                ja += 1u32;
                                            }

                                            lb += 1u32;
                                        }
                                        la += 1u32;
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

            qi += qi_step;
        }
    }
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
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Phase 27 (27-04, D-06): spinor gradient now evaluates via the Plan-02 derivative
    // wrapper `cart_to_spinor_sf_derivative_3c2e` (SPHERICAL aux-k). The early reject is
    // removed; the Spinor arm in the staging match below folds the device cart blocks.

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

    // The whole per-triple numeric core (`fill_g_tensor_2e` → `gout_ip1` →
    // component-leading transpose) now runs on the device kernel
    // (`center_3c2e_ip1_kernel`). The strides/nroots/ibase come from `grad_shape`
    // (`build_2e_shape(li+1, lj, 0, lk)`); `kbase` is always false for this mapping
    // so only the kbase==false HRR branches are emitted. The device produces the
    // per-(ci,cj,ck) component-leading `cart_blocks` the staging tail expects.
    let _ = two_e_shape_as_f12(&grad_shape); // (host-side reference bridge; unused on the device path)

    // Flatten primitive data the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();

    // One triple through the batched kernel — a one-triple launch group, so the
    // per-tuple compatibility API and a wide batch execute the *same* code
    // (Task 35-D). The five backend arms this replaced were identical apart from
    // their runtime type.
    let cart_blocks: Vec<f64> = run_3c2e_deriv_single(
        backend,
        ThreeC2eDerivFamily::Ip1,
        li as u32,
        lj as u32,
        lk as u32,
        &grad_shape,
        common_factor,
        ri,
        rj,
        rk,
        [n_prim_i as u32, n_prim_j as u32, n_prim_k as u32],
        [n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32],
        [&exps_i, &exps_j, &exps_k],
        [&coeff_i, &coeff_j, &coeff_k],
    );

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
                            let base =
                                ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len + comp * block_len;
                            let sph =
                                cart_to_sph_3c2e(&cart_blocks[base..base + block_len], li, lj, lk);
                            for mk in 0..nsk {
                                let kidx = ck * nsk + mk;
                                for mj in 0..nsj {
                                    let jidx = cj * nsj + mj;
                                    for mi in 0..nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * mk);
                                        let dst =
                                            staging_comp_base + iidx + di * (jidx + dj * kidx);
                                        staging[dst] = F::from_f64_lossy(sph[src]);
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
                            let base =
                                ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len + comp * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for kc in 0..nck {
                                let kidx = ck * nck + kc;
                                for jc in 0..ncj {
                                    let jidx = cj * ncj + jc;
                                    for ic in 0..nci {
                                        let iidx = ci * nci + ic;
                                        let src = ic + nci * (jc + ncj * kc);
                                        let dst =
                                            staging_comp_base + iidx + di * (jidx + dj * kidx);
                                        staging[dst] = F::from_f64_lossy(block[src]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            // 27-04 (D-06): fold the device cart blocks to spinor via the Plan-02
            // derivative wrapper. Aux-k is SPHERICAL nsph(lk); only bra i / ket j are
            // spinor-sized (4l+2). The wrapper owns the KET→BRA transpose and the
            // per-(comp,k) cart→sph(k) + sf_2d fold (no transpose lives here).
            //
            // W5-02: the wrapper consumes the device cart_blocks as
            // `[((ci*n_ctr_j+cj)*n_ctr_k+ck)][comp][k][j][i]` and composes
            // contraction-major on all three axes, so a general-contracted aux-k is
            // handled rather than rejected.
            cart_to_spinor_sf_derivative_3c2e::<F>(
                staging,
                &cart_blocks,
                3,
                li,
                shell_i.kappa,
                lj,
                shell_j.kappa,
                lk,
                n_ctr_i,
                n_ctr_j,
                n_ctr_k,
            )?;
        }
    }

    // Per-symbol nonzero sentinel (precision-aware; matches the scalar path).
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// `int3c2e_ip2` gradient launch — the `∇` auxiliary-`k`-center derivative of the
/// three-center two-electron Coulomb integral (DRV1-05).
///
/// Mirrors [`launch_center_3c2e_ip1`], but takes the derivative on the auxiliary `k`
/// center instead of the bra `i`. RESEARCH Pitfall 2: cintx's 3c2e g-tensor maps the
/// real aux `k` into the 2e `ll` slot (the 2e `lk` slot is a phantom s-function), so
/// the ip2 derivative must nabla the `ll` slot via `nabla1l_2e` — `nabla1k_2e` would
/// touch the phantom slot.
///
///   - 2e "ij side"  ← real `(i, j)` (the bra; NOT raised for ip2)
///   - 2e `ll` slot   ← real `k` (raised to `lk+1` for the `∇_k` headroom)
///   - 2e `lk` slot   ← phantom s-function (`lk_ceil = 0`, exponent `0`)
///
/// Builds the plain Coulomb G-tensor through the SHARED 2e recurrence with the
/// `ll = lk+1` headroom (`build_2e_shape(li, lj, 0, lk+1)`), applies `nabla1l_2e`
/// with exponent `ak`, and emits 3-component component-leading `[3, nk, nj, ni]`
/// F-order (same convention as `int3c2e_ip1`).
///
/// Guards (fail-closed):
///   - `Representation::Spinor` → `UnsupportedApi` (D-06).
///   - `grad_shape.nroots > 5` → `UnsupportedApi` (D-13): the `lk→lk+1` raise can
///     push high-l triples past the rys_root1..5 ceiling; reject BEFORE any rys
///     dispatch.
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_ip2<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Phase 27 (27-04, D-06): spinor gradient now evaluates via the same Plan-02
    // derivative wrapper `cart_to_spinor_sf_derivative_3c2e` (SPHERICAL aux-k) — ip2
    // shares ip1's buffer SHAPE, differing only in which center is differentiated
    // (the device kernel already picked nabla1l). The early reject is removed.

    // 3c2e kl mapping into the 2e shape (Pitfall 2): real k → 2e `ll` slot, phantom
    // 2e `lk` slot = 0. For ip2 the bra `i` is NOT raised; the real aux k (`ll` slot)
    // is raised to `lk+1` so `nabla1l_2e` can read index lk+1.
    let grad_shape = build_2e_shape(li as usize, lj as usize, 0, lk as usize + 1);

    // D-13: the elevated ll can push nroots past the rys_root1..5 ceiling. Reject
    // fail-closed BEFORE any rys_roots_host call (which would otherwise panic).
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

    // The whole per-triple numeric core (`fill_g_tensor_2e` → `nabla1l_2e` →
    // `gout_ipn(L)` → component-leading transpose) runs on the device kernel
    // (`center_3c2e_ip2_kernel`). The strides/nroots/ibase come from `grad_shape`
    // (`build_2e_shape(li, lj, 0, lk+1)`); `kbase` is always false for this mapping
    // so only the kbase==false HRR branches are emitted.
    let _ = two_e_shape_as_f12(&grad_shape); // (host-side reference bridge; unused on the device path)

    // Flatten primitive data the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();

    // One triple through the batched kernel — a one-triple launch group, so the
    // per-tuple compatibility API and a wide batch execute the *same* code
    // (Task 35-D). The five backend arms this replaced were identical apart from
    // their runtime type.
    let cart_blocks: Vec<f64> = run_3c2e_deriv_single(
        backend,
        ThreeC2eDerivFamily::Ip2,
        li as u32,
        lj as u32,
        lk as u32,
        &grad_shape,
        common_factor,
        ri,
        rj,
        rk,
        [n_prim_i as u32, n_prim_j as u32, n_prim_k as u32],
        [n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32],
        [&exps_i, &exps_j, &exps_k],
        [&coeff_i, &coeff_j, &coeff_k],
    );

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
                            let base =
                                ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len + comp * block_len;
                            let sph =
                                cart_to_sph_3c2e(&cart_blocks[base..base + block_len], li, lj, lk);
                            for mk in 0..nsk {
                                let kidx = ck * nsk + mk;
                                for mj in 0..nsj {
                                    let jidx = cj * nsj + mj;
                                    for mi in 0..nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * mk);
                                        let dst =
                                            staging_comp_base + iidx + di * (jidx + dj * kidx);
                                        staging[dst] = F::from_f64_lossy(sph[src]);
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
                            let base =
                                ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len + comp * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for kc in 0..nck {
                                let kidx = ck * nck + kc;
                                for jc in 0..ncj {
                                    let jidx = cj * ncj + jc;
                                    for ic in 0..nci {
                                        let iidx = ci * nci + ic;
                                        let src = ic + nci * (jc + ncj * kc);
                                        let dst =
                                            staging_comp_base + iidx + di * (jidx + dj * kidx);
                                        staging[dst] = F::from_f64_lossy(block[src]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            // 27-04 (D-06): ip2 spinor gradient folds the device cart blocks through
            // the SAME derivative wrapper as ip1 (identical buffer shape per the spike;
            // the device kernel already chose the aux/ket-center gradient). Aux-k stays
            // SPHERICAL nsph(lk); only bra i / ket j are spinor-sized (4l+2). No
            // transpose lives here (D-06: it is owned inside the wrapper).
            // W5-02: aux-k carries its own contraction axis inside the wrapper, so a
            // general-contracted aux-k is composed rather than rejected.
            cart_to_spinor_sf_derivative_3c2e::<F>(
                staging,
                &cart_blocks,
                3,
                li,
                shell_i.kappa,
                lj,
                shell_j.kappa,
                lk,
                n_ctr_i,
                n_ctr_j,
                n_ctr_k,
            )?;
        }
    }

    // Per-symbol nonzero sentinel (precision-aware; matches the scalar path).
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// Which center the `int3c2e_ipip*` double-nabla is applied to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HessKind {
    /// `int3c2e_ipip1` — ∇² on the bra center i (`gout_ipip1`, `nabla1i_2e`, li+2).
    Ipip1,
    /// `int3c2e_ipip2` — ∇² on the auxiliary k center (mapped to the 2e `ll` slot
    /// via `gout_ipip2_l`, `nabla1l_2e`, ll=lk+2 headroom). KET-side (D-09).
    Ipip2,
    /// Mixed derivative on the two real bra centers i and j.
    Ipvip1,
    /// Mixed derivative on bra i and the real auxiliary center in the 2e l slot.
    Ip1ip2,
}

/// Shared HOST launcher for the two multi-center 3c2e rank-9 Hessian families
/// (`int3c2e_ipip1` bra-side, `int3c2e_ipip2` ket-side — HESS-03).
///
/// 3c2e kl mapping (Pitfall 2): the real aux k → 2e `ll` slot, the 2e `lk` slot is
/// a phantom s-function. ipip1 raises bra `i` to `li+2`; ipip2 raises the real aux
/// (`ll` slot) to `lk+2`. The G-tensor is built through the SHARED 2e recurrence
/// (`fill_g_tensor_2e`); the family's verbatim gout (`gout_ipip1` / `gout_ipip2_l`)
/// emits the 9 column-major Hessian components. Emits `[9, nk, nj, ni]` F-order.
///
/// HOST-routed so the `+2` raise can reach nroots 6..12 (FND-02); nroots>12 stays
/// fail-closed. Spinor → `UnsupportedApi` (D-11).
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_hess<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    kind: HessKind,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    const NCOMP: usize = 9;
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: match kind {
                HessKind::Ipip1 => "spinor int3c2e_ipip1 Hessian".to_owned(),
                HessKind::Ipip2 => "spinor int3c2e_ipip2 Hessian".to_owned(),
                HessKind::Ipvip1 => "spinor int3c2e_ipvip1 Hessian".to_owned(),
                HessKind::Ip1ip2 => "spinor int3c2e_ip1ip2 Hessian".to_owned(),
            },
        });
    }

    // ipip1: bra i raised +2; real aux k (ll slot) at base lk.
    // ipip2: bra i at base; real aux k (ll slot) raised +2.
    let hess_shape = match kind {
        HessKind::Ipip1 => build_2e_shape(li as usize + 2, lj as usize, 0, lk as usize),
        HessKind::Ipip2 => build_2e_shape(li as usize, lj as usize, 0, lk as usize + 2),
        HessKind::Ipvip1 => build_2e_shape(li as usize + 1, lj as usize + 1, 0, lk as usize),
        HessKind::Ip1ip2 => build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize + 1),
    };

    if hess_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", hess_shape.nroots),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    // The 2e `lk` slot is a phantom s-function on the same center as the real aux k.
    let rl = rk;
    let rk_phantom = rk;

    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let block_len = nci * ncj * nck;
    let total_len = NCOMP * block_len;

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    let hess_f12_shape = two_e_shape_as_f12(&hess_shape);

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * total_len];

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];

                // Per-primitive Gaussian-overlap prefactors (the `pdata.fac`
                // exp(-mu*r²) terms) — same as the device ip1/ip2 host bridge
                // (host_ip1_cart_blocks): bra pair (i,j) and ket pair (phantom-k, aux-l).
                let pdata_ij =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
                let pdata_kl = compute_pdata_host(
                    0.0,
                    ak,
                    rk_phantom[0],
                    rk_phantom[1],
                    rk_phantom[2],
                    rl[0],
                    rl[1],
                    rl[2],
                    1.0,
                    1.0,
                );
                let fac_env = common_factor * pdata_ij.fac * pdata_kl.fac;

                // 2e G-tensor with the real aux k in the `ll` slot (al = ak) and a
                // phantom s in the `lk` slot (ak_2e = 0). Mirrors the 3c2e ip1/ip2
                // host bridge (host_ip1_cart_blocks in the test module).
                let g = fill_g_tensor_2e(
                    ai,
                    aj,
                    0.0,
                    ak,
                    &ri,
                    &rj,
                    &rk_phantom,
                    &rl,
                    hess_shape,
                    fac_env,
                );

                // Verbatim Hessian gout: ipip1 nabla²_i (bra), ipip2 nabla²_l (ket aux).
                let gout = match kind {
                    HessKind::Ipip1 => gout_ipip1(
                        &g,
                        &hess_f12_shape,
                        li as usize,
                        lj as usize,
                        0,
                        lk as usize,
                        ai,
                    ),
                    HessKind::Ipip2 => gout_ipip2_l(
                        &g,
                        &hess_f12_shape,
                        li as usize,
                        lj as usize,
                        0,
                        lk as usize,
                        ak,
                    ),
                    HessKind::Ipvip1 => gout_ipvip1(
                        &g,
                        &hess_f12_shape,
                        li as usize,
                        lj as usize,
                        0,
                        lk as usize,
                        ai,
                        aj,
                    ),
                    HessKind::Ip1ip2 => gout_ip1ip2_l(
                        &g,
                        &hess_f12_shape,
                        li as usize,
                        lj as usize,
                        0,
                        lk as usize,
                        ai,
                        ak,
                    ),
                };

                for ci in 0..n_ctr_i {
                    let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                        for ck in 0..n_ctr_k {
                            let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                            let weight = coeff_i * coeff_j * coeff_k;
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len;
                            for n in 0..block_len {
                                for comp in 0..NCOMP {
                                    cart_blocks[base + comp * block_len + n] +=
                                        weight * gout[n * NCOMP + comp];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Component-leading `[9, nk, nj, ni]` F-order write.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let sph_block = di * dj * dk;
            for comp in 0..NCOMP {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            let base =
                                ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len + comp * block_len;
                            let sph =
                                cart_to_sph_3c2e(&cart_blocks[base..base + block_len], li, lj, lk);
                            for mk in 0..nsk {
                                let kidx = ck * nsk + mk;
                                for mj in 0..nsj {
                                    let jidx = cj * nsj + mj;
                                    for mi in 0..nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * mk);
                                        let dst =
                                            staging_comp_base + iidx + di * (jidx + dj * kidx);
                                        staging[dst] = F::from_f64_lossy(sph[src]);
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
            for comp in 0..NCOMP {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            let base =
                                ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len + comp * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for kc in 0..nck {
                                let kidx = ck * nck + kc;
                                for jc in 0..ncj {
                                    let jidx = cj * ncj + jc;
                                    for ic in 0..nci {
                                        let iidx = ci * nci + ic;
                                        let src = ic + nci * (jc + ncj * kc);
                                        let dst =
                                            staging_comp_base + iidx + di * (jidx + dj * kidx);
                                        staging[dst] = F::from_f64_lossy(block[src]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => unreachable!("spinor int3c2e Hessian rejected above"),
    }

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// `int3c2e_ipip1` — ∇² on the bra center i (HESS-03). Thin wrapper over
/// [`launch_center_3c2e_hess`] with [`HessKind::Ipip1`].
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_hess1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    launch_center_3c2e_hess::<F>(
        plan,
        shell_i,
        shell_j,
        shell_k,
        li,
        lj,
        lk,
        HessKind::Ipip1,
        staging,
    )
}

/// `int3c2e_ipspsp1` (W4-05) — the 3-centre σ·p gradient, `int3c2e.c:668`,
/// `ng = {2, 1, 0, 0, 3, 4, 1, 3}`. Spinor only, per the manifest-wide σ-family
/// precedent (`int1e_sp`, `int2e_spsp1`, … are all spinor-only rows).
///
/// Its gout is byte-for-byte the `int2e_ipspsp1` gout evaluated with a phantom
/// `l`-shell — identical cascade, `s[]` table and fold (verified against
/// `grad2.c:183`) — so it reuses `f12::gout_ip_sigma(0, …)` rather than duplicating it.
/// Following the file's existing convention the REAL auxiliary `k` sits in the 2e `ll`
/// slot and the `lk` slot holds a phantom `s`; the cascade touches only the `i` and `j`
/// legs, so that slot choice does not enter the recurrence.
///
/// Electron 1 folds through the σ transform `c2s_si_3c2e1`
/// ([`cart_to_spinor_si_3c2e1`]); the auxiliary index is spherical, as libcint's
/// `sph2e_inner` makes it.
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_ipspsp1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // 12 host gout components = 3 tensor axes x the 4 sigma blocks (gc_x/gc_y/gc_z/gc_1).
    const NGOUT: usize = 12;
    const RANK: usize = 3;

    if plan.representation != Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "int3c2e_ipspsp1 is a sigma family (spinor only); requested {}",
                plan.representation
            ),
        });
    }

    let shape = build_2e_shape(li as usize + 2, lj as usize + 1, 0, lk as usize);
    if shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", shape.nroots),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    let rl = rk;
    let rk_phantom = rk;

    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsk = nsph(lk);
    let block_len = nci * ncj * nck;
    let total_len = NGOUT * block_len;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    let f12_shape = two_e_shape_as_f12(&shape);
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * total_len];

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                let pdata_ij =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
                let pdata_kl = compute_pdata_host(
                    0.0,
                    ak,
                    rk_phantom[0],
                    rk_phantom[1],
                    rk_phantom[2],
                    rl[0],
                    rl[1],
                    rl[2],
                    1.0,
                    1.0,
                );
                let fac_env = common_factor * pdata_ij.fac * pdata_kl.fac;

                let g =
                    fill_g_tensor_2e(ai, aj, 0.0, ak, &ri, &rj, &rk_phantom, &rl, shape, fac_env);

                let gout = crate::kernels::f12::gout_ip_sigma(
                    0,
                    &g,
                    &f12_shape,
                    li as usize,
                    lj as usize,
                    0,
                    lk as usize,
                    ai,
                    aj,
                    0.0,
                    ak,
                );

                for ci in 0..n_ctr_i {
                    let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                        for ck in 0..n_ctr_k {
                            let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                            let weight = coeff_i * coeff_j * coeff_k;
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len;
                            for n in 0..block_len {
                                for comp in 0..NGOUT {
                                    cart_blocks[base + comp * block_len + n] +=
                                        weight * gout[n * NGOUT + comp];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let di = spinor_len(li, shell_i.kappa as i32);
    let dj = spinor_len(lj, shell_j.kappa as i32);
    let n2c_i = n_ctr_i * di;
    let n2c_j = n_ctr_j * dj;
    let n_aux = n_ctr_k * nsk;
    let spinor_block = n2c_i * n2c_j * n_aux * 2;
    if staging.len() < RANK * spinor_block {
        return Err(cintxRsError::BufferTooSmall {
            required: RANK * spinor_block,
            provided: staging.len(),
        });
    }

    let mut tmp = vec![F::from_f64_lossy(0.0); di * dj * nsk * 2];
    for axis in 0..RANK {
        let staging_axis_base = axis * spinor_block;
        for ci in 0..n_ctr_i {
            for cj in 0..n_ctr_j {
                for ck in 0..n_ctr_k {
                    let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len;
                    // gout component index is `tensor_axis * 4 + sigma_block`
                    // (CINT3c2e_spinor_drv advances gctr by nc*ncomp_e1 per e2 block).
                    let blk = |sigma: usize| {
                        let off = base + (axis * 4 + sigma) * block_len;
                        &cart_blocks[off..off + block_len]
                    };
                    cart_to_spinor_si_3c2e1::<F>(
                        &mut tmp,
                        blk(0),
                        blk(1),
                        blk(2),
                        blk(3),
                        li,
                        shell_i.kappa,
                        lj,
                        shell_j.kappa,
                        lk,
                    )?;
                    for mk in 0..nsk {
                        let kidx = ck * nsk + mk;
                        for j_sp in 0..dj {
                            let jidx = cj * dj + j_sp;
                            for i_sp in 0..di {
                                let iidx = ci * di + i_sp;
                                let src = ((mk * dj + j_sp) * di + i_sp) * 2;
                                let dst =
                                    staging_axis_base + ((kidx * n2c_j + jidx) * n2c_i + iidx) * 2;
                                staging[dst] = tmp[src];
                                staging[dst + 1] = tmp[src + 1];
                            }
                        }
                    }
                }
            }
        }
    }

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;
    let staging_bytes = std::mem::size_of_val(staging);
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

/// `int3c2e_ipip2` — ∇² on the auxiliary k center (KET headroom, HESS-03). Thin
/// wrapper over [`launch_center_3c2e_hess`] with [`HessKind::Ipip2`].
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_hess2<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    launch_center_3c2e_hess::<F>(
        plan,
        shell_i,
        shell_j,
        shell_k,
        li,
        lj,
        lk,
        HessKind::Ipip2,
        staging,
    )
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
            backend, plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }

    // Phase 23 DRV1-05: int3c2e_ip2 (∇ on the auxiliary k center). The derivative is
    // applied via nabla1l_2e on the 2e `ll` slot (real aux k), NOT nabla1k_2e (which
    // would touch the phantom slot — RESEARCH Pitfall 2).
    if plan.descriptor.operator_name() == "ip2" {
        return launch_center_3c2e_ip2::<F>(
            backend, plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }

    // Phase 25 HESS-03: int3c2e_ipip1 (∇² on bra center i) and int3c2e_ipip2 (∇²
    // on the auxiliary k center — KET headroom). Both rank-9, HOST-routed through
    // `fill_g_tensor_2e` (the +2 raise can reach nroots 6..12, FND-02).
    // W4-05: int3c2e_ipspsp1 (sigma-p gradient, spinor only).
    if plan.descriptor.operator_name() == "ipspsp1" {
        return launch_center_3c2e_ipspsp1::<F>(
            plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }

    if plan.descriptor.operator_name() == "ipip1" {
        return launch_center_3c2e_hess1::<F>(
            plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }
    if plan.descriptor.operator_name() == "ipip2" {
        return launch_center_3c2e_hess2::<F>(
            plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }
    if plan.descriptor.operator_name() == "ipvip1" {
        return launch_center_3c2e_hess::<F>(
            plan,
            shell_i_in,
            shell_j_in,
            shell_k,
            li_in,
            lj_in,
            lk,
            HessKind::Ipvip1,
            staging,
        );
    }
    if plan.descriptor.operator_name() == "ip1ip2" {
        return launch_center_3c2e_hess::<F>(
            plan,
            shell_i_in,
            shell_j_in,
            shell_k,
            li_in,
            lj_in,
            lk,
            HessKind::Ip1ip2,
            staging,
        );
    }

    let swap_ij = li_in < lj_in;
    let (shell_i, shell_j, li, lj) = if swap_ij {
        (shell_j_in, shell_i_in, lj_in, li_in)
    } else {
        (shell_i_in, shell_j_in, li_in, lj_in)
    };
    // D-PBC-24: `rys_order` is `(li + lj + lk)/2 + 1` — 3c2e folds the auxiliary
    // shell into the 2e `ll` slot with `lk_ceil = 0` (g3c2e.c:70). Short range
    // DOUBLES the root count for `rys_order <= 3` (g3c2e.c:70-77), and that is
    // what `g_stride_i` / `g_size` are built from, so it has to be applied here.
    let range_omega = plan.operator_env_params.range_omega;
    let rys_order = (li as usize + lj as usize + lk as usize) / 2 + 1;
    let nrys_roots = cintx_runtime::range_omega::nrys_roots_for(rys_order, range_omega);
    // D-PBC-24 stage 4: the device `#[cube]` kernel has no omega branch, so a
    // range-separated 3c2e routes to the host engine — explicitly and logged.
    // The host chain (`fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`)
    // is the one `scalar_device_tests` already proves device-equivalent, and it
    // serves nroots up to the host Rys ceiling, so the device ceiling below does
    // not apply to it.
    let route_host = cintx_runtime::range_omega::is_range_separated(range_omega);
    if route_host {
        tracing::debug!(
            family = "3c2e",
            omega = range_omega.unwrap_or(0.0),
            rys_order,
            nrys_roots,
            "range-separated 3c2e routed to the host Rys engine (D-PBC-24 stage 4)"
        );
    }
    // Task 33-03: the ceiling is the backend's and the family's, not a
    // constant. It is `MAX_DEVICE_NROOTS` unless `extended-device-rys` is
    // compiled in, this backend's FMA-fusion probe passed, *and* `int3c2e` has
    // been flipped onto the inline Wheeler entry — in which case it serves
    // 6..=12 and this class no longer has to be refused.
    let nrys_ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int3c2e,
    );
    // The HOST engine's own ceiling. `rys_roots_host` panics above 12 (that is
    // where the vendor itself would need quadmath), so the range-separated route
    // fails closed with a typed error instead of reaching it.
    if route_host && nrys_roots > 12 {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "range_omega:3c2e:nroots={nrys_roots}: the host Rys engine serves                  nroots<=12 (l=({li},{lj},{lk}))"
            ),
        });
    }
    if !route_host && nrys_roots > nrys_ceiling {
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

    // Dispatch onto the resolved backend's device client (compute in f64), or
    // onto the host engine when range separation is in play (D-PBC-24).
    let cart_buf: Vec<f64> = if route_host {
        host_3c2e_cart_blocks(
            li,
            lj,
            lk,
            ri,
            rj,
            rk,
            &exps_i,
            &exps_j,
            &exps_k,
            &coeff_i,
            &coeff_j,
            &coeff_k,
            n_ctr_i,
            n_ctr_j,
            n_ctr_k,
            nrys_roots,
            common_factor,
            range_omega,
        )?
    } else {
        match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(client) => run_3c2e_device::<cubecl::cpu::CpuRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                nrys_roots as u32,
                ri,
                rj,
                rk,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &coeff_i,
                &coeff_j,
                &coeff_k,
            ),
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(client, _) => run_3c2e_device::<cubecl_wgpu::WgpuRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                nrys_roots as u32,
                ri,
                rj,
                rk,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &coeff_i,
                &coeff_j,
                &coeff_k,
            ),
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(client) => run_3c2e_device::<cubecl_cuda::CudaRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                nrys_roots as u32,
                ri,
                rj,
                rk,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &coeff_i,
                &coeff_j,
                &coeff_k,
            ),
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(client) => run_3c2e_device::<cubecl_hip::HipRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                nrys_roots as u32,
                ri,
                rj,
                rk,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &coeff_i,
                &coeff_j,
                &coeff_k,
            ),
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(client, _) => run_3c2e_device::<cubecl_wgpu::WgpuRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                nrys_roots as u32,
                ri,
                rj,
                rk,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &coeff_i,
                &coeff_j,
                &coeff_k,
            ),
        }
    };

    // `cart_buf` holds `n_ctr_i * n_ctr_j * n_ctr_k` Cartesian blocks in the
    // *canonical* (li >= lj) shell order, contraction-major. Restore the caller's
    // (i, j) order: the Cartesian axes transpose with the shells, and so does the
    // contraction index — `swap_ij` swapped which input shell is canonical `i`.
    let n_ctr_i_in = shell_i_in.nctr as usize;
    let n_ctr_j_in = shell_j_in.nctr as usize;
    let cart_block = nci * ncj * nck;
    let (nci_in, ncj_in) = (ncart(li_in), ncart(lj_in));
    let cart_block_in = nci_in * ncj_in * nck;

    let mut cart_out = vec![0.0_f64; n_ctr_i_in * n_ctr_j_in * n_ctr_k * cart_block_in];
    for ci in 0..n_ctr_i_in {
        for cj in 0..n_ctr_j_in {
            for ck in 0..n_ctr_k {
                // libcint's 3c2e recurrence chooses ibase adaptively (li > lj).
                // We evaluate in canonical order li>=lj and transpose back when
                // the input had li<lj.
                let (ca, cb) = if swap_ij { (cj, ci) } else { (ci, cj) };
                let src = ((ca * n_ctr_j + cb) * n_ctr_k + ck) * cart_block;
                let dst = ((ci * n_ctr_j_in + cj) * n_ctr_k + ck) * cart_block_in;
                let block = &cart_buf[src..src + cart_block];
                if swap_ij {
                    let transposed = transpose_ij_3idx(block, nci, ncj, nck);
                    cart_out[dst..dst + cart_block_in].copy_from_slice(&transposed);
                } else {
                    cart_out[dst..dst + cart_block_in].copy_from_slice(block);
                }
            }
        }
    }

    // Apply cart-to-sph/spinor or copy Cartesian, casting to F at the staging
    // write. The AO index of contraction `c` and component `m` is `c*n<comp>+m`,
    // so a general contraction is scattered rather than copied.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i_in * nsi_in;
            let dj = n_ctr_j_in * nsj_in;
            for ci in 0..n_ctr_i_in {
                for cj in 0..n_ctr_j_in {
                    for ck in 0..n_ctr_k {
                        let base = ((ci * n_ctr_j_in + cj) * n_ctr_k + ck) * cart_block_in;
                        let sph = cart_to_sph_3c2e(
                            &cart_out[base..base + cart_block_in],
                            li_in,
                            lj_in,
                            lk,
                        );
                        for mk in 0..nsk {
                            let kidx = ck * nsk + mk;
                            for mj in 0..nsj_in {
                                let jidx = cj * nsj_in + mj;
                                for mi in 0..nsi_in {
                                    let iidx = ci * nsi_in + mi;
                                    let dst = iidx + di * (jidx + dj * kidx);
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(
                                            sph[mi + nsi_in * (mj + nsj_in * mk)],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            // cart_to_spinor_sf_3c2e is generic over F: CintFloat (Plan 04) and
            // consumes exactly one Cartesian block; general contraction is not
            // wired through it, so fail closed rather than transforming only the
            // first block.
            if n_ctr_i_in != 1 || n_ctr_j_in != 1 || n_ctr_k != 1 {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!(
                        "3c2e-spinor:general-contraction nctr=({n_ctr_i_in},{n_ctr_j_in},{n_ctr_k})"
                    ),
                });
            }
            let kappa_i = shell_i_in.kappa;
            let kappa_j = shell_j_in.kappa;
            cart_to_spinor_sf_3c2e::<F>(staging, &cart_out, li_in, kappa_i, lj_in, kappa_j, lk)?;
        }
        Representation::Cart => {
            let di = n_ctr_i_in * nci_in;
            let dj = n_ctr_j_in * ncj_in;
            for ci in 0..n_ctr_i_in {
                for cj in 0..n_ctr_j_in {
                    for ck in 0..n_ctr_k {
                        let base = ((ci * n_ctr_j_in + cj) * n_ctr_k + ck) * cart_block_in;
                        for mk in 0..nck {
                            let kidx = ck * nck + mk;
                            for mj in 0..ncj_in {
                                let jidx = cj * ncj_in + mj;
                                for mi in 0..nci_in {
                                    let iidx = ci * nci_in + mi;
                                    let dst = iidx + di * (jidx + dj * kidx);
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(
                                            cart_out[base + mi + nci_in * (mj + ncj_in * mk)],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Per-symbol nonzero sentinel
    // WR-06: precision-aware sentinel so f32 stale lanes (< f32 noise floor ~1e-7)
    // are not counted. The outer F32 arm already bounds staging to out_elems, so this
    // scan cannot touch stale upper-half lanes.
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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
            launch_center_3c2e_typed::<f32>(
                backend,
                plan,
                specialization,
                &mut staging_f32[..out_elems],
            )
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
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell,
        };
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| {
            Arc::new(
                Shell::try_new(
                    atom_idx,
                    0,
                    1,
                    1,
                    0,
                    Representation::Cart,
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells =
            Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(22),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let mut plan = ExecutionPlan::new(
            OperatorId::new(22),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging_outer = vec![0.0_f64; 1];
        let mut staging_typed = vec![0.0_f64; 1];

        // Call outer dispatcher
        let result_outer = launch_center_3c2e(&backend, &plan, &spec, &mut staging_outer);
        assert!(
            result_outer.is_ok(),
            "outer f64 3c2e should succeed: {:?}",
            result_outer
        );

        // Call typed inner directly (RED: compile fails until launch_center_3c2e_typed defined)
        let result_typed =
            launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging_typed);
        assert!(
            result_typed.is_ok(),
            "typed f64 3c2e should succeed: {:?}",
            result_typed
        );

        // Byte-identical check
        assert_eq!(
            staging_outer[0].to_bits(),
            staging_typed[0].to_bits(),
            "f64 outer and typed 3c2e should be byte-identical: outer={} typed={}",
            staging_outer[0],
            staging_typed[0]
        );
        assert!(
            staging_outer[0].is_finite() && staging_outer[0].abs() > 1e-30,
            "3c2e s-s-s value should be finite and nonzero: {}",
            staging_outer[0]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1d: launch_center_3c2e F32 path runs without panic.
    // RED: compile fails until launch_center_3c2e dispatches on plan.precision.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c2e_f32_smoke() {
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell,
        };
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| {
            Arc::new(
                Shell::try_new(
                    atom_idx,
                    0,
                    1,
                    1,
                    0,
                    Representation::Cart,
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells =
            Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(22),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let mut plan = ExecutionPlan::new(
            OperatorId::new(22),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_center_3c2e(&backend, &plan, &spec, &mut staging);
        assert!(
            result.is_ok(),
            "F32 3c2e should succeed without panic: {:?}",
            result
        );

        let staging_f32 = bytemuck::cast_slice::<f64, f32>(&staging);
        assert!(
            staging_f32[0].is_finite(),
            "F32 3c2e result should be finite: {}",
            staging_f32[0]
        );
        assert!(
            staging_f32[0] > 0.0,
            "F32 3c2e result should be positive: {}",
            staging_f32[0]
        );
    }

    #[test]
    fn test_fill_g_tensor_3c2e_sss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.1, 0.2];
        let pair = compute_pdata_host(1.0, 1.0, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        let g = fill_g_tensor_3c2e(&pair, 1.0, ri, rk, 0, 0, 0, 1, 1.0, None)
            .expect("full range is supported")
            .expect("full range is never screened out");
        assert_eq!(g.len(), 3, "s-s-s should produce one root x one n x one m");
        assert!(
            g[2].abs() > 1e-20,
            "gz root must be non-zero for s-s-s primitive"
        );
    }

    #[test]
    fn test_contract_3c2e_sss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.1, 0.2];
        let pair = compute_pdata_host(1.0, 1.0, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        let g2d = fill_g_tensor_3c2e(&pair, 1.0, ri, rk, 0, 0, 0, 1, 1.0, None)
            .expect("full range is supported")
            .expect("full range is never screened out");
        let g_split = split_ij_hrr(
            &g2d,
            0,
            0,
            0,
            1,
            [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]],
        );
        let out = contract_3c2e(&g_split, 0, 0, 0, 1);
        assert_eq!(out.len(), 1);
        assert!(
            out[0].abs() > 1e-20,
            "contracted s-s-s 3c2e value must be non-zero"
        );
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
        let g2d = fill_g_tensor_3c2e(&pair, ak, ri, rk, li, lj, lk, nrys, fac_env, None)
            .expect("full range is supported")
            .expect("full range is never screened out");
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
            ai,
            aj,
            ak,
            ri,
            rj,
            rk,
            li,
            lj,
            lk,
            common_factor,
            coeff_i,
            coeff_j,
            coeff_k,
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

        assert_eq!(
            host.len(),
            dev.len(),
            "length mismatch li={li} lj={lj} lk={lk}"
        );
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
        // Flattened three-shell basis: one primitive, one contraction each.
        let exps = [1.0_f32, 1.0, 1.0];
        let coeffs = [1.0_f32, 1.0, 1.0];
        let centers = [0.0_f32, 0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.7, 0.2];
        let shell_meta: [u32; 12] = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1];
        // `[si, sj, sk, out_off, class]` — one class, index 0.
        let triples: [u32; 5] = [0, 1, 2, 0, 0];
        let class_shape: [u32; THREE_C2E_SHAPE_STRIDE] = [0, 0, 0];
        let g_zero = [0.0_f32; 3];
        let gs_zero = [0.0_f32; 3];
        let work_zero = [0.0_f32; 1];
        let out_zero = [0.0_f32; 1];

        let exps_h = client.create_from_slice(f32::as_bytes(&exps));
        let coeffs_h = client.create_from_slice(f32::as_bytes(&coeffs));
        let centers_h = client.create_from_slice(f32::as_bytes(&centers));
        let meta_h = client.create_from_slice(u32::as_bytes(&shell_meta));
        let triples_h = client.create_from_slice(u32::as_bytes(&triples));
        let shape_h = client.create_from_slice(u32::as_bytes(&class_shape));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let gs_h = client.create_from_slice(f32::as_bytes(&gs_zero));
        let work_h = client.create_from_slice(f32::as_bytes(&work_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let class_factor = [((PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(0)
            * common_fac_sp(0)
            * common_fac_sp(0)) as f32];
        let factor_h = client.create_from_slice(f32::as_bytes(&class_factor));
        // The extended-Rys tables are an unconditional kernel argument; an
        // `nroots = 1` smoke launch never reads them.
        let rys_tables = crate::math::rys_wheeler::ext_rys_tables();
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));

        center_3c2e_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_h, exps.len()) },
            unsafe { ArrayArg::from_raw_parts(coeffs_h, coeffs.len()) },
            unsafe { ArrayArg::from_raw_parts(centers_h, centers.len()) },
            unsafe { ArrayArg::from_raw_parts(meta_h, shell_meta.len()) },
            unsafe { ArrayArg::from_raw_parts(triples_h, triples.len()) },
            unsafe { ArrayArg::from_raw_parts(shape_h, class_shape.len()) },
            unsafe { ArrayArg::from_raw_parts(factor_h, class_factor.len()) },
            unsafe { ArrayArg::from_raw_parts(rys_tab_h, EXT_TABLES_LEN) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(gs_h, 3) },
            unsafe { ArrayArg::from_raw_parts(work_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            PIE4 as f32,
            0.0_f32, // prim_tol: the f32 smoke launch never screens
            1u32,    // n_triples
            1u32,    // n_cubes
            3u32,    // g_stride (one slab, unpadded)
            3u32,    // split_stride
            1u32,    // work_slab
            1u32,    // nroots
            // One cube, one triple: the shape these single slabs are sized for.
            0u32,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 3c2e kernel result must be finite");
        assert!(out > 0.0, "s-s-s 3c2e f32 result should be positive: {out}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// int3c2e_ip1 device-vs-host cross-check (CpuRuntime, f64). The CubeCL kernel
// (`center_3c2e_ip1_kernel`) must reproduce the host per-triple component-leading
// `cart_blocks` (the pre-transform Cartesian tensor: `fill_g_tensor_2e` →
// `gout_ip1` → component-leading transpose) exactly.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod ip1_device_tests {
    use super::*;

    /// Host reference: the per-triple component-leading `[3, nck, ncj, nci]`
    /// Cartesian block for a single-primitive single-contraction shell triple,
    /// via the verbatim host `fill_g_tensor_2e` + `gout_ip1` chain.
    ///
    /// `pub(super)`: the ip2 device test reuses this as the `ip2 != ip1` reference
    /// (Pitfall 2 wrong-slot guard).
    pub(super) fn host_ip1_cart_blocks(
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
        let grad_shape = build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize);
        let grad_f12_shape = two_e_shape_as_f12(&grad_shape);
        let pdata_ij =
            compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let pdata_kl =
            compute_pdata_host(0.0, ak, rk[0], rk[1], rk[2], rk[0], rk[1], rk[2], 1.0, 1.0);
        let fac_env = common_factor * pdata_ij.fac * pdata_kl.fac;
        let g = fill_g_tensor_2e(ai, aj, 0.0, ak, &ri, &rj, &rk, &rk, grad_shape, fac_env);
        let gout = crate::kernels::f12::gout_ip1(
            &g,
            &grad_f12_shape,
            li as usize,
            lj as usize,
            0,
            lk as usize,
            ai,
        );

        let block_len = ncart(li) * ncart(lj) * ncart(lk);
        let weight = coeff_i * coeff_j * coeff_k;
        let mut out = vec![0.0_f64; 3 * block_len];
        for n in 0..block_len {
            for comp in 0..3usize {
                out[comp * block_len + n] += weight * gout[n * 3 + comp];
            }
        }
        out
    }

    /// A CPU `ResolvedBackend` for the derivative device tests.
    ///
    /// The batched entry point takes a backend rather than a raw client because
    /// it uploads the flattened basis through the backend's arm.
    fn test_cpu_backend() -> ResolvedBackend {
        ResolvedBackend::from_intent(&cintx_runtime::BackendIntent {
            backend: cintx_runtime::BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend")
    }

    fn assert_device_matches_host_ip1(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64) {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.4_f64, -0.2, 0.6];
        let rk = [0.1_f64, 0.7, 0.3];
        let coeff_i = 0.9_f64;
        let coeff_j = 1.1_f64;
        let coeff_k = 0.8_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk);
        let grad_shape = build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize);

        let host = host_ip1_cart_blocks(
            ai,
            aj,
            ak,
            ri,
            rj,
            rk,
            li,
            lj,
            lk,
            common_factor,
            coeff_i,
            coeff_j,
            coeff_k,
        );
        let dev = run_3c2e_deriv_single(
            &test_cpu_backend(),
            ThreeC2eDerivFamily::Ip1,
            li as u32,
            lj as u32,
            lk as u32,
            &grad_shape,
            common_factor,
            ri,
            rj,
            rk,
            [1, 1, 1],
            [1, 1, 1],
            [&[ai], &[aj], &[ak]],
            [&[coeff_i], &[coeff_j], &[coeff_k]],
        );

        assert_eq!(
            host.len(),
            dev.len(),
            "length mismatch li={li} lj={lj} lk={lk}"
        );
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host ip1 mismatch li={li} lj={lj} lk={lk} idx={idx}: \
                 host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    #[test]
    fn test_ip1_device_matches_host_sss() {
        assert_device_matches_host_ip1(0, 0, 0, 0.8, 1.0, 1.2);
    }

    #[test]
    fn test_ip1_device_matches_host_pss() {
        assert_device_matches_host_ip1(1, 0, 0, 1.3, 0.8, 0.9);
    }

    #[test]
    fn test_ip1_device_matches_host_sps() {
        assert_device_matches_host_ip1(0, 1, 0, 0.7, 1.1, 0.6);
    }

    #[test]
    fn test_ip1_device_matches_host_ssp() {
        assert_device_matches_host_ip1(0, 0, 1, 0.9, 0.5, 0.7);
    }

    #[test]
    fn test_ip1_device_matches_host_pps() {
        assert_device_matches_host_ip1(1, 1, 0, 0.6, 0.9, 0.8);
    }

    #[test]
    fn test_ip1_device_matches_host_psp() {
        assert_device_matches_host_ip1(1, 0, 1, 0.7, 1.2, 0.55);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// int3c2e_ip2 device-vs-host cross-check (CpuRuntime, f64). The CubeCL kernel
// (`center_3c2e_ip2_kernel`) must reproduce the host per-triple component-leading
// `cart_blocks` (`fill_g_tensor_2e` with the `ll = lk+1` headroom →
// `gout_ipn(Nabla1Center::L)` → component-leading transpose) exactly.
//
// Pitfall 2 guard: the ip2 derivative is on the auxiliary k (the 2e `ll` slot via
// `nabla1l_2e`), NOT the bra i. A `test_ip2_not_equal_ip1` assertion catches a
// wrong-slot nabla — if ip2 == ip1, the derivative hit the wrong center.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod ip2_device_tests {
    use super::*;

    /// Host reference: the per-triple component-leading `[3, nck, ncj, nci]`
    /// Cartesian block for a single-primitive single-contraction shell triple,
    /// via the verbatim host `fill_g_tensor_2e` + `gout_ipn(Nabla1Center::L)` chain.
    fn host_ip2_cart_blocks(
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
        // ip2: bra NOT raised; the real aux k (ll slot) raised to lk+1.
        let grad_shape = build_2e_shape(li as usize, lj as usize, 0, lk as usize + 1);
        let grad_f12_shape = two_e_shape_as_f12(&grad_shape);
        let pdata_ij =
            compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let pdata_kl =
            compute_pdata_host(0.0, ak, rk[0], rk[1], rk[2], rk[0], rk[1], rk[2], 1.0, 1.0);
        let fac_env = common_factor * pdata_ij.fac * pdata_kl.fac;
        let g = fill_g_tensor_2e(ai, aj, 0.0, ak, &ri, &rj, &rk, &rk, grad_shape, fac_env);
        // nabla on the L (ll) slot — real aux k — at base lk, exponent ak.
        let gout = crate::kernels::f12::gout_ipn(
            &g,
            &grad_f12_shape,
            li as usize,
            lj as usize,
            0,
            lk as usize,
            crate::kernels::f12::Nabla1Center::L,
            ak,
        );

        let block_len = ncart(li) * ncart(lj) * ncart(lk);
        let weight = coeff_i * coeff_j * coeff_k;
        let mut out = vec![0.0_f64; 3 * block_len];
        for n in 0..block_len {
            for comp in 0..3usize {
                out[comp * block_len + n] += weight * gout[n * 3 + comp];
            }
        }
        out
    }

    /// A CPU `ResolvedBackend` for the derivative device tests.
    ///
    /// The batched entry point takes a backend rather than a raw client because
    /// it uploads the flattened basis through the backend's arm.
    fn test_cpu_backend() -> ResolvedBackend {
        ResolvedBackend::from_intent(&cintx_runtime::BackendIntent {
            backend: cintx_runtime::BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend")
    }

    fn device_ip2(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64) -> Vec<f64> {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.4_f64, -0.2, 0.6];
        let rk = [0.1_f64, 0.7, 0.3];
        let coeff_i = 0.9_f64;
        let coeff_j = 1.1_f64;
        let coeff_k = 0.8_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk);
        let grad_shape = build_2e_shape(li as usize, lj as usize, 0, lk as usize + 1);
        run_3c2e_deriv_single(
            &test_cpu_backend(),
            ThreeC2eDerivFamily::Ip2,
            li as u32,
            lj as u32,
            lk as u32,
            &grad_shape,
            common_factor,
            ri,
            rj,
            rk,
            [1, 1, 1],
            [1, 1, 1],
            [&[ai], &[aj], &[ak]],
            [&[coeff_i], &[coeff_j], &[coeff_k]],
        )
    }

    fn assert_device_matches_host_ip2(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64) {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.4_f64, -0.2, 0.6];
        let rk = [0.1_f64, 0.7, 0.3];
        let coeff_i = 0.9_f64;
        let coeff_j = 1.1_f64;
        let coeff_k = 0.8_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk);

        let host = host_ip2_cart_blocks(
            ai,
            aj,
            ak,
            ri,
            rj,
            rk,
            li,
            lj,
            lk,
            common_factor,
            coeff_i,
            coeff_j,
            coeff_k,
        );
        let dev = device_ip2(li, lj, lk, ai, aj, ak);

        assert_eq!(
            host.len(),
            dev.len(),
            "length mismatch li={li} lj={lj} lk={lk}"
        );
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host ip2 mismatch li={li} lj={lj} lk={lk} idx={idx}: \
                 host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    #[test]
    fn test_ip2_device_matches_host_sss() {
        assert_device_matches_host_ip2(0, 0, 0, 0.8, 1.0, 1.2);
    }

    #[test]
    fn test_ip2_device_matches_host_pss() {
        assert_device_matches_host_ip2(1, 0, 0, 1.3, 0.8, 0.9);
    }

    #[test]
    fn test_ip2_device_matches_host_sps() {
        assert_device_matches_host_ip2(0, 1, 0, 0.7, 1.1, 0.6);
    }

    #[test]
    fn test_ip2_device_matches_host_ssp() {
        assert_device_matches_host_ip2(0, 0, 1, 0.9, 0.5, 0.7);
    }

    #[test]
    fn test_ip2_device_matches_host_pds() {
        // NON-SQUARE bra (p×d): a square block is transpose-symmetric and hides
        // axis/layout bugs (RESEARCH anti-pattern).
        assert_device_matches_host_ip2(1, 2, 0, 0.6, 0.9, 0.8);
    }

    #[test]
    fn test_ip2_device_matches_host_psp() {
        assert_device_matches_host_ip2(1, 0, 1, 0.7, 1.2, 0.55);
    }

    /// Pitfall 2: the ip2 derivative is on the auxiliary k (the `ll` slot), so its
    /// output MUST differ from ip1 (∇ on the bra i). If they match, the nabla hit the
    /// wrong slot. Use a NON-SQUARE i×j (p×d) block so a transposed layout cannot
    /// accidentally match.
    #[test]
    fn test_ip2_not_equal_ip1() {
        let (li, lj, lk) = (1u8, 2u8, 0u8);
        let (ai, aj, ak) = (0.6_f64, 0.9, 0.8);
        let ip2 = device_ip2(li, lj, lk, ai, aj, ak);

        // ip1 host reference (∇ on bra i), same triple.
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.4_f64, -0.2, 0.6];
        let rk = [0.1_f64, 0.7, 0.3];
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk);
        let ip1 = ip1_device_tests::host_ip1_cart_blocks(
            ai,
            aj,
            ak,
            ri,
            rj,
            rk,
            li,
            lj,
            lk,
            common_factor,
            0.9,
            1.1,
            0.8,
        );

        assert_eq!(ip2.len(), ip1.len(), "ip1/ip2 length mismatch");
        let any_diff = ip2
            .iter()
            .zip(ip1.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(
            any_diff,
            "int3c2e_ip2 output equals int3c2e_ip1 — the nabla hit the wrong slot (Pitfall 2)"
        );
        // Sanity: ip2 must be non-trivial.
        assert!(
            ip2.iter().any(|v| v.abs() > 1e-12),
            "int3c2e_ip2 output is all zeros"
        );
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
//   - spinor (27-04, D-06): int3c2e_ip1 with Representation::Spinor is SUPPORTED via
//     the cart_to_spinor_sf_derivative_3c2e wrapper (byte-identity vs libcint proven
//     in cintx-oracle/tests/spinor_deriv_parity.rs).
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
    fn build_ip1_plan(li: u8, lj: u8, lk: u8) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
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
    fn build_plain_plan(li: u8, lj: u8, lk: u8) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
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
        let all_equal = staging.iter().all(|v| (v - staging[0]).abs() <= 1e-15);
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
        let (plain, _) = run(
            &basis_plain,
            shells_plain,
            op_plain,
            Representation::Spheric,
        )
        .unwrap();

        // plain has 3 AOs (p,s,s); ip1 has 9 (3 comps × 3 AO).
        assert_eq!(plain.len(), 3, "plain (p,s,s) 3c2e should be 3 AOs");
        assert_eq!(
            ip1.len(),
            9,
            "ip1 (p,s,s) 3c2e should be 9 (3 comps × 3 AO)"
        );

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

    // Spinor (27-04, D-06): int3c2e_ip1 with Representation::Spinor is now SUPPORTED
    // via the cart_to_spinor_sf_derivative_3c2e wrapper (the pre-27 UnsupportedApi
    // rejection was lifted in plan 27-04). Byte-identity vs vendored libcint is proven
    // in cintx-oracle/tests/spinor_deriv_parity.rs; this is a launcher-level smoke test
    // that the arm computes (no longer UnsupportedApi) and writes a correctly-sized,
    // nonzero spinor block. The plan is built Spheric then flipped to Spinor — the cart
    // blocks are representation-independent; only the final fold differs.
    #[test]
    fn test_int3c2e_ip1_spinor_supported() {
        let (basis, shells, op) = build_ip1_plan(0, 0, 0);
        let opts = ExecutionOptions::default();
        let q =
            query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        // (s,s,s) spinor int3c2e_ip1 output: di=dj=spinor_len(0,0)=2, aux-k SPHERICAL
        // nsk=nsph(0)=1, ncomp=3, complex interleaved → 3*(2*2*1*2) = 24 f64.
        let mut staging = vec![0.0_f64; 24];
        let result = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            result.is_ok(),
            "spinor int3c2e_ip1 is SUPPORTED since plan 27-04 (derivative wrapper), got: {result:?}"
        );
        assert!(
            staging.iter().any(|v| v.abs() > 1e-18),
            "spinor int3c2e_ip1 produced all-zero output: {staging:?}"
        );
    }
}
