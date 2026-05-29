//! `origk` family: origin-k-displaced 3c1e integrals (`cint3c1e_a.c`).
//! Split out of the original single-file `unstable.rs`.
//!
//! # Host / device split (quick-260529-twi Task 5)
//!
//! ON DEVICE (`#[cube(launch)] origk_scalar_kernel<F>`): the per-primitive
//! 3-center 1e G-tensor build (VRR in the combined j+k direction, HRR for i,
//! HRR for k-separation) followed by the r^n Cartesian contraction
//! (`contract_origk`), accumulated over primitive + contraction triples into a
//! per-shell-triple Cartesian block, in f64-internal / `F`-output. This covers
//! the SCALAR origk symbols `int3c1e_{r2,r4,r6}_origk_sph`.
//!
//! ON HOST: the `cart_to_sph_3c1e` representation transform (its coefficient
//! tables are host-only) + the AO scatter into `staging`.
//!
//! DEFERRED-TO-HOST (documented, mirrors the 1e/2e ports deferring derivative
//! sub-paths): the `ip1` bra-gradient variants `int3c1e_ip1_{r2,r4,r6}_origk_sph`
//! (host `contract_origk_ip1` + `g1e_d_i_3c1e`). The ip1 ladder applies a
//! separable bra-derivative `D_I` after the r^n expansion and is a distinct
//! derivative port; the scalar device path branches on `origk_variant` and the
//! ip1 variants keep the existing host code path unchanged.
//!
//! NOTE on Rys roots: origk is a 1e displaced-overlap-style operator and needs
//! NO Rys quadrature (`fill_g_tensor_3c1e_origk` builds the G-tensor directly via
//! exponential VRR/HRR). The `#[comptime] nroots` kernel parameter is carried for
//! parity with the family-port template but is numerically unused; the
//! `MAX_DEVICE_NROOTS` guard fail-closes on the angular-momentum sum instead.

use super::shared::{SQRTPI, cart_comps, common_fac_sp, make_exec_stats};
use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c1e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// Maximum angular-momentum sum the device origk kernel accepts. `origk` has no
/// Rys roots, so this is a fail-closed bound on `nmax = li_ceil + lj + lk_ceil`
/// (the same numeric ceiling as the host path's `g_alloc` sizing). Mirrors the
/// `MAX_DEVICE_NROOTS` naming used by the other family ports.
const MAX_DEVICE_NROOTS: usize = 5;

// ─────────────────────────────────────────────────────────────────────────────
// Origk family: origin-k-displaced 3c1e integrals
// ─────────────────────────────────────────────────────────────────────────────

/// Variant metadata for origk operators.
struct OrigkVariant {
    k_inc: u8,
    i_inc: u8,
    ncomp: usize,
}

fn origk_variant(op_name: &str) -> Result<OrigkVariant, cintxRsError> {
    match op_name {
        "r2_origk" => Ok(OrigkVariant { k_inc: 2, i_inc: 0, ncomp: 1 }),
        "r4_origk" => Ok(OrigkVariant { k_inc: 4, i_inc: 0, ncomp: 1 }),
        "r6_origk" => Ok(OrigkVariant { k_inc: 6, i_inc: 0, ncomp: 1 }),
        "ip1_r2_origk" => Ok(OrigkVariant { k_inc: 2, i_inc: 1, ncomp: 3 }),
        "ip1_r4_origk" => Ok(OrigkVariant { k_inc: 4, i_inc: 1, ncomp: 3 }),
        "ip1_r6_origk" => Ok(OrigkVariant { k_inc: 6, i_inc: 1, ncomp: 3 }),
        _ => Err(cintxRsError::UnsupportedApi {
            requested: format!("origk variant '{}' not supported", op_name),
        }),
    }
}

/// G1E_D_I for 3c1e: nabla in i-direction.
/// f[i=0, j, k] = -2*ai * g[i=1, j, k]
/// f[i>0, j, k] = i * g[i-1, j, k] + (-2*ai) * g[i+1, j, k]
fn g1e_d_i_3c1e(
    g: &[f64],
    g_alloc: usize,
    li: usize,
    lj: usize,
    lk: usize,
    dj: usize,
    dk: usize,
    ai: f64,
) -> Vec<f64> {
    let mut f = vec![0.0_f64; 3 * g_alloc];
    let ai2 = -2.0 * ai;
    for axis in 0..3 {
        let off = axis * g_alloc;
        for k in 0..=lk {
            for j in 0..=lj {
                let ptr = j * dj + k * dk;
                // i=0
                f[off + ptr] = ai2 * g[off + ptr + 1];
                // i>0
                for i in 1..=li {
                    f[off + ptr + i] = (i as f64) * g[off + ptr + i - 1] + ai2 * g[off + ptr + i + 1];
                }
            }
        }
    }
    f
}

/// Contract origk r^n gout for ncomp=1 variants (r2, r4, r6).
///
/// G1E_R_K shifts by dk in the k-direction. The gout patterns mirror origi
/// but in the k-index instead of i-index.
///
/// r2: g3[k] = g0[k+2] for each axis, s = g3x*g0y*g0z + g0x*g3y*g0z + g0x*g0y*g3z
/// r4: mirrors the r4_origi pattern but with k-shifts
/// r6: same pattern extended to 6th power
///
/// Retained as the host reference for the in-crate device-vs-host cross-check;
/// the live scalar path now computes via [`origk_scalar_kernel`].
#[allow(dead_code)]
fn contract_origk(
    g0: &[f64],
    g_alloc: usize,
    li: u8,
    lj: u8,
    lk: u8,
    dli: usize,
    dlj: usize,
    dk: usize,
    r_power: u8,
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let dj = dli; // g_stride_j = dli
    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);
    let _ = dlj;

    let mut out = vec![0.0_f64; nci * ncj * nck];

    let gx = 0usize;
    let gy = g_alloc;
    let gz = 2 * g_alloc;

    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let bx = ix as usize + jx as usize * dj + kx as usize * dk;
                let by = iy as usize + jy as usize * dj + ky as usize * dk;
                let bz = iz as usize + jz as usize * dj + kz as usize * dk;

                let s = match r_power {
                    2 => {
                        // r^2: sum_d g0[d+2] * g0[other] * g0[other]
                        g0[gx + bx + 2 * dk] * g0[gy + by] * g0[gz + bz]
                            + g0[gx + bx] * g0[gy + by + 2 * dk] * g0[gz + bz]
                            + g0[gx + bx] * g0[gy + by] * g0[gz + bz + 2 * dk]
                    }
                    4 => {
                        // r^4: same pattern as origi_r4 but with dk shifts
                        g0[gx + bx + 4 * dk] * g0[gy + by] * g0[gz + bz]
                            + 2.0 * g0[gx + bx + 2 * dk] * g0[gy + by + 2 * dk] * g0[gz + bz]
                            + 2.0 * g0[gx + bx + 2 * dk] * g0[gy + by] * g0[gz + bz + 2 * dk]
                            + g0[gx + bx] * g0[gy + by + 4 * dk] * g0[gz + bz]
                            + 2.0 * g0[gx + bx] * g0[gy + by + 2 * dk] * g0[gz + bz + 2 * dk]
                            + g0[gx + bx] * g0[gy + by] * g0[gz + bz + 4 * dk]
                    }
                    6 => {
                        // r^6 from cint3c1e_a.c (g3=+2dk, g60=+4dk, g63=+6dk chains).
                        g0[gx + bx + 6 * dk] * g0[gy + by] * g0[gz + bz]
                            + 3.0 * g0[gx + bx + 4 * dk] * g0[gy + by + 2 * dk] * g0[gz + bz]
                            + 3.0 * g0[gx + bx + 4 * dk] * g0[gy + by] * g0[gz + bz + 2 * dk]
                            + 3.0 * g0[gx + bx + 2 * dk] * g0[gy + by + 4 * dk] * g0[gz + bz]
                            + 6.0 * g0[gx + bx + 2 * dk] * g0[gy + by + 2 * dk] * g0[gz + bz + 2 * dk]
                            + 3.0 * g0[gx + bx + 2 * dk] * g0[gy + by] * g0[gz + bz + 4 * dk]
                            + g0[gx + bx] * g0[gy + by + 6 * dk] * g0[gz + bz]
                            + 3.0 * g0[gx + bx] * g0[gy + by + 4 * dk] * g0[gz + bz + 2 * dk]
                            + 3.0 * g0[gx + bx] * g0[gy + by + 2 * dk] * g0[gz + bz + 4 * dk]
                            + g0[gx + bx] * g0[gy + by] * g0[gz + bz + 6 * dk]
                    }
                    _ => 0.0,
                };

                out[(k_idx * ncj + j_idx) * nci + i_idx] += s;
            }
        }
    }

    out
}

/// Contract origk ip1 variants (ncomp=3): nabla on i + r^n on k.
///
/// Each r_power level has a specific gout formula from cint3c1e_a.c.
/// D_I and R_K commute since they operate on different indices, so
/// D_I(g0 + n*dk) = g_di + n*dk.
fn contract_origk_ip1(
    g0: &[f64],
    g_alloc: usize,
    li: u8,
    lj: u8,
    lk: u8,
    dli: usize,
    _dlj: usize,
    dk: usize,
    r_power: u8,
    ai: f64,
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let ncomp = 3;
    let dj = dli;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);

    let mut out = vec![0.0_f64; nci * ncj * nck * ncomp];

    // Build D_I(g0): nabla in i-direction on the full g0 tensor (including elevated k)
    let g_di = g1e_d_i_3c1e(g0, g_alloc, li as usize, lj as usize, (lk as usize) + (r_power as usize), dj, dk, ai);

    let cart_size = nci * ncj * nck;

    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let bx = ix as usize + jx as usize * dj + kx as usize * dk;
                let by = iy as usize + jy as usize * dj + ky as usize * dk;
                let bz = iz as usize + jz as usize * dj + kz as usize * dk;

                let n = (k_idx * ncj + j_idx) * nci + i_idx;

                // Helper closures for readability
                let g = |axis: usize, base: usize, k_shift: usize| g0[axis * g_alloc + base + k_shift * dk];
                let di = |axis: usize, base: usize, k_shift: usize| g_di[axis * g_alloc + base + k_shift * dk];

                let (s0, s1, s2) = match r_power {
                    2 => {
                        // ip1_r2: g3=+2dk, g4=D_I, g7=D_I+2dk
                        let s0 = di(0, bx, 2) * g(1, by, 0) * g(2, bz, 0)
                            + di(0, bx, 0) * g(1, by, 2) * g(2, bz, 0)
                            + di(0, bx, 0) * g(1, by, 0) * g(2, bz, 2);
                        let s1 = g(0, bx, 2) * di(1, by, 0) * g(2, bz, 0)
                            + g(0, bx, 0) * di(1, by, 2) * g(2, bz, 0)
                            + g(0, bx, 0) * di(1, by, 0) * g(2, bz, 2);
                        let s2 = g(0, bx, 2) * g(1, by, 0) * di(2, bz, 0)
                            + g(0, bx, 0) * g(1, by, 2) * di(2, bz, 0)
                            + g(0, bx, 0) * g(1, by, 0) * di(2, bz, 2);
                        (s0, s1, s2)
                    }
                    4 => {
                        // ip1_r4 from cint3c1e_a.c lines 415-420.
                        let s0 = di(0, bx, 4) * g(1, by, 0) * g(2, bz, 0)
                            + 2.0 * di(0, bx, 2) * g(1, by, 2) * g(2, bz, 0)
                            + 2.0 * di(0, bx, 2) * g(1, by, 0) * g(2, bz, 2)
                            + di(0, bx, 0) * g(1, by, 4) * g(2, bz, 0)
                            + 2.0 * di(0, bx, 0) * g(1, by, 2) * g(2, bz, 2)
                            + di(0, bx, 0) * g(1, by, 0) * g(2, bz, 4);
                        let s1 = g(0, bx, 4) * di(1, by, 0) * g(2, bz, 0)
                            + 2.0 * g(0, bx, 2) * di(1, by, 2) * g(2, bz, 0)
                            + 2.0 * g(0, bx, 2) * di(1, by, 0) * g(2, bz, 2)
                            + g(0, bx, 0) * di(1, by, 4) * g(2, bz, 0)
                            + 2.0 * g(0, bx, 0) * di(1, by, 2) * g(2, bz, 2)
                            + g(0, bx, 0) * di(1, by, 0) * g(2, bz, 4);
                        let s2 = g(0, bx, 4) * g(1, by, 0) * di(2, bz, 0)
                            + 2.0 * g(0, bx, 2) * g(1, by, 2) * di(2, bz, 0)
                            + 2.0 * g(0, bx, 2) * g(1, by, 0) * di(2, bz, 2)
                            + g(0, bx, 0) * g(1, by, 4) * di(2, bz, 0)
                            + 2.0 * g(0, bx, 0) * g(1, by, 2) * di(2, bz, 2)
                            + g(0, bx, 0) * g(1, by, 0) * di(2, bz, 4);
                        (s0, s1, s2)
                    }
                    6 => {
                        // ip1_r6: D_I applied to the r^6 expansion.
                        let s0 = di(0, bx, 6) * g(1, by, 0) * g(2, bz, 0)
                            + 3.0 * di(0, bx, 4) * g(1, by, 2) * g(2, bz, 0)
                            + 3.0 * di(0, bx, 4) * g(1, by, 0) * g(2, bz, 2)
                            + 3.0 * di(0, bx, 2) * g(1, by, 4) * g(2, bz, 0)
                            + 6.0 * di(0, bx, 2) * g(1, by, 2) * g(2, bz, 2)
                            + 3.0 * di(0, bx, 2) * g(1, by, 0) * g(2, bz, 4)
                            + di(0, bx, 0) * g(1, by, 6) * g(2, bz, 0)
                            + 3.0 * di(0, bx, 0) * g(1, by, 4) * g(2, bz, 2)
                            + 3.0 * di(0, bx, 0) * g(1, by, 2) * g(2, bz, 4)
                            + di(0, bx, 0) * g(1, by, 0) * g(2, bz, 6);
                        let s1 = g(0, bx, 6) * di(1, by, 0) * g(2, bz, 0)
                            + 3.0 * g(0, bx, 4) * di(1, by, 2) * g(2, bz, 0)
                            + 3.0 * g(0, bx, 4) * di(1, by, 0) * g(2, bz, 2)
                            + 3.0 * g(0, bx, 2) * di(1, by, 4) * g(2, bz, 0)
                            + 6.0 * g(0, bx, 2) * di(1, by, 2) * g(2, bz, 2)
                            + 3.0 * g(0, bx, 2) * di(1, by, 0) * g(2, bz, 4)
                            + g(0, bx, 0) * di(1, by, 6) * g(2, bz, 0)
                            + 3.0 * g(0, bx, 0) * di(1, by, 4) * g(2, bz, 2)
                            + 3.0 * g(0, bx, 0) * di(1, by, 2) * g(2, bz, 4)
                            + g(0, bx, 0) * di(1, by, 0) * g(2, bz, 6);
                        let s2 = g(0, bx, 6) * g(1, by, 0) * di(2, bz, 0)
                            + 3.0 * g(0, bx, 4) * g(1, by, 2) * di(2, bz, 0)
                            + 3.0 * g(0, bx, 4) * g(1, by, 0) * di(2, bz, 2)
                            + 3.0 * g(0, bx, 2) * g(1, by, 4) * di(2, bz, 0)
                            + 6.0 * g(0, bx, 2) * g(1, by, 2) * di(2, bz, 2)
                            + 3.0 * g(0, bx, 2) * g(1, by, 0) * di(2, bz, 4)
                            + g(0, bx, 0) * g(1, by, 6) * di(2, bz, 0)
                            + 3.0 * g(0, bx, 0) * g(1, by, 4) * di(2, bz, 2)
                            + 3.0 * g(0, bx, 0) * g(1, by, 2) * di(2, bz, 4)
                            + g(0, bx, 0) * g(1, by, 0) * di(2, bz, 6);
                        (s0, s1, s2)
                    }
                    _ => (0.0, 0.0, 0.0),
                };

                out[0 * cart_size + n] += s0;
                out[1 * cart_size + n] += s1;
                out[2 * cart_size + n] += s2;
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Device `#[cube(launch)]` scalar origk kernel (r2/r4/r6) — generic over F.
// ─────────────────────────────────────────────────────────────────────────────

/// VRR in the combined j+k direction on one axis sub-block (stride `dj`),
/// reproducing the host `fill_g_tensor_3c1e_origk` j-loop with explicit base.
/// `disp = -rjrijk[d]`, `aijk1 = 0.5/aijk`, levels `1..=nmax`.
#[cube]
fn origk_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, disp: F, aijk1: F, dj: u32, nmax: u32) {
    if nmax >= 1u32 {
        g[(base + dj) as usize] = disp * g[base as usize];
        let mut j = 1u32;
        while j < nmax {
            g[(base + (j + 1u32) * dj) as usize] =
                aijk1 * F::cast_from(j) * g[(base + (j - 1u32) * dj) as usize]
                    + disp * g[(base + j * dj) as usize];
            j += 1u32;
        }
    }
}

/// HRR for the i-direction on one axis sub-block (i-stride = 1, j-stride = dj),
/// reproducing the host `fill_g_tensor_3c1e_origk` i-loop.
#[cube]
fn origk_hrr_i_axis<F: Float>(g: &mut Array<F>, base: u32, rirj_d: F, dj: u32, li: u32, nmax: u32) {
    let mut i = 1u32;
    while i <= li {
        let j_max = nmax - i;
        let mut j = 0u32;
        while j <= j_max {
            let idx_out = base + i + j * dj;
            let idx_hi = base + (i - 1u32) + (j + 1u32) * dj;
            let idx_lo = base + (i - 1u32) + j * dj;
            g[idx_out as usize] = g[idx_hi as usize] - rirj_d * g[idx_lo as usize];
            j += 1u32;
        }
        i += 1u32;
    }
}

/// HRR for the k-separation on one axis sub-block (k-stride = dk),
/// reproducing the host `fill_g_tensor_3c1e_origk` k-loop.
#[cube]
fn origk_hrr_k_axis<F: Float>(
    g: &mut Array<F>,
    base: u32,
    rjrk_d: F,
    dj: u32,
    dk: u32,
    li: u32,
    lk: u32,
    mmax: u32,
) {
    let mut k = 1u32;
    while k <= lk {
        let j_max = mmax - k;
        let mut j = 0u32;
        while j <= j_max {
            let kb = k * dk + j * dj;
            let mut i = 0u32;
            while i <= li {
                let idx = base + kb + i;
                let idx_hi = idx + dj - dk;
                let idx_lo = idx - dk;
                g[idx as usize] = g[idx_hi as usize] + rjrk_d * g[idx_lo as usize];
                i += 1u32;
            }
            j += 1u32;
        }
        k += 1u32;
    }
}

/// Device scalar origk kernel: per-shell-triple Cartesian block for the
/// `int3c1e_{r2,r4,r6}_origk_sph` operators, generic over `F: Float`.
///
/// Single work item (`UNIT_POS == 0`). Inlines the host pipeline
/// `fill_g_tensor_3c1e_origk` -> `contract_origk`, accumulated over primitive
/// triples (k outer, j, i inner) and contraction triples (ck, cj, ci).
///
/// Output `cart_out` layout matches host `contract_origk`:
///   `out[(k_idx * ncj + j_idx) * nci + i_idx]` per (ci,cj,ck) block,
///   blocks ordered `(ck * nctr_j + cj) * nctr_i + ci`, i fastest.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn origk_scalar_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    exps_k: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    coeff_k: &Array<F>,
    g: &mut Array<F>,
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
    li: u32,
    lj: u32,
    lk: u32,
    li_ceil: u32,
    lk_ceil: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    dli: u32,
    dlj: u32,
    g_alloc: u32,
    #[comptime] r_power: u32,
    #[comptime] _nroots: u32,
) {
    if UNIT_POS == 0u32 {
        // Strides (mirror host: dj = dli, dk = dli*dlj).
        let dj = dli;
        let dk = dli * dlj;
        let nmax = li_ceil + lj + lk_ceil;
        let mmax = lj + lk_ceil;

        let gx = 0u32;
        let gy = g_alloc;
        let gz = 2u32 * g_alloc;
        let total_g = 3u32 * g_alloc;

        // Cartesian component counts (host ncart()).
        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
        let block_len = nci * ncj * nck;
        let out_total = nctr_i * nctr_j * nctr_k * block_len;

        // Zero the accumulation buffer.
        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // ── Primitive loop (k outer, j, i inner) ─────────────────────────────
        let mut kp = 0u32;
        while kp < nprim_k {
            let ak = exps_k[kp as usize];
            let mut jp = 0u32;
            while jp < nprim_j {
                let aj = exps_j[jp as usize];
                let mut ip = 0u32;
                while ip < nprim_i {
                    let ai = exps_i[ip as usize];
                    let aijk = ai + aj + ak;

                    // eijk = (ai*aj*rr_ij + ai*ak*rr_ik + aj*ak*rr_jk) / aijk
                    let rijx = rix - rjx;
                    let rijy = riy - rjy;
                    let rijz = riz - rjz;
                    let rikx = rix - rkx;
                    let riky = riy - rky;
                    let rikz = riz - rkz;
                    let rjkx = rjx - rkx;
                    let rjky = rjy - rky;
                    let rjkz = rjz - rkz;
                    let rr_ij = rijx * rijx + rijy * rijy + rijz * rijz;
                    let rr_ik = rikx * rikx + riky * riky + rikz * rikz;
                    let rr_jk = rjkx * rjkx + rjky * rjky + rjkz * rjkz;
                    let eijk = (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;

                    // expcutoff = 60.0 (host). Skip negligible primitives.
                    let mut skip = false;
                    if eijk > F::new(60.0) {
                        skip = true;
                    }

                    if !skip {
                        let dijk = F::exp(-eijk) / (aijk * F::sqrt(aijk));
                        let fac = common_factor * dijk;

                        // ── fill_g_tensor_3c1e_origk (li_ceil, lj, lk_ceil) ─────
                        let mut gi = 0u32;
                        while gi < total_g {
                            g[gi as usize] = F::new(0.0);
                            gi += 1u32;
                        }
                        g[gx as usize] = F::new(1.0);
                        g[gy as usize] = F::new(1.0);
                        g[gz as usize] = fac;

                        if nmax >= 1u32 {
                            let aijk1 = F::new(0.5) / aijk;

                            // rjrijk[d] = rj[d] - (ai*ri[d]+aj*rj[d]+ak*rk[d])/aijk
                            let px = (ai * rix + aj * rjx + ak * rkx) / aijk;
                            let py = (ai * riy + aj * rjy + ak * rky) / aijk;
                            let pz = (ai * riz + aj * rjz + ak * rkz) / aijk;
                            let dispx = -(rjx - px);
                            let dispy = -(rjy - py);
                            let dispz = -(rjz - pz);

                            // VRR (combined j+k direction) per axis.
                            origk_vrr_axis::<F>(g, gx, dispx, aijk1, dj, nmax);
                            origk_vrr_axis::<F>(g, gy, dispy, aijk1, dj, nmax);
                            origk_vrr_axis::<F>(g, gz, dispz, aijk1, dj, nmax);

                            // HRR for i-direction (rirj = ri - rj) per axis.
                            if li_ceil >= 1u32 {
                                origk_hrr_i_axis::<F>(g, gx, rijx, dj, li_ceil, nmax);
                                origk_hrr_i_axis::<F>(g, gy, rijy, dj, li_ceil, nmax);
                                origk_hrr_i_axis::<F>(g, gz, rijz, dj, li_ceil, nmax);
                            }

                            // HRR for k-separation (rjrk = rj - rk) per axis.
                            if lk_ceil >= 1u32 {
                                origk_hrr_k_axis::<F>(g, gx, rjkx, dj, dk, li_ceil, lk_ceil, mmax);
                                origk_hrr_k_axis::<F>(g, gy, rjky, dj, dk, li_ceil, lk_ceil, mmax);
                                origk_hrr_k_axis::<F>(g, gz, rjkz, dj, dk, li_ceil, lk_ceil, mmax);
                            }
                        }

                        // ── Contract into every (ci,cj,ck) block ───────────────
                        let mut ck = 0u32;
                        while ck < nctr_k {
                            let cval_k = coeff_k[(kp * nctr_k + ck) as usize];
                            let mut cj = 0u32;
                            while cj < nctr_j {
                                let cval_j = coeff_j[(jp * nctr_j + cj) as usize];
                                let mut ci = 0u32;
                                while ci < nctr_i {
                                    let cval_i = coeff_i[(ip * nctr_i + ci) as usize];
                                    let weight = cval_i * cval_j * cval_k;
                                    let base = ((ck * nctr_j + cj) * nctr_i + ci) * block_len;

                                    // Cartesian component triples (k outer, j, i inner),
                                    // matching host cart_comps ordering (lx descending).
                                    let mut k_idx = 0u32;
                                    let mut ka = 0u32;
                                    while ka <= lk {
                                        let kx = lk - ka;
                                        let lk_minus_kx = lk - kx;
                                        let mut kb = 0u32;
                                        while kb <= lk_minus_kx {
                                            let ky = lk_minus_kx - kb;
                                            let kz = lk - kx - ky;

                                            let mut j_idx = 0u32;
                                            let mut ja = 0u32;
                                            while ja <= lj {
                                                let jx = lj - ja;
                                                let lj_minus_jx = lj - jx;
                                                let mut jb = 0u32;
                                                while jb <= lj_minus_jx {
                                                    let jy = lj_minus_jx - jb;
                                                    let jz = lj - jx - jy;

                                                    let mut i_idx = 0u32;
                                                    let mut ia = 0u32;
                                                    while ia <= li {
                                                        let ix = li - ia;
                                                        let li_minus_ix = li - ix;
                                                        let mut ib = 0u32;
                                                        while ib <= li_minus_ix {
                                                            let iy = li_minus_ix - ib;
                                                            let iz = li - ix - iy;

                                                            let bx = ix + jx * dj + kx * dk;
                                                            let by = iy + jy * dj + ky * dk;
                                                            let bz = iz + jz * dj + kz * dk;

                                                            let mut s = F::new(0.0);
                                                            if comptime!(r_power == 2u32) {
                                                                s = g[(gx + bx + 2u32 * dk) as usize] * g[(gy + by) as usize] * g[(gz + bz) as usize]
                                                                    + g[(gx + bx) as usize] * g[(gy + by + 2u32 * dk) as usize] * g[(gz + bz) as usize]
                                                                    + g[(gx + bx) as usize] * g[(gy + by) as usize] * g[(gz + bz + 2u32 * dk) as usize];
                                                            } else if comptime!(r_power == 4u32) {
                                                                s = g[(gx + bx + 4u32 * dk) as usize] * g[(gy + by) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(2.0) * g[(gx + bx + 2u32 * dk) as usize] * g[(gy + by + 2u32 * dk) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(2.0) * g[(gx + bx + 2u32 * dk) as usize] * g[(gy + by) as usize] * g[(gz + bz + 2u32 * dk) as usize]
                                                                    + g[(gx + bx) as usize] * g[(gy + by + 4u32 * dk) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(2.0) * g[(gx + bx) as usize] * g[(gy + by + 2u32 * dk) as usize] * g[(gz + bz + 2u32 * dk) as usize]
                                                                    + g[(gx + bx) as usize] * g[(gy + by) as usize] * g[(gz + bz + 4u32 * dk) as usize];
                                                            } else {
                                                                // r_power == 6
                                                                s = g[(gx + bx + 6u32 * dk) as usize] * g[(gy + by) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(3.0) * g[(gx + bx + 4u32 * dk) as usize] * g[(gy + by + 2u32 * dk) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(3.0) * g[(gx + bx + 4u32 * dk) as usize] * g[(gy + by) as usize] * g[(gz + bz + 2u32 * dk) as usize]
                                                                    + F::new(3.0) * g[(gx + bx + 2u32 * dk) as usize] * g[(gy + by + 4u32 * dk) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(6.0) * g[(gx + bx + 2u32 * dk) as usize] * g[(gy + by + 2u32 * dk) as usize] * g[(gz + bz + 2u32 * dk) as usize]
                                                                    + F::new(3.0) * g[(gx + bx + 2u32 * dk) as usize] * g[(gy + by) as usize] * g[(gz + bz + 4u32 * dk) as usize]
                                                                    + g[(gx + bx) as usize] * g[(gy + by + 6u32 * dk) as usize] * g[(gz + bz) as usize]
                                                                    + F::new(3.0) * g[(gx + bx) as usize] * g[(gy + by + 4u32 * dk) as usize] * g[(gz + bz + 2u32 * dk) as usize]
                                                                    + F::new(3.0) * g[(gx + bx) as usize] * g[(gy + by + 2u32 * dk) as usize] * g[(gz + bz + 4u32 * dk) as usize]
                                                                    + g[(gx + bx) as usize] * g[(gy + by) as usize] * g[(gz + bz + 6u32 * dk) as usize];
                                                            }

                                                            let out_idx = base + (k_idx * ncj + j_idx) * nci + i_idx;
                                                            cart_out[out_idx as usize] =
                                                                cart_out[out_idx as usize] + weight * s;

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
                                    ci += 1u32;
                                }
                                cj += 1u32;
                            }
                            ck += 1u32;
                        }
                    }

                    ip += 1u32;
                }
                jp += 1u32;
            }
            kp += 1u32;
        }
    }
}

/// Device dispatcher: build buffers, monomorphize over (r_power, nroots), launch
/// the scalar origk kernel on runtime `R`, read back the Cartesian block.
#[allow(clippy::too_many_arguments)]
fn run_origk_scalar_device<R: Runtime>(
    client: &ComputeClient<R>,
    r_power: u32,
    nroots: u32,
    li: u32,
    lj: u32,
    lk: u32,
    li_ceil: u32,
    lk_ceil: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    dli: u32,
    dlj: u32,
    g_alloc: u32,
    common_factor: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
) -> Vec<f64> {
    let nci = ((li + 1) * (li + 2) / 2) as usize;
    let ncj = ((lj + 1) * (lj + 2) / 2) as usize;
    let nck = ((lk + 1) * (lk + 2) / 2) as usize;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * (nctr_k as usize) * nci * ncj * nck;
    let g_total = 3 * (g_alloc as usize);

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let exps_k_h = client.create_from_slice(f64::as_bytes(exps_k));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let coeff_k_h = client.create_from_slice(f64::as_bytes(coeff_k));

    let g_zero = vec![0.0_f64; g_total];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($rp:expr, $nr:expr) => {
            origk_scalar_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_k_h.clone(), exps_k.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_k_h.clone(), coeff_k.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), g_total) },
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
                li,
                lj,
                lk,
                li_ceil,
                lk_ceil,
                nprim_i,
                nprim_j,
                nprim_k,
                nctr_i,
                nctr_j,
                nctr_k,
                dli,
                dlj,
                g_alloc,
                $rp,
                $nr,
            )
        };
    }

    // nroots is unused numerically (origk has no Rys roots); the comptime arg is
    // carried for template parity. Monomorphize on r_power only.
    let nr = if nroots == 0 { 1u32 } else { nroots };
    match r_power {
        2 => launch_with!(2u32, nr),
        4 => launch_with!(4u32, nr),
        _ => launch_with!(6u32, nr),
    };

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_origk_scalar_device`] (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_origk_scalar_on_backend(
    backend: &ResolvedBackend,
    r_power: u32,
    nroots: u32,
    li: u32,
    lj: u32,
    lk: u32,
    li_ceil: u32,
    lk_ceil: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    dli: u32,
    dlj: u32,
    g_alloc: u32,
    common_factor: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_origk_scalar_device::<cubecl::cpu::CpuRuntime>(
            client, r_power, nroots, li, lj, lk, li_ceil, lk_ceil, nprim_i, nprim_j, nprim_k,
            nctr_i, nctr_j, nctr_k, dli, dlj, g_alloc, common_factor, ri, rj, rk, exps_i, exps_j,
            exps_k, coeff_i, coeff_j, coeff_k,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_origk_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, r_power, nroots, li, lj, lk, li_ceil, lk_ceil, nprim_i, nprim_j, nprim_k,
            nctr_i, nctr_j, nctr_k, dli, dlj, g_alloc, common_factor, ri, rj, rk, exps_i, exps_j,
            exps_k, coeff_i, coeff_j, coeff_k,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_origk_scalar_device::<cubecl_cuda::CudaRuntime>(
            client, r_power, nroots, li, lj, lk, li_ceil, lk_ceil, nprim_i, nprim_j, nprim_k,
            nctr_i, nctr_j, nctr_k, dli, dlj, g_alloc, common_factor, ri, rj, rk, exps_i, exps_j,
            exps_k, coeff_i, coeff_j, coeff_k,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_origk_scalar_device::<cubecl_hip::HipRuntime>(
            client, r_power, nroots, li, lj, lk, li_ceil, lk_ceil, nprim_i, nprim_j, nprim_k,
            nctr_i, nctr_j, nctr_k, dli, dlj, g_alloc, common_factor, ri, rj, rk, exps_i, exps_j,
            exps_k, coeff_i, coeff_j, coeff_k,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_origk_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, r_power, nroots, li, lj, lk, li_ceil, lk_ceil, nprim_i, nprim_j, nprim_k,
            nctr_i, nctr_j, nctr_k, dli, dlj, g_alloc, common_factor, ri, rj, rk, exps_i, exps_j,
            exps_k, coeff_i, coeff_j, coeff_k,
        ),
    }
}

/// Origk family launcher: dispatches 6 origin-k-displaced 3c1e variants.
///
/// These use the standard 3c1e G-tensor fill (same as center_3c1e) but with
/// elevated ceiling k-angular momentum. The r^n operator is encoded as dk shifts.
///
/// SCALAR r2/r4/r6 variants run the `#[cube(launch)] origk_scalar_kernel` device
/// kernel (dispatched on all 5 backend arms). The ip1 gradient variants keep the
/// existing host path (deferred-to-host, see module doc).
pub fn launch_origk(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    let op_name = plan.descriptor.operator_name();
    let variant = origk_variant(op_name)?;

    let shells = plan.shells.as_slice();
    if shells.len() < 3 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_origk",
            detail: "origk requires 3 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);

    // Ceiling angular momenta
    let li_ceil = li as u32 + variant.i_inc as u32;
    let lk_ceil = lk as u32 + variant.k_inc as u32;

    let dli = (li_ceil + 1) as usize;
    let dlj = (lj as usize) + (lk_ceil as usize) + 1;
    let dlk = (lk_ceil + 1) as usize;

    let nmax = (li_ceil + lj as u32 + lk_ceil) as usize;
    let vrr_nmax = dli + (lj as usize) + (lk_ceil as usize);
    let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);

    let dk = dli * dlj;

    let common_factor = SQRTPI * PI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
    let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
    let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
    let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

    let expcutoff = 60.0_f64;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    let cart_buf: Vec<f64> = if variant.ncomp == 1 {
        // ── SCALAR r2/r4/r6: device path. Fail-closed nmax guard. ────────────
        if nmax > MAX_DEVICE_NROOTS * 2 + 1 {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_origk",
                detail: format!(
                    "origk device kernel nmax {nmax} exceeds MAX_DEVICE_NROOTS-derived bound"
                ),
            });
        }

        run_origk_scalar_on_backend(
            backend,
            variant.k_inc as u32,
            1, // nroots unused
            li as u32,
            lj as u32,
            lk as u32,
            li_ceil,
            lk_ceil,
            n_prim_i as u32,
            n_prim_j as u32,
            n_prim_k as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            n_ctr_k as u32,
            dli as u32,
            dlj as u32,
            g_alloc as u32,
            common_factor,
            ri,
            rj,
            rk,
            &shell_i.exponents,
            &shell_j.exponents,
            &shell_k.exponents,
            &shell_i.coefficients,
            &shell_j.coefficients,
            &shell_k.coefficients,
        )
    } else {
        // ── ip1 gradient variants: host path (deferred-to-host). ─────────────
        let mut cart_buf = vec![0.0_f64; nci * ncj * nck * variant.ncomp];
        for kp in 0..n_prim_k {
            let ak = shell_k.exponents[kp];
            for jp in 0..n_prim_j {
                let aj = shell_j.exponents[jp];
                for ip in 0..n_prim_i {
                    let ai = shell_i.exponents[ip];
                    let aijk = ai + aj + ak;
                    let eijk = (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;
                    if eijk > expcutoff {
                        continue;
                    }

                    let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());
                    let fac = common_factor * dijk;

                    let g = fill_g_tensor_3c1e_origk(
                        fac, ai, aj, ak, ri, rj, rk, rirj,
                        li_ceil, lj as u32, lk_ceil,
                        dli, dlj, g_alloc,
                    );

                    let prim_buf =
                        contract_origk_ip1(&g, g_alloc, li, lj, lk, dli, dlj, dk, variant.k_inc, ai);

                    for ck in 0..n_ctr_k {
                        let coeff_k = shell_k.coefficients[kp * n_ctr_k + ck];
                        for cj in 0..n_ctr_j {
                            let coeff_j = shell_j.coefficients[jp * n_ctr_j + cj];
                            for ci in 0..n_ctr_i {
                                let coeff_i = shell_i.coefficients[ip * n_ctr_i + ci];
                                let weight = coeff_i * coeff_j * coeff_k;
                                for idx in 0..prim_buf.len() {
                                    cart_buf[idx] += weight * prim_buf[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        cart_buf
    };

    // Apply c2s transform (host part of the split).
    if variant.ncomp == 1 {
        match plan.representation {
            Representation::Spheric => {
                let sph = cart_to_sph_3c1e(&cart_buf, li, lj, lk);
                let sph_size = nsi * nsj * nsk;
                let copy_len = staging.len().min(sph.len()).min(sph_size);
                staging[..copy_len].copy_from_slice(&sph[..copy_len]);
            }
            _ => {
                let copy_len = staging.len().min(cart_buf.len());
                staging[..copy_len].copy_from_slice(&cart_buf[..copy_len]);
            }
        }
    } else {
        // ncomp > 1: c2s each component
        let cart_size = nci * ncj * nck;
        let sph_size = nsi * nsj * nsk;
        match plan.representation {
            Representation::Spheric => {
                for comp in 0..variant.ncomp {
                    let cart_slice = &cart_buf[comp * cart_size..(comp + 1) * cart_size];
                    let sph = cart_to_sph_3c1e(cart_slice, li, lj, lk);
                    let sph_off = comp * sph_size;
                    let copy_len = staging.len().saturating_sub(sph_off).min(sph.len()).min(sph_size);
                    if copy_len > 0 {
                        staging[sph_off..sph_off + copy_len].copy_from_slice(&sph[..copy_len]);
                    }
                }
            }
            _ => {
                let copy_len = staging.len().min(cart_buf.len());
                staging[..copy_len].copy_from_slice(&cart_buf[..copy_len]);
            }
        }
    }

    Ok(make_exec_stats(plan, staging))
}

/// Fill G-tensor for 3c1e origk with elevated ceiling k-angular momentum.
/// Identical to center_3c1e::fill_g_tensor_3c1e but parameterized with ceiling values.
fn fill_g_tensor_3c1e_origk(
    fac: f64,
    ai: f64,
    aj: f64,
    ak: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rirj: [f64; 3],
    li: u32,
    lj: u32,
    lk: u32,
    dli: usize,
    dlj: usize,
    g_alloc: usize,
) -> Vec<f64> {
    let nmax = (li + lj + lk) as usize;
    let mmax = (lj + lk) as usize;

    let mut g = vec![0.0_f64; 3 * g_alloc];

    let aijk = ai + aj + ak;
    let aijk1 = 0.5_f64 / aijk;

    g[0] = 1.0;
    g[g_alloc] = 1.0;
    g[2 * g_alloc] = fac;

    if nmax == 0 {
        return g;
    }

    let dj_local = dli;

    let rjrijk = [
        rj[0] - (ai * ri[0] + aj * rj[0] + ak * rk[0]) / aijk,
        rj[1] - (ai * ri[1] + aj * rj[1] + ak * rk[1]) / aijk,
        rj[2] - (ai * ri[2] + aj * rj[2] + ak * rk[2]) / aijk,
    ];

    // VRR: combined j+k direction
    for d in 0..3 {
        let off = d * g_alloc;
        let disp = -rjrijk[d];
        g[off + dj_local] = disp * g[off];
        let mut j = 1usize;
        while j < nmax {
            g[off + (j + 1) * dj_local] =
                aijk1 * j as f64 * g[off + (j - 1) * dj_local] + disp * g[off + j * dj_local];
            j += 1;
        }
    }

    // HRR for i-direction
    for d in 0..3 {
        let off = d * g_alloc;
        let rirj_d = rirj[d];
        for i in 1..=(li as usize) {
            let j_max = nmax - i;
            for j in 0..=j_max {
                let idx_out = i + j * dj_local;
                let idx_hi = (i - 1) + (j + 1) * dj_local;
                let idx_lo = (i - 1) + j * dj_local;
                g[off + idx_out] = g[off + idx_hi] - rirj_d * g[off + idx_lo];
            }
        }
    }

    // HRR for k-separation
    let dk = dli * dlj;
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];

    for d in 0..3 {
        let off = d * g_alloc;
        let rjrk_d = rjrk[d];
        for k in 1..=(lk as usize) {
            for j in 0..=(mmax - k) {
                let base = k * dk + j * dj_local;
                for i in 0..=li as usize {
                    let idx = base + i;
                    let idx_hi = idx + dj_local - dk;
                    let idx_lo = idx - dk;
                    g[off + idx] = g[off + idx_hi] + rjrk_d * g[off + idx_lo];
                }
            }
        }
    }

    g
}

// ─────────────────────────────────────────────────────────────────────────────
// In-crate device-vs-host + f32 genericity tests (CpuRuntime, feature = "cpu").
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host reference: single-primitive, single-contraction origk Cartesian block
    /// via fill_g_tensor_3c1e_origk -> contract_origk for the given r_power.
    fn host_origk_block(
        r_power: u8,
        ai: f64,
        aj: f64,
        ak: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rk: [f64; 3],
        li: u8,
        lj: u8,
        lk: u8,
    ) -> Vec<f64> {
        let li_ceil = li as u32;
        let lk_ceil = lk as u32 + r_power as u32;
        let dli = (li_ceil + 1) as usize;
        let dlj = (lj as usize) + (lk_ceil as usize) + 1;
        let dlk = (lk_ceil + 1) as usize;
        let vrr_nmax = dli + (lj as usize) + (lk_ceil as usize);
        let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);
        let dk = dli * dlj;

        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
        let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
        let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
        let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
        let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

        let common_factor =
            SQRTPI * PI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

        let aijk = ai + aj + ak;
        let eijk = (ai * aj * rr_ij + ai * ak * rr_ik + aj * ak * rr_jk) / aijk;
        let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());
        let fac = common_factor * dijk;

        let g = fill_g_tensor_3c1e_origk(
            fac, ai, aj, ak, ri, rj, rk, rirj, li_ceil, lj as u32, lk_ceil, dli, dlj, g_alloc,
        );
        contract_origk(&g, g_alloc, li, lj, lk, dli, dlj, dk, r_power)
    }

    fn assert_close(host: &[f64], dev: &[f64], tag: &str) {
        assert_eq!(host.len(), dev.len(), "length mismatch ({tag})");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch ({tag}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    #[test]
    fn test_device_matches_host_origk() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let rk = [0.2_f64, -0.4, 0.3];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        let ak = 0.7_f64;

        for &(li, lj, lk) in &[
            (0u8, 0u8, 0u8),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
        ] {
            for &r_power in &[2u8, 4u8, 6u8] {
                let host = host_origk_block(r_power, ai, aj, ak, ri, rj, rk, li, lj, lk);

                let li_ceil = li as u32;
                let lk_ceil = lk as u32 + r_power as u32;
                let dli = (li_ceil + 1) as usize;
                let dlj = (lj as usize) + (lk_ceil as usize) + 1;
                let dlk = (lk_ceil + 1) as usize;
                let vrr_nmax = dli + (lj as usize) + (lk_ceil as usize);
                let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);
                let common_factor =
                    SQRTPI * PI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

                let dev = run_origk_scalar_device::<cubecl::cpu::CpuRuntime>(
                    &cpu_client(),
                    r_power as u32,
                    1,
                    li as u32,
                    lj as u32,
                    lk as u32,
                    li_ceil,
                    lk_ceil,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    dli as u32,
                    dlj as u32,
                    g_alloc as u32,
                    common_factor,
                    ri,
                    rj,
                    rk,
                    &[ai],
                    &[aj],
                    &[ak],
                    &[1.0],
                    &[1.0],
                    &[1.0],
                );
                assert_close(&host, &dev, &format!("origk r{r_power} li={li} lj={lj} lk={lk}"));
            }
        }
    }

    /// Genericity: the kernel launches for F=f32 — an s-s-s r2 block on
    /// CpuRuntime returns a finite output.
    #[test]
    fn test_origk_scalar_kernel_generic_f32() {
        let client = cpu_client();
        // s-s-s, r_power=2: li_ceil=0, lk_ceil=2, dli=1, dlj=3, dlk=3.
        let li_ceil = 0u32;
        let lk_ceil = 2u32;
        let dli = 1usize;
        let dlj = 3usize;
        let dlk = 3usize;
        let vrr_nmax = dli + 0 + (lk_ceil as usize);
        let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);
        let g_total = 3 * g_alloc;
        let out_len = 1usize;

        let exps_i = [0.9_f32];
        let exps_j = [1.3_f32];
        let exps_k = [0.7_f32];
        let coeff = [1.0_f32];
        let g_zero = vec![0.0_f32; g_total];
        let out_zero = vec![0.0_f32; out_len];

        let ei_h = client.create_from_slice(f32::as_bytes(&exps_i));
        let ej_h = client.create_from_slice(f32::as_bytes(&exps_j));
        let ek_h = client.create_from_slice(f32::as_bytes(&exps_k));
        let ci_h = client.create_from_slice(f32::as_bytes(&coeff));
        let cj_h = client.create_from_slice(f32::as_bytes(&coeff));
        let ck_h = client.create_from_slice(f32::as_bytes(&coeff));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let common_factor = (SQRTPI * PI * common_fac_sp(0) * common_fac_sp(0) * common_fac_sp(0)) as f32;

        origk_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(ei_h, 1) },
            unsafe { ArrayArg::from_raw_parts(ej_h, 1) },
            unsafe { ArrayArg::from_raw_parts(ek_h, 1) },
            unsafe { ArrayArg::from_raw_parts(ci_h, 1) },
            unsafe { ArrayArg::from_raw_parts(cj_h, 1) },
            unsafe { ArrayArg::from_raw_parts(ck_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, g_total) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
            0.0_f32,
            0.0,
            0.0,
            0.6,
            0.5,
            0.7,
            0.2,
            -0.4,
            0.3,
            common_factor,
            0, // li
            0, // lj
            0, // lk
            li_ceil,
            lk_ceil,
            1,
            1,
            1,
            1,
            1,
            1,
            dli as u32,
            dlj as u32,
            g_alloc as u32,
            2u32, // r_power
            1u32, // nroots (unused)
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 origk r2 s-s-s result must be finite: {out}");
    }
}
