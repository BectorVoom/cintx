//! Phase 29 Wave 1 — 1e Group-4 relativistic σ-operator families (REL-01/02).
//!
//! Wires the seven `int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor`
//! launcher arms onto the correct per-family host transform. Each launcher:
//!   1. builds the family Cartesian G-tensor on-device (overlap base for
//!      sigma/sr/srsr/spsp; Rys nuclear base for spnucsp/sprinvsp/srnucsr),
//!   2. emits the four component-leading `gc_x/gc_y/gc_z/gc_1` cart blocks the
//!      host spin-included transform reads in order (one 4-block group per
//!      σ-tensor component — 1 group for all families except `sigma`, which
//!      emits 3 groups, one per Pauli matrix σ_x/σ_y/σ_z),
//!   3. folds each group through the per-family transform and scatters the
//!      `di*dj*2` spinor sub-block into the contraction-major AO grid.
//!
//! Per-family transform map (RESEARCH §Per-Family gout & Transform Map,
//! transcribed from `autocode/intor3.c`):
//!   spsp                              → `cart_to_spinor_sf_2d`  (c2s_sf_1e, scalar ∇²)
//!   spnucsp/sprinvsp/srsr/srnucsr     → `cart_to_spinor_si_2d`  (c2s_si_1e,  real bra-σ)
//!   sr/sigma                          → `cart_to_spinor_si_2di` (c2s_si_1ei, imaginary ket)
//!
//! `int1e_sigma` is **component_rank 3** (empirically measured in
//! `rel_1e_sigma_parity::test_sigma_rank_measured`): `CINT1e_spinor_drv` loops
//! the c2s transform `ng[7]=3` times (cint1e.c:269), producing the three stacked
//! Pauli σ-matrices. Every other family is component_rank 1 (the σ-axis folds
//! into the transform).

use crate::backend::ResolvedBackend;
use crate::transform::c2s::ncart;
use crate::transform::c2spinor::{
    cart_to_spinor_sf_2d, cart_to_spinor_si_2d, cart_to_spinor_si_2di, spinor_len,
};
use cintx_core::CintFloat;
use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// sqrt(pi) — G-tensor base-case normalization (matches `g1e.c` `SQRTPI`).
const SQRTPI: f64 = 1.7724538509055159_f64;

/// libcint `CINTcommon_fac_sp` s/p normalization factor (matches
/// `one_electron::common_fac_sp` / `sigma_p::common_fac_sp`).
pub(crate) fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64, // 1/(2*sqrt(pi))
        1 => 0.488602511902919921_f64, // sqrt(3/(4*pi))
        _ => 1.0,
    }
}

/// Number of cart blocks per σ-tensor component group: `gc_x, gc_y, gc_z, gc_1`.
const N_GC: u32 = 4;

// ─────────────────────────────────────────────────────────────────────────────
//  Family selector (comptime). Drives the gout assembly inside the kernels.
//
//   0 = sigma   (overlap; 3 σ-groups, each [gc_x|gc_y|gc_z|0] = [-s0 on diag])
//   1 = sr      (overlap + R_I bra; gout -s0,-s1,-s2,0)
//   2 = srsr    (overlap + R_J/R_I composed; gout s5-s7,s6-s2,s1-s3,s0+s4+s8)
//   3 = spsp    (overlap + D_J/D_I composed; gout scalar s0+s4+s8 → gc_1 only)
//   (nuclear families 4/5/6 use the Rys kernel below)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub(crate) fn family_id(op: &str) -> Option<u32> {
    match op {
        "sigma" => Some(0),
        "sr" => Some(1),
        "srsr" => Some(2),
        "spsp" => Some(3),
        "spnucsp" => Some(10),
        "srnucsr" => Some(11),
        "sprinvsp" => Some(12),
        _ => None,
    }
}

/// Number of σ-tensor component groups (output spinor matrices). 3 for `sigma`,
/// 1 otherwise.
fn family_rank(op: &str) -> usize {
    if op == "sigma" { 3 } else { 1 }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-axis overlap VRR / HRR helpers (shared with sigma_p; duplicated to keep
//  this module self-contained).
// ─────────────────────────────────────────────────────────────────────────────

#[cube]
fn ov_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, rijrx: F, aij2: F, nmax: u32) {
    if nmax >= 1u32 {
        g[(base + 1u32) as usize] = rijrx * g[base as usize];
        let mut n = 1u32;
        while n < nmax {
            g[(base + n + 1u32) as usize] = F::cast_from(n) * aij2 * g[(base + n - 1u32) as usize]
                + rijrx * g[(base + n) as usize];
            n += 1u32;
        }
    }
}

#[cube]
fn ov_hrr_axis<F: Float>(g: &mut Array<F>, base: u32, rirj: F, dj: u32, li_max: u32, lj: u32) {
    let mut j = 1u32;
    while j <= lj {
        let i_max = li_max - j;
        let mut i = 0u32;
        while i <= i_max {
            let idx_out = base + j * dj + i;
            let idx_hi = base + (j - 1u32) * dj + (i + 1u32);
            let idx_lo = base + (j - 1u32) * dj + i;
            g[idx_out as usize] = g[idx_hi as usize] + rirj * g[idx_lo as usize];
            i += 1u32;
        }
        j += 1u32;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Overlap-engine σ kernel (sigma/sr/srsr/spsp). Single UNIT_POS==0 work item.
//
//  Builds the overlap G-tensor `g` with headroom: i up to li+1, j up to lj+1
//  (so the R_I/R_J/D_I/D_J index shifts read in-range). nmax = li+lj+2,
//  lj_ext = lj+1. Output is `rank * N_GC`-block component-leading per (ci,cj):
//  group `grp` (0..rank), block `b` (0..4) at base + (grp*N_GC + b)*block_len.
// ─────────────────────────────────────────────────────────────────────────────

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sigma_ov_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    gc_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    sqrtpi: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    #[comptime] family: u32,
) {
    if UNIT_POS == 0u32 {
        // Headroom: +2 in i (D_I/R_I needs i+1; composed needs +2), +1 in j.
        let nmax = li + lj + 2u32;
        let lj_ext = lj + 1u32;
        let dj = nmax + 1u32;
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let rank = if comptime!(family == 0u32) {
            3u32
        } else {
            1u32
        };
        let total_len = rank * N_GC * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        let mut oi = 0u32;
        while oi < out_total {
            gc_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                ov_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                ov_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                ov_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                if lj_ext >= 1u32 {
                    ov_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    ov_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    ov_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                }

                let ai2 = F::new(-2.0) * ai;
                let aj2 = F::new(-2.0) * aj;

                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;
                        let base = (ci * nctr_j + cj) * total_len;

                        let mut cj_idx = 0u32;
                        let mut ja = 0u32;
                        while ja <= lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            let mut jb = 0u32;
                            while jb <= lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

                                let mut ci_idx = 0u32;
                                let mut ia = 0u32;
                                while ia <= li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    let mut ib = 0u32;
                                    while ib <= li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        // Base index of (ix,iy,iz | jx,jy,jz) per axis.
                                        let bx = jx * dj + ix;
                                        let by = jy * dj + iy;
                                        let bz = jz * dj + iz;

                                        let g0x = g[(gx + bx) as usize];
                                        let g0y = g[(gy + by) as usize];
                                        let g0z = g[(gz + bz) as usize];

                                        let elem = cj_idx * nci + ci_idx;

                                        if comptime!(family == 0u32) {
                                            // sigma: s0 = g0x*g0y*g0z; 3 groups, each
                                            // [-s0 on its diag block, 0 elsewhere].
                                            let s0 = g0x * g0y * g0z;
                                            let v = weight * (-s0);
                                            // group 0 (σ_x): gc_x = -s0
                                            let g0base = base + 0u32 * N_GC * block_len;
                                            gc_out[(g0base + elem) as usize] += v;
                                            // group 1 (σ_y): gc_y = -s0
                                            let g1base = base + 1u32 * N_GC * block_len;
                                            gc_out[(g1base + block_len + elem) as usize] += v;
                                            // group 2 (σ_z): gc_z = -s0
                                            let g2base = base + 2u32 * N_GC * block_len;
                                            gc_out[(g2base + 2u32 * block_len + elem) as usize] +=
                                                v;
                                        } else if comptime!(family == 1u32) {
                                            // sr: g1 = R_I(g0) → read at i+1.
                                            let g1x = g[(gx + bx + 1u32) as usize];
                                            let g1y = g[(gy + by + 1u32) as usize];
                                            let g1z = g[(gz + bz + 1u32) as usize];
                                            let s0 = g1x * g0y * g0z;
                                            let s1 = g0x * g1y * g0z;
                                            let s2 = g0x * g0y * g1z;
                                            // gout: -s0,-s1,-s2,0 → gc_x,gc_y,gc_z,gc_1
                                            gc_out[(base + elem) as usize] += weight * (-s0);
                                            gc_out[(base + block_len + elem) as usize] +=
                                                weight * (-s1);
                                            gc_out[(base + 2u32 * block_len + elem) as usize] +=
                                                weight * (-s2);
                                            // gc_1 stays 0.
                                        } else if comptime!(family == 2u32) {
                                            // srsr: g1=R_J(g0), g2=R_I(g0), g3=R_I(g1).
                                            //   R_J read at +dj, R_I read at +1.
                                            // g0[n]   = g[b]
                                            // g1[n]   = g[b+dj]       (R_J)
                                            // g2[n]   = g[b+1]        (R_I)
                                            // g3[n]   = g[b+dj+1]     (R_I of R_J)
                                            let s_idx_x0 = gx + bx;
                                            let s_idx_y0 = gy + by;
                                            let s_idx_z0 = gz + bz;
                                            sigma_srsr_accum::<F>(
                                                g, gc_out, base, block_len, elem, weight, dj,
                                                s_idx_x0, s_idx_y0, s_idx_z0,
                                            );
                                        } else {
                                            // spsp: g1=D_J(g0), g2=D_I(g0), g3=D_I(g1).
                                            // gout scalar s0+s4+s8 → gc_1 (block 3).
                                            sigma_spsp_accum::<F>(
                                                g,
                                                gc_out,
                                                base,
                                                block_len,
                                                elem,
                                                weight,
                                                dj,
                                                ai2,
                                                aj2,
                                                gx + bx,
                                                gy + by,
                                                gz + bz,
                                                ix,
                                                iy,
                                                iz,
                                                jx,
                                                jy,
                                                jz,
                                            );
                                        }

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }
                                cj_idx += 1u32;
                                jb += 1u32;
                            }
                            ja += 1u32;
                        }
                        cj += 1u32;
                    }
                    ci += 1u32;
                }
                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// srsr per-cart accumulation: composes R_J then R_I on each axis via index
/// shifts, forms the 9 `s[..]` products, and writes the rank-4 gout
/// `(s5-s7, s6-s2, s1-s3, s0+s4+s8)` into gc_x/gc_y/gc_z/gc_1.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sigma_srsr_accum<F: Float>(
    g: &Array<F>,
    gc_out: &mut Array<F>,
    base: u32,
    block_len: u32,
    elem: u32,
    weight: F,
    dj: u32,
    ix0: u32,
    iy0: u32,
    iz0: u32,
) {
    // Per axis: g0 = g[b], g1 = R_J = g[b+dj], g2 = R_I = g[b+1], g3 = g[b+dj+1].
    let g0x = g[ix0 as usize];
    let g1x = g[(ix0 + dj) as usize];
    let g2x = g[(ix0 + 1u32) as usize];
    let g3x = g[(ix0 + dj + 1u32) as usize];
    let g0y = g[iy0 as usize];
    let g1y = g[(iy0 + dj) as usize];
    let g2y = g[(iy0 + 1u32) as usize];
    let g3y = g[(iy0 + dj + 1u32) as usize];
    let g0z = g[iz0 as usize];
    let g1z = g[(iz0 + dj) as usize];
    let g2z = g[(iz0 + 1u32) as usize];
    let g3z = g[(iz0 + dj + 1u32) as usize];

    let s0 = g3x * g0y * g0z;
    let s1 = g2x * g1y * g0z;
    let s2 = g2x * g0y * g1z;
    let s3 = g1x * g2y * g0z;
    let s4 = g0x * g3y * g0z;
    let s5 = g0x * g2y * g1z;
    let s6 = g1x * g0y * g2z;
    let s7 = g0x * g1y * g2z;
    let s8 = g0x * g0y * g3z;

    gc_out[(base + elem) as usize] += weight * (s5 - s7);
    gc_out[(base + block_len + elem) as usize] += weight * (s6 - s2);
    gc_out[(base + 2u32 * block_len + elem) as usize] += weight * (s1 - s3);
    gc_out[(base + 3u32 * block_len + elem) as usize] += weight * (s0 + s4 + s8);
}

/// spsp per-cart accumulation: composes D_J then D_I (nabla) on each axis, forms
/// the 9 products, and writes the scalar gout `s0+s4+s8` into gc_1 (block 3).
/// gc_x/gc_y/gc_z stay zero (the SF transform reads only gc_1).
#[cube]
#[allow(clippy::too_many_arguments)]
fn sigma_spsp_accum<F: Float>(
    g: &Array<F>,
    gc_out: &mut Array<F>,
    base: u32,
    block_len: u32,
    elem: u32,
    weight: F,
    dj: u32,
    ai2: F,
    aj2: F,
    ix0: u32,
    iy0: u32,
    iz0: u32,
    ixp: u32,
    iyp: u32,
    izp: u32,
    jxp: u32,
    jyp: u32,
    jzp: u32,
) {
    // g1 = D_J(g0): f[i] = j*g[i-dj] + aj2*g[i+dj]  (j-direction nabla).
    // g2 = D_I(g0): f[i] = i*g[i-1]  + ai2*g[i+1]   (i-direction nabla).
    // g3 = D_I(g1).
    let g1x = nabla_j::<F>(g, ix0, dj, aj2, jxp);
    let g2x = nabla_i::<F>(g, ix0, ai2, ixp);
    let g3x = nabla_i_of_j::<F>(g, ix0, dj, ai2, aj2, ixp, jxp);
    let g1y = nabla_j::<F>(g, iy0, dj, aj2, jyp);
    let g2y = nabla_i::<F>(g, iy0, ai2, iyp);
    let g3y = nabla_i_of_j::<F>(g, iy0, dj, ai2, aj2, iyp, jyp);
    let g1z = nabla_j::<F>(g, iz0, dj, aj2, jzp);
    let g2z = nabla_i::<F>(g, iz0, ai2, izp);
    let g3z = nabla_i_of_j::<F>(g, iz0, dj, ai2, aj2, izp, jzp);

    let g0x = g[ix0 as usize];
    let g0y = g[iy0 as usize];
    let g0z = g[iz0 as usize];

    let s0 = g3x * g0y * g0z;
    let s4 = g0x * g3y * g0z;
    let s8 = g0x * g0y * g3z;
    // suppress unused warnings for g1/g2 (only s0,s4,s8 used for the scalar gout)
    let _ = (g1x, g2x, g1y, g2y, g1z, g2z);

    // spsp gout: scalar s0+s4+s8 → gc_1 (block 3). gc_x/gc_y/gc_z = 0.
    gc_out[(base + 3u32 * block_len + elem) as usize] += weight * (s0 + s4 + s8);
}

/// D_J(g) at index `idx0` (cart j-exponent `jexp`): j*g[idx0-dj] + aj2*g[idx0+dj].
#[cube]
fn nabla_j<F: Float>(g: &Array<F>, idx0: u32, dj: u32, aj2: F, jexp: u32) -> F {
    if jexp == 0u32 {
        aj2 * g[(idx0 + dj) as usize]
    } else {
        F::cast_from(jexp) * g[(idx0 - dj) as usize] + aj2 * g[(idx0 + dj) as usize]
    }
}

/// D_I(g) at index `idx0` (cart i-exponent `iexp`): i*g[idx0-1] + ai2*g[idx0+1].
#[cube]
fn nabla_i<F: Float>(g: &Array<F>, idx0: u32, ai2: F, iexp: u32) -> F {
    if iexp == 0u32 {
        ai2 * g[(idx0 + 1u32) as usize]
    } else {
        F::cast_from(iexp) * g[(idx0 - 1u32) as usize] + ai2 * g[(idx0 + 1u32) as usize]
    }
}

/// D_I(D_J(g)) at index `idx0`. Applies D_J first (forms g1 at the i-stencil
/// points idx0-1, idx0+1), then D_I across them. The j-exponent `jexp` is fixed
/// (D_I does not change the j-order); the D_J stencil uses idx±dj.
#[cube]
fn nabla_i_of_j<F: Float>(
    g: &Array<F>,
    idx0: u32,
    dj: u32,
    ai2: F,
    aj2: F,
    iexp: u32,
    jexp: u32,
) -> F {
    // g1 = D_J(g) sampled at the i-shifted points needed by D_I.
    if iexp == 0u32 {
        let g1_ip = nabla_j::<F>(g, idx0 + 1u32, dj, aj2, jexp);
        ai2 * g1_ip
    } else {
        let g1_im = nabla_j::<F>(g, idx0 - 1u32, dj, aj2, jexp);
        let g1_ip = nabla_j::<F>(g, idx0 + 1u32, dj, aj2, jexp);
        F::cast_from(iexp) * g1_im + ai2 * g1_ip
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device dispatch for the overlap-engine kernel.
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_sigma_ov_device<R: Runtime>(
    client: &ComputeClient<R>,
    family: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax_u = li_u + lj_u + 2;
    let lj_ext_u = lj_u + 1;
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let rank = if family == 0 { 3 } else { 1 };
    let out_len = (nctr_i as usize) * (nctr_j as usize) * rank * (N_GC as usize) * nci * ncj;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    // Scratch + output buffers: allocate directly on device via client.empty.
    let g_h = client.empty(total_g * std::mem::size_of::<f64>());
    let out_h = client.empty(out_len * std::mem::size_of::<f64>());

    // SAFETY: Input and scratch buffer lengths match exact dimensions.
    // In-kernel loops strictly bound indices to valid array ranges.
    macro_rules! launch_with {
        ($fam:expr) => {
            unsafe {
                sigma_ov_kernel::launch_unchecked::<f64, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()),
                    ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()),
                    ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()),
                    ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()),
                    ArrayArg::from_raw_parts(g_h.clone(), total_g),
                    ArrayArg::from_raw_parts(out_h.clone(), out_len),
                    ri[0],
                    ri[1],
                    ri[2],
                    rj[0],
                    rj[1],
                    rj[2],
                    SQRTPI,
                    std::f64::consts::PI,
                    li,
                    lj,
                    nprim_i,
                    nprim_j,
                    nctr_i,
                    nctr_j,
                    $fam,
                );
            }
        };
    }

    match family {
        0 => launch_with!(0u32),
        1 => launch_with!(1u32),
        2 => launch_with!(2u32),
        _ => launch_with!(3u32),
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

#[allow(clippy::too_many_arguments)]
fn run_sigma_ov_on_backend(
    backend: &ResolvedBackend,
    family: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
    let out = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_sigma_ov_device::<cubecl::cpu::CpuRuntime>(
            client, family, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_sigma_ov_device::<cubecl_wgpu::WgpuRuntime>(
            client, family, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_sigma_ov_device::<cubecl_cuda::CudaRuntime>(
            client, family, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_sigma_ov_device::<cubecl_hip::HipRuntime>(
            client, family, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_sigma_ov_device::<cubecl_wgpu::WgpuRuntime>(
            client, family, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
    };
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public per-family launcher (overlap + nuclear). Folds each σ-component group
//  through the per-family transform and scatters into the contraction-major grid.
// ─────────────────────────────────────────────────────────────────────────────

/// Transform selector per family (RESEARCH §Per-Family Map).
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum TransformKind {
    /// spsp → c2s_sf_1e (scalar, reads gc_1 only).
    Sf,
    /// spnucsp/sprinvsp/srsr/srnucsr → c2s_si_1e (real bra-σ-mix).
    Si,
    /// sr/sigma → c2s_si_1ei (imaginary-ket bra-σ-mix).
    SiI,
}

fn family_transform(op: &str) -> TransformKind {
    match op {
        "spsp" => TransformKind::Sf,
        "sr" | "sigma" => TransformKind::SiI,
        _ => TransformKind::Si, // spnucsp/sprinvsp/srsr/srnucsr
    }
}

/// Fold one component group's four gc blocks through the family transform into
/// the `di*dj*2` scratch (column-major bra-fastest).
#[allow(clippy::too_many_arguments)]
fn fold_group<F: CintFloat>(
    kind: TransformKind,
    scratch: &mut [F],
    gc_x: &[f64],
    gc_y: &[f64],
    gc_z: &[f64],
    gc_1: &[f64],
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
) -> Result<(), cintxRsError> {
    match kind {
        TransformKind::Sf => {
            // SF transform reads a single BRA-major cart block (cart[i*ncj+j]),
            // but the device kernel emits KET-major (gc_1[j*nci+i]). Unlike si_2d/
            // si_2di, cart_to_spinor_sf_2d does NOT own the KET→BRA transpose, so
            // transpose the scalar block here before the SF fold.
            let nci = ncart(li);
            let ncj = ncart(lj);
            let mut bra_major = vec![0.0f64; nci * ncj];
            for i in 0..nci {
                for j in 0..ncj {
                    bra_major[i * ncj + j] = gc_1[j * nci + i];
                }
            }
            cart_to_spinor_sf_2d::<F>(scratch, &bra_major, li, kappa_i, lj, kappa_j)
        }
        TransformKind::Si => {
            cart_to_spinor_si_2d::<F>(scratch, gc_x, gc_y, gc_z, gc_1, li, kappa_i, lj, kappa_j)
        }
        TransformKind::SiI => {
            cart_to_spinor_si_2di::<F>(scratch, gc_x, gc_y, gc_z, gc_1, li, kappa_i, lj, kappa_j)
        }
    }
}

/// Unified 1e Group-4 σ Spinor launcher for one shell pair `(i, j)`.
///
/// `op` is the operator name without the `int1e_`/`_spinor` affixes
/// (`spsp`/`spnucsp`/`sprinvsp`/`srsr`/`srnucsr`/`sr`/`sigma`). Output layout is
/// interleaved-complex, column-major (ket-spinor outer, bra-spinor inner),
/// contraction-major, with `component_rank` σ-matrices stacked
/// (`sigma`→3, all others→1):
/// `out[grp*(ni_sp*nj_sp*2) + (j_global*ni_sp + i_global)*2 + {0:re,1:im}]`.
///
/// Each call applies its OWN fail-closed staging guard BEFORE any write
/// (Phase-28 CR-01: inline arms bypass any launch_*_pair guard).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_sigma_family_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    op: &str,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let rank = family_rank(op);
    let transform = family_transform(op);

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;

    // Fail-closed staging guard BEFORE any write (OOM-safe stop, no partial writes).
    let staging_required = ni_sp * nj_sp * 2 * rank;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // ── Build the family cart gc blocks (rank*N_GC blocks per (ci,cj)). ──
    let mut gc = build_sigma_cart(
        op,
        backend,
        li,
        lj,
        nprim_i,
        nprim_j,
        nctr_i,
        nctr_j,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        origin_coords,
        origin_charges,
    )?;

    // s/p normalization scale (matches the scalar/gradient 1e arms).
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
    if (sp_scale - 1.0).abs() > 1e-15 {
        for v in gc.iter_mut() {
            *v *= sp_scale;
        }
    }

    let total_len = rank * (N_GC as usize) * block_len;
    let group_out = ni_sp * nj_sp * 2; // one spinor matrix's flat length

    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            let base = (ci * nctr_j + cj) * total_len;
            for grp in 0..rank {
                let gbase = base + grp * (N_GC as usize) * block_len;
                let gc_x = &gc[gbase..gbase + block_len];
                let gc_y = &gc[gbase + block_len..gbase + 2 * block_len];
                let gc_z = &gc[gbase + 2 * block_len..gbase + 3 * block_len];
                let gc_1 = &gc[gbase + 3 * block_len..gbase + 4 * block_len];

                fold_group::<F>(
                    transform,
                    &mut scratch,
                    gc_x,
                    gc_y,
                    gc_z,
                    gc_1,
                    li,
                    kappa_i,
                    lj,
                    kappa_j,
                )?;

                // Scatter the di*dj*2 spinor sub-block into the contraction-major
                // grid for this σ-group. scratch is column-major (j*di+i)*2.
                let grp_off = grp * group_out;
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = grp_off + (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Build the family cart gc blocks for `op` (overlap or nuclear engine).
#[allow(clippy::too_many_arguments)]
fn build_sigma_cart(
    op: &str,
    backend: &ResolvedBackend,
    li: u8,
    lj: u8,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Result<Vec<f64>, cintxRsError> {
    let fam = family_id(op).ok_or_else(|| cintxRsError::UnsupportedApi {
        requested: format!("int1e_{op}_spinor is not a Group-4 1e σ family"),
    })?;

    if fam < 10 {
        // Overlap engine (sigma/sr/srsr/spsp).
        run_sigma_ov_on_backend(
            backend,
            fam,
            li as u32,
            lj as u32,
            nprim_i as u32,
            nprim_j as u32,
            nctr_i as u32,
            nctr_j as u32,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
        )
    } else {
        // Nuclear engine (spnucsp/srnucsr/sprinvsp) — implemented in sigma_1e_nuc.
        super::sigma_1e_nuc::run_sigma_nuc_on_backend(
            op,
            backend,
            li,
            lj,
            nprim_i,
            nprim_j,
            nctr_i,
            nctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            origin_coords,
            origin_charges,
        )
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 30 (GIAO-03 / Sub-wave 1a): GIAO×σ 1e family dispatch.
//
//  Wires the three Sub-wave-1a GIAO×σ 1e families — `spgsp` (8-G London overlap,
//  net-new engine), `cg_sa10sp` and `giao_sa10sp` (proven in 30-00) — through a
//  single dispatch table that routes each to its `sigma_p.rs` launcher. All three
//  are component_rank 3 and use the imaginary-ket `c2s_si_1ei` transform (SiI).
//
//  Each arm applies its OWN fail-closed full-block staging guard before calling
//  the launcher (the launchers also guard, but per Phase-28 CR-01 every inline
//  dispatch arm must fail closed independently — defense in depth, no partial
//  writes). spgsp's origin is `ri` (G1E_R0I) + London `ri−rj`; cg uses the gauge
//  `dri = ri − common_orig`; giao is cg at `origin = [0,0,0]`.
// ═════════════════════════════════════════════════════════════════════════════

/// GIAO×σ 1e family selector for Sub-wave 1a (symbol-name resolution; the
/// returned id is informational, NOT a positional OperatorId — Pitfall 6).
#[allow(dead_code)]
pub(crate) fn giao_family_id(op: &str) -> Option<u32> {
    match op {
        "spgsp" => Some(0),
        "cg_sa10sp" => Some(1),
        "giao_sa10sp" => Some(2),
        // Sub-wave 1c: the rank-9 Rys+gauge sa01 families.
        "cg_sa10sa01" => Some(3),
        "giao_sa10sa01" => Some(4),
        // Sub-wave 1b: the rank-3 Rys+gauge nuclear families.
        "cg_sa10nucsp" => Some(5),
        "giao_sa10nucsp" => Some(6),
        // Sub-wave 1d: the spg-Rys/London families (nuclear rank 3 / rinv rank 9).
        "spgnucsp" => Some(7),
        "spgsa01" => Some(8),
        _ => None,
    }
}

/// Component_rank for the GIAO×σ 1e families. Sub-wave-1a families are rank 3
/// (ng[7]); the Sub-wave-1c sa01 families are **rank 9** (ng[7]=9 — the GIAO
/// g-factor raises the rank; registering at rank 3 truncates 6 of 9 tensor
/// components, Pitfall 3).
#[allow(dead_code)]
pub(crate) fn giao_family_rank(op: &str) -> usize {
    match op {
        "spgsp" | "cg_sa10sp" | "giao_sa10sp" => 3,
        "cg_sa10nucsp" | "giao_sa10nucsp" => 3,
        "cg_sa10sa01" | "giao_sa10sa01" => 9,
        // Sub-wave 1d spg-Rys/London: spgnucsp rank 3 (ng[7]=3), spgsa01 rank 9
        // (ng[7]=9 — the GIAO g-factor raises the rank; truncating to 3 drops 6 of
        // 9 tensor components, Pitfall 3 / T-30-01d-02).
        "spgnucsp" => 3,
        "spgsa01" => 9,
        _ => 0,
    }
}

/// Transform selector for the GIAO×σ 1e families. The Sub-wave-1a sp families use
/// the imaginary-ket SiI (`cart_to_spinor_si_2di`); the Sub-wave-1c **sa01**
/// families use the REAL `Si` (`cart_to_spinor_si_2d` / c2s_si_1e) — the spin
/// operator sits on the ket-side angular operator, not as an imaginary ∇ phase
/// (Pitfall 2).
#[allow(dead_code)]
pub(crate) fn giao_family_transform(op: &str) -> TransformKind {
    let _ = TransformKind::Sf; // keep the variant referenced for non-spgsp arms
    match op {
        // The rank-9 sa01 families route through the REAL si transform (incl. the
        // Sub-wave-1d spg-Rys/London spgsa01).
        "cg_sa10sa01" | "giao_sa10sa01" | "spgsa01" => TransformKind::Si,
        // The sp/nucsp arms (incl. spgnucsp) are imaginary-ket (c2s_si_1ei).
        _ => TransformKind::SiI,
    }
}

/// Unified GIAO×σ 1e Spinor launcher for one shell pair `(i, j)` (Sub-wave 1a).
///
/// `op` is the operator name without the `int1e_`/`_spinor` affixes
/// (`spgsp` / `cg_sa10sp` / `giao_sa10sp`). `common_orig` is the gauge origin read
/// from `plan.operator_env_params.common_orig.unwrap_or([0.0;3])`; it is used ONLY
/// for `cg_sa10sp` (the gauge shift `dri = ri − common_orig`). `spgsp` ignores it
/// (its origin is `ri`, London phase `ri − rj`); `giao_sa10sp` uses `[0,0,0]`.
///
/// Output layout: 3 stacked spinor matrices, column-major (ket outer, bra inner),
/// interleaved complex, contraction-major:
/// `out[grp*(ni_sp*nj_sp*2) + (j_global*ni_sp + i_global)*2 + {0:re,1:im}]`.
///
/// Each arm applies its OWN fail-closed full-block staging guard
/// (`ni_sp*nj_sp*2*rank`) BEFORE any write (Phase-28 CR-01).
#[allow(clippy::too_many_arguments)]
pub fn launch_int1e_giao_sigma_family_spinor_pair<F: CintFloat>(
    backend: &ResolvedBackend,
    op: &str,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nprim_i: usize,
    nprim_j: usize,
    nctr_i: usize,
    nctr_j: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    common_orig: [f64; 3],
    rinv_orig: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    use crate::kernels::sigma_p::{
        launch_int1e_cg_sa10sp_spinor_pair, launch_int1e_giao_sa10sp_spinor_pair,
        launch_int1e_sa10nucsp_spinor_pair, launch_int1e_sa10sa01_spinor_pair,
        launch_int1e_spgnucsp_spinor_pair, launch_int1e_spgsa01_spinor_pair,
        launch_int1e_spgsp_spinor_pair,
    };

    let rank = giao_family_rank(op);
    if rank == 0 {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("int1e_{op}_spinor is not a Sub-wave-1a GIAO×σ 1e family"),
        });
    }

    // Fail-closed full-block staging guard BEFORE any write (Phase-28 CR-01: each
    // inline dispatch arm guards independently, no partial writes).
    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;
    let staging_required = ni_sp * nj_sp * 2 * rank;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    match op {
        "spgsp" => {
            // 8-G London overlap; origin = ri, London phase = ri − rj (no common_orig).
            // Its OWN BufferTooSmall guard runs inside the launcher too.
            launch_int1e_spgsp_spinor_pair::<F>(
                backend, li, kappa_i, lj, kappa_j, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj,
                exps_i, exps_j, coeff_i, coeff_j, staging,
            )
        }
        "cg_sa10sp" => {
            // Gauge fold: dri = ri − common_orig.
            let dri = [
                ri[0] - common_orig[0],
                ri[1] - common_orig[1],
                ri[2] - common_orig[2],
            ];
            launch_int1e_cg_sa10sp_spinor_pair::<F>(
                backend, li, kappa_i, lj, kappa_j, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, dri,
                exps_i, exps_j, coeff_i, coeff_j, staging,
            )
        }
        "giao_sa10sp" => {
            // Natural-center: cg at origin = [0,0,0] (G1E_R_I).
            launch_int1e_giao_sa10sp_spinor_pair::<F>(
                backend, li, kappa_i, lj, kappa_j, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj,
                exps_i, exps_j, coeff_i, coeff_j, staging,
            )
        }
        "cg_sa10sa01" => {
            // Rank-9 Rys+gauge. cg gauge shift dri = ri − common_orig (G2E_RCI).
            // rinv center = env[PTR_RINV_ORIG] (single rinv, charge +1; intor3.c
            // int1e_type=1) — threaded through `rinv_orig`, NOT a hardcoded [0,0,0]
            // (30-01c root cause: vendor reads PTR_RINV_ORIG; a non-zero center must
            // be honored). The launcher carries its OWN ×9 staging + nroots guards.
            let dri = [
                ri[0] - common_orig[0],
                ri[1] - common_orig[1],
                ri[2] - common_orig[2],
            ];
            launch_int1e_sa10sa01_spinor_pair::<F>(
                backend, li, kappa_i, lj, kappa_j, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj,
                rinv_orig, dri, exps_i, exps_j, coeff_i, coeff_j, staging,
            )
        }
        "giao_sa10sa01" => {
            // Rank-9 Rys+gauge. giao natural bra center: x1i origin = ri (G2E_R_I).
            // rinv center = env[PTR_RINV_ORIG] (same single-center rinv as cg).
            launch_int1e_sa10sa01_spinor_pair::<F>(
                backend, li, kappa_i, lj, kappa_j, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj,
                rinv_orig, ri, exps_i, exps_j, coeff_i, coeff_j, staging,
            )
        }
        "cg_sa10nucsp" => {
            // Rank-3 Rys+gauge nuclear. cg gauge shift dri = ri − common_orig
            // (G2E_RCI). The launcher carries its OWN ×3 staging guard; the
            // fail-closed nroots guard is inside run_sigma_nuc_gauge_on_backend.
            let dri = [
                ri[0] - common_orig[0],
                ri[1] - common_orig[1],
                ri[2] - common_orig[2],
            ];
            launch_int1e_sa10nucsp_spinor_pair::<F>(
                backend,
                li,
                kappa_i,
                lj,
                kappa_j,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                ri,
                rj,
                dri,
                exps_i,
                exps_j,
                coeff_i,
                coeff_j,
                origin_coords,
                origin_charges,
                staging,
            )
        }
        "giao_sa10nucsp" => {
            // Rank-3 Rys+gauge nuclear. giao natural bra center: x1i origin =
            // [0,0,0] (G2E_R_I, no gauge shift).
            launch_int1e_sa10nucsp_spinor_pair::<F>(
                backend,
                li,
                kappa_i,
                lj,
                kappa_j,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                ri,
                rj,
                [0.0, 0.0, 0.0],
                exps_i,
                exps_j,
                coeff_i,
                coeff_j,
                origin_coords,
                origin_charges,
                staging,
            )
        }
        "spgnucsp" => {
            // Sub-wave 1d: NUCLEAR Rys + 8-G London. London x1i origin = ri (G2E_R0I),
            // London phase = ri − rj; does NOT read PTR_COMMON_ORIG. Nuclear centers
            // are the atom-summed origin_coords/origin_charges (int1e_type 2). rank 3,
            // SiI. The launcher carries its OWN ×3 staging + nroots guards.
            launch_int1e_spgnucsp_spinor_pair::<F>(
                backend,
                li,
                kappa_i,
                lj,
                kappa_j,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                ri,
                rj,
                exps_i,
                exps_j,
                coeff_i,
                coeff_j,
                origin_coords,
                origin_charges,
                staging,
            )
        }
        "spgsa01" => {
            // Sub-wave 1d: RINV Rys + 8-G London (both-side g1). London x1i origin =
            // ri (G2E_R0I), London phase = ri − rj; does NOT read PTR_COMMON_ORIG.
            // rinv center = env[PTR_RINV_ORIG] (single center, charge +1, int1e_type
            // 1) — threaded via `rinv_orig`, NOT a hardcoded [0,0,0] (the 30-01c root
            // cause). rank 9, REAL Si. Own ×9 staging + nroots guards.
            launch_int1e_spgsa01_spinor_pair::<F>(
                backend, li, kappa_i, lj, kappa_j, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj,
                rinv_orig, exps_i, exps_j, coeff_i, coeff_j, staging,
            )
        }
        other => Err(cintxRsError::UnsupportedApi {
            requested: format!("int1e_{other}_spinor is not a GIAO×σ 1e family"),
        }),
    }
}
