//! `origk` family: origin-k-displaced 3c1e integrals (`cint3c1e_a.c`).
//! Split out of the original single-file `unstable.rs`;
//! move-only — function bodies are unchanged.

use super::shared::{SQRTPI, cart_comps, common_fac_sp, make_exec_stats};
use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c1e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use std::f64::consts::PI;


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
                        // r^6 from cint3c1e_a.c lines 240-249:
                        // g63x*g0y*g0z
                        // + g60x*g3y*g0z * 3  (g60=+4dk, g3=+2dk)
                        // + g60x*g0y*g3z * 3
                        // + g48x*g15y*g0z * 3  (g48=+2dk, g15=+4dk)
                        // + g48x*g12y*g3z * 6  (g12=+2dk)
                        // + g48x*g0y*g15z * 3
                        // + g0x*g63y*g0z
                        // + g0x*g60y*g3z * 3
                        // + g0x*g48y*g15z * 3
                        // + g0x*g0y*g63z
                        //
                        // Mapping: g3=+2dk, g12=+2dk, g15=+4dk, g48=+2dk, g60=+4dk, g63=+6dk
                        // Wait, need to trace the R_K chains more carefully:
                        //   g1 = g0 + dk (k+5)
                        //   g3 = g1 + dk = g0 + 2dk (k+4)
                        //   g4 = g0 + dk (k+3)
                        //   g7 = g3 + dk = g0 + 3dk  -- wait, g7 = (g3 from line 227) + dk
                        // Let me retrace from the C code:
                        //   G1E_R_K(g1, g0, ..., k+5)  -> g1 = g0 + dk
                        //   G1E_R_K(g3, g1, ..., k+4)  -> g3 = g1 + dk = g0 + 2dk
                        //   G1E_R_K(g4, g0, ..., k+3)  -> g4 = g0 + dk
                        //   G1E_R_K(g7, g3, ..., k+3)  -> g7 = g3 + dk = g0 + 3dk
                        //   G1E_R_K(g12, g4, ..., k+2) -> g12 = g4 + dk = g0 + 2dk
                        //   G1E_R_K(g15, g7, ..., k+2) -> g15 = g7 + dk = g0 + 4dk
                        //   G1E_R_K(g16, g0, ..., k+1) -> g16 = g0 + dk
                        //   G1E_R_K(g28, g12, ..., k+1) -> g28 = g12 + dk = g0 + 3dk
                        //   G1E_R_K(g31, g15, ..., k+1) -> g31 = g15 + dk = g0 + 5dk
                        //   G1E_R_K(g48, g16, ..., k+0) -> g48 = g16 + dk = g0 + 2dk
                        //   G1E_R_K(g60, g28, ..., k+0) -> g60 = g28 + dk = g0 + 4dk
                        //   G1E_R_K(g63, g31, ..., k+0) -> g63 = g31 + dk = g0 + 6dk
                        //
                        // So: g3=+2dk, g12=+2dk, g15=+4dk, g48=+2dk, g60=+4dk, g63=+6dk
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

    let gx = 0usize;
    let gy = g_alloc;
    let gz = 2 * g_alloc;

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
                        // ip1_r4 from cint3c1e_a.c lines 415-420:
                        // g3=+2dk, g12=+2dk, g15=+4dk, g16=D_I, g19=D_I+2dk, g28=D_I+2dk, g31=D_I+4dk
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
                        // ip1_r6: D_I applied to the r^6 expansion
                        // Each component: D_I on one axis, r^6 polynomial on the other two axes
                        // The r^6 polynomial is the same as contract_origk r6 but with D_I on one axis
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

/// Origk family launcher: dispatches 6 origin-k-displaced 3c1e variants.
///
/// These use the standard 3c1e G-tensor fill (same as center_3c1e) but with
/// elevated ceiling k-angular momentum. The r^n operator is encoded as dk shifts.
pub fn launch_origk(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    let _ = backend;

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

    let dj = dli;
    let dk = dli * dlj;

    let common_factor = SQRTPI * PI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
    let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
    let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
    let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

    let expcutoff = 60.0_f64;

    let mut cart_buf = vec![0.0_f64; nci * ncj * nck * variant.ncomp];

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

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

                let prim_buf = if variant.ncomp == 1 {
                    let r_power = variant.k_inc;
                    contract_origk(&g, g_alloc, li, lj, lk, dli, dlj, dk, r_power)
                } else {
                    // ip1 variants
                    contract_origk_ip1(&g, g_alloc, li, lj, lk, dli, dlj, dk, variant.k_inc, ai)
                };

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

    // Apply c2s transform
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
