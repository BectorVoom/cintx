//! `origi` family: origin-displaced r^n one-electron integrals
//! (`cint1e_a.c`). Split out of the original single-file `unstable.rs`;
//! move-only — function bodies are unchanged.

use super::shared::{SQRTPI, cart_comps, common_fac_sp, make_exec_stats};
use crate::backend::ResolvedBackend;
use crate::math::obara_saika::{hrr_step_host, vrr_step_host};
use crate::math::pdata::{PairData, compute_pdata_host};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Origi family: origin-displaced r^n 1e integrals
// ─────────────────────────────────────────────────────────────────────────────

/// Variant metadata for origi operators.
///
/// `i_inc`: extra i-angular momentum (from ng[0])
/// `j_inc`: extra j-angular momentum (from ng[1])
/// `ncomp`: number of output components
struct OrigiVariant {
    i_inc: u8,
    j_inc: u8,
    ncomp: usize,
}

fn origi_variant(op_name: &str) -> Result<OrigiVariant, cintxRsError> {
    match op_name {
        "r2_origi" => Ok(OrigiVariant { i_inc: 2, j_inc: 0, ncomp: 1 }),
        "r4_origi" => Ok(OrigiVariant { i_inc: 4, j_inc: 0, ncomp: 1 }),
        "r2_origi_ip2" => Ok(OrigiVariant { i_inc: 2, j_inc: 1, ncomp: 3 }),
        "r4_origi_ip2" => Ok(OrigiVariant { i_inc: 4, j_inc: 1, ncomp: 3 }),
        _ => Err(cintxRsError::UnsupportedApi {
            requested: format!("origi variant '{}' not supported", op_name),
        }),
    }
}

/// Apply G1E_R_I: f = g + stride_i (pointer offset by 1 in the i-direction).
///
/// In a 1e G-tensor with g_stride_i = 1, this simply means reading at index i+1
/// instead of i. We implement it by copying the shifted slice into a new buffer.
#[allow(dead_code)]
fn g1e_r_i(g: &[f64], g_size: usize) -> Vec<f64> {
    // f[n] = g[n + 1] per axis. stride_i = 1 for 1e.
    let mut f = vec![0.0_f64; 3 * g_size];
    for axis in 0..3 {
        let off = axis * g_size;
        for n in 0..g_size - 1 {
            f[off + n] = g[off + n + 1];
        }
    }
    f
}

/// Apply G1E_D_J: nabla in j-direction.
/// f[j=0, i] = -2*aj * g[j=1, i]
/// f[j>0, i] = j * g[j-1, i] + (-2*aj) * g[j+1, i]
fn g1e_d_j(g: &[f64], g_size: usize, li: usize, lj: usize, _lk: usize, dj: usize, aj: f64) -> Vec<f64> {
    let mut f = vec![0.0_f64; 3 * g_size];
    let aj2 = -2.0 * aj;
    for axis in 0..3 {
        let off = axis * g_size;
        // j=0
        for i in 0..=li {
            f[off + i] = aj2 * g[off + i + dj];
        }
        // j=1..lj
        for j in 1..=lj {
            for i in 0..=li {
                let ptr = j * dj + i;
                f[off + ptr] = (j as f64) * g[off + ptr - dj] + aj2 * g[off + ptr + dj];
            }
        }
    }
    f
}

/// Contract origi r^2 gout: sum over xyz of g3[ix]*g0[iy]*g0[iz]
/// where g3 = G1E_R_I(G1E_R_I(g0)). g3 is g shifted by 2 in i-direction.
fn contract_origi_r2(g0: &[f64], g_size: usize, li: u8, lj: u8, dj: usize) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let mut out = vec![0.0_f64; nci * ncj];

    // g3 = g0 shifted by +2 in i-direction (G1E_R_I applied twice)
    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let base_x = jx as usize * dj + ix as usize;
            let base_y = jy as usize * dj + iy as usize;
            let base_z = jz as usize * dj + iz as usize;

            // g3[ix] = g0[ix+2], g0[iy] = g0[iy], g0[iz] = g0[iz]
            let s = g0[0 * g_size + base_x + 2] * g0[1 * g_size + base_y] * g0[2 * g_size + base_z]
                  + g0[0 * g_size + base_x] * g0[1 * g_size + base_y + 2] * g0[2 * g_size + base_z]
                  + g0[0 * g_size + base_x] * g0[1 * g_size + base_y] * g0[2 * g_size + base_z + 2];
            out[ci_idx * ncj + cj_idx] += s;
        }
    }
    out
}

/// Contract origi r^4 gout.
/// From libcint cint1e_a.c:
///   g15[ix]*g0[iy]*g0[iz]
/// + g12[ix]*g3[iy]*g0[iz]*2
/// + g12[ix]*g0[iy]*g3[iz]*2
/// + g0[ix]*g15[iy]*g0[iz]
/// + g0[ix]*g12[iy]*g3[iz]*2
/// + g0[ix]*g0[iy]*g15[iz]
///
/// where g3 = g0 + 2*stride_i (r_i^2), g12 = g0 + 2*stride_i (same as g3 for r_i^2),
/// g15 = g0 + 4*stride_i (r_i^4).
///
/// Examining the C code more carefully:
///   g1 = g0 + stride (i.e., g0[ix+1])
///   g3 = g1 + stride = g0[ix+2]
///   g4 = g0 + stride = g0[ix+1]  (separate chain for cross terms)
///   g7 = g3 + stride = g0[ix+3]... wait, need to re-read the code.
///
/// From cint1e_a.c lines 96-101:
///   G1E_R_I(g1, g0, i_l+3, j_l, 0)    -> g1 = g0 + 1
///   G1E_R_I(g3, g1, i_l+2, j_l, 0)    -> g3 = g1 + 1 = g0 + 2
///   G1E_R_I(g4, g0, i_l+1, j_l, 0)    -> g4 = g0 + 1
///   G1E_R_I(g7, g3, i_l+1, j_l, 0)    -> g7 = g3 + 1 = g0 + 3
///   G1E_R_I(g12, g4, i_l+0, j_l, 0)   -> g12 = g4 + 1 = g0 + 2
///   G1E_R_I(g15, g7, i_l+0, j_l, 0)   -> g15 = g7 + 1 = g0 + 4
///
/// So: g3 = g0+2, g12 = g0+2, g15 = g0+4
/// gout = g15x*g0y*g0z + g12x*g3y*g0z*2 + g12x*g0y*g3z*2
///      + g0x*g15y*g0z + g0x*g12y*g3z*2 + g0x*g0y*g15z
///
/// = g0[ix+4]*g0[iy]*g0[iz] + 2*g0[ix+2]*g0[iy+2]*g0[iz]
///   + 2*g0[ix+2]*g0[iy]*g0[iz+2] + g0[ix]*g0[iy+4]*g0[iz]
///   + 2*g0[ix]*g0[iy+2]*g0[iz+2] + g0[ix]*g0[iy]*g0[iz+4]
fn contract_origi_r4(g0: &[f64], g_size: usize, li: u8, lj: u8, dj: usize) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let mut out = vec![0.0_f64; nci * ncj];

    let gx = 0usize;
    let gy = g_size;
    let gz = 2 * g_size;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let bx = jx as usize * dj + ix as usize;
            let by = jy as usize * dj + iy as usize;
            let bz = jz as usize * dj + iz as usize;

            let s = g0[gx + bx + 4] * g0[gy + by] * g0[gz + bz]
                + 2.0 * g0[gx + bx + 2] * g0[gy + by + 2] * g0[gz + bz]
                + 2.0 * g0[gx + bx + 2] * g0[gy + by] * g0[gz + bz + 2]
                + g0[gx + bx] * g0[gy + by + 4] * g0[gz + bz]
                + 2.0 * g0[gx + bx] * g0[gy + by + 2] * g0[gz + bz + 2]
                + g0[gx + bx] * g0[gy + by] * g0[gz + bz + 4];
            out[ci_idx * ncj + cj_idx] += s;
        }
    }
    out
}

/// Contract origi r^2 ip2 gout (3-component).
/// From cint1e_a.c lines 163-167:
///   g1 = D_J(g0)    -> nabla_j on g0
///   g2 = g0 + 1     -> R_I on g0
///   g3 = g1 + 1     -> R_I on D_J(g0)
///   g6 = g2 + 1     -> R_I on R_I on g0 = g0 shifted +2
///   g7 = g3 + 1     -> R_I on R_I on D_J(g0) = g1 shifted +2
///
/// s[0] = g7x*g0y*g0z + g1x*g6y*g0z + g1x*g0y*g6z
/// s[1] = g6x*g1y*g0z + g0x*g7y*g0z + g0x*g1y*g6z
/// s[2] = g6x*g0y*g1z + g0x*g6y*g1z + g0x*g0y*g7z
///
/// g6 = g0 + 2 in i, g7 = g1 + 2 in i (where g1 = D_J(g0))
fn contract_origi_r2_ip2(
    g0: &[f64],
    g_size: usize,
    li: u8,
    lj: u8,
    dj: usize,
    aj: f64,
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let ncomp = 3;
    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let mut out = vec![0.0_f64; nci * ncj * ncomp];

    // Build g1 = D_J(g0) with lj+1 coverage
    // We need li_ceil = li + 2, lj_ceil = lj + 1 for D_J, then R_I shifts
    let g1 = g1e_d_j(g0, g_size, (li as usize) + 2, (lj as usize), 0, dj, aj);

    let gx = 0usize;
    let gy = g_size;
    let gz = 2 * g_size;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let bx = jx as usize * dj + ix as usize;
            let by = jy as usize * dj + iy as usize;
            let bz = jz as usize * dj + iz as usize;

            let n = ci_idx * ncj + cj_idx;

            // g6 = g0[..+2], g7 = g1[..+2], g1 = D_J(g0)
            let g0x = g0[gx + bx]; let g0y = g0[gy + by]; let g0z = g0[gz + bz];
            let g1x = g1[gx + bx]; let g1y = g1[gy + by]; let g1z = g1[gz + bz];
            let g6x = g0[gx + bx + 2]; let g6y = g0[gy + by + 2]; let g6z = g0[gz + bz + 2];
            let g7x = g1[gx + bx + 2]; let g7y = g1[gy + by + 2]; let g7z = g1[gz + bz + 2];

            let cart_size = nci * ncj;
            out[0 * cart_size + n] += g7x * g0y * g0z + g1x * g6y * g0z + g1x * g0y * g6z;
            out[1 * cart_size + n] += g6x * g1y * g0z + g0x * g7y * g0z + g0x * g1y * g6z;
            out[2 * cart_size + n] += g6x * g0y * g1z + g0x * g6y * g1z + g0x * g0y * g7z;
        }
    }
    out
}

/// Contract origi r^4 ip2 gout (3-component).
/// From cint1e_a.c lines 254-276, tracing the G1E_R_I and G1E_D_J chains:
///   g1 = D_J(g0)                     shift: D_J
///   g2 = g0 + 1                      shift: R_I^1
///   g3 = g1 + 1 = D_J(g0) + R_I     shift: D_J + R_I^1
///   g6 = g2 + 1 = g0 + 2            shift: R_I^2
///   g7 = g3 + 1 = D_J(g0) + 2       shift: D_J + R_I^2
///   g8 = g0 + 1                      shift: R_I^1  (separate chain)
///   g9 = g1 + 1 = D_J(g0) + 1       shift: D_J + R_I^1
///   g14 = g6 + 1 = g0 + 3           shift: R_I^3
///   g15 = g7 + 1 = D_J(g0) + 3      shift: D_J + R_I^3
///   g24 = g8 + 1 = g0 + 2           shift: R_I^2
///   g25 = g9 + 1 = D_J(g0) + 2      shift: D_J + R_I^2
///   g30 = g14 + 1 = g0 + 4          shift: R_I^4
///   g31 = g15 + 1 = D_J(g0) + 4     shift: D_J + R_I^4
///
/// So: g0[+n] = g0 shifted by n in i; g1[+n] = D_J(g0) shifted by n in i
/// Mapping: g3=g0+2, g6=g0+2, g7=g1+2, g12=g0+2, g14=g0+3, g15=g1+3,
///          g24=g0+2, g25=g1+2, g30=g0+4, g31=g1+4
///
/// From the gout formula (lines 271-276):
///   s[0] = g31x*g0y*g0z + 2*g25x*g6y*g0z + 2*g25x*g0y*g6z
///        + g1x*g30y*g0z + 2*g1x*g24y*g6z + g1x*g0y*g30z
///   (where g6=g0+2, g24=g0+2, g25=g1+2, g30=g0+4, g31=g1+4)
fn contract_origi_r4_ip2(
    g0: &[f64],
    g_size: usize,
    li: u8,
    lj: u8,
    dj: usize,
    aj: f64,
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let ncomp = 3;
    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let mut out = vec![0.0_f64; nci * ncj * ncomp];

    // Build g1 = D_J(g0) with enough headroom: li_ceil = li + 4, lj = lj
    let g1 = g1e_d_j(g0, g_size, (li as usize) + 4, lj as usize, 0, dj, aj);

    let gx = 0usize;
    let gy = g_size;
    let gz = 2 * g_size;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let bx = jx as usize * dj + ix as usize;
            let by = jy as usize * dj + iy as usize;
            let bz = jz as usize * dj + iz as usize;

            let n = ci_idx * ncj + cj_idx;

            // Shortcuts for g0 and g1 at various i-shifts
            let g0v = |axis_off: usize, base: usize, shift: usize| g0[axis_off + base + shift];
            let g1v = |axis_off: usize, base: usize, shift: usize| g1[axis_off + base + shift];

            // s[0] = g31x*g0y*g0z + 2*g25x*g6y*g0z + 2*g25x*g0y*g6z
            //       + g1x*g30y*g0z + 2*g1x*g24y*g6z + g1x*g0y*g30z
            // g31=g1+4, g25=g1+2, g6=g0+2, g30=g0+4, g24=g0+2
            let s0 = g1v(gx, bx, 4) * g0v(gy, by, 0) * g0v(gz, bz, 0)
                + 2.0 * g1v(gx, bx, 2) * g0v(gy, by, 2) * g0v(gz, bz, 0)
                + 2.0 * g1v(gx, bx, 2) * g0v(gy, by, 0) * g0v(gz, bz, 2)
                + g1v(gx, bx, 0) * g0v(gy, by, 4) * g0v(gz, bz, 0)
                + 2.0 * g1v(gx, bx, 0) * g0v(gy, by, 2) * g0v(gz, bz, 2)
                + g1v(gx, bx, 0) * g0v(gy, by, 0) * g0v(gz, bz, 4);

            // s[1]: swap x and y roles in g1/g0
            let s1 = g0v(gx, bx, 4) * g1v(gy, by, 0) * g0v(gz, bz, 0)
                + 2.0 * g0v(gx, bx, 2) * g1v(gy, by, 2) * g0v(gz, bz, 0)
                + 2.0 * g0v(gx, bx, 2) * g1v(gy, by, 0) * g0v(gz, bz, 2)
                + g0v(gx, bx, 0) * g1v(gy, by, 4) * g0v(gz, bz, 0)
                + 2.0 * g0v(gx, bx, 0) * g1v(gy, by, 2) * g0v(gz, bz, 2)
                + g0v(gx, bx, 0) * g1v(gy, by, 0) * g0v(gz, bz, 4);

            // s[2]: swap x and z roles in g1/g0
            let s2 = g0v(gx, bx, 4) * g0v(gy, by, 0) * g1v(gz, bz, 0)
                + 2.0 * g0v(gx, bx, 2) * g0v(gy, by, 2) * g1v(gz, bz, 0)
                + 2.0 * g0v(gx, bx, 2) * g0v(gy, by, 0) * g1v(gz, bz, 2)
                + g0v(gx, bx, 0) * g0v(gy, by, 4) * g1v(gz, bz, 0)
                + 2.0 * g0v(gx, bx, 0) * g0v(gy, by, 2) * g1v(gz, bz, 2)
                + g0v(gx, bx, 0) * g0v(gy, by, 0) * g1v(gz, bz, 4);

            let cart_size = nci * ncj;
            out[0 * cart_size + n] += s0;
            out[1 * cart_size + n] += s1;
            out[2 * cart_size + n] += s2;
        }
    }
    out
}

/// Origi family launcher: dispatches 4 origin-displaced r^n 1e integral variants.
///
/// These are standard 1e overlap integrals with the G-tensor built at higher ceiling
/// angular momentum. The r^n operator is encoded as pointer offsets (G1E_R_I) in the
/// gout function, which in our flat G-tensor translates to index shifts in the i-direction.
pub fn launch_origi(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    let _ = backend;

    let op_name = plan.descriptor.operator_name();
    let variant = origi_variant(op_name)?;

    let shells = plan.shells.as_slice();
    if shells.len() < 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_origi",
            detail: "origi requires 2 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    // Ceiling angular momenta include the i_inc and j_inc from the variant ng array.
    let li_ceil = li as u32 + variant.i_inc as u32;
    let lj_ceil = lj as u32 + variant.j_inc as u32;
    let nmax = li_ceil + lj_ceil;

    // G-tensor per-axis size: (nmax+1) * (lj_ceil+1)
    let g_per_axis = ((nmax + 1) * (lj_ceil + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let mut cart_buf = vec![0.0_f64; nci * ncj * variant.ncomp];

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

            // Build G-tensor with the origi ceiling angular momentum
            let g = fill_g_tensor_origi(&pd, ri, rj, nmax, lj_ceil);

            // Contract based on variant
            let prim_buf = match op_name {
                "r2_origi" => contract_origi_r2(&g, g_per_axis, li, lj, dj),
                "r4_origi" => contract_origi_r4(&g, g_per_axis, li, lj, dj),
                "r2_origi_ip2" => contract_origi_r2_ip2(&g, g_per_axis, li, lj, dj, aj),
                "r4_origi_ip2" => contract_origi_r4_ip2(&g, g_per_axis, li, lj, dj, aj),
                _ => unreachable!(),
            };

            for ci in 0..n_ctr_i {
                let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                for cj in 0..n_ctr_j {
                    let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                    let weight = coeff_i * coeff_j;
                    for k in 0..prim_buf.len() {
                        cart_buf[k] += weight * prim_buf[k];
                    }
                }
            }
        }
    }

    // Apply sp normalization
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
    if (sp_scale - 1.0).abs() > 1e-15 {
        for v in cart_buf.iter_mut() {
            *v *= sp_scale;
        }
    }

    // For multi-component ip2: apply c2s to each component separately
    if variant.ncomp == 1 {
        match plan.representation {
            Representation::Spheric => {
                let sph_size = nsi * nsj;
                if staging.len() >= sph_size {
                    cart_to_sph_1e(&cart_buf, &mut staging[..sph_size], li, lj);
                }
            }
            _ => {
                let copy_len = staging.len().min(cart_buf.len());
                staging[..copy_len].copy_from_slice(&cart_buf[..copy_len]);
            }
        }
    } else {
        // ncomp > 1: c2s each component, layout: comp slowest
        match plan.representation {
            Representation::Spheric => {
                let sph_size = nsi * nsj;
                let cart_size = nci * ncj;
                for comp in 0..variant.ncomp {
                    let cart_slice = &cart_buf[comp * cart_size..(comp + 1) * cart_size];
                    let sph_off = comp * sph_size;
                    if sph_off + sph_size <= staging.len() {
                        cart_to_sph_1e(cart_slice, &mut staging[sph_off..sph_off + sph_size], li, lj);
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

/// Fill G-tensor for origi (standard 1e overlap with elevated ceiling angular momentum).
/// Identical to one_electron::fill_g_tensor_overlap.
fn fill_g_tensor_origi(
    pd: &PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    nmax: u32,
    lj: u32,
) -> Vec<f64> {
    let g_per_axis = ((nmax + 1) * (lj + 1)) as usize;
    let mut g = vec![0.0_f64; 3 * g_per_axis];

    let aij = pd.zeta_ab;
    let gz0 = pd.fac * SQRTPI * PI / (aij * aij.sqrt());

    let gx = 0;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;

    g[gx] = 1.0;
    g[gy] = 1.0;
    g[gz] = gz0;

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let rijrx = [rp[0] - ri[0], rp[1] - ri[1], rp[2] - ri[2]];

    if nmax >= 1 {
        vrr_step_host(&mut g[gx..gx + g_per_axis], rijrx[0], pd.aij2, nmax, 1);
        vrr_step_host(&mut g[gy..gy + g_per_axis], rijrx[1], pd.aij2, nmax, 1);
        vrr_step_host(&mut g[gz..gz + g_per_axis], rijrx[2], pd.aij2, nmax, 1);
    }

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    if lj >= 1 {
        let di = 1u32;
        let dj = nmax + 1;
        hrr_step_host(&mut g[gx..gx + g_per_axis], rirj[0], di, dj, nmax, lj);
        hrr_step_host(&mut g[gy..gy + g_per_axis], rirj[1], di, dj, nmax, lj);
        hrr_step_host(&mut g[gz..gz + g_per_axis], rirj[2], di, dj, nmax, lj);
    }

    g
}
