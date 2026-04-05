//! Launch functions for unstable-source API families.
//!
//! Phase 14 Wave 2 — implements origi, origk, and ssc families.
//! Grids and breit remain stubs (Wave 2 plan 03/04).
//!
//! Families covered:
//!   - origi: origin-displaced r^n one-electron integrals (cint1e_a.c)
//!   - grids: grid-point integrals with NGRIDS env parameter (cint1e_grids.c) [stub]
//!   - breit: Breit spinor-only two-electron integrals (breit.c) [stub]
//!   - origk: origin-k-displaced three-center one-electron integrals (cint3c1e_a.c)
//!   - ssc: spin-spin contact three-center two-electron integral (cint3c2e.c)

use crate::backend::ResolvedBackend;
use crate::math::obara_saika::{hrr_step_host, vrr_step_host};
use crate::math::pdata::{PairData, compute_pdata_host};
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, cart_to_sph_3c1e, cart_to_sph_3c2e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

use std::f64::consts::PI;

/// sqrt(pi) constant.
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

fn make_exec_stats(plan: &ExecutionPlan<'_>, staging: &[f64]) -> ExecutionStats {
    let not0 = staging.iter().filter(|&&v| v.abs() > 1e-18).count() as i32;
    let staging_bytes = staging.len() * std::mem::size_of::<f64>();
    ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: staging_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: staging_bytes,
        not0,
        fallback_reason: plan.workspace.fallback_reason,
    }
}

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

// ─────────────────────────────────────────────────────────────────────────────
// SSC family: spin-spin contact 3c2e integral
// ─────────────────────────────────────────────────────────────────────────────

/// SSC launcher: same gout as standard int3c2e but with SSC c2s transform.
///
/// In the SSC c2s variant, the k-shell stays in Cartesian while i and j are
/// transformed to spherical. This differs from normal c2s_sph_3c2e1 which
/// transforms all three shells to spherical.
pub fn launch_ssc(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    let _ = backend;

    let shells = plan.shells.as_slice();
    if shells.len() < 3 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_ssc",
            detail: "ssc requires 3 shells".to_owned(),
        });
    }

    let shell_i_in = &shells[0];
    let shell_j_in = &shells[1];
    let shell_k = &shells[2];

    let li_in = shell_i_in.ang_momentum;
    let lj_in = shell_j_in.ang_momentum;
    let lk = shell_k.ang_momentum;

    // Same ibase canonicalization as center_3c2e
    let swap_ij = li_in < lj_in;
    let (shell_i, shell_j, li, lj) = if swap_ij {
        (shell_j_in, shell_i_in, lj_in, li_in)
    } else {
        (shell_i_in, shell_j_in, li_in, lj_in)
    };

    let nrys_roots = (li as usize + lj as usize + lk as usize) / 2 + 1;
    if nrys_roots > 5 {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{nrys_roots}"),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // SSC: same gout as standard 3c2e (CINTgout2e), same G-tensor fill
    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsi_in = nsph(li_in);
    let nsj_in = nsph(lj_in);
    // SSC: k stays Cartesian
    let nk_ssc = nck;

    let mut cart_buf = vec![0.0_f64; nci * ncj * nck];

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

                let pair = compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );
                let fac_env = common_factor * pair.fac;
                let g2d = fill_g_tensor_3c2e_ssc(
                    &pair, ak, ri, rk, li, lj, lk, nrys_roots, fac_env,
                );
                let g_split = split_ij_hrr_ssc(&g2d, li, lj, lk, nrys_roots, rirj);
                let prim_buf = contract_3c2e_ssc(&g_split, li, lj, lk, nrys_roots);

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

    let cart_out = if swap_ij {
        transpose_ij_3idx(&cart_buf, nci, ncj, nck)
    } else {
        cart_buf
    };

    // SSC c2s: spherical on i,j; Cartesian on k
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_3c2e_ssc(&cart_out, li_in, lj_in, lk);
            let out_size = nsi_in * nsj_in * nk_ssc;
            let copy_len = staging.len().min(sph.len()).min(out_size);
            staging[..copy_len].copy_from_slice(&sph[..copy_len]);
        }
        _ => {
            let copy_len = staging.len().min(cart_out.len());
            staging[..copy_len].copy_from_slice(&cart_out[..copy_len]);
        }
    }

    Ok(make_exec_stats(plan, staging))
}

/// Transpose a flat 3-index buffer from (i,j,k) to (j,i,k) ordering.
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

/// SSC c2s: apply spherical transform to i and j, leave k in Cartesian.
///
/// This mirrors libcint's c2s_sph_3c2e1_ssc: c2s_ket_sph on j, c2s_bra_sph on i,
/// k stays as nfk Cartesian functions.
fn cart_to_sph_3c2e_ssc(cart: &[f64], li: u8, lj: u8, lk: u8) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk); // stays Cartesian
    let nsi = nsph(li);
    let nsj = nsph(lj);

    // Use existing cart_to_sph_1e infrastructure for the i,j part of each k slice
    // Cart layout: (k * ncj + j) * nci + i
    // For each k, extract the nci x ncj block, transform i,j to spherical, write to output

    let mut out = vec![0.0_f64; nsi * nsj * nck];

    for k in 0..nck {
        // Extract the 2D slice for this k
        let mut ij_cart = vec![0.0_f64; nci * ncj];
        for j in 0..ncj {
            for i in 0..nci {
                ij_cart[i * ncj + j] = cart[(k * ncj + j) * nci + i];
            }
        }

        // Apply 1e-style c2s to the i,j block
        let mut ij_sph = vec![0.0_f64; nsi * nsj];
        cart_to_sph_1e(&ij_cart, &mut ij_sph, li, lj);

        // Write into output: (k * nsj + j) * nsi + i
        for j in 0..nsj {
            for i in 0..nsi {
                out[(k * nsj + j) * nsi + i] = ij_sph[i * nsj + j];
            }
        }
    }

    out
}

// Reuse the 3c2e G-tensor fill and HRR from center_3c2e, adapted for SSC.
// SSC uses the exact same G-tensor as standard 3c2e (CINTgout2e).

fn fill_g_tensor_3c2e_ssc(
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
    let akl = ak;
    let p = [pair.center_p_x, pair.center_p_y, pair.center_p_z];

    let xij_kl = p[0] - rk[0];
    let yij_kl = p[1] - rk[1];
    let zij_kl = p[2] - rk[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

    let a1 = aij * akl;
    let a0 = a1 / (aij + akl);
    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_env;
    let x_rys = a0 * rr;
    let (u_roots, w_weights) = rys_roots_host(nrys_roots, x_rys);

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
        let c0p = [tmp3 * xij_kl, tmp3 * yij_kl, tmp3 * zij_kl];

        g[irys] = 1.0;
        g[g_size + irys] = 1.0;
        g[2 * g_size + irys] = w_weights[irys] * fac1;

        for axis in 0..3 {
            let axis_off = axis * g_size;
            let c00_axis = c00[axis];
            let c0p_axis = c0p[axis];

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

            if mmax > 0 {
                let mut s_prev = g[axis_off + irys];
                let mut s1 = c0p_axis * s_prev;
                g[axis_off + irys + dm] = s1;
                for m in 1..mmax {
                    let s2 = c0p_axis * s1 + m as f64 * b01 * s_prev;
                    g[axis_off + irys + (m + 1) * dm] = s2;
                    s_prev = s1;
                    s1 = s2;
                }

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

fn split_ij_hrr_ssc(
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

fn contract_3c2e_ssc(g: &[f64], li: u8, lj: u8, lk: u8, nrys_roots: usize) -> Vec<f64> {
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
                    let idx_x = ((root * nk + kx as usize) * nj + jx as usize) * ni + ix as usize;
                    let idx_y = ((root * nk + ky as usize) * nj + jy as usize) * ni + iy as usize;
                    let idx_z = ((root * nk + kz as usize) * nj + jz as usize) * ni + iz as usize;
                    val += g[gx_off + idx_x] * g[gy_off + idx_y] * g[gz_off + idx_z];
                }
                out[(k_idx * ncj + j_idx) * nci + i_idx] += val;
            }
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Stubs for grids and breit (pending Wave 2 plan 03/04)
// ─────────────────────────────────────────────────────────────────────────────

/// Stub for grids family (int1e_grids and derivative variants).
/// Implementation pending in Phase 14 Plan 03.
pub fn launch_grids(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "grids: stub -- implementation pending".to_owned(),
    })
}

/// Stub for breit family (int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor).
/// Implementation pending in Phase 14 Plan 03.
pub fn launch_breit(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "breit: stub -- implementation pending".to_owned(),
    })
}
