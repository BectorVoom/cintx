//! `origi` family: origin-displaced r^n one-electron integrals
//! (`cint1e_a.c`).
//!
//! # Host / device split (quick-260529-twi Task 2)
//!
//! The SCALAR r2/r4 paths (`int1e_r2_origi_sph`, `int1e_r4_origi_sph`) now run a
//! real CubeCL `#[cube(launch)]` device kernel ([`origi_scalar_kernel`]) generic
//! over `F: Float`, dispatched on all five `ResolvedBackend` arms
//! (CPU / Wgpu / Cuda / ROCm-HIP / Metal) via [`run_origi_scalar_on_backend`].
//! The kernel builds the per-primitive overlap G-tensor (VRR + HRR) at the origi
//! ceiling angular momentum and performs the r2/r4 Cartesian contraction
//! ON-DEVICE, in f64-internal / F-output, accumulating ONE `nci*ncj` Cartesian
//! block (i-major: `out[ci_idx*ncj + cj_idx]`) summed over every primitive pair
//! AND contraction pair -- exactly matching the host scalar path.
//!
//! ON HOST (host part of the split, UNCHANGED): the `cart_to_sph_1e`
//! coefficient-table transform (its tables are host-only) + the `common_fac_sp`
//! s/p normalization + the AO scatter into `staging`.
//!
//! DEFERRED-TO-HOST (documented, mirrors the 1e/2e ports deferring derivative
//! sub-paths): the `ip2` variants (`int1e_r2_origi_ip2_sph`,
//! `int1e_r4_origi_ip2_sph`) keep their existing host path. The ip2 path adds a
//! second `G1E_D_J` ket-derivative ladder (`g1e_d_j` + `contract_origi_*_ip2`)
//! that is a separable port; it is left on host this task and selected by an
//! early branch on `origi_variant` inside `launch_origi`.

use super::shared::{SQRTPI, cart_comps, common_fac_sp, make_exec_stats};
use crate::backend::ResolvedBackend;
use crate::math::obara_saika::{hrr_step_host, vrr_step_host};
use crate::math::pdata::PairData;
use crate::math::pdata::compute_pdata_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// Maximum bra+ket angular ceiling the device G-tensor path supports. origi's
/// scalar contraction reads i-shifts up to +4 (r4), so the G-tensor must already
/// carry that headroom (it does: `nmax = li_ceil + lj_ceil`). This fail-closed
/// guard mirrors the `MAX_DEVICE_NROOTS` guard in `one_electron.rs`; origi uses no
/// Rys roots (pure overlap recurrence), so it bounds the combined ceiling instead.
const MAX_DEVICE_LSUM: u32 = 12;

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
        "r2_origi" => Ok(OrigiVariant {
            i_inc: 2,
            j_inc: 0,
            ncomp: 1,
        }),
        "r4_origi" => Ok(OrigiVariant {
            i_inc: 4,
            j_inc: 0,
            ncomp: 1,
        }),
        "r2_origi_ip2" => Ok(OrigiVariant {
            i_inc: 2,
            j_inc: 1,
            ncomp: 3,
        }),
        "r4_origi_ip2" => Ok(OrigiVariant {
            i_inc: 4,
            j_inc: 1,
            ncomp: 3,
        }),
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
///
/// Host reference for the device port (`origi_dj_axis` / `origi_ip2_kernel`); the
/// live ip2 path now runs on-device, so this is retained as the device-vs-host
/// oracle.
#[allow(dead_code)]
fn g1e_d_j(
    g: &[f64],
    g_size: usize,
    li: usize,
    lj: usize,
    _lk: usize,
    dj: usize,
    aj: f64,
) -> Vec<f64> {
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
///
/// Host reference for the device port (`origi_scalar_kernel`); the live scalar
/// path now runs on-device, so this is retained as the device-vs-host oracle.
#[allow(dead_code)]
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
///
/// Host reference for the device port (`origi_scalar_kernel`); the live scalar
/// path now runs on-device, so this is retained as the device-vs-host oracle.
#[allow(dead_code)]
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
///
/// Host reference for the device port (`origi_ip2_kernel`); the live ip2 path now
/// runs on-device, so this is retained as the device-vs-host oracle.
#[allow(dead_code)]
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
            let g0x = g0[gx + bx];
            let g0y = g0[gy + by];
            let g0z = g0[gz + bz];
            let g1x = g1[gx + bx];
            let g1y = g1[gy + by];
            let g1z = g1[gz + bz];
            let g6x = g0[gx + bx + 2];
            let g6y = g0[gy + by + 2];
            let g6z = g0[gz + bz + 2];
            let g7x = g1[gx + bx + 2];
            let g7y = g1[gy + by + 2];
            let g7z = g1[gz + bz + 2];

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
///
/// Host reference for the device port (`origi_ip2_kernel`); the live ip2 path now
/// runs on-device, so this is retained as the device-vs-host oracle.
#[allow(dead_code)]
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

    let mut cart_buf = vec![0.0_f64; nci * ncj * variant.ncomp];

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    // r_power encodes the scalar r^n operator: r2 -> 2, r4 -> 4. The ip2 variants
    // carry j_inc>0 and stay on host (deferred -- see module doc).
    let is_scalar = variant.j_inc == 0;
    let r_power: u32 = variant.i_inc as u32; // 2 for r2, 4 for r4

    if is_scalar {
        // -- DEVICE PATH (origi scalar r2/r4) --------------------------------
        // Fail-closed guard on the combined angular ceiling (overlap recurrence,
        // no Rys roots -- bounds nmax instead of nroots).
        if nmax > MAX_DEVICE_LSUM {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_origi",
                detail: format!(
                    "origi scalar device path requires nmax <= {MAX_DEVICE_LSUM}, got {nmax}"
                ),
            });
        }

        let dev = run_origi_scalar_on_backend(
            backend,
            r_power,
            li as u32,
            lj as u32,
            n_prim_i as u32,
            n_prim_j as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            ri,
            rj,
            &shell_i.exponents[..n_prim_i],
            &shell_j.exponents[..n_prim_j],
            &shell_i.coefficients[..n_prim_i * n_ctr_i],
            &shell_j.coefficients[..n_prim_j * n_ctr_j],
        );
        let copy = cart_buf.len().min(dev.len());
        cart_buf[..copy].copy_from_slice(&dev[..copy]);
    } else {
        // -- DEVICE PATH (origi ip2 r2/r4) -----------------------------------
        // Same overlap-recurrence ceiling guard as the scalar path; the ip2
        // ceiling is one higher in j (lj_ceil = lj + 1) -> nmax is +1.
        if nmax > MAX_DEVICE_LSUM {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "cubecl_origi",
                detail: format!(
                    "origi ip2 device path requires nmax <= {MAX_DEVICE_LSUM}, got {nmax}"
                ),
            });
        }

        let dev = run_origi_ip2_on_backend(
            backend,
            r_power,
            li as u32,
            lj as u32,
            n_prim_i as u32,
            n_prim_j as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            ri,
            rj,
            &shell_i.exponents[..n_prim_i],
            &shell_j.exponents[..n_prim_j],
            &shell_i.coefficients[..n_prim_i * n_ctr_i],
            &shell_j.coefficients[..n_prim_j * n_ctr_j],
        );
        let copy = cart_buf.len().min(dev.len());
        cart_buf[..copy].copy_from_slice(&dev[..copy]);
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
                        cart_to_sph_1e(
                            cart_slice,
                            &mut staging[sph_off..sph_off + sph_size],
                            li,
                            lj,
                        );
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
fn fill_g_tensor_origi(pd: &PairData, ri: [f64; 3], rj: [f64; 3], nmax: u32, lj: u32) -> Vec<f64> {
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
//  Device kernel — `#[cube(launch)]`, generic over `F: Float`
//
//  On-device port of the host scalar origi pipeline
//  (`fill_g_tensor_origi` -> `contract_origi_r2` / `contract_origi_r4`).
//  Single work item (`UNIT_POS == 0`): iterates primitive pairs (pi,pj) and
//  contraction pairs (ci,cj) in-kernel and accumulates ONE `nci*ncj` Cartesian
//  block (i-major: `out[ci_idx*ncj + cj_idx]`) summed over all (pi,pj)x(ci,cj),
//  matching the host `cart_buf` exactly. f64-internal / F-output.
//
//  The `#[comptime] r_power` arg (2 or 4) selects the r2 vs r4 contraction
//  branch via `comptime!`. The overlap G-tensor is built at the origi ceiling
//  angular momentum (nmax = li + r_power + lj), reusing the per-axis VRR/HRR
//  `#[cube]` helpers below (direct `#[cube]` calls are allowed — they are NOT
//  plain-fn calls).
// ─────────────────────────────────────────────────────────────────────────────

/// Per-axis 1e overlap VRR into the `g` sub-block starting at `base`
/// (fixed-center recurrence; stride 1). Mirrors `vrr_step_host`.
#[cube]
fn origi_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, rijrx: F, aij2: F, nmax: u32) {
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

/// Per-axis HRR into the `g` sub-block starting at `base` (i-stride = 1).
/// Mirrors `hrr_step_host` with `di = 1`.
#[cube]
fn origi_hrr_axis<F: Float>(g: &mut Array<F>, base: u32, rirj: F, dj: u32, li_max: u32, lj: u32) {
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

/// On-device scalar origi r2/r4 G-tensor fill + contraction for ONE shell pair.
///
/// Faithful port of `fill_g_tensor_origi` + `contract_origi_r2`/`contract_origi_r4`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn origi_scalar_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
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
    #[comptime] r_power: u32,
) {
    if UNIT_POS == 0u32 {
        // Origi ceiling angular momentum: li_ceil = li + r_power, lj_ceil = lj.
        let li_ceil = li + r_power;
        let nmax = li_ceil + lj;
        let dj = nmax + 1u32; // stride between consecutive j-levels within an axis
        let g_per_axis = (nmax + 1u32) * (lj + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;

        // Zero the single accumulation block.
        let mut oi = 0u32;
        while oi < block_len {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let rirjx = rix - rjx;
        let rirjy = riy - rjy;
        let rirjz = riz - rjz;

        // -- Primitive loop --------------------------------------------------
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                // Pair data in F (norm_i = norm_j = 1.0), matching compute_pdata_host.
                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // Build the overlap G-tensor for this primitive pair into `g`.
                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                origi_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                origi_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                origi_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                if lj >= 1u32 {
                    origi_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj);
                    origi_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj);
                    origi_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj);
                }

                // Contract into the single block over all (ci,cj).
                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;

                        // Cartesian component loops: ci outer, cj inner; lx descending
                        // (matches host cart_comps). Output index = ci_idx*ncj + cj_idx.
                        let mut ci_idx = 0u32;
                        let mut ia = 0u32;
                        while ia <= li {
                            let ix = li - ia;
                            let li_minus_ix = li - ix;
                            let mut ib = 0u32;
                            while ib <= li_minus_ix {
                                let iy = li_minus_ix - ib;
                                let iz = li - ix - iy;

                                let mut cj_idx = 0u32;
                                let mut ja = 0u32;
                                while ja <= lj {
                                    let jx = lj - ja;
                                    let lj_minus_jx = lj - jx;
                                    let mut jb = 0u32;
                                    while jb <= lj_minus_jx {
                                        let jy = lj_minus_jx - jb;
                                        let jz = lj - jx - jy;

                                        let bx = jx * dj + ix;
                                        let by = jy * dj + iy;
                                        let bz = jz * dj + iz;

                                        // g0 reads at i-shift 0/2/4.
                                        let g0x = g[(gx + bx) as usize];
                                        let g0y = g[(gy + by) as usize];
                                        let g0z = g[(gz + bz) as usize];
                                        let g2x = g[(gx + bx + 2u32) as usize];
                                        let g2y = g[(gy + by + 2u32) as usize];
                                        let g2z = g[(gz + bz + 2u32) as usize];

                                        let mut val = F::new(0.0);
                                        if comptime!(r_power == 2u32) {
                                            // r^2: g2x*g0y*g0z + g0x*g2y*g0z + g0x*g0y*g2z
                                            val =
                                                g2x * g0y * g0z + g0x * g2y * g0z + g0x * g0y * g2z;
                                        } else {
                                            // r^4: g4x*g0y*g0z + 2*g2x*g2y*g0z
                                            //    + 2*g2x*g0y*g2z + g0x*g4y*g0z
                                            //    + 2*g0x*g2y*g2z + g0x*g0y*g4z
                                            let g4x = g[(gx + bx + 4u32) as usize];
                                            let g4y = g[(gy + by + 4u32) as usize];
                                            let g4z = g[(gz + bz + 4u32) as usize];
                                            val = g4x * g0y * g0z
                                                + F::new(2.0) * g2x * g2y * g0z
                                                + F::new(2.0) * g2x * g0y * g2z
                                                + g0x * g4y * g0z
                                                + F::new(2.0) * g0x * g2y * g2z
                                                + g0x * g0y * g4z;
                                        }

                                        cart_out[(ci_idx * ncj + cj_idx) as usize] += weight * val;

                                        cj_idx += 1u32;
                                        jb += 1u32;
                                    }
                                    ja += 1u32;
                                }

                                ci_idx += 1u32;
                                ib += 1u32;
                            }
                            ia += 1u32;
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

/// Dispatch [`origi_scalar_kernel`] at `f64` on a resolved backend's client and
/// read back the i-major Cartesian accumulator (one `nci*ncj` block). Generic
/// over `R: Runtime`; device compute is `f64` (module precision policy). The
/// `r_power` comptime arg is selected at the `launch::<f64, R>` call site by a
/// host-side match (CubeCL cannot pass comptime args dynamically).
#[allow(clippy::too_many_arguments)]
fn run_origi_scalar_device<R: Runtime>(
    client: &ComputeClient<R>,
    r_power: u32,
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
    let nmax_u = li_u + (r_power as usize) + lj_u;
    let g_per_axis = (nmax_u + 1) * (lj_u + 1);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = nci * ncj;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    let g_zero = vec![0.0_f64; 3 * g_per_axis];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($rp:expr) => {
            origi_scalar_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3 * g_per_axis) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
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
                $rp,
            )
        };
    }

    if r_power == 2 {
        launch_with!(2u32);
    } else {
        launch_with!(4u32);
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_origi_scalar_device`] — copies the exact arm
/// set used by `one_electron.rs::run_1e_scalar_on_backend` (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_origi_scalar_on_backend(
    backend: &ResolvedBackend,
    r_power: u32,
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
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_origi_scalar_device::<cubecl::cpu::CpuRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_origi_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_origi_scalar_device::<cubecl_cuda::CudaRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_origi_scalar_device::<cubecl_hip::HipRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_origi_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `origi_ip2_kernel`, `#[cube(launch)]`, generic over `F: Float`
//
//  On-device port of the host origi `ip2` pipeline
//  (`fill_g_tensor_origi` -> `g1e_d_j` -> `contract_origi_r2_ip2` /
//  `contract_origi_r4_ip2`). Single work item (`UNIT_POS == 0`): iterates
//  primitive pairs (pi,pj) and contraction pairs (ci,cj) in-kernel and
//  accumulates the 3-component `nci*ncj` Cartesian output summed over all
//  (pi,pj)x(ci,cj). Output layout is COMP-SLOWEST:
//  `out[comp*nci*ncj + ci_idx*ncj + cj_idx]` — matching the host `cart_buf` /
//  `contract_origi_r*_ip2` 3-component buffer exactly. f64-internal / F-output.
//
//  The ip2 path elevates the j-ceiling by one (lj_ceil = lj + 1) so the
//  `G1E_D_J` ket-derivative ladder (`origi_dj_axis`) can read the j+1 level.
//  The `#[comptime] r_power` arg (2 or 4) selects the r2 vs r4 ip2 contraction
//  branch via `comptime!`.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-axis `G1E_D_J` ket-derivative into the `f` sub-block at `f_base`, reading
/// the overlap sub-block at `g_base`. Mirrors the host [`g1e_d_j`] for one axis:
///   f[j=0, i]   = (-2*aj) * g[j=1, i]            (i.e. g[i + dj])
///   f[j>0, i]   = j * g[j-1, i] + (-2*aj) * g[j+1, i]
/// `li_dj` is the i-ceiling covered (li + r_power), `lj` the (un-elevated) j max.
#[cube]
#[allow(clippy::too_many_arguments)]
fn origi_dj_axis<F: Float>(
    g: &Array<F>,
    f: &mut Array<F>,
    g_base: u32,
    f_base: u32,
    li_dj: u32,
    lj: u32,
    dj: u32,
    aj2: F,
) {
    // j = 0 : f[i] = aj2 * g[i + dj]
    let mut i = 0u32;
    while i <= li_dj {
        f[(f_base + i) as usize] = aj2 * g[(g_base + i + dj) as usize];
        i += 1u32;
    }
    // j = 1..=lj : f[j*dj + i] = j*g[j*dj + i - dj] + aj2*g[j*dj + i + dj]
    let mut j = 1u32;
    while j <= lj {
        let mut ii = 0u32;
        while ii <= li_dj {
            let ptr = j * dj + ii;
            f[(f_base + ptr) as usize] = F::cast_from(j) * g[(g_base + ptr - dj) as usize]
                + aj2 * g[(g_base + ptr + dj) as usize];
            ii += 1u32;
        }
        j += 1u32;
    }
}

/// On-device origi `ip2` (r2/r4) G-tensor fill + D_J ladder + 3-component
/// contraction for ONE shell pair.
///
/// Faithful port of `fill_g_tensor_origi` + `g1e_d_j` +
/// `contract_origi_r2_ip2`/`contract_origi_r4_ip2`. The j-ceiling is elevated by
/// one (`lj_ceil = lj + 1`) so the derivative ladder has the j+1 level available.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn origi_ip2_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    cart_out: &mut Array<F>,
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
    #[comptime] r_power: u32,
) {
    if UNIT_POS == 0u32 {
        // Origi ip2 ceiling angular momenta: li_ceil = li + r_power,
        // lj_ceil = lj + 1 (the extra j level feeds the G1E_D_J ladder).
        let li_ceil = li + r_power;
        let lj_ceil = lj + 1u32;
        let nmax = li_ceil + lj_ceil;
        let dj = nmax + 1u32; // stride between consecutive j-levels within an axis
        let g_per_axis = (nmax + 1u32) * (lj_ceil + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_out = 3u32 * block_len;

        // Zero the 3-component accumulation block.
        let mut oi = 0u32;
        while oi < total_out {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let rirjx = rix - rjx;
        let rirjy = riy - rjy;
        let rirjz = riz - rjz;

        let two = F::new(2.0);

        // -- Primitive loop --------------------------------------------------
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                // Pair data in F (norm_i = norm_j = 1.0), matching compute_pdata_host.
                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // Build the overlap G-tensor (g0) for this primitive pair into `g`.
                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                origi_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                origi_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                origi_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                if lj_ceil >= 1u32 {
                    origi_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ceil);
                    origi_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ceil);
                    origi_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ceil);
                }

                // Build g1 = D_J(g0). Zero first, then fill per axis. The host
                // g1e_d_j covers i in 0..=li_dj where li_dj = li + r_power.
                let mut g1i = 0u32;
                while g1i < total_g {
                    g1[g1i as usize] = F::new(0.0);
                    g1i += 1u32;
                }
                let aj2 = F::new(-2.0) * aj;
                let li_dj = li + r_power;
                origi_dj_axis::<F>(g, g1, gx, gx, li_dj, lj, dj, aj2);
                origi_dj_axis::<F>(g, g1, gy, gy, li_dj, lj, dj, aj2);
                origi_dj_axis::<F>(g, g1, gz, gz, li_dj, lj, dj, aj2);

                // Contract into the 3-component block over all (ci,cj).
                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;

                        // Cartesian component loops: ci outer, cj inner; lx descending
                        // (matches host cart_comps). Block index = ci_idx*ncj + cj_idx.
                        let mut ci_idx = 0u32;
                        let mut ia = 0u32;
                        while ia <= li {
                            let ix = li - ia;
                            let li_minus_ix = li - ix;
                            let mut ib = 0u32;
                            while ib <= li_minus_ix {
                                let iy = li_minus_ix - ib;
                                let iz = li - ix - iy;

                                let mut cj_idx = 0u32;
                                let mut ja = 0u32;
                                while ja <= lj {
                                    let jx = lj - ja;
                                    let lj_minus_jx = lj - jx;
                                    let mut jb = 0u32;
                                    while jb <= lj_minus_jx {
                                        let jy = lj_minus_jx - jb;
                                        let jz = lj - jx - jy;

                                        let bx = jx * dj + ix;
                                        let by = jy * dj + iy;
                                        let bz = jz * dj + iz;

                                        // g0 / g1 reads at i-shift 0/2/4.
                                        let g0x = g[(gx + bx) as usize];
                                        let g0y = g[(gy + by) as usize];
                                        let g0z = g[(gz + bz) as usize];
                                        let g1x = g1[(gx + bx) as usize];
                                        let g1y = g1[(gy + by) as usize];
                                        let g1z = g1[(gz + bz) as usize];
                                        let g0x2 = g[(gx + bx + 2u32) as usize];
                                        let g0y2 = g[(gy + by + 2u32) as usize];
                                        let g0z2 = g[(gz + bz + 2u32) as usize];

                                        let mut s0 = F::new(0.0);
                                        let mut s1 = F::new(0.0);
                                        let mut s2 = F::new(0.0);

                                        if comptime!(r_power == 2u32) {
                                            // r2_ip2: g6=g0+2, g7=g1+2 (g1 = D_J(g0)).
                                            let g7x = g1[(gx + bx + 2u32) as usize];
                                            let g7y = g1[(gy + by + 2u32) as usize];
                                            let g7z = g1[(gz + bz + 2u32) as usize];
                                            // s[0] = g7x*g0y*g0z + g1x*g6y*g0z + g1x*g0y*g6z
                                            s0 = g7x * g0y * g0z
                                                + g1x * g0y2 * g0z
                                                + g1x * g0y * g0z2;
                                            // s[1] = g6x*g1y*g0z + g0x*g7y*g0z + g0x*g1y*g6z
                                            s1 = g0x2 * g1y * g0z
                                                + g0x * g7y * g0z
                                                + g0x * g1y * g0z2;
                                            // s[2] = g6x*g0y*g1z + g0x*g6y*g1z + g0x*g0y*g7z
                                            s2 = g0x2 * g0y * g1z
                                                + g0x * g0y2 * g1z
                                                + g0x * g0y * g7z;
                                        } else {
                                            // r4_ip2: g30=g0+4, g31=g1+4 (plus +2 shifts).
                                            let g0x4 = g[(gx + bx + 4u32) as usize];
                                            let g0y4 = g[(gy + by + 4u32) as usize];
                                            let g0z4 = g[(gz + bz + 4u32) as usize];
                                            let g1x2 = g1[(gx + bx + 2u32) as usize];
                                            let g1y2 = g1[(gy + by + 2u32) as usize];
                                            let g1z2 = g1[(gz + bz + 2u32) as usize];
                                            let g1x4 = g1[(gx + bx + 4u32) as usize];
                                            let g1y4 = g1[(gy + by + 4u32) as usize];
                                            let g1z4 = g1[(gz + bz + 4u32) as usize];

                                            // s[0]: g1 on x.
                                            s0 = g1x4 * g0y * g0z
                                                + two * g1x2 * g0y2 * g0z
                                                + two * g1x2 * g0y * g0z2
                                                + g1x * g0y4 * g0z
                                                + two * g1x * g0y2 * g0z2
                                                + g1x * g0y * g0z4;
                                            // s[1]: g1 on y.
                                            s1 = g0x4 * g1y * g0z
                                                + two * g0x2 * g1y2 * g0z
                                                + two * g0x2 * g1y * g0z2
                                                + g0x * g1y4 * g0z
                                                + two * g0x * g1y2 * g0z2
                                                + g0x * g1y * g0z4;
                                            // s[2]: g1 on z.
                                            s2 = g0x4 * g0y * g1z
                                                + two * g0x2 * g0y2 * g1z
                                                + two * g0x2 * g0y * g1z2
                                                + g0x * g0y4 * g1z
                                                + two * g0x * g0y2 * g1z2
                                                + g0x * g0y * g1z4;
                                        }

                                        let blk = ci_idx * ncj + cj_idx;
                                        cart_out[blk as usize] += weight * s0;
                                        cart_out[(block_len + blk) as usize] += weight * s1;
                                        cart_out[(2u32 * block_len + blk) as usize] += weight * s2;

                                        cj_idx += 1u32;
                                        jb += 1u32;
                                    }
                                    ja += 1u32;
                                }

                                ci_idx += 1u32;
                                ib += 1u32;
                            }
                            ia += 1u32;
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

/// Dispatch [`origi_ip2_kernel`] at `f64` on a resolved backend's client and read
/// back the comp-slowest 3-component Cartesian accumulator
/// (`out[comp*nci*ncj + ci*ncj + cj]`). Generic over `R: Runtime`; device compute
/// is `f64` (module precision policy). The `r_power` comptime arg is selected at
/// the `launch::<f64, R>` call site by a host-side match.
#[allow(clippy::too_many_arguments)]
fn run_origi_ip2_device<R: Runtime>(
    client: &ComputeClient<R>,
    r_power: u32,
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
    // ip2 ceiling: li_ceil = li + r_power, lj_ceil = lj + 1.
    let nmax_u = li_u + (r_power as usize) + lj_u + 1;
    let lj_ceil_u = lj_u + 1;
    let g_per_axis = (nmax_u + 1) * (lj_ceil_u + 1);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = 3 * nci * ncj;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    let g_zero = vec![0.0_f64; 3 * g_per_axis];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_zero = vec![0.0_f64; 3 * g_per_axis];
    let g1_h = client.create_from_slice(f64::as_bytes(&g1_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($rp:expr) => {
            origi_ip2_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3 * g_per_axis) },
                unsafe { ArrayArg::from_raw_parts(g1_h.clone(), 3 * g_per_axis) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
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
                $rp,
            )
        };
    }

    if r_power == 2 {
        launch_with!(2u32);
    } else {
        launch_with!(4u32);
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_origi_ip2_device`] — mirrors
/// [`run_origi_scalar_on_backend`] (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_origi_ip2_on_backend(
    backend: &ResolvedBackend,
    r_power: u32,
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
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_origi_ip2_device::<cubecl::cpu::CpuRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_origi_ip2_device::<cubecl_wgpu::WgpuRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_origi_ip2_device::<cubecl_cuda::CudaRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_origi_ip2_device::<cubecl_hip::HipRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_origi_ip2_device::<cubecl_wgpu::WgpuRuntime>(
            client, r_power, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
    }
}

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;

    fn cpu_client() -> cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::Runtime;
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host reference: single-primitive, single-contraction origi scalar block.
    fn host_origi_block(
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        li: u8,
        lj: u8,
        r_power: u32,
    ) -> Vec<f64> {
        let li_ceil = li as u32 + r_power;
        let lj_ceil = lj as u32;
        let nmax = li_ceil + lj_ceil;
        let g_per_axis = ((nmax + 1) * (lj_ceil + 1)) as usize;
        let dj = (nmax + 1) as usize;
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let g = fill_g_tensor_origi(&pd, ri, rj, nmax, lj_ceil);
        if r_power == 2 {
            contract_origi_r2(&g, g_per_axis, li, lj, dj)
        } else {
            contract_origi_r4(&g, g_per_axis, li, lj, dj)
        }
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

    /// Device-vs-host cross-check for r2 AND r4 over (s,s),(p,s),(s,p),(p,p),(d,s).
    #[test]
    fn test_device_matches_host_origi_scalar() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &r_power in &[2u32, 4u32] {
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1), (2, 0)] {
                let host = host_origi_block(ai, aj, ri, rj, li, lj, r_power);
                let dev = run_origi_scalar_device::<cubecl::cpu::CpuRuntime>(
                    &cpu_client(),
                    r_power,
                    li as u32,
                    lj as u32,
                    1,
                    1,
                    1,
                    1,
                    ri,
                    rj,
                    &[ai],
                    &[aj],
                    &[1.0],
                    &[1.0],
                );
                assert_close(&host, &dev, &format!("origi r{r_power} li={li} lj={lj}"));
            }
        }
    }

    /// Multi-primitive / multi-contraction device-vs-host parity (exercises the
    /// in-kernel (pi,pj)x(ci,cj) accumulation against the host loop).
    #[test]
    fn test_device_matches_host_origi_multiprim() {
        let ri = [0.1_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let exps_i = [1.7_f64, 0.5];
        let exps_j = [2.1_f64, 0.8];
        let coeff_i = [0.6_f64, 0.4]; // nprim=2, nctr=1
        let coeff_j = [0.7_f64, 0.3];
        for &r_power in &[2u32, 4u32] {
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (1, 1)] {
                // Host reference: full primitive + contraction accumulation.
                let li_ceil = li as u32 + r_power;
                let nmax = li_ceil + lj as u32;
                let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
                let dj = (nmax + 1) as usize;
                let nci = ncart(li);
                let ncj = ncart(lj);
                let mut host = vec![0.0_f64; nci * ncj];
                for pi in 0..2usize {
                    for pj in 0..2usize {
                        let pd = compute_pdata_host(
                            exps_i[pi], exps_j[pj], ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0,
                            1.0,
                        );
                        let g = fill_g_tensor_origi(&pd, ri, rj, nmax, lj as u32);
                        let prim = if r_power == 2 {
                            contract_origi_r2(&g, g_per_axis, li, lj, dj)
                        } else {
                            contract_origi_r4(&g, g_per_axis, li, lj, dj)
                        };
                        let w = coeff_i[pi] * coeff_j[pj];
                        for k in 0..prim.len() {
                            host[k] += w * prim[k];
                        }
                    }
                }
                let dev = run_origi_scalar_device::<cubecl::cpu::CpuRuntime>(
                    &cpu_client(),
                    r_power,
                    li as u32,
                    lj as u32,
                    2,
                    2,
                    1,
                    1,
                    ri,
                    rj,
                    &exps_i,
                    &exps_j,
                    &coeff_i,
                    &coeff_j,
                );
                assert_close(&host, &dev, &format!("origi-mp r{r_power} li={li} lj={lj}"));
            }
        }
    }

    /// Genericity: the kernel compiles and launches for `F = f32` (s-s yields a
    /// finite output). Same shape as `one_electron.rs`'s f32 smoke test.
    #[test]
    fn test_origi_scalar_kernel_generic_f32() {
        let client = cpu_client();
        let exps_i = [1.0_f32];
        let exps_j = [1.0_f32];
        let coeff_i = [1.0_f32];
        let coeff_j = [1.0_f32];
        // s-s, r_power=2: li=lj=0 -> nmax=2, g_per_axis=3 -> 9 g elements; out_len=1.
        let g_zero = [0.0_f32; 9];
        let out_zero = [0.0_f32; 1];

        let exps_i_h = client.create_from_slice(f32::as_bytes(&exps_i));
        let exps_j_h = client.create_from_slice(f32::as_bytes(&exps_j));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&coeff_i));
        let coeff_j_h = client.create_from_slice(f32::as_bytes(&coeff_j));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        origi_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 9) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32, // rix
            0.0,     // riy
            0.0,     // riz
            0.0,     // rjx
            0.0,     // rjy
            1.4,     // rjz
            SQRTPI as f32,
            std::f64::consts::PI as f32,
            0,    // li
            0,    // lj
            1,    // nprim_i
            1,    // nprim_j
            1,    // nctr_i
            1,    // nctr_j
            2u32, // r_power
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 origi r2 kernel result must be finite");
        assert!(
            out > 0.0,
            "s-s origi r2 f32 result should be positive: {out}"
        );
    }

    // ── ip2 device-vs-host ────────────────────────────────────────────────

    /// Host reference: single-primitive, single-contraction origi ip2
    /// 3-component block (comp-slowest `out[comp*nci*ncj + ci*ncj + cj]`).
    fn host_origi_ip2_block(
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        li: u8,
        lj: u8,
        r_power: u32,
    ) -> Vec<f64> {
        // ip2 ceiling: li_ceil = li + r_power, lj_ceil = lj + 1.
        let li_ceil = li as u32 + r_power;
        let lj_ceil = lj as u32 + 1;
        let nmax = li_ceil + lj_ceil;
        let g_per_axis = ((nmax + 1) * (lj_ceil + 1)) as usize;
        let dj = (nmax + 1) as usize;
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let g = fill_g_tensor_origi(&pd, ri, rj, nmax, lj_ceil);
        if r_power == 2 {
            contract_origi_r2_ip2(&g, g_per_axis, li, lj, dj, aj)
        } else {
            contract_origi_r4_ip2(&g, g_per_axis, li, lj, dj, aj)
        }
    }

    /// Device-vs-host cross-check for r2_ip2 AND r4_ip2 over the required pairs
    /// (s,s),(p,s),(s,p),(p,p),(d,s), 3-component comp-slowest layout, atol 1e-12.
    #[test]
    fn test_device_matches_host_origi_ip2() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &r_power in &[2u32, 4u32] {
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1), (2, 0)] {
                let host = host_origi_ip2_block(ai, aj, ri, rj, li, lj, r_power);
                let dev = run_origi_ip2_device::<cubecl::cpu::CpuRuntime>(
                    &cpu_client(),
                    r_power,
                    li as u32,
                    lj as u32,
                    1,
                    1,
                    1,
                    1,
                    ri,
                    rj,
                    &[ai],
                    &[aj],
                    &[1.0],
                    &[1.0],
                );
                assert_close(
                    &host,
                    &dev,
                    &format!("origi-ip2 r{r_power} li={li} lj={lj}"),
                );
            }
        }
    }

    /// Multi-primitive / multi-contraction ip2 device-vs-host parity (exercises
    /// the in-kernel (pi,pj)x(ci,cj) accumulation against the host loop).
    #[test]
    fn test_device_matches_host_origi_ip2_multiprim() {
        let ri = [0.1_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let exps_i = [1.7_f64, 0.5];
        let exps_j = [2.1_f64, 0.8];
        let coeff_i = [0.6_f64, 0.4]; // nprim=2, nctr=1
        let coeff_j = [0.7_f64, 0.3];
        for &r_power in &[2u32, 4u32] {
            for &(li, lj) in &[(0u8, 0u8), (1, 0), (1, 1)] {
                let li_ceil = li as u32 + r_power;
                let lj_ceil = lj as u32 + 1;
                let nmax = li_ceil + lj_ceil;
                let g_per_axis = ((nmax + 1) * (lj_ceil + 1)) as usize;
                let dj = (nmax + 1) as usize;
                let nci = ncart(li);
                let ncj = ncart(lj);
                let mut host = vec![0.0_f64; nci * ncj * 3];
                for pi in 0..2usize {
                    for pj in 0..2usize {
                        let pd = compute_pdata_host(
                            exps_i[pi], exps_j[pj], ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0,
                            1.0,
                        );
                        let g = fill_g_tensor_origi(&pd, ri, rj, nmax, lj_ceil);
                        let prim = if r_power == 2 {
                            contract_origi_r2_ip2(&g, g_per_axis, li, lj, dj, exps_j[pj])
                        } else {
                            contract_origi_r4_ip2(&g, g_per_axis, li, lj, dj, exps_j[pj])
                        };
                        let w = coeff_i[pi] * coeff_j[pj];
                        for k in 0..prim.len() {
                            host[k] += w * prim[k];
                        }
                    }
                }
                let dev = run_origi_ip2_device::<cubecl::cpu::CpuRuntime>(
                    &cpu_client(),
                    r_power,
                    li as u32,
                    lj as u32,
                    2,
                    2,
                    1,
                    1,
                    ri,
                    rj,
                    &exps_i,
                    &exps_j,
                    &coeff_i,
                    &coeff_j,
                );
                assert_close(
                    &host,
                    &dev,
                    &format!("origi-ip2-mp r{r_power} li={li} lj={lj}"),
                );
            }
        }
    }

    /// Genericity: the ip2 kernel compiles and launches for `F = f32` (s-s yields
    /// a 3-component finite output).
    #[test]
    fn test_origi_ip2_kernel_generic_f32() {
        let client = cpu_client();
        let exps_i = [1.0_f32];
        let exps_j = [1.0_f32];
        let coeff_i = [1.0_f32];
        let coeff_j = [1.0_f32];
        // s-s, r_power=2: li=lj=0 -> li_ceil=2, lj_ceil=1, nmax=3,
        // g_per_axis=(nmax+1)*(lj_ceil+1)=4*2=8 -> 24 g elements; out_len=3.
        let g_per_axis = 8usize;
        let g_zero = [0.0_f32; 24];
        let g1_zero = [0.0_f32; 24];
        let out_zero = [0.0_f32; 3];

        let exps_i_h = client.create_from_slice(f32::as_bytes(&exps_i));
        let exps_j_h = client.create_from_slice(f32::as_bytes(&exps_j));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&coeff_i));
        let coeff_j_h = client.create_from_slice(f32::as_bytes(&coeff_j));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let g1_h = client.create_from_slice(f32::as_bytes(&g1_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        origi_ip2_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_per_axis) },
            unsafe { ArrayArg::from_raw_parts(g1_h, 3 * g_per_axis) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 3) },
            0.0_f32, // rix
            0.0,     // riy
            0.0,     // riz
            0.0,     // rjx
            0.0,     // rjy
            1.4,     // rjz
            SQRTPI as f32,
            std::f64::consts::PI as f32,
            0,    // li
            0,    // lj
            1,    // nprim_i
            1,    // nprim_j
            1,    // nctr_i
            1,    // nctr_j
            2u32, // r_power
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw);
        for c in 0..3 {
            assert!(
                out[c].is_finite(),
                "f32 origi r2_ip2 kernel component {c} must be finite"
            );
        }
    }
}
