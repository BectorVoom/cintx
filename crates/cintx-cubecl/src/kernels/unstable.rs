//! Kernel launch functions for unstable-source API families.
//!
//! Phase 14 — all five unstable-source families implemented:
//!   - grids: grid-point integrals with NGRIDS env parameter (cint1e_grids.c)
//!   - breit: Breit spinor-only 2e integrals (breit.c)
//!   - origi: origin-displaced r^n one-electron integrals (cint1e_a.c)
//!   - origk: origin-k-displaced 3c1e integrals (cint3c1e_a.c)
//!   - ssc: spin-spin contact 3c2e integral (cint3c2e.c)

use crate::backend::ResolvedBackend;
use crate::math::obara_saika::{hrr_step_host, vrr_step_host, vrr_2e_step_host};
use crate::math::pdata::{PairData, compute_pdata_host};
use crate::math::rys::{rys_root1_host, rys_root2_host, rys_roots_host};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, cart_to_sph_3c1e, cart_to_sph_3c2e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_4d;
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats, planner::GridsEnvParams};
use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
/// Same as one_electron.rs: CINTcommon_fac_sp(l).
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

/// Apply nabla_i derivative to G-tensor (bra gradient).
///
/// Formula: `D_i[g][j, i] = i * g[j, i-1] - 2*ai * g[j, i+1]`
/// where i is the bra VRR index (stride 1) and j is the HRR ket index (stride dj).
/// For i=0: `D_i = -2*ai * g[j, 1]`.
///
/// The derivative requires the source G-tensor to have one extra bra level (nmax+1).
/// Result is stored in `df` which has the same layout as `g`.
fn nabla_i_host(
    df: &mut [f64],
    g: &[f64],
    ai: f64,
    li: u32,
    lj: u32,
    nmax: u32,
    g_per_axis: usize,
) {
    let ai2 = -2.0 * ai;
    let dj_stride = (nmax + 1) as usize; // stride between j-levels

    for j in 0..=(lj as usize) {
        // i = 0: D_i[j, 0] = -2*ai * g[j, 1]
        df[j * dj_stride] = ai2 * g[j * dj_stride + 1];

        // i = 1..li: D_i[j, i] = i * g[j, i-1] - 2*ai * g[j, i+1]
        for i in 1..=(li as usize) {
            let i_f = i as f64;
            let val = i_f * g[j * dj_stride + (i - 1)] + ai2 * g[j * dj_stride + (i + 1)];
            df[j * dj_stride + i] = val;
        }
    }

    let _ = g_per_axis; // used for layout validation elsewhere
}

/// Apply nabla_j derivative to G-tensor (ket gradient).
///
/// Formula: `D_j[g][j, i] = j * g[j-1, i] - 2*aj * g[j+1, i]`
/// where j is the HRR ket index (stride dj) and i is the bra VRR index (stride 1).
/// For j=0: `D_j = -2*aj * g[1, i]`.
///
/// The derivative requires the source G-tensor to have one extra ket level (lj+1).
fn nabla_j_host(
    df: &mut [f64],
    g: &[f64],
    aj: f64,
    li: u32,
    lj: u32,
    nmax: u32,
) {
    let aj2 = -2.0 * aj;
    let dj_stride = (nmax + 1) as usize;

    // j = 0: D_j[0, i] = -2*aj * g[1, i]
    for i in 0..=(li as usize) {
        df[i] = aj2 * g[dj_stride + i];
    }

    // j = 1..lj: D_j[j, i] = j * g[j-1, i] - 2*aj * g[j+1, i]
    for j in 1..=(lj as usize) {
        let j_f = j as f64;
        for i in 0..=(li as usize) {
            let val = j_f * g[(j - 1) * dj_stride + i] + aj2 * g[(j + 1) * dj_stride + i];
            df[j * dj_stride + i] = val;
        }
    }
}

/// Apply nabla_i and nabla_j derivatives to a full 3-axis G-tensor (g[gx|gy|gz]).
///
/// Returns a 3-axis derivative tensor of the same layout.
fn apply_nabla_i_3axis(
    g: &[f64],
    ai: f64,
    li: u32,
    lj: u32,
    nmax: u32,
    g_per_axis: usize,
) -> Vec<f64> {
    let mut df = vec![0.0_f64; 3 * g_per_axis];
    nabla_i_host(&mut df[0..g_per_axis], &g[0..g_per_axis], ai, li, lj, nmax, g_per_axis);
    nabla_i_host(&mut df[g_per_axis..2*g_per_axis], &g[g_per_axis..2*g_per_axis], ai, li, lj, nmax, g_per_axis);
    nabla_i_host(&mut df[2*g_per_axis..3*g_per_axis], &g[2*g_per_axis..3*g_per_axis], ai, li, lj, nmax, g_per_axis);
    df
}

fn apply_nabla_j_3axis(
    g: &[f64],
    aj: f64,
    li: u32,
    lj: u32,
    nmax: u32,
    g_per_axis: usize,
) -> Vec<f64> {
    let mut df = vec![0.0_f64; 3 * g_per_axis];
    nabla_j_host(&mut df[0..g_per_axis], &g[0..g_per_axis], aj, li, lj, nmax);
    nabla_j_host(&mut df[g_per_axis..2*g_per_axis], &g[g_per_axis..2*g_per_axis], aj, li, lj, nmax);
    nabla_j_host(&mut df[2*g_per_axis..3*g_per_axis], &g[2*g_per_axis..3*g_per_axis], aj, li, lj, nmax);
    df
}

/// Compute the Rys-quadrature G-tensor for one primitive pair and one "nuclear" center.
///
/// Uses the same algorithm as nuclear attraction in one_electron.rs, but with
/// a user-supplied center `rc` (grid point) instead of atomic coordinates.
/// The prefactor is `2*pi * fac / zeta` (no atomic charge factor).
///
/// Returns accumulated Cartesian integral buffer of size nci * ncj.
fn grids_contract_nuclear_like(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    li: u8,
    lj: u8,
    nmax_extra: u32, // extra j-levels for derivative operators (0 for base)
    lj_extra: u32,   // extra HRR ket levels for derivative operators
) -> Vec<f64> {
    let nmax = (li + lj) as u32 + nmax_extra;
    let lj_hrr = lj as u32 + lj_extra;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let mut out = vec![0.0_f64; nci * ncj];

    let nrys_roots = (li + lj) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj_hrr + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];

    // Vector from P to grid center: crij = rc - rp
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];

    // Boys argument x = zeta * |P - rc|^2
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    // Get Rys roots and weights
    let (u_arr, w_arr) = if nrys_roots == 1 {
        let (u0, w0) = rys_root1_host(x_boys);
        ([u0, 0.0], [w0, 0.0])
    } else {
        let (u, w) = rys_root2_host(x_boys);
        (u, w)
    };

    // Grids prefactor: 2*pi * fac / zeta (same as nuclear but no -Z_c charge factor)
    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;

    // For each Rys root
    for n in 0..nrys_roots as usize {
        let u_n = u_arr[n];
        let w_n = w_arr[n];

        let tau = u_n / (1.0 + u_n);

        // Modified recurrence coefficient b10 = aij2 * (1 - tau)
        let rt = pd.aij2 * (1.0 - tau);

        // VRR c00: (P - ri) + tau * crij = (P - ri) + tau * (rc - rp)
        let c00 = [
            (rp[0] - ri[0]) + tau * crij[0],
            (rp[1] - ri[1]) + tau * crij[1],
            (rp[2] - ri[2]) + tau * crij[2],
        ];

        let gz0_root = fac1 * w_n;

        let mut g_root = vec![0.0_f64; 3 * g_per_axis];

        let gx_off = 0;
        let gy_off = g_per_axis;
        let gz_off = 2 * g_per_axis;

        g_root[gx_off] = 1.0;
        g_root[gy_off] = 1.0;
        g_root[gz_off] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g_root[gx_off..gx_off + g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[gy_off..gy_off + g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[gz_off..gz_off + g_per_axis], c00[2], rt, nmax, 1);
        }

        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        if lj_hrr >= 1 {
            let di = 1u32;
            let dj_stride = nmax + 1;
            hrr_step_host(
                &mut g_root[gx_off..gx_off + g_per_axis],
                rirj[0],
                di,
                dj_stride,
                nmax,
                lj_hrr,
            );
            hrr_step_host(
                &mut g_root[gy_off..gy_off + g_per_axis],
                rirj[1],
                di,
                dj_stride,
                nmax,
                lj_hrr,
            );
            hrr_step_host(
                &mut g_root[gz_off..gz_off + g_per_axis],
                rirj[2],
                di,
                dj_stride,
                nmax,
                lj_hrr,
            );
        }

        // Contract this root's contribution (base operator: gx*gy*gz)
        for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let vx = g_root[gx_off + jx as usize * dj + ix as usize];
                let vy = g_root[gy_off + jy as usize * dj + iy as usize];
                let vz = g_root[gz_off + jz as usize * dj + iz as usize];
                out[ci_idx * ncj + cj_idx] += vx * vy * vz;
            }
        }
    }

    out
}

/// Compute grids-ip (nabla_i) Cartesian integral for one primitive pair at one grid point.
///
/// Returns 3 * nci * ncj elements (3 Cartesian derivative components: x, y, z).
/// Output layout: [comp][ci][cj] where comp=0..2 is the nabla_i direction.
fn grids_contract_ip(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // IP requires G-tensor built with li+1 bra levels to allow nabla_i.
    // nmax for VRR = (li+1) + lj (we add 1 to bra for derivative)
    let nmax = (li + 1 + lj) as u32;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nrys_roots = (li + 1 + lj) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    let (u_arr, w_arr) = if nrys_roots == 1 {
        let (u0, w0) = rys_root1_host(x_boys);
        ([u0, 0.0], [w0, 0.0])
    } else {
        let (u, w) = rys_root2_host(x_boys);
        (u, w)
    };

    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // Accumulate over Rys roots
    let mut g0_acc = vec![0.0_f64; 3 * g_per_axis];

    for n in 0..nrys_roots as usize {
        let u_n = u_arr[n];
        let w_n = w_arr[n];
        let tau = u_n / (1.0 + u_n);
        let rt = pd.aij2 * (1.0 - tau);
        let c00 = [
            (rp[0] - ri[0]) + tau * crij[0],
            (rp[1] - ri[1]) + tau * crij[1],
            (rp[2] - ri[2]) + tau * crij[2],
        ];
        let gz0_root = fac1 * w_n;

        let mut g_root = vec![0.0_f64; 3 * g_per_axis];
        g_root[0] = 1.0;
        g_root[g_per_axis] = 1.0;
        g_root[2 * g_per_axis] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g_root[0..g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[g_per_axis..2*g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[2*g_per_axis..3*g_per_axis], c00[2], rt, nmax, 1);
        }

        if lj >= 1 {
            hrr_step_host(&mut g_root[0..g_per_axis], rirj[0], 1, nmax+1, nmax, lj as u32);
            hrr_step_host(&mut g_root[g_per_axis..2*g_per_axis], rirj[1], 1, nmax+1, nmax, lj as u32);
            hrr_step_host(&mut g_root[2*g_per_axis..3*g_per_axis], rirj[2], 1, nmax+1, nmax, lj as u32);
        }

        for k in 0..3 * g_per_axis {
            g0_acc[k] += g_root[k];
        }
    }

    // Apply nabla_i to the accumulated G-tensor
    let g1 = apply_nabla_i_3axis(&g0_acc, ai, li as u32, lj as u32, nmax, g_per_axis);

    // Contract: 3 components (x, y, z)
    let mut out = vec![0.0_f64; 3 * nci * ncj];

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            // Component x: D_ix * g0y * g0z
            let g0x = g0_acc[jx as usize * dj + ix as usize];
            let g0y = g0_acc[g_per_axis + jy as usize * dj + iy as usize];
            let g0z = g0_acc[2*g_per_axis + jz as usize * dj + iz as usize];
            let g1x = g1[jx as usize * dj + ix as usize];
            let g1y = g1[g_per_axis + jy as usize * dj + iy as usize];
            let g1z = g1[2*g_per_axis + jz as usize * dj + iz as usize];

            let base_idx = ci_idx * ncj + cj_idx;
            out[0 * nci * ncj + base_idx] += g1x * g0y * g0z;  // comp x
            out[1 * nci * ncj + base_idx] += g0x * g1y * g0z;  // comp y
            out[2 * nci * ncj + base_idx] += g0x * g0y * g1z;  // comp z
        }
    }

    out
}

/// Compute grids-ipip (nabla_i^2) Cartesian integral for one primitive pair.
///
/// Returns 9 * nci * ncj elements (9 = 3x3 second-derivative components).
/// Output layout matches libcint: column-transposed ordering.
fn grids_contract_ipip(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // IPIP requires G-tensor with li+2 bra levels.
    let nmax = (li + 2 + lj) as u32;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nrys_roots = (li + 2 + lj) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    let (u_arr, w_arr) = if nrys_roots == 1 {
        let (u0, w0) = rys_root1_host(x_boys);
        ([u0, 0.0], [w0, 0.0])
    } else {
        let (u, w) = rys_root2_host(x_boys);
        (u, w)
    };

    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    let mut g0_acc = vec![0.0_f64; 3 * g_per_axis];

    for n in 0..nrys_roots as usize {
        let u_n = u_arr[n];
        let w_n = w_arr[n];
        let tau = u_n / (1.0 + u_n);
        let rt = pd.aij2 * (1.0 - tau);
        let c00 = [
            (rp[0] - ri[0]) + tau * crij[0],
            (rp[1] - ri[1]) + tau * crij[1],
            (rp[2] - ri[2]) + tau * crij[2],
        ];
        let gz0_root = fac1 * w_n;

        let mut g_root = vec![0.0_f64; 3 * g_per_axis];
        g_root[0] = 1.0;
        g_root[g_per_axis] = 1.0;
        g_root[2*g_per_axis] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g_root[0..g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[g_per_axis..2*g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[2*g_per_axis..3*g_per_axis], c00[2], rt, nmax, 1);
        }

        if lj >= 1 {
            hrr_step_host(&mut g_root[0..g_per_axis], rirj[0], 1, nmax+1, nmax, lj as u32);
            hrr_step_host(&mut g_root[g_per_axis..2*g_per_axis], rirj[1], 1, nmax+1, nmax, lj as u32);
            hrr_step_host(&mut g_root[2*g_per_axis..3*g_per_axis], rirj[2], 1, nmax+1, nmax, lj as u32);
        }

        for k in 0..3 * g_per_axis {
            g0_acc[k] += g_root[k];
        }
    }

    // g1 = D_i(g0) applied with li+1 bra levels
    let g1 = apply_nabla_i_3axis(&g0_acc, ai, li as u32 + 1, lj as u32, nmax, g_per_axis);
    // g2 = D_i(g0) applied with li bra levels (same as g1 but li not li+1)
    let g2 = apply_nabla_i_3axis(&g0_acc, ai, li as u32, lj as u32, nmax, g_per_axis);
    // g3 = D_i(g1) applied to g1 with li levels
    let g3 = apply_nabla_i_3axis(&g1, ai, li as u32, lj as u32, nmax, g_per_axis);

    // Contract: 9 components
    // libcint ipip layout (from autocode): s[0..8] -> gout with transposition
    // gout[n*9+0] = s0, gout[n*9+1] = s3, gout[n*9+2] = s6, ... (column-transposed)
    let mut s = vec![0.0_f64; 9 * nci * ncj];

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let base = ci_idx * ncj + cj_idx;
            let g0x = g0_acc[jx as usize * dj + ix as usize];
            let g0y = g0_acc[g_per_axis + jy as usize * dj + iy as usize];
            let g0z = g0_acc[2*g_per_axis + jz as usize * dj + iz as usize];
            let g1x = g1[jx as usize * dj + ix as usize];
            let g1y = g1[g_per_axis + jy as usize * dj + iy as usize];
            let g1z = g1[2*g_per_axis + jz as usize * dj + iz as usize];
            let g2x = g2[jx as usize * dj + ix as usize];
            let g2y = g2[g_per_axis + jy as usize * dj + iy as usize];
            let g2z = g2[2*g_per_axis + jz as usize * dj + iz as usize];
            let g3x = g3[jx as usize * dj + ix as usize];
            let g3y = g3[g_per_axis + jy as usize * dj + iy as usize];
            let g3z = g3[2*g_per_axis + jz as usize * dj + iz as usize];

            // s[0..8] = [g3x*g0y*g0z, g2x*g1y*g0z, g2x*g0y*g1z,
            //            g1x*g2y*g0z, g0x*g3y*g0z, g0x*g2y*g1z,
            //            g1x*g0y*g2z, g0x*g1y*g2z, g0x*g0y*g3z]
            let s0 = g3x * g0y * g0z;
            let s1 = g2x * g1y * g0z;
            let s2 = g2x * g0y * g1z;
            let s3 = g1x * g2y * g0z;
            let s4 = g0x * g3y * g0z;
            let s5 = g0x * g2y * g1z;
            let s6 = g1x * g0y * g2z;
            let s7 = g0x * g1y * g2z;
            let s8 = g0x * g0y * g3z;

            // libcint ipip gout[n*9+k]: s0 s3 s6 s1 s4 s7 s2 s5 s8 (column-transposed)
            s[0 * nci * ncj + base] += s0;
            s[1 * nci * ncj + base] += s3;
            s[2 * nci * ncj + base] += s6;
            s[3 * nci * ncj + base] += s1;
            s[4 * nci * ncj + base] += s4;
            s[5 * nci * ncj + base] += s7;
            s[6 * nci * ncj + base] += s2;
            s[7 * nci * ncj + base] += s5;
            s[8 * nci * ncj + base] += s8;
        }
    }

    s
}

/// Compute grids-ipvip (nabla_i x nabla_j) Cartesian integral.
///
/// Returns 9 * nci * ncj elements. Same 9-component layout as ipip but with
/// nabla on both bra (i) and ket (j).
fn grids_contract_ipvip(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    aj: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // IPVIP requires li+1 bra and lj+1 ket levels.
    let nmax = (li + 1 + lj + 1) as u32;
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nrys_roots = (li + 1 + lj + 1) as u32 / 2 + 1;
    let g_per_axis = ((nmax + 1) * (lj as u32 + 2)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
    let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

    let (u_arr, w_arr) = if nrys_roots == 1 {
        let (u0, w0) = rys_root1_host(x_boys);
        ([u0, 0.0], [w0, 0.0])
    } else {
        let (u, w) = rys_root2_host(x_boys);
        (u, w)
    };

    let fac1 = 2.0 * std::f64::consts::PI * pd.fac / pd.zeta_ab;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    let mut g0_acc = vec![0.0_f64; 3 * g_per_axis];

    for n in 0..nrys_roots as usize {
        let u_n = u_arr[n];
        let w_n = w_arr[n];
        let tau = u_n / (1.0 + u_n);
        let rt = pd.aij2 * (1.0 - tau);
        let c00 = [
            (rp[0] - ri[0]) + tau * crij[0],
            (rp[1] - ri[1]) + tau * crij[1],
            (rp[2] - ri[2]) + tau * crij[2],
        ];
        let gz0_root = fac1 * w_n;

        let mut g_root = vec![0.0_f64; 3 * g_per_axis];
        g_root[0] = 1.0;
        g_root[g_per_axis] = 1.0;
        g_root[2*g_per_axis] = gz0_root;

        if nmax >= 1 {
            vrr_2e_step_host(&mut g_root[0..g_per_axis], c00[0], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[g_per_axis..2*g_per_axis], c00[1], rt, nmax, 1);
            vrr_2e_step_host(&mut g_root[2*g_per_axis..3*g_per_axis], c00[2], rt, nmax, 1);
        }

        // HRR to lj+1 for derivative on ket
        let lj_hrr = lj as u32 + 1;
        hrr_step_host(&mut g_root[0..g_per_axis], rirj[0], 1, nmax+1, nmax, lj_hrr);
        hrr_step_host(&mut g_root[g_per_axis..2*g_per_axis], rirj[1], 1, nmax+1, nmax, lj_hrr);
        hrr_step_host(&mut g_root[2*g_per_axis..3*g_per_axis], rirj[2], 1, nmax+1, nmax, lj_hrr);

        for k in 0..3 * g_per_axis {
            g0_acc[k] += g_root[k];
        }
    }

    // g1 = D_j(g0)
    let g1 = apply_nabla_j_3axis(&g0_acc, aj, li as u32 + 1, lj as u32, nmax, g_per_axis);
    // g2 = D_i(g0)
    let g2 = apply_nabla_i_3axis(&g0_acc, ai, li as u32, lj as u32 + 1, nmax, g_per_axis);
    // g3 = D_i(g1)
    let g3 = apply_nabla_i_3axis(&g1, ai, li as u32, lj as u32, nmax, g_per_axis);

    let mut s = vec![0.0_f64; 9 * nci * ncj];

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let base = ci_idx * ncj + cj_idx;
            let g0x = g0_acc[jx as usize * dj + ix as usize];
            let g0y = g0_acc[g_per_axis + jy as usize * dj + iy as usize];
            let g0z = g0_acc[2*g_per_axis + jz as usize * dj + iz as usize];
            let g1x = g1[jx as usize * dj + ix as usize];
            let g1y = g1[g_per_axis + jy as usize * dj + iy as usize];
            let g1z = g1[2*g_per_axis + jz as usize * dj + iz as usize];
            let g2x = g2[jx as usize * dj + ix as usize];
            let g2y = g2[g_per_axis + jy as usize * dj + iy as usize];
            let g2z = g2[2*g_per_axis + jz as usize * dj + iz as usize];
            let g3x = g3[jx as usize * dj + ix as usize];
            let g3y = g3[g_per_axis + jy as usize * dj + iy as usize];
            let g3z = g3[2*g_per_axis + jz as usize * dj + iz as usize];

            let s0 = g3x * g0y * g0z;
            let s1 = g2x * g1y * g0z;
            let s2 = g2x * g0y * g1z;
            let s3 = g1x * g2y * g0z;
            let s4 = g0x * g3y * g0z;
            let s5 = g0x * g2y * g1z;
            let s6 = g1x * g0y * g2z;
            let s7 = g0x * g1y * g2z;
            let s8 = g0x * g0y * g3z;

            s[0 * nci * ncj + base] += s0;
            s[1 * nci * ncj + base] += s1;
            s[2 * nci * ncj + base] += s2;
            s[3 * nci * ncj + base] += s3;
            s[4 * nci * ncj + base] += s4;
            s[5 * nci * ncj + base] += s5;
            s[6 * nci * ncj + base] += s6;
            s[7 * nci * ncj + base] += s7;
            s[8 * nci * ncj + base] += s8;
        }
    }

    s
}

/// Compute grids-spvsp (sigma-p . 1/r . sigma-p) Cartesian integral.
///
/// Returns 4 * nci * ncj elements. Uses the same G-tensor as ipvip but
/// combines the 9 intermediates into 4 spvsp components per libcint autocode.
///
/// spvsp gout: [s5-s7, s6-s2, s1-s3, s0+s4+s8]
fn grids_contract_spvsp(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    rc: [f64; 3],
    ai: f64,
    aj: f64,
    li: u8,
    lj: u8,
) -> Vec<f64> {
    // Same G-tensor as ipvip
    let ipvip = grids_contract_ipvip(pd, ri, rj, rc, ai, aj, li, lj);
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nij = nci * ncj;

    // ipvip s[0..8]: s0 s1 s2 s3 s4 s5 s6 s7 s8
    // spvsp gout: s5-s7, s6-s2, s1-s3, s0+s4+s8
    let mut out = vec![0.0_f64; 4 * nij];

    for k in 0..nij {
        let s0 = ipvip[0 * nij + k];
        let s1 = ipvip[1 * nij + k];
        let s2 = ipvip[2 * nij + k];
        let s3 = ipvip[3 * nij + k];
        let s4 = ipvip[4 * nij + k];
        let s5 = ipvip[5 * nij + k];
        let s6 = ipvip[6 * nij + k];
        let s7 = ipvip[7 * nij + k];
        let s8 = ipvip[8 * nij + k];

        out[0 * nij + k] = s5 - s7;
        out[1 * nij + k] = s6 - s2;
        out[2 * nij + k] = s1 - s3;
        out[3 * nij + k] = s0 + s4 + s8;
    }

    out
}

/// Core grids kernel: compute integrals for all grid points in `grids_params`.
///
/// For each grid point g, computes the integral and writes to:
///   `staging[comp * ngrids * nsi * nsj + j_sph * ngrids * nsi + i_sph * ngrids + g]`
///
/// This matches the libcint c2s_sph_1e_grids output layout where ngrids is the
/// innermost (fastest-varying) index.
fn launch_grids_kernel(
    plan: &ExecutionPlan<'_>,
    grids_params: &GridsEnvParams,
    ncomp: usize,
    staging: &mut [f64],
    contract_fn: impl Fn(
        &crate::math::pdata::PairData,
        [f64; 3],
        [f64; 3],
        [f64; 3],
        f64,
        f64,
        u8,
        u8,
    ) -> Vec<f64>,
) -> Result<(), cintxRsError> {
    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;

    let ngrids = grids_params.ngrids;
    let grid_coords = &grids_params.grid_coords;

    if grid_coords.len() < ngrids {
        return Err(cintxRsError::InvalidEnvParam {
            param: "PTR_GRIDS",
            reason: format!(
                "grid_coords length {} < ngrids {}",
                grid_coords.len(),
                ngrids
            ),
        });
    }

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    // For each grid point
    for g in 0..ngrids {
        let rc = grid_coords[g];

        // Accumulate over primitive pairs for this grid point
        let mut cart_buf = vec![0.0_f64; ncomp * nci * ncj];

        for pi in 0..n_prim_i {
            let ai = shell_i.exponents[pi];
            for pj in 0..n_prim_j {
                let aj = shell_j.exponents[pj];

                let pd = compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );

                let prim_buf = contract_fn(&pd, ri, rj, rc, ai, aj, li, lj);

                // Accumulate over contractions
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

        // Apply sp_scale
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_buf.iter_mut() {
                *v *= sp_scale;
            }
        }

        // Transform cart to sph for each component and write to staging
        // Output layout: staging[comp * ngrids * nsi * nsj + j * ngrids * nsi + i * ngrids + g]
        for c in 0..ncomp {
            let cart_comp = &cart_buf[c * nci * ncj..(c + 1) * nci * ncj];
            let mut sph_buf = vec![0.0_f64; nsi * nsj];
            cart_to_sph_1e(cart_comp, &mut sph_buf, li, lj);

            let comp_offset = c * ngrids * nsi * nsj;
            for j_sph in 0..nsj {
                for i_sph in 0..nsi {
                    // libcint layout: out[g + i * ngrids + j * ngrids * ni]
                    let idx = comp_offset + j_sph * ngrids * nsi + i_sph * ngrids + g;
                    if idx < staging.len() {
                        staging[idx] = sph_buf[i_sph * nsj + j_sph];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Build ExecutionStats for a completed grids kernel call.
fn grids_stats(plan: &ExecutionPlan<'_>, written: usize) -> ExecutionStats {
    let staging_bytes = written * std::mem::size_of::<f64>();
    ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: staging_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: staging_bytes,
        not0: written as i32, // conservatively report all as non-zero
        fallback_reason: plan.workspace.fallback_reason,
    }
}

/// Real grids kernel launch function.
///
/// Implements int1e_grids_sph and derivative variants.
/// Output layout matches libcint c2s_sph_1e_grids:
///   staging[comp * ngrids * di * dj + j * ngrids * di + i * ngrids + g]
pub fn launch_grids(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "grids" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_grids",
            detail: format!(
                "canonical_family mismatch: expected grids, got {}",
                specialization.canonical_family()
            ),
        });
    }

    // Suppress backend: host-side pipeline executes natively.
    let _ = backend;

    let shells = plan.shells.as_slice();
    if shells.len() < 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_grids",
            detail: "grids kernel requires exactly 2 shells".to_owned(),
        });
    }

    // Grids params must be populated by the caller (raw compat reads from env[11..12]).
    let grids_params = plan
        .operator_env_params
        .grids_params
        .as_ref()
        .ok_or_else(|| cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            reason: "grids_params not populated — caller must set env[11] and env[12]".to_owned(),
        })?;

    // The env array is not available in the ExecutionPlan; we need it passed through a side
    // channel. For the raw compat path, the kernel receives env through grids_params.ptr_grids
    // which was already adjusted to point at the correct grid range start.
    // We need the actual env array. This is passed as a "basis extension" in the plan.
    // However, the current plan API doesn't carry env directly. We need to access it.
    //
    // Grid coordinates are carried in grids_params.grid_coords, populated by the caller
    // (eval_raw extracts from env[ptr_grids..] at call time before dispatching the plan).

    let op_name = plan.descriptor.operator_name();

    let ngrids = grids_params.ngrids;
    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let nsi = nsph(li);
    let nsj = nsph(lj);

    match op_name {
        "grids" => {
            let ncomp = 1usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(plan, grids_params, ncomp, &mut staging[..required], |pd, ri, rj, rc, _ai, _aj, li, lj| {
                grids_contract_nuclear_like(pd, ri, rj, rc, li, lj, 0, 0)
            })?;
            Ok(grids_stats(plan, required))
        }
        "grids_ip" => {
            let ncomp = 3usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(plan, grids_params, ncomp, &mut staging[..required], |pd, ri, rj, rc, ai, _aj, li, lj| {
                grids_contract_ip(pd, ri, rj, rc, ai, li, lj)
            })?;
            Ok(grids_stats(plan, required))
        }
        "grids_ipvip" => {
            let ncomp = 9usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(plan, grids_params, ncomp, &mut staging[..required], |pd, ri, rj, rc, ai, aj, li, lj| {
                grids_contract_ipvip(pd, ri, rj, rc, ai, aj, li, lj)
            })?;
            Ok(grids_stats(plan, required))
        }
        "grids_spvsp" => {
            let ncomp = 4usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(plan, grids_params, ncomp, &mut staging[..required], |pd, ri, rj, rc, ai, aj, li, lj| {
                grids_contract_spvsp(pd, ri, rj, rc, ai, aj, li, lj)
            })?;
            Ok(grids_stats(plan, required))
        }
        "grids_ipip" => {
            let ncomp = 9usize;
            let required = ncomp * ngrids * nsi * nsj;
            if staging.len() < required {
                return Err(cintxRsError::BufferTooSmall {
                    required,
                    provided: staging.len(),
                });
            }
            launch_grids_kernel(plan, grids_params, ncomp, &mut staging[..required], |pd, ri, rj, rc, ai, _aj, li, lj| {
                grids_contract_ipip(pd, ri, rj, rc, ai, li, lj)
            })?;
            Ok(grids_stats(plan, required))
        }
        other => {
            Err(cintxRsError::UnsupportedApi {
                requested: format!("grids operator '{}' is not supported", other),
            })
        }
    }
}

/// Stub for origi family (int1e_r2_origi, int1e_r4_origi, ip2 derivatives).
/// Implementation pending.

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
    //
    // libcint int2e_breit_r1p2_spinor and int2e_breit_r2p2_spinor use c2s_sf_2e1i + c2s_sf_2e2i
    // (iket variants), which apply a phase of i to both the j-ket (step 1) and l-ket (step 2).
    // The combined phase is i_j * i_l = i^2 = -1 relative to the regular c2s_sf_2e1 + c2s_sf_2e2
    // transform used by ordinary 2e integrals. We apply cart_to_spinor_sf_4d (regular) and then
    // negate, matching the iket phase convention.
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
    // Negate to account for c2s_sf_2e1i + c2s_sf_2e2i phase convention.
    for v in staging.iter_mut() {
        *v = -*v;
    }

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