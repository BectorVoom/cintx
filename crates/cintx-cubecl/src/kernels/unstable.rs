//! Kernel launch functions for unstable-source API families.
//!
//! Phase 14 Wave 2 — grids family implementation.
//! Other families (origi, breit, origk, ssc) remain stubbed.
//!
//! Families covered:
//!   - grids: grid-point integrals with NGRIDS env parameter (cint1e_grids.c)
//!   - origi: origin-displaced r^n one-electron integrals (cint1e_a.c) [stub]
//!   - breit: Breit spinor-only two-electron integrals (breit.c) [stub]
//!   - origk: origin-k-displaced three-center one-electron integrals (cint3c1e_a.c) [stub]
//!   - ssc: spin-spin contact three-center two-electron integral (cint3c2e.c) [stub]

use crate::backend::ResolvedBackend;
use crate::math::obara_saika::{hrr_step_host, vrr_2e_step_host};
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::{rys_root1_host, rys_root2_host};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats, planner::GridsEnvParams};

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
pub fn launch_origi(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "origi: stub — implementation pending".to_owned(),
    })
}

/// Stub for breit family (int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor).
/// Implementation pending.
pub fn launch_breit(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "breit: stub — implementation pending".to_owned(),
    })
}

/// Stub for origk family (int3c1e_r2/r4/r6_origk and ip1 derivatives).
/// Implementation pending.
pub fn launch_origk(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "origk: stub — implementation pending".to_owned(),
    })
}

/// Stub for ssc family (int3c2e_sph_ssc).
/// Implementation pending.
pub fn launch_ssc(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "ssc: stub — implementation pending".to_owned(),
    })
}
