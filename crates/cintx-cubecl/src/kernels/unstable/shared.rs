//! Cross-family shared helpers for the unstable-source-api kernel families.
//!
//! These items were factored out of the original single-file `unstable.rs`
//! during the per-family module split (move-only: bodies are unchanged;
//! only visibility was widened to `pub(crate)` so the family submodules
//! can reference them).

use crate::transform::c2s::{cart_to_sph_1e_into, cart_to_sph_3c1e_into, ncart, nsph};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

/// sqrt(pi) constant — matches libcint `SQRTPI`.
pub(crate) const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
/// Same as one_electron.rs: CINTcommon_fac_sp(l).
pub(crate) fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
pub(crate) fn cart_comps(l: u8) -> Vec<(u8, u8, u8)> {
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
fn nabla_j_host(df: &mut [f64], g: &[f64], aj: f64, li: u32, lj: u32, nmax: u32) {
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
pub(crate) fn apply_nabla_i_3axis(
    g: &[f64],
    ai: f64,
    li: u32,
    lj: u32,
    nmax: u32,
    g_per_axis: usize,
) -> Vec<f64> {
    let mut df = vec![0.0_f64; 3 * g_per_axis];
    nabla_i_host(
        &mut df[0..g_per_axis],
        &g[0..g_per_axis],
        ai,
        li,
        lj,
        nmax,
        g_per_axis,
    );
    nabla_i_host(
        &mut df[g_per_axis..2 * g_per_axis],
        &g[g_per_axis..2 * g_per_axis],
        ai,
        li,
        lj,
        nmax,
        g_per_axis,
    );
    nabla_i_host(
        &mut df[2 * g_per_axis..3 * g_per_axis],
        &g[2 * g_per_axis..3 * g_per_axis],
        ai,
        li,
        lj,
        nmax,
        g_per_axis,
    );
    df
}

pub(crate) fn apply_nabla_j_3axis(
    g: &[f64],
    aj: f64,
    li: u32,
    lj: u32,
    nmax: u32,
    g_per_axis: usize,
) -> Vec<f64> {
    let mut df = vec![0.0_f64; 3 * g_per_axis];
    nabla_j_host(&mut df[0..g_per_axis], &g[0..g_per_axis], aj, li, lj, nmax);
    nabla_j_host(
        &mut df[g_per_axis..2 * g_per_axis],
        &g[g_per_axis..2 * g_per_axis],
        aj,
        li,
        lj,
        nmax,
    );
    nabla_j_host(
        &mut df[2 * g_per_axis..3 * g_per_axis],
        &g[2 * g_per_axis..3 * g_per_axis],
        aj,
        li,
        lj,
        nmax,
    );
    df
}

pub(crate) fn make_exec_stats(plan: &ExecutionPlan<'_>, staging: &[f64]) -> ExecutionStats {
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
//  General-contraction (`nctr > 1`) output scatter
//
//  libcint emits ONE dense array per component whose per-axis extent is
//  `nblk * nctr`, with the contraction index the MAJOR (outer) one WITHIN each
//  axis: `i_global = ci*nblk_i + i_idx` (`CINT1e_drv` / `CINT3c1e_drv` +
//  `c2s_{cart,sph}_{1e,3c2e1}`; see `counts[0] = (i_l*2+1) * x_ctr[0]`). It is
//  NOT a stack of independent per-contraction blocks, so a launcher cannot just
//  concatenate them — the contraction and angular indices interleave per axis.
//
//  Device kernels in this module hand back the per-`(ci,cj[,ck])` Cartesian
//  blocks back to back, component-slowest. These two helpers apply the cart→sph
//  transform per block and place each block at its interleaved offsets. With
//  `nctr == 1` they reduce to a single block written at the natural offsets.
// ─────────────────────────────────────────────────────────────────────────────

/// Scatter the per-`(ci,cj)` Cartesian blocks of a 1e family into `staging`.
///
/// `cart` is `[comp][cj][ci][j_idx][i_idx]` (i fastest) with `cj` the slower
/// contraction index; `staging` receives
/// `[comp][j_global][i_global]` with `i_global = ci*nblk_i + i_idx`.
pub(crate) fn scatter_1e_ctr_blocks(
    cart: &[f64],
    li: u8,
    lj: u8,
    spheric: bool,
    n_ctr_i: usize,
    n_ctr_j: usize,
    ncomp: usize,
    staging: &mut [f64],
) {
    let (nci, ncj) = (ncart(li), ncart(lj));
    let cart_block = nci * ncj;
    let (nblk_i, nblk_j) = if spheric {
        (nsph(li), nsph(lj))
    } else {
        (nci, ncj)
    };
    let ni_full = n_ctr_i * nblk_i;
    let nj_full = n_ctr_j * nblk_j;
    let comp_stride = ni_full * nj_full;
    let ctr_total = n_ctr_i * n_ctr_j;

    let mut block = vec![0.0_f64; nblk_i * nblk_j];
    let mut scratch = Vec::new();

    for comp in 0..ncomp {
        for cj in 0..n_ctr_j {
            for ci in 0..n_ctr_i {
                let src = (comp * ctr_total + cj * n_ctr_i + ci) * cart_block;
                let Some(src_block) = cart.get(src..src + cart_block) else {
                    continue;
                };
                if spheric {
                    cart_to_sph_1e_into(src_block, &mut block, li, lj, &mut scratch);
                } else {
                    block.copy_from_slice(src_block);
                }
                let comp_base = comp * comp_stride;
                for j_idx in 0..nblk_j {
                    let j_global = cj * nblk_j + j_idx;
                    let row = comp_base + j_global * ni_full + ci * nblk_i;
                    let Some(dst) = staging.get_mut(row..row + nblk_i) else {
                        continue;
                    };
                    dst.copy_from_slice(&block[j_idx * nblk_i..(j_idx + 1) * nblk_i]);
                }
            }
        }
    }
}

/// Scatter the per-`(ci,cj,ck)` Cartesian blocks of a 3c1e family into `staging`.
///
/// `cart` is `[comp][ck][cj][ci][k_idx][j_idx][i_idx]` (i fastest); `staging`
/// receives `[comp][k_global][j_global][i_global]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_3c1e_ctr_blocks(
    cart: &[f64],
    li: u8,
    lj: u8,
    lk: u8,
    spheric: bool,
    n_ctr_i: usize,
    n_ctr_j: usize,
    n_ctr_k: usize,
    ncomp: usize,
    staging: &mut [f64],
) {
    let (nci, ncj, nck) = (ncart(li), ncart(lj), ncart(lk));
    let cart_block = nci * ncj * nck;
    let (nblk_i, nblk_j, nblk_k) = if spheric {
        (nsph(li), nsph(lj), nsph(lk))
    } else {
        (nci, ncj, nck)
    };
    let ni_full = n_ctr_i * nblk_i;
    let nj_full = n_ctr_j * nblk_j;
    let nk_full = n_ctr_k * nblk_k;
    let comp_stride = ni_full * nj_full * nk_full;
    let ctr_total = n_ctr_i * n_ctr_j * n_ctr_k;

    let mut block = vec![0.0_f64; nblk_i * nblk_j * nblk_k];
    let mut scratch = Vec::new();

    for comp in 0..ncomp {
        for ck in 0..n_ctr_k {
            for cj in 0..n_ctr_j {
                for ci in 0..n_ctr_i {
                    let slot = (ck * n_ctr_j + cj) * n_ctr_i + ci;
                    let src = (comp * ctr_total + slot) * cart_block;
                    let Some(src_block) = cart.get(src..src + cart_block) else {
                        continue;
                    };
                    if spheric {
                        cart_to_sph_3c1e_into(src_block, li, lj, lk, &mut block, &mut scratch);
                    } else {
                        block.copy_from_slice(src_block);
                    }
                    let comp_base = comp * comp_stride;
                    for k_idx in 0..nblk_k {
                        let k_global = ck * nblk_k + k_idx;
                        for j_idx in 0..nblk_j {
                            let j_global = cj * nblk_j + j_idx;
                            let row =
                                comp_base + (k_global * nj_full + j_global) * ni_full + ci * nblk_i;
                            let Some(dst) = staging.get_mut(row..row + nblk_i) else {
                                continue;
                            };
                            let src_row = (k_idx * nblk_j + j_idx) * nblk_i;
                            dst.copy_from_slice(&block[src_row..src_row + nblk_i]);
                        }
                    }
                }
            }
        }
    }
}
