//! Cross-family shared helpers for the unstable-source-api kernel families.
//!
//! These items were factored out of the original single-file `unstable.rs`
//! during the per-family module split (move-only: bodies are unchanged;
//! only visibility was widened to `pub(crate)` so the family submodules
//! can reference them).

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
pub(crate) fn apply_nabla_i_3axis(
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
    nabla_j_host(&mut df[g_per_axis..2*g_per_axis], &g[g_per_axis..2*g_per_axis], aj, li, lj, nmax);
    nabla_j_host(&mut df[2*g_per_axis..3*g_per_axis], &g[2*g_per_axis..3*g_per_axis], aj, li, lj, nmax);
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
