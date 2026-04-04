//! Host-side 3c1e integral kernel: three-center one-electron overlap.
//!
//! Implements the G-tensor fill + contraction + c2s pipeline from
//! libcint `g3c1e.c` (`CINTg3c1e_ovlp`) and `cint3c1e.c` (`CINT3c1e_loop_nopt`).
//!
//! # Algorithm
//! For each contracted shell triple (i, j, k):
//! 1. Compute per-primitive prefactor from three-center Gaussian overlap exponent.
//! 2. Fill G-tensor via VRR in combined (j+k) direction, then HRR to split i and k.
//! 3. Contract Cartesian components weighted by ci*cj*ck contraction coefficients.
//! 4. Accumulate over all primitive triples (ip, jp, kp).
//! 5. Apply `common_fac_sp` scaling for s/p shells.
//! 6. Apply cart-to-sph transform if representation is Spheric.
//!
//! # G-tensor layout (from CINTinit_int3c1e_EnvVars)
//! ```text
//! dli = li + 1,  dlj = lj + lk + 1,  dlk = lk + 1
//! g_stride_i = 1
//! g_stride_j = dli
//! g_stride_k = dli * dlj
//! g_size = dli * dlj * dlk
//! ```
//!
//! # Key difference from 1e
//! Three centers, triple primitive loops, combined j+k VRR dimension followed
//! by i-HRR and k-separation HRR. No Rys quadrature — this is a three-center
//! Gaussian product overlap.

use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c1e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Matches `CINTcommon_fac_sp` from libcint `cart2sph.c`:
///   - l=0: 0.282094791773878 = 1/(2*sqrt(pi)) = Y_0^0
///   - l=1: 0.488602511902920 = sqrt(3/(4*pi))
///   - l>=2: 1.0
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Follows libcint `CINTcart_comp` ordering:
/// for lx = l..=0, for ly = l-lx..=0, nz = l - lx - ly.
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

/// Fill the G-tensor for a 3c1e overlap primitive triple.
///
/// Implements `CINTg3c1e_ovlp` from `libcint-master/src/g3c1e.c`.
///
/// The G-tensor has layout (per axis):
/// ```text
///   g_size = dli * dlj * dlk
///   dli = li + 1,  dlj = lj + lk + 1,  dlk = lk + 1
///   g_stride_i = 1
///   g_stride_j = dli       (in final layout after HRR)
///   g_stride_k = dli * dlj (in final layout after HRR)
/// ```
///
/// Returned as flat `3 * g_size` array: `[gx | gy | gz]`.
///
/// Parameters:
/// - `fac`: per-primitive prefactor (envs->fac[0] = dijk in libcint)
/// - `ai`, `aj`, `ak`: primitive exponents
/// - `ri`, `rj`, `rk`: center coordinates
/// - `rirj`: `ri - rj` (pre-computed by caller)
/// - `li`, `lj`, `lk`: angular momenta (ceiling values = actual l for overlap)
fn fill_g_tensor_3c1e(
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
) -> Vec<f64> {
    let dli = (li + 1) as usize;
    let dlj = (lj + lk + 1) as usize; // combined j+k dimension
    let dlk = (lk + 1) as usize;
    let g_size = dli * dlj * dlk;

    // Combined angular momentum counters
    let nmax = (li + lj + lk) as usize; // total VRR length in combined dimension
    let mmax = (lj + lk) as usize; // used for k-separation HRR bound

    // We need g_size big enough to hold the combined VRR buffer AND the final split layout.
    // From libcint: g_size = max(dli*dlj*dlk, dli*nmax) where nmax = li + dlj.
    // The first phase of VRR builds columns of length nmax+1 in a dli*(nmax+1) buffer.
    let vrr_nmax = li as usize + dlj; // = li + lj + lk + 1
    // CORRECT: max of the two products, not dli*dlj*(max of dlk vs vrr_nmax).
    let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);

    let mut g = vec![0.0_f64; 3 * g_alloc];

    let aijk = ai + aj + ak;
    let aijk1 = 0.5_f64 / aijk; // 0.5 / (ai+aj+ak)

    // G-tensor base case: gx[0]=1, gy[0]=1, gz[0]=fac
    // (the per-primitive prefactor includes the normalization)
    let gx_off = 0usize;
    let gy_off = g_alloc;
    let gz_off = 2 * g_alloc;

    g[gx_off] = 1.0;
    g[gy_off] = 1.0;
    g[gz_off] = fac;

    if nmax == 0 {
        // All s-type shells: nothing more to fill
        return g;
    }

    // In the first VRR phase, g is treated as a 1D array with stride dj = dli (local stride).
    // The local stride for the combined j+k direction is dj_local = li+1.
    // After VRR builds combined dimension 0..nmax, HRR shifts i and k.
    let dj_local = dli; // = li + 1

    // Compute rjrijk = Rj - G where G = (ai*Ri + aj*Rj + ak*Rk) / aijk
    // (displacement from J center to the three-center weighted center)
    // From g3c1e.c line 157-159:
    //   rjrijk[d] = rj[d] - (ai*ri[d] + aj*rj[d] + ak*rk[d]) / aijk
    let rjrijk = [
        rj[0] - (ai * ri[0] + aj * rj[0] + ak * rk[0]) / aijk,
        rj[1] - (ai * ri[1] + aj * rj[1] + ak * rk[1]) / aijk,
        rj[2] - (ai * ri[2] + aj * rj[2] + ak * rk[2]) / aijk,
    ];

    // VRR: fill combined j+k direction (0 to nmax levels) using rjrijk displacement.
    // From g3c1e.c lines 161-168:
    //   g[dj] = -rjrijk * g[0]
    //   g[(j+1)*dj] = aijk1 * j * g[(j-1)*dj] - rjrijk * g[j*dj],  j = 1..nmax-1
    //
    // Note: the sign is NEGATIVE (g[dj] = -rjrijk*g[0]). This differs from the 1e VRR
    // which uses +rijrx. The 3c1e VRR uses the displacement to the three-center G point.
    for d in 0..3 {
        let off = d * g_alloc;
        let disp = -rjrijk[d]; // negative sign from the formula
        // j=1 level:
        g[off + dj_local] = disp * g[off];
        // j=2..nmax:
        let mut j = 1usize;
        while j < nmax {
            g[off + (j + 1) * dj_local] =
                aijk1 * j as f64 * g[off + (j - 1) * dj_local] + disp * g[off + j * dj_local];
            j += 1;
        }
    }

    // HRR for i-direction: shift angular momentum from combined-j to i.
    // From g3c1e.c lines 171-177:
    //   for i = 1..li:
    //     for j = 0..nmax-i:
    //       g[i + j*dj] = g[i-1 + (j+1)*dj] - rirj[d] * g[i-1 + j*dj]
    // rirj in libcint is Ri - Rj (set in CINTinit_int3c1e_EnvVars line 74-76).
    // Our `rirj` parameter is ri - rj so we use `rirj[d]` directly.
    for d in 0..3 {
        let off = d * g_alloc;
        let rirj_d = rirj[d]; // = ri[d] - rj[d]
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

    // Switch to the final layout stride: dj = dli (g_stride_j) and dk = dli * dlj (g_stride_k).
    // HRR for k-separation: separate the combined j+k dimension into j and k.
    // From g3c1e.c lines 179-188:
    //   dj = g_stride_j = dli
    //   for k = 1..lk:
    //     for j = 0..mmax-k:
    //       off = k*dk + j*dj
    //       for i = off..off+li:
    //         g[i] = g[i + dj - dk] + rjrk * g[i - dk]
    //
    // rjrk = Rj - Rk (libcint: rjrk computed in loop header)
    let dk = dli * dlj; // = g_stride_k
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];

    for d in 0..3 {
        let off = d * g_alloc;
        let rjrk_d = rjrk[d];
        for k in 1..=(lk as usize) {
            for j in 0..=(mmax - k) {
                let base = k * dk + j * dj_local;
                // i = base..base+li (flat slice across i-range)
                for i in 0..=li as usize {
                    let idx = base + i;
                    // g[idx] = g[idx + dj - dk] + rjrk * g[idx - dk]
                    // Note: dj_local = dli = g_stride_j in final layout
                    let idx_hi = idx + dj_local - dk;
                    let idx_lo = idx - dk;
                    g[off + idx] = g[off + idx_hi] + rjrk_d * g[off + idx_lo];
                }
            }
        }
    }

    g
}

/// Contract G-tensor for 3c1e overlap operator.
///
/// Loops over all (i, j, k) Cartesian products and produces the flat Cartesian
/// integral buffer of size `ncart(li) * ncart(lj) * ncart(lk)`.
///
/// Output layout: i is fastest (innermost), k is slowest (outermost).
/// This matches libcint's `CINTg3c1e_index_xyz` column-major ordering.
fn contract_3c1e_ovlp(g: &[f64], li: u8, lj: u8, lk: u8, g_size: usize) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let dli = (li as usize) + 1;
    let dlj = (lj as usize) + (lk as usize) + 1;

    let dj = dli; // g_stride_j = dli
    let dk = dli * dlj; // g_stride_k

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);

    let gx = 0usize;
    let gy = g_size;
    let gz = 2 * g_size;

    // Output layout: i fastest, j middle, k slowest
    // out[(k_idx * ncj + j_idx) * nci + i_idx]
    let mut out = vec![0.0_f64; nci * ncj * nck];

    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                // G-tensor index: g[ix + jx*dj + kx*dk], etc.
                let vx = g[gx + ix as usize + jx as usize * dj + kx as usize * dk];
                let vy = g[gy + iy as usize + jy as usize * dj + ky as usize * dk];
                let vz = g[gz + iz as usize + jz as usize * dj + kz as usize * dk];
                out[(k_idx * ncj + j_idx) * nci + i_idx] += vx * vy * vz;
            }
        }
    }

    out
}

/// Launch the 3c1e kernel for a contracted shell triple.
///
/// Implements the three-center one-electron overlap integral per
/// `cint3c1e.c` `CINT3c1e_loop_nopt` with `CINTg3c1e_ovlp` G-fill.
///
/// Replaces the zero-returning stub from Phase 9.
pub fn launch_center_3c1e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "3c1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c1e",
            detail: format!(
                "canonical_family mismatch for 3c1e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    // Host-side execution: backend not used for GPU dispatch
    let _ = backend;

    // Require exactly 3 shells
    let shells = plan.shells.as_slice();
    if shells.len() < 3 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c1e",
            detail: format!(
                "3c1e kernel requires exactly 3 shells, got {}",
                shells.len()
            ),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;

    // Atom coordinates
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    // Pre-compute pairwise displacements and squared distances
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let rirk = [ri[0] - rk[0], ri[1] - rk[1], ri[2] - rk[2]];
    let rjrk = [rj[0] - rk[0], rj[1] - rk[1], rj[2] - rk[2]];
    let rr_ij = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
    let rr_ik = rirk[0] * rirk[0] + rirk[1] * rirk[1] + rirk[2] * rirk[2];
    let rr_jk = rjrk[0] * rjrk[0] + rjrk[1] * rjrk[1] + rjrk[2] * rjrk[2];

    // common_factor = sqrt(pi) * pi * fac_sp(li) * fac_sp(lj) * fac_sp(lk)
    // From CINTinit_int3c1e_EnvVars line 55-57.
    let common_factor = SQRTPI
        * std::f64::consts::PI
        * common_fac_sp(li)
        * common_fac_sp(lj)
        * common_fac_sp(lk);

    // Output size in Cartesian
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);

    // Accumulated Cartesian integral buffer (i fastest, k slowest)
    let mut cart_buf = vec![0.0_f64; nci * ncj * nck];

    // G-tensor size parameters
    let dli = (li as usize) + 1;
    let dlj = (lj as usize) + (lk as usize) + 1;
    let dlk = (lk as usize) + 1;
    let vrr_nmax = dli + (lj as usize) + (lk as usize); // = li + lj + lk + 1
    let g_size = (dli * dlj * dlk).max(dli * vrr_nmax);

    // Primitive loops: kp, jp, ip (matching CINT3c1e_loop_nopt order)
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_i = shell_i.nprim as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_i = shell_i.nctr as usize;

    // expcutoff: from libcint default EXPCUTOFF = 60 (used to skip negligible primitives)
    let expcutoff = 60.0_f64;

    for kp in 0..n_prim_k {
        let ak = shell_k.exponents[kp];
        let ajakrr_base = ak * rr_jk; // ak * aj * rr_jk factored later

        for jp in 0..n_prim_j {
            let aj = shell_j.exponents[jp];
            let ajakrr = aj * ajakrr_base; // aj * ak * rr_jk

            for ip in 0..n_prim_i {
                let ai = shell_i.exponents[ip];
                let aijk = ai + aj + ak;
                let aiajrr = ai * aj * rr_ij;
                let aiakrr = ai * ak * rr_ik;

                // Exponential screening: eijk = (ai*aj*rr_ij + ai*ak*rr_ik + aj*ak*rr_jk) / aijk
                let eijk = (aiajrr + aiakrr + ajakrr) / aijk;
                if eijk > expcutoff {
                    continue;
                }

                // Per-primitive prefactor (before contraction coefficients):
                // dijk = common_factor * exp(-eijk) / (aijk * sqrt(aijk))
                // The contraction coefficients are multiplied outside.
                // Note: in CINT3c1e_loop_nopt, common_factor is factored into fac1k, fac1j, fac1i
                // along with contraction coefficients. Here we accumulate over contractions
                // separately, so the primitive G-tensor uses dijk (without coefficients).
                let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());
                let fac = common_factor * dijk; // envs->fac[0] equivalent (without ci*cj*ck)

                // Fill G-tensor for this primitive triple
                let g = fill_g_tensor_3c1e(
                    fac, ai, aj, ak, ri, rj, rk, rirj, li as u32, lj as u32, lk as u32,
                );

                // Contract Cartesian components from G-tensor
                let prim_buf = contract_3c1e_ovlp(&g, li, lj, lk, g_size);

                // Accumulate with contraction coefficients
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

    // Apply cart-to-sph transform or copy Cartesian to staging
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_3c1e(&cart_buf, li, lj, lk);
            let sph_size = nsi * nsj * nsk;
            let copy_len = staging.len().min(sph.len()).min(sph_size);
            staging[..copy_len].copy_from_slice(&sph[..copy_len]);
        }
        _ => {
            // Cartesian: copy directly
            let copy_len = staging.len().min(cart_buf.len());
            staging[..copy_len].copy_from_slice(&cart_buf[..copy_len]);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // Test: s-s-s overlap at same center should be nonzero
    #[test]
    fn test_fill_g_tensor_3c1e_sss() {
        // Simple s-s-s case: ai=aj=ak=1, all at origin
        let fac = 1.0_f64;
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let ak = 1.0_f64;
        let ri = [0.0_f64; 3];
        let rj = [0.0_f64; 3];
        let rk = [0.0_f64; 3];
        let rirj = [0.0_f64; 3];
        let li = 0u32;
        let lj = 0u32;
        let lk = 0u32;

        let g = fill_g_tensor_3c1e(fac, ai, aj, ak, ri, rj, rk, rirj, li, lj, lk);

        // For s-s-s: gx[0]=1, gy[0]=1, gz[0]=fac=1
        assert!((g[0] - 1.0).abs() < 1e-15, "gx[0] should be 1.0");
        assert!((g[1] - 1.0).abs() < 1e-15, "gy[0] should be 1.0");
        assert!((g[2] - 1.0).abs() < 1e-15, "gz[0] should be 1.0");
    }

    // Test: contract_3c1e_ovlp for s-s-s returns scalar
    #[test]
    fn test_contract_3c1e_ovlp_sss() {
        // For s-s-s: li=lj=lk=0, g has single elements gx[0]=1,gy[0]=1,gz[0]=fac
        let fac = 2.5_f64;
        let g = vec![1.0_f64, 1.0, fac]; // [gx|gy|gz] each of size 1
        let out = contract_3c1e_ovlp(&g, 0, 0, 0, 1);
        assert_eq!(out.len(), 1);
        assert!((out[0] - fac).abs() < 1e-14, "s-s-s overlap should equal gz[0] = {}", fac);
    }
}
