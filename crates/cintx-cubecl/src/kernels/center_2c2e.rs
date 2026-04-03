//! Host-side 2c2e (two-center two-electron Coulomb) integral kernel.
//!
//! Implements the G-tensor fill + contraction + cart-to-sph pipeline following
//! libcint `g2c2e.c` / `g2e.c` `CINTg0_2e` + `CINTg0_2e_2d`.
//!
//! # Algorithm
//! For each contracted shell pair (i, k):
//! 1. Compute Rys argument x = rho * |ri - rk|^2 where rho = ai*ak/(ai+ak).
//! 2. Fetch nrys_roots Rys quadrature roots u[] and weights w[] via rys_roots_host.
//! 3. For each Rys root: compute recurrence coefficients (b00, b10, b01, c00, c0p)
//!    and fill the G-tensor via VRR in both i and k directions.
//! 4. Contract G-tensor elements over Cartesian component triples (ix, iy, iz) x (kx, ky, kz).
//! 5. Accumulate contracted values weighted by primitive contraction coefficients.
//! 6. Apply common_fac_sp(li) * common_fac_sp(lk) scaling.
//! 7. Apply cart_to_sph_2c2e if Spheric representation is requested.
//!
//! Source: libcint-master/src/g2c2e.c (CINTinit_int2c2e_EnvVars) and
//!         libcint-master/src/g2e.c (CINTg0_2e, CINTg0_2e_2d).

use crate::backend::ResolvedBackend;
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2c2e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Mirrors libcint `CINTcommon_fac_sp(l)` from `g2e.c` / `g1e.c`:
///   l=0 (s): 1/(2*sqrt(pi)) = 0.282094791773878143
///   l=1 (p): sqrt(3/(4*pi)) = 0.488602511902919921
///   l>=2:    1.0 (embedded in c2s coefficient tables)
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

/// Fill the 2c2e G-tensor for one primitive pair (ai, ak) using Rys quadrature.
///
/// Returns flat `[gx | gy | gz]` each of size `g_size = nrys * (li+1) * (lk+1)`.
/// Layout: `g[axis*g_size + k_level*nrys*(li+1) + i_level*nrys + root]`
/// where `dn = nrys` (i-stride) and `dm = nrys*(li+1)` (k-stride).
///
/// Source: libcint-master/src/g2e.c `CINTg0_2e` + `CINTg0_2e_2d` (lines 4425-4566, 272-424).
fn fill_g_tensor_2c2e(
    ai: f64,
    ak: f64,
    ri: &[f64; 3],
    rk: &[f64; 3],
    li: u8,
    lk: u8,
    fac_pair: f64,
) -> Vec<f64> {
    let nmax = li as usize;   // VRR max in i direction
    let mmax = lk as usize;   // VRR max in k direction
    let nrys_roots = (li as usize + lk as usize) / 2 + 1;

    // G-tensor strides (matching libcint g2c2e.c CINTinit_int2c2e_EnvVars):
    //   g_stride_i = nrys_roots
    //   g_stride_k = nrys_roots * (li+1)
    let dn = nrys_roots;                      // i-level stride
    let dm = nrys_roots * (li as usize + 1); // k-level stride
    let g_size = nrys_roots * (li as usize + 1) * (lk as usize + 1);

    let mut g = vec![0.0_f64; 3 * g_size];

    // Displacement and Rys argument
    let xij_kl = ri[0] - rk[0];
    let yij_kl = ri[1] - rk[1];
    let zij_kl = ri[2] - rk[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

    // For 2c2e: aij = ai, akl = ak (j_l = l_l = 0)
    let aij = ai;
    let akl = ak;
    let a1 = aij * akl;
    let a0 = a1 / (aij + akl); // rho = ai*ak/(ai+ak)

    // Common factor: sqrt(a0 / (a1^3)) = sqrt(rho / (ai*ak)^3) = 1/sqrt(ai*ak*(ai+ak))
    // Source: g2e.c CINTg0_2e line 4441: fac1 = sqrt(a0/(a1*a1*a1)) * envs->fac[0]
    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_pair;

    // Rys argument x = a0 * rr
    let x_rys = a0 * rr;

    // Get Rys roots and weights
    let (u_roots, w_weights) = rys_roots_host(nrys_roots, x_rys);

    // For 2c2e: rx_in_rijrx = ri (rij = ri), rx_in_rklrx = rk (rkl = rk)
    // So: rijrx = rij - rx_in_rijrx = ri - ri = (0,0,0)
    //     rklrx = rkl - rx_in_rklrx = rk - rk = (0,0,0)
    let rijrx = [0.0_f64; 3];
    let rklrx = [0.0_f64; 3];

    // Per-root recurrence coefficients and G-tensor fill
    // Source: g2e.c CINTg0_2e lines 4542-4563, then CINTg0_2e_2d lines 285-421
    for irys in 0..nrys_roots {
        // u2 = a0 * u[irys]   (a0 = rho = ai*ak/(ai+ak))
        let u2 = a0 * u_roots[irys];

        // tmp4 = 0.5 / (u2*(aij+akl) + a1)
        // tmp5 = u2 * tmp4
        let tmp4 = 0.5 / (u2 * (aij + akl) + a1);
        let tmp5 = u2 * tmp4;

        // b00, b10, b01
        let b00 = tmp5;
        let b10 = tmp5 + tmp4 * akl;
        let b01 = tmp5 + tmp4 * aij;

        // c00 displacement: rijrx - tmp2 * (rij - rkl)
        // where tmp2 = tmp1 * akl = 2*tmp5*akl
        let tmp2 = 2.0 * tmp5 * akl;
        let tmp3 = 2.0 * tmp5 * aij;
        let c00 = [
            rijrx[0] - tmp2 * xij_kl,
            rijrx[1] - tmp2 * yij_kl,
            rijrx[2] - tmp2 * zij_kl,
        ];
        // c0p displacement: rklrx + tmp3 * (rij - rkl)
        let c0p = [
            rklrx[0] + tmp3 * xij_kl,
            rklrx[1] + tmp3 * yij_kl,
            rklrx[2] + tmp3 * zij_kl,
        ];

        // Base case: gx[irys] = 1, gy[irys] = 1, gz[irys] = w[irys]*fac1
        // Source: g2e.c CINTg0_2e lines 4517-4521, CINTg0_2e_2d lines 285-289
        g[irys] = 1.0;           // gx[root=irys, i=0, k=0]
        g[g_size + irys] = 1.0; // gy[root=irys, i=0, k=0]
        g[2 * g_size + irys] = w_weights[irys] * fac1; // gz carries weight

        // VRR in i direction (nmax levels): CINTg0_2e_2d lines 305-331
        // gx[irys + (n+1)*dn] = c00x * gx[irys + n*dn] + n * b10 * gx[irys + (n-1)*dn]
        if nmax > 0 {
            for axis in 0..3 {
                let base = axis * g_size;
                let c = c00[axis];
                // n=0: g[n+1] = c00 * g[0]
                let s0 = g[base + irys];
                let mut s1 = c * s0;
                g[base + irys + dn] = s1;
                // n=1..nmax-1: g[n+1] = c00*g[n] + n*b10*g[n-1]
                let mut s_prev = s0;
                for n in 1..nmax {
                    let s2 = c * s1 + n as f64 * b10 * s_prev;
                    g[base + irys + (n + 1) * dn] = s2;
                    s_prev = s1;
                    s1 = s2;
                }
            }
        }

        // VRR in k direction (mmax levels): CINTg0_2e_2d lines 334-390
        // gx[irys + (m+1)*dm] = c0p * gx[irys + m*dm] + m*b01 * gx[irys + (m-1)*dm]
        if mmax > 0 {
            for axis in 0..3 {
                let base = axis * g_size;
                let c = c0p[axis];
                // m=0: g[k=1, i=0] = c0p * g[k=0, i=0]
                let s0 = g[base + irys];
                let mut s1 = c * s0;
                g[base + irys + dm] = s1;
                // m=1..mmax-1
                let mut s_prev = s0;
                for m in 1..mmax {
                    let s2 = c * s1 + m as f64 * b01 * s_prev;
                    g[base + irys + (m + 1) * dm] = s2;
                    s_prev = s1;
                    s1 = s2;
                }
            }

            // Mixed i+k recurrence (CINTg0_2e_2d lines 362-391):
            // gx[irys + (n)*dn + (m+1)*dm] = c0p * gx[irys + n*dn + m*dm]
            //   + m*b01 * gx[irys + n*dn + (m-1)*dm] + b00 * gx[irys + n*dn + m*dm - dn]
            // NOTE: only needed when nmax >= 1 (has i levels)
            if nmax > 0 {
                for axis in 0..3 {
                    let base = axis * g_size;
                    let c = c0p[axis];
                    // For each i level n=0..nmax, fill k levels m=0..mmax
                    for n in 0..=nmax {
                        let i_off = irys + n * dn;
                        // m=0 (base row for this i-level): already filled from i-VRR
                        // We need to fill m=1..mmax for n>0
                        // m=0: g[i=n, k=1] = c0p * g[i=n, k=0] + b00 * g[i=n-1, k=0] ... wait
                        // Actually from g2e.c lines 367-391, for n>=1 it's:
                        // g[n*dn + (m+1)*dm] = c0p*g[n*dn+m*dm] + m*b01*g[n*dn+(m-1)*dm] + b00*g[(n-1)*dn+m*dm]
                        // For n=0 it's already filled (pure k-VRR above)
                        if n == 0 {
                            continue;
                        }
                        // Build k levels for this i-level
                        // m=0: already in g[i_off] = g[irys + n*dn] (from i-VRR)
                        // m=1: c0p*g[n,0] + b00*g[n-1,0]
                        let s0_k0 = g[base + i_off]; // g[n, k=0]
                        let prev_i_k0 = g[base + irys + (n - 1) * dn]; // g[n-1, k=0]
                        let mut s1 = c * s0_k0 + b00 * prev_i_k0;
                        g[base + i_off + dm] = s1;
                        // m=1..mmax-1: c0p*g[n,m] + m*b01*g[n,m-1] + b00*g[n-1,m]
                        let mut s_prev = s0_k0;
                        for m in 1..mmax {
                            let prev_i_km = g[base + irys + (n - 1) * dn + m * dm]; // g[n-1, m]
                            let s2 = c * s1 + m as f64 * b01 * s_prev + b00 * prev_i_km;
                            g[base + i_off + (m + 1) * dm] = s2;
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

/// Real 2c2e integral kernel following the g2c2e.c + g2e.c algorithm.
///
/// Replaces the zero-returning stub. Implements the full G-tensor fill
/// + primitive contraction + cart-to-sph pipeline for the two-center
/// two-electron Coulomb integral (`int2c2e_sph`).
pub fn launch_center_2c2e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "2c2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_2c2e",
            detail: format!(
                "canonical_family mismatch for 2c2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    // Host-side pipeline — no GPU dispatch needed.
    let _ = backend;

    let shells = plan.shells.as_slice();
    if shells.len() < 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_2c2e",
            detail: "2c2e kernel requires exactly 2 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_k = &shells[1];

    let li = shell_i.ang_momentum;
    let lk = shell_k.ang_momentum;

    // Atom coordinates
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    // Output sizes
    let nci = ncart(li);
    let nck = ncart(lk);
    let nsi = nsph(li);
    let nsk = nsph(lk);

    // Accumulated Cartesian integral buffer: layout is [nck * nci]
    // i is fastest (innermost), k is slowest (outermost) — libcint column-major convention
    let mut cart_buf = vec![0.0_f64; nci * nck];

    // Primitive loop over (pi, pk) pairs
    let n_prim_i = shell_i.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // common_factor from g2c2e.c CINTinit_int2c2e_EnvVars:
    //   common_factor = PI^3 * 2 / sqrt(PI) * fac_sp_i * fac_sp_k
    // Note: fac_sp scaling is applied after accumulation (below), not per-primitive.
    // The PI^3 * 2 / sqrt(PI) factor is incorporated into the per-primitive gz[0] via fac1
    // and fac_pair below.
    //
    // fac_pair = exp(-ai*ak/(ai+ak)*|ri-rk|^2) [the Gaussian overlap pre-exponential]
    //   (norm_i = norm_k = 1.0 since coefficients carry the normalization)
    //
    // gz[irys] = w[irys] * fac1 where:
    //   fac1 = sqrt(a0/(a1^3)) * fac_pair  with a0 = rho, a1 = ai*ak
    //
    // But libcint also has common_factor = PI^3*2/sqrt(PI) in envs->common_factor, applied
    // separately in the gout accumulation step (cint2c2e.c CINTg2c2e0_2c2e).
    // Looking at cint2e.c / g2e.c more carefully: the gz weight carries w[irys]*fac1
    // where fac1 = sqrt(a0/(a1^3)) = 1/sqrt(ai*ak*(ai+ak)).
    // Then the full scale PI^(3/2)*2/sqrt(PI) = 2*PI comes from the 3D integration:
    // integral of e^{-a r^2} dr = (pi/a)^{3/2}, giving factor PI^{3/2} per electron pair,
    // total PI^3/((aij*akl)^{3/2}), divided by sqrt(aij+akl). The Rys weights w absorb
    // part of this. The exact scale from g2e.c is that gz[0] = w[irys]*fac1 already encodes
    // the full normalization: fac1 = sqrt(pi/(aij+akl)) * PI / (aij*akl)^{1/2} ...
    // Let's follow exactly: fac1 = sqrt(a0/(a1^3)) * envs->fac where fac = exp(-rho*rr)*ci*ck.
    // This is the full per-primitive contribution scale; the Rys weights w[irys] are
    // dimensionless quadrature weights (not pre-multiplied by the common PI factor).
    //
    // The common_factor (PI^3*2/sqrt(PI)) from g2c2e.c must be applied separately.
    // It is NOT in fac1 — it's applied in the output gout step of cint2c2e.c.
    // So: cart_buf[elem] += ci*ck * common_factor * sum_roots(gx*gy*gz)
    // where gz[root] = w[root] * fac1 = w[root] * sqrt(a0/(a1^3)) * exp(-rho*rr).

    // PI^3 * 2 / sqrt(PI) = 2 * PI^{5/2} = 2 * PI^2 * sqrt(PI)
    // From g2c2e.c line 44: (M_PI*M_PI*M_PI)*2/SQRTPI
    let common_factor_base = (PI * PI * PI) * 2.0 / SQRTPI;

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];

        for pk in 0..n_prim_k {
            let ak = shell_k.exponents[pk];

            // Gaussian overlap prefactor: exp(-ai*ak/(ai+ak) * |ri-rk|^2)
            // (norm = 1.0 since contraction coefficients carry normalization)
            let aij = ai;
            let akl = ak;
            let rho = aij * akl / (aij + akl);
            let rr = {
                let dx = ri[0] - rk[0];
                let dy = ri[1] - rk[1];
                let dz = ri[2] - rk[2];
                dx * dx + dy * dy + dz * dz
            };
            let fac_pair = (-rho * rr).exp();

            // Fill G-tensor via Rys quadrature
            let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, li, lk, fac_pair);

            // Determine G-tensor layout parameters
            let nrys_roots = (li as usize + lk as usize) / 2 + 1;
            let dn = nrys_roots;
            let dm = nrys_roots * (li as usize + 1);
            let g_size = nrys_roots * (li as usize + 1) * (lk as usize + 1);

            let ci_comps = cart_comps(li);
            let ck_comps = cart_comps(lk);

            // Contract: sum over Rys roots and Cartesian components
            // Output layout: i fastest, k slowest: out[ci_idx + ck_idx * nci]
            // (matches libcint column-major: i fastest)
            let mut prim_buf = vec![0.0_f64; nci * nck];

            for (ck_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
                for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let mut val = 0.0_f64;
                    for irys in 0..nrys_roots {
                        // G-tensor index: g[axis*g_size + k_level*dm + i_level*dn + root]
                        let vx = g[0 * g_size + kx as usize * dm + ix as usize * dn + irys];
                        let vy = g[1 * g_size + ky as usize * dm + iy as usize * dn + irys];
                        let vz = g[2 * g_size + kz as usize * dm + iz as usize * dn + irys];
                        val += vx * vy * vz;
                    }
                    // i fastest: prim_buf[ci_idx + ck_idx*nci]
                    prim_buf[ci_idx + ck_idx * nci] = val;
                }
            }

            // Accumulate over contractions with common_factor_base
            for ci in 0..n_ctr_i {
                let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                for ck in 0..n_ctr_k {
                    let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                    let weight = coeff_i * coeff_k * common_factor_base;
                    for elem in 0..prim_buf.len() {
                        cart_buf[elem] += weight * prim_buf[elem];
                    }
                }
            }
        }
    }

    // Apply CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lk) scaling.
    // Source: g2c2e.c CINTinit_int2c2e_EnvVars line 44-45.
    let sp_scale = common_fac_sp(li) * common_fac_sp(lk);
    if (sp_scale - 1.0).abs() > 1e-15 {
        for v in cart_buf.iter_mut() {
            *v *= sp_scale;
        }
    }

    // Apply cart-to-sph transform or copy Cartesian to staging
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_2c2e(&cart_buf, li, lk);
            let copy_len = staging.len().min(sph.len());
            staging[..copy_len].copy_from_slice(&sph[..copy_len]);
        }
        _ => {
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;

    /// Smoke test: s-s pair should produce a positive non-zero value.
    #[test]
    fn test_fill_g_tensor_2c2e_ss_nonzero() {
        // Two s-shells at different centers
        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 2.0]; // 2 bohr apart
        let ai = 1.0_f64;
        let ak = 1.0_f64;
        let fac_pair = (-0.5 * 4.0_f64).exp(); // rho * r^2 = 0.5 * 4 = 2

        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, 0, 0, fac_pair);

        // For s-s: g_size = 1, g = [gx, gy, gz] each size 1
        // gz[0] = w[0] * fac1 where fac1 = sqrt(a0/(a1^3)) * fac_pair
        // This should be positive and non-zero
        assert!(g.len() == 3, "s-s G-tensor should have 3 elements (gx,gy,gz)");
        let gz = g[2]; // gz[0]
        assert!(gz > 0.0, "s-s gz[0] should be positive: got {gz:.6e}");
    }

    /// Smoke test: p-p pair should produce a non-trivial G-tensor.
    #[test]
    fn test_fill_g_tensor_2c2e_pp_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 2.0];
        let ai = 0.5_f64;
        let ak = 0.5_f64;
        let rho = ai * ak / (ai + ak);
        let rr = 4.0_f64;
        let fac_pair = (-rho * rr).exp();

        // li=1, lk=1 => nrys = (1+1)/2+1 = 2, g_size = 2*2*2 = 8
        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, 1, 1, fac_pair);
        assert_eq!(g.len(), 3 * 8, "p-p G-tensor size mismatch");

        // At least some gz entries should be non-zero
        let gz = &g[2 * 8..3 * 8];
        let nonzero = gz.iter().filter(|&&v| v.abs() > 1e-20).count();
        assert!(nonzero > 0, "p-p G-tensor gz should have non-zero entries");
    }
}
