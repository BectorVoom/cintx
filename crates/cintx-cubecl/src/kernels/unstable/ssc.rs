//! `ssc` family: spin-spin contact 3c2e integral (`cint3c2e.c`).
//! Split out of the original single-file `unstable.rs`.
//!
//! HOST/DEVICE SPLIT (Phase 21 D-04 convention; quick task 260529-twi Task 6):
//!
//! - ON DEVICE (`ssc_scalar_kernel` `#[cube(launch)]`, generic over `F: Float`):
//!   the FULL ssc scalar numeric pipeline — per-primitive-triple Rys-root
//!   2e-style G-tensor build (`fill_g_tensor_3c2e_ssc`), the ij→(i,j) HRR split
//!   (`split_ij_hrr_ssc`), and the 3c2e Cartesian contraction
//!   (`contract_3c2e_ssc`), weighted + accumulated over every primitive triple
//!   (ip,jp,kp) AND every contraction triple (ci,cj,ck) into a SINGLE i-fastest
//!   Cartesian block `cart_out[(k*ncj+j)*nci+i]` (size nci*ncj*nck). This matches
//!   the host loop exactly: the host also collapses all contractions into one
//!   `cart_buf` of size nci*ncj*nck. Intermediate arithmetic in `F` (run at f64
//!   by the launcher), output written in `F`.
//!
//! - ON HOST (the host part of the honest split, UNCHANGED): the SSC
//!   representation transform — `transpose_ij_3idx` (ibase canonicalization
//!   un-swap) + `cart_to_sph_3c2e_ssc` (spherical on i,j; Cartesian on k; the c2s
//!   coefficient tables are host-only) + the AO scatter into `staging`. The device
//!   kernel produces the Cartesian block in the i-fastest `(k*ncj+j)*nci+i` layout
//!   that the host `transpose_ij_3idx` / `cart_to_sph_3c2e_ssc` consume verbatim.
//!
//! - NO derivative sub-paths to defer: `int3c2e_sph_ssc` is the ONLY ssc symbol.
//!
//! ssc is a 3-center 2e integral (shls:[i32;3]). The G-tensor recurrence is the
//! standard 2e Rys VRR with the bra (i,j) merged into one index (nmax=li+lj) and
//! the ket = real-k shell (mmax=lk). The `split_ij_hrr_ssc` then HRRs the merged
//! bra index back into separate i,j using `rirj`. ALL strides (dn/dm/g_size,
//! nmax/mmax, ni/nj/nk/axis_size) are computed host-side and passed in as runtime
//! u32; `#[comptime] nroots` selects rys_root{1..5}.

use super::shared::{SQRTPI, common_fac_sp, make_exec_stats};
use crate::backend::ResolvedBackend;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
// Host reference helpers (cart_comps / PairData / compute_pdata_host /
// rys_roots_host) are used only by the `#[cfg(test)]` cross-check oracle.
#[cfg(test)]
use super::shared::cart_comps;
#[cfg(test)]
use crate::math::pdata::{PairData, compute_pdata_host};
#[cfg(test)]
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate.
/// ssc `nroots = (li+lj+lk)/2 + 1`, so this covers `l-sum <= 8`.
const MAX_DEVICE_NROOTS: usize = 5;

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
    if nrys_roots > MAX_DEVICE_NROOTS {
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

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // Device-kernel stride metadata (mirrors fill_g_tensor_3c2e_ssc + split_ij_hrr_ssc).
    let nmax = li as usize + lj as usize;
    let mmax = lk as usize;
    let dn = nrys_roots;
    let dm = nrys_roots * (nmax + 1);
    let g_size = nrys_roots * (nmax + 1) * (mmax + 1);
    let ni = li as usize + 1;
    let nj = lj as usize + 1;
    let nk = lk as usize + 1;
    let axis_size = nrys_roots * nk * nj * ni;

    // Flatten the f64 per-shell exps/coeffs the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();

    let out_len = nci * ncj * nck;

    // FULL ssc scalar pipeline on-device (CPU / Wgpu / Cuda / ROCm / Metal).
    let cart_buf: Vec<f64> = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_ssc_scalar_device::<cubecl::cpu::CpuRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, dn as u32, dm as u32,
            g_size as u32, nmax as u32, mmax as u32, ni as u32, nj as u32, nk as u32,
            axis_size as u32, nrys_roots as u32, ri, rj, rk, rirj, common_factor, &exps_i, &exps_j,
            &exps_k, &coeff_i, &coeff_j, &coeff_k, out_len,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_ssc_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, dn as u32, dm as u32,
            g_size as u32, nmax as u32, mmax as u32, ni as u32, nj as u32, nk as u32,
            axis_size as u32, nrys_roots as u32, ri, rj, rk, rirj, common_factor, &exps_i, &exps_j,
            &exps_k, &coeff_i, &coeff_j, &coeff_k, out_len,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_ssc_scalar_device::<cubecl_cuda::CudaRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, dn as u32, dm as u32,
            g_size as u32, nmax as u32, mmax as u32, ni as u32, nj as u32, nk as u32,
            axis_size as u32, nrys_roots as u32, ri, rj, rk, rirj, common_factor, &exps_i, &exps_j,
            &exps_k, &coeff_i, &coeff_j, &coeff_k, out_len,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_ssc_scalar_device::<cubecl_hip::HipRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, dn as u32, dm as u32,
            g_size as u32, nmax as u32, mmax as u32, ni as u32, nj as u32, nk as u32,
            axis_size as u32, nrys_roots as u32, ri, rj, rk, rirj, common_factor, &exps_i, &exps_j,
            &exps_k, &coeff_i, &coeff_j, &coeff_k, out_len,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_ssc_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lj as u32, lk as u32, n_prim_i as u32, n_prim_j as u32,
            n_prim_k as u32, n_ctr_i as u32, n_ctr_j as u32, n_ctr_k as u32, dn as u32, dm as u32,
            g_size as u32, nmax as u32, mmax as u32, ni as u32, nj as u32, nk as u32,
            axis_size as u32, nrys_roots as u32, ri, rj, rk, rirj, common_factor, &exps_i, &exps_j,
            &exps_k, &coeff_i, &coeff_j, &coeff_k, out_len,
        ),
    };

    // HOST part of the split: un-swap the ibase canonicalization, then SSC c2s.
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

// ─────────────────────────────────────────────────────────────────────────────
//  Host reference helpers — `#[cfg(test)]` only.
//
//  The production scalar ssc path runs `ssc_scalar_kernel` on-device; these host
//  fns are the `device_tests` cross-check oracle (verbatim from the pre-port
//  implementation, bodies unchanged).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
//  Scalar ssc device kernel — `#[cube(launch)]`, generic over `F: Float`
//
//  Faithful inline port of the host SCALAR pipeline
//  `fill_g_tensor_3c2e_ssc` → `split_ij_hrr_ssc` → `contract_3c2e_ssc`,
//  accumulated over all primitive triples (ip,jp,kp) and contraction triples
//  (ci,cj,ck) into a SINGLE i-fastest Cartesian block `cart_out[(k*ncj+j)*nci+i]`
//  (matching the host single-block collapse). Scratch `g` holds the 3-axis VRR
//  G-tensor (3*g_size); scratch `gsplit` holds the 3-axis HRR-split tensor
//  (3*axis_size); scratch `work` is the per-(axis,k,root) j-HRR temp
//  (nj*(nmax+1)). `#[comptime] nroots` selects rys_root{1..5}.
// ─────────────────────────────────────────────────────────────────────────────

/// Single-work-item scalar ssc kernel. See module note above.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn ssc_scalar_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    exps_k: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    coeff_k: &Array<F>,
    g: &mut Array<F>,
    gsplit: &mut Array<F>,
    work: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    rkx: F,
    rky: F,
    rkz: F,
    rirjx: F,
    rirjy: F,
    rirjz: F,
    common_factor: F,
    pie4: F,
    li: u32,
    lj: u32,
    lk: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    dn: u32,
    dm: u32,
    g_size: u32,
    nmax: u32,
    mmax: u32,
    ni: u32,
    nj: u32,
    nk: u32,
    axis_size: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let nrys = nroots;
        let total_g = 3u32 * g_size;
        let total_split = 3u32 * axis_size;
        let work_stride = nmax + 1u32;
        let work_len = nj * work_stride;

        let nfi = (li + 1u32) * (li + 2u32) / 2u32;
        let nfj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let nfk = (lk + 1u32) * (lk + 2u32) / 2u32;
        let out_len = nfi * nfj * nfk;

        // Zero the single accumulation block.
        let mut oi = 0u32;
        while oi < out_len {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // Primitive triple loop (ip, jp, kp).
        let mut ip = 0u32;
        while ip < nprim_i {
            let ai = exps_i[ip as usize];
            let mut jp = 0u32;
            while jp < nprim_j {
                let aj = exps_j[jp as usize];

                // Inline compute_pdata_host(ai, aj, ri, rj, 1.0, 1.0):
                //   zeta_ab = ai+aj, center_p = (ai*ri+aj*rj)/zeta_ab,
                //   fac = exp(-ai*aj/zeta_ab * |ri-rj|^2)   (norms = 1.0).
                let aij = ai + aj;
                let px = (ai * rix + aj * rjx) / aij;
                let py = (ai * riy + aj * rjy) / aij;
                let pz = (ai * riz + aj * rjz) / aij;
                let dijx = rix - rjx;
                let dijy = riy - rjy;
                let dijz = riz - rjz;
                let rr_ij = dijx * dijx + dijy * dijy + dijz * dijz;
                let fac_ij = F::exp(-ai * aj / aij * rr_ij);

                let mut kp = 0u32;
                while kp < nprim_k {
                    let ak = exps_k[kp as usize];
                    let akl = ak;

                    let xij_kl = px - rkx;
                    let yij_kl = py - rky;
                    let zij_kl = pz - rkz;
                    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

                    let a1 = aij * akl;
                    let a0 = a1 / (aij + akl);
                    // fac_env = common_factor * pair.fac; fac1 = sqrt(a0/(a1^3))*fac_env.
                    let fac_env = common_factor * fac_ij;
                    let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * fac_env;
                    let x_rys = a0 * rr;

                    // Rys roots/weights (comptime nroots branch).
                    if comptime!(nroots == 1u32) {
                        rys_root1::<F>(x_rys, urys, wrys, pie4);
                    } else if comptime!(nroots == 2u32) {
                        rys_root2::<F>(x_rys, urys, wrys, pie4);
                    } else if comptime!(nroots == 3u32) {
                        rys_root3::<F>(x_rys, urys, wrys, pie4);
                    } else if comptime!(nroots == 4u32) {
                        rys_root4::<F>(x_rys, urys, wrys, pie4);
                    } else {
                        rys_root5::<F>(x_rys, urys, wrys, pie4);
                    }

                    // rijrx = center_p - ri (host fill uses ri here).
                    let rijrxx = px - rix;
                    let rijrxy = py - riy;
                    let rijrxz = pz - riz;

                    // ── Build the [gx|gy|gz] G-tensor (fill_g_tensor_3c2e_ssc) ──
                    let mut gi = 0u32;
                    while gi < total_g {
                        g[gi as usize] = F::new(0.0);
                        gi += 1u32;
                    }

                    let mut irys = 0u32;
                    while irys < nrys {
                        let u2 = a0 * urys[irys as usize];
                        let tmp4 = F::new(0.5) / (u2 * (aij + akl) + a1);
                        let tmp5 = u2 * tmp4;
                        let b00 = tmp5;
                        let b10 = tmp5 + tmp4 * akl;
                        let b01 = tmp5 + tmp4 * aij;

                        let tmp2 = F::new(2.0) * tmp5 * akl;
                        let tmp3 = F::new(2.0) * tmp5 * aij;

                        // Seed for this root.
                        g[irys as usize] = F::new(1.0);
                        g[(g_size + irys) as usize] = F::new(1.0);
                        g[(2u32 * g_size + irys) as usize] = wrys[irys as usize] * fac1;

                        let mut axis = 0u32;
                        while axis < 3u32 {
                            let axis_off = axis * g_size;
                            let mut xkl = xij_kl;
                            let mut rijrx = rijrxx;
                            if axis == 1u32 {
                                xkl = yij_kl;
                                rijrx = rijrxy;
                            }
                            if axis == 2u32 {
                                xkl = zij_kl;
                                rijrx = rijrxz;
                            }
                            let c00_axis = rijrx - tmp2 * xkl;
                            let c0p_axis = tmp3 * xkl;

                            // n-recurrence (bra index, stride dn).
                            if nmax > 0u32 {
                                let mut s_prev = g[(axis_off + irys) as usize];
                                let mut s1 = c00_axis * s_prev;
                                g[(axis_off + irys + dn) as usize] = s1;
                                let mut n = 1u32;
                                while n < nmax {
                                    let s2 = c00_axis * s1 + F::cast_from(n) * b10 * s_prev;
                                    g[(axis_off + irys + (n + 1u32) * dn) as usize] = s2;
                                    s_prev = s1;
                                    s1 = s2;
                                    n += 1u32;
                                }
                            }

                            if mmax > 0u32 {
                                // m-recurrence on the n=0 column (ket index, stride dm).
                                let mut s_prev = g[(axis_off + irys) as usize];
                                let mut s1 = c0p_axis * s_prev;
                                g[(axis_off + irys + dm) as usize] = s1;
                                let mut m = 1u32;
                                while m < mmax {
                                    let s2 = c0p_axis * s1 + F::cast_from(m) * b01 * s_prev;
                                    g[(axis_off + irys + (m + 1u32) * dm) as usize] = s2;
                                    s_prev = s1;
                                    s1 = s2;
                                    m += 1u32;
                                }

                                // n>0, m>0 cross terms (b00 coupling).
                                if nmax > 0u32 {
                                    let mut n2 = 1u32;
                                    while n2 <= nmax {
                                        let i_off = irys + n2 * dn;
                                        let s0_k0 = g[(axis_off + i_off) as usize];
                                        let prev_i_k0 =
                                            g[(axis_off + irys + (n2 - 1u32) * dn) as usize];
                                        let mut s1c = c0p_axis * s0_k0 + b00 * prev_i_k0;
                                        g[(axis_off + i_off + dm) as usize] = s1c;
                                        let mut s_prevc = s0_k0;
                                        let mut m2 = 1u32;
                                        while m2 < mmax {
                                            let prev_i_km = g[(axis_off
                                                + irys
                                                + (n2 - 1u32) * dn
                                                + m2 * dm)
                                                as usize];
                                            let s2c = c0p_axis * s1c
                                                + F::cast_from(m2) * b01 * s_prevc
                                                + b00 * prev_i_km;
                                            g[(axis_off + i_off + (m2 + 1u32) * dm) as usize] = s2c;
                                            s_prevc = s1c;
                                            s1c = s2c;
                                            m2 += 1u32;
                                        }
                                        n2 += 1u32;
                                    }
                                }
                            }

                            axis += 1u32;
                        }
                        irys += 1u32;
                    }

                    // ── ij→(i,j) HRR split (split_ij_hrr_ssc) ────────────────
                    let mut si = 0u32;
                    while si < total_split {
                        gsplit[si as usize] = F::new(0.0);
                        si += 1u32;
                    }

                    let mut saxis = 0u32;
                    while saxis < 3u32 {
                        let axis_in_off = saxis * g_size;
                        let axis_out_off = saxis * axis_size;
                        let mut rirj = rirjx;
                        if saxis == 1u32 {
                            rirj = rirjy;
                        }
                        if saxis == 2u32 {
                            rirj = rirjz;
                        }

                        let mut k = 0u32;
                        while k <= mmax {
                            let mut root = 0u32;
                            while root < nrys {
                                // Load the merged-bra column into `work`.
                                let mut wz = 0u32;
                                while wz < work_len {
                                    work[wz as usize] = F::new(0.0);
                                    wz += 1u32;
                                }
                                let mut iw = 0u32;
                                while iw <= nmax {
                                    work[iw as usize] = g[(axis_in_off + root + iw * dn + k * dm)
                                        as usize];
                                    iw += 1u32;
                                }

                                // j-HRR: work[j][i] = rirj*work[j-1][i] + work[j-1][i+1].
                                let mut j = 1u32;
                                while j <= lj {
                                    let prev = (j - 1u32) * work_stride;
                                    let cur = j * work_stride;
                                    let i_max = nmax - j;
                                    let mut i = 0u32;
                                    while i <= i_max {
                                        work[(cur + i) as usize] = rirj
                                            * work[(prev + i) as usize]
                                            + work[(prev + i + 1u32) as usize];
                                        i += 1u32;
                                    }
                                    j += 1u32;
                                }

                                // Scatter into gsplit[(root*nk+k)*nj+j)*ni+i].
                                let mut j2 = 0u32;
                                while j2 <= lj {
                                    let mut i2 = 0u32;
                                    while i2 <= li {
                                        let out_idx = ((root * nk + k) * nj + j2) * ni + i2;
                                        gsplit[(axis_out_off + out_idx) as usize] =
                                            work[(j2 * work_stride + i2) as usize];
                                        i2 += 1u32;
                                    }
                                    j2 += 1u32;
                                }

                                root += 1u32;
                            }
                            k += 1u32;
                        }

                        saxis += 1u32;
                    }

                    // ── Contract into the Cartesian block (contract_3c2e_ssc) ──
                    // cart_out[(k_idx*ncj+j_idx)*nci+i_idx], i fastest.
                    // cart_comps walk: descending lx then ly (matches host cart_comps).
                    let gy_off = axis_size;
                    let gz_off = 2u32 * axis_size;

                    // Load this triple's contraction weight product across all
                    // (ci,cj,ck) — accumulated into the single block (host collapse).
                    let mut k_idx = 0u32;
                    let mut ka = 0u32;
                    while ka <= lk {
                        let kx = lk - ka;
                        let lk_minus = lk - kx;
                        let mut kb = 0u32;
                        while kb <= lk_minus {
                            let ky = lk_minus - kb;
                            let kz = lk - kx - ky;

                            let mut j_idx = 0u32;
                            let mut ja = 0u32;
                            while ja <= lj {
                                let jx = lj - ja;
                                let lj_minus = lj - jx;
                                let mut jb = 0u32;
                                while jb <= lj_minus {
                                    let jy = lj_minus - jb;
                                    let jz = lj - jx - jy;

                                    let mut i_idx = 0u32;
                                    let mut ia = 0u32;
                                    while ia <= li {
                                        let ix = li - ia;
                                        let li_minus = li - ix;
                                        let mut ib = 0u32;
                                        while ib <= li_minus {
                                            let iy = li_minus - ib;
                                            let iz = li - ix - iy;

                                            let mut val = F::new(0.0);
                                            let mut root = 0u32;
                                            while root < nrys {
                                                let idx_x =
                                                    ((root * nk + kx) * nj + jx) * ni + ix;
                                                let idx_y =
                                                    ((root * nk + ky) * nj + jy) * ni + iy;
                                                let idx_z =
                                                    ((root * nk + kz) * nj + jz) * ni + iz;
                                                val += gsplit[idx_x as usize]
                                                    * gsplit[(gy_off + idx_y) as usize]
                                                    * gsplit[(gz_off + idx_z) as usize];
                                                root += 1u32;
                                            }

                                            // Sum coeff weights over all contractions
                                            // into the single block (host behavior).
                                            let mut wsum = F::new(0.0);
                                            let mut ci = 0u32;
                                            while ci < nctr_i {
                                                let cvi =
                                                    coeff_i[(ip * nctr_i + ci) as usize];
                                                let mut cj = 0u32;
                                                while cj < nctr_j {
                                                    let cvj =
                                                        coeff_j[(jp * nctr_j + cj) as usize];
                                                    let mut ck = 0u32;
                                                    while ck < nctr_k {
                                                        let cvk = coeff_k
                                                            [(kp * nctr_k + ck) as usize];
                                                        wsum += cvi * cvj * cvk;
                                                        ck += 1u32;
                                                    }
                                                    cj += 1u32;
                                                }
                                                ci += 1u32;
                                            }

                                            let oidx =
                                                (k_idx * nfj + j_idx) * nfi + i_idx;
                                            cart_out[oidx as usize] += wsum * val;

                                            i_idx += 1u32;
                                            ib += 1u32;
                                        }
                                        ia += 1u32;
                                    }
                                    j_idx += 1u32;
                                    jb += 1u32;
                                }
                                ja += 1u32;
                            }
                            k_idx += 1u32;
                            kb += 1u32;
                        }
                        ka += 1u32;
                    }

                    kp += 1u32;
                }
                jp += 1u32;
            }
            ip += 1u32;
        }
    }
}

/// Dispatch [`ssc_scalar_kernel`] at `f64` on a resolved backend client and read
/// back the single i-fastest Cartesian block (`out_len = nci*ncj*nck`).
#[allow(clippy::too_many_arguments)]
fn run_ssc_scalar_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    lk: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    dn: u32,
    dm: u32,
    g_size: u32,
    nmax: u32,
    mmax: u32,
    ni: u32,
    nj: u32,
    nk: u32,
    axis_size: u32,
    nroots: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rirj: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
    out_len: usize,
) -> Vec<f64> {
    let nroots_u = nroots as usize;
    let g_size_u = g_size as usize;
    let axis_size_u = axis_size as usize;
    let work_len = (nj * (nmax + 1)) as usize;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let exps_k_h = client.create_from_slice(f64::as_bytes(exps_k));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let coeff_k_h = client.create_from_slice(f64::as_bytes(coeff_k));

    let g_zero = vec![0.0_f64; 3 * g_size_u];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let gsplit_zero = vec![0.0_f64; 3 * axis_size_u];
    let gsplit_h = client.create_from_slice(f64::as_bytes(&gsplit_zero));
    let work_zero = vec![0.0_f64; work_len.max(1)];
    let work_h = client.create_from_slice(f64::as_bytes(&work_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($nr:literal) => {
            ssc_scalar_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h, exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h, exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_k_h, exps_k.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h, coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_k_h, coeff_k.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size_u) },
                unsafe { ArrayArg::from_raw_parts(gsplit_h, 3 * axis_size_u) },
                unsafe { ArrayArg::from_raw_parts(work_h, work_len.max(1)) },
                unsafe { ArrayArg::from_raw_parts(u_h, nroots_u) },
                unsafe { ArrayArg::from_raw_parts(w_h, nroots_u) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                rk[0],
                rk[1],
                rk[2],
                rirj[0],
                rirj[1],
                rirj[2],
                common_factor,
                PIE4,
                li,
                lj,
                lk,
                nprim_i,
                nprim_j,
                nprim_k,
                nctr_i,
                nctr_j,
                nctr_k,
                dn,
                dm,
                g_size,
                nmax,
                mmax,
                ni,
                nj,
                nk,
                axis_size,
                $nr,
            )
        };
    }

    match nroots {
        1 => launch_with!(1u32),
        2 => launch_with!(2u32),
        3 => launch_with!(3u32),
        4 => launch_with!(4u32),
        _ => launch_with!(5u32),
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

#[cfg(all(test, feature = "cpu"))]
mod device_tests {
    use super::*;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host single-triple Cartesian reference:
    /// fill_g_tensor_3c2e_ssc → split_ij_hrr_ssc → contract_3c2e_ssc for one
    /// primitive triple and one contraction (coeff weights applied).
    #[allow(clippy::too_many_arguments)]
    fn host_cart_ssc(
        li: u8,
        lj: u8,
        lk: u8,
        ai: f64,
        aj: f64,
        ak: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rk: [f64; 3],
        common_factor: f64,
        ci: f64,
        cj: f64,
        ck: f64,
    ) -> Vec<f64> {
        let nrys_roots = (li as usize + lj as usize + lk as usize) / 2 + 1;
        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
        let pair = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let fac_env = common_factor * pair.fac;
        let g2d = fill_g_tensor_3c2e_ssc(&pair, ak, ri, rk, li, lj, lk, nrys_roots, fac_env);
        let g_split = split_ij_hrr_ssc(&g2d, li, lj, lk, nrys_roots, rirj);
        let prim = contract_3c2e_ssc(&g_split, li, lj, lk, nrys_roots);
        let weight = ci * cj * ck;
        prim.iter().map(|&v| v * weight).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_device_matches_host(li: u8, lj: u8, lk: u8, ai: f64, aj: f64, ak: f64) {
        // NOTE: the device path is wired AFTER the host ibase swap (li>=lj). These
        // triples are pre-canonicalized (li>=lj) so this direct comparison drives
        // the same kernel code launch_ssc uses; the (s,p,s) triple exercises the
        // ij-HRR split branch (lj>0).
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 1.1];
        let rk = [0.7_f64, 0.3, 0.0];
        let ci = 0.9_f64;
        let cj = 1.1_f64;
        let ck = 0.8_f64;
        let common_factor =
            (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

        let nrys_roots = (li as usize + lj as usize + lk as usize) / 2 + 1;
        let nmax = li as usize + lj as usize;
        let mmax = lk as usize;
        let dn = nrys_roots;
        let dm = nrys_roots * (nmax + 1);
        let g_size = nrys_roots * (nmax + 1) * (mmax + 1);
        let ni = li as usize + 1;
        let nj = lj as usize + 1;
        let nk = lk as usize + 1;
        let axis_size = nrys_roots * nk * nj * ni;

        let nci = ncart(li);
        let ncj = ncart(lj);
        let nck = ncart(lk);
        let out_len = nci * ncj * nck;
        let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

        let host = host_cart_ssc(li, lj, lk, ai, aj, ak, ri, rj, rk, common_factor, ci, cj, ck);
        let dev = run_ssc_scalar_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
            li as u32,
            lj as u32,
            lk as u32,
            1,
            1,
            1,
            1,
            1,
            1,
            dn as u32,
            dm as u32,
            g_size as u32,
            nmax as u32,
            mmax as u32,
            ni as u32,
            nj as u32,
            nk as u32,
            axis_size as u32,
            nrys_roots as u32,
            ri,
            rj,
            rk,
            rirj,
            common_factor,
            &[ai],
            &[aj],
            &[ak],
            &[ci],
            &[cj],
            &[ck],
            out_len,
        );

        assert_eq!(host.len(), dev.len(), "length mismatch for ({li},{lj},{lk})");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch ({li},{lj},{lk}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    // (s,s,s): nroots=1, no HRR split (lj=0), mmax=0.
    #[test]
    fn test_ssc_device_sss() {
        assert_device_matches_host(0, 0, 0, 1.0, 0.8, 0.9);
    }

    // (p,s,s): li>lj, nmax=1, exercises the n-recurrence; no ij-HRR (lj=0).
    #[test]
    fn test_ssc_device_pss() {
        assert_device_matches_host(1, 0, 0, 0.8, 1.0, 0.9);
    }

    // (s,p,s): lj>0 → exercises the ij-HRR split branch. NOTE: pre-canonicalized
    // here as (s,p,..) directly so the kernel itself runs the j-HRR (the host
    // launch_ssc would swap to (p,s) first, but the kernel math is what we check).
    #[test]
    fn test_ssc_device_sps() {
        assert_device_matches_host(0, 1, 0, 1.0, 0.8, 0.9);
    }

    // (s,s,p): mmax=1 → exercises the ket m-recurrence and b00 cross term.
    #[test]
    fn test_ssc_device_ssp() {
        assert_device_matches_host(0, 0, 1, 1.0, 0.8, 0.9);
    }

    // (p,p,s): li=lj=1, nmax=2 → ij-HRR split with i+1 mixing; nroots=2.
    #[test]
    fn test_ssc_device_pps() {
        assert_device_matches_host(1, 1, 0, 0.8, 0.9, 1.0);
    }

    // (d,s,s): higher li, nmax=2, nroots=2.
    #[test]
    fn test_ssc_device_dss() {
        assert_device_matches_host(2, 0, 0, 0.7, 1.0, 0.9);
    }

    /// Genericity: the kernel compiles AND launches for F = f32. An s-s-s f32
    /// launch yields a finite result.
    #[test]
    fn test_ssc_scalar_kernel_generic_f32() {
        let client = cpu_client();
        let li = 0u32;
        let lj = 0u32;
        let lk = 0u32;
        let nrys_roots = 1usize;
        let nmax = 0usize;
        let mmax = 0usize;
        let dn = nrys_roots;
        let dm = nrys_roots * (nmax + 1);
        let g_size = nrys_roots * (nmax + 1) * (mmax + 1);
        let ni = 1usize;
        let nj = 1usize;
        let nk = 1usize;
        let axis_size = nrys_roots * nk * nj * ni;
        let work_len = nj * (nmax + 1);

        let g_zero = vec![0.0_f32; 3 * g_size];
        let gsplit_zero = vec![0.0_f32; 3 * axis_size];
        let work_zero = vec![0.0_f32; work_len.max(1)];
        let rys_zero = vec![0.0_f32; nrys_roots];
        let out_zero = [0.0_f32; 1];
        let one = [1.0_f32];

        let exps_i_h = client.create_from_slice(f32::as_bytes(&one));
        let exps_j_h = client.create_from_slice(f32::as_bytes(&one));
        let exps_k_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_j_h = client.create_from_slice(f32::as_bytes(&one));
        let coeff_k_h = client.create_from_slice(f32::as_bytes(&one));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let gsplit_h = client.create_from_slice(f32::as_bytes(&gsplit_zero));
        let work_h = client.create_from_slice(f32::as_bytes(&work_zero));
        let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let common_factor =
            ((PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(0) * common_fac_sp(0) * common_fac_sp(0))
                as f32;

        ssc_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size) },
            unsafe { ArrayArg::from_raw_parts(gsplit_h, 3 * axis_size) },
            unsafe { ArrayArg::from_raw_parts(work_h, work_len.max(1)) },
            unsafe { ArrayArg::from_raw_parts(u_h, nrys_roots) },
            unsafe { ArrayArg::from_raw_parts(w_h, nrys_roots) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32,
            0.0,
            0.0,
            0.0,
            0.0,
            1.1,
            0.7,
            0.3,
            0.0,
            0.0,
            0.0,
            -1.1,
            common_factor,
            PIE4 as f32,
            li,
            lj,
            lk,
            1,
            1,
            1,
            1,
            1,
            1,
            dn as u32,
            dm as u32,
            g_size as u32,
            nmax as u32,
            mmax as u32,
            ni as u32,
            nj as u32,
            nk as u32,
            axis_size as u32,
            1u32,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 scalar ssc kernel result must be finite");
    }
}
