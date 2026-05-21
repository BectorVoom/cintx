//! Host-side 3c2e (three-center two-electron Coulomb) integral kernel.
//!
//! Implements the G-tensor fill + contraction + c2s pipeline following
//! libcint `g3c2e.c` / `cint3c2e.c` with shared 2e recurrence machinery from `g2e.c`.
//!
//! # Pitfall 4 mapping (critical)
//! 3c2e has real shells `(i, j, k)` but libcint reuses 2e machinery by mapping:
//! - 2e "ij side"  <- real `(i, j)`
//! - 2e "kl side"  <- real `k` mapped into the 2e `ll` slot
//! - 2e `lk` slot is a phantom s-function (`lk_ceil = 0`, `ak = 0`)
//! This file follows that mapping explicitly: the third center `k` is treated as
//! the 2e `ll` angular channel, with only one real "ket-side" angular axis.

use crate::backend::ResolvedBackend;
use crate::math::pdata::{PairData, compute_pdata_host};
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_3c2e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_3c2e;
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Matches libcint `CINTcommon_fac_sp(l)`:
///   l=0: 1/(2*sqrt(pi))
///   l=1: sqrt(3/(4*pi))
///   l>=2: 1.0
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Follows libcint `CINTcart_comp` ordering.
fn cart_comps(l: u8) -> Vec<(usize, usize, usize)> {
    let mut comps = Vec::new();
    let l = l as i32;
    let mut lx = l;
    while lx >= 0 {
        let mut ly = l - lx;
        while ly >= 0 {
            let lz = l - lx - ly;
            comps.push((lx as usize, ly as usize, lz as usize));
            ly -= 1;
        }
        lx -= 1;
    }
    comps
}

/// Fill the 2d 3c2e G-tensor for one primitive triple (ip, jp, kp).
///
/// The returned tensor is `[gx | gy | gz]` where each axis block has shape:
/// `[m=0..lk][n=0..(li+lj)][root=0..nrys-1]`, root-fastest.
///
/// This is the shared 2e recurrence stage before ij-HRR splitting:
/// - `n` corresponds to combined `(i+j)` angular order
/// - `m` corresponds to real third-center `k` angular order (2e ll-slot)
fn fill_g_tensor_3c2e(
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
    let akl = ak; // 3c2e mapping: 2e "kl" pair uses only the real k shell (l-slot), phantom k-slot has exponent 0.
    let p = [pair.center_p_x, pair.center_p_y, pair.center_p_z];

    // 2e-style pair displacement: rij - rkl with rij=P and rkl=Rk (mapped ll slot).
    let xij_kl = p[0] - rk[0];
    let yij_kl = p[1] - rk[1];
    let zij_kl = p[2] - rk[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

    let a1 = aij * akl;
    let a0 = a1 / (aij + akl);
    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_env;
    let x_rys = a0 * rr;
    let (u_roots, w_weights) = rys_roots_host(nrys_roots, x_rys);

    // 3c2e uses 2e recurrence with rx_in_rijrx = Ri and rx_in_rklrx = Rk.
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
        // rklrx = rkl - rk = 0 for the mapped ll-slot center, so c0p is only the coupling term.
        let c0p = [tmp3 * xij_kl, tmp3 * yij_kl, tmp3 * zij_kl];

        // Base for this Rys root.
        g[irys] = 1.0;
        g[g_size + irys] = 1.0;
        g[2 * g_size + irys] = w_weights[irys] * fac1;

        for axis in 0..3 {
            let axis_off = axis * g_size;
            let c00_axis = c00[axis];
            let c0p_axis = c0p[axis];

            // VRR in combined ij direction (n-axis).
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

            // VRR in mapped k(ll)-direction (m-axis), including b00 cross-coupling.
            if mmax > 0 {
                // n=0 ladder over m
                let mut s_prev = g[axis_off + irys];
                let mut s1 = c0p_axis * s_prev;
                g[axis_off + irys + dm] = s1;
                for m in 1..mmax {
                    let s2 = c0p_axis * s1 + m as f64 * b01 * s_prev;
                    g[axis_off + irys + (m + 1) * dm] = s2;
                    s_prev = s1;
                    s1 = s2;
                }

                // n>0 ladders over m with b00 cross term.
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

/// Split ij angular momentum for ibase=true layout.
///
/// Input `n` channel is the ij-base ladder (i-like axis) from 2e-style VRR.
/// We recover explicit `(i,j)` channels via HRR transfer along j:
/// `g(i,j,...) = (Ri-Rj) * g(i,j-1,...) + g(i+1,j-1,...)`.
///
/// Input:  `[axis][m][n][root]` from `fill_g_tensor_3c2e`
/// Output: `[axis][root][k][j][i]` (i fastest inside each root block).
fn split_ij_hrr(
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
                // Work rows are j (0..lj), columns are i-base index (0..li+lj).
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

/// Contract HRR-split G-tensor into Cartesian integral buffer.
///
/// Output layout: i fastest, j middle, k slowest:
/// `out[(k * ncj + j) * nci + i]`.
fn contract_3c2e(g: &[f64], li: u8, lj: u8, lk: u8, nrys_roots: usize) -> Vec<f64> {
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
                    let idx_x = ((root * nk + kx) * nj + jx) * ni + ix;
                    let idx_y = ((root * nk + ky) * nj + jy) * ni + iy;
                    let idx_z = ((root * nk + kz) * nj + jz) * ni + iz;
                    val += g[gx_off + idx_x] * g[gy_off + idx_y] * g[gz_off + idx_z];
                }
                out[(k_idx * ncj + j_idx) * nci + i_idx] += val;
            }
        }
    }

    out
}

/// Transpose a flat 3-index buffer from `(i,j,k)` to `(j,i,k)` ordering.
///
/// Input/output are both i-fastest, then j, then k slowest:
/// `idx = (k * nj + j) * ni + i`.
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

/// Generic inner for the 3c2e launcher.
///
/// Contains the full algorithm of `launch_center_3c2e` parameterized over the
/// output float type `F: CintFloat`. Intermediate computations (G-tensor, cart_buf)
/// remain `f64`; precision conversion happens only at the final staging write via
/// `F::from_f64_lossy`. Preserves the li>=lj canonicalization + transpose-back.
fn launch_center_3c2e_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "3c2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c2e",
            detail: format!(
                "canonical_family mismatch for 3c2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    // Host-side execution: no GPU dispatch in this phase.
    let _ = backend;

    let shells = plan.shells.as_slice();
    if shells.len() < 3 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_3c2e",
            detail: format!(
                "3c2e kernel requires exactly 3 shells, got {}",
                shells.len()
            ),
        });
    }

    let shell_i_in = &shells[0];
    let shell_j_in = &shells[1];
    let shell_k = &shells[2];

    let li_in = shell_i_in.ang_momentum;
    let lj_in = shell_j_in.ang_momentum;
    let lk = shell_k.ang_momentum;
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

    // Coordinates
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];

    // From CINTinit_int3c2e_EnvVars:
    // common_factor = pi^3 * 2 / sqrt(pi) * fac_sp(i) * fac_sp(j) * fac_sp(k)
    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsi_in = nsph(li_in);
    let nsj_in = nsph(lj_in);
    let nsk = nsph(lk);

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
                let g2d = fill_g_tensor_3c2e(
                    &pair, ak, ri, rk, li, lj, lk, nrys_roots, fac_env,
                );
                let g_split = split_ij_hrr(&g2d, li, lj, lk, nrys_roots, rirj);
                let prim_buf = contract_3c2e(&g_split, li, lj, lk, nrys_roots);

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
        // libcint's 3c2e recurrence chooses ibase adaptively (li > lj).
        // We evaluate in canonical order li>=lj and transpose back when input had li<lj.
        transpose_ij_3idx(&cart_buf, nci, ncj, nck)
    } else {
        cart_buf
    };

    // Apply cart-to-sph/spinor or copy Cartesian, casting to F at the staging write.
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_3c2e(&cart_out, li_in, lj_in, lk);
            let sph_size = nsi_in * nsj_in * nsk;
            let copy_len = staging.len().min(sph_size);
            for (dst, &src) in staging[..copy_len].iter_mut().zip(sph[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
        Representation::Spinor => {
            // cart_to_spinor_sf_3c2e is generic over F: CintFloat (Plan 04).
            let kappa_i = shell_i_in.kappa;
            let kappa_j = shell_j_in.kappa;
            cart_to_spinor_sf_3c2e::<F>(
                staging, &cart_out,
                li_in, kappa_i, lj_in, kappa_j, lk,
            )?;
        }
        Representation::Cart => {
            let copy_len = staging.len().min(cart_out.len());
            for (dst, &src) in staging[..copy_len].iter_mut().zip(cart_out[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
    }

    // Per-symbol nonzero sentinel
    // WR-06: precision-aware sentinel so f32 stale lanes (< f32 noise floor ~1e-7)
    // are not counted. The outer F32 arm already bounds staging to out_elems, so this
    // scan cannot touch stale upper-half lanes.
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = staging.len() * std::mem::size_of::<F>();
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

/// Outer precision dispatcher for the 3c2e kernel.
///
/// Keeps the registered `FamilyLaunchFn` signature unchanged. Internally matches on
/// `plan.precision` and delegates to `launch_center_3c2e_typed::<F>`, reinterpreting
/// staging via `bytemuck::cast_slice_mut` for the F32 arm (A5 proven sound).
/// CR-01: captures the true output element count BEFORE the bytemuck cast and bounds
/// the typed inner to that count, returning `BufferTooSmall` if the view cannot hold it.
pub fn launch_center_3c2e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_center_3c2e_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            // CR-01: capture the true output element count BEFORE the bytemuck cast.
            // api.rs sizes Vec<f64> to chunk_len == the TRUE output element count;
            // after cast staging_f32.len() == chunk_len*2, so out_elems = staging.len() pre-cast.
            let out_elems = staging.len(); // f64 slice length == TRUE output element count
            let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
            if staging_f32.len() < out_elems {
                return Err(cintxRsError::BufferTooSmall {
                    required: out_elems,
                    provided: staging_f32.len(),
                });
            }
            launch_center_3c2e_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1c: launch_center_3c2e_typed::<f64> is byte-identical to the
    // existing launch_center_3c2e at f64 (center_3c2e_parity).
    // RED: compile fails until launch_center_3c2e_typed is defined.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c2e_parity_f64() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(22), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(22), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging_outer = vec![0.0_f64; 1];
        let mut staging_typed = vec![0.0_f64; 1];

        // Call outer dispatcher
        let result_outer = launch_center_3c2e(&backend, &plan, &spec, &mut staging_outer);
        assert!(result_outer.is_ok(), "outer f64 3c2e should succeed: {:?}", result_outer);

        // Call typed inner directly (RED: compile fails until launch_center_3c2e_typed defined)
        let result_typed = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging_typed);
        assert!(result_typed.is_ok(), "typed f64 3c2e should succeed: {:?}", result_typed);

        // Byte-identical check
        assert_eq!(staging_outer[0].to_bits(), staging_typed[0].to_bits(),
            "f64 outer and typed 3c2e should be byte-identical: outer={} typed={}", staging_outer[0], staging_typed[0]);
        assert!(staging_outer[0].is_finite() && staging_outer[0].abs() > 1e-30,
            "3c2e s-s-s value should be finite and nonzero: {}", staging_outer[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-1d: launch_center_3c2e F32 path runs without panic.
    // RED: compile fails until launch_center_3c2e dispatches on plan.precision.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_center_3c2e_f32_smoke() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_c = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b, atom_c].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| Arc::new(Shell::try_new(
            atom_idx, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_a = make_s_shell(0);
        let shell_b = make_s_shell(1);
        let shell_c = make_s_shell(2);
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone(), shell_c.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b, shell_c]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(OperatorId::new(22), Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(OperatorId::new(22), Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_center_3c2e(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "F32 3c2e should succeed without panic: {:?}", result);

        let staging_f32 = bytemuck::cast_slice::<f64, f32>(&staging);
        assert!(staging_f32[0].is_finite(), "F32 3c2e result should be finite: {}", staging_f32[0]);
        assert!(staging_f32[0] > 0.0, "F32 3c2e result should be positive: {}", staging_f32[0]);
    }

    #[test]
    fn test_fill_g_tensor_3c2e_sss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.1, 0.2];
        let pair = compute_pdata_host(
            1.0, 1.0, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
        );

        let g = fill_g_tensor_3c2e(&pair, 1.0, ri, rk, 0, 0, 0, 1, 1.0);
        assert_eq!(g.len(), 3, "s-s-s should produce one root x one n x one m");
        assert!(g[2].abs() > 1e-20, "gz root must be non-zero for s-s-s primitive");
    }

    #[test]
    fn test_contract_3c2e_sss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 0.5];
        let rk = [0.0_f64, 0.1, 0.2];
        let pair = compute_pdata_host(
            1.0, 1.0, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
        );

        let g2d = fill_g_tensor_3c2e(&pair, 1.0, ri, rk, 0, 0, 0, 1, 1.0);
        let g_split = split_ij_hrr(&g2d, 0, 0, 0, 1, [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]]);
        let out = contract_3c2e(&g_split, 0, 0, 0, 1);
        assert_eq!(out.len(), 1);
        assert!(out[0].abs() > 1e-20, "contracted s-s-s 3c2e value must be non-zero");
    }
}
