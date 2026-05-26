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
use crate::kernels::two_electron::{build_2e_shape, fill_g_tensor_2e, two_e_shape_as_f12};
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
/// `int3c2e_ip1` gradient launch — the `∇_A` first-center derivative of the
/// three-center two-electron Coulomb integral (GRAD-08 / Risk R1).
///
/// Mirrors `two_electron.rs::launch_two_electron_ip1`, applying the 3c2e Pitfall-4
/// kl mapping (file header lines 6-12):
///   - 2e "ij side"  ← real `(i, j)` (the bra; `i` raised to `li+1` for `∇_i`)
///   - 2e `ll` slot   ← real `k` (the only real ket-side angular axis)
///   - 2e `lk` slot   ← phantom s-function (`lk_ceil = 0`, exponent `ak = 0`)
///
/// Builds the plain Coulomb G-tensor through the SHARED 2e recurrence
/// ([`fill_g_tensor_2e`]) with `li_ceil = li+1` headroom, reuses
/// [`crate::kernels::f12::gout_ip1`] verbatim, and emits 3-component component-leading
/// `[3, nk, nj, ni]` F-order (same convention as `int2e_ip1`).
///
/// Guards (fail-closed):
///   - `Representation::Spinor` → `UnsupportedApi` (R5 / T-21-06-04).
///   - `grad_shape.nroots > 5` → `UnsupportedApi` (R2 / T-21-06-04): the `li→li+1`
///     raise can push high-l triples past the rys_root1..5 ceiling; reject BEFORE
///     any rys dispatch.
///
/// No `swap_ij` canonicalization: the derivative acts on the first (`i`) shell, and
/// the output keeps the caller's `(i, j, k)` shell order.
#[allow(clippy::too_many_arguments)]
fn launch_center_3c2e_ip1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    shell_i: &cintx_core::Shell,
    shell_j: &cintx_core::Shell,
    shell_k: &cintx_core::Shell,
    li: u8,
    lj: u8,
    lk: u8,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // R5 / T-21-06-04: spinor gradient is not supported. Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int3c2e_ip1 gradient".to_owned(),
        });
    }

    // 3c2e kl mapping into the 2e shape (Pitfall-4): real k → 2e `ll` slot, phantom
    // 2e `lk` slot = 0; bra `i` raised to `li+1` so `nabla1i_2e` can read index li+1.
    let grad_shape = build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize);

    // R2 / T-21-06-04: the elevated li can push nroots past the rys_root1..5 ceiling.
    // Reject fail-closed BEFORE any rys_roots_host call (which would otherwise panic).
    if grad_shape.nroots > 5 {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    // From CINTinit_int3c2e_EnvVars (same prefactor as the scalar 3c2e path).
    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk);

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let block_len = nci * ncj * nck; // per-component Cartesian AO product
    let total_len = 3 * block_len;

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // Per-contraction-triple component-leading Cartesian accumulator.
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    for kp in 0..n_prim_k {
        let ak = shell_k.exponents[kp];
        for jp in 0..n_prim_j {
            let aj = shell_j.exponents[jp];
            for ip in 0..n_prim_i {
                let ai = shell_i.exponents[ip];

                // 3c2e mapping for the 2e G-tensor fill:
                //   bra pair (i, j): ai, aj at ri, rj.
                //   ket pair: phantom 2e `lk` shell (exponent 0, at the real-k center)
                //   in the lk-slot; real k in the 2e `ll` slot (exponent ak at rk).
                //
                // `fill_g_tensor_2e` computes the bra-ket Rys prefactor but NOT the
                // intra-pair Gaussian product factors — those come from the pdata
                // `fac` (g1e.c:134) and must be folded into `fac_env`, exactly as
                // `launch_two_electron_ip1` does (`quartet_fac = common_factor *
                // pdata_ij.fac * pdata_kl.fac`). For the phantom-real_k ket pair the
                // product factor is `exp(-0) = 1`, but we compute it for fidelity.
                let pdata_ij = compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );
                let pdata_kl = compute_pdata_host(
                    0.0, ak, rk[0], rk[1], rk[2], rk[0], rk[1], rk[2], 1.0, 1.0,
                );
                let fac_env = common_factor * pdata_ij.fac * pdata_kl.fac;
                let g = fill_g_tensor_2e(
                    ai, aj, 0.0, ak, &ri, &rj, &rk, &rk, grad_shape, fac_env,
                );

                // Reuse gout_ip1 verbatim (f12.rs). Called at BASE li (the G-tensor
                // carries the li+1 headroom). With the phantom 2e lk=0, the gout
                // n-walk [ll, lk, lj, li] collapses to [real_k, (phantom size 1),
                // j, i] → effectively [k][j][i] (i fastest), matching the scalar
                // 3c2e cart layout the cart_to_sph_3c2e transform expects.
                let gout = crate::kernels::f12::gout_ip1(
                    &g,
                    &grad_f12_shape,
                    li as usize,
                    lj as usize,
                    0, // phantom 2e lk slot
                    lk as usize, // real k in the 2e ll slot
                    ai,
                );

                for ci in 0..n_ctr_i {
                    let coeff_i = shell_i.coefficients[ip * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let coeff_j = shell_j.coefficients[jp * n_ctr_j + cj];
                        for ck in 0..n_ctr_k {
                            let coeff_k = shell_k.coefficients[kp * n_ctr_k + ck];
                            let weight = coeff_i * coeff_j * coeff_k;
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len;
                            // TRANSPOSE interleaved gout[n*3+comp] into component-leading.
                            for n in 0..block_len {
                                for comp in 0..3usize {
                                    cart_blocks[base + comp * block_len + n] +=
                                        weight * gout[n * 3 + comp];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write component-leading `[3, nk, nj, ni]` F-order to staging. Per component,
    // the per-triple block is the i-fastest `[nk][nj][ni]` Cartesian tensor — run
    // the cart→sph 3c2e transform per component for the sph rep.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let sph_block = di * dj * dk;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len
                                + comp * block_len;
                            let sph = cart_to_sph_3c2e(
                                &cart_blocks[base..base + block_len],
                                li,
                                lj,
                                lk,
                            );
                            for mk in 0..nsk {
                                let kidx = ck * nsk + mk;
                                for mj in 0..nsj {
                                    let jidx = cj * nsj + mj;
                                    for mi in 0..nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * mk);
                                        let dst = staging_comp_base
                                            + iidx
                                            + di * (jidx + dj * kidx);
                                        if dst < staging.len() {
                                            staging[dst] = F::from_f64_lossy(sph[src]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            let di = n_ctr_i * nci;
            let dj = n_ctr_j * ncj;
            let dk = n_ctr_k * nck;
            let cart_block = di * dj * dk;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            let base = ((ci * n_ctr_j + cj) * n_ctr_k + ck) * total_len
                                + comp * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for kc in 0..nck {
                                let kidx = ck * nck + kc;
                                for jc in 0..ncj {
                                    let jidx = cj * ncj + jc;
                                    for ic in 0..nci {
                                        let iidx = ci * nci + ic;
                                        let src = ic + nci * (jc + ncj * kc);
                                        let dst = staging_comp_base
                                            + iidx
                                            + di * (jidx + dj * kidx);
                                        if dst < staging.len() {
                                            staging[dst] = F::from_f64_lossy(block[src]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => unreachable!("spinor int3c2e_ip1 rejected above"),
    }

    // Per-symbol nonzero sentinel (precision-aware; matches the scalar path).
    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
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

    // int3c2e_ip1 gradient path (Plan 21-06 / GRAD-08 / Risk R1).
    //
    // The scalar path below is operator-blind and silently returns the PLAIN 3c2e
    // integral; pyscf-grad's DF-gradient runtime consumes `int3c2e_ip1` as the
    // `∇_A` first-center DERIVATIVE. This branch ships the real derivative, reusing
    // `gout_ip1` (f12.rs, made pub(crate) by 21-05) verbatim — the same `∇_i` math
    // as `int2e_ip1` (two_electron.rs). It preserves the 3c2e Pitfall-4 kl mapping
    // (file header lines 6-12): real k is mapped to the 2e `ll` slot and the 2e `lk`
    // slot is a phantom s-function (`lk_ceil = 0`, exponent 0). The G-tensor is built
    // through the SAME 2e recurrence (`fill_g_tensor_2e`) the contraction expects.
    //
    // After 21-02's manifest change `id.operator` for int3c2e_ip1 is "ip1".
    if plan.descriptor.operator_name() == "ip1" {
        return launch_center_3c2e_ip1::<F>(
            plan, shell_i_in, shell_j_in, shell_k, li_in, lj_in, lk, staging,
        );
    }

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

// ─────────────────────────────────────────────────────────────────────────────
// int3c2e_ip1 gradient tests (Plan 21-06 / GRAD-08 / Risk R1).
//
// The latent bug: `launch_center_3c2e_typed` was operator-blind and returned the
// PLAIN (scalar) 3c2e integral for `int3c2e_ip1`. These tests pin the REAL
// 3-component derivative:
//   - component count: (s,s,s) → 3; (p,s,s) → 3×3 = 9 (the 3× multiplier is the
//     proof the scalar stub is gone — the stub returned 1×nci*ncj*nck).
//   - NOT-equal-to-plain: the (p,s,s) ip1 output is NOT element-wise equal to the
//     plain int3c2e output broadcast across components (regression-proof for R1).
//   - determinism (D-10): repeated evaluation is bit-identical.
//   - spinor (R5): int3c2e_ip1 with Representation::Spinor returns UnsupportedApi.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod ip1_tests {
    use super::*;
    use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
    use crate::specialization::SpecializationKey;
    use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell, ShellTuple};
    use cintx_ops::resolver::Resolver;
    use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
    use std::sync::Arc;

    /// Build a 3-shell (li, lj, lk) triple plan for the int3c2e_ip1 sph operator.
    fn build_ip1_plan(
        li: u8,
        lj: u8,
        lk: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom2 = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1, atom2].into_boxed_slice());

        let mk = |atom_index: u32, l: u8| {
            Arc::new(
                Shell::try_new(
                    atom_index,
                    l,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.8_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let s0 = mk(0, li);
        let s1 = mk(1, lj);
        let s2 = mk(2, lk);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2]).unwrap();

        let op = Resolver::descriptor_by_symbol("int3c2e_ip1_sph")
            .expect("int3c2e_ip1_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    /// Build a plain int3c2e_sph triple plan (same shells/centers) for the
    /// NOT-equal-to-plain regression comparison.
    fn build_plain_plan(
        li: u8,
        lj: u8,
        lk: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom2 = Atom::try_new(8, [0.7, 0.7, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1, atom2].into_boxed_slice());

        let mk = |atom_index: u32, l: u8| {
            Arc::new(
                Shell::try_new(
                    atom_index,
                    l,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.8_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let s0 = mk(0, li);
        let s1 = mk(1, lj);
        let s2 = mk(2, lk);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2]).unwrap();

        let op = Resolver::descriptor_by_symbol("int3c2e_sph")
            .expect("int3c2e_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    fn run(
        basis: &BasisSet,
        shells: ShellTuple,
        op: cintx_core::OperatorId,
        rep: Representation,
    ) -> Result<(Vec<f64>, ExecutionStats), cintxRsError> {
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, rep, basis, shells.clone(), &opts)?;
        let mut plan = ExecutionPlan::new(op, rep, basis, shells, &q)?;
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let out_elems = plan.output_layout.staging_elements;
        let mut staging = vec![0.0_f64; out_elems];
        let stats = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok((staging, stats))
    }

    // Component count (the 3× multiplier proves the scalar stub is gone). The scalar
    // stub only WROTE `nci*ncj*nck` elements (3 for (p,s,s)) and left the remaining
    // 6 lanes at zero. The real derivative must FILL all 3 component blocks with
    // genuinely-derived values. We assert the kernel writes a nonzero value into
    // each of the 3 component blocks (not just into the first block).
    #[test]
    fn test_int3c2e_ip1_component_count() {
        let (basis, shells, op) = build_ip1_plan(0, 0, 0);
        let (staging, _) = run(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(
            staging.len(),
            3,
            "(s,s,s) int3c2e_ip1 should produce 3 components, got {}",
            staging.len()
        );
        // s-s-s ∇_i with the test geometry (all atoms in the z=0 plane): the x and y
        // gradient components are nonzero; the z component vanishes by symmetry. A
        // genuine 3-component derivative therefore fills ≥2 lanes with distinct
        // nonzero values — the scalar stub (which broadcast a single scalar) would
        // give 3 identical lanes. We assert at least 2 nonzero AND not all equal.
        let nonzero = staging.iter().filter(|v| v.abs() > 1e-12).count();
        assert!(
            nonzero >= 2,
            "(s,s,s) int3c2e_ip1 must fill ≥2 component lanes (in-plane ∇_i): {staging:?}"
        );
        let all_equal = staging
            .iter()
            .all(|v| (v - staging[0]).abs() <= 1e-15);
        assert!(
            !all_equal,
            "(s,s,s) int3c2e_ip1 lanes are all identical — the scalar stub (R1) is \
             NOT closed: {staging:?}"
        );

        let (basis, shells, op) = build_ip1_plan(1, 0, 0);
        let (staging, _) = run(&basis, shells, op, Representation::Spheric).unwrap();
        // sph p = 3 AOs; 3 components × 3×1×1 = 9. The scalar stub gave 3 written.
        assert_eq!(
            staging.len(),
            9,
            "(p,s,s) int3c2e_ip1 should produce 9 outputs (3 comps × 3 AO), got {}",
            staging.len()
        );
        // PROOF the scalar stub is gone: every one of the 3 component blocks (each
        // of length nci*ncj*nck = 3) must carry a nonzero derived value. The stub
        // wrote only the first block (lanes 0..3) and left lanes 3..9 at zero, so
        // blocks 1 and 2 would be all-zero under the stub.
        let block_len = 3usize; // nci(p)*ncj(s)*nck(s) = 3*1*1
        for comp in 0..3usize {
            let block = &staging[comp * block_len..(comp + 1) * block_len];
            assert!(
                block.iter().any(|v| v.abs() > 1e-12),
                "(p,s,s) int3c2e_ip1 component block {comp} is all-zero — the scalar \
                 stub (R1) is NOT closed. staging={staging:?}"
            );
        }
    }

    // NOT-equal-to-plain (R1 regression proof): the (p,s,s) ip1 output must NOT be
    // element-wise equal to the plain int3c2e output broadcast across the 3 comps.
    // The scalar stub WROTE `plain` into the first component block (lanes 0..3) and
    // zeros into the rest, so the regression-proof here is the first block: a real
    // derivative ∇_i is NOT equal to the plain integral value itself.
    #[test]
    fn test_int3c2e_ip1_not_equal_to_plain() {
        let (basis_ip1, shells_ip1, op_ip1) = build_ip1_plan(1, 0, 0);
        let (ip1, _) = run(&basis_ip1, shells_ip1, op_ip1, Representation::Spheric).unwrap();

        let (basis_plain, shells_plain, op_plain) = build_plain_plan(1, 0, 0);
        let (plain, _) = run(&basis_plain, shells_plain, op_plain, Representation::Spheric).unwrap();

        // plain has 3 AOs (p,s,s); ip1 has 9 (3 comps × 3 AO).
        assert_eq!(plain.len(), 3, "plain (p,s,s) 3c2e should be 3 AOs");
        assert_eq!(ip1.len(), 9, "ip1 (p,s,s) 3c2e should be 9 (3 comps × 3 AO)");

        // The FIRST component block (lanes 0..3) is what the scalar stub wrote `plain`
        // into. A real ∇_i derivative differs from the plain integral value. If the
        // stub were still in place, ip1[0..3] would equal `plain` exactly.
        let first_block = &ip1[0..3];
        let first_block_equals_plain = first_block
            .iter()
            .zip(plain.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-12);
        assert!(
            !first_block_equals_plain,
            "int3c2e_ip1 first component block is byte-equal to the plain integral — \
             the scalar stub (R1) is NOT closed. ip1[0..3]={first_block:?} plain={plain:?}"
        );
        // The derivative must produce nonzero values across all 3 component blocks.
        assert!(
            ip1.iter().any(|v| v.abs() > 1e-12),
            "int3c2e_ip1 (p,s,s) output is all-zero: {ip1:?}"
        );
    }

    // Determinism (D-10): repeated evaluation is bit-identical.
    #[test]
    fn test_int3c2e_ip1_determinism() {
        let (basis, shells, op) = build_ip1_plan(1, 0, 0);
        let (out1, _) = run(&basis, shells.clone(), op, Representation::Spheric).unwrap();
        let (out2, _) = run(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "int3c2e_ip1 output not bit-identical across two evaluations"
            );
        }
    }

    // Spinor (R5): int3c2e_ip1 with Representation::Spinor returns UnsupportedApi.
    #[test]
    fn test_int3c2e_ip1_spinor_unsupported() {
        let (basis, shells, op) = build_ip1_plan(0, 0, 0);
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 6];
        let result = launch_center_3c2e_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "spinor int3c2e_ip1 should return UnsupportedApi, got: {result:?}"
        );
    }
}
