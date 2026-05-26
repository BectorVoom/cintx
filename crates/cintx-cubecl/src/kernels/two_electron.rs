//! Host-side 2e (four-center electron-repulsion) integral kernel.
//!
//! Implements the libcint `g2e.c` recurrence pipeline:
//! 1. Rys roots/weights per primitive quartet (`rys_roots_host`).
//! 2. 2D VRR fill (`CINTg0_2e_2d` equivalent).
//! 3. Branch-specific 4D HRR transfer (ibase/kbase adaptive stride choice).
//! 4. Cartesian contraction + optional `cart_to_sph_2e` transform.

use crate::backend::ResolvedBackend;
use crate::math::pdata::compute_pdata_host;
use crate::math::rys::rys_roots_host;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, ncart, nsph};
use crate::transform::c2spinor::cart_to_spinor_sf_4d;
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
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

#[derive(Clone, Copy, Debug)]
struct TwoEShape {
    nroots: usize,
    nmax: usize,
    mmax: usize,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
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

/// Initialize stride/layout metadata following `CINTinit_int2e_EnvVars`.
fn build_2e_shape(li: usize, lj: usize, lk: usize, ll: usize) -> TwoEShape {
    let nroots = (li + lj + lk + ll) / 2 + 1;
    let nmax = li + lj;
    let mmax = lk + ll;

    // Adaptive branch selection from libcint (strict >).
    let ibase = li > lj;
    let kbase = lk > ll;

    let (dli, dlj) = if ibase {
        (li + lj + 1, lj + 1)
    } else {
        (li + 1, li + lj + 1)
    };
    let (dlk, dll) = if kbase {
        (lk + ll + 1, ll + 1)
    } else {
        (lk + 1, lk + ll + 1)
    };

    let di = nroots;
    let dk = nroots * dli;
    let dl = nroots * dli * dlk;
    let dj = nroots * dli * dlk * dll;
    let g_size = nroots * dli * dlk * dll * dlj;

    let g2d_ijmax = if ibase { di } else { dj };
    let g2d_klmax = if kbase { dk } else { dl };

    TwoEShape {
        nroots,
        nmax,
        mmax,
        li,
        lj,
        lk,
        ll,
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

#[inline]
fn vrr_fill_axis(
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

/// HRR branch for `ibase=false && kbase=false` (`CINTg0_lj2d_4d`).
fn hrr_lj2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.li == 0 && shape.lk == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rirj[axis];

        for i in 1..=shape.li {
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
        for j in 0..=shape.lj {
            for k in 1..=shape.lk {
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

/// HRR branch for `ibase=false && kbase=true` (`CINTg0_kj2d_4d`).
fn hrr_kj2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.li == 0 && shape.ll == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rirj[axis];

        for i in 1..=shape.li {
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
        for j in 0..=shape.lj {
            for l in 1..=shape.ll {
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

/// HRR branch for `ibase=true && kbase=false` (`CINTg0_il2d_4d`).
fn hrr_il2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.lj == 0 && shape.lk == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rkrl[axis];

        for k in 1..=shape.lk {
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
        for j in 1..=shape.lj {
            for l in 0..=shape.ll {
                for k in 0..=shape.lk {
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

/// HRR branch for `ibase=true && kbase=true` (`CINTg0_ik2d_4d`).
fn hrr_ik2d_4d(g: &mut [f64], shape: TwoEShape, rirj: [f64; 3], rkrl: [f64; 3]) {
    if shape.lj == 0 && shape.ll == 0 {
        return;
    }

    let nroots = shape.nroots;
    for axis in 0..3 {
        let off = axis * shape.g_size;
        let rx = rkrl[axis];

        for l in 1..=shape.ll {
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
        for j in 1..=shape.lj {
            for l in 0..=shape.ll {
                for k in 0..=shape.lk {
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

/// Fill the full `[gx|gy|gz]` tensor for one primitive quartet.
fn fill_g_tensor_2e(
    ai: f64,
    aj: f64,
    ak: f64,
    al: f64,
    ri: &[f64; 3],
    rj: &[f64; 3],
    rk: &[f64; 3],
    rl: &[f64; 3],
    shape: TwoEShape,
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
        vrr_fill_axis(
            gx,
            irys,
            shape.nmax,
            shape.mmax,
            shape.g2d_ijmax,
            shape.g2d_klmax,
            c00[0],
            c0p[0],
            b10,
            b01,
            b00,
        );
        vrr_fill_axis(
            gy,
            irys,
            shape.nmax,
            shape.mmax,
            shape.g2d_ijmax,
            shape.g2d_klmax,
            c00[1],
            c0p[1],
            b10,
            b01,
            b00,
        );
        vrr_fill_axis(
            gz,
            irys,
            shape.nmax,
            shape.mmax,
            shape.g2d_ijmax,
            shape.g2d_klmax,
            c00[2],
            c0p[2],
            b10,
            b01,
            b00,
        );
    }

    // HRR transfer to final (i,k,l,j) layout with branch-specific ordering.
    if shape.kbase {
        if shape.ibase {
            hrr_ik2d_4d(&mut g, shape, rirj, rkrl);
        } else {
            hrr_kj2d_4d(&mut g, shape, rirj, rkrl);
        }
    } else if shape.ibase {
        hrr_il2d_4d(&mut g, shape, rirj, rkrl);
    } else {
        hrr_lj2d_4d(&mut g, shape, rirj, rkrl);
    }

    g
}

/// Contract `[gx|gy|gz]` into Cartesian 2e tensor with output order:
/// `out[i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk]` (i fastest, l slowest).
fn contract_2e_cart(g: &[f64], shape: TwoEShape, li: u8, lj: u8, lk: u8, ll: u8) -> Vec<f64> {
    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);
    let ck_comps = cart_comps(lk);
    let cl_comps = cart_comps(ll);

    let gx_off = 0usize;
    let gy_off = shape.g_size;
    let gz_off = 2 * shape.g_size;

    let mut out = vec![0.0_f64; nfi * nfj * nfk * nfl];

    for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
        for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let mut sum = 0.0_f64;
                    for irys in 0..shape.nroots {
                        let x_idx = irys
                            + ix as usize * shape.di
                            + kx as usize * shape.dk
                            + lx as usize * shape.dl
                            + jx as usize * shape.dj;
                        let y_idx = irys
                            + iy as usize * shape.di
                            + ky as usize * shape.dk
                            + ly as usize * shape.dl
                            + jy as usize * shape.dj;
                        let z_idx = irys
                            + iz as usize * shape.di
                            + kz as usize * shape.dk
                            + lz as usize * shape.dl
                            + jz as usize * shape.dj;
                        sum += g[gx_off + x_idx] * g[gy_off + y_idx] * g[gz_off + z_idx];
                    }
                    let out_idx = i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                    out[out_idx] = sum;
                }
            }
        }
    }

    out
}

/// Generic inner for the 2e launcher. See `launch_two_electron` for the dispatch rationale.
///
/// Intermediate computations (G-tensor, cart_buf) remain `f64`; output staging
/// is written via `F::from_f64_lossy`. The `f64` monomorphization is byte-identical
/// to the pre-generic code.
fn launch_two_electron_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: format!(
                "canonical_family mismatch for 2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    // Host-side implementation (no CubeCL launch path here).
    let _ = backend;

    let shells = plan.shells.as_slice();
    if shells.len() < 4 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: "2e kernel requires exactly 4 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;
    let lk = shell_k.ang_momentum;
    let ll = shell_l.ang_momentum;

    let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);

    // Keep branch logic explicit for auditability against libcint Pitfall 1.
    let _ibase_kbase_used = (shape.ibase, shape.kbase);

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    let rl = atoms[shell_l.atom_index as usize].coord_bohr;

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let block_len = nfi * nfj * nfk * nfl;

    // Pitfall 2: all four common_fac_sp factors are required for 2e.
    let sp_factor = common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk) * common_fac_sp(ll);
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * sp_factor;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    // Per-contraction-quad Cartesian accumulators (mirrors the one_electron.rs
    // general-contraction fix, cintx 9af2164). A generally-contracted shell
    // (nctr>1 — e.g. cc-pVDZ's 9s→3s / 4p→2p blocks) emits ONE
    // nfi*nfj*nfk*nfl Cartesian block PER (ci,cj,ck,cl) contraction quad. The
    // contraction index is the OUTER (slow) AO index within each shell axis and
    // the angular component the inner index — libcint "contraction-major" AO
    // ordering. For the segmented case (all nctr==1) this is a single block,
    // byte-identical to the prior single-buffer path. Block (ci,cj,ck,cl) starts
    // at flat offset (((ci*n_ctr_j + cj)*n_ctr_k + ck)*n_ctr_l + cl) * block_len.
    // The previous code summed every contraction quad into ONE buffer and
    // scattered it as if nctr==1, leaving the inner-contraction AO blocks zero
    // (near-zero ERIs → SCF divergence for cc-pVDZ etc.).
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * block_len];

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

                    let g = fill_g_tensor_2e(ai, aj, ak, al, &ri, &rj, &rk, &rl, shape, quartet_fac);
                    let prim_cart = contract_2e_cart(&g, shape, li, lj, lk, ll);

                    for ci in 0..n_ctr_i {
                        let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                        for cj in 0..n_ctr_j {
                            let coeff_j = shell_j.coefficients[pj * n_ctr_j + cj];
                            for ck in 0..n_ctr_k {
                                let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                                for cl in 0..n_ctr_l {
                                    let coeff_l = shell_l.coefficients[pl * n_ctr_l + cl];
                                    let weight = coeff_i * coeff_j * coeff_k * coeff_l;
                                    let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l
                                        + cl)
                                        * block_len;
                                    for idx in 0..block_len {
                                        cart_blocks[base + idx] += weight * prim_cart[idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Representation dispatch: intermediate transforms use f64 temp buffers;
    // final values cast to F via F::from_f64_lossy.
    match plan.representation {
        Representation::Spheric => {
            // Per-contraction-quad cart→sph, scattered into the contraction-major
            // AO grid. di = n_ctr_i * nsi = shell_i.ao_per_shell(). cart_to_sph_2e
            // emits an i-fastest [nsl][nsk][nsj][nsi] block
            // (sph[mi + nsi*(mj + nsj*(mk + nsk*ml))]); the downstream stitch
            // (pyscf-gto evaluate_arity4) reads F-order i-fastest with AO index
            // ci*nsi+mi, so dst = ii + di*(jj + dj*(kk + dk*ll)). For all-nctr==1
            // (di==nsi, …) this is byte-identical to the prior linear copy.
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    for ck in 0..n_ctr_k {
                        for cl in 0..n_ctr_l {
                            let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                * block_len;
                            let sph =
                                cart_to_sph_2e(&cart_blocks[base..base + block_len], li, lj, lk, ll);
                            for ml in 0..nsl {
                                let lidx = cl * nsl + ml;
                                for mk in 0..nsk {
                                    let kidx = ck * nsk + mk;
                                    for mj in 0..nsj {
                                        let jidx = cj * nsj + mj;
                                        for mi in 0..nsi {
                                            let iidx = ci * nsi + mi;
                                            let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                            let dst = iidx + di * (jidx + dj * (kidx + dk * lidx));
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
        }
        Representation::Spinor => {
            // Single-contraction spinor preserves the exact prior behavior. General
            // contraction (nctr>1) is not wired for the 4D spinor transform (and is
            // not exercised by the non-relativistic callers); return an explicit
            // error rather than silently truncating to the (0,0,0,0) block.
            if n_ctr_i != 1 || n_ctr_j != 1 || n_ctr_k != 1 || n_ctr_l != 1 {
                return Err(cintxRsError::UnsupportedApi {
                    requested: "spinor 2e with general contraction (nctr>1)".to_owned(),
                });
            }
            // cart_blocks is exactly one nfi*nfj*nfk*nfl block here — identical to
            // the old cart_buf.
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            let kappa_k = shell_k.kappa;
            let kappa_l = shell_l.kappa;
            cart_to_spinor_sf_4d::<F>(
                staging, &cart_blocks,
                li, kappa_i, lj, kappa_j,
                lk, kappa_k, ll, kappa_l,
            )?;
        }
        Representation::Cart => {
            // Each contraction block is i-fastest [nfl][nfk][nfj][nfi]; scatter it
            // into the contraction-major AO grid (di = n_ctr_i*nfi) i-fastest to
            // match the Cart stitch. For all-nctr==1 this is the prior linear copy.
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    for ck in 0..n_ctr_k {
                        for cl in 0..n_ctr_l {
                            let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                * block_len;
                            let block = &cart_blocks[base..base + block_len];
                            for lc in 0..nfl {
                                let lidx = cl * nfl + lc;
                                for kc in 0..nfk {
                                    let kidx = ck * nfk + kc;
                                    for jc in 0..nfj {
                                        let jidx = cj * nfj + jc;
                                        for ic in 0..nfi {
                                            let iidx = ci * nfi + ic;
                                            let src = ic + nfi * (jc + nfj * (kc + nfk * lc));
                                            let dst = iidx + di * (jidx + dj * (kidx + dk * lidx));
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
        }
    }

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

/// Host-side 2e launcher — outer precision dispatcher.
///
/// Keeps the registered `FamilyLaunchFn` signature unchanged so the `as FamilyLaunchFn`
/// cast in `kernels/mod.rs` compiles. Dispatches on `plan.precision` to the generic inner.
pub fn launch_two_electron(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_two_electron_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            // F32 arm: capture the true output element count BEFORE the bytemuck cast.
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
            launch_two_electron_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// int2e_ip1 gradient tests (Plan 21-05)
//
// The behavior contract (per the plan):
//   - nroots guard: a quartet whose gradient nroots = (li+1+lj+lk+ll)/2+1 > 5
//     (e.g. an all-f quartet) returns UnsupportedApi; an s/p/d quartet does not.
//   - component count: an (s,s,s,s) quartet produces 3 outputs; a (p,s,s,s)
//     quartet produces 3 * 3*1*1*1 = 9.
//   - determinism: repeated evaluation is bit-identical (ordered reduction, D-10).
//   - spinor: int2e_ip1 with Representation::Spinor returns UnsupportedApi (R5).
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

    /// Build a 4-shell same-l quartet plan for the int2e_ip1 sph operator.
    ///
    /// Returns the plan plus a correctly-sized f64 staging buffer (the runtime
    /// planner already multiplies the AO product by the manifest `component_rank=3`).
    fn build_ip1_plan(
        l: u8,
        rep: Representation,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        // Two atoms so the four shells are not all on the same center (a same-center
        // s,s,s,s ERI gradient is nonzero only off-center; off-center keeps the math
        // exercised, but for the unit contract we only need shape/guard behavior).
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());

        let mk = |atom_index: u32| {
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
        let s0 = mk(0);
        let s1 = mk(1);
        let s2 = mk(0);
        let s3 = mk(1);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();

        let op = Resolver::descriptor_by_symbol("int2e_ip1_sph")
            .expect("int2e_ip1_sph must be in manifest")
            .id;
        let _ = rep;
        (basis, shells, op)
    }

    /// Build a (li, lj, lk, ll) quartet plan with explicit per-shell angular momenta.
    fn build_ip1_plan_lll(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
    ) -> (BasisSet, ShellTuple, cintx_core::OperatorId) {
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());

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
        let s2 = mk(0, lk);
        let s3 = mk(1, ll);

        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();

        let op = Resolver::descriptor_by_symbol("int2e_ip1_sph")
            .expect("int2e_ip1_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    fn run_ip1(
        basis: &BasisSet,
        shells: ShellTuple,
        op: cintx_core::OperatorId,
        rep: Representation,
    ) -> Result<(Vec<f64>, ExecutionStats), cintxRsError> {
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, rep, basis, shells.clone(), &opts)?;
        let mut plan = ExecutionPlan::new(op, rep, basis, shells, &q)?;
        plan.precision = cintx_core::PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        // Size staging to the planner-declared output element count (includes the
        // 3-component axis via component_rank=3).
        let out_elems = plan.output_layout.staging_elements;
        let mut staging = vec![0.0_f64; out_elems];
        let stats = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok((staging, stats))
    }

    // nroots guard (R2 / T-21-05-01): an all-f quartet has gradient
    // nroots = (3+1 + 3 + 3 + 3)/2 + 1 = 13/2 + 1 = 7 > 5 → UnsupportedApi.
    // An s/p/d quartet (max d,d,d,d → gradient nroots = (2+1+2+2+2)/2+1 = 5) is OK.
    #[test]
    fn test_int2e_ip1_nroots_guard() {
        let (basis, shells, op) = build_ip1_plan_lll(3, 3, 3, 3);
        let result = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "all-f int2e_ip1 quartet (nroots=7) must return UnsupportedApi, got: {:?}",
            result.map(|(s, _)| s.len())
        );

        // (d,d,d,d) gradient nroots = (2+1+2+2+2)/2 + 1 = 4 + 1 = 5 ≤ 5 → allowed.
        let (basis, shells, op) = build_ip1_plan_lll(2, 2, 2, 2);
        let ok = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "(d,d,d,d) int2e_ip1 quartet (nroots=5) must be allowed, got: {:?}",
            ok.err()
        );
    }

    // Component count: (s,s,s,s) → 3 nonzero-capable outputs; (p,s,s,s) → 9 (sph p = 3).
    #[test]
    fn test_int2e_ip1_component_count() {
        let (basis, shells, op) = build_ip1_plan(0, Representation::Spheric);
        let (staging, _stats) = run_ip1(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(
            staging.len(),
            3,
            "(s,s,s,s) int2e_ip1 should produce 3 components, got {}",
            staging.len()
        );

        let (basis, shells, op) = build_ip1_plan_lll(1, 0, 0, 0);
        let (staging, _stats) = run_ip1(&basis, shells, op, Representation::Spheric).unwrap();
        // sph p = 3 AOs; 3 components × 3×1×1×1 = 9.
        assert_eq!(
            staging.len(),
            9,
            "(p,s,s,s) int2e_ip1 should produce 9 outputs, got {}",
            staging.len()
        );
    }

    // Determinism (D-10): repeated evaluation is bit-identical.
    #[test]
    fn test_int2e_ip1_determinism() {
        let (basis, shells, op) = build_ip1_plan_lll(1, 1, 0, 0);
        let (out1, _) = run_ip1(&basis, shells.clone(), op, Representation::Spheric).unwrap();
        let (out2, _) = run_ip1(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "int2e_ip1 output not bit-identical across two evaluations"
            );
        }
    }

    // Spinor (R5): int2e_ip1 with Representation::Spinor returns UnsupportedApi.
    #[test]
    fn test_int2e_ip1_spinor_unsupported() {
        // Build with sph for a valid workspace query, then force Spinor on the plan.
        let (basis, shells, op) = build_ip1_plan(0, Representation::Spheric);
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = cintx_core::PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 6];
        let result = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "spinor int2e_ip1 should return UnsupportedApi, got: {:?}",
            result
        );
    }
}
