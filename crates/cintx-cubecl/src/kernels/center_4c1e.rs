//! Host-side 4c1e (four-center one-electron overlap) integral kernel.
//!
//! Implements the libcint `g4c1e.c` polynomial 1D recurrence pipeline:
//! 1. Polynomial recurrence fill (NOT Rys quadrature — nroots=1 always).
//! 2. Branch-specific 4D HRR transfer reusing two_electron.rs functions.
//! 3. Cartesian contraction + optional `cart_to_sph_2e` transform.
//!
//! # Algorithm difference from 2e
//! The 2e kernel uses Rys quadrature roots to fill the G-tensor.
//! The 4c1e kernel uses a polynomial 1D recurrence from g4c1e.c:
//!   `buf[0] = 1`, `buf[1] = -r1r12 * buf[0]`,
//!   `buf[i+1] = 0.5*i/aijkl * buf[i-1] - r1r12 * buf[i]`
//! with `nroots = 1` hardcoded (no quadrature roots needed).

use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, ncart, nsph};
use cintx_core::{Representation, cintxRsError};
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

fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("outside Validated4C1E ({reason})"),
    }
}

/// G-tensor layout metadata for 4c1e.
///
/// Reuses the same stride/layout as TwoEShape from two_electron.rs, but
/// nroots is always 1 (no Rys quadrature) and g_size is recomputed accordingly.
#[derive(Clone, Copy, Debug)]
struct Shape4c1e {
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
    g_size: usize,
}

/// Build layout metadata for 4c1e with nroots forced to 1.
///
/// Follows the same stride logic as `build_2e_shape` from two_electron.rs,
/// but overrides nroots=1 since 4c1e uses polynomial recurrence, not Rys quadrature.
fn build_4c1e_shape(li: usize, lj: usize, lk: usize, ll: usize) -> Shape4c1e {
    let nroots = 1usize; // 4c1e always uses nroots=1 (polynomial recurrence)
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

    Shape4c1e {
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
        g_size,
    }
}

/// Fill G-tensor for one primitive quartet using polynomial 1D recurrence (g4c1e.c).
///
/// Key algorithm from g4c1e.c:
/// ```text
/// aijkl = aij + akl
/// fac factor applied only to z-axis initial value: buf[0] = fac / (aijkl * sqrt(aijkl))
/// x and y axes: buf[0] = 1.0
/// buf[1] = -r1r12 * buf[0]
/// buf[i+1] = 0.5 * i / aijkl * buf[i-1] - r1r12 * buf[i]   for i=1..nmax+mmax-1
/// 2D shift: buf[j*db + i] = buf[(j-1)*db + i+1] + r1r2 * buf[(j-1)*db + i]
/// ```
///
/// CRITICAL: The fac factor (which encodes the z-axis Gaussian prefactor) is ONLY applied
/// to the z-axis initial value. x and y axes start with buf[0] = 1.0. The final
/// 3-axis product during contraction forms the correct integral value.
///
/// `g` must be zeroed, size `3 * shape.g_size`, layout `[gx | gy | gz]` with nroots=1.
fn fill_4c1e_g_tensor(
    g: &mut [f64],
    shape: &Shape4c1e,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    rij: [f64; 3],
    rkl: [f64; 3],
    aij: f64,
    akl: f64,
    fac: f64,
) {
    let aijkl = aij + akl;
    let nmax = shape.nmax;
    let mmax = shape.mmax;

    for axis in 0..3usize {
        let off = axis * shape.g_size;

        // Determine base center per g4c1e.c ibase/kbase selection.
        // r1 is the "main" center, r2 is the "auxiliary" center.
        let (r1, r2) = if nmax >= mmax {
            // ij pair is larger: r1 from ij pair, r2 from kl pair
            let r1 = if shape.ibase { ri[axis] } else { rj[axis] };
            let r2 = if shape.kbase { rk[axis] } else { rl[axis] };
            (r1, r2)
        } else {
            // kl pair is larger: r1 from kl pair, r2 from ij pair
            let r1 = if shape.kbase { rk[axis] } else { rl[axis] };
            let r2 = if shape.ibase { ri[axis] } else { rj[axis] };
            (r1, r2)
        };

        // Weighted center P = (aij * Rij + akl * Rkl) / aijkl
        let rp = (aij * rij[axis] + akl * rkl[axis]) / aijkl;
        let r1r12 = r1 - rp; // displacement from r1 to weighted center

        // 1D polynomial recurrence scratch buffer, size db * (bigger + 1)
        let db = nmax + mmax + 1; // total polynomial degree + 1
        let bigger = nmax.max(mmax);
        let mut buf = vec![0.0f64; db * (bigger + 1)];

        // Initial value: z-axis gets the prefactor, x and y start at 1.
        buf[0] = if axis == 2 {
            fac / (aijkl * aijkl.sqrt())
        } else {
            1.0
        };

        // Recurrence: buf[i+1] = 0.5*i/aijkl * buf[i-1] - r1r12 * buf[i]
        if nmax + mmax > 0 {
            buf[1] = -r1r12 * buf[0];
        }
        for i in 1..(nmax + mmax) {
            buf[i + 1] = 0.5 * (i as f64) / aijkl * buf[i - 1] - r1r12 * buf[i];
        }

        // 2D shift fill: buf[j*db + i] for j > 0
        // This builds the "auxiliary" polynomial dimension.
        let r1r2 = r1 - r2;
        for j in 1..=bigger {
            for i in 0..db.saturating_sub(j) {
                buf[j * db + i] = buf[(j - 1) * db + i + 1] + r1r2 * buf[(j - 1) * db + i];
            }
        }

        // Map from polynomial buf to G-tensor.
        // For nmax >= mmax: ij pair is the "main" dim (i index maps to buf col),
        //   kl pair is auxiliary (j index maps to buf row).
        //   g[di*i + dk*k + dl*l] = buf[j_aux * db + i_main]
        // For nmax < mmax: kl pair is main, ij is auxiliary — swap roles.
        if nmax >= mmax {
            // i iterates main (0..=nmax), m iterates auxiliary (0..=mmax)
            for m in 0..=mmax {
                for n in 0..=nmax {
                    // In ibase=true: stride layout has di for i-direction (base),
                    // In ibase=false: dj for j-direction (base), but nroots=1 so di=1.
                    // With nroots=1, di=1, so the index stride is:
                    // g[off + n*di + m*dm] where dm is the kl-direction stride.
                    // The kl direction corresponds to dk (kbase) or dl (!kbase).
                    let g2d_ij = if shape.ibase { shape.di } else { shape.dj };
                    let g2d_kl = if shape.kbase { shape.dk } else { shape.dl };
                    let idx = off + n * g2d_ij + m * g2d_kl;
                    g[idx] = buf[m * db + n];
                }
            }
        } else {
            // kl is main, ij is auxiliary — swap n and m roles
            for n in 0..=nmax {
                for m in 0..=mmax {
                    let g2d_kl = if shape.kbase { shape.dk } else { shape.dl };
                    let g2d_ij = if shape.ibase { shape.di } else { shape.dj };
                    let idx = off + m * g2d_kl + n * g2d_ij;
                    g[idx] = buf[n * db + m];
                }
            }
        }
    }
}

/// HRR branch for `ibase=false && kbase=false`.
fn hrr_lj2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=false && kbase=true`.
fn hrr_kj2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=true && kbase=false`.
fn hrr_il2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=true && kbase=true`.
fn hrr_ik2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// Contract `[gx|gy|gz]` into Cartesian 4c1e tensor.
///
/// Output order: `out[i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk]` (i fastest, l slowest).
/// With nroots=1 the Rys root loop reduces to a single iteration at irys=0.
fn contract_4c1e_cart(g: &[f64], shape: &Shape4c1e, li: u8, lj: u8, lk: u8, ll: u8) -> Vec<f64> {
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

    // nroots=1: irys=0 is the only root
    let irys = 0usize;

    for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
        for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
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
                    let val = g[gx_off + x_idx] * g[gy_off + y_idx] * g[gz_off + z_idx];
                    let out_idx =
                        i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                    out[out_idx] += val;
                }
            }
        }
    }

    out
}

fn ensure_validated_4c1e(
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
) -> Result<(), cintxRsError> {
    // D-05: Spinor rejection FIRST — before feature gate or any other check.
    if matches!(plan.representation, Representation::Spinor) {
        return Err(validated_4c1e_error("spinor representation not supported for 4c1e"));
    }

    if specialization.canonical_family() != "4c1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_4c1e",
            detail: format!(
                "canonical_family mismatch for 4c1e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    if !matches!(
        plan.representation,
        Representation::Cart | Representation::Spheric
    ) {
        return Err(validated_4c1e_error("representation must be cart/sph"));
    }
    if !plan.descriptor.entry.component_rank.trim().is_empty()
        && plan.descriptor.entry.component_rank != "scalar"
    {
        return Err(validated_4c1e_error("component rank must be scalar"));
    }
    if plan
        .shells
        .as_slice()
        .iter()
        .any(|shell| shell.ang_momentum > 4)
    {
        return Err(validated_4c1e_error("max(l)>4"));
    }

    Ok(())
}

pub fn launch_center_4c1e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    ensure_validated_4c1e(plan, specialization)?;

    // Host-side implementation (no CubeCL GPU launch path).
    let _ = backend;

    let shells = plan.shells.as_slice();
    if shells.len() < 4 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_4c1e",
            detail: "4c1e kernel requires exactly 4 shells".to_owned(),
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

    let shape = build_4c1e_shape(li as usize, lj as usize, lk as usize, ll as usize);

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

    // Accumulated Cartesian integral buffer.
    let mut cart_buf = vec![0.0_f64; nfi * nfj * nfk * nfl];

    // Common factor from cint4c1e.c CINT4c1e_loop_nopt:
    //   common_fac = envs->common_factor * SQRTPI * M_PI * sp_factors
    // where envs->common_factor = 1 (set in CINTinit_int4c1e_EnvVars).
    // So the 4c1e common factor is: sqrt(pi) * pi * sp_factors.
    //
    // NOTE: This differs from the 2e formula (pi^3 * 2 / sqrt(pi)) — do NOT use
    // the 2e formula here. The 4c1e integral uses a different normalization.
    let sp_factor = common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk) * common_fac_sp(ll);
    let common_factor = SQRTPI * PI * sp_factor;

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
            let aij = ai + aj;

            // Pair weighted center for ij
            let rij = [
                (ai * ri[0] + aj * rj[0]) / aij,
                (ai * ri[1] + aj * rj[1]) / aij,
                (ai * ri[2] + aj * rj[2]) / aij,
            ];

            // Pair overlap prefactor: exp(-ai*aj/aij * |ri-rj|^2)
            let dx_ij = ri[0] - rj[0];
            let dy_ij = ri[1] - rj[1];
            let dz_ij = ri[2] - rj[2];
            let rr_ij = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
            let fac_ij = f64::exp(-ai * aj / aij * rr_ij);

            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let akl = ak + al;

                    // Pair weighted center for kl
                    let rkl = [
                        (ak * rk[0] + al * rl[0]) / akl,
                        (ak * rk[1] + al * rl[1]) / akl,
                        (ak * rk[2] + al * rl[2]) / akl,
                    ];

                    // Pair overlap prefactor: exp(-ak*al/akl * |rk-rl|^2)
                    let dx_kl = rk[0] - rl[0];
                    let dy_kl = rk[1] - rl[1];
                    let dz_kl = rk[2] - rl[2];
                    let rr_kl = dx_kl * dx_kl + dy_kl * dy_kl + dz_kl * dz_kl;
                    let fac_kl = f64::exp(-ak * al / akl * rr_kl);

                    // Cross-pair exponential: exp(-a0 * |rij - rkl|^2)
                    // where a0 = aij*akl/(aij+akl) is the reduced exponent.
                    // This matches libcint's eijkl = eij + ekl + a0*SQUARE(rij-rkl).
                    let aijkl = aij + akl;
                    let a0 = aij * akl / aijkl;
                    let dx_ijkl = rij[0] - rkl[0];
                    let dy_ijkl = rij[1] - rkl[1];
                    let dz_ijkl = rij[2] - rkl[2];
                    let rr_ijkl = dx_ijkl * dx_ijkl + dy_ijkl * dy_ijkl + dz_ijkl * dz_ijkl;
                    let fac_ijkl = f64::exp(-a0 * rr_ijkl);

                    // Quartet prefactor: common_factor * exp_ij * exp_kl * exp_ijkl_cross
                    let quartet_fac = common_factor * fac_ij * fac_kl * fac_ijkl;

                    // Fill G-tensor using polynomial recurrence (nroots=1).
                    let mut g = vec![0.0_f64; 3 * shape.g_size];
                    fill_4c1e_g_tensor(
                        &mut g,
                        &shape,
                        ri,
                        rj,
                        rk,
                        rl,
                        rij,
                        rkl,
                        aij,
                        akl,
                        quartet_fac,
                    );

                    // Apply HRR (same 4-branch selection as two_electron.rs).
                    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
                    let rkrl = [rk[0] - rl[0], rk[1] - rl[1], rk[2] - rl[2]];

                    if shape.kbase {
                        if shape.ibase {
                            hrr_ik2d_4d(&mut g, &shape, rirj, rkrl);
                        } else {
                            hrr_kj2d_4d(&mut g, &shape, rirj, rkrl);
                        }
                    } else if shape.ibase {
                        hrr_il2d_4d(&mut g, &shape, rirj, rkrl);
                    } else {
                        hrr_lj2d_4d(&mut g, &shape, rirj, rkrl);
                    }

                    // Contract G-tensor to Cartesian output.
                    let prim_cart = contract_4c1e_cart(&g, &shape, li, lj, lk, ll);

                    // Accumulate with contraction coefficients.
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

    // Apply cart-to-sph transform if representation is Spheric.
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_2e(&cart_buf, li, lj, lk, ll);
            let sph_size = nsi * nsj * nsk * nsl;
            let copy_len = staging.len().min(sph.len()).min(sph_size);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_4c1e_g_tensor_ssss() {
        // s-s-s-s: nmax=0, mmax=0, shape.g_size=1, nroots=1
        let shape = build_4c1e_shape(0, 0, 0, 0);
        assert_eq!(shape.nroots, 1);
        assert_eq!(shape.g_size, 1);

        let mut g = vec![0.0_f64; 3 * shape.g_size];
        let ri = [0.0_f64; 3];
        let rj = [0.0_f64; 3];
        let rk = [1.0_f64, 0.0, 0.0];
        let rl = [1.0_f64, 0.0, 0.0];
        let rij = [0.0_f64; 3];
        let rkl = [1.0_f64, 0.0, 0.0];
        let aij = 2.0;
        let akl = 2.0;
        let fac = 1.5_f64;

        fill_4c1e_g_tensor(&mut g, &shape, ri, rj, rk, rl, rij, rkl, aij, akl, fac);

        // gx[0]=1, gy[0]=1, gz[0] = fac / (aijkl * sqrt(aijkl))
        let aijkl = aij + akl;
        let expected_gz = fac / (aijkl * aijkl.sqrt());
        assert!((g[0] - 1.0).abs() < 1e-14, "gx[0] should be 1.0, got {}", g[0]);
        assert!((g[1] - 1.0).abs() < 1e-14, "gy[0] should be 1.0, got {}", g[1]);
        assert!(
            (g[2] - expected_gz).abs() < 1e-14,
            "gz[0] should be {expected_gz}, got {}",
            g[2]
        );
    }

    #[test]
    fn test_build_4c1e_shape_nroots_one() {
        // Verify nroots is always 1 regardless of angular momenta
        for li in 0..=2 {
            for lj in 0..=2 {
                for lk in 0..=2 {
                    for ll in 0..=2 {
                        let shape = build_4c1e_shape(li, lj, lk, ll);
                        assert_eq!(
                            shape.nroots, 1,
                            "nroots must be 1 for 4c1e, got {} for li={li} lj={lj} lk={lk} ll={ll}",
                            shape.nroots
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_spinor_rejected_first() {
        // Spinor check must return an error with "spinor" in the message.
        // We test the validated_4c1e_error path directly.
        let err = validated_4c1e_error("spinor representation not supported for 4c1e");
        match &err {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("spinor"),
                    "Error message must contain 'spinor', got: {requested}"
                );
            }
            _ => panic!("Expected UnsupportedApi error"),
        }
    }
}
