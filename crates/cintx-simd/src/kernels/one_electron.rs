use crate::boys::{rys_root1_simd, rys_root2_simd};
use crate::kernels::recurrence::{hrr_1e_axis, kin_d2_axis, vrr_1e_axis, vrr_nuc_axis};
use crate::vector::SimdFloat;
use std::f64::consts::PI;

pub const SQRTPI: f64 = 1.7724538509055160272981674833411451_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Ensures exact libcint result compatibility.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64, // 1 / (2 * sqrt(pi))
        1 => 0.488602511902919921_f64, // sqrt(3 / (4 * pi))
        _ => 1.0,
    }
}

/// Compute number of Cartesian basis functions for angular momentum `l`.
#[inline(always)]
pub const fn ncart(l: u8) -> usize {
    ((l as usize) + 1) * ((l as usize) + 2) / 2
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Follows libcint `CINTcart_comp` ordering:
/// for lx = l..=0, for ly = l-lx..=0, lz = l - lx - ly.
pub fn cart_comps(l: u8) -> Vec<(u8, u8, u8)> {
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

/// Description of an atomic center with nuclear charge and Cartesian coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomCoord {
    pub charge: f64,
    pub coord: [f64; 3],
}

/// Evaluation configuration and inputs for 1-electron SIMD integral kernels.
#[derive(Clone, Debug)]
pub struct OneElectronInput<'a> {
    pub li: u8,
    pub lj: u8,
    pub ri: [f64; 3],
    pub rj: [f64; 3],
    pub exps_i: &'a [f64],
    pub exps_j: &'a [f64],
    pub coeff_i: &'a [f64],
    pub coeff_j: &'a [f64],
    pub atoms: &'a [AtomCoord],
}

/// SIMD-vectorized evaluator for 1-electron integrals (overlap, kinetic, nuclear).
pub struct SimdOneElectronKernel;

impl SimdOneElectronKernel {
    /// Evaluate 1-electron overlap integrals (`int1e_ovlp`) with SIMD vectorization.
    ///
    /// Writes output into `out` in Cartesian component order (cj outer, ci inner).
    pub fn eval_ovlp<V: SimdFloat>(input: &OneElectronInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lj = input.lj as usize;
        let nci = ncart(input.li);
        let _ncj = ncart(input.lj);

        out.fill(0.0);

        let nmax = li + lj;
        let dj = nmax + 1;
        let g_per_axis = (nmax + 1) * (lj + 1);
        let total_g = 3 * g_per_axis;
        let gx = 0;
        let gy = g_per_axis;
        let gz = 2 * g_per_axis;

        let mut g = vec![V::splat(V::Scalar::default()); total_g];

        let rirjx_val = input.ri[0] - input.rj[0];
        let rirjy_val = input.ri[1] - input.rj[1];
        let rirjz_val = input.ri[2] - input.rj[2];
        let rr_val = rirjx_val * rirjx_val + rirjy_val * rirjy_val + rirjz_val * rirjz_val;

        let nprim_i = input.exps_i.len();
        let nprim_j = input.exps_j.len();

        let norm_fac = common_fac_sp(input.li) * common_fac_sp(input.lj);

        // Process primitive pairs with SIMD lane chunking
        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            let mut pj = 0;
            while pj < nprim_j {
                let chunk_size = (nprim_j - pj).min(V::LANES);

                let mut aj_arr = [1.0; 8];
                let mut weight_arr = [0.0; 8];
                for lane in 0..chunk_size {
                    let aj_v = input.exps_j[pj + lane];
                    let cj_v = input.coeff_j[pj + lane];
                    aj_arr[lane] = aj_v;
                    weight_arr[lane] = ci_val * cj_v;
                }

                let ai = V::from_f64(ai_val);
                let aj = V::from_f64_slice(&aj_arr[..V::LANES], 1.0);
                let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                let zeta = ai + aj;
                let aij2 = V::from_f64(0.5) / zeta;
                let rirjx = V::from_f64(rirjx_val);
                let rirjy = V::from_f64(rirjy_val);
                let rirjz = V::from_f64(rirjz_val);
                let rr = V::from_f64(rr_val);

                let fac = (-ai * aj / zeta * rr).exp();
                let px = (ai * V::from_f64(input.ri[0]) + aj * V::from_f64(input.rj[0])) / zeta;
                let py = (ai * V::from_f64(input.ri[1]) + aj * V::from_f64(input.rj[1])) / zeta;
                let pz = (ai * V::from_f64(input.ri[2]) + aj * V::from_f64(input.rj[2])) / zeta;

                let pi_const = V::from_f64(PI);
                let sqrtpi = V::from_f64(SQRTPI);
                let base_factor = fac * sqrtpi * pi_const / (zeta * zeta.sqrt());

                // Zero G-tensor
                g.fill(V::splat(V::Scalar::default()));
                g[gx] = V::from_f64(1.0);
                g[gy] = V::from_f64(1.0);
                g[gz] = base_factor;

                // Overlap VRR
                vrr_1e_axis(&mut g, gx, px - V::from_f64(input.ri[0]), aij2, nmax);
                vrr_1e_axis(&mut g, gy, py - V::from_f64(input.ri[1]), aij2, nmax);
                vrr_1e_axis(&mut g, gz, pz - V::from_f64(input.ri[2]), aij2, nmax);

                // Overlap HRR
                if lj >= 1 {
                    hrr_1e_axis(&mut g, gx, rirjx, dj, nmax, lj);
                    hrr_1e_axis(&mut g, gy, rirjy, dj, nmax, lj);
                    hrr_1e_axis(&mut g, gz, rirjz, dj, nmax, lj);
                }

                // Contract Cartesian components
                let mut cj_idx = 0;
                for ja in 0..=lj {
                    let jx = lj - ja;
                    let lj_minus_jx = lj - jx;
                    for jb in 0..=lj_minus_jx {
                        let jy = lj_minus_jx - jb;
                        let jz = lj - jx - jy;

                        let mut ci_idx = 0;
                        for ia in 0..=li {
                            let ix = li - ia;
                            let li_minus_ix = li - ix;
                            for ib in 0..=li_minus_ix {
                                let iy = li_minus_ix - ib;
                                let iz = li - ix - iy;

                                let vx = g[gx + jx * dj + ix];
                                let vy = g[gy + jy * dj + iy];
                                let vz = g[gz + jz * dj + iz];
                                let val = vx * vy * vz;

                                let term = (weight * val).reduce_add();
                                let term_f64: f64 = term.into();
                                out[cj_idx * nci + ci_idx] += term_f64 * norm_fac;

                                ci_idx += 1;
                            }
                        }
                        cj_idx += 1;
                    }
                }

                pj += chunk_size;
            }
        }
    }

    /// Evaluate 1-electron kinetic energy integrals (`int1e_kin`) with SIMD vectorization.
    pub fn eval_kin<V: SimdFloat>(input: &OneElectronInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lj = input.lj as usize;
        let nci = ncart(input.li);
        let _ncj = ncart(input.lj);

        out.fill(0.0);

        let lj_ext = lj + 2;
        let nmax = li + lj + 2;
        let dj = nmax + 1;
        let g_per_axis = (nmax + 1) * (lj_ext + 1);
        let total_g = 3 * g_per_axis;
        let gx = 0;
        let gy = g_per_axis;
        let gz = 2 * g_per_axis;

        let mut g = vec![V::splat(V::Scalar::default()); total_g];

        let rirjx_val = input.ri[0] - input.rj[0];
        let rirjy_val = input.ri[1] - input.rj[1];
        let rirjz_val = input.ri[2] - input.rj[2];
        let rr_val = rirjx_val * rirjx_val + rirjy_val * rirjy_val + rirjz_val * rirjz_val;

        let nprim_i = input.exps_i.len();
        let nprim_j = input.exps_j.len();
        let norm_fac = common_fac_sp(input.li) * common_fac_sp(input.lj);

        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            let mut pj = 0;
            while pj < nprim_j {
                let chunk_size = (nprim_j - pj).min(V::LANES);

                let mut aj_arr = [1.0; 8];
                let mut weight_arr = [0.0; 8];
                for lane in 0..chunk_size {
                    let aj_v = input.exps_j[pj + lane];
                    let cj_v = input.coeff_j[pj + lane];
                    aj_arr[lane] = aj_v;
                    weight_arr[lane] = ci_val * cj_v;
                }

                let ai = V::from_f64(ai_val);
                let aj = V::from_f64_slice(&aj_arr[..V::LANES], 1.0);
                let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                let zeta = ai + aj;
                let aij2 = V::from_f64(0.5) / zeta;
                let rirjx = V::from_f64(rirjx_val);
                let rirjy = V::from_f64(rirjy_val);
                let rirjz = V::from_f64(rirjz_val);
                let rr = V::from_f64(rr_val);

                let fac = (-ai * aj / zeta * rr).exp();
                let px = (ai * V::from_f64(input.ri[0]) + aj * V::from_f64(input.rj[0])) / zeta;
                let py = (ai * V::from_f64(input.ri[1]) + aj * V::from_f64(input.rj[1])) / zeta;
                let pz = (ai * V::from_f64(input.ri[2]) + aj * V::from_f64(input.rj[2])) / zeta;

                let pi_const = V::from_f64(PI);
                let sqrtpi = V::from_f64(SQRTPI);
                let base_factor = fac * sqrtpi * pi_const / (zeta * zeta.sqrt());

                g.fill(V::splat(V::Scalar::default()));
                g[gx] = V::from_f64(1.0);
                g[gy] = V::from_f64(1.0);
                g[gz] = base_factor;

                vrr_1e_axis(&mut g, gx, px - V::from_f64(input.ri[0]), aij2, nmax);
                vrr_1e_axis(&mut g, gy, py - V::from_f64(input.ri[1]), aij2, nmax);
                vrr_1e_axis(&mut g, gz, pz - V::from_f64(input.ri[2]), aij2, nmax);

                if lj_ext >= 1 {
                    hrr_1e_axis(&mut g, gx, rirjx, dj, nmax, lj_ext);
                    hrr_1e_axis(&mut g, gy, rirjy, dj, nmax, lj_ext);
                    hrr_1e_axis(&mut g, gz, rirjz, dj, nmax, lj_ext);
                }

                let mut cj_idx = 0;
                for ja in 0..=lj {
                    let jx = lj - ja;
                    let lj_minus_jx = lj - jx;
                    for jb in 0..=lj_minus_jx {
                        let jy = lj_minus_jx - jb;
                        let jz = lj - jx - jy;

                        let mut ci_idx = 0;
                        for ia in 0..=li {
                            let ix = li - ia;
                            let li_minus_ix = li - ix;
                            for ib in 0..=li_minus_ix {
                                let iy = li_minus_ix - ib;
                                let iz = li - ix - iy;

                                let nx = jx * dj + ix;
                                let ny = jy * dj + iy;
                                let nz = jz * dj + iz;

                                let vx0 = g[gx + nx];
                                let vy0 = g[gy + ny];
                                let vz0 = g[gz + nz];

                                let g3x = kin_d2_axis(&g, gx, nx, dj, jx, aj);
                                let g3y = kin_d2_axis(&g, gy, ny, dj, jy, aj);
                                let g3z = kin_d2_axis(&g, gz, nz, dj, jz, aj);

                                let val = V::from_f64(-0.5)
                                    * (g3x * vy0 * vz0 + vx0 * g3y * vz0 + vx0 * vy0 * g3z);

                                let term = (weight * val).reduce_add();
                                let term_f64: f64 = term.into();
                                out[cj_idx * nci + ci_idx] += term_f64 * norm_fac;

                                ci_idx += 1;
                            }
                        }
                        cj_idx += 1;
                    }
                }

                pj += chunk_size;
            }
        }
    }

    /// Evaluate 1-electron nuclear attraction integrals (`int1e_nuc`) with SIMD vectorization.
    pub fn eval_nuc<V: SimdFloat>(input: &OneElectronInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lj = input.lj as usize;
        let nci = ncart(input.li);
        let _ncj = ncart(input.lj);

        out.fill(0.0);

        let nmax = li + lj;
        let dj = nmax + 1;
        let g_per_axis = (nmax + 1) * (lj + 1);
        let total_g = 3 * g_per_axis;
        let gx = 0;
        let gy = g_per_axis;
        let gz = 2 * g_per_axis;

        let mut g = vec![V::splat(V::Scalar::default()); total_g];

        let rirjx_val = input.ri[0] - input.rj[0];
        let rirjy_val = input.ri[1] - input.rj[1];
        let rirjz_val = input.ri[2] - input.rj[2];
        let rr_val = rirjx_val * rirjx_val + rirjy_val * rirjy_val + rirjz_val * rirjz_val;

        let nroots = (li + lj) / 2 + 1;
        let norm_fac = common_fac_sp(input.li) * common_fac_sp(input.lj);

        let nprim_i = input.exps_i.len();
        let nprim_j = input.exps_j.len();

        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            let mut pj = 0;
            while pj < nprim_j {
                let chunk_size = (nprim_j - pj).min(V::LANES);

                let mut aj_arr = [1.0; 8];
                let mut weight_arr = [0.0; 8];
                for lane in 0..chunk_size {
                    let aj_v = input.exps_j[pj + lane];
                    let cj_v = input.coeff_j[pj + lane];
                    aj_arr[lane] = aj_v;
                    weight_arr[lane] = ci_val * cj_v;
                }

                let ai = V::from_f64(ai_val);
                let aj = V::from_f64_slice(&aj_arr[..V::LANES], 1.0);
                let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                let zeta = ai + aj;
                let aij2 = V::from_f64(0.5) / zeta;
                let rirjx = V::from_f64(rirjx_val);
                let rirjy = V::from_f64(rirjy_val);
                let rirjz = V::from_f64(rirjz_val);
                let rr = V::from_f64(rr_val);

                let fac = (-ai * aj / zeta * rr).exp();
                let px = (ai * V::from_f64(input.ri[0]) + aj * V::from_f64(input.rj[0])) / zeta;
                let py = (ai * V::from_f64(input.ri[1]) + aj * V::from_f64(input.rj[1])) / zeta;
                let pz = (ai * V::from_f64(input.ri[2]) + aj * V::from_f64(input.rj[2])) / zeta;

                let pi_const = V::from_f64(PI);

                // Accumulate over nuclear centers
                for atom in input.atoms {
                    let z_c = V::from_f64(atom.charge);
                    let rcx = V::from_f64(atom.coord[0]);
                    let rcy = V::from_f64(atom.coord[1]);
                    let rcz = V::from_f64(atom.coord[2]);

                    let crijx = rcx - px;
                    let crijy = rcy - py;
                    let crijz = rcz - pz;
                    let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                    let fac1 = V::from_f64(2.0) * pi_const * (V::from_f64(0.0) - z_c) * fac / zeta;

                    // Evaluate Rys quadrature roots (1 or 2 roots supported in SIMD kernel)
                    if nroots == 1 {
                        let (u_n, w_n) = rys_root1_simd(x_boys);
                        let tau = u_n / (V::from_f64(1.0) + u_n);
                        let rt = aij2 * (V::from_f64(1.0) - tau);

                        let c00x = (px - V::from_f64(input.ri[0])) + tau * crijx;
                        let c00y = (py - V::from_f64(input.ri[1])) + tau * crijy;
                        let c00z = (pz - V::from_f64(input.ri[2])) + tau * crijz;

                        g.fill(V::splat(V::Scalar::default()));
                        g[gx] = V::from_f64(1.0);
                        g[gy] = V::from_f64(1.0);
                        g[gz] = fac1 * w_n;

                        vrr_nuc_axis(&mut g, gx, c00x, rt, nmax);
                        vrr_nuc_axis(&mut g, gy, c00y, rt, nmax);
                        vrr_nuc_axis(&mut g, gz, c00z, rt, nmax);

                        if lj >= 1 {
                            hrr_1e_axis(&mut g, gx, rirjx, dj, nmax, lj);
                            hrr_1e_axis(&mut g, gy, rirjy, dj, nmax, lj);
                            hrr_1e_axis(&mut g, gz, rirjz, dj, nmax, lj);
                        }

                        let mut cj_idx = 0;
                        for ja in 0..=lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            for jb in 0..=lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

                                let mut ci_idx = 0;
                                for ia in 0..=li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    for ib in 0..=li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        let vx = g[gx + jx * dj + ix];
                                        let vy = g[gy + jy * dj + iy];
                                        let vz = g[gz + jz * dj + iz];
                                        let val = vx * vy * vz;

                                        let term = (weight * val).reduce_add();
                                        let term_f64: f64 = term.into();
                                        out[cj_idx * nci + ci_idx] += term_f64 * norm_fac;

                                        ci_idx += 1;
                                    }
                                }
                                cj_idx += 1;
                            }
                        }
                    } else if nroots == 2 {
                        let ([u0, u1], [w0, w1]) = rys_root2_simd(x_boys);
                        let roots_w = [(u0, w0), (u1, w1)];

                        for (u_n, w_n) in roots_w {
                            let tau = u_n / (V::from_f64(1.0) + u_n);
                            let rt = aij2 * (V::from_f64(1.0) - tau);

                            let c00x = (px - V::from_f64(input.ri[0])) + tau * crijx;
                            let c00y = (py - V::from_f64(input.ri[1])) + tau * crijy;
                            let c00z = (pz - V::from_f64(input.ri[2])) + tau * crijz;

                            g.fill(V::splat(V::Scalar::default()));
                            g[gx] = V::from_f64(1.0);
                            g[gy] = V::from_f64(1.0);
                            g[gz] = fac1 * w_n;

                            vrr_nuc_axis(&mut g, gx, c00x, rt, nmax);
                            vrr_nuc_axis(&mut g, gy, c00y, rt, nmax);
                            vrr_nuc_axis(&mut g, gz, c00z, rt, nmax);

                            if lj >= 1 {
                                hrr_1e_axis(&mut g, gx, rirjx, dj, nmax, lj);
                                hrr_1e_axis(&mut g, gy, rirjy, dj, nmax, lj);
                                hrr_1e_axis(&mut g, gz, rirjz, dj, nmax, lj);
                            }

                            let mut cj_idx = 0;
                            for ja in 0..=lj {
                                let jx = lj - ja;
                                let lj_minus_jx = lj - jx;
                                for jb in 0..=lj_minus_jx {
                                    let jy = lj_minus_jx - jb;
                                    let jz = lj - jx - jy;

                                    let mut ci_idx = 0;
                                    for ia in 0..=li {
                                        let ix = li - ia;
                                        let li_minus_ix = li - ix;
                                        for ib in 0..=li_minus_ix {
                                            let iy = li_minus_ix - ib;
                                            let iz = li - ix - iy;

                                            let vx = g[gx + jx * dj + ix];
                                            let vy = g[gy + jy * dj + iy];
                                            let vz = g[gz + jz * dj + iz];
                                            let val = vx * vy * vz;

                                            let term = (weight * val).reduce_add();
                                            let term_f64: f64 = term.into();
                                            out[cj_idx * nci + ci_idx] += term_f64 * norm_fac;

                                            ci_idx += 1;
                                        }
                                    }
                                    cj_idx += 1;
                                }
                            }
                        }
                    }
                }

                pj += chunk_size;
            }
        }
    }
}
