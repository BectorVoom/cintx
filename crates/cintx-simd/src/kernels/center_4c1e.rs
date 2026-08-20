use crate::kernels::one_electron::{common_fac_sp, ncart};
use crate::vector::SimdFloat;
use std::f64::consts::PI;

/// Input parameters for 4-center 1-electron overlap integral evaluation (`(ij|kl)_{1e}`).
#[derive(Clone, Debug)]
pub struct Center4c1eInput<'a> {
    pub li: u8,
    pub lj: u8,
    pub lk: u8,
    pub ll: u8,
    pub ri: [f64; 3],
    pub rj: [f64; 3],
    pub rk: [f64; 3],
    pub rl: [f64; 3],
    pub exps_i: &'a [f64],
    pub exps_j: &'a [f64],
    pub exps_k: &'a [f64],
    pub exps_l: &'a [f64],
    pub coeff_i: &'a [f64],
    pub coeff_j: &'a [f64],
    pub coeff_k: &'a [f64],
    pub coeff_l: &'a [f64],
}

/// SIMD-vectorized 4-center 1-electron overlap integral kernel (`(ij|kl)_{1e}`).
pub struct SimdCenter4c1eKernel;

impl SimdCenter4c1eKernel {
    /// Evaluate 4-center 1-electron overlap integrals (`(ij|kl)_{1e}`).
    ///
    /// Writes results into `out` in Cartesian component ordering (`l` outer, `k` middle-2, `j` middle-1, `i` inner).
    pub fn eval<V: SimdFloat>(input: &Center4c1eInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lj = input.lj as usize;
        let lk = input.lk as usize;
        let ll = input.ll as usize;

        let nfi = ncart(input.li);
        let nfj = ncart(input.lj);
        let nfk = ncart(input.lk);
        let nfl = ncart(input.ll);
        let total_comps = nfi * nfj * nfk * nfl;

        out[..total_comps].fill(0.0);

        let l_tot = li + lj + lk + ll;
        let ljkl = lj + lk + ll;
        let lkl = lk + ll;

        let norm_fac = common_fac_sp(input.li)
            * common_fac_sp(input.lj)
            * common_fac_sp(input.lk)
            * common_fac_sp(input.ll);

        let rirjx_val = input.ri[0] - input.rj[0];
        let rirjy_val = input.ri[1] - input.rj[1];
        let rirjz_val = input.ri[2] - input.rj[2];

        let rjrkx_val = input.rj[0] - input.rk[0];
        let rjrky_val = input.rj[1] - input.rk[1];
        let rjrkz_val = input.rj[2] - input.rk[2];

        let rkrlx_val = input.rk[0] - input.rl[0];
        let rkrly_val = input.rk[1] - input.rl[1];
        let rkrlz_val = input.rk[2] - input.rl[2];

        let di = 1;
        let dj = li + 1;
        let dk = (li + 1) * (lj + 1);
        let dl = (li + 1) * (lj + 1) * (lk + 1);
        let hrr_size = (li + 1) * (lj + 1) * (lk + 1) * (ll + 1);

        let mut g1d_x = vec![V::splat(V::Scalar::default()); l_tot + 1];
        let mut g1d_y = vec![V::splat(V::Scalar::default()); l_tot + 1];
        let mut g1d_z = vec![V::splat(V::Scalar::default()); l_tot + 1];

        let mut g_i_jkl_x = vec![V::splat(V::Scalar::default()); (li + 1) * (ljkl + 1)];
        let mut g_i_jkl_y = vec![V::splat(V::Scalar::default()); (li + 1) * (ljkl + 1)];
        let mut g_i_jkl_z = vec![V::splat(V::Scalar::default()); (li + 1) * (ljkl + 1)];

        let mut g_ij_kl_x = vec![V::splat(V::Scalar::default()); (li + 1) * (lj + 1) * (lkl + 1)];
        let mut g_ij_kl_y = vec![V::splat(V::Scalar::default()); (li + 1) * (lj + 1) * (lkl + 1)];
        let mut g_ij_kl_z = vec![V::splat(V::Scalar::default()); (li + 1) * (lj + 1) * (lkl + 1)];

        let mut g_ijkl = vec![V::splat(V::Scalar::default()); 3 * hrr_size];

        let nprim_i = input.exps_i.len();
        let nprim_j = input.exps_j.len();
        let nprim_k = input.exps_k.len();
        let nprim_l = input.exps_l.len();

        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            for pj in 0..nprim_j {
                let aj_val = input.exps_j[pj];
                let cj_val = input.coeff_j[pj];

                let dij = dist_sq(input.ri, input.rj);

                for pk in 0..nprim_k {
                    let ak_val = input.exps_k[pk];
                    let ck_val = input.coeff_k[pk];

                    let dik = dist_sq(input.ri, input.rk);
                    let djk = dist_sq(input.rj, input.rk);

                    let mut pl = 0;
                    while pl < nprim_l {
                        let chunk_size = (nprim_l - pl).min(V::LANES);

                        let mut al_arr = [1.0; 8];
                        let mut weight_arr = [0.0; 8];
                        for lane in 0..chunk_size {
                            al_arr[lane] = input.exps_l[pl + lane];
                            weight_arr[lane] =
                                ci_val * cj_val * ck_val * input.coeff_l[pl + lane] * norm_fac;
                        }

                        let ai = V::from_f64(ai_val);
                        let aj = V::from_f64(aj_val);
                        let ak = V::from_f64(ak_val);
                        let al = V::from_f64_slice(&al_arr[..V::LANES], 1.0);
                        let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                        let rl_x = V::from_f64(input.rl[0]);
                        let rl_y = V::from_f64(input.rl[1]);
                        let rl_z = V::from_f64(input.rl[2]);

                        let dil = (V::from_f64(input.ri[0]) - rl_x)
                            * (V::from_f64(input.ri[0]) - rl_x)
                            + (V::from_f64(input.ri[1]) - rl_y) * (V::from_f64(input.ri[1]) - rl_y)
                            + (V::from_f64(input.ri[2]) - rl_z) * (V::from_f64(input.ri[2]) - rl_z);

                        let djl = (V::from_f64(input.rj[0]) - rl_x)
                            * (V::from_f64(input.rj[0]) - rl_x)
                            + (V::from_f64(input.rj[1]) - rl_y) * (V::from_f64(input.rj[1]) - rl_y)
                            + (V::from_f64(input.rj[2]) - rl_z) * (V::from_f64(input.rj[2]) - rl_z);

                        let dkl = (V::from_f64(input.rk[0]) - rl_x)
                            * (V::from_f64(input.rk[0]) - rl_x)
                            + (V::from_f64(input.rk[1]) - rl_y) * (V::from_f64(input.rk[1]) - rl_y)
                            + (V::from_f64(input.rk[2]) - rl_z) * (V::from_f64(input.rk[2]) - rl_z);

                        let zeta = ai + aj + ak + al;
                        let wx = (ai * V::from_f64(input.ri[0])
                            + aj * V::from_f64(input.rj[0])
                            + ak * V::from_f64(input.rk[0])
                            + al * rl_x)
                            / zeta;
                        let wy = (ai * V::from_f64(input.ri[1])
                            + aj * V::from_f64(input.rj[1])
                            + ak * V::from_f64(input.rk[1])
                            + al * rl_y)
                            / zeta;
                        let wz = (ai * V::from_f64(input.ri[2])
                            + aj * V::from_f64(input.rj[2])
                            + ak * V::from_f64(input.rk[2])
                            + al * rl_z)
                            / zeta;

                        let exp_sum = ai * aj * V::from_f64(dij)
                            + ai * ak * V::from_f64(dik)
                            + ai * al * dil
                            + aj * ak * V::from_f64(djk)
                            + aj * al * djl
                            + ak * al * dkl;
                        let s0 = (V::from_f64(PI) / zeta).pow(V::from_f64(1.5))
                            * (-exp_sum / zeta).exp();

                        // 1D VRR
                        let wi_x = wx - V::from_f64(input.ri[0]);
                        let wi_y = wy - V::from_f64(input.ri[1]);
                        let wi_z = wz - V::from_f64(input.ri[2]);

                        g1d_x[0] = V::from_f64(1.0);
                        g1d_y[0] = V::from_f64(1.0);
                        g1d_z[0] = s0;

                        if l_tot >= 1 {
                            g1d_x[1] = wi_x;
                            g1d_y[1] = wi_y;
                            g1d_z[1] = wi_z * s0;

                            for n in 1..l_tot {
                                let fac = V::from_f64(0.5 * n as f64) / zeta;
                                g1d_x[n + 1] = wi_x * g1d_x[n] + fac * g1d_x[n - 1];
                                g1d_y[n + 1] = wi_y * g1d_y[n] + fac * g1d_y[n - 1];
                                g1d_z[n + 1] = wi_z * g1d_z[n] + fac * g1d_z[n - 1];
                            }
                        }

                        // HRR 1: split i and (j+k+l)
                        let rirj_x = V::from_f64(rirjx_val);
                        let rirj_y = V::from_f64(rirjy_val);
                        let rirj_z = V::from_f64(rirjz_val);

                        for jkl in 0..=ljkl {
                            for i in 0..=(l_tot - jkl) {
                                if i <= li {
                                    let out_idx = jkl * (li + 1) + i;
                                    if jkl == 0 {
                                        g_i_jkl_x[out_idx] = g1d_x[i];
                                        g_i_jkl_y[out_idx] = g1d_y[i];
                                        g_i_jkl_z[out_idx] = g1d_z[i];
                                    } else {
                                        let prev_idx = (jkl - 1) * (li + 1) + i;
                                        let prev_next_i = (jkl - 1) * (li + 1) + (i + 1);
                                        g_i_jkl_x[out_idx] =
                                            g_i_jkl_x[prev_next_i] + rirj_x * g_i_jkl_x[prev_idx];
                                        g_i_jkl_y[out_idx] =
                                            g_i_jkl_y[prev_next_i] + rirj_y * g_i_jkl_y[prev_idx];
                                        g_i_jkl_z[out_idx] =
                                            g_i_jkl_z[prev_next_i] + rirj_z * g_i_jkl_z[prev_idx];
                                    }
                                }
                            }
                        }

                        // HRR 2: split j and (k+l)
                        let rjrk_x = V::from_f64(rjrkx_val);
                        let rjrk_y = V::from_f64(rjrky_val);
                        let rjrk_z = V::from_f64(rjrkz_val);

                        for kl in 0..=lkl {
                            for j in 0..=(ljkl - kl) {
                                if j <= lj {
                                    for i in 0..=li {
                                        let out_idx = (kl * (lj + 1) + j) * (li + 1) + i;
                                        if kl == 0 {
                                            g_ij_kl_x[out_idx] = g_i_jkl_x[j * (li + 1) + i];
                                            g_ij_kl_y[out_idx] = g_i_jkl_y[j * (li + 1) + i];
                                            g_ij_kl_z[out_idx] = g_i_jkl_z[j * (li + 1) + i];
                                        } else {
                                            let prev_idx = ((kl - 1) * (lj + 1) + j) * (li + 1) + i;
                                            let prev_next_j =
                                                ((kl - 1) * (lj + 1) + (j + 1)) * (li + 1) + i;
                                            g_ij_kl_x[out_idx] = g_ij_kl_x[prev_next_j]
                                                + rjrk_x * g_ij_kl_x[prev_idx];
                                            g_ij_kl_y[out_idx] = g_ij_kl_y[prev_next_j]
                                                + rjrk_y * g_ij_kl_y[prev_idx];
                                            g_ij_kl_z[out_idx] = g_ij_kl_z[prev_next_j]
                                                + rjrk_z * g_ij_kl_z[prev_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // HRR 3: split k and l
                        let rkrl_x = V::from_f64(rkrlx_val);
                        let rkrl_y = V::from_f64(rkrly_val);
                        let rkrl_z = V::from_f64(rkrlz_val);

                        for l in 0..=ll {
                            for k in 0..=(lkl - l) {
                                if k <= lk {
                                    for j in 0..=lj {
                                        for i in 0..=li {
                                            let out_idx = l * dl + k * dk + j * dj + i * di;
                                            if l == 0 {
                                                g_ijkl[out_idx] =
                                                    g_ij_kl_x[(k * (lj + 1) + j) * (li + 1) + i];
                                                g_ijkl[hrr_size + out_idx] =
                                                    g_ij_kl_y[(k * (lj + 1) + j) * (li + 1) + i];
                                                g_ijkl[2 * hrr_size + out_idx] =
                                                    g_ij_kl_z[(k * (lj + 1) + j) * (li + 1) + i];
                                            } else {
                                                let prev_idx =
                                                    (l - 1) * dl + k * dk + j * dj + i * di;
                                                let prev_next_k =
                                                    (l - 1) * dl + (k + 1) * dk + j * dj + i * di;
                                                g_ijkl[out_idx] =
                                                    g_ijkl[prev_next_k] + rkrl_x * g_ijkl[prev_idx];
                                                g_ijkl[hrr_size + out_idx] = g_ijkl
                                                    [hrr_size + prev_next_k]
                                                    + rkrl_y * g_ijkl[hrr_size + prev_idx];
                                                g_ijkl[2 * hrr_size + out_idx] = g_ijkl
                                                    [2 * hrr_size + prev_next_k]
                                                    + rkrl_z * g_ijkl[2 * hrr_size + prev_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Cartesian Contraction
                        let mut comp_idx = 0;
                        for lx in (0..=ll).rev() {
                            for ly in (0..=(ll - lx)).rev() {
                                let lz = ll - lx - ly;

                                for kx in (0..=lk).rev() {
                                    for ky in (0..=(lk - kx)).rev() {
                                        let kz = lk - kx - ky;

                                        for jx in (0..=lj).rev() {
                                            for jy in (0..=(lj - jx)).rev() {
                                                let jz = lj - jx - jy;

                                                for ix in (0..=li).rev() {
                                                    for iy in (0..=(li - ix)).rev() {
                                                        let iz = li - ix - iy;

                                                        let idx_x =
                                                            lx * dl + kx * dk + jx * dj + ix * di;
                                                        let idx_y = hrr_size
                                                            + ly * dl
                                                            + ky * dk
                                                            + jy * dj
                                                            + iy * di;
                                                        let idx_z = 2 * hrr_size
                                                            + lz * dl
                                                            + kz * dk
                                                            + jz * dj
                                                            + iz * di;

                                                        let val = g_ijkl[idx_x]
                                                            * g_ijkl[idx_y]
                                                            * g_ijkl[idx_z]
                                                            * weight;
                                                        out[comp_idx] += val.reduce_add().into();
                                                        comp_idx += 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        pl += chunk_size;
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}
