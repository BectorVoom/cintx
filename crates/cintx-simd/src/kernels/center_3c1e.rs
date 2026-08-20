use crate::kernels::one_electron::{SQRTPI, cart_comps, common_fac_sp, ncart};
use crate::vector::SimdFloat;
use std::f64::consts::PI;

/// Input parameters for 3-center 1-electron overlap integral evaluation (`(i|O_k|j)`).
#[derive(Clone, Debug)]
pub struct Center3c1eInput<'a> {
    pub li: u8,
    pub lj: u8,
    pub lk: u8,
    pub ri: [f64; 3],
    pub rj: [f64; 3],
    pub rk: [f64; 3],
    pub exps_i: &'a [f64],
    pub exps_j: &'a [f64],
    pub exps_k: &'a [f64],
    pub coeff_i: &'a [f64],
    pub coeff_j: &'a [f64],
    pub coeff_k: &'a [f64],
}

/// SIMD-vectorized 3-center 1-electron overlap integral kernel (`(i|O_k|j)`).
pub struct SimdCenter3c1eKernel;

impl SimdCenter3c1eKernel {
    /// Evaluate 3-center 1-electron overlap integrals (`(i|O_k|j)`).
    ///
    /// Writes results into `out` in Cartesian component ordering (`k` outer, `j` middle, `i` inner).
    pub fn eval<V: SimdFloat>(input: &Center3c1eInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lj = input.lj as usize;
        let lk = input.lk as usize;

        let nci = ncart(input.li);
        let ncj = ncart(input.lj);
        let nck = ncart(input.lk);
        let total_comps = nci * ncj * nck;

        out[..total_comps].fill(0.0);

        let dli = li + 1;
        let dlj = lj + lk + 1;
        let dlk = lk + 1;
        let vrr_nmax = li + lj + lk + 1;
        let g_alloc = (dli * dlj * dlk).max(dli * vrr_nmax);
        let total_g = 3 * g_alloc;

        let dj = dli;
        let dk = dli * dlj;
        let nmax = (li + lj + lk) as usize;
        let mmax = (lj + lk) as usize;

        let mut g = vec![V::splat(V::Scalar::default()); total_g];

        let common_factor_val = SQRTPI
            * PI
            * common_fac_sp(input.li)
            * common_fac_sp(input.lj)
            * common_fac_sp(input.lk);

        let xij = input.ri[0] - input.rj[0];
        let yij = input.ri[1] - input.rj[1];
        let zij = input.ri[2] - input.rj[2];
        let xik = input.ri[0] - input.rk[0];
        let yik = input.ri[1] - input.rk[1];
        let zik = input.ri[2] - input.rk[2];
        let xjk = input.rj[0] - input.rk[0];
        let yjk = input.rj[1] - input.rk[1];
        let zjk = input.rj[2] - input.rk[2];
        let rr_ij = xij * xij + yij * yij + zij * zij;
        let rr_ik = xik * xik + yik * yik + zik * zik;
        let rr_jk = xjk * xjk + yjk * yjk + zjk * zjk;

        let rirj = [xij, yij, zij];
        let rjrk = [xjk, yjk, zjk];

        let nprim_i = input.exps_i.len();
        let nprim_j = input.exps_j.len();
        let nprim_k = input.exps_k.len();

        let ci_comps = cart_comps(input.li);
        let cj_comps = cart_comps(input.lj);
        let ck_comps = cart_comps(input.lk);

        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            for pj in 0..nprim_j {
                let aj_val = input.exps_j[pj];
                let cj_val = input.coeff_j[pj];

                let mut pk = 0;
                while pk < nprim_k {
                    let chunk_size = (nprim_k - pk).min(V::LANES);

                    let mut ak_arr = [1.0; 8];
                    let mut weight_arr = [0.0; 8];
                    for lane in 0..chunk_size {
                        ak_arr[lane] = input.exps_k[pk + lane];
                        weight_arr[lane] = ci_val * cj_val * input.coeff_k[pk + lane];
                    }

                    let ai = V::from_f64(ai_val);
                    let aj = V::from_f64(aj_val);
                    let ak = V::from_f64_slice(&ak_arr[..V::LANES], 1.0);
                    let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                    let aijk = ai + aj + ak;
                    let aiajrr = ai * aj * V::from_f64(rr_ij);
                    let aiakrr = ai * ak * V::from_f64(rr_ik);
                    let ajakrr = aj * ak * V::from_f64(rr_jk);
                    let eijk = (aiajrr + aiakrr + ajakrr) / aijk;

                    let dijk = (-eijk).exp() / (aijk * aijk.sqrt());
                    let fac = V::from_f64(common_factor_val) * dijk * weight;
                    let aijk1 = V::from_f64(0.5) / aijk;

                    g.fill(V::splat(V::Scalar::default()));

                    g[0] = V::from_f64(1.0);
                    g[g_alloc] = V::from_f64(1.0);
                    g[2 * g_alloc] = fac;

                    let gx_w = (ai * V::from_f64(input.ri[0])
                        + aj * V::from_f64(input.rj[0])
                        + ak * V::from_f64(input.rk[0]))
                        / aijk;
                    let gy_w = (ai * V::from_f64(input.ri[1])
                        + aj * V::from_f64(input.rj[1])
                        + ak * V::from_f64(input.rk[1]))
                        / aijk;
                    let gz_w = (ai * V::from_f64(input.ri[2])
                        + aj * V::from_f64(input.rj[2])
                        + ak * V::from_f64(input.rk[2]))
                        / aijk;

                    let rjrijkx = V::from_f64(input.rj[0]) - gx_w;
                    let rjrijky = V::from_f64(input.rj[1]) - gy_w;
                    let rjrijkz = V::from_f64(input.rj[2]) - gz_w;

                    for axis in 0..3 {
                        let off = axis * g_alloc;
                        let disp = match axis {
                            0 => -rjrijkx,
                            1 => -rjrijky,
                            _ => -rjrijkz,
                        };
                        let rirj_d = V::from_f64(rirj[axis]);
                        let rjrk_d = V::from_f64(rjrk[axis]);

                        // VRR over combined j+k
                        if nmax >= 1 {
                            g[off + dj] = disp * g[off];
                            for j in 1..nmax {
                                let hi = aijk1 * V::from_f64(j as f64) * g[off + (j - 1) * dj]
                                    + disp * g[off + j * dj];
                                g[off + (j + 1) * dj] = hi;
                            }
                        }

                        // i-HRR
                        if li >= 1 {
                            for i in 1..=li {
                                let j_max = nmax - i;
                                for j in 0..=j_max {
                                    let idx_out = i + j * dj;
                                    let idx_hi = (i - 1) + (j + 1) * dj;
                                    let idx_lo = (i - 1) + j * dj;
                                    g[off + idx_out] = g[off + idx_hi] - rirj_d * g[off + idx_lo];
                                }
                            }
                        }

                        // k-separation HRR
                        if lk >= 1 {
                            for k in 1..=lk {
                                let j_max = mmax - k;
                                for j in 0..=j_max {
                                    let base = k * dk + j * dj;
                                    for i in 0..=li {
                                        let idx = base + i;
                                        let idx_hi = idx + dj - dk;
                                        let idx_lo = idx - dk;
                                        g[off + idx] = g[off + idx_hi] + rjrk_d * g[off + idx_lo];
                                    }
                                }
                            }
                        }
                    }

                    // Cartesian Contraction
                    let gx = 0;
                    let gy = g_alloc;
                    let gz = 2 * g_alloc;

                    for (ck_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
                        for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                                let vx = g[gx + ix as usize + jx as usize * dj + kx as usize * dk];
                                let vy = g[gy + iy as usize + jy as usize * dj + ky as usize * dk];
                                let vz = g[gz + iz as usize + jz as usize * dj + kz as usize * dk];

                                let out_idx = (ck_idx * ncj + cj_idx) * nci + ci_idx;
                                let term = vx * vy * vz;
                                out[out_idx] += term.reduce_add().into();
                            }
                        }
                    }

                    pk += chunk_size;
                }
            }
        }
    }
}
