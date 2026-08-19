use crate::boys::rys_roots_simd;
use crate::kernels::one_electron::{common_fac_sp, ncart, SQRTPI};
use crate::kernels::recurrence::vrr_2e_2d_axis;
use crate::vector::SimdFloat;
use std::f64::consts::PI;

/// Input parameters for 2-center 2-electron integral evaluation (`int2c2e`).
#[derive(Clone, Debug)]
pub struct Center2c2eInput<'a> {
    pub li: u8,
    pub lk: u8,
    pub ri: [f64; 3],
    pub rk: [f64; 3],
    pub exps_i: &'a [f64],
    pub exps_k: &'a [f64],
    pub coeff_i: &'a [f64],
    pub coeff_k: &'a [f64],
}

/// SIMD-vectorized 2-center 2-electron repulsion integral kernel (`(i|k)`).
pub struct SimdCenter2c2eKernel;

impl SimdCenter2c2eKernel {
    /// Evaluate 2-center 2-electron repulsion integrals (`int2c2e`).
    ///
    /// Writes results into `out` in Cartesian component ordering (`k` outer, `i` inner).
    pub fn eval<V: SimdFloat>(input: &Center2c2eInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lk = input.lk as usize;
        let nci = ncart(input.li);
        let _nck = ncart(input.lk);

        out.fill(0.0);

        let nmax = li;
        let mmax = lk;
        let nroots = (input.li + input.lk) as usize / 2 + 1;

        let dn = nroots;
        let dm = nroots * (nmax + 1);
        let g_size = nroots * (nmax + 1) * (mmax + 1);
        let total_g = 3 * g_size;
        let gx = 0;
        let gy = g_size;
        let gz = 2 * g_size;

        let mut g = vec![V::splat(V::Scalar::default()); total_g];

        let dx_val = input.ri[0] - input.rk[0];
        let dy_val = input.ri[1] - input.rk[1];
        let dz_val = input.ri[2] - input.rk[2];
        let rr_val = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

        let common_factor_val =
            (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(input.li) * common_fac_sp(input.lk);

        let nprim_i = input.exps_i.len();
        let nprim_k = input.exps_k.len();

        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            let mut pk = 0;
            while pk < nprim_k {
                let chunk_size = (nprim_k - pk).min(V::LANES);

                let mut ak_arr = [1.0; 8];
                let mut weight_arr = [0.0; 8];
                for lane in 0..chunk_size {
                    ak_arr[lane] = input.exps_k[pk + lane];
                    weight_arr[lane] = ci_val * input.coeff_k[pk + lane];
                }

                let ai = V::from_f64(ai_val);
                let ak = V::from_f64_slice(&ak_arr[..V::LANES], 1.0);
                let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                let aij = ai;
                let akl = ak;
                let a1 = aij * akl;
                let a0 = a1 / (aij + akl);

                let rr = V::from_f64(rr_val);
                let x_rys = a0 * rr;
                let common_factor = V::from_f64(common_factor_val);
                let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * common_factor;

                let dx = V::from_f64(dx_val);
                let dy = V::from_f64(dy_val);
                let dz = V::from_f64(dz_val);

                g.fill(V::splat(V::Scalar::default()));

                let (u_vec, w_vec) = rys_roots_simd(nroots, x_rys);
                for irys in 0..nroots {
                    let u2 = a0 * u_vec[irys];
                    let tmp4 = V::from_f64(0.5) / (u2 * (aij + akl) + a1);
                    let tmp5 = u2 * tmp4;
                    let b00 = tmp5;
                    let b10 = tmp5 + tmp4 * akl;
                    let b01 = tmp5 + tmp4 * aij;
                    let tmp2 = V::from_f64(2.0) * tmp5 * akl;
                    let tmp3 = V::from_f64(2.0) * tmp5 * aij;

                    let c00x = -tmp2 * dx;
                    let c00y = -tmp2 * dy;
                    let c00z = -tmp2 * dz;
                    let c0px = tmp3 * dx;
                    let c0py = tmp3 * dy;
                    let c0pz = tmp3 * dz;

                    g[gx + irys] = V::from_f64(1.0);
                    g[gy + irys] = V::from_f64(1.0);
                    g[gz + irys] = w_vec[irys] * fac1;

                    vrr_2e_2d_axis(
                        &mut g, gx, irys, nmax, mmax, dn, dm, c00x, c0px, b10, b01, b00,
                    );
                    vrr_2e_2d_axis(
                        &mut g, gy, irys, nmax, mmax, dn, dm, c00y, c0py, b10, b01, b00,
                    );
                    vrr_2e_2d_axis(
                        &mut g, gz, irys, nmax, mmax, dn, dm, c00z, c0pz, b10, b01, b00,
                    );
                }

                // Contract Cartesian components
                let mut ck_idx = 0;
                for ka in 0..=lk {
                    let kx = lk - ka;
                    let lk_minus_kx = lk - kx;
                    for kb in 0..=lk_minus_kx {
                        let ky = lk_minus_kx - kb;
                        let kz = lk - kx - ky;

                        let mut ci_idx = 0;
                        for ia in 0..=li {
                            let ix = li - ia;
                            let li_minus_ix = li - ix;
                            for ib in 0..=li_minus_ix {
                                let iy = li_minus_ix - ib;
                                let iz = li - ix - iy;

                                let mut total_term = V::splat(V::Scalar::default());
                                for r in 0..nroots {
                                    let idx_x = r + ix * dn + kx * dm;
                                    let idx_y = r + iy * dn + ky * dm;
                                    let idx_z = r + iz * dn + kz * dm;

                                    let vx = g[gx + idx_x];
                                    let vy = g[gy + idx_y];
                                    let vz = g[gz + idx_z];
                                    total_term = total_term + vx * vy * vz;
                                }

                                let term = (weight * total_term).reduce_add();
                                let term_f64: f64 = term.into();
                                out[ck_idx * nci + ci_idx] += term_f64;

                                ci_idx += 1;
                            }
                        }
                        ck_idx += 1;
                    }
                }

                pk += chunk_size;
            }
        }
    }
}
