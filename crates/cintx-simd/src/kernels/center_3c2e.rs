use crate::boys::{rys_root1_simd, rys_root2_simd};
use crate::kernels::one_electron::{cart_comps, common_fac_sp, ncart, SQRTPI};
use crate::kernels::recurrence::vrr_2e_2d_axis;
use crate::vector::SimdFloat;
use std::f64::consts::PI;

/// Input parameters for 3-center 2-electron integral evaluation (`int3c2e`, `(ij|k)`).
#[derive(Clone, Debug)]
pub struct Center3c2eInput<'a> {
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

/// SIMD-vectorized 3-center 2-electron Coulomb integral kernel (`(ij|k)`).
pub struct SimdCenter3c2eKernel;

impl SimdCenter3c2eKernel {
    /// Evaluate 3-center 2-electron Coulomb integrals (`(ij|k)`).
    ///
    /// Writes results into `out` in Cartesian component ordering (`k` outer, `j` middle, `i` inner).
    pub fn eval<V: SimdFloat>(input: &Center3c2eInput<'_>, out: &mut [f64]) {
        let li = input.li as usize;
        let lj = input.lj as usize;
        let lk = input.lk as usize;

        let nfi = ncart(input.li);
        let nfj = ncart(input.lj);
        let nfk = ncart(input.lk);
        let total_comps = nfi * nfj * nfk;

        out[..total_comps].fill(0.0);

        let nmax = li + lj;
        let mmax = lk;
        let nroots = (input.li + input.lj + input.lk) as usize / 2 + 1;

        let dn = nroots;
        let dm = nroots * (nmax + 1);
        let g2d_size = nroots * (nmax + 1) * (mmax + 1);
        let total_g = 3 * g2d_size;
        let gx = 0;
        let gy = g2d_size;
        let gz = 2 * g2d_size;

        let ni = li + 1;
        let nj = lj + 1;
        let nk = lk + 1;
        let axis_size = nroots * nk * nj * ni;
        let total_split = 3 * axis_size;

        let mut g = vec![V::splat(V::Scalar::default()); total_g];
        let mut g_split = vec![V::splat(V::Scalar::default()); total_split];

        let dx_ij_val = input.ri[0] - input.rj[0];
        let dy_ij_val = input.ri[1] - input.rj[1];
        let dz_ij_val = input.ri[2] - input.rj[2];
        let rr_ij_val = dx_ij_val * dx_ij_val + dy_ij_val * dy_ij_val + dz_ij_val * dz_ij_val;

        let rirj_val = [
            input.ri[0] - input.rj[0],
            input.ri[1] - input.rj[1],
            input.ri[2] - input.rj[2],
        ];

        let common_factor_val = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(input.li)
            * common_fac_sp(input.lj)
            * common_fac_sp(input.lk);

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

                let aij_val = ai_val + aj_val;
                let fac_ij_val = (-ai_val * aj_val / aij_val * rr_ij_val).exp();

                let px_val = (ai_val * input.ri[0] + aj_val * input.rj[0]) / aij_val;
                let py_val = (ai_val * input.ri[1] + aj_val * input.rj[1]) / aij_val;
                let pz_val = (ai_val * input.ri[2] + aj_val * input.rj[2]) / aij_val;

                let px = V::from_f64(px_val);
                let py = V::from_f64(py_val);
                let pz = V::from_f64(pz_val);

                let aij = V::from_f64(aij_val);

                let mut pk = 0;
                while pk < nprim_k {
                    let chunk_size = (nprim_k - pk).min(V::LANES);

                    let mut ak_arr = [1.0; 8];
                    let mut weight_arr = [0.0; 8];
                    for lane in 0..chunk_size {
                        ak_arr[lane] = input.exps_k[pk + lane];
                        weight_arr[lane] = ci_val * cj_val * input.coeff_k[pk + lane];
                    }

                    let ak = V::from_f64_slice(&ak_arr[..V::LANES], 1.0);
                    let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                    let akl = ak;
                    let a1 = aij * akl;
                    let a0 = a1 / (aij + akl);

                    let rk_x = V::from_f64(input.rk[0]);
                    let rk_y = V::from_f64(input.rk[1]);
                    let rk_z = V::from_f64(input.rk[2]);

                    let dx_pk = px - rk_x;
                    let dy_pk = py - rk_y;
                    let dz_pk = pz - rk_z;
                    let rr_pk = dx_pk * dx_pk + dy_pk * dy_pk + dz_pk * dz_pk;

                    let fac_ij = V::from_f64(fac_ij_val);
                    let common_factor = V::from_f64(common_factor_val);
                    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * common_factor * fac_ij;
                    let x_rys = a0 * rr_pk;

                    let ri_x = V::from_f64(input.ri[0]);
                    let ri_y = V::from_f64(input.ri[1]);
                    let ri_z = V::from_f64(input.ri[2]);

                    let rx_ij_x = px - ri_x;
                    let rx_ij_y = py - ri_y;
                    let rx_ij_z = pz - ri_z;

                    g.fill(V::splat(V::Scalar::default()));

                    if nroots == 1 {
                        let (u0, w0) = rys_root1_simd(x_rys);
                        let u2 = a0 * u0;
                        let tmp4 = V::from_f64(0.5) / (u2 * (aij + akl) + a1);
                        let tmp5 = u2 * tmp4;
                        let b00 = tmp5;
                        let b10 = tmp5 + tmp4 * akl;
                        let b01 = tmp5 + tmp4 * aij;
                        let tmp2 = V::from_f64(2.0) * tmp5 * akl;
                        let tmp3 = V::from_f64(2.0) * tmp5 * aij;

                        let c00x = rx_ij_x - tmp2 * dx_pk;
                        let c00y = rx_ij_y - tmp2 * dy_pk;
                        let c00z = rx_ij_z - tmp2 * dz_pk;
                        let c0px = tmp3 * dx_pk;
                        let c0py = tmp3 * dy_pk;
                        let c0pz = tmp3 * dz_pk;

                        g[gx] = V::from_f64(1.0);
                        g[gy] = V::from_f64(1.0);
                        g[gz] = w0 * fac1;

                        vrr_2e_2d_axis(
                            &mut g, gx, 0, nmax, mmax, dn, dm, c00x, c0px, b10, b01, b00,
                        );
                        vrr_2e_2d_axis(
                            &mut g, gy, 0, nmax, mmax, dn, dm, c00y, c0py, b10, b01, b00,
                        );
                        vrr_2e_2d_axis(
                            &mut g, gz, 0, nmax, mmax, dn, dm, c00z, c0pz, b10, b01, b00,
                        );
                    } else if nroots == 2 {
                        let (u, w) = rys_root2_simd(x_rys);
                        for irys in 0..2 {
                            let u2 = a0 * u[irys];
                            let tmp4 = V::from_f64(0.5) / (u2 * (aij + akl) + a1);
                            let tmp5 = u2 * tmp4;
                            let b00 = tmp5;
                            let b10 = tmp5 + tmp4 * akl;
                            let b01 = tmp5 + tmp4 * aij;
                            let tmp2 = V::from_f64(2.0) * tmp5 * akl;
                            let tmp3 = V::from_f64(2.0) * tmp5 * aij;

                            let c00x = rx_ij_x - tmp2 * dx_pk;
                            let c00y = rx_ij_y - tmp2 * dy_pk;
                            let c00z = rx_ij_z - tmp2 * dz_pk;
                            let c0px = tmp3 * dx_pk;
                            let c0py = tmp3 * dy_pk;
                            let c0pz = tmp3 * dz_pk;

                            g[gx + irys] = V::from_f64(1.0);
                            g[gy + irys] = V::from_f64(1.0);
                            g[gz + irys] = w[irys] * fac1;

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
                    }

                    // HRR Transfer on bra j (j <- i)
                    let work_stride = nmax + 1;
                    let mut work = vec![V::splat(V::Scalar::default()); nj * work_stride];

                    for axis in 0..3 {
                        let axis_in_off = axis * g2d_size;
                        let axis_out_off = axis * axis_size;
                        let rirj = V::from_f64(rirj_val[axis]);

                        for k in 0..=mmax {
                            for root in 0..nroots {
                                for i in 0..=nmax {
                                    work[i] = g[axis_in_off + root + i * dn + k * dm];
                                }

                                for j in 1..=lj {
                                    let prev = (j - 1) * work_stride;
                                    let cur = j * work_stride;
                                    let i_max = nmax - j;
                                    for i in 0..=i_max {
                                        work[cur + i] = rirj * work[prev + i] + work[prev + i + 1];
                                    }
                                }

                                for j in 0..=lj {
                                    for i in 0..=li {
                                        let out_idx = ((root * nk + k) * nj + j) * ni + i;
                                        g_split[axis_out_off + out_idx] = work[j * work_stride + i];
                                    }
                                }
                            }
                        }
                    }

                    // Cartesian Contraction
                    let gx_off = 0usize;
                    let gy_off = axis_size;
                    let gz_off = 2 * axis_size;

                    for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
                        for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                            for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                                let mut sum = V::splat(V::Scalar::default());
                                for r in 0..nroots {
                                    let x_idx = ((r * nk + kx as usize) * nj + jx as usize) * ni + ix as usize;
                                    let y_idx = ((r * nk + ky as usize) * nj + jy as usize) * ni + iy as usize;
                                    let z_idx = ((r * nk + kz as usize) * nj + jz as usize) * ni + iz as usize;

                                    sum = sum + g_split[gx_off + x_idx] * g_split[gy_off + y_idx] * g_split[gz_off + z_idx];
                                }

                                let out_idx = (k_idx * nfj + j_idx) * nfi + i_idx;
                                let val = sum * weight;
                                out[out_idx] += val.reduce_add().into();
                            }
                        }
                    }

                    pk += chunk_size;
                }
            }
        }
    }
}
