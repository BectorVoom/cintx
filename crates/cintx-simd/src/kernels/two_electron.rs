use crate::boys::{rys_root1_simd, rys_root2_simd};
use crate::kernels::one_electron::{cart_comps, common_fac_sp, ncart, SQRTPI};
use crate::vector::SimdFloat;
use std::f64::consts::PI;

/// Input parameters for 2-electron integral evaluation (`int2e`).
#[derive(Clone, Debug)]
pub struct TwoElectronInput<'a> {
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

fn build_2e_shape(li: usize, lj: usize, lk: usize, ll: usize) -> TwoEShape {
    let nroots = (li + lj + lk + ll) / 2 + 1;
    let nmax = li + lj;
    let mmax = lk + ll;

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

/// SIMD-vectorized 2-electron integral kernel (`int2e`).
pub struct SimdTwoElectronKernel;

impl SimdTwoElectronKernel {
    /// Evaluate 2-electron repulsion integrals (`int2e`).
    ///
    /// Writes results into `out` in Cartesian component ordering (`l` outer, `k` middle-2, `j` middle-1, `i` inner).
    pub fn eval<V: SimdFloat>(input: &TwoElectronInput<'_>, out: &mut [f64]) {
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

        let shape = build_2e_shape(li, lj, lk, ll);
        let nroots = shape.nroots;
        let mut g = vec![V::splat(V::Scalar::default()); 3 * shape.g_size];

        let common_factor_val = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(input.li)
            * common_fac_sp(input.lj)
            * common_fac_sp(input.lk)
            * common_fac_sp(input.ll);

        let (rx_in_rijrx_val, rirj_val) = if shape.ibase {
            (
                input.ri,
                [
                    input.ri[0] - input.rj[0],
                    input.ri[1] - input.rj[1],
                    input.ri[2] - input.rj[2],
                ],
            )
        } else {
            (
                input.rj,
                [
                    input.rj[0] - input.ri[0],
                    input.rj[1] - input.ri[1],
                    input.rj[2] - input.ri[2],
                ],
            )
        };

        let (rx_in_rklrx_val, rkrl_val) = if shape.kbase {
            (
                input.rk,
                [
                    input.rk[0] - input.rl[0],
                    input.rk[1] - input.rl[1],
                    input.rk[2] - input.rl[2],
                ],
            )
        } else {
            (
                input.rl,
                [
                    input.rl[0] - input.rk[0],
                    input.rl[1] - input.rk[1],
                    input.rl[2] - input.rk[2],
                ],
            )
        };

        let nprim_i = input.exps_i.len();
        let nprim_j = input.exps_j.len();
        let nprim_k = input.exps_k.len();
        let nprim_l = input.exps_l.len();

        let ci_comps = cart_comps(input.li);
        let cj_comps = cart_comps(input.lj);
        let ck_comps = cart_comps(input.lk);
        let cl_comps = cart_comps(input.ll);

        for pi in 0..nprim_i {
            let ai_val = input.exps_i[pi];
            let ci_val = input.coeff_i[pi];

            for pj in 0..nprim_j {
                let aj_val = input.exps_j[pj];
                let cj_val = input.coeff_j[pj];

                let aij_val = ai_val + aj_val;
                let dx_ij_val = input.ri[0] - input.rj[0];
                let dy_ij_val = input.ri[1] - input.rj[1];
                let dz_ij_val = input.ri[2] - input.rj[2];
                let rr_ij_val =
                    dx_ij_val * dx_ij_val + dy_ij_val * dy_ij_val + dz_ij_val * dz_ij_val;
                let fac_ij_val = (-ai_val * aj_val / aij_val * rr_ij_val).exp();

                let px_val = (ai_val * input.ri[0] + aj_val * input.rj[0]) / aij_val;
                let py_val = (ai_val * input.ri[1] + aj_val * input.rj[1]) / aij_val;
                let pz_val = (ai_val * input.ri[2] + aj_val * input.rj[2]) / aij_val;

                let px = V::from_f64(px_val);
                let py = V::from_f64(py_val);
                let pz = V::from_f64(pz_val);
                let aij = V::from_f64(aij_val);

                let rijrx_x = px - V::from_f64(rx_in_rijrx_val[0]);
                let rijrx_y = py - V::from_f64(rx_in_rijrx_val[1]);
                let rijrx_z = pz - V::from_f64(rx_in_rijrx_val[2]);

                for pk in 0..nprim_k {
                    let ak_val = input.exps_k[pk];
                    let ck_val = input.coeff_k[pk];

                    let mut pl = 0;
                    while pl < nprim_l {
                        let chunk_size = (nprim_l - pl).min(V::LANES);

                        let mut al_arr = [1.0; 8];
                        let mut weight_arr = [0.0; 8];
                        for lane in 0..chunk_size {
                            al_arr[lane] = input.exps_l[pl + lane];
                            weight_arr[lane] =
                                ci_val * cj_val * ck_val * input.coeff_l[pl + lane];
                        }

                        let ak = V::from_f64(ak_val);
                        let al = V::from_f64_slice(&al_arr[..V::LANES], 1.0);
                        let weight = V::from_f64_slice(&weight_arr[..V::LANES], 0.0);

                        let akl = ak + al;
                        let a1 = aij * akl;
                        let a0 = a1 / (aij + akl);

                        let rk_x = V::from_f64(input.rk[0]);
                        let rk_y = V::from_f64(input.rk[1]);
                        let rk_z = V::from_f64(input.rk[2]);

                        let rl_x = V::from_f64(input.rl[0]);
                        let rl_y = V::from_f64(input.rl[1]);
                        let rl_z = V::from_f64(input.rl[2]);

                        let dx_kl = rk_x - rl_x;
                        let dy_kl = rk_y - rl_y;
                        let dz_kl = rk_z - rl_z;
                        let rr_kl = dx_kl * dx_kl + dy_kl * dy_kl + dz_kl * dz_kl;

                        let qx = (ak * rk_x + al * rl_x) / akl;
                        let qy = (ak * rk_y + al * rl_y) / akl;
                        let qz = (ak * rk_z + al * rl_z) / akl;

                        let dx_pq = px - qx;
                        let dy_pq = py - qy;
                        let dz_pq = pz - qz;
                        let rr_pq = dx_pq * dx_pq + dy_pq * dy_pq + dz_pq * dz_pq;

                        let fac_ij = V::from_f64(fac_ij_val);
                        let fac_kl = (-ak * al / akl * rr_kl).exp();

                        let common_factor = V::from_f64(common_factor_val);
                        let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * common_factor * fac_ij * fac_kl;
                        let x_rys = a0 * rr_pq;

                        let rklrx_x = qx - V::from_f64(rx_in_rklrx_val[0]);
                        let rklrx_y = qy - V::from_f64(rx_in_rklrx_val[1]);
                        let rklrx_z = qz - V::from_f64(rx_in_rklrx_val[2]);

                        g.fill(V::splat(V::Scalar::default()));

                        let gy_off = shape.g_size;
                        let gz_off = 2 * shape.g_size;

                        if nroots == 1 {
                            let (u0, w0) = rys_root1_simd(x_rys);
                            g[0] = V::from_f64(1.0);
                            g[gy_off] = V::from_f64(1.0);
                            g[gz_off] = w0 * fac1;

                            let u2 = a0 * u0;
                            let tmp4 = V::from_f64(0.5) / (u2 * (aij + akl) + a1);
                            let tmp5 = u2 * tmp4;
                            let tmp1 = V::from_f64(2.0) * tmp5;
                            let tmp2 = tmp1 * akl;
                            let tmp3 = tmp1 * aij;

                            let b00 = tmp5;
                            let b10 = tmp5 + tmp4 * akl;
                            let b01 = tmp5 + tmp4 * aij;

                            let c00x = rijrx_x - tmp2 * dx_pq;
                            let c00y = rijrx_y - tmp2 * dy_pq;
                            let c00z = rijrx_z - tmp2 * dz_pq;

                            let c0px = rklrx_x + tmp3 * dx_pq;
                            let c0py = rklrx_y + tmp3 * dy_pq;
                            let c0pz = rklrx_z + tmp3 * dz_pq;

                            let (gx, rest) = g.split_at_mut(shape.g_size);
                            let (gy, gz) = rest.split_at_mut(shape.g_size);

                            vrr_fill_axis_simd(
                                gx,
                                0,
                                shape.nmax,
                                shape.mmax,
                                shape.g2d_ijmax,
                                shape.g2d_klmax,
                                c00x,
                                c0px,
                                b10,
                                b01,
                                b00,
                            );
                            vrr_fill_axis_simd(
                                gy,
                                0,
                                shape.nmax,
                                shape.mmax,
                                shape.g2d_ijmax,
                                shape.g2d_klmax,
                                c00y,
                                c0py,
                                b10,
                                b01,
                                b00,
                            );
                            vrr_fill_axis_simd(
                                gz,
                                0,
                                shape.nmax,
                                shape.mmax,
                                shape.g2d_ijmax,
                                shape.g2d_klmax,
                                c00z,
                                c0pz,
                                b10,
                                b01,
                                b00,
                            );
                        } else if nroots == 2 {
                            let (u, w) = rys_root2_simd(x_rys);
                            for irys in 0..2 {
                                g[irys] = V::from_f64(1.0);
                                g[gy_off + irys] = V::from_f64(1.0);
                                g[gz_off + irys] = w[irys] * fac1;

                                let u2 = a0 * u[irys];
                                let tmp4 = V::from_f64(0.5) / (u2 * (aij + akl) + a1);
                                let tmp5 = u2 * tmp4;
                                let tmp1 = V::from_f64(2.0) * tmp5;
                                let tmp2 = tmp1 * akl;
                                let tmp3 = tmp1 * aij;

                                let b00 = tmp5;
                                let b10 = tmp5 + tmp4 * akl;
                                let b01 = tmp5 + tmp4 * aij;

                                let c00x = rijrx_x - tmp2 * dx_pq;
                                let c00y = rijrx_y - tmp2 * dy_pq;
                                let c00z = rijrx_z - tmp2 * dz_pq;

                                let c0px = rklrx_x + tmp3 * dx_pq;
                                let c0py = rklrx_y + tmp3 * dy_pq;
                                let c0pz = rklrx_z + tmp3 * dz_pq;

                                let (gx, rest) = g.split_at_mut(shape.g_size);
                                let (gy, gz) = rest.split_at_mut(shape.g_size);

                                vrr_fill_axis_simd(
                                    gx,
                                    irys,
                                    shape.nmax,
                                    shape.mmax,
                                    shape.g2d_ijmax,
                                    shape.g2d_klmax,
                                    c00x,
                                    c0px,
                                    b10,
                                    b01,
                                    b00,
                                );
                                vrr_fill_axis_simd(
                                    gy,
                                    irys,
                                    shape.nmax,
                                    shape.mmax,
                                    shape.g2d_ijmax,
                                    shape.g2d_klmax,
                                    c00y,
                                    c0py,
                                    b10,
                                    b01,
                                    b00,
                                );
                                vrr_fill_axis_simd(
                                    gz,
                                    irys,
                                    shape.nmax,
                                    shape.mmax,
                                    shape.g2d_ijmax,
                                    shape.g2d_klmax,
                                    c00z,
                                    c0pz,
                                    b10,
                                    b01,
                                    b00,
                                );
                            }
                        }

                        // HRR Transfer in-place
                        let rirj = [
                            V::from_f64(rirj_val[0]),
                            V::from_f64(rirj_val[1]),
                            V::from_f64(rirj_val[2]),
                        ];
                        let rkrl = [
                            V::from_f64(rkrl_val[0]),
                            V::from_f64(rkrl_val[1]),
                            V::from_f64(rkrl_val[2]),
                        ];

                        if shape.kbase {
                            if shape.ibase {
                                hrr_ik2d_4d_simd(&mut g, shape, rirj, rkrl);
                            } else {
                                hrr_kj2d_4d_simd(&mut g, shape, rirj, rkrl);
                            }
                        } else if shape.ibase {
                            hrr_il2d_4d_simd(&mut g, shape, rirj, rkrl);
                        } else {
                            hrr_lj2d_4d_simd(&mut g, shape, rirj, rkrl);
                        }

                        // Cartesian Contraction
                        let gx_off = 0usize;
                        let gy_off = shape.g_size;
                        let gz_off = 2 * shape.g_size;

                        for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
                            for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
                                for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                                    for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                                        let mut sum = V::splat(V::Scalar::default());
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
                                            sum = sum + g[gx_off + x_idx] * g[gy_off + y_idx] * g[gz_off + z_idx];
                                        }

                                        let out_idx = i_idx
                                            + j_idx * nfi
                                            + k_idx * nfi * nfj
                                            + l_idx * nfi * nfj * nfk;
                                        let val = sum * weight;
                                        out[out_idx] += val.reduce_add().into();
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

#[inline]
fn vrr_fill_axis_simd<V: SimdFloat>(
    g_axis: &mut [V],
    root: usize,
    nmax: usize,
    mmax: usize,
    dn: usize,
    dm: usize,
    c00: V,
    c0p: V,
    b10: V,
    b01: V,
    b00: V,
) {
    if nmax > 0 {
        let mut s0 = g_axis[root];
        let mut s1 = c00 * s0;
        g_axis[root + dn] = s1;
        for n in 1..nmax {
            let s2 = c00 * s1 + V::from_f64(n as f64) * b10 * s0;
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
            let s2 = c0p * s1 + V::from_f64(m as f64) * b01 * s0;
            g_axis[root + (m + 1) * dm] = s2;
            s0 = s1;
            s1 = s2;
        }

        if nmax > 0 {
            let mut s0n = g_axis[root + dn];
            let mut s1n = c0p * s0n + b00 * g_axis[root];
            g_axis[root + dn + dm] = s1n;
            for m in 1..mmax {
                let s2n = c0p * s1n + V::from_f64(m as f64) * b01 * s0n + b00 * g_axis[root + m * dm];
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
                let s2 = c00 * s1
                    + V::from_f64(n as f64) * b10 * s0
                    + V::from_f64(m as f64) * b00 * g_axis[j + n * dn - dm];
                g_axis[j + (n + 1) * dn] = s2;
                s0 = s1;
                s1 = s2;
            }
        }
    }
}

fn hrr_lj2d_4d_simd<V: SimdFloat>(g: &mut [V], shape: TwoEShape, rirj: [V; 3], rkrl: [V; 3]) {
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

fn hrr_kj2d_4d_simd<V: SimdFloat>(g: &mut [V], shape: TwoEShape, rirj: [V; 3], rkrl: [V; 3]) {
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

fn hrr_il2d_4d_simd<V: SimdFloat>(g: &mut [V], shape: TwoEShape, rirj: [V; 3], rkrl: [V; 3]) {
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

fn hrr_ik2d_4d_simd<V: SimdFloat>(g: &mut [V], shape: TwoEShape, rirj: [V; 3], rkrl: [V; 3]) {
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
