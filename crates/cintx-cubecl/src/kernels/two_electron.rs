//! Host-side 2e (four-center electron-repulsion) integral kernel.
//!
//! Implements the libcint `g2e.c` recurrence pipeline:
//! 1. Rys roots/weights per primitive quartet (`rys_roots_host`).
//! 2. 2D VRR fill (`CINTg0_2e_2d` equivalent).
//! 3. Branch-specific 4D HRR transfer (ibase/kbase adaptive stride choice).
//! 4. Cartesian contraction + optional `cart_to_sph_2e` transform.

// Transcribed verbatim from vendored libcint 6.1.3 (and, in `cintx-basis`, from the
// Lanczos reference these normalization constants come from). Result compatibility
// is decided by the exact bits these literals carry, so none is truncated to the
// shortest form that round-trips.
#![allow(clippy::excessive_precision)]
// The `as usize` / `as u32` casts here are load-bearing under `#[cube]`: the
// CubeCL builtins (`UNIT_POS`, `CUBE_DIM`, ...) expand to `NativeExpand<u32>`,
// and `Array` indexing takes a `usize`, so the uniform `(expr) as usize` form is
// what lets an index expression be swapped between a literal and a variable.
// Clippy sees the post-expansion type and reads them as redundant.
#![allow(clippy::unnecessary_cast)]
// Index-carrying loops (`for axis in 0..3`, `for i in 0..n`) index several
// parallel arrays or a strided buffer, and the index itself names an axis,
// component or stride. An iterator rewrite would hide exactly that.
#![allow(clippy::needless_range_loop)]
// Kernel launches take the whole shape contract as positional arguments — that
// is the CubeCL calling convention, not a design choice — and the host wrappers
// mirror it so the two can be read side by side.
#![allow(clippy::too_many_arguments)]

use crate::backend::ResolvedBackend;
use crate::kernels::f12::Gauge2eKind;
use crate::kernels::pair_table::{PAIR_DATA_STRIDE, PAIR_INDEX_STRIDE};
use crate::math::pdata::compute_pdata_host;
use crate::math::root_vec::{roots_load, vrr_fill_axis_roots};
use crate::math::rys::rys_roots_fixed_rt;
use crate::math::rys_wheeler::{
    EXT_TABLES_LEN, ext_rys_out_slots, ext_rys_slots, rys_roots_ext_dev,
};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, ncart, nsph};
use crate::transform::c2spinor::{
    cart_to_spinor_sf_2e1, cart_to_spinor_sf_2e2, cart_to_spinor_sf_4d, cart_to_spinor_si_2e1,
    cart_to_spinor_si_2e1i, cart_to_spinor_si_2e2, cart_to_spinor_si_2e2i, spinor_len,
};
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// sqrt(pi) constant — matches libcint `SQRTPI`.
const SQRTPI: f64 = 1.7724538509055160272981674833411451_f64;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
// Verbatim libcint literal, not `std::f64::consts::FRAC_PI_4`: result compatibility
// with upstream is decided by the exact bits this file feeds the Rys kernels, so
// the constant is transcribed from `rys_roots.c` rather than recomputed.
#[allow(clippy::approx_constant)]
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the HOST Rys engine (`rys_roots_host` → `rys_wheeler`) evaluates
/// (Phase 25 FND-02). The host gradient/Hessian path uses the Wheeler nroots 6..12
/// engine; the vendor build caps at 12 (quadmath disabled), so nroots>12 stays
/// fail-closed (T-25-03). What the *device* kernels may serve is no longer a
/// constant here: it is `device_rys_ceiling::device_nroots_ceiling(backend,
/// RysFamily::Int2e)`, which is `BASE_DEVICE_NROOTS` (5) unless the
/// `extended-device-rys` feature, the backend's FMA probe and this family's
/// flip all agree (task 33-03).
const HOST_RYS_NROOTS_CEILING: usize = 12;

/// Spherical harmonic normalization prefactor for s and p shells.
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// The 2e common prefactor `(π³·2/√π) · ∏ common_fac_sp(l)` for a shell quartet
/// `(li,lj,lk,ll)` — the same value `launch_two_electron_typed` builds before
/// dispatch. Exposed so external drivers (the D-03 transform parity test) can
/// invoke [`launch_int2e_spsp1_spinor_quartet`] with the identical normalization
/// the eval_raw path uses, without duplicating the constant.
pub fn int2e_common_factor(li: u8, lj: u8, lk: u8, ll: u8) -> f64 {
    // Left-to-right, one `CINTcommon_fac_sp` at a time — `g2e.c:54-56` is a
    // single chained expression, so grouping the four into an `sp_factor` first
    // and scaling once rounds differently.
    (PI * PI * PI) * 2.0 / SQRTPI
        * common_fac_sp(li)
        * common_fac_sp(lj)
        * common_fac_sp(lk)
        * common_fac_sp(ll)
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

/// `pub(crate)` (Phase 21 D-04): shared with `center_3c2e.rs::int3c2e_ip1`, which
/// builds its derivative G-tensor through the SAME 2e recurrence machinery using the
/// 3c2e Pitfall-4 kl mapping (real `k` → 2e `ll` slot; 2e `lk` slot is a phantom
/// s-function). The struct carries the identical field set as
/// [`crate::kernels::f12::F12Shape`] so `gout_ip1` can be reused verbatim.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TwoEShape {
    pub(crate) nroots: usize,
    /// `rys_order = (Σ l_ceil)/2 + 1` BEFORE any short-range doubling
    /// (`g2e.c:74-79`). Equal to `nroots` for full and long range; half of it
    /// under short range at `rys_order <= 3`, where libcint evaluates SR as
    /// "full minus long range" over `2 * rys_order` roots (D-PBC-24 §3.1).
    pub(crate) rys_order: usize,
    pub(crate) nmax: usize,
    pub(crate) mmax: usize,
    pub(crate) li: usize,
    pub(crate) lj: usize,
    pub(crate) lk: usize,
    pub(crate) ll: usize,
    pub(crate) ibase: bool,
    pub(crate) kbase: bool,
    pub(crate) di: usize,
    pub(crate) dk: usize,
    pub(crate) dl: usize,
    pub(crate) dj: usize,
    pub(crate) g2d_ijmax: usize,
    pub(crate) g2d_klmax: usize,
    pub(crate) g_size: usize,
}

/// Initialize stride/layout metadata following `CINTinit_int2e_EnvVars`.
///
/// `pub(crate)` (Phase 21 D-04): `center_3c2e.rs::int3c2e_ip1` calls this with the
/// 3c2e kl mapping `build_2e_shape(li+1, lj, 0, lk)` (phantom `lk=0`, real k in the
/// `ll` slot, bra `i` raised to `li+1` for the `∇_i` headroom).
pub(crate) fn build_2e_shape(li: usize, lj: usize, lk: usize, ll: usize) -> TwoEShape {
    build_2e_shape_omega(li, lj, lk, ll, None)
}

/// [`build_2e_shape`] with the D-PBC-24 short-range Rys-root doubling applied.
///
/// `g2e.c:76-79` doubles `nrys_roots` when `omega < 0 && rys_order <= 3`, and
/// `nrys_roots` is what `g_stride_i` / `g_stride_k` / `g_size` are built from —
/// so the doubling has to happen HERE, in the stride/layout metadata, not just
/// in the root evaluation. `nmax`, `mmax`, `ibase` and `kbase` are unaffected:
/// they depend on the angular momenta only.
pub(crate) fn build_2e_shape_omega(
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    range_omega: Option<f64>,
) -> TwoEShape {
    let rys_order = (li + lj + lk + ll) / 2 + 1;
    let nroots = cintx_runtime::range_omega::nrys_roots_for(rys_order, range_omega);
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
        rys_order,
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
///
/// `pub(crate)` (Phase 21 D-04): `center_3c2e.rs::int3c2e_ip1` fills its derivative
/// G-tensor through this exact recurrence using the 3c2e kl mapping (phantom 2e `lk`
/// shell with exponent `ak=0` at the real-k center, real k in the 2e `ll` slot).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_g_tensor_2e(
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
    // `range_omega = None` is the `omega == 0` arm of `CINTg0_2e`: it can
    // neither fail closed nor screen a primitive out, so both layers unwrap.
    fill_g_tensor_2e_range(ai, aj, ak, al, ri, rj, rk, rl, shape, fac_env, None)
        .expect("full-range Coulomb is always supported")
        .expect("full-range Coulomb never screens a primitive out")
}

/// [`fill_g_tensor_2e`] under a range-separation parameter ω (D-PBC-24).
///
/// `shape` must come from
/// [`build_2e_shape_omega`]`(.., range_omega)` so its `nroots` strides match the
/// root count this evaluates. Returns `Ok(None)` when the short-range integrand
/// is past `EXPCUTOFF_SR` and libcint would contribute nothing for this
/// primitive quartet (`g2e.c:4460`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_g_tensor_2e_range(
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
    range_omega: Option<f64>,
) -> Result<Option<Vec<f64>>, cintxRsError> {
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

    // D-PBC-24: the shared `CINTg0_2e` omega branch (g2e.c:4443-4512). The
    // `omega == 0` arm is the plain `rys_roots_host(shape.nroots, x_rys)` with
    // `fac1` unchanged, so full range stays byte-identical.
    let Some(roots) = crate::math::range_separation::rys_roots_range_separated(
        shape.rys_order,
        shape.nroots,
        x_rys,
        a0,
        fac1,
        range_omega,
    )?
    else {
        return Ok(None);
    };
    let (u_roots, mut w_weights, fac1) = (roots.u, roots.w, roots.fac1);
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

    Ok(Some(g))
}

/// Contract `[gx|gy|gz]` into Cartesian 2e tensor with output order:
/// `out[i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk]` (i fastest, l slowest).
///
/// Test-only since quick-260529-q4k (see `cart_comps` note): the production scalar
/// 2e path runs `two_electron_scalar_kernel` on-device; this host reference is the
/// `device_tests` cross-check oracle.
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

/// Bridge a plain-2e [`TwoEShape`] into the [`crate::kernels::f12::F12Shape`] that
/// [`crate::kernels::f12::gout_ip1`] / `nabla1i_2e` consume.
///
/// The two structs carry the IDENTICAL field set (di/dj/dk/dl/nroots/nmax/mmax/
/// li/lj/lk/ll/ibase/kbase/g2d_ijmax/g2d_klmax/g_size). The gradient math is
/// F12-free, so this 1:1 field copy lets the plain-Coulomb gradient reuse the
/// exact verbatim derivative code (Phase 21 D-04).
///
/// `pub(crate)` (Phase 21 D-04): `center_3c2e.rs::int3c2e_ip1` reuses this bridge so
/// its 3c2e derivative G-tensor can be fed to the verbatim `gout_ip1` contraction.
#[inline]
pub(crate) fn two_e_shape_as_f12(shape: &TwoEShape) -> crate::kernels::f12::F12Shape {
    crate::kernels::f12::F12Shape {
        nroots: shape.nroots,
        nmax: shape.nmax,
        mmax: shape.mmax,
        li: shape.li,
        lj: shape.lj,
        lk: shape.lk,
        ll: shape.ll,
        ibase: shape.ibase,
        kbase: shape.kbase,
        di: shape.di,
        dk: shape.dk,
        dl: shape.dl,
        dj: shape.dj,
        g2d_ijmax: shape.g2d_ijmax,
        g2d_klmax: shape.g2d_klmax,
        g_size: shape.g_size,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scalar 2e device kernel — `#[cube(launch)]`, generic over `F: Float`
//
//  Faithful inline port of the host SCALAR pipeline
//  `fill_g_tensor_2e` → `contract_2e_cart`, accumulated over all primitive
//  quartets (pi,pj,pk,pl) and contraction quads (ci,cj,ck,cl) into a per-quad
//  i-fastest Cartesian block buffer. Intermediate arithmetic in `F` (run at f64
//  by the launcher), output written to `cart_out` in `F`.
//
//  ALL strides (di,dk,dl,dj,g_size,nmax,mmax,g2d_ijmax,g2d_klmax) plus the
//  ibase/kbase flags are computed host-side via `build_2e_shape` and passed in as
//  runtime u32 — the adaptive dli/dlj/dlk/dll branch logic is NOT recomputed
//  on-device (avoids if-expressions). `#[comptime] nroots` selects rys_root{1..5}.
// ─────────────────────────────────────────────────────────────────────────────

/// Single-work-item scalar 2e kernel. See module note above.
/// One stage of the staged general contraction (GTH plan, C1).
///
/// libcint contracts a generally contracted quartet in four stages
/// (`cint2e.c:193-262`, `PRIM2CTR`): the primitive block is folded into
/// `gctri[ci]` once per primitive quartet, `gctri` into `gctrj[cj][ci]` once
/// per `j` primitive, `gctrj` into `gctrk[ck][cj][ci]` once per `k` primitive
/// and `gctrk` into the output once per `l` primitive. The cost is
/// `nprim^4·nctr_i + nprim^3·nctr_i·nctr_j + …` multiply-adds per element
/// instead of `nprim^4·nctr_i·nctr_j·nctr_k·nctr_l` — a 16x reduction for a
/// TZVP-MOLOPT `(pp|pp)` quartet — and, more to the point on any backend, it
/// removes the read-modify-write of every contraction block per primitive
/// quartet.
///
/// This helper is the middle two stages, which share one shape: `inner`
/// consecutive blocks of `block_len` at `src_off` are scaled by this
/// primitive's `nctr` coefficients and written to `dst_off + (c*inner +
/// inner_idx)*block_len`. `assign == 1` is libcint's `CINTprim_to_ctr_0`
/// (first write since the stage was emptied), otherwise `_1` (accumulate).
///
/// A shell with `nctr == 1` had its coefficient folded into the primitive
/// weight, exactly as libcint folds it into `fac1{l,k,j,i}`, so its stage
/// multiplies by one. Only the lane's own elements (`q % lanes == lane`) are
/// touched in every stage, so no barrier separates the stages in the
/// cooperative decomposition.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_contract<F: Float>(
    ctr: &mut Array<F>,
    src_off: u32,
    dst_off: u32,
    inner: u32,
    coeffs: &Array<F>,
    coff: u32,
    p: u32,
    nctr: u32,
    block_len: u32,
    lane: u32,
    lanes: u32,
    assign: u32,
) {
    let mut c = 0u32;
    while c < nctr {
        let mut cv = F::new(1.0_f32);
        if nctr > 1u32 {
            cv = coeffs[(coff + p * nctr + c) as usize];
        }
        let mut inner_idx = 0u32;
        while inner_idx < inner {
            let src = src_off + inner_idx * block_len;
            let dst = dst_off + (c * inner + inner_idx) * block_len;
            let mut q = lane;
            while q < block_len {
                let term = cv * ctr[(src + q) as usize];
                if assign == 1u32 {
                    ctr[(dst + q) as usize] = term;
                } else {
                    ctr[(dst + q) as usize] += term;
                }
                q += lanes;
            }
            inner_idx += 1u32;
        }
        c += 1u32;
    }
}

/// The last stage of [`stage_contract`]'s scheme: `gctrk[ck][cj][ci]` scaled
/// by this `l` primitive's coefficients, accumulated into the quartet's
/// output blocks in the batch layout
/// `(((ci*nctr_j+cj)*nctr_k+ck)*nctr_l+cl)*block_len` (`i` slowest among the
/// contractions, the layout the host and device transforms both read).
/// `cart_out` was zeroed at the top of the quartet, so this always
/// accumulates.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_contract_out<F: Float>(
    ctr: &Array<F>,
    cart_out: &mut Array<F>,
    src_off: u32,
    out_off: u32,
    coeffs: &Array<F>,
    coff_l: u32,
    pl: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    nctr_l: u32,
    block_len: u32,
    lane: u32,
    lanes: u32,
) {
    let mut cl = 0u32;
    while cl < nctr_l {
        let mut cvl = F::new(1.0_f32);
        if nctr_l > 1u32 {
            cvl = coeffs[(coff_l + pl * nctr_l + cl) as usize];
        }
        let mut ck = 0u32;
        while ck < nctr_k {
            let mut cj = 0u32;
            while cj < nctr_j {
                let mut ci = 0u32;
                while ci < nctr_i {
                    let src = src_off + ((ck * nctr_j + cj) * nctr_i + ci) * block_len;
                    let dst =
                        out_off + (((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl) * block_len;
                    let mut q = lane;
                    while q < block_len {
                        cart_out[(dst + q) as usize] += cvl * ctr[(src + q) as usize];
                        q += lanes;
                    }
                    ci += 1u32;
                }
                cj += 1u32;
            }
            ck += 1u32;
        }
        cl += 1u32;
    }
}

/// One HRR raise chain, at a **comptime** root width.
///
/// Every one of the eight raises the cooperative arm can run — two per
/// `(ibase, kbase)` arm — has the same shape once it is written down:
///
/// ```text
/// for a in 1..=a_max:                  # the index being raised, stride `sa`
///     for b in 0..=(m_max - a):        # the free index of the 2D block, `sb`
///         for r in r_lo..r_lo + nr:    # the Rys roots this task owns
///             g[base + a*sa + b*sb + r] = coef * g[…- sa] + g[…- sa + sb]
/// ```
///
/// `base` carries the axis plane and whichever indices the *task* fixed (the
/// third index of the first raise, the pair of the second), so the four
/// `(ibase, kbase)` arms differ only in which strides they hand over — and
/// `lj2d`'s and `kj2d`'s `nb`-stepping inner loops are the `b` sweep of this
/// same nest, which is what `CINTg0_kj2d_4d` (g2e.c:552) walks.
///
/// **The root width is comptime**, which is the point (plan §23, and the
/// `#[unroll]` chapter of the CubeCL manual). It was a runtime `while r <
/// r_hi` bound by a value read from the class row, so the innermost statement
/// of the recurrence — three index computations, two loads, one FMA, one store
/// — carried a compare, an add and a backward branch it could not lose, on a
/// part that is latency-bound per cube. A cooperative task owns exactly one
/// root (`rw == 1`) or the whole order (`rw == nroots`, one to five), so the
/// caller picks the width with one branch per *task* and the loop disappears.
///
/// Same expressions, same operands, same order: the slab is bit-identical to
/// the nest this replaces.
#[cube]
#[allow(clippy::too_many_arguments)]
fn hrr_chain<F: Float>(
    g: &mut Slice<F, ReadWrite>,
    base: u32,
    a_max: u32,
    m_max: u32,
    sa: u32,
    sb: u32,
    coef: F,
    r_lo: u32,
    r_hi: u32,
    #[comptime] nr: u32,
) {
    let mut a = 1u32;
    while a <= a_max {
        let arow = base + a * sa;
        let mut b = 0u32;
        while b <= (m_max - a) {
            let pb = arow + b * sb;
            if comptime!(nr > 0u32) {
                #[unroll]
                for r in 0..nr {
                    let idx = pb + r_lo + r;
                    g[idx as usize] = coef * g[(idx - sa) as usize] + g[(idx - sa + sb) as usize];
                }
            } else {
                let mut r = r_lo;
                while r < r_hi {
                    let idx = pb + r;
                    g[idx as usize] = coef * g[(idx - sa) as usize] + g[(idx - sa + sb) as usize];
                    r += 1u32;
                }
            }
            b += 1u32;
        }
        a += 1u32;
    }
}

/// [`hrr_chain`] at the one width a cooperative task actually takes.
///
/// A task owns a single root wherever the lanes of a sub-group share the root
/// axis (`rw == 1`), which is every cooperative launch the planner makes, and
/// the whole order only in the `coop=lane0` A/B arm and where a sub-group is
/// one lane. So the ladder here is **not** the five-arm one F1 (§15) put on
/// the Rys solvers: emitting widths two to five would compile four copies of
/// the chain nest per raise site that no dispatch ever enters, and a program
/// that carries five Rys solvers has no code budget to spend on dead unrolls.
/// §23.2 measured that: the five-arm form cost 16% on DZVP-MOLOPT-SR.
///
/// Width one is unrolled — it is the straight-line body the recurrence wants,
/// with the compare, the add and the backward branch of the old `while r <
/// r_hi` gone — and everything else takes the loop it always had.
#[cube]
#[allow(clippy::too_many_arguments)]
fn hrr_chain_w<F: Float>(
    g: &mut Slice<F, ReadWrite>,
    base: u32,
    a_max: u32,
    m_max: u32,
    sa: u32,
    sb: u32,
    coef: F,
    r_lo: u32,
    r_hi: u32,
) {
    if r_hi - r_lo == 1u32 {
        hrr_chain::<F>(g, base, a_max, m_max, sa, sb, coef, r_lo, r_hi, 1u32);
    } else {
        hrr_chain::<F>(g, base, a_max, m_max, sa, sb, coef, r_lo, r_hi, 0u32);
    }
}

/// The contraction's per-element Rys sum, `Σ_r gx·gy·gz`, at a **comptime**
/// root count (F1, §15).
///
/// Before the launch-group fusion `nroots` was the kernel's comptime parameter
/// and this loop was `#[unroll]`ed into the contraction. Fusing the Rys orders
/// into one dispatch makes `nroots` a per-quartet *runtime* value, and a
/// dynamic trip count of one to five would put a compare and a branch on the
/// hottest three-load-two-multiply statement the kernel has. So the loop keeps
/// its unrolled form and the caller selects the width with one branch per
/// element instead — a branch that is perfectly predicted, because every
/// element of a quartet shares its `nroots`.
///
/// The accumulation starts at zero and adds the roots in ascending order, which
/// is what the `#[unroll]`ed loop did; the result is bit-identical.
#[cube]
fn root_dot<F: Float>(
    g: &Slice<F, ReadWrite>,
    ax: u32,
    ay: u32,
    az: u32,
    #[comptime] width: u32,
) -> F {
    let mut sum = F::new(0.0_f32);
    #[unroll]
    for r in 0..width {
        sum += g[(ax + r) as usize] * g[(ay + r) as usize] * g[(az + r) as usize];
    }
    sum
}

/// The contraction's element walk for one primitive quartet, at a **comptime**
/// root width (plan §23).
///
/// F1 (§15) made `nroots` a runtime column of the class row and kept
/// [`root_dot`] unrolled by selecting its width with "one branch per element".
/// That branch sat *inside* the walk over the block's Cartesian elements —
/// eighty-one of them on a TZVP-MOLOPT `(pp|pp)` — so the hottest loop the
/// kernel has carried a five-way ladder that no backend can hoist for itself:
/// `ctr` and `cart_out` are kernel arguments, and a compiler that cannot prove
/// them unaliased cannot lift a branch across their stores. On the per-unit
/// shape the contraction is half the kernel (`probe:no-ctr` 2.0x–2.2x), and a
/// branch in the body is also what stops the walk vectorising.
///
/// The ladder moves to the caller ([`contract_block_elems_w`]), which takes it
/// once per primitive quartet, and the body left behind is straight-line. The
/// coefficient bases — `coff + p * nctr`, a property of the *primitive
/// quartet* — come out of the walk with it.
///
/// `arm` is the accumulation the quartet uses, decided once where `use_acc`,
/// `is_uncontracted` and `use_staged` are: `0` the private accumulator (S2),
/// `1` the segmented read-modify-write, `2` libcint's staged `i` stage (C1),
/// `3` the naive every-quad arm that `CINTX_2E_CONTRACT=naive` selects. Same
/// expressions, same operands, same order — bit-identical to the ladder-inside
/// form.
#[cube]
#[allow(clippy::too_many_arguments)]
fn contract_block_elems<F: Float>(
    g_slab: &Slice<F, ReadWrite>,
    class_idx: &Array<u32>,
    coeffs: &Array<F>,
    acc: &mut Slice<F, ReadWrite>,
    ctr: &mut Array<F>,
    cart_out: &mut Array<F>,
    idx_off: u32,
    gb: u32,
    g_size: u32,
    block_len: u32,
    q_start: u32,
    lanes: u32,
    weight: F,
    out_off: u32,
    ctr_i_off: u32,
    mb: u32,
    iempty: u32,
    arm: u32,
    coff_i: u32,
    pi: u32,
    nctr_i: u32,
    coff_j: u32,
    pj: u32,
    nctr_j: u32,
    coff_k: u32,
    pk: u32,
    nctr_k: u32,
    coff_l: u32,
    pl: u32,
    nctr_l: u32,
    #[comptime] shared_tier: u32,
    #[comptime] nr: u32,
) {
    let gby = gb + g_size;
    let gbz = gb + 2u32 * g_size;
    let cbi = coff_i + pi * nctr_i;
    let cbj = coff_j + pj * nctr_j;
    let cbk = coff_k + pk * nctr_k;
    let cbl = coff_l + pl * nctr_l;
    let mut q_elem = q_start;
    let mut acc_slot: u32 = 0u32;
    while q_elem < block_len {
        let t = idx_off + 3u32 * q_elem;
        let ax = gb + class_idx[t as usize];
        let ay = gby + class_idx[(t + 1u32) as usize];
        let az = gbz + class_idx[(t + 2u32) as usize];
        let sum = root_dot::<F>(g_slab, ax, ay, az, nr);
        if arm == 0u32 {
            // Read-modify-write rather than `+=`: a `Slice` index does not
            // carry the compound-assignment expansion an `Array` does. Same
            // operands, same order.
            if iempty == 1u32 {
                acc[acc_slot as usize] = weight * sum;
            } else {
                let prev = acc[acc_slot as usize];
                acc[acc_slot as usize] = prev + weight * sum;
            }
        } else if arm == 1u32 {
            if iempty == 1u32 {
                cart_out[(out_off + q_elem) as usize] = weight * sum;
            } else {
                cart_out[(out_off + q_elem) as usize] += weight * sum;
            }
        } else if arm == 2u32 {
            // The `i` stage: this primitive quartet into `gctri[ci][q]`.
            let w = weight * sum;
            let mut ci = 0u32;
            while ci < nctr_i {
                let mut cvi = F::new(1.0_f32);
                if nctr_i > 1u32 {
                    cvi = coeffs[(cbi + ci) as usize];
                }
                // The meta row carries the first four (§22.5); wider shells
                // reload.
                if comptime!(shared_tier > 0u32) {
                    if ci < 4u32 {
                        cvi = g_slab[(mb + 4u32 + ci) as usize];
                    }
                }
                let idx = ctr_i_off + ci * block_len + q_elem;
                if iempty == 1u32 {
                    ctr[idx as usize] = cvi * w;
                } else {
                    ctr[idx as usize] += cvi * w;
                }
                ci += 1u32;
            }
        } else {
            naive_quad_accumulate::<F>(
                coeffs, cart_out, sum, out_off, q_elem, block_len, cbi, nctr_i, cbj, nctr_j, cbk,
                nctr_k, cbl, nctr_l,
            );
        }
        q_elem += lanes;
        acc_slot += 1u32;
    }
}

/// The naive contraction arm: this element into every one of the quartet's
/// `nctr_i·nctr_j·nctr_k·nctr_l` output blocks.
///
/// Its own function so it is compiled **once** rather than in each of
/// [`contract_block_elems`]'s five comptime widths. It is the
/// `CINTX_2E_CONTRACT=naive` A/B arm and no default run enters it, so the code
/// it costs a program is code spent on nothing; the staged arm (C1) is what
/// every default dispatch takes. §23.2 is where that mattered.
#[cube]
#[allow(clippy::too_many_arguments)]
fn naive_quad_accumulate<F: Float>(
    coeffs: &Array<F>,
    cart_out: &mut Array<F>,
    sum: F,
    out_off: u32,
    q_elem: u32,
    block_len: u32,
    cbi: u32,
    nctr_i: u32,
    cbj: u32,
    nctr_j: u32,
    cbk: u32,
    nctr_k: u32,
    cbl: u32,
    nctr_l: u32,
) {
    let mut ci = 0u32;
    while ci < nctr_i {
        let cvi = coeffs[(cbi + ci) as usize];
        let mut cj = 0u32;
        while cj < nctr_j {
            let cvj = coeffs[(cbj + cj) as usize];
            let mut ck = 0u32;
            while ck < nctr_k {
                let cvk = coeffs[(cbk + ck) as usize];
                let mut cl = 0u32;
                while cl < nctr_l {
                    let cvl = coeffs[(cbl + cl) as usize];
                    let w4 = cvi * cvj * cvk * cvl;
                    let qbase = (((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl) * block_len;
                    let oidx = out_off + qbase + q_elem;
                    cart_out[oidx as usize] += w4 * sum;
                    cl += 1u32;
                }
                ck += 1u32;
            }
            cj += 1u32;
        }
        ci += 1u32;
    }
}

/// [`contract_block_elems`] with the root width chosen once per primitive
/// quartet.
///
/// The ladder F1 (§15) put on the Rys solvers, on [`root_dot`] and on
/// [`hrr_chain_w`], hoisted out of the element walk: a quartet's `nroots` is
/// the same for every element of its block, so the test belongs here and not
/// eighty-one levels in. Only the widths this dispatch can carry are emitted.
/// Above order five the orders are never fused, so `nroots == nr_max`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn contract_block_elems_w<F: Float>(
    g_slab: &Slice<F, ReadWrite>,
    class_idx: &Array<u32>,
    coeffs: &Array<F>,
    acc: &mut Slice<F, ReadWrite>,
    ctr: &mut Array<F>,
    cart_out: &mut Array<F>,
    idx_off: u32,
    gb: u32,
    g_size: u32,
    block_len: u32,
    q_start: u32,
    lanes: u32,
    weight: F,
    out_off: u32,
    ctr_i_off: u32,
    mb: u32,
    iempty: u32,
    arm: u32,
    coff_i: u32,
    pi: u32,
    nctr_i: u32,
    coff_j: u32,
    pj: u32,
    nctr_j: u32,
    coff_k: u32,
    pk: u32,
    nctr_k: u32,
    coff_l: u32,
    pl: u32,
    nctr_l: u32,
    nroots: u32,
    #[comptime] shared_tier: u32,
    #[comptime] nr_max: u32,
) {
    if comptime!(nr_max > 5u32) {
        contract_block_elems::<F>(
            g_slab,
            class_idx,
            coeffs,
            acc,
            ctr,
            cart_out,
            idx_off,
            gb,
            g_size,
            block_len,
            q_start,
            lanes,
            weight,
            out_off,
            ctr_i_off,
            mb,
            iempty,
            arm,
            coff_i,
            pi,
            nctr_i,
            coff_j,
            pj,
            nctr_j,
            coff_k,
            pk,
            nctr_k,
            coff_l,
            pl,
            nctr_l,
            shared_tier,
            comptime!(nr_max),
        );
    } else if nroots == 1u32 {
        contract_block_elems::<F>(
            g_slab,
            class_idx,
            coeffs,
            acc,
            ctr,
            cart_out,
            idx_off,
            gb,
            g_size,
            block_len,
            q_start,
            lanes,
            weight,
            out_off,
            ctr_i_off,
            mb,
            iempty,
            arm,
            coff_i,
            pi,
            nctr_i,
            coff_j,
            pj,
            nctr_j,
            coff_k,
            pk,
            nctr_k,
            coff_l,
            pl,
            nctr_l,
            shared_tier,
            1u32,
        );
    } else if nroots == 2u32 {
        if comptime!(nr_max >= 2u32) {
            contract_block_elems::<F>(
                g_slab,
                class_idx,
                coeffs,
                acc,
                ctr,
                cart_out,
                idx_off,
                gb,
                g_size,
                block_len,
                q_start,
                lanes,
                weight,
                out_off,
                ctr_i_off,
                mb,
                iempty,
                arm,
                coff_i,
                pi,
                nctr_i,
                coff_j,
                pj,
                nctr_j,
                coff_k,
                pk,
                nctr_k,
                coff_l,
                pl,
                nctr_l,
                shared_tier,
                2u32,
            );
        }
    } else if nroots == 3u32 {
        if comptime!(nr_max >= 3u32) {
            contract_block_elems::<F>(
                g_slab,
                class_idx,
                coeffs,
                acc,
                ctr,
                cart_out,
                idx_off,
                gb,
                g_size,
                block_len,
                q_start,
                lanes,
                weight,
                out_off,
                ctr_i_off,
                mb,
                iempty,
                arm,
                coff_i,
                pi,
                nctr_i,
                coff_j,
                pj,
                nctr_j,
                coff_k,
                pk,
                nctr_k,
                coff_l,
                pl,
                nctr_l,
                shared_tier,
                3u32,
            );
        }
    } else if nroots == 4u32 {
        if comptime!(nr_max >= 4u32) {
            contract_block_elems::<F>(
                g_slab,
                class_idx,
                coeffs,
                acc,
                ctr,
                cart_out,
                idx_off,
                gb,
                g_size,
                block_len,
                q_start,
                lanes,
                weight,
                out_off,
                ctr_i_off,
                mb,
                iempty,
                arm,
                coff_i,
                pi,
                nctr_i,
                coff_j,
                pj,
                nctr_j,
                coff_k,
                pk,
                nctr_k,
                coff_l,
                pl,
                nctr_l,
                shared_tier,
                4u32,
            );
        }
    } else {
        if comptime!(nr_max >= 5u32) {
            contract_block_elems::<F>(
                g_slab,
                class_idx,
                coeffs,
                acc,
                ctr,
                cart_out,
                idx_off,
                gb,
                g_size,
                block_len,
                q_start,
                lanes,
                weight,
                out_off,
                ctr_i_off,
                mb,
                iempty,
                arm,
                coff_i,
                pi,
                nctr_i,
                coff_j,
                pj,
                nctr_j,
                coff_k,
                pk,
                nctr_k,
                coff_l,
                pl,
                nctr_l,
                shared_tier,
                5u32,
            );
        }
    }
}

/// The vector VRR arm for one primitive quartet: the three axes' 2D
/// recurrences with the Rys root index folded into `N` vector lanes (V1, §12).
///
/// Lifted out of the kernel body by the launch-group fusion (§15). A `Vector`
/// width is a *type-level* size, so it cannot follow a runtime `nroots`; the
/// widths this arm serves are instantiated here as `Const<2..5>` and the kernel
/// picks one with a runtime branch per primitive quartet. That is why the
/// `#[define(N)]` dynamic size the kernel used to carry is gone: the width now
/// reaches `vrr_fill_axis_roots` from the concrete type at each call site.
///
/// Statement for statement the block it replaces, so the slab it writes is
/// bit-identical to both the scalar arm and the pre-fusion vector arm.
#[cube]
#[allow(clippy::too_many_arguments)]
fn vrr_build_axes_roots<F: Float, N: Size>(
    g_slab: &mut Slice<F, ReadWrite>,
    urys: &Slice<F, ReadWrite>,
    wrys: &Slice<F, ReadWrite>,
    gx_off: u32,
    g_size: u32,
    nmax: u32,
    mmax: u32,
    g2d_ijmax: u32,
    g2d_klmax: u32,
    a0: F,
    aij: F,
    akl: F,
    a1: F,
    fac1: F,
    xij_kl: F,
    yij_kl: F,
    zij_kl: F,
    rijrxx: F,
    rijrxy: F,
    rijrxz: F,
    rklrxx: F,
    rklrxy: F,
    rklrxz: F,
    #[comptime] width: usize,
) {
    let u2 = Vector::<F, N>::new(a0) * roots_load::<F, N>(urys, 0u32, width);
    let tmp4 = Vector::<F, N>::new(F::new(0.5_f32))
        / (u2 * Vector::<F, N>::new(aij + akl) + Vector::<F, N>::new(a1));
    let tmp5 = u2 * tmp4;
    let tmp1 = Vector::<F, N>::new(F::new(2.0_f32)) * tmp5;
    let tmp2 = tmp1 * Vector::<F, N>::new(akl);
    let tmp3 = tmp1 * Vector::<F, N>::new(aij);
    let b00 = tmp5;
    let b10 = tmp5 + tmp4 * Vector::<F, N>::new(akl);
    let b01 = tmp5 + tmp4 * Vector::<F, N>::new(aij);

    #[unroll]
    for axis in 0..3u32 {
        let off = gx_off + axis * g_size;
        let mut xkl = xij_kl;
        let mut rijrx = rijrxx;
        let mut rklrx = rklrxx;
        if axis == 1u32 {
            xkl = yij_kl;
            rijrx = rijrxy;
            rklrx = rklrxy;
        } else if axis == 2u32 {
            xkl = zij_kl;
            rijrx = rijrxz;
            rklrx = rklrxz;
        }
        let c00 = Vector::<F, N>::new(rijrx) - tmp2 * Vector::<F, N>::new(xkl);
        let c0p = Vector::<F, N>::new(rklrx) + tmp3 * Vector::<F, N>::new(xkl);

        // `gx`/`gy` seed at one, `gz` at the root's Rys weight times the
        // primitive's scale factor — the same three values the scalar arm
        // writes, by the same rule. Statement-level mutation rather than a
        // value-returning `if`, for the reason the scalar arm records.
        let mut seed = Vector::<F, N>::new(F::new(1.0_f32));
        if axis == 2u32 {
            seed = roots_load::<F, N>(wrys, 0u32, width) * Vector::<F, N>::new(fac1);
        }

        vrr_fill_axis_roots::<F, N>(
            g_slab, off, nmax, mmax, g2d_ijmax, g2d_klmax, c00, c0p, b00, b10, b01, seed, width,
        );
    }
}

/// Batched scalar 2e kernel — one cube per shell quartet (Task 34-B).
///
/// The kernel evaluates a whole **launch group** in one dispatch. A group is
/// every quartet sharing the kernel's three comptime parameters — `ibase`,
/// `kbase` and `nroots`, i.e. the HRR branch and the Rys order. The G-tensor
/// extents are *runtime* scalars, so several `(li,lj,lk,ll)` classes coexist in
/// one dispatch, each carrying its own shape row (Task 35-M1). Everything that
/// varies is read through flat arrays plus an index table:
///
/// - `exps` / `coeffs` — every shell's primitives concatenated;
/// - `centers` — 3 floats per shell;
/// - `shell_meta` — 4 `u32` per shell: `[exp_off, coeff_off, nprim, nctr]`;
/// - `quartets` — 6 `u32` per quartet: `[si, sj, sk, sl, out_off, class]`;
/// - `class_shape` — [`TWO_E_SHAPE_STRIDE`] `u32` per class: `li,lj,lk,ll,
///   di,dk,dl,dj,g_size,nmax,mmax,g2d_ijmax,g2d_klmax`;
/// - `class_factor` — one `common_factor` per class.
///
/// `g` is a per-slot slab of `3 * g_size_max` over the group, bound only by
/// a dispatch whose `shared_tier` is `0` (B1, §22); the Rys roots and weights are
/// **kernel-local** arrays (every read of them sits inside the same
/// `lane == 0` region that writes them), so they need neither a buffer nor
/// a per-slot offset.
///
/// # Two decompositions, selected by the comptime `per_unit` flag
///
/// A *slot* is one work-item of the quartet grid-stride; a *lane* is one unit
/// inside the cooperative group that shares a slot.
///
/// - `per_unit == 0` — **one quartet per cube**, the whole cube cooperating on
///   it: the G-tensor build runs on lane 0 and the contraction is split
///   `q_elem % lanes == lane`, with two `sync_cube()` barriers per primitive
///   quartet. This is the shape GPU backends want, where `sync_cube` is a
///   workgroup barrier and `cube_count` is real parallelism.
/// - `per_unit == 1` — **one quartet per unit**: `lanes == 1`, `lane == 0`, so
///   every unit runs a whole quartet alone in its own G slab and *no barrier is
///   reachable*. This is the shape the CubeCL CPU runtime wants, where a unit
///   is an OS thread, `sync_cube` is a global spin-wait, and `cube_count`
///   lowers to a sequential `scf.for` inside each unit — i.e. the cube, not the
///   grid, is the only parallelism axis there (Task 34-A0's finding).
///
/// The barrier must be comptime-removed rather than merely skipped: under
/// `per_unit == 1` units walk *different* quartets, so their trip counts differ
/// and any barrier inside the quartet loop is divergent.
///
/// # Every way the two arms differ, and why each one has to (§18)
///
/// The rule is that CPU and GPU run the *same method*: the same grouping, the
/// same ket-pair split, the same partition, the same arithmetic in the same
/// order. Everything below is a consequence of one fact — the CubeCL CPU
/// runtime makes a unit an OS thread and `cube_count` a sequential loop, so the
/// grid is not a parallelism axis there — and nothing below is a policy choice.
/// **The two arms are bit-identical to each other**, which is what
/// `two_e_cooperative_arm::cooperative_g_build_is_bit_identical_on_*` pins on
/// the CPU backend, where both can be run.
///
/// | difference | forced by |
/// |---|---|
/// | `sync_cube()` present or comptime-removed | units walk different quartets under `per_unit`, so a barrier in the quartet loop is divergent |
/// | the G build split `task % build_lanes` (S3) | there is one lane per slot under `per_unit`; the residue test admits every task and collapses to the scalar loop |
/// | the contraction split `q_elem % lanes` | the same: `lanes == 1` |
/// | the vector VRR width (§12) | it folds the roots into lanes, which is free where a slot *is* one lane and costs S3's `3·nroots` tasks where a slot is a cube |
/// | [`acc_capacity`] 256 vs 64 slots | private storage is per work item, and a 256-wide cube holding 256 slots each would spill |
/// | cube width and count | `cooperative_cube_dim` sizes a cube from the Cartesian block; `per_unit_cube_dim` sizes a thread pool from the rows |
/// | the G tensor's home and the block of primitive quartets per cube (B1, §22) | shared memory is per cube, so the cooperative arm keeps the G tensor in a comptime-tiered `SharedMemory` region and builds `b_max` primitive quartets at once, one per lane sub-group; a unit has no cube to share with, so the per-unit arm keeps its global slab and a block of one — the contraction still walks the block in row order, so the sums are the same sums |
///
/// What is deliberately *not* on that list any more: which Rys orders share a
/// dispatch (F1, §15 — both fuse), whether a quartet's ket range is split (G1,
/// §16–18 — both do, by the same per-quartet rule), and whether a memory limit
/// changes the arithmetic (§18 — it does not, on either).
///
/// See the module note above for the comptime/runtime split.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// F1 (§15) selects a per-order arm with `if comptime!(nr_max >= k) { if nroots
// == k { … } }`. Collapsing the two into one `&&` would put a comptime
// condition and a runtime one in the same expression, which is not a shape this
// frontend folds reliably — the module note above records what that costs — and
// the nesting is what keeps the outer test a pure JIT-time elision.
#[allow(clippy::collapsible_if)]
fn two_electron_scalar_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    quartets: &Array<u32>,
    class_shape: &Array<u32>,
    class_factor: &Array<F>,
    class_idx: &Array<u32>,
    rys_tab: &Array<f64>,
    pair_data: &Array<F>,
    pair_index: &Array<u32>,
    pair_offset: &Array<u32>,
    slot_bounds: &Array<u32>,
    g: &mut Array<F>,
    ctr: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    prim_tol: F,
    expcutoff: F,
    nbas: u32,
    acc_slots_max: u32,
    n_quartets: u32,
    n_cubes: u32,
    g_stride: u32,
    ctr_stride: u32,
    ctr_mode: u32,
    coop_build: u32,
    #[comptime] ibase: u32,
    #[comptime] kbase: u32,
    // F1 (§15): the widest Rys order this dispatch carries, not *the* Rys
    // order. Every quartet's own `nroots` is a runtime value read from its
    // class row; `nr_max` is what the kernel's comptime shapes are sized to
    // and which per-order arms it emits at all — the Rys solvers, the vector
    // VRR widths and the contraction's unrolled root sum. It stays part of the
    // kernel's identity so a dispatch that needs a wider order compiles its
    // own program.
    #[comptime] nr_max: u32,
    #[comptime] per_unit: u32,
    #[comptime] shared_tier: u32,
    #[comptime] row_stride: u32,
) {
    let cube_pos = CUBE_POS as u32;

    // Slot / lane decomposition — see the doc comment above.
    //
    // `slot` is this work-item's index in the quartet grid-stride and doubles as
    // its private G-slab index; `n_slots` is the stride. `lane`/`lanes` describe
    // the cooperative group sharing one slot: the whole cube when
    // `per_unit == 0`, a group of one when `per_unit == 1`.
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    // `coop` is 1 in the cooperative decomposition and 0 in the per-unit one;
    // `punit` is its complement. The selection is written as arithmetic on these
    // two comptime-folded flags rather than as a `comptime!` if/else, because
    // the `UNIT_POS`/`CUBE_DIM` builtins expand to `NativeExpand<u32>`, which
    // will not unify with the `u32` literal the other arm would carry.
    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;

    // One slot per cube when cooperating, one slot per unit otherwise.
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    // `(lane, lanes)` collapses to `(0, 1)` in the per-unit decomposition — a
    // cooperative group of one, so every `lane == 0` guard admits every unit and
    // the `q_elem % lanes == lane` split is the identity.
    let lane = unit_pos * coop;
    let lanes = (cube_dim - 1u32) * coop + 1u32;

    // ── S3 / B1: who builds the G tensor ──────────────────────────────────
    //
    // Decided per quartet now, inside the loop below: the lanes of a cube are
    // cut into `b_max` sub-groups (B1, plan §22), one primitive quartet each,
    // and the S3 `(axis, root)` ownership — `split` by default, `lane0` for
    // the A/B — is taken *within* a sub-group. Both collapse to "one group of
    // one lane, owning everything" under the per-unit decomposition.

    // ── B1: the G tensor's home — a tiered shared-memory region, or the
    //        per-slot global slab ──────────────────────────────────────────
    //
    // The Rys recurrences read and write the G tensor in their innermost
    // loops, and every step of a recurrence depends on the one before it. In
    // the cooperative decomposition that slab lived in *global* memory, so a
    // `(pp|pp)` primitive quartet was a chain of a few dozen dependent
    // global-memory round trips on nine lanes of a ninety-six-lane cube
    // (plan §22.1). Shared memory is where such a chain belongs.
    //
    // `shared_tier` is the comptime extent of the shared G region in `f64`
    // slots, `0` for the global slab. It is *tiered* rather than sized to the
    // class ([`SHARED_G_TIERS`]): `SharedMemory::new` takes a comptime extent,
    // so a per-class size would be a program per class — the merge that
    // Task 35-M1 undid — while a single 48 KiB allocation admits one workgroup
    // per compute unit and was measured at 0.31x–0.57x (§22.1). A tier is the
    // smallest of three that holds the dispatch's widest class, and every
    // narrower class in the dispatch fills the rest of it with *more
    // primitive quartets in flight* (`b_max` below), which is what buys the
    // latency back.
    //
    // The region's first [`G_META_SLOTS`] entries carry, per primitive
    // quartet of the block in flight, the state its builder found it in
    // (screened out, under tolerance, live); the sub-slabs follow.
    //
    // `Slice<F, ReadWrite>` is what makes this a selection rather than a second
    // copy of the recurrences: `Array` and `SharedMemory` both produce one, so
    // the recurrences below are written once and bind whichever they were
    // given. Every access to the G tensor below must go through `g_slab`,
    // never through the `g` parameter: indexing `g` directly still compiles
    // and is correct only in the global configuration, where `g_slab` aliases
    // it. That is what `.planning/notes/rocm-shared-memory-miscompile.md`
    // records.
    let mut shared_g = SharedMemory::<F>::new(comptime!(if shared_tier > 0u32 {
        (shared_tier + G_META_SLOTS) as usize
    } else {
        1usize
    }));
    let mut g_slab = if comptime!(shared_tier > 0u32) {
        shared_g.to_slice_mut()
    } else {
        g.to_slice_mut()
    };

    // Shared memory is per cube, so the cooperative region starts after its
    // meta slots; the global slab is one per slot and is indexed. Written as
    // arithmetic on a comptime-folded flag rather than as a conditional
    // assignment, for the same reason `coop`/`punit` above are.
    let global_slab = if comptime!(shared_tier > 0u32) {
        0u32
    } else {
        1u32
    };
    let in_shared = 1u32 - global_slab;
    let slab_base = slot * g_stride * global_slab + comptime!(G_META_SLOTS) * in_shared;

    // Rys roots/weights are written and read entirely inside the `lane == 0`
    // region below, so they are per-unit private storage rather than buffers.
    // The extent follows `nroots`: five for the polynomial-fit kernels, and
    // exactly `nroots` once the inline extended entry (task 33-01) serves the
    // class. The caller's fail-closed guard is what keeps `nroots` inside
    // `device_nroots_ceiling(backend, RysFamily::Int2e)`.
    let mut urys = Array::<F>::new(comptime!(ext_rys_slots(nr_max)));
    let mut wrys = Array::<F>::new(comptime!(ext_rys_slots(nr_max)));
    // The extended entry is f64-only — its double-double arms are what buy the
    // accuracy — so it lands in its own pair and is cast into `urys`/`wrys`.
    // Both collapse to one element when the arm is not emitted.
    let mut uext = Array::<f64>::new(comptime!(ext_rys_out_slots(nr_max)));
    let mut wext = Array::<f64>::new(comptime!(ext_rys_out_slots(nr_max)));
    // S2's accumulator, hoisted out of the quartet loop: its capacity is
    // comptime, so it never depended on the quartet, and it is zeroed per
    // quartet below in any case.
    let mut acc_store = Array::<F>::new(comptime!(acc_capacity(per_unit)));
    // Handed to [`contract_block_elems`] as a slice, which is what a `#[cube]`
    // helper can take; every access below goes through it.
    let mut acc = acc_store.to_slice_mut();

    // Grid-stride over the quartet list: one quartet per slot when the grid is
    // wide enough, a strided sweep when it is capped. Under `per_unit == 0`
    // every unit of a cube walks the same `qi`, so the `sync_cube()` calls stay
    // convergent; under `per_unit == 1` the barriers are comptime-removed and
    // units are free to walk different quartets.
    //
    // The stride is derived from the launch argument `n_cubes`, not the
    // `CUBE_COUNT` builtin: `cubecl-cpu` 0.10 rejects that builtin outright
    // (`compiler/visitor/args_manager.rs`: "Unsupported builtin was used:
    // CubeCount"), and the host already knows the value it passed to
    // `CubeCount::Static`.
    //
    // Under `per_unit == 1` the walk is *blocked* rather than interleaved: each
    // unit takes one contiguous run of quartets. Neighbouring quartets write
    // neighbouring `cart_out` blocks, so an interleaved assignment would put
    // every unit's accumulation on the same handful of cache lines.
    // Same `coop`/`punit` arithmetic as above, for the same reason:
    //   per-unit  -> [bounds[slot], bounds[slot + 1])  step 1
    //   coop      -> [slot,         n_quartets)        step n_slots
    //
    // K2 (GTH plan §10): the per-unit ranges come from the host as
    // `slot_bounds` — `n_slots + 1` row indices, cost-balanced by default
    // (`per_unit_slot_bounds`) so a unit that draws the `(pp|pp)` rows of a
    // generally contracted class gets fewer of them than a unit drawing
    // `(ss|ss)`. `CINTX_2E_BALANCE=uniform` restores `ceil(n / n_slots)`
    // rows each. Which unit evaluates a quartet cannot change its value, so
    // the two partitions are bit-identical by construction. The index is
    // `slot * punit`, so the cooperative arm reads the two-element
    // placeholder it is handed and keeps its interleaved walk.
    let bidx = slot * punit;
    let qi_start = slot_bounds[bidx as usize] * punit + slot * coop;
    let mut qi_stop = slot_bounds[(bidx + 1u32) as usize] * punit + n_quartets * coop;
    if qi_stop > n_quartets {
        qi_stop = n_quartets;
    }
    let qi_step = n_slots * coop + punit;

    let mut qi = qi_start;
    while qi < qi_stop {
        // `row_stride` is comptime and part of the kernel's identity, so a
        // change to the row layout changes the compiled program's cache key
        // (the HIP cache is keyed by signature, not body: a body-only change
        // once ran a stale kernel against the new table and page-faulted).
        let qrow = qi * row_stride;
        let si = quartets[qrow as usize];
        let sj = quartets[(qrow + 1u32) as usize];
        let sk = quartets[(qrow + 2u32) as usize];
        let sl = quartets[(qrow + 3u32) as usize];
        let out_off = quartets[(qrow + 4u32) as usize];

        // ── Per-class shape (Task 35-M1) ──────────────────────────────────
        //
        // A launch class used to be one `(li,lj,lk,ll)` tuple, because every
        // shape scalar below was a launch argument. They are all *runtime*
        // scalars though — only `ibase`, `kbase` and `nroots` are comptime — so
        // one dispatch can carry every l-quartet that shares those three, with
        // the shape read per quartet from `class_shape`. On def2-SVP that takes
        // 69 launches down to 16.
        //
        // The G slab is sized to the widest `g_size` in the dispatch and each
        // class indexes only the first `3 * g_size` of it, so a narrow class
        // reads and writes exactly the elements it did when it launched alone —
        // which is why the merge is bit-identical, not merely close.
        let cls = quartets[(qrow + 5u32) as usize];
        let srow = cls * comptime!(TWO_E_SHAPE_STRIDE as u32);
        let li = class_shape[srow as usize];
        let lj = class_shape[(srow + 1u32) as usize];
        let lk = class_shape[(srow + 2u32) as usize];
        let ll = class_shape[(srow + 3u32) as usize];
        let di = class_shape[(srow + 4u32) as usize];
        let dk = class_shape[(srow + 5u32) as usize];
        let dl = class_shape[(srow + 6u32) as usize];
        let dj = class_shape[(srow + 7u32) as usize];
        let g_size = class_shape[(srow + 8u32) as usize];
        let nmax = class_shape[(srow + 9u32) as usize];
        let mmax = class_shape[(srow + 10u32) as usize];
        let g2d_ijmax = class_shape[(srow + 11u32) as usize];
        let g2d_klmax = class_shape[(srow + 12u32) as usize];
        // K1: where this class's Cartesian index table starts in `class_idx`.
        let idx_off = class_shape[(srow + 13u32) as usize];
        // F1 (§15): this class's Rys order. A *runtime* scalar since the
        // launch-group fusion — the dispatch carries every order up to
        // `nr_max`, and which one a quartet takes is a property of its class,
        // not of the compiled program.
        let nroots = class_shape[(srow + 14u32) as usize];
        let common_factor = class_factor[cls as usize];

        let nfi = (li + 1u32) * (li + 2u32) / 2u32;
        let nfj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let nfk = (lk + 1u32) * (lk + 2u32) / 2u32;
        let nfl = (ll + 1u32) * (ll + 2u32) / 2u32;
        let block_len = nfi * nfj * nfk * nfl;

        let mi = si * 4u32;
        let eoff_i = shell_meta[mi as usize];
        let coff_i = shell_meta[(mi + 1u32) as usize];
        let nctr_i = shell_meta[(mi + 3u32) as usize];
        let mj = sj * 4u32;
        let eoff_j = shell_meta[mj as usize];
        let coff_j = shell_meta[(mj + 1u32) as usize];
        let nctr_j = shell_meta[(mj + 3u32) as usize];
        let mk = sk * 4u32;
        let eoff_k = shell_meta[mk as usize];
        let coff_k = shell_meta[(mk + 1u32) as usize];
        let nctr_k = shell_meta[(mk + 3u32) as usize];
        let ml2 = sl * 4u32;
        let eoff_l = shell_meta[ml2 as usize];
        let coff_l = shell_meta[(ml2 + 1u32) as usize];
        let nctr_l = shell_meta[(ml2 + 3u32) as usize];

        let ci3 = si * 3u32;
        let rix = centers[ci3 as usize];
        let riy = centers[(ci3 + 1u32) as usize];
        let riz = centers[(ci3 + 2u32) as usize];
        let cj3 = sj * 3u32;
        let rjx = centers[cj3 as usize];
        let rjy = centers[(cj3 + 1u32) as usize];
        let rjz = centers[(cj3 + 2u32) as usize];
        let ck3 = sk * 3u32;
        let rkx = centers[ck3 as usize];
        let rky = centers[(ck3 + 1u32) as usize];
        let rkz = centers[(ck3 + 2u32) as usize];
        let cl3 = sl * 3u32;
        let rlx = centers[cl3 as usize];
        let rly = centers[(cl3 + 1u32) as usize];
        let rlz = centers[(cl3 + 2u32) as usize];

        let out_len = nctr_i * nctr_j * nctr_k * nctr_l * block_len;

        let is_uncontracted =
            (nctr_i == 1u32) && (nctr_j == 1u32) && (nctr_k == 1u32) && (nctr_l == 1u32);

        // ── Staged general contraction (GTH plan, C1) ─────────────────────
        //
        // A generally contracted quartet used to fold every primitive quartet
        // into every one of its `nctr_i·nctr_j·nctr_k·nctr_l` output blocks —
        // 81 read-modify-writes of `cart_out` per element per primitive
        // quartet on a TZVP-MOLOPT `(pp|pp)`. libcint's staging
        // (`cint2e.c:193-262`) is reproduced instead: see [`stage_contract`].
        // The three intermediates live in this slot's `ctr` slab, laid out
        // `gctri[ci][q]`, `gctrj[cj][ci][q]`, `gctrk[ck][cj][ci][q]`, `q` the
        // Cartesian element. `ctr_mode` is a runtime scalar so both settings
        // are one compiled program (`CINTX_2E_CONTRACT=naive` is the A/B).
        //
        // The `*empty` flags and `prev_p*` are libcint's own: a stage is
        // *assigned* on its first write after being emptied and accumulated
        // after, and a stage is flushed when the primitive that owns it
        // changes. The pair rows are ordered `(pl, pk)` and `(pj, pi)` by
        // `PairTable::push_shell_pair`, which is what makes "the primitive
        // changed" detectable on a compacted list. Every quantity here is
        // cube-uniform (every lane walks the same rows), so the branches stay
        // convergent.
        // Modes 3 and 4 are attribution probes on the *staged* path (see
        // `set_contraction_mode`): they keep its bookkeeping and skip a phase.
        let use_staged = (!is_uncontracted) && (ctr_mode == 1u32 || ctr_mode >= 3u32);
        // `lane`/`lanes` carry the builtin's `NativeExpand` type, which does
        // not unify with a cube helper's `u32` parameter; rebased onto plain
        // values here, once per quartet.
        let mut lane_u = 0u32;
        lane_u += lane;
        let mut lanes_u = 0u32;
        lanes_u += lanes;
        let leni = nctr_i * block_len;
        let lenj = leni * nctr_j;
        let ctr_i_off = slot * ctr_stride;
        let ctr_j_off = ctr_i_off + leni;
        let ctr_k_off = ctr_j_off + lenj;
        let mut kempty = 1u32;
        let mut prev_pl = 0xFFFF_FFFFu32;

        // ── S2: accumulate this quartet in private storage where it fits ──
        //
        // The contraction sum used to land in `cart_out` — a kernel argument,
        // so a pointer the compiler cannot prove unaliased — once per primitive
        // quartet per element. A `(ss|ss)` class on SO2/def2-TZVP walks up to
        // 2 401 primitive quartets, so its single output element was loaded and
        // stored 2 401 times through memory the optimizer had to reload.
        //
        // A lane owns the elements where `q_elem % lanes == lane`, so it needs
        // `ceil(block_len / lanes)` slots. `ACC_SLOTS` covers every class up to
        // `nroots = 3` in the per-unit decomposition — the classes carrying
        // 93-98% of all primitive work — and every class at all once a
        // cooperative cube splits the block across its lanes. Anything wider
        // falls back to the original read-modify-write, so the ceiling bounds
        // stack use without bounding what the kernel can evaluate.
        //
        // Accumulation order is unchanged: the same primitive quartets are
        // summed into the same element in the same sequence, just in a
        // different place. The result is bit-identical by construction, which
        // is what `CINTX_2E_ACCUMULATE=global` exists to check.
        let acc_slots = (out_len + lanes - 1u32) / lanes;
        // Two ceilings, and both are real: the comptime one is what the array
        // can hold on this decomposition, the runtime one is the A/B switch.
        let use_acc = is_uncontracted
            && (acc_slots <= acc_slots_max)
            && (acc_slots <= comptime!(acc_capacity(per_unit) as u32));
        if use_acc {
            let mut ai = 0u32;
            while ai < acc_slots {
                acc[ai as usize] = F::new(0.0_f32);
                ai += 1u32;
            }
        }
        // libcint's `gempty`, at quartet scope: the first contributing
        // primitive quartet *assigns* into the accumulator (or the output) and
        // only later ones add. `0.0 + x == x` for every `x` except `-0.0`,
        // where it gives `+0.0`, so zero-then-always-accumulate silently loses
        // the sign of a zero the vendor keeps. `iempty` cannot serve here: it
        // is reset per ket primitive pair for the staged arm, while the
        // accumulator's lifetime is the whole quartet.
        let mut qempty: u32 = 1u32;

        // Zero this quartet's output block across the slot's lanes. Skipped
        // under the accumulator, which writes every element it owns exactly
        // once at the flush below rather than adding into it.
        if !use_acc {
            let mut oi = lane;
            while oi < out_len {
                cart_out[(out_off + oi) as usize] = F::new(0.0_f32);
                oi += lanes;
            }
        }
        // Which accumulation this quartet's element walk takes, decided
        // here beside the flags it mirrors rather than re-tested per element
        // (§23): `0` the private accumulator, `1` the segmented
        // read-modify-write, `2` the staged `i` stage, `3` the naive arm.
        let mut ctr_arm = 3u32;
        if use_acc {
            ctr_arm = 0u32;
        } else if is_uncontracted {
            ctr_arm = 1u32;
        } else if use_staged {
            ctr_arm = 2u32;
        }
        if comptime!(per_unit == 0u32) {
            sync_cube();
        }

        // ibase/kbase-selected reference centers and the HRR displacement
        // vectors rirj / rkrl: properties of the quartet's centres, so they
        // are formed once here rather than once per primitive quartet.
        let mut rx_ij_x = rjx;
        let mut rx_ij_y = rjy;
        let mut rx_ij_z = rjz;
        let mut rirjx = rjx - rix;
        let mut rirjy = rjy - riy;
        let mut rirjz = rjz - riz;
        if comptime!(ibase == 1u32) {
            rx_ij_x = rix;
            rx_ij_y = riy;
            rx_ij_z = riz;
            rirjx = rix - rjx;
            rirjy = riy - rjy;
            rirjz = riz - rjz;
        }
        let mut rx_kl_x = rlx;
        let mut rx_kl_y = rly;
        let mut rx_kl_z = rlz;
        let mut rkrlx = rlx - rkx;
        let mut rkrly = rly - rky;
        let mut rkrlz = rlz - rkz;
        if comptime!(kbase == 1u32) {
            rx_kl_x = rkx;
            rx_kl_y = rky;
            rx_kl_z = rkz;
            rkrlx = rkx - rlx;
            rkrly = rky - rly;
            rkrlz = rkz - rlz;
        }

        // ── B1: primitive quartets in flight per cube (plan §22) ──────────
        //
        // A cube's lanes are cut into `b_max` sub-groups of `sg` lanes. Each
        // sub-group builds one primitive quartet's G tensor into its own
        // sub-slab of the shared region, `b_max` of them at once, and then the
        // whole cube contracts the block in *row order* — the same primitive
        // quartets summed into the same elements in the same sequence, so the
        // result is bit-identical to the one-at-a-time walk. `b_max` is bounded
        // by what the tier holds, by giving every `(axis, root)` VRR task of a
        // sub-group its own lane, and by [`B_MAX`]. It is one wherever the G
        // tensor is global — the per-unit decomposition, and a class too wide
        // for any tier — which is exactly the shape that was there before.
        //
        // Within a sub-group the S3 ownership is `task % own_lanes == own_lane`
        // over the sub-group's lanes (`lane0` mode: sub-lane 0 owns every task).
        // `builds` is whether this lane takes any task at all: not under
        // `lane0` off sub-lane 0, and not in a sub-group past `b_max` — the
        // lanes left over when `lanes` is not a multiple of `sg`.
        let g3 = 3u32 * g_size;
        let tier_slots = comptime!(shared_tier);
        let b_cap = comptime!(B_MAX);
        let mut b_max: u32 = 1u32;
        if comptime!(shared_tier > 0u32) {
            b_max = tier_slots / g3;
            let by_lanes = lanes_u / (3u32 * nroots);
            if by_lanes < b_max {
                b_max = by_lanes;
            }
            if b_max > b_cap {
                b_max = b_cap;
            }
            if b_max < 1u32 {
                b_max = 1u32;
            }
        }
        let sg = lanes_u / b_max;
        let sub = lane_u / sg;
        let sub_lane = lane_u - sub * sg;
        let mut builds: u32 = 1u32;
        let mut own_lane = sub_lane;
        let mut own_lanes = sg;
        if coop_build == 0u32 {
            own_lane = 0u32;
            own_lanes = 1u32;
            if sub_lane != 0u32 {
                builds = 0u32;
            }
        }
        if sub >= b_max {
            builds = 0u32;
        }
        // Roots per build task. On a group of one lane — the per-unit arm,
        // and `lane0` — a task takes every root, so the innermost loop of
        // every chain is the contiguous root index it always was on the CPU
        // (§22.5 measured the strided form at 1.2x–1.3x there); where lanes
        // share a sub-group a task is one root, the finer split.
        let mut rw = 1u32;
        if own_lanes == 1u32 {
            rw = nroots;
        }
        let nrg = (nroots + rw - 1u32) / rw;
        // This sub-group's build sub-slab, and its three axis planes.
        let g_off = slab_base + sub * g3;

        // ── The owned-task decode, hoisted to the quartet (§23) ───────────
        //
        // Each of the three phases below deals its tasks out as
        // `own_lane, own_lane + own_lanes, …` and decodes the *first* one into
        // its `(axis, root group, chain index)` coordinates, carrying the rest.
        // That decode is seven unsigned divisions, and every operand of it —
        // `nrg`, `own_lane`, `builds`, and the class's `l`/`d`/`nmax`/`mmax`
        // scalars — is fixed for the whole quartet, while the code sat inside
        // the ket/bra loops and so ran once per *block of primitive quartets*.
        // For a class too wide for any shared tier a block is one primitive
        // quartet, which made it seven divisions per primitive quartet on the
        // longest rows in the list — the rows that set the launch floor. No
        // backend has an integer divide instruction; each one is a
        // reciprocal sequence of a dozen or more.
        //
        // Same values, computed once. `builds == 0` still parks a lane past
        // the end of its task list, which is what makes it skip the phase.
        let n_vrr = 3u32 * nrg;
        let mut vrr_task0 = own_lane;
        if builds == 0u32 {
            vrr_task0 = n_vrr;
        }
        let vrr_axis0 = vrr_task0 / nrg;
        let vrr_rg0 = vrr_task0 - vrr_axis0 * nrg;

        // The two HRR raises are the cooperative arm's alone; on the per-unit
        // arm the committed nest runs instead and these fold away with it.
        let mut n_a = 0u32;
        let mut a_task0 = 0u32;
        let mut a_axis0 = 0u32;
        let mut a_rg0 = 0u32;
        let mut a_p0 = 0u32;
        let mut n_b = 0u32;
        let mut b_task0 = 0u32;
        let mut b_axis0 = 0u32;
        let mut b_rg0 = 0u32;
        let mut b_p10 = 0u32;
        let mut b_p20 = 0u32;
        let mut pa = 0u32;
        let mut w1 = 0u32;
        let mut w2 = 0u32;
        if comptime!(per_unit == 0u32) {
            pa = nmax + 1u32;
            if comptime!(ibase == 0u32) {
                pa = mmax + 1u32;
            }
            n_a = 3u32 * nrg * pa;
            a_task0 = own_lane;
            if builds == 0u32 {
                a_task0 = n_a;
            }
            a_axis0 = a_task0 / (nrg * pa);
            let a_rem = a_task0 - a_axis0 * (nrg * pa);
            a_rg0 = a_rem / pa;
            a_p0 = a_rem - a_rg0 * pa;

            // `ni` is libcint's `ptr .. ptr + dk` in steps of `di`
            // (`CINTg0_kj2d_4d`, g2e.c:552) — the count of `i` planes.
            let ni = dk / di;
            let mut pb = (ll + 1u32) * (lk + 1u32);
            w1 = ll + 1u32;
            w2 = lk + 1u32;
            if comptime!(ibase == 0u32) {
                pb = (lj + 1u32) * ni;
                w1 = lj + 1u32;
                w2 = ni;
            }
            n_b = 3u32 * nrg * pb;
            b_task0 = own_lane;
            if builds == 0u32 {
                b_task0 = n_b;
            }
            b_axis0 = b_task0 / (nrg * pb);
            let b_rem = b_task0 - b_axis0 * (nrg * pb);
            b_rg0 = b_rem / pb;
            let p0 = b_rem - b_rg0 * pb;
            b_p10 = p0 / w2;
            b_p20 = p0 - b_p10 * w2;
        }

        // ── Primitive-pair loop (S1) ──────────────────────────────────────
        //
        // Ket outer, bra inner: libcint's own nesting (`cint2e.c:192-230`).
        // Two things follow from it, and both are the point.
        //
        // 1. The ket's product centre `rkl`, its overlap exponential and its
        //    `ccekl` estimate are formed **once per ket primitive pair**. The
        //    four-deep `nprim` loop this replaces rebuilt them
        //    `nprim_i * nprim_j` times over — a division, three multiply-adds
        //    and an `exp` per rebuild.
        // 2. The accumulation order over primitive quartets becomes the
        //    vendor's, so the f64 rounding of the contraction sum is the
        //    rounding libcint's is.
        //
        // Both pair rows are read from the resident table built by
        // [`crate::kernels::pair_table`], which also carries the vendor's
        // pair-level `expcutoff` screen: a row that is present survived it.
        // The two tests below are the vendor's remaining two, and they are
        // uniform across a cube (every lane of a cooperative launch walks the
        // same `kl` and the same block of `ij` rows), so the barriers inside
        // stay convergent.
        let ij_slot = si * nbas + sj;
        let ij_start = pair_offset[ij_slot as usize];
        let ij_stop = pair_offset[(ij_slot + 1u32) as usize];
        // G1: the ket range is the row's, not the pair table's — the same
        // `pair_offset[sk*nbas+sl] ..` span for an unsplit row, a slice of it
        // when the quartet is spread over several cubes (`expand_kl_split`).
        let kl_start = quartets[(qrow + 6u32) as usize];
        let kl_stop = quartets[(qrow + 7u32) as usize];

        let mut kl_row = kl_start;
        while kl_row < kl_stop {
            let kl_d = kl_row * comptime!(PAIR_DATA_STRIDE as u32);
            let ccekl = pair_data[(kl_d + 4u32) as usize];

            // `cint2e.c:205` — the whole ket pair is under the threshold.
            if ccekl <= expcutoff {
                let kl_i = kl_row * comptime!(PAIR_INDEX_STRIDE as u32);
                let pk = pair_index[kl_i as usize];
                let pl = pair_index[(kl_i + 1u32) as usize];
                // A new `l` primitive: fold the finished `gctrk` into the
                // output with the previous one's coefficients.
                if use_staged && pl != prev_pl {
                    if kempty == 0u32 {
                        stage_contract_out::<F>(
                            ctr, cart_out, ctr_k_off, out_off, coeffs, coff_l, prev_pl, nctr_i,
                            nctr_j, nctr_k, nctr_l, block_len, lane_u, lanes_u,
                        );
                        kempty = 1u32;
                    }
                    prev_pl = pl;
                }
                let mut jempty = 1u32;
                let mut iempty: u32 = 1u32;
                let mut prev_pj = 0xFFFF_FFFFu32;
                let ak = exps[(eoff_k + pk) as usize];
                let al = exps[(eoff_l + pl) as usize];
                let akl = ak + al;

                // The **ket** pair data, inline — `CINT2e_loop_nopt`
                // (`cint2e.c:202-213`), which is the path a caller with no
                // optimizer takes and the one the oracle compares against:
                //
                // ```c
                // akl = ak[kp] + al[lp];
                // ekl = rr_kl * ak[kp] * al[lp] / akl;
                // rkl[0] = (ak[kp]*rk[0] + al[lp]*rl[0]) / akl;
                // ekl = exp(-ekl);
                // ```
                //
                // This is **not** `CINTset_pairdata`'s form, which the pair
                // table stores and the *bra* still uses (`pdata_base`, built by
                // `CINTset_pairdata`, is the bra's alone in this loop). The ket
                // divides by `akl` instead of reusing a reciprocal, and takes
                // the weighted-sum centre instead of the interpolation — the
                // same value in exact arithmetic, one ULP apart in `f64`.
                let klx = rkx - rlx;
                let kly = rky - rly;
                let klz = rkz - rlz;
                let rr_kl = klx * klx + kly * kly + klz * klz;
                let fac_kl = F::exp(F::new(0.0_f32) - (rr_kl * ak * al / akl));
                let rklx = (ak * rkx + al * rlx) / akl;
                let rkly = (ak * rky + al * rly) / akl;
                let rklz = (ak * rkz + al * rlz) / akl;
                // `cint2e.c:212`: what is left of the budget for a bra pair.
                let eijcutoff = expcutoff - ccekl;
                let rklrxx = rklx - rx_kl_x;
                let rklrxy = rkly - rx_kl_y;
                let rklrxz = rklz - rx_kl_z;

                // The bra rows in blocks of `b_max` (B1). A block's build
                // phase runs its rows side by side, one per sub-group; its
                // contraction phase walks them in row order.
                let mut ij_row = ij_start;
                while ij_row < ij_stop {
                    let mut nb = ij_stop - ij_row;
                    if nb > b_max {
                        nb = b_max;
                    }

                    // ── Build phase: sub-group `sub` takes row `ij_row + sub` ──
                    //
                    // `state` is what this sub-group found its row to be:
                    // 0 — the bra pair cannot reach the budget (`cint2e.c:232`),
                    //     nothing happens for it;
                    // 1 — the pair is in, but the primitive quartet's scale
                    //     factor is under `prim_tol`: the staged contraction's
                    //     bookkeeping still runs for it, exactly as before,
                    //     and nothing else does;
                    // 2 — live: G tensor built, contracted.
                    let mut state: u32 = 0u32;
                    let mut pi_b: u32 = 0u32;
                    let mut pj_b: u32 = 0u32;
                    let mut aij = F::new(0.0_f32);
                    let mut a0 = F::new(0.0_f32);
                    let mut a1 = F::new(0.0_f32);
                    let mut x_rys = F::new(0.0_f32);
                    let mut fac1 = F::new(0.0_f32);
                    let mut xij_kl = F::new(0.0_f32);
                    let mut yij_kl = F::new(0.0_f32);
                    let mut zij_kl = F::new(0.0_f32);
                    let mut rijrxx = F::new(0.0_f32);
                    let mut rijrxy = F::new(0.0_f32);
                    let mut rijrxz = F::new(0.0_f32);
                    if sub < nb {
                        let brow = ij_row + sub;
                        let ij_d = brow * comptime!(PAIR_DATA_STRIDE as u32);
                        let cceij = pair_data[(ij_d + 4u32) as usize];
                        // `cint2e.c:232` — this bra pair cannot reach the budget.
                        if cceij <= eijcutoff {
                            state = 1u32;
                            let rijx = pair_data[ij_d as usize];
                            let rijy = pair_data[(ij_d + 1u32) as usize];
                            let rijz = pair_data[(ij_d + 2u32) as usize];
                            let fac_ij = pair_data[(ij_d + 3u32) as usize];
                            let ij_i = brow * comptime!(PAIR_INDEX_STRIDE as u32);
                            let pi = pair_index[ij_i as usize];
                            let pj = pair_index[(ij_i + 1u32) as usize];
                            pi_b = pi;
                            pj_b = pj;
                            let ai = exps[(eoff_i + pi) as usize];
                            let aj = exps[(eoff_j + pj) as usize];
                            aij = ai + aj;

                            xij_kl = rijx - rklx;
                            yij_kl = rijy - rkly;
                            zij_kl = rijz - rklz;
                            let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

                            a1 = aij * akl;
                            a0 = a1 / (aij + akl);
                            x_rys = a0 * rr;

                            // Primitive-quartet screening (Task 34-D).
                            //
                            // `fac1` is the scalar every element of this primitive
                            // quartet's G tensor is built from: `gz` starts at
                            // `wrys[irys] * fac1` and `gx`/`gy` start at 1, so the
                            // whole contribution scales with it. Screening here —
                            // rather than on `fac_ij * fac_kl` alone — keeps the
                            // `sqrt(a0 / a1^3)` factor in the bound, which is not
                            // O(1): for diffuse primitives `a1` is small and that
                            // square root is large, so a prefactor-only test would
                            // discard contributions it had not actually bounded.
                            //
                            // At `prim_tol == 0` (the default) the only quartets
                            // dropped are those whose `fac1` underflowed to exactly
                            // zero, whose contribution is exactly zero — which is
                            // why the tolerance-zero identity gate holds bit for
                            // bit. The Rys weights and the VRR/HRR coefficients are
                            // *not* bounded by one, so a non-zero tolerance is a
                            // proxy, not a certificate: set it well below the
                            // accuracy actually wanted.
                            // `envs->fac[0]`, built in `CINT2e_loop`'s own
                            // nesting order (`cint2e.c:75-119`):
                            //
                            // ```c
                            // fac1l = common_factor * cl[lp];
                            // fac1k = fac1l * ck[kp];
                            // fac1j = fac1k * cj[jp];
                            // fac1i = fac1j * ci[ip] * expij * expkl;
                            // ```
                            //
                            // then `fac1 = sqrt(a0/a1^3) * fac1[0]`
                            // (`g2e.c:17`). Each uncontracted side's
                            // coefficient enters here, at the seed of the G
                            // tensor, instead of multiplying the finished
                            // contraction — the recurrence is linear in it, so
                            // the two agree to within a few ULP and not
                            // bit-for-bit.
                            let mut fac_env = common_factor;
                            if nctr_l == 1u32 {
                                fac_env = fac_env * coeffs[(coff_l + pl) as usize];
                            }
                            if nctr_k == 1u32 {
                                fac_env = fac_env * coeffs[(coff_k + pk) as usize];
                            }
                            if nctr_j == 1u32 {
                                fac_env = fac_env * coeffs[(coff_j + pj) as usize];
                            }
                            if nctr_i == 1u32 {
                                fac_env = fac_env * coeffs[(coff_i + pi) as usize];
                            }
                            // `expijkl = pdata_ij->eij * ekl` and then
                            // `fac1i = fac1j * ci[ip] * expijkl`
                            // (`cint2e.c:238-240`): the two exponentials
                            // multiply *each other* first, not the running
                            // product in turn.
                            fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * (fac_env * (fac_ij * fac_kl));
                            // On the magnitude: now that the contraction
                            // coefficients ride inside `fac1`, its sign is the
                            // sign of their product, and a contracted s shell
                            // routinely carries a negative one (`-0.0999…` for
                            // O-2s in STO-3G). Testing the signed value would
                            // discard every primitive quartet whose coefficient
                            // product is negative — the same reason the 1e
                            // nuclear arm tests `F::abs(fac1)`.
                            if F::abs(fac1) > prim_tol {
                                state = 2u32;
                            }
                            rijrxx = rijx - rx_ij_x;
                            rijrxy = rijy - rx_ij_y;
                            rijrxz = rijz - rx_ij_z;
                        }
                    }

                    // Probe 3 skips the whole G build, probe 4 only the roots;
                    // the contraction then reads a stale slab and the output is
                    // undefined, which is the deal every probe makes.
                    let mut do_build: u32 = 0u32;
                    if state == 2u32 {
                        do_build = 1u32;
                    }
                    if ctr_mode == 3u32 {
                        do_build = 0u32;
                    }
                    let mut do_roots: u32 = do_build;
                    if ctr_mode == 4u32 {
                        do_roots = 0u32;
                    }
                    if do_roots == 1u32 {
                        // ── S3: every lane of the sub-group computes the Rys roots ──
                        //
                        // `urys`/`wrys` are per-work-item private arrays and
                        // `rys_rootN` is a pure function of `x_rys`, so every
                        // lane computing them lands on identical values with
                        // no barrier and no shared storage. It is what lets
                        // each lane own a slice of the G build below without a
                        // barrier to hand the roots around first.
                        //
                        // F1 (§15): the *order* is a runtime value now, so
                        // the fixed-order solvers are selected by a runtime
                        // branch and only the orders this dispatch can
                        // carry are emitted — `comptime!(nr_max >= k)`
                        // keeps a `nroots <= 3` program from compiling
                        // `rys_root5`. The branch costs one predicted test
                        // per primitive quartet against a solver body of
                        // tens of operations, and every lane still walks
                        // the same arm (a quartet's order is cube-uniform).
                        if comptime!(nr_max <= 5u32) {
                            // `rys_roots_fixed_rt` is the whole of
                            // `CINTrys_roots` for a *runtime* order: the two
                            // global table branches first, the per-order fit
                            // only in the band between them. It keeps the same
                            // `nr_max` guards, so a `nroots <= 3` program still
                            // does not compile `rys_root5`.
                            rys_roots_fixed_rt::<F>(
                                rys_tab, x_rys, &mut urys, &mut wrys, pie4, nroots, nr_max,
                            );
                        } else {
                            // nroots 6..=12: the inline Wheeler/Jacobi
                            // entry (task 33-01), reachable only once
                            // `device_nroots_ceiling` was raised for
                            // this family on this backend.
                            //
                            // F1 (§15) does *not* fuse these orders: the
                            // extended solver takes its order at comptime,
                            // and merging 6..=12 into one dispatch would
                            // emit seven double-double solvers where the
                            // fixed-order arms emit five short ones. A
                            // class above five therefore keeps a dispatch
                            // of its own, and `nr_max == nroots` there.
                            rys_roots_ext_dev(
                                rys_tab,
                                f64::cast_from(x_rys),
                                &mut uext,
                                &mut wext,
                                nr_max,
                            );
                            #[unroll]
                            for iext in 0..nr_max {
                                urys[iext as usize] = F::cast_from(uext[iext as usize]);
                                wrys[iext as usize] = F::cast_from(wext[iext as usize]);
                            }
                        }
                    }
                    if do_build == 1u32 {
                        // ── S3: build the [gx|gy|gz] tensor cooperatively ─────
                        //
                        // It parallelises without any reduction, because the
                        // recurrences never cross an axis or a Rys root. The
                        // VRR at `(axis, root)` touches only
                        // `off + root + n*dn + m*dm`, and `dn`/`dm` are
                        // multiples of `nroots` in this root-fastest layout,
                        // so every read and write it makes stays inside its
                        // own residue class. So `3 * nroots` independent
                        // tasks, handed out `task % own_lanes == own_lane`
                        // across the sub-group, each element still computed by
                        // exactly the expression that computed it before —
                        // **the result is bit-identical**, which is the gate
                        // rather than a divergence budget.
                        //
                        // The per-unit decomposition owns the whole root axis
                        // inside one unit, so the `nroots` independent
                        // recurrences collapse into a single `Vector` chain of
                        // the same length (V1, §12). Elementwise ops on the same
                        // operands in the same order, so the slab comes out
                        // bit-identical. The cooperative arm keeps the scalar
                        // loop: there its `3 * nroots` tasks are real work for
                        // real lanes.
                        //
                        // F1 (§15): `nroots` is a runtime value and a `Vector`
                        // width is a type-level size, so the four widths this
                        // arm serves are instantiated as `Const<2..5>` and
                        // chosen by one branch per primitive quartet.
                        // `vec_built` records whether an arm ran, so the
                        // scalar fallback below is one test rather than a
                        // repeat of the same five-way ladder.
                        let mut vec_built: u32 = 0u32;
                        if comptime!(per_unit == 1u32 && nr_max > 1u32) {
                            let uslice = urys.to_slice_mut();
                            let wslice = wrys.to_slice_mut();
                            if comptime!(nr_max >= 2u32) {
                                if nroots == 2u32 {
                                    vrr_build_axes_roots::<F, Const<2>>(
                                        &mut g_slab,
                                        &uslice,
                                        &wslice,
                                        g_off,
                                        g_size,
                                        nmax,
                                        mmax,
                                        g2d_ijmax,
                                        g2d_klmax,
                                        a0,
                                        aij,
                                        akl,
                                        a1,
                                        fac1,
                                        xij_kl,
                                        yij_kl,
                                        zij_kl,
                                        rijrxx,
                                        rijrxy,
                                        rijrxz,
                                        rklrxx,
                                        rklrxy,
                                        rklrxz,
                                        2usize,
                                    );
                                    vec_built = 1u32;
                                }
                            }
                            if comptime!(nr_max >= 3u32) {
                                if nroots == 3u32 {
                                    vrr_build_axes_roots::<F, Const<3>>(
                                        &mut g_slab,
                                        &uslice,
                                        &wslice,
                                        g_off,
                                        g_size,
                                        nmax,
                                        mmax,
                                        g2d_ijmax,
                                        g2d_klmax,
                                        a0,
                                        aij,
                                        akl,
                                        a1,
                                        fac1,
                                        xij_kl,
                                        yij_kl,
                                        zij_kl,
                                        rijrxx,
                                        rijrxy,
                                        rijrxz,
                                        rklrxx,
                                        rklrxy,
                                        rklrxz,
                                        3usize,
                                    );
                                    vec_built = 1u32;
                                }
                            }
                            if comptime!(nr_max >= 4u32) {
                                if nroots == 4u32 {
                                    vrr_build_axes_roots::<F, Const<4>>(
                                        &mut g_slab,
                                        &uslice,
                                        &wslice,
                                        g_off,
                                        g_size,
                                        nmax,
                                        mmax,
                                        g2d_ijmax,
                                        g2d_klmax,
                                        a0,
                                        aij,
                                        akl,
                                        a1,
                                        fac1,
                                        xij_kl,
                                        yij_kl,
                                        zij_kl,
                                        rijrxx,
                                        rijrxy,
                                        rijrxz,
                                        rklrxx,
                                        rklrxy,
                                        rklrxz,
                                        4usize,
                                    );
                                    vec_built = 1u32;
                                }
                            }
                            if comptime!(nr_max >= 5u32) {
                                if nroots == 5u32 {
                                    vrr_build_axes_roots::<F, Const<5>>(
                                        &mut g_slab,
                                        &uslice,
                                        &wslice,
                                        g_off,
                                        g_size,
                                        nmax,
                                        mmax,
                                        g2d_ijmax,
                                        g2d_klmax,
                                        a0,
                                        aij,
                                        akl,
                                        a1,
                                        fac1,
                                        xij_kl,
                                        yij_kl,
                                        zij_kl,
                                        rijrxx,
                                        rijrxy,
                                        rijrxz,
                                        rklrxx,
                                        rklrxy,
                                        rklrxz,
                                        5usize,
                                    );
                                    vec_built = 1u32;
                                }
                            }
                        }
                        if vec_built == 0u32 {
                            // Owned tasks only (§22.4): a lane walks the
                            // `(axis, root)` tasks `own_lane, own_lane +
                            // own_lanes, …` and decodes each, instead of every
                            // lane enumerating every task and testing it — the
                            // loop control and the modulo were most of what an
                            // idle lane did. The per-root coefficients are
                            // formed per task, by the same expressions.
                            let mut task = vrr_task0;
                            let mut axis = vrr_axis0;
                            let mut rg = vrr_rg0;
                            while task < n_vrr {
                                let r_lo = rg * rw;
                                let mut r_hi = r_lo + rw;
                                if r_hi > nroots {
                                    r_hi = nroots;
                                }
                                let mut irys2 = r_lo;
                                while irys2 < r_hi {
                                    let u2 = a0 * urys[irys2 as usize];
                                    let tmp4 = F::new(0.5_f32) / (u2 * (aij + akl) + a1);
                                    let tmp5 = u2 * tmp4;
                                    let tmp1 = F::new(2.0_f32) * tmp5;
                                    let tmp2 = tmp1 * akl;
                                    let tmp3 = tmp1 * aij;
                                    let b00 = tmp5;
                                    let b10 = tmp5 + tmp4 * akl;
                                    let b01 = tmp5 + tmp4 * aij;

                                    let off = g_off + axis * g_size;
                                    let mut xkl = xij_kl;
                                    let mut rijrx = rijrxx;
                                    let mut rklrx = rklrxx;
                                    if axis == 1u32 {
                                        xkl = yij_kl;
                                        rijrx = rijrxy;
                                        rklrx = rklrxy;
                                    } else if axis == 2u32 {
                                        xkl = zij_kl;
                                        rijrx = rijrxz;
                                        rklrx = rklrxz;
                                    }
                                    let c00 = rijrx - tmp2 * xkl;
                                    let c0p = rklrx + tmp3 * xkl;

                                    // Inline vrr_fill_axis(g[off..], irys2, nmax, mmax,
                                    //   dn=g2d_ijmax, dm=g2d_klmax, c00, c0p, b10, b01, b00).
                                    let root = irys2;
                                    let dn = g2d_ijmax;
                                    let dm = g2d_klmax;

                                    // The seed for this slice. `gx`/`gy` start at
                                    // one, `gz` at the root's Rys weight times the
                                    // primitive's scale factor. Statement-level
                                    // mutation rather than a value-returning `if`:
                                    // the latter does not lower the way ordinary
                                    // Rust does inside `#[cube]`, which cost the
                                    // device c2s pass 1 127 wrong values once.
                                    let mut seed = F::new(1.0_f32);
                                    if axis == 2u32 {
                                        seed = wrys[root as usize] * fac1;
                                    }
                                    g_slab[(off + root) as usize] = seed;

                                    if nmax > 0u32 {
                                        let mut s0 = g_slab[(off + root) as usize];
                                        let mut s1 = c00 * s0;
                                        g_slab[(off + root + dn) as usize] = s1;
                                        let mut n = 1u32;
                                        while n < nmax {
                                            let s2 = c00 * s1 + F::cast_from(n) * b10 * s0;
                                            g_slab[(off + root + (n + 1u32) * dn) as usize] = s2;
                                            s0 = s1;
                                            s1 = s2;
                                            n += 1u32;
                                        }
                                    }

                                    if mmax > 0u32 {
                                        let mut s0 = g_slab[(off + root) as usize];
                                        let mut s1 = c0p * s0;
                                        g_slab[(off + root + dm) as usize] = s1;
                                        let mut m = 1u32;
                                        while m < mmax {
                                            let s2 = c0p * s1 + F::cast_from(m) * b01 * s0;
                                            g_slab[(off + root + (m + 1u32) * dm) as usize] = s2;
                                            s0 = s1;
                                            s1 = s2;
                                            m += 1u32;
                                        }

                                        if nmax > 0u32 {
                                            let mut s0n = g_slab[(off + root + dn) as usize];
                                            let mut s1n =
                                                c0p * s0n + b00 * g_slab[(off + root) as usize];
                                            g_slab[(off + root + dn + dm) as usize] = s1n;
                                            let mut m2 = 1u32;
                                            while m2 < mmax {
                                                let s2n = c0p * s1n
                                                    + F::cast_from(m2) * b01 * s0n
                                                    + b00 * g_slab[(off + root + m2 * dm) as usize];
                                                g_slab[(off + root + dn + (m2 + 1u32) * dm)
                                                    as usize] = s2n;
                                                s0n = s1n;
                                                s1n = s2n;
                                                m2 += 1u32;
                                            }
                                        }
                                    }

                                    if nmax > 0u32 {
                                        let mut m3 = 1u32;
                                        while m3 <= mmax {
                                            let offm = m3 * dm;
                                            let jbase = offm + root;
                                            let mut s0 = g_slab[(off + jbase) as usize];
                                            let mut s1 = g_slab[(off + jbase + dn) as usize];
                                            let mut n2 = 1u32;
                                            while n2 < nmax {
                                                let s2 = c00 * s1
                                                    + F::cast_from(n2) * b10 * s0
                                                    + F::cast_from(m3)
                                                        * b00
                                                        * g_slab
                                                            [(off + jbase + n2 * dn - dm) as usize];
                                                g_slab[(off + jbase + (n2 + 1u32) * dn) as usize] =
                                                    s2;
                                                s0 = s1;
                                                s1 = s2;
                                                n2 += 1u32;
                                            }
                                            m3 += 1u32;
                                        }
                                    }
                                    irys2 += 1u32;
                                }
                                task += own_lanes;
                                // Advance the decoded `(axis, root group)` with
                                // the task, carrying — no division per task,
                                // which the per-unit arm (every task owned)
                                // paid in full (§22.5).
                                rg += own_lanes;
                                while rg >= nrg {
                                    rg -= nrg;
                                    axis += 1u32;
                                }
                            }
                        }
                    }
                    // The HRR reads every root's VRR output at its `(axis,
                    // root)`, which another lane of the sub-group wrote.
                    if comptime!(per_unit == 0u32) {
                        sync_cube();
                    }

                    // ── HRR transfer on the per-unit arm: the nest as it was ──
                    //
                    // A unit walks every `(axis, root)` itself, and the old
                    // nest's two innermost loops sweep the contiguous
                    // `(i, root)` span of each element — which is what the
                    // CPU vectorises. The per-chain form below, dealt out
                    // across a sub-group's lanes, measured 1.1x–1.3x slower
                    // here (§22.5), so the per-unit arm keeps its nest and the
                    // cooperative arm takes the split. Same expressions, same
                    // operands: the dump comparison holds both to the bit.
                    if comptime!(per_unit == 1u32) {
                        if do_build == 1u32 {
                            // ── HRR transfer (branch by comptime kbase/ibase) ──────
                            #[unroll]
                            for axis2 in 0..3u32 {
                                let off = g_off + axis2 * g_size;
                                let mut rirj = rirjx;
                                let mut rkrl = rkrlx;
                                if axis2 == 1u32 {
                                    rirj = rirjy;
                                    rkrl = rkrly;
                                } else if axis2 == 2u32 {
                                    rirj = rirjz;
                                    rkrl = rkrlz;
                                }

                                // S3: the roots this lane owns on this axis,
                                // under the same `axis * nroots + root` map
                                // the VRR used — so a lane's HRR reads only
                                // the slice its own VRR wrote, and no barrier
                                // separates them. The owned roots are an
                                // arithmetic progression, so the loops below
                                // step by `lanes` instead of testing every
                                // root. Per-unit collapses it exactly:
                                // `lanes == 1` and `lane == 0` give
                                // `r_first == 0` and a step of one, which is
                                // the loop that was there before.

                                if comptime!(kbase == 1u32 && ibase == 1u32) {
                                    // ik2d: i then k done; transfer dl←dk (ll), dj←di (lj).
                                    let mut l = 1u32;
                                    while l <= ll {
                                        let mut k = 0u32;
                                        while k <= (mmax - l) {
                                            let mut i = 0u32;
                                            while i <= nmax {
                                                let ptr = l * dl + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nroots {
                                                    let idx = ptr + r;
                                                    g_slab[(off + idx) as usize] = rkrl
                                                        * g_slab[(off + idx - dl) as usize]
                                                        + g_slab[(off + idx - dl + dk) as usize];
                                                    r += 1u32;
                                                }
                                                i += 1u32;
                                            }
                                            k += 1u32;
                                        }
                                        l += 1u32;
                                    }
                                    let mut j = 1u32;
                                    while j <= lj {
                                        let mut l2 = 0u32;
                                        while l2 <= ll {
                                            let mut k2 = 0u32;
                                            while k2 <= lk {
                                                let ptr = j * dj + l2 * dl + k2 * dk;
                                                let mut i2 = 0u32;
                                                while i2 <= (nmax - j) {
                                                    let pbase = ptr + i2 * di;
                                                    let mut r = 0u32;
                                                    while r < nroots {
                                                        let idx = pbase + r;
                                                        g_slab[(off + idx) as usize] = rirj
                                                            * g_slab[(off + idx - dj) as usize]
                                                            + g_slab
                                                                [(off + idx - dj + di) as usize];
                                                        r += 1u32;
                                                    }
                                                    i2 += 1u32;
                                                }
                                                k2 += 1u32;
                                            }
                                            l2 += 1u32;
                                        }
                                        j += 1u32;
                                    }
                                } else if comptime!(kbase == 1u32 && ibase == 0u32) {
                                    // kj2d: i raise (dj←di), then l raise (dl←dk).
                                    let mut i = 1u32;
                                    while i <= li {
                                        let mut j = 0u32;
                                        while j <= (nmax - i) {
                                            let mut k = 0u32;
                                            while k <= mmax {
                                                let ptr = j * dj + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nroots {
                                                    let idx = ptr + r;
                                                    g_slab[(off + idx) as usize] = rirj
                                                        * g_slab[(off + idx - di) as usize]
                                                        + g_slab[(off + idx - di + dj) as usize];
                                                    r += 1u32;
                                                }
                                                k += 1u32;
                                            }
                                            j += 1u32;
                                        }
                                        i += 1u32;
                                    }
                                    let mut l = 1u32;
                                    while l <= ll {
                                        let mut k = 0u32;
                                        while k <= (mmax - l) {
                                            let mut j = 0u32;
                                            while j <= lj {
                                                let ptr = l * dl + k * dk + j * dj;
                                                // libcint `CINTg0_kj2d_4d` (g2e.c:552)
                                                // walks `ptr .. ptr + dk`, and so does the
                                                // host `hrr_kj2d_4d`. This loop is the
                                                // flattened form of that range, so its
                                                // bound is `dk`, not `di`.
                                                //
                                                // With `ibase == 0`, `di == nroots` and
                                                // `dk == nroots * (li + 1)`, so a `di`
                                                // bound silently under-writes every
                                                // `i >= 1` plane. That was invisible to the
                                                // existing (s,s,p,s) device test — it has
                                                // `li == 0`, where `dk == di` — and to any
                                                // `ll == 0` class, where this loop never
                                                // runs at all.
                                                // S3: `n` flattens `(i, root)` at
                                                // stride `di == nroots` here, so
                                                // this is the nest it always was,
                                                // with the root axis partitioned.
                                                let mut nb = 0u32;
                                                while nb < dk {
                                                    let mut r = 0u32;
                                                    while r < nroots {
                                                        let idx = ptr + nb + r;
                                                        g_slab[(off + idx) as usize] = rkrl
                                                            * g_slab[(off + idx - dl) as usize]
                                                            + g_slab
                                                                [(off + idx - dl + dk) as usize];
                                                        r += 1u32;
                                                    }
                                                    nb += di;
                                                }
                                                j += 1u32;
                                            }
                                            k += 1u32;
                                        }
                                        l += 1u32;
                                    }
                                } else if comptime!(kbase == 0u32 && ibase == 1u32) {
                                    // il2d: k raise (dl←dk), then j raise (dj←di).
                                    let mut k = 1u32;
                                    while k <= lk {
                                        let mut l = 0u32;
                                        while l <= (mmax - k) {
                                            let mut i = 0u32;
                                            while i <= nmax {
                                                let ptr = l * dl + k * dk + i * di;
                                                let mut r = 0u32;
                                                while r < nroots {
                                                    let idx = ptr + r;
                                                    g_slab[(off + idx) as usize] = rkrl
                                                        * g_slab[(off + idx - dk) as usize]
                                                        + g_slab[(off + idx - dk + dl) as usize];
                                                    r += 1u32;
                                                }
                                                i += 1u32;
                                            }
                                            l += 1u32;
                                        }
                                        k += 1u32;
                                    }
                                    let mut j = 1u32;
                                    while j <= lj {
                                        let mut l = 0u32;
                                        while l <= ll {
                                            let mut k2 = 0u32;
                                            while k2 <= lk {
                                                let ptr = j * dj + l * dl + k2 * dk;
                                                let mut i2 = 0u32;
                                                while i2 <= (nmax - j) {
                                                    let pbase = ptr + i2 * di;
                                                    let mut r = 0u32;
                                                    while r < nroots {
                                                        let idx = pbase + r;
                                                        g_slab[(off + idx) as usize] = rirj
                                                            * g_slab[(off + idx - dj) as usize]
                                                            + g_slab
                                                                [(off + idx - dj + di) as usize];
                                                        r += 1u32;
                                                    }
                                                    i2 += 1u32;
                                                }
                                                k2 += 1u32;
                                            }
                                            l += 1u32;
                                        }
                                        j += 1u32;
                                    }
                                } else {
                                    // lj2d: i raise (dj←di), then k raise (dl←dk).
                                    let mut i = 1u32;
                                    while i <= li {
                                        let mut j = 0u32;
                                        while j <= (nmax - i) {
                                            let mut l = 0u32;
                                            while l <= mmax {
                                                let ptr = j * dj + l * dl + i * di;
                                                let mut r = 0u32;
                                                while r < nroots {
                                                    let idx = ptr + r;
                                                    g_slab[(off + idx) as usize] = rirj
                                                        * g_slab[(off + idx - di) as usize]
                                                        + g_slab[(off + idx - di + dj) as usize];
                                                    r += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            j += 1u32;
                                        }
                                        i += 1u32;
                                    }
                                    let mut j2 = 0u32;
                                    while j2 <= lj {
                                        let mut k = 1u32;
                                        while k <= lk {
                                            let mut l = 0u32;
                                            while l <= (mmax - k) {
                                                let ptr = j2 * dj + l * dl + k * dk;
                                                // S3: `n` flattens `(i, root)` at
                                                // stride `di == nroots` here, so
                                                // this is the nest it always was,
                                                // with the root axis partitioned.
                                                let mut nb = 0u32;
                                                while nb < dk {
                                                    let mut r = 0u32;
                                                    while r < nroots {
                                                        let idx = ptr + nb + r;
                                                        g_slab[(off + idx) as usize] = rkrl
                                                            * g_slab[(off + idx - dk) as usize]
                                                            + g_slab
                                                                [(off + idx - dk + dl) as usize];
                                                        r += 1u32;
                                                    }
                                                    nb += di;
                                                }
                                                l += 1u32;
                                            }
                                            k += 1u32;
                                        }
                                        j2 += 1u32;
                                    }
                                }
                            }
                        }
                    }
                    // ── HRR transfer, first raise (branch by comptime kbase/ibase) ──
                    //
                    // Each raise is a chain along *one* index — the one being
                    // raised — and independent across every other index, so the
                    // work is dealt out as one task per `(axis, root, <the other
                    // index>)` chain across the sub-group's lanes, each lane
                    // walking only the tasks it owns (§22.4) and the chain
                    // itself serially. Every element is still the same
                    // expression over the same operands, so the slab is
                    // bit-identical to the old nest. A lane's reads inside a
                    // raise are its own writes or the previous phase's, which
                    // is what the barriers around each raise guarantee.
                    //
                    // The task's third index is `i` where `ibase == 1` (the
                    // raise runs over `l` or `k`) and `k` or `l` where
                    // `ibase == 0` (it runs over `i`).
                    if comptime!(per_unit == 0u32) {
                        if do_build == 1u32 {
                            let mut task = a_task0;
                            let mut axis2 = a_axis0;
                            let mut rg = a_rg0;
                            let mut p = a_p0;
                            while task < n_a {
                                let r_lo = rg * rw;
                                let mut r_hi = r_lo + rw;
                                if r_hi > nroots {
                                    r_hi = nroots;
                                }
                                let off = g_off + axis2 * g_size;
                                let mut rirj = rirjx;
                                let mut rkrl = rkrlx;
                                if axis2 == 1u32 {
                                    rirj = rirjy;
                                    rkrl = rkrly;
                                } else if axis2 == 2u32 {
                                    rirj = rirjz;
                                    rkrl = rkrlz;
                                }

                                // The chain this task owns, in the one shape all four arms
                                // share: `a` from 1 to `a_max` at `sa` — the index being
                                // raised — and `b` from 0 to `m_max - a` at `sb`. The arm
                                // decides only which strides those are, and which index the
                                // task fixed into `cbase`; the raise itself is
                                // [`hrr_chain`], at a comptime root width.
                                let mut cbase = off + p * di;
                                let mut a_max = ll;
                                let mut m_max = mmax;
                                let mut sa = dl;
                                let mut sb = dk;
                                let mut coef = rkrl;
                                if comptime!(kbase == 1u32 && ibase == 1u32) {
                                    // ik2d, first raise: dl←dk (ll), chain along l, p = i.
                                } else if comptime!(kbase == 1u32 && ibase == 0u32) {
                                    // kj2d, first raise: dj←di (li), chain along i, p = k.
                                    cbase = off + p * dk;
                                    a_max = li;
                                    m_max = nmax;
                                    sa = di;
                                    sb = dj;
                                    coef = rirj;
                                } else if comptime!(kbase == 0u32 && ibase == 1u32) {
                                    // il2d, first raise: dl←dk (lk), chain along k, p = i.
                                    a_max = lk;
                                    sa = dk;
                                    sb = dl;
                                } else {
                                    // lj2d, first raise: dj←di (li), chain along i, p = l.
                                    cbase = off + p * dl;
                                    a_max = li;
                                    m_max = nmax;
                                    sa = di;
                                    sb = dj;
                                    coef = rirj;
                                }
                                hrr_chain_w::<F>(
                                    &mut g_slab,
                                    cbase,
                                    a_max,
                                    m_max,
                                    sa,
                                    sb,
                                    coef,
                                    r_lo,
                                    r_hi,
                                );
                                task += own_lanes;
                                p += own_lanes;
                                while p >= pa {
                                    p -= pa;
                                    rg += 1u32;
                                    if rg >= nrg {
                                        rg -= nrg;
                                        axis2 += 1u32;
                                    }
                                }
                            }
                        }
                    }
                    if comptime!(per_unit == 0u32) {
                        sync_cube();
                    }

                    // ── HRR transfer, second raise ────────────────────────
                    //
                    // The task's other indices are `(l, k)` where `ibase == 1`
                    // (the raise runs over `j`) and `(j, i)` where `ibase == 0`
                    // (it runs over `l` or `k`); `ni` is the count of `i`
                    // planes, libcint's `ptr .. ptr + dk` in steps of `di`
                    // (`CINTg0_kj2d_4d`, g2e.c:552).
                    if comptime!(per_unit == 0u32) {
                        if do_build == 1u32 {
                            // The task's other pair `(p1, p2)` — `(l, k)` or
                            // `(j, i)` — is decoded once per quartet above and
                            // then carried.
                            let mut task = b_task0;
                            let mut axis2 = b_axis0;
                            let mut rg = b_rg0;
                            let mut p1 = b_p10;
                            let mut p2 = b_p20;
                            while task < n_b {
                                let r_lo = rg * rw;
                                let mut r_hi = r_lo + rw;
                                if r_hi > nroots {
                                    r_hi = nroots;
                                }
                                let off = g_off + axis2 * g_size;
                                let mut rirj = rirjx;
                                let mut rkrl = rkrlx;
                                if axis2 == 1u32 {
                                    rirj = rirjy;
                                    rkrl = rkrly;
                                } else if axis2 == 2u32 {
                                    rirj = rirjz;
                                    rkrl = rkrlz;
                                }

                                // Same chain, one raise later: the task fixed a *pair* of
                                // indices this time, and they are what `cbase` carries.
                                let mut cbase = off + p1 * dl + p2 * dk;
                                let mut a_max = lj;
                                let mut m_max = nmax;
                                let mut sa = dj;
                                let mut sb = di;
                                let mut coef = rirj;
                                if comptime!(kbase == 1u32 && ibase == 1u32) {
                                    // ik2d, second raise: dj←di (lj), chain along j, p = (l, k).
                                } else if comptime!(kbase == 1u32 && ibase == 0u32) {
                                    // kj2d, second raise: dl←dk (ll), chain along l, p = (j, i).
                                    cbase = off + p1 * dj + p2 * di;
                                    a_max = ll;
                                    m_max = mmax;
                                    sa = dl;
                                    sb = dk;
                                    coef = rkrl;
                                } else if comptime!(kbase == 0u32 && ibase == 1u32) {
                                    // il2d, second raise: dj←di (lj), chain along j, p = (l, k).
                                } else {
                                    // lj2d, second raise: dl←dk (lk), chain along k, p = (j, i).
                                    cbase = off + p1 * dj + p2 * di;
                                    a_max = lk;
                                    m_max = mmax;
                                    sa = dk;
                                    sb = dl;
                                    coef = rkrl;
                                }
                                hrr_chain_w::<F>(
                                    &mut g_slab,
                                    cbase,
                                    a_max,
                                    m_max,
                                    sa,
                                    sb,
                                    coef,
                                    r_lo,
                                    r_hi,
                                );
                                task += own_lanes;
                                p2 += own_lanes;
                                while p2 >= w2 {
                                    p2 -= w2;
                                    p1 += 1u32;
                                }
                                while p1 >= w1 {
                                    p1 -= w1;
                                    rg += 1u32;
                                    if rg >= nrg {
                                        rg -= nrg;
                                        axis2 += 1u32;
                                    }
                                }
                            }
                        }
                    }
                    // Publish this row's meta for the contraction phase, which
                    // walks every row of the block: the state, the primitive
                    // weight and the `i` coefficients the builder already has
                    // — read back from shared memory in ~100 cycles where the
                    // contraction used to spend two dependent uniform global
                    // loads per row (§22.5). Only the shared region has a
                    // block wider than one; elsewhere the registers serve.
                    if comptime!(shared_tier > 0u32) {
                        if sub < nb {
                            if sub_lane == 0u32 {
                                let m = sub * comptime!(META_STRIDE);
                                g_slab[m as usize] = F::cast_from(state);
                                // The weight: the four coefficients' product
                                // for a segmented quartet, the segmented
                                // shells' product for a staged one.
                                // Every uncontracted side's coefficient is
                                // already inside `fac1`, where `CINT2e_loop`
                                // puts it, so the block weight is 1 and only a
                                // *contracted* side's coefficient is still
                                // applied — by `stage_contract_out`, which is
                                // libcint's `PRIM2CTR`.
                                let w = F::new(1.0_f32);
                                g_slab[(m + 1u32) as usize] = w;
                                g_slab[(m + 2u32) as usize] = F::cast_from(pi_b);
                                g_slab[(m + 3u32) as usize] = F::cast_from(pj_b);
                                let mut ci = 0u32;
                                while ci < 4u32 {
                                    let mut cvi = F::new(1.0_f32);
                                    if nctr_i > 1u32 {
                                        if ci < nctr_i {
                                            cvi = coeffs[(coff_i + pi_b * nctr_i + ci) as usize];
                                        }
                                    }
                                    g_slab[(m + 4u32 + ci) as usize] = cvi;
                                    ci += 1u32;
                                }
                            }
                        }
                    }
                    if comptime!(per_unit == 0u32) {
                        sync_cube();
                    }

                    // ── Contraction phase: the block's rows, in row order ──
                    let mut b = 0u32;
                    while b < nb {
                        // The row's meta: from the registers where the block is
                        // one row (`b == sub == 0`), from shared memory
                        // otherwise. `prim_weight` and `fold` are one value —
                        // the arm that uses each is exclusive.
                        let mut st: u32 = state;
                        let mut pi: u32 = pi_b;
                        let mut pj: u32 = pj_b;
                        let mut w_b = F::new(1.0_f32);
                        let mb = b * comptime!(META_STRIDE);
                        if comptime!(shared_tier > 0u32) {
                            let flag = g_slab[mb as usize];
                            st = 0u32;
                            if flag > F::new(0.5_f32) {
                                st = 1u32;
                            }
                            if flag > F::new(1.5_f32) {
                                st = 2u32;
                            }
                            w_b = g_slab[(mb + 1u32) as usize];
                            pi = u32::cast_from(g_slab[(mb + 2u32) as usize]);
                            pj = u32::cast_from(g_slab[(mb + 3u32) as usize]);
                        }
                        if comptime!(shared_tier == 0u32) {
                            if st == 2u32 {
                                // The global-slab arm used to rebuild the
                                // primitive weight here rather than read the
                                // meta row. There is nothing left to rebuild:
                                // every segmented shell's coefficient is inside
                                // `fac1` (`cint2e.c:75-119`) and every generally
                                // contracted one is applied by its stage, so the
                                // weight is 1 on both arms.
                                w_b = F::new(1.0_f32);
                            }
                        }
                        if st != 0u32 {
                            // A new `j` primitive: fold `gctri` into `gctrj`.
                            if use_staged && pj != prev_pj {
                                if iempty == 0u32 {
                                    stage_contract::<F>(
                                        ctr, ctr_i_off, ctr_j_off, nctr_i, coeffs, coff_j, prev_pj,
                                        nctr_j, block_len, lane_u, lanes_u, jempty,
                                    );
                                    jempty = 0u32;
                                    iempty = 1u32;
                                }
                                prev_pj = pj;
                            }
                        }
                        if st == 2u32 {
                            // This row's sub-slab, and where the element walk starts.
                            let gb = slab_base + b * g3;
                            let mut q_start = lane_u;
                            if ctr_mode == 2u32 {
                                q_start = block_len;
                            }

                            // ── Contract into per-quad Cartesian blocks cooperatively ──
                            //
                            // K1 (GTH plan §10): the Cartesian elements of a class are
                            // walked through its index table — three G offsets per
                            // element, built once on the host by
                            // `TwoELaunchGroup::push_class` in the order the five-deep
                            // `(l, k, j, i)` nest walked them, `i` fastest — instead of
                            // that nest being re-run per primitive quartet. libcint does
                            // the same (`CINTg2e_index_xyz`, `idx` in `cint2e.c`). Each
                            // lane strides through its own elements, and each element is
                            // the same expression over the same roots in the same order,
                            // accumulated into the same place, so the result is
                            // bit-identical (`gth_profile`'s dump comparison is the gate).
                            // `ctr_mode == 2` is the measurement probe: the G build runs
                            // and the contraction does not, so the difference against the
                            // default is the contraction's share. The output is undefined
                            // under it; only `gth_profile` sets it.
                            //
                            // §23: the walk itself is [`contract_block_elems`], and the
                            // Rys-width ladder that used to sit inside it is taken here,
                            // once per primitive quartet.
                            contract_block_elems_w::<F>(
                                &g_slab,
                                class_idx,
                                coeffs,
                                &mut acc,
                                ctr,
                                cart_out,
                                idx_off,
                                gb,
                                g_size,
                                block_len,
                                q_start,
                                lanes_u,
                                w_b,
                                out_off,
                                ctr_i_off,
                                mb,
                                // The staged arm's flag is per ket primitive
                                // pair; arms 0/1 accumulate for the whole
                                // quartet and use `qempty`.
                                if use_staged { iempty } else { qempty },
                                ctr_arm,
                                coff_i,
                                pi,
                                nctr_i,
                                coff_j,
                                pj,
                                nctr_j,
                                coff_k,
                                pk,
                                nctr_k,
                                coff_l,
                                pl,
                                nctr_l,
                                nroots,
                                shared_tier,
                                nr_max,
                            );
                            if use_staged {
                                iempty = 0u32;
                            }
                            qempty = 0u32;
                        }
                        b += 1u32;
                    }
                    // The next block's build overwrites the sub-slabs this
                    // contraction read.
                    if comptime!(per_unit == 0u32) {
                        sync_cube();
                    }
                    ij_row += nb;
                }
                // End of this ket primitive pair's bra loop: the last `j`
                // primitive's `gctri`, then this `k` primitive's `gctrj`.
                if use_staged {
                    if iempty == 0u32 {
                        stage_contract::<F>(
                            ctr, ctr_i_off, ctr_j_off, nctr_i, coeffs, coff_j, prev_pj, nctr_j,
                            block_len, lane_u, lanes_u, jempty,
                        );
                        jempty = 0u32;
                    }
                    if jempty == 0u32 {
                        stage_contract::<F>(
                            ctr,
                            ctr_j_off,
                            ctr_k_off,
                            nctr_i * nctr_j,
                            coeffs,
                            coff_k,
                            pk,
                            nctr_k,
                            block_len,
                            lane_u,
                            lanes_u,
                            kempty,
                        );
                        kempty = 0u32;
                    }
                }
            } // end ket `ccekl` cutoff
            kl_row += 1u32;
        }
        // The last `l` primitive's `gctrk` (staged path).
        if use_staged && kempty == 0u32 {
            stage_contract_out::<F>(
                ctr, cart_out, ctr_k_off, out_off, coeffs, coff_l, prev_pl, nctr_i, nctr_j, nctr_k,
                nctr_l, block_len, lane_u, lanes_u,
            );
        }

        // S2: one write per element this lane owns, replacing one
        // read-modify-write per primitive quartet.
        if use_acc {
            let mut oi = lane;
            let mut oslot: u32 = 0u32;
            while oi < out_len {
                cart_out[(out_off + oi) as usize] = acc[oslot as usize];
                oi += lanes;
                oslot += 1u32;
            }
        }
        if comptime!(per_unit == 0u32) {
            sync_cube();
        }

        qi += qi_step;
    }
}

/// The S2 accumulator ceiling, with `CINTX_2E_ACCUMULATE` applied.
///
/// `global` disables the private accumulator and restores the per-primitive
/// read-modify-write into `cart_out`. Anything else — including unset — uses
/// [`ACC_SLOTS_DEFAULT`].
///
/// A runtime scalar rather than a comptime one, so both settings are the *same
/// compiled program*: an A/B that recompiled the kernel would be measuring the
/// JIT as much as the change. It doubles as a real knob for a backend whose
/// per-work-item storage is tighter than this host's.
pub fn accumulator_slots_max() -> u32 {
    let current = ACCUMULATOR_SLOTS_MAX.load(std::sync::atomic::Ordering::Relaxed);
    if current != u32::MAX {
        return current;
    }
    let from_env = if std::env::var("CINTX_2E_ACCUMULATE")
        .is_ok_and(|value| value.eq_ignore_ascii_case("global"))
    {
        0
    } else {
        ACC_SLOTS_DEFAULT as u32
    };
    ACCUMULATOR_SLOTS_MAX.store(from_env, std::sync::atomic::Ordering::Relaxed);
    from_env
}

/// `u32::MAX` until [`accumulator_slots_max`] resolves the environment.
static ACCUMULATOR_SLOTS_MAX: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

/// Override the S2 accumulator ceiling for the rest of this process.
///
/// Exists so the A/B can alternate settings *inside one process*, interleaved
/// and repeated. Absolute times on the development host vary by up to 2x
/// between processes for identical work, so a cross-process comparison of two
/// settings measures the machine rather than the change — which is exactly the
/// trap this setter is here to avoid.
pub fn set_accumulator_slots_max(slots: u32) {
    ACCUMULATOR_SLOTS_MAX.store(slots, std::sync::atomic::Ordering::Relaxed);
}

/// The staged-contraction switch, with `CINTX_2E_CONTRACT` applied.
///
/// `1` (the default) contracts a generally contracted quartet in libcint's
/// four stages (`stage_contract`); `naive`, i.e. `0`, restores the
/// per-primitive-quartet fold into every contraction block. A segmented
/// quartet (`nctr == 1` on all four shells) is unaffected by either setting.
///
/// A runtime scalar rather than a comptime one, for the reason
/// [`accumulator_slots_max`] gives: the A/B must be one compiled program.
pub fn contraction_mode() -> u32 {
    let current = CONTRACTION_MODE.load(std::sync::atomic::Ordering::Relaxed);
    if current != u32::MAX {
        return current;
    }
    let from_env = if std::env::var("CINTX_2E_CONTRACT")
        .is_ok_and(|value| value.eq_ignore_ascii_case("naive"))
    {
        0
    } else {
        1
    };
    CONTRACTION_MODE.store(from_env, std::sync::atomic::Ordering::Relaxed);
    from_env
}

/// `u32::MAX` until [`contraction_mode`] resolves the environment.
static CONTRACTION_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Override the contraction mode for the rest of this process: `true` is the
/// staged scheme, `false` the naive fold. For in-process A/B measurement, as
/// [`set_accumulator_slots_max`] is.
pub fn set_staged_contraction(staged: bool) {
    CONTRACTION_MODE.store(u32::from(staged), std::sync::atomic::Ordering::Relaxed);
}

/// Measurement probe: run the G build without the contraction, for the rest
/// of this process. **The output is undefined** — every element the kernel
/// would have contracted is left as whatever the buffer held — so this is
/// for attributing time inside a profile run and nothing else; `gth_profile`
/// is its only caller. [`set_staged_contraction`] restores a real mode.
#[doc(hidden)]
pub fn set_contraction_probe() {
    set_contraction_mode(2);
}

/// Set the raw contraction mode: `0` naive, `1` staged, `2` the no-contraction
/// probe, `3` the no-G-build probe (roots, VRR and HRR skipped; the staged
/// contraction runs over a stale slab), `4` the no-roots probe (VRR/HRR run on
/// stale roots). Every probe's output is undefined. Measurement aid only.
#[doc(hidden)]
pub fn set_contraction_mode(mode: u32) {
    CONTRACTION_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
}

/// The per-unit partition switch, with `CINTX_2E_BALANCE` applied (K2).
///
/// `1` (the default) hands each unit a contiguous run of quartet rows whose
/// *estimated cost* is a `1/n_slots` share of the dispatch
/// ([`per_unit_slot_bounds`]); `uniform` (`0`) hands each unit
/// `ceil(n / n_slots)` rows, which was the only shape before K2. Runtime
/// rather than comptime for the reason [`accumulator_slots_max`] gives.
pub fn balance_mode() -> u32 {
    let current = BALANCE_MODE.load(std::sync::atomic::Ordering::Relaxed);
    if current != u32::MAX {
        return current;
    }
    let from_env = if std::env::var("CINTX_2E_BALANCE")
        .is_ok_and(|value| value.eq_ignore_ascii_case("uniform"))
    {
        0
    } else {
        1
    };
    BALANCE_MODE.store(from_env, std::sync::atomic::Ordering::Relaxed);
    from_env
}

/// `u32::MAX` until [`balance_mode`] resolves the environment.
static BALANCE_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Pin the per-unit partition for the rest of this process: `Some(true)` is
/// the cost-balanced partition, `Some(false)` the uniform one, `None`
/// re-reads `CINTX_2E_BALANCE`. For in-process A/B measurement.
pub fn set_two_e_balance(balanced: Option<bool>) {
    BALANCE_MODE.store(
        balanced.map_or(u32::MAX, u32::from),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Row bounds of the per-unit walk: `n_slots + 1` indices into a group's
/// quartet rows, slot `s` taking `[bounds[s], bounds[s + 1])` (K2).
///
/// Uniform: `ceil(n / n_slots)` rows per slot, the shape the kernel computed
/// for itself before K2. Balanced: contiguous runs cut where the prefix sum
/// of `cost` crosses each `1/n_slots` share of the total, so a dispatch whose
/// rows are appended class by class — cheap `(ss|ss)` first, `(pp|pp)` with
/// `7^4` primitive quartets and 81 contraction blocks last — no longer hands
/// the last unit all of the expensive rows. Both shapes cover every row
/// exactly once and are monotone, which is all the kernel relies on.
fn per_unit_slot_bounds(cost: &[u64], n_slots: usize, balanced: bool) -> Vec<u32> {
    let n = cost.len();
    let n_slots = n_slots.max(1);
    let mut bounds = Vec::with_capacity(n_slots + 1);
    if !balanced || n == 0 {
        let chunk = n.div_ceil(n_slots).max(1);
        for s in 0..=n_slots {
            bounds.push((s * chunk).min(n) as u32);
        }
        return bounds;
    }
    let total: u128 = cost.iter().map(|&c| u128::from(c)).sum();
    bounds.push(0);
    let mut cursor = 0_usize;
    let mut prefix: u128 = 0;
    for s in 1..n_slots {
        // The share this slot's range should end at, in cost units.
        let target = total * s as u128 / n_slots as u128;
        // Advance to the first row whose prefix reaches the target, then take
        // one more when that cut lands closer — whole rows only.
        while cursor < n && prefix + u128::from(cost[cursor]) <= target {
            prefix += u128::from(cost[cursor]);
            cursor += 1;
        }
        // Distances, not differences. `prefix` can already sit *past* the
        // target: an earlier slot's step-back carries the cursor beyond every
        // target it overshot, which happens whenever slots outnumber rows,
        // since consecutive targets are then closer together than one row is
        // wide. `abs_diff` keeps that case inside the same expression rather
        // than underflowing it, and it answers correctly — a cursor already
        // past the target only moves further away by taking another row.
        if cursor < n {
            let before = prefix.abs_diff(target);
            let after = (prefix + u128::from(cost[cursor])).abs_diff(target);
            if after < before {
                prefix += u128::from(cost[cursor]);
                cursor += 1;
            }
        }
        bounds.push(cursor as u32);
    }
    bounds.push(n as u32);
    bounds
}

/// Relative cost of one quartet row, for [`per_unit_slot_bounds`] (K2).
///
/// Primitive quartets after the pair screen, times the per-primitive-quartet
/// work: the contraction's `3 * nroots` G loads per Cartesian element plus
/// the `i` stage's `nctr_i` accumulations, and the VRR/HRR over the three
/// axes of the G tensor. A proxy, not a model — it only has to rank rows.
fn quartet_cost_estimate(prim_quartets: u64, params: &TwoEClassParams, nctr_i: u32) -> u64 {
    let block_len = (ncart(params.li as u8)
        * ncart(params.lj as u8)
        * ncart(params.lk as u8)
        * ncart(params.ll as u8)) as u64;
    let per_prim = block_len * (3 * u64::from(params.nroots) + 2 * u64::from(nctr_i))
        + 10 * u64::from(params.g_size)
        + prim_fixed_cost();
    prim_quartets.max(1) * per_prim
}

/// What a primitive quartet costs *before* any of its arithmetic (§23).
///
/// [`quartet_cost_estimate`]'s other two terms are proportional to the work:
/// the contraction's `3 · nroots` G loads per Cartesian element and the
/// VRR/HRR over the three axes. This one is what a primitive quartet costs
/// merely for being one — the pair rows and exponents, the `f64` division and
/// square root of the screen, the Rys solve, and on the cooperative shape the
/// four barriers a block pays. It is flat in the class.
///
/// # Why it is not small
///
/// It was `100`, which put an `(ss|ss)` primitive quartet at a twentieth of a
/// `(pp|pp)` one. §22.6 measured the ratio on gfx1151 at nearer a third and
/// left the constant alone because it should be measured rather than guessed;
/// [§23.2] is that measurement. The term matters because it is what
/// [`kl_split_plan`] ranks by: under-charging the narrow classes leaves their
/// rows unsplit, and a dispatch costs as long as its longest row — on
/// H2O/TZVP-MOLOPT two thirds of the run is the launch floor, and an unsplit
/// `(ss|ss)` row of 2 401 primitive quartets is what sets it.
///
/// Raising it is *not* the same as lowering [`KL_SPLIT_TARGET_PART_COST`].
/// The target cost scales every class alike, and the classes whose partial
/// blocks are expensive are the wide ones; this term is flat, so it splits the
/// narrow rows — whose blocks are a few hundred bytes — and leaves the wide
/// ones, already at their ket-range cap, where they are.
///
/// `CINTX_2E_COST_FIXED` overrides it, which is how it was swept.
fn prim_fixed_cost() -> u64 {
    use std::sync::OnceLock;
    static FIXED: OnceLock<u64> = OnceLock::new();
    *FIXED.get_or_init(|| {
        std::env::var("CINTX_2E_COST_FIXED")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(PRIM_FIXED_COST)
    })
}

/// The default [`prim_fixed_cost`], swept in §23.2.
const PRIM_FIXED_COST: u64 = 100;

/// The most [`quartet_cost_estimate`] work one item of the ket-pair split may
/// carry (G1, §18).
///
/// A work item is a cube on the cooperative shape and a row of K2's partition
/// on the per-unit one, and the split serves both: **occupancy** there — one
/// quartet per cube leaves a small molecule with a few dozen workgroups on a
/// GPU of tens of compute units (§10.5) — and **balance** here, because K2 cuts
/// whole rows, so a dispatch whose costliest quartet exceeds a `1/units` share
/// of the total cannot be balanced until that quartet is divisible.
///
/// # Why this is an absolute cost and not a share of the dispatch
///
/// It was `total / (8 · units)` until §18, and that made the split a function
/// of *how the work list happened to be batched*. Two consequences, and both
/// are the kind of thing that should not be true:
///
/// - **CPU and GPU split the same quartet differently**, because they chunk
///   differently and report different `parallel_units`. The two backends then
///   disagree for a reason that has nothing to do with the hardware.
/// - **Chunking changed the answer.** A run under `memory_limit_bytes`, or the
///   same quartets evaluated in two calls instead of one, re-associated the ket
///   sum differently — `chunked_evaluation_is_bit_identical_to_unchunked` and
///   `tuned_and_untuned_dispatches_agree_bit_for_bit` are the contracts that
///   says otherwise, and they are worth more than the targeting was.
///
/// Against an absolute cap the split is a pure function of the quartet, so the
/// same quartet splits the same way in every batch, on every backend, and the
/// pre-flight memory plan can predict it exactly — [`plan_batch_bytes`] charges
/// what [`kl_split_plan`] will ask for, with no client and no guessing.
///
/// The value is [swept in §18.3]; `CINTX_2E_KL_SPLIT_COST` overrides it.
const KL_SPLIT_TARGET_PART_COST: u64 = 1_000_000;

/// Ceiling on the partial blocks **one quartet** may add, in bytes (G1, §18).
///
/// The whole-group budget this replaces was the other half of the batch
/// dependence: which quartets got narrowed depended on what else was in the
/// dispatch. Per quartet it is again a pure function of the quartet — and it is
/// what bounds `(parts - 1) · block` for a wide class, where the ket-range cap
/// alone would allow 63 copies of a 10 000-element `(ff|ff)` block.
const KL_SPLIT_PARTIAL_BUDGET_BYTES_PER_QUARTET: usize = 1024 * 1024;

/// The per-part cost cap, with `CINTX_2E_KL_SPLIT_COST` applied.
fn kl_split_part_cost() -> u64 {
    use std::sync::OnceLock;
    static COST: OnceLock<u64> = OnceLock::new();
    *COST.get_or_init(|| {
        std::env::var("CINTX_2E_KL_SPLIT_COST")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(KL_SPLIT_TARGET_PART_COST)
    })
}

/// Widest ket-pair split (G1). A GTH-MOLOPT ket has at most 49 primitive
/// pairs, a def2-TZVP sulfur ket at most 36 surviving ones; beyond this a
/// part would hold no rows at all.
const KL_SPLIT_MAX: usize = 64;

/// The ket-pair split override, with `CINTX_2E_KL_SPLIT` applied (G1).
///
/// `None` (the default, or `auto`) lets [`kl_split_plan`] size the split
/// from the hardware; `Some(1)` (`0`, `1`, `off`) disables it; `Some(n)`
/// pins it. The A/B switch for the GPU profile.
fn kl_split_override() -> Option<usize> {
    let mut current = KL_SPLIT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if current == KL_SPLIT_UNRESOLVED {
        current = match std::env::var("CINTX_2E_KL_SPLIT").ok().as_deref() {
            None | Some("") | Some("auto") => KL_SPLIT_AUTO,
            Some("off") => 1,
            Some(value) => value.parse::<u32>().map_or(KL_SPLIT_AUTO, |n| n.max(1)),
        };
        KL_SPLIT_OVERRIDE.store(current, std::sync::atomic::Ordering::Relaxed);
    }
    (current != KL_SPLIT_AUTO).then_some(current as usize)
}

const KL_SPLIT_UNRESOLVED: u32 = u32::MAX;
const KL_SPLIT_AUTO: u32 = u32::MAX - 1;
static KL_SPLIT_OVERRIDE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(KL_SPLIT_UNRESOLVED);

/// Pin the ket-pair split (G1) for the rest of this process: `Some(1)` turns
/// it off, `Some(n)` spreads every cooperative quartet over `n` cubes, `None`
/// restores the hardware-sized default. For in-process A/B measurement, and
/// for exercising the split on the CPU backend's pinned cooperative arm,
/// where the default never selects it.
pub fn set_two_e_kl_split(parts: Option<u32>) {
    KL_SPLIT_OVERRIDE.store(
        parts.map_or(KL_SPLIT_AUTO, |n| n.max(1)),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// How many ket-pair parts each quartet of `group` is split into (G1, §16).
///
/// One count per quartet row, in row order. `1` means "not split": that
/// quartet keeps the whole ket range, writes straight into the group's output
/// and costs no partial buffer at all.
///
/// **A pure function of the quartet.** Nothing here reads the group's total,
/// the backend, the chunking or the unit count, and that is the point (§18):
/// the same quartet splits the same way in every batch and on every backend, so
/// re-associating the ket sum cannot make the answer depend on how the caller
/// happened to slice the work list, and CPU and GPU differ only where the
/// hardware makes them.
///
/// # Why per quartet and not per group
///
/// The split was sized once per group until §16, and that was survivable only
/// while a group held one Rys order — a dispatch of `(dd|dd)` quartets is
/// uniform, so one number fits it. F1 (§15) made groups *heterogeneous*: one
/// dispatch now carries an `(ss|ss)` quartet next to a `(dd|dd)` one whose
/// serial walk is two hundred times longer, and a per-group split either
/// starves the long one or buys the short one parts it cannot use. The
/// measurement (§16.1) is unambiguous: forcing the fused cooperative dispatch
/// back to the split the *unfused* one chose turned 0.82x into 1.29x, while
/// pinning the cube width — the other mechanism §15.4 suspected — changed
/// nothing at all.
///
/// # The rule
///
/// A work item's serial cost is its quartet's cost over its part count, so give
/// each quartet `ceil(cost / KL_SPLIT_TARGET_PART_COST)` parts — bounded by its
/// own ket range (a part with no rows is waste), by [`KL_SPLIT_MAX`], and by
/// the partial blocks one quartet may add
/// ([`KL_SPLIT_PARTIAL_BUDGET_BYTES_PER_QUARTET`]). Cheap quartets fall out at
/// one part and cost nothing; the expensive few take the parts.
///
/// A pinned `CINTX_2E_KL_SPLIT=n` still means "n parts for every quartet",
/// clamped per row, which is what the forced-split gates rely on.
fn kl_split_plan(group: &TwoELaunchGroup) -> Vec<u32> {
    let n = group.len();
    if n == 0 {
        return Vec::new();
    }
    let block_lens = quartet_block_lens(&group.quartets, group.out_len);
    // A quartet can be cut into at most as many parts as it has ket rows, as
    // `KL_SPLIT_MAX` allows, and as its own partial-block budget affords.
    let ceilings: Vec<u32> = group
        .quartets
        .chunks_exact(QUARTET_ROW_STRIDE)
        .zip(&block_lens)
        .map(|(row, &block)| {
            let by_rows = (row[7] - row[6]).max(1) as usize;
            let by_bytes = 1 + KL_SPLIT_PARTIAL_BUDGET_BYTES_PER_QUARTET
                / (block * std::mem::size_of::<f64>()).max(1);
            by_rows.min(KL_SPLIT_MAX).min(by_bytes).max(1) as u32
        })
        .collect();

    if let Some(pinned) = kl_split_override() {
        return ceilings
            .iter()
            .map(|&ceiling| (pinned as u32).clamp(1, ceiling))
            .collect();
    }
    let per_part = u128::from(kl_split_part_cost());
    group
        .quartet_cost
        .iter()
        .zip(&ceilings)
        .map(|(&cost, &ceiling)| {
            let want = u128::from(cost).div_ceil(per_part);
            (want.min(u128::from(ceiling)) as u32).max(1)
        })
        .collect()
}

/// The `f64` a group's ket-pair split adds past its `out_len` (G1, §18).
///
/// The same expression [`expand_kl_split`] lays out, so the pre-flight memory
/// plan and the real allocation cannot drift apart — which is what lets the
/// split run under a `memory_limit_bytes` at all.
fn kl_split_extra_len(group: &TwoELaunchGroup, splits: &[u32]) -> usize {
    quartet_block_lens(&group.quartets, group.out_len)
        .iter()
        .zip(splits)
        .map(|(&block, &parts)| (parts as usize - 1) * block)
        .sum()
}

/// Cartesian output elements each quartet row of a group writes.
///
/// Derived from the rows rather than stored: `build_launch_groups` appends
/// quartets in order with `out_len` accumulating, so a row's block is the gap
/// to the next row's offset and the last row's is the gap to `out_len`.
fn quartet_block_lens(rows: &[u32], out_len: usize) -> Vec<usize> {
    let offsets: Vec<usize> = rows
        .chunks_exact(QUARTET_ROW_STRIDE)
        .map(|row| row[4] as usize)
        .collect();
    (0..offsets.len())
        .map(|q| offsets.get(q + 1).copied().unwrap_or(out_len) - offsets[q])
        .collect()
}

/// What [`expand_kl_split`] lays out for one group.
struct KlSplitLayout {
    /// The expanded quartet rows, [`QUARTET_ROW_STRIDE`] `u32` each.
    rows: Vec<u32>,
    /// [`KL_REDUCE_ROW_STRIDE`] `u32` per **split** quartet.
    reduce_table: Vec<u32>,
    /// One [`quartet_cost_estimate`] per *expanded* row, in row order.
    ///
    /// K2 partitions rows, not quartets, so it needs a cost per row or its
    /// slice of `quartet_cost` is the wrong length and the wrong shape. A
    /// part's cost is its quartet's over the part count: the parts tile the ket
    /// range in equal-sized pieces and the cost estimate is linear in it.
    row_cost: Vec<u64>,
    /// `f64` the partial region needs beyond the group's `out_len`.
    extra_len: usize,
}

/// Spread each row of `rows` over its own number of ket-pair parts (G1, §16).
///
/// **Part 0 of every quartet writes straight into the group's output** at the
/// offset it already carried, and parts 1.. write into a compact region past
/// `out_len`, quartet by quartet. So a quartet that is not split costs nothing
/// — no partial block, no copy — which is what makes a per-quartet plan cheap
/// enough to be the default: on a fused dispatch most quartets take one part.
///
/// The reduce table is four `u32` per **split** quartet — `[extra_base,
/// extra_parts, out_off, block_len]`, `extra_base` absolute in the combined
/// buffer. A quartet that took one part is absent: its answer is already in
/// place and [`reduce_kl_partials`] accumulates in place, so there is nothing
/// to carry.
///
/// Ket rows stay in libcint's `(pl, pk)` order inside each part, so a part's
/// accumulation is a contiguous slice of the vendor's; only the final sum over
/// parts re-associates it, and it sums them in part order, so the result is
/// deterministic. A part may be empty — its block is zeroed and nothing else.
fn expand_kl_split(
    rows: &[u32],
    splits: &[u32],
    block_lens: &[usize],
    costs: &[u64],
    out_len: usize,
) -> KlSplitLayout {
    let n = splits.len();
    let total_rows: usize = splits.iter().map(|&s| s as usize).sum();
    let mut expanded = Vec::with_capacity(total_rows * QUARTET_ROW_STRIDE);
    let mut table = Vec::with_capacity(n * KL_REDUCE_ROW_STRIDE);
    let mut row_cost = Vec::with_capacity(total_rows);
    let mut extra = 0usize;
    for (q, row) in rows.chunks_exact(QUARTET_ROW_STRIDE).enumerate() {
        let parts = splits[q].max(1);
        let (lo, hi) = (row[6], row[7]);
        let len = hi - lo;
        let block = block_lens[q];
        let part_cost = costs[q].div_ceil(u64::from(parts));
        // Only a quartet that actually split needs reducing; the rest already
        // hold their answer at `row[4]` and are left alone (§17). The base is
        // **absolute**, the same offset the part rows carry, so the reduce needs
        // no knowledge of where the partial region starts.
        if parts > 1 {
            table.extend_from_slice(&[(out_len + extra) as u32, parts - 1, row[4], block as u32]);
        }
        for part in 0..parts {
            let a = lo + len * part / parts;
            let b = lo + len * (part + 1) / parts;
            // Part 0 keeps the quartet's own output offset; the rest take a
            // block of the partial region, which starts at `out_len`.
            let out_off = if part == 0 {
                row[4]
            } else {
                (out_len + extra + (part as usize - 1) * block) as u32
            };
            expanded.extend_from_slice(&[row[0], row[1], row[2], row[3], out_off, row[5], a, b]);
            row_cost.push(part_cost);
        }
        extra += (parts as usize - 1) * block;
    }
    KlSplitLayout {
        rows: expanded,
        reduce_table: table,
        row_cost,
        extra_len: extra,
    }
}

/// `u32` per quartet of [`reduce_kl_partials`]'s table:
/// `extra_base, extra_parts, out_off, block_len`.
const KL_REDUCE_ROW_STRIDE: usize = 4;

/// Fold a ket-split dispatch's partial blocks into the group's output (G1,
/// §16): one cube per **split** quartet, grid-stride over them, its units
/// striding the quartet's Cartesian block.
///
/// `combined` holds the group's output in `[0, out_len)` — part 0 of every
/// quartet, written there by the evaluation kernel itself — followed by the
/// compact partial region [`expand_kl_split`] laid out. The extras are summed
/// **in place** onto part 0, so a quartet that took no parts is not named in
/// the table and is not touched at all: the group's output is already where it
/// belongs, and the caller trims the partial region off the handle rather than
/// copying `out_len` elements past it (§17).
///
/// A quartet's parts are summed `p = 0, 1, …` in a fixed order, so the result
/// is deterministic; starting the accumulator from part 0's value in place is
/// the same sequence of additions the per-group reduce performed.
///
/// The read and the write are the same buffer and never the same element: a
/// work item owns one `(quartet, element)` and reads only that quartet's parts,
/// which live past `out_len` and are written by nobody here.
#[cube(launch_unchecked)]
fn reduce_kl_partials<F: Float>(
    combined: &mut Array<F>,
    table: &Array<u32>,
    n_entries: u32,
    n_cubes: u32,
    #[comptime] row_stride: u32,
) {
    let width = CUBE_DIM as u32;
    let mut q = CUBE_POS as u32;
    while q < n_entries {
        let t = q * row_stride;
        let base = table[t as usize];
        let extra_parts = table[(t + 1u32) as usize];
        let out_off = table[(t + 2u32) as usize];
        let block = table[(t + 3u32) as usize];
        let mut e = UNIT_POS as u32;
        while e < block {
            let mut acc = combined[(out_off + e) as usize];
            let mut p = 0u32;
            while p < extra_parts {
                acc += combined[(base + p * block + e) as usize];
                p += 1u32;
            }
            combined[(out_off + e) as usize] = acc;
            e += width;
        }
        q += n_cubes;
    }
}

/// Run [`reduce_kl_partials`] over `combined`, in place.
///
/// One cube per split quartet, capped by the grid. A quartet's Cartesian block
/// is the parallel axis, so the cube's units stride it — which is why the width
/// follows the backend's *parallelism* rather than its planes. On a plane-less
/// runtime `backend_plane_cube_dim` is one unit, and one unit walking every
/// element is a sequential pass over the whole output; §17 needs this path on
/// the CPU, so it takes the per-unit width there instead.
fn reduce_kl_partials_inplace<R: Runtime>(
    client: &ComputeClient<R>,
    combined: &cubecl::server::Handle,
    combined_len: usize,
    table: &[u32],
    probe: &Arc<Mutex<crate::memory_probe::DeviceMemoryProbe>>,
) {
    let n_entries = (table.len() / KL_REDUCE_ROW_STRIDE) as u32;
    if n_entries == 0 {
        return;
    }
    let table_h = client.create_from_slice(u32::as_bytes(table));
    probe
        .lock()
        .expect("device memory probe poisoned")
        .charge_tables(std::mem::size_of_val(table), 1);
    let cube_dim = if crate::plane::launch_hardware(client).has_planes {
        crate::plane::backend_plane_cube_dim::<R>(client)
    } else {
        CubeDim::new_1d(crate::plane::per_unit_width(
            client,
            combined_len,
            1,
            usize::MAX,
        ))
    };
    let cubes = crate::plane::grid_cube_count(client, n_entries as usize);
    // SAFETY: `combined` holds `combined_len` elements. Every table row names a
    // `[out_off, block_len)` inside the group's output and an
    // `[extra_base, extra_base + extra_parts * block_len)` inside the partial
    // region, both laid out by `expand_kl_split` within `combined_len`.
    unsafe {
        reduce_kl_partials::launch_unchecked::<f64, R>(
            client,
            crate::plane::cube_count_1d(cubes),
            cube_dim,
            ArrayArg::from_raw_parts(combined.clone(), combined_len),
            ArrayArg::from_raw_parts(table_h, table.len()),
            n_entries,
            cubes,
            KL_REDUCE_ROW_STRIDE as u32,
        );
    }
}

/// The cooperative G-build switch, with `CINTX_2E_COOP_BUILD` applied.
///
/// `lane0` restores the pre-S3 shape — lane 0 builds the whole G tensor while
/// the rest of the cube waits at the barrier. Anything else, including unset,
/// splits the `3 * nroots` `(axis, root)` slices across the cube.
///
/// Both settings produce **bit-identical** values: the slices are disjoint in
/// a root-fastest layout whose every stride is a multiple of `nroots`, so an
/// element is computed by the same expression from the same inputs either way,
/// only on a different lane. That is what makes the timing question separable
/// from the correctness one, and `two_e_cooperative_arm.rs` asserts it against
/// the per-unit arm.
pub fn cooperative_build_mode() -> u32 {
    let current = COOPERATIVE_BUILD_MODE.load(std::sync::atomic::Ordering::Relaxed);
    if current != u32::MAX {
        return current;
    }
    let from_env = u32::from(
        !std::env::var("CINTX_2E_COOP_BUILD")
            .is_ok_and(|value| value.eq_ignore_ascii_case("lane0")),
    );
    COOPERATIVE_BUILD_MODE.store(from_env, std::sync::atomic::Ordering::Relaxed);
    from_env
}

/// `u32::MAX` until [`cooperative_build_mode`] resolves the environment.
static COOPERATIVE_BUILD_MODE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

/// Override the cooperative G-build mode for the rest of this process:
/// `true` splits the build across lanes (S3), `false` keeps it on lane 0.
/// For in-process A/B measurement, as [`set_accumulator_slots_max`] is.
pub fn set_cooperative_build_split(split: bool) {
    COOPERATIVE_BUILD_MODE.store(u32::from(split), std::sync::atomic::Ordering::Relaxed);
}

/// Is the shared-memory G tensor enabled (S3 / B1)?
///
/// **On by default since B1 (plan §22)**; `CINTX_2E_SHARED_G=0` opts out to
/// the per-slot global slab, which is the A/B arm. The cooperative arm keeps
/// its G tensor in a [`SHARED_G_TIERS`]-sized `SharedMemory` region whenever
/// the dispatch's widest class fits one and the backend reports room.
///
/// # History
///
/// This was first recorded as a ROCm backend defect: with the shared slab
/// selected, `int2e_sph` came back as uninitialised memory. The bisect is in
/// `.planning/notes/rocm-shared-memory-miscompile.md`, and its conclusion was
/// wrong. The fault was in cintx — the S3 edit rebound 6 of the 46 G-tensor
/// accesses to `g_slab` and left the whole VRR/HRR indexing the raw `g`
/// parameter, so under `g_in_shared` the recurrence ran across two buffers.
/// Every access now goes through `g_slab` (the invariant is stated at its
/// declaration), and `def2_batch_rocm_parity` passes with the switch set,
/// with the same numbers as the global-slab control.
///
/// Two things the bisect *did* establish and are worth keeping: compiled
/// kernels are cached by `KernelId`, which hashes no body or source, so a
/// body-only edit can run the previous binary (`rm -rf crates/*/target/hip
/// target/hip`, or change the signature); and an output buffer from
/// `client.empty` can read back as an earlier launch's data, so a probe must
/// seed it with a sentinel.
///
/// The single 48 KiB allocation this switch first selected measured
/// 0.31x–0.57x on gfx1151 (§22.1) — one workgroup per compute unit. The
/// tiers, and the block of primitive quartets each tier holds, are what
/// turned the same shared region into the win §22 records.
fn shared_g_enabled() -> bool {
    let current = SHARED_G_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
    if current != u32::MAX {
        return current != 0;
    }
    let from_env = u32::from(!std::env::var("CINTX_2E_SHARED_G").is_ok_and(|value| value == "0"));
    SHARED_G_ENABLED.store(from_env, std::sync::atomic::Ordering::Relaxed);
    from_env != 0
}

/// `u32::MAX` until [`shared_g_enabled`] resolves the environment.
static SHARED_G_ENABLED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Put the cooperative G slab in shared memory for the rest of this process,
/// overriding `CINTX_2E_SHARED_G`.
///
/// Unlike the other measurement switches this one is **comptime** inside the
/// kernel (`shared_tier` selects which buffer the recurrences bind), so the
/// two settings are two compiled programs. An A/B across it has to warm both
/// before timing either, or it measures the JIT.
pub fn set_shared_g_enabled(enabled: bool) {
    SHARED_G_ENABLED.store(u32::from(enabled), std::sync::atomic::Ordering::Relaxed);
}

/// Read a positive-integer env override once per process.
///
/// Every `CINTX_2E_*` knob below is an A/B measurement aid, not part of the
/// public contract: unset, each falls back to the backend-derived default.
fn env_u32_override(var: &'static str) -> Option<u32> {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

/// Does this backend want the **one quartet per unit** decomposition?
///
/// Task 34-A0 established the CubeCL CPU runtime's shape: a cube unit is an OS
/// thread, `sync_cube` is a global spin-wait over every unit, and `cube_count`
/// lowers to a sequential `scf.for` *inside* each unit. So on CPU the cube is
/// the only real parallelism axis, and the way to use it is to give each unit a
/// whole quartet rather than a slice of one quartet's contraction — which also
/// removes the two barriers per primitive quartet entirely (they are
/// comptime-dropped at `per_unit == 1`).
///
/// On GPU backends the opposite holds — `sync_cube` is a cheap workgroup
/// barrier and the grid is real parallelism — so they keep the cooperative
/// one-quartet-per-cube shape from Task 34-B.
///
/// `CINTX_2E_PER_UNIT=0|1` pins it for A/B measurement.
fn two_e_per_unit<R: Runtime>(client: &ComputeClient<R>) -> bool {
    match two_e_per_unit_override() {
        Some(value) => value,
        None => !crate::plane::has_planes(client),
    }
}

/// `PER_UNIT_OVERRIDE` states: unresolved, backend default, or pinned.
const PER_UNIT_UNRESOLVED: u32 = u32::MAX;
const PER_UNIT_AUTO: u32 = u32::MAX - 1;
static PER_UNIT_OVERRIDE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(PER_UNIT_UNRESOLVED);

/// The pinned decomposition, or `None` to let the backend decide.
fn two_e_per_unit_override() -> Option<bool> {
    let mut current = PER_UNIT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if current == PER_UNIT_UNRESOLVED {
        current = std::env::var("CINTX_2E_PER_UNIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map_or(PER_UNIT_AUTO, |value| u32::from(value != 0));
        PER_UNIT_OVERRIDE.store(current, std::sync::atomic::Ordering::Relaxed);
    }
    if current == PER_UNIT_AUTO {
        None
    } else {
        Some(current != 0)
    }
}

/// Pin the decomposition for the rest of this process, overriding
/// `CINTX_2E_PER_UNIT`; `None` restores the backend default.
///
/// Exists so the cooperative arm — the shape every GPU backend runs, and the
/// one S3 parallelises — can be exercised on the CPU backend inside one
/// process, against the per-unit arm it must agree with bit for bit. That is
/// the S3 gate, and it needs no GPU to run.
///
/// The cooperative arm is *slow* on the CubeCL CPU runtime (a unit is an OS
/// thread and `sync_cube` a global spin barrier), so a caller pinning it
/// should also pin a narrow cube through [`set_two_e_cube_dim`] and keep the
/// work list short. It is a correctness vehicle there, never a timing one.
pub fn set_two_e_per_unit(per_unit: Option<bool>) {
    let value = per_unit.map_or(PER_UNIT_AUTO, u32::from);
    PER_UNIT_OVERRIDE.store(value, std::sync::atomic::Ordering::Relaxed);
}

/// Cube dimension for [`two_electron_scalar_kernel`].
///
/// Two regimes, matching [`two_e_per_unit`]:
///
/// - **per-unit (CPU)** — the cube dimension *is* the thread count, because
///   each unit owns a whole quartet. It is sized to
///   the backend's own core count (`hardware.num_cpu_cores`), clamped by the
///   quartet count (no point spawning threads with no quartet to take) and by
///   the per-unit G-slab budget.
/// - **cooperative (GPU)** — the contraction block is split across the cube
///   (`q_elem % lanes == lane`) and the G build runs on lane 0, so the useful
///   width is the contraction block length;
///   [`crate::plane::cooperative_cube_dim`] rounds it up to a whole number of
///   the backend's planes.
///
/// Task 34-A0 measured why the CPU case must never take the cooperative shape:
/// the kernel's two `sync_cube()` calls sit **inside** the primitive-quartet
/// loop, so a wide cube on the CPU runtime was 28x to ~4.9e5x slower
/// (`artifacts/34-A0_cube_dim_ab.md`).
///
/// `CINTX_2E_CUBE_DIM` pins it for A/B measurement and is not part of the
/// public contract.
static CUBE_DIM_OVERRIDE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

/// The pinned cube width, or `None` for the heuristic.
fn two_e_cube_dim_override() -> Option<u32> {
    let mut current = CUBE_DIM_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if current == u32::MAX {
        current = env_u32_override("CINTX_2E_CUBE_DIM").unwrap_or(0);
        CUBE_DIM_OVERRIDE.store(current, std::sync::atomic::Ordering::Relaxed);
    }
    if current == 0 { None } else { Some(current) }
}

/// Pin the cube width for the rest of this process, overriding
/// `CINTX_2E_CUBE_DIM`; `None` restores the heuristic. The companion to
/// [`set_two_e_per_unit`], for the same reason.
pub fn set_two_e_cube_dim(width: Option<u32>) {
    CUBE_DIM_OVERRIDE.store(width.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

fn two_e_cube_dim<R: Runtime>(
    client: &ComputeClient<R>,
    block_len: u32,
    n_quartets: usize,
    g_size: usize,
    ctr_len: usize,
) -> CubeDim {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let _ = &OVERRIDE;
    if let Some(dim) = two_e_cube_dim_override() {
        return CubeDim::new_1d(dim);
    }
    if two_e_per_unit::<R>(client) {
        return CubeDim::new_1d(per_unit_cube_dim(client, n_quartets, g_size, ctr_len));
    }
    crate::plane::cooperative_cube_dim(client, block_len)
}

/// Unit count for the per-unit decomposition.
///
/// Each unit needs its own `3 * g_size` G slab, so the thread count is capped
/// by [`MAX_BATCH_SCRATCH_BYTES`] as well as by hardware parallelism and by the
/// number of quartets actually available to take.
fn per_unit_cube_dim<R: Runtime>(
    client: &ComputeClient<R>,
    n_quartets: usize,
    g_size: usize,
    ctr_len: usize,
) -> u32 {
    let per_slab = slot_scratch_bytes(g_size, ctr_len);
    let by_memory = (MAX_BATCH_SCRATCH_BYTES / per_slab.max(1)).max(1);
    // One quartet is `nprim^4` primitive quartets through the full VRR/HRR build —
    // far more than the ~2 us it costs to wake a unit, so every available unit is
    // worth using even for a class of a few dozen quartets (measured: 16 units beat
    // 4 by ~1.8x on a 45-quartet class).
    crate::plane::per_unit_width(client, n_quartets, 1, by_memory)
}

/// Stride, in `f64` elements, between one slot's G slab and the next.
///
/// A slab holds `3 * g_size` elements; the stride rounds that up to a 64-byte
/// cache line so that concurrent slots — OS threads, in the per-unit
/// decomposition — never share a line while writing the G tensor.
fn g_slab_stride(g_size: usize) -> usize {
    /// f64 elements per 64-byte cache line.
    const LINE: usize = 8;
    (3 * g_size).div_ceil(LINE) * LINE
}

/// Stride, in `f64` elements, between one slot's staged-contraction scratch
/// and the next (GTH plan, C1); `0` when no quartet in the dispatch is
/// generally contracted. Padded to a cache line for the reason
/// [`g_slab_stride`] is.
fn ctr_slab_stride(ctr_len: usize) -> usize {
    const LINE: usize = 8;
    ctr_len.div_ceil(LINE) * LINE
}

/// `f64` elements of staged-contraction scratch one quartet needs: the three
/// intermediates `gctri`, `gctrj`, `gctrk` of [`stage_contract`], or `0` for
/// a segmented quartet, which never enters the staged path.
fn staged_ctr_len(nctr: [u32; 4], cart_block: usize) -> usize {
    let [ni, nj, nk, _] = nctr;
    if nctr.iter().all(|&n| n == 1) {
        return 0;
    }
    let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
    (ni + ni * nj + ni * nj * nk) * cart_block
}

/// Bytes of scratch one slot owns: its G slab plus its contraction slab.
fn slot_scratch_bytes(g_size: usize, ctr_len: usize) -> usize {
    (g_slab_stride(g_size) + ctr_slab_stride(ctr_len)) * std::mem::size_of::<f64>()
}

/// Ceiling on the per-launch G-tensor scratch slab, shared by both
/// decompositions ([`two_e_cube_count`] sizes cubes, [`per_unit_cube_dim`]
/// sizes units, and each owns one `3 * g_size` slab).
const MAX_BATCH_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

/// Class-uniform shape parameters shared by every quartet in one launch.
///
/// These derive from `(li,lj,lk,ll)` alone, which is exactly what
/// `cintx-driver`'s launch-class bucketing holds constant, so they are launch
/// arguments rather than per-quartet data.
#[derive(Clone, Copy, Debug)]
pub struct TwoEClassParams {
    pub li: u32,
    pub lj: u32,
    pub lk: u32,
    pub ll: u32,
    pub di: u32,
    pub dk: u32,
    pub dl: u32,
    pub dj: u32,
    pub g_size: u32,
    pub nmax: u32,
    pub mmax: u32,
    pub g2d_ijmax: u32,
    pub g2d_klmax: u32,
    pub ibase: u32,
    pub kbase: u32,
    pub nroots: u32,
    pub common_factor: f64,
}

impl TwoEClassParams {
    /// Derive the class parameters from an angular-momentum quartet.
    #[must_use]
    pub fn new(li: u8, lj: u8, lk: u8, ll: u8) -> Self {
        let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
        Self {
            li: li as u32,
            lj: lj as u32,
            lk: lk as u32,
            ll: ll as u32,
            di: shape.di as u32,
            dk: shape.dk as u32,
            dl: shape.dl as u32,
            dj: shape.dj as u32,
            g_size: shape.g_size as u32,
            nmax: shape.nmax as u32,
            mmax: shape.mmax as u32,
            g2d_ijmax: shape.g2d_ijmax as u32,
            g2d_klmax: shape.g2d_klmax as u32,
            ibase: shape.ibase as u32,
            kbase: shape.kbase as u32,
            nroots: shape.nroots as u32,
            common_factor: int2e_common_factor(li, lj, lk, ll),
        }
    }
}

/// Number of cubes to dispatch for a batch of `n_quartets`.
///
/// On the CubeCL CPU runtime `cube_count` lowers to a sequential `scf.for`
/// inside each unit (see [`crate::plane::cooperative_cube_dim`]), so a wide
/// grid buys nothing but multiplies the G-tensor scratch; one cube, walking the
/// list grid-stride, is both fastest and smallest. On GPU backends the grid is
/// the parallelism axis, so it is one cube per quartet, capped so the scratch
/// slab stays within `MAX_BATCH_SCRATCH_BYTES`.
fn two_e_cube_count<R: Runtime>(
    client: &ComputeClient<R>,
    n_quartets: usize,
    g_size: usize,
    ctr_len: usize,
) -> u32 {
    if two_e_per_unit::<R>(client) {
        // The units carry the parallelism; `cube_count` on the CPU runtime is a
        // sequential loop, so a second cube would only duplicate G slabs.
        return 1;
    }
    let per_cube = slot_scratch_bytes(g_size, ctr_len);
    let by_memory = (MAX_BATCH_SCRATCH_BYTES / per_cube.max(1)).max(1);
    // B1 (§22.4): one cube per row put thousands of cubes — and thousands of
    // contraction-scratch slabs, 158 MiB on SO2/TZVP-MOLOPT — into a dispatch
    // the device runs a few dozen cubes of at a time. The grid is capped at
    // [`cooperative_cubes_per_unit`] per compute unit and walks the rest
    // grid-stride; which cube evaluates a row cannot change its value.
    let hw = crate::plane::launch_hardware(client);
    let by_concurrency =
        (hw.parallel_units as usize * cooperative_cubes_per_unit() as usize).max(1);
    crate::plane::grid_cube_count(client, n_quartets.min(by_memory).min(by_concurrency))
}

/// Cubes per compute unit a cooperative 2e dispatch is capped at (B1, §22.4);
/// `CINTX_2E_CUBES_PER_UNIT` overrides it for a sweep.
///
/// Well above what any tier lets a unit hold resident (at most sixteen by
/// shared memory, fewer by waves), so the queue never runs dry, and the
/// per-cube scratch — the contraction stages, which stay in global memory —
/// is bounded by the machine rather than by the work list.
pub const COOPERATIVE_CUBES_PER_UNIT: u32 = 64;

fn cooperative_cubes_per_unit() -> u32 {
    use std::sync::OnceLock;
    static CAP: OnceLock<u32> = OnceLock::new();
    *CAP.get_or_init(|| {
        env_u32_override("CINTX_2E_CUBES_PER_UNIT").unwrap_or(COOPERATIVE_CUBES_PER_UNIT)
    })
}

/// Flattened basis shared by every launch class in one batched run.
///
/// Uploaded **once per run** rather than once per class (Task 34-C): the whole
/// point of batching is that the shell data stops being per-launch payload.
#[derive(Clone, Debug, Default)]
pub(crate) struct TwoEFlatBasis {
    /// Every shell's primitive exponents, concatenated.
    pub(crate) exps: Vec<f64>,
    /// Every shell's contraction coefficients, concatenated, primitive-major.
    pub(crate) coeffs: Vec<f64>,
    /// Three coordinates per shell.
    pub(crate) centers: Vec<f64>,
    /// `[exp_off, coeff_off, nprim, nctr]` per shell.
    pub(crate) shell_meta: Vec<u32>,
}

/// Flatten a shell list into the concatenated arrays the kernel indexes.
fn flatten_2e_basis(shells: &[BatchShell]) -> TwoEFlatBasis {
    let mut basis = TwoEFlatBasis::default();
    basis.centers.reserve(shells.len() * 3);
    basis.shell_meta.reserve(shells.len() * 4);
    for shell in shells {
        basis.shell_meta.push(basis.exps.len() as u32);
        basis.shell_meta.push(basis.coeffs.len() as u32);
        basis.shell_meta.push(shell.nprim);
        basis.shell_meta.push(shell.nctr);
        basis
            .exps
            .extend_from_slice(&shell.exponents[..shell.nprim as usize]);
        basis
            .coeffs
            .extend_from_slice(&shell.coefficients[..(shell.nprim * shell.nctr) as usize]);
        basis.centers.extend_from_slice(&shell.center);
    }
    basis
}

impl TwoEFlatBasis {
    /// Bytes this basis costs to upload.
    #[must_use]
    pub(crate) fn upload_bytes(&self) -> usize {
        (self.exps.len() + self.coeffs.len() + self.centers.len()) * std::mem::size_of::<f64>()
            + self.shell_meta.len() * std::mem::size_of::<u32>()
    }
}

/// The widest shared-memory G region (S3): the last of [`SHARED_G_TIERS`].
///
/// 6 144 elements is 48 KiB, the shared-memory budget a workgroup can count on
/// across the backends this project targets. It holds `3 * g_size` for every
/// def2 class up to `nroots = 5` — which is 99.7% of SO2/def2-TZVP's quartets
/// and every quartet of def2-SVP. The `nroots` 6 and 7 classes need 72 KiB and
/// 129 KiB and keep the global slab.
pub const SHARED_G_SLOTS: usize = 6144;

/// The comptime extents, in `f64` slots, a cooperative dispatch's shared G
/// region may take (B1, plan §22) — the smallest holding `3 * max_g_size`.
///
/// Three, not one: `SharedMemory::new` takes a comptime extent, so the extent
/// is part of the compiled program, and a single 48 KiB allocation admits one
/// workgroup per compute unit on gfx1151's 64 KiB — measured at 0.31x–0.57x
/// (§22.1). A per-class extent would be a program per class, which is the
/// dispatch merge Task 35-M1 made. Each tier is filled with as many primitive
/// quartets in flight as it holds (`b_max` in the kernel), so a `(pp|pp)`
/// class in the 16 KiB tier builds six G tensors per cube at once.
pub const SHARED_G_TIERS: [u32; 5] = [512, 1024, 2048, 4096, 6144];

/// How many primitive quartets a class's tier should hold at once, where a
/// tier that wide exists (B1). `CINTX_2E_B_TARGET` overrides it for a sweep.
///
/// The tier trades occupancy for block width: a 4 KiB tier admits sixteen
/// cubes per 64 KiB compute unit and a 16 KiB one four, while a `(pp|pp)`
/// class blocks one primitive quartet in the first and six in the second.
/// Swept at 1, 2, 4 and 16 on ROCm (§22.5): two was best on CH4 with either
/// basis and on H2O/TZVP-MOLOPT, and what it mostly tunes is how many tiers
/// — and so how many dispatches — a work list spreads over.
pub const B_TARGET_DEFAULT: u32 = 2;

fn b_target() -> u32 {
    use std::sync::OnceLock;
    static TARGET: OnceLock<u32> = OnceLock::new();
    *TARGET.get_or_init(|| env_u32_override("CINTX_2E_B_TARGET").unwrap_or(B_TARGET_DEFAULT))
}

/// Leading slots of the shared G region that carry one [`META_STRIDE`]-wide
/// row per primitive quartet of the block in flight; the G sub-slabs follow.
pub const G_META_SLOTS: u32 = B_MAX * META_STRIDE;

/// Slots per meta row: the builder's state (0 screened out, 1 under
/// tolerance, 2 live), the primitive weight, `pi`, `pj`, and the first four
/// `i`-contraction coefficients — everything the contraction phase would
/// otherwise reload through two dependent uniform global loads per row
/// (§22.5).
pub const META_STRIDE: u32 = 8;

/// Ceiling on primitive quartets a cube builds at once (B1).
///
/// The contraction phase walks the block serially, so the block is bounded
/// even where a tiny class would let the tier and the lanes admit hundreds;
/// and the meta rows cost `8 * B_MAX` slots of every tier.
pub const B_MAX: u32 = 32;

/// The shared-memory tier a class of G extent `g_size` dispatches under, or
/// `0` for the global slab (B1, plan §22).
///
/// A pure function of the class and the backend's shared-memory limit: the
/// smallest tier holding [`b_target`] G tensors of `3 * g_size`, or failing
/// that the widest tier holding one, among the tiers whose allocation (tier
/// plus the meta slots) the backend reports room for. `0` — the per-slot
/// global slab, one primitive quartet per cube — for a class wider than every
/// such tier.
pub fn class_shared_tier(g_size: usize, max_shared_bytes: usize) -> u32 {
    let need = 3 * g_size;
    let cap = shared_tier_cap();
    let fits = |tier: u32| {
        tier <= cap
            && (tier + G_META_SLOTS) as usize * std::mem::size_of::<f64>() <= max_shared_bytes
    };
    let want = need * b_target() as usize;
    if let Some(tier) = SHARED_G_TIERS
        .iter()
        .copied()
        .find(|&tier| want <= tier as usize && fits(tier))
    {
        return tier;
    }
    SHARED_G_TIERS
        .iter()
        .copied()
        .filter(|&tier| need <= tier as usize && fits(tier))
        .max()
        .unwrap_or(0)
}

/// The widest tier a class may take, in slots; wider classes keep the global
/// slab. `CINTX_2E_TIER_CAP` overrides it for a sweep.
///
/// A tier is the whole cube's shared memory, so a 32 KiB or 48 KiB tier
/// admits two or one cubes per 64 KiB compute unit — and for the classes
/// that need one (`(dd|dd)` at 3 375 slots) it holds a single G tensor, so it
/// buys latency and nothing else. Swept on ROCm against the committed kernel
/// in one session (§22.5): at 2 048 slots SO2 lost 10–20% on both bases while
/// CH4 and H2O gained 1.4x–1.6x; at 1 024, eight cubes per unit, SO2 gains
/// 1.1x–1.6x too and CH4/DZVP keeps 1.2x. No workload regresses at 1 024, which
/// is why it is the default; 512 loses everywhere.
pub const SHARED_G_TIER_CAP: u32 = 1024;

fn shared_tier_cap() -> u32 {
    use std::sync::OnceLock;
    static CAP: OnceLock<u32> = OnceLock::new();
    *CAP.get_or_init(|| env_u32_override("CINTX_2E_TIER_CAP").unwrap_or(SHARED_G_TIER_CAP))
}

/// The shared-memory budget a cooperative dispatch on `backend` may size its
/// tier from, or `None` where the G tensor stays in the global slab: the
/// per-unit decomposition (shared memory is per cube, and on the CPU runtime
/// it is ordinary cache anyway) and the `CINTX_2E_SHARED_G=0` opt-out.
///
/// Consulted once per plan, like the fusion decision, so the grouping a
/// pre-flight budget was computed for is the grouping that is dispatched.
///
/// Public for the same reason as [`two_e_nroots_fusion`]: it is the other
/// backend-level input to a launch signature. `None` — the per-slot global slab
/// — under the per-unit decomposition or when the shared-memory G tensor is
/// switched off.
pub fn shared_tier_limit(backend: &ResolvedBackend) -> Option<usize> {
    if !shared_g_enabled() {
        return None;
    }
    fn limit<R: Runtime>(client: &ComputeClient<R>) -> Option<usize> {
        if two_e_per_unit::<R>(client) {
            None
        } else {
            Some(client.properties().hardware.max_shared_memory_size)
        }
    }
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => limit::<cubecl::cpu::CpuRuntime>(client),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => limit::<cubecl_wgpu::WgpuRuntime>(client),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => limit::<cubecl_cuda::CudaRuntime>(client),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => limit::<cubecl_hip::HipRuntime>(client),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => limit::<cubecl_wgpu::WgpuRuntime>(client),
    }
}

/// The kernel's comptime signature — everything a dispatch must hold constant.
///
/// `two_electron_scalar_kernel` specializes on exactly three parameters:
/// `ibase` and `kbase` select the HRR branch, `nroots` selects the Rys root
/// solver and the unrolled root loops. Every other shape scalar is a runtime
/// value, so quartets that differ in `(li,lj,lk,ll)` but agree here can share a
/// dispatch (Task 35-M1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TwoELaunchSignature {
    pub ibase: u32,
    pub kbase: u32,
    /// The Rys **bucket**, not the Rys order (F1, §15).
    ///
    /// [`FUSED_NROOTS_BUCKET`] for every class the fixed-order solvers serve
    /// (`nroots <= 5`) — those share one dispatch and carry their own order in
    /// their class row — and the order itself above that, where the extended
    /// solver's comptime order makes fusion cost more than it saves.
    pub nroots: u32,
    /// The shared-memory tier the class's G tensor takes (B1, plan §22),
    /// `0` for the per-slot global slab — see [`class_shared_tier`].
    ///
    /// Part of the signature, not a per-dispatch choice, because the tier is
    /// comptime in the kernel and because a dispatch's cube width and tier are
    /// sized to its *widest* class: fusing a `(ss|ss)` class under a
    /// `(dd|dd)` one put every narrow row on a 256-lane cube at a 32 KiB tier,
    /// which measured 1.13x–1.56x *slower* than no fusion at all (§22.3).
    /// Keyed by tier, the narrow classes keep the narrow geometry.
    pub tier: u32,
}

/// The bucket every `nroots <= `[`MAX_FUSED_NROOTS`] class dispatches under.
///
/// Zero is not a Rys order, so it cannot collide with an unfused one.
pub const FUSED_NROOTS_BUCKET: u32 = 0;

/// The widest Rys order the fused dispatch carries (F1, §15).
///
/// One through five are the fixed-order polynomial solvers `rys_root1..5`:
/// short bodies, and five of them in one program is the same code the five
/// separate programs held. Six and above is `rys_roots_ext_dev`, whose
/// double-double Wheeler/Jacobi arms are an order of magnitude larger and
/// which takes its order at comptime; fusing those would emit seven of them
/// per dispatch to merge classes that carry a fraction of a percent of any
/// work list's primitive quartets. They keep a dispatch each.
pub const MAX_FUSED_NROOTS: u32 = 5;

impl TwoELaunchSignature {
    /// The signature an angular-momentum class dispatches under.
    ///
    /// `fuse` is [`two_e_nroots_fusion`] for the backend this batch runs on —
    /// the caller passes it rather than the callee reading it, so that one
    /// grouping decision covers a whole plan and cannot change between the
    /// pre-flight budget and the dispatch.
    ///
    /// `shared_limit` is the backend's shared-memory budget in bytes for a
    /// cooperative dispatch, or `None` where the G tensor stays in the global
    /// slab (the per-unit decomposition, or the switch off) — see
    /// [`shared_tier_limit`].
    #[must_use]
    pub fn of(params: &TwoEClassParams, fuse: bool, shared_limit: Option<usize>) -> Self {
        Self {
            ibase: params.ibase,
            kbase: params.kbase,
            nroots: if fuse && params.nroots <= MAX_FUSED_NROOTS {
                FUSED_NROOTS_BUCKET
            } else {
                params.nroots
            },
            tier: shared_limit.map_or(0, |limit| class_shared_tier(params.g_size as usize, limit)),
        }
    }
}

/// Does this backend's decomposition want the Rys-order fusion (F1, §15)?
///
/// **Both, since §16.** The fusion is a load-balancing change, and each
/// decomposition needed its own thing balanced first.
///
/// On the per-unit shape a dispatch's quartets are partitioned across every
/// unit, and a dispatch holding one `(dd|dd)` quartet leaves fifteen of sixteen
/// units idle while it runs; fusing the Rys orders into one dispatch per
/// `(ibase, kbase)` puts the whole work list in one partition, and that was
/// worth 1.16x-1.48x with nothing else changed (§15.3).
///
/// On the cooperative shape the same move first measured **0.82x**, and §15.4
/// guessed at two mechanisms. The attribution (§16.1) cleared one of them —
/// pinning the cube width changed the number by 0.000x — and convicted the
/// other: the ket-pair split was sized once per *group*, which
/// only ever fitted a dispatch of one Rys order. Sizing it per quartet
/// ([`kl_split_plan`]) is what makes a fused, heterogeneous dispatch work, and
/// with it the fusion is worth 1.2x-1.5x on ROCm as well.
///
/// Public because a launch signature cannot be derived without it: it is one of
/// the two backend-level decisions `TwoELaunchSignature::of` folds in, and
/// `def2_batches_launch_once_per_signature` re-derives the signature set from
/// the bucket list to check the dispatch count against something other than the
/// planner's own bookkeeping.
pub fn two_e_nroots_fusion() -> bool {
    nroots_fusion_override().unwrap_or(true)
}

/// The Rys-order launch-group fusion override (F1, §15), with `CINTX_2E_FUSE`
/// applied.
///
/// `None` is the backend default; `Some(false)` restores one dispatch per
/// `(ibase, kbase, nroots)`, which is the A/B this section's ratios are quoted
/// from; `Some(true)` forces the fusion on a decomposition that would not
/// choose it. Unlike the kernel's other switches this one *does* change the
/// compiled program (`nr_max` is comptime), so the two arms are two programs;
/// that is why it is a grouping switch read on the host rather than a kernel
/// scalar. Both arms produce bit-identical output on the per-unit shape, which
/// is what `gth_profile`'s dump comparison holds them to.
fn nroots_fusion_override() -> Option<bool> {
    let mut current = NROOTS_FUSION.load(std::sync::atomic::Ordering::Relaxed);
    if current == FUSE_UNRESOLVED {
        current = match std::env::var("CINTX_2E_FUSE").ok().as_deref() {
            None | Some("") | Some("auto") => FUSE_AUTO,
            Some("off") | Some("0") => 0,
            Some(_) => 1,
        };
        NROOTS_FUSION.store(current, std::sync::atomic::Ordering::Relaxed);
    }
    if current == FUSE_AUTO {
        None
    } else {
        Some(current == 1)
    }
}

/// `NROOTS_FUSION` states: unresolved, backend default, or pinned.
const FUSE_UNRESOLVED: u32 = u32::MAX;
const FUSE_AUTO: u32 = u32::MAX - 1;
static NROOTS_FUSION: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(FUSE_UNRESOLVED);

/// Pin the Rys-order fusion for the rest of this process: `Some(true)` fuses,
/// `Some(false)` restores one dispatch per Rys order, `None` restores the
/// backend default. For in-process A/B measurement.
///
/// A work list already planned is unaffected — the grouping is read when a
/// [`crate::ResidentTwoEBasis`] plans a batch, so set this before the call
/// whose grouping is being measured.
pub fn set_two_e_nroots_fusion(fused: Option<bool>) {
    NROOTS_FUSION.store(
        fused.map_or(FUSE_AUTO, u32::from),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Accumulator slots in the **per-unit** decomposition (S2).
///
/// A work item owns `ceil(block_len / lanes)` output elements, and in the
/// per-unit shape `lanes == 1`, so this ceiling is on the whole Cartesian block.
/// 256 f64 is 2 KB per unit — 32 KB across the 16 units of this host's CPU
/// decomposition.
///
/// It covers every `nroots <= 3` class, which is where 93-98% of all primitive
/// work sits on the def2 workloads. An `(ff|ff)` block is 10 000 Cartesian
/// elements and does not fit; that class takes the read-modify-write path it
/// always had, and carries under 0.1% of the work.
pub const ACC_SLOTS_DEFAULT: usize = 256;

/// Accumulator slots in the **cooperative** decomposition.
///
/// The array is per work item, and in the cooperative shape a work item is a
/// lane: at [`ACC_SLOTS_DEFAULT`] a 256-wide cube would carry 512 KB of private
/// storage and spill, costing far more occupancy than the read-modify-write it
/// replaces.
///
/// It can afford to be small because a lane's share shrinks as the cube widens.
/// `cooperative_cube_dim` sizes the cube from the block itself, so
/// `ceil(block_len / lanes)` stays near 1 for narrow classes and reaches 40 for
/// the widest `(ff|ff)` block at 256 lanes. 64 covers every class either way.
pub const ACC_SLOTS_COOPERATIVE: usize = 64;

/// Accumulator capacity for a decomposition, folded at JIT time.
const fn acc_capacity(per_unit: u32) -> usize {
    if per_unit == 1 {
        ACC_SLOTS_DEFAULT
    } else {
        ACC_SLOTS_COOPERATIVE
    }
}

/// `u32` per quartet row of the device quartet table:
/// `si, sj, sk, sl, out_off, class, kl_lo, kl_hi`.
///
/// `kl_lo..kl_hi` (G1) is the range of ket primitive-pair rows this row
/// evaluates — the whole of `pair_offset[sk*nbas+sl] ..` for an unsplit
/// quartet, a slice of it when [`expand_kl_split`] has spread the quartet
/// over several cubes. Carried on the row rather than derived in the kernel
/// so the split is a host-side table change and nothing else.
pub(crate) const QUARTET_ROW_STRIDE: usize = 8;

/// `u32` shape scalars per class row of the device shape table:
/// `li,lj,lk,ll,di,dk,dl,dj,g_size,nmax,mmax,g2d_ijmax,g2d_klmax,idx_off,nroots`.
///
/// `idx_off` (K1) is where the class's Cartesian index table starts in
/// [`TwoELaunchGroup::class_idx`], and `nroots` (F1, §15) is the class's Rys
/// order — a runtime column since orders are fused into one dispatch.
const TWO_E_SHAPE_STRIDE: usize = 15;

/// One dispatch: every quartet sharing a [`TwoELaunchSignature`] (Task 35-M1).
///
/// The group carries one shape row per angular-momentum class it merged, and
/// each quartet names its class in the sixth column of its table row. The
/// G-tensor slab is sized to the *widest* class in the group; the measured
/// spread over the def2-SVP envelope is `g_size` 27..144 within a signature, so
/// the slab stays tens of KB per slot and well inside
/// [`MAX_BATCH_SCRATCH_BYTES`].
#[derive(Clone, Debug)]
pub struct TwoELaunchGroup {
    /// The comptime parameters every quartet here shares.
    pub signature: TwoELaunchSignature,
    /// [`TWO_E_SHAPE_STRIDE`] `u32` per merged class.
    pub class_shape: Vec<u32>,
    /// One `common_factor` per merged class.
    pub class_factor: Vec<f64>,
    /// Every merged class's Cartesian index table, concatenated (K1): three
    /// `u32` G offsets per Cartesian element, `[base_x, base_y, base_z]`,
    /// in the contraction's element order (`l, k, j, i` descending
    /// components, `i` fastest); a class's table starts at the `idx_off` of
    /// its shape row. libcint's `idx` from `CINTg2e_index_xyz`.
    pub class_idx: Vec<u32>,
    /// [`QUARTET_ROW_STRIDE`] `u32` per quartet:
    /// `[si, sj, sk, sl, out_off, class, kl_lo, kl_hi]`.
    pub quartets: Vec<u32>,
    /// Total Cartesian output elements across this group's quartets.
    pub out_len: usize,
    /// Widest `g_size` in the group — what the per-slot G slab is sized to.
    pub max_g_size: u32,
    /// Widest Rys order merged into this dispatch (F1, §15) — the kernel's
    /// comptime `nr_max`. It sizes the private root arrays and decides which
    /// per-order arms the program emits; each quartet still takes the order in
    /// its own class row.
    pub max_nroots: u32,
    /// Widest Cartesian contraction block — the cooperative cube's parallel width.
    pub max_block_len: u32,
    /// Widest staged-contraction scratch any quartet here needs
    /// ([`staged_ctr_len`]); `0` when every quartet is segmented.
    pub max_ctr_len: u32,
    /// One [`quartet_cost_estimate`] per quartet row, in row order (K2).
    pub quartet_cost: Vec<u64>,
    /// Contraction quads (`nctr_i·nctr_j·nctr_k·nctr_l`) per quartet row, in
    /// row order (§19).
    ///
    /// The device cart-to-sph transform's unit of work is one `(quartet, quad)`
    /// pair, not a whole quartet, and only the grouping knows the shells' `nctr`
    /// — so it is recorded here rather than re-derived from `shell_meta` on the
    /// device or from the basis on the host.
    pub quad_count: Vec<u32>,
}

impl TwoELaunchGroup {
    /// An empty group for `signature`.
    #[must_use]
    pub fn new(signature: TwoELaunchSignature) -> Self {
        Self {
            signature,
            class_shape: Vec::new(),
            class_factor: Vec::new(),
            class_idx: Vec::new(),
            quartets: Vec::new(),
            out_len: 0,
            max_g_size: 0,
            max_nroots: 0,
            max_block_len: 0,
            max_ctr_len: 0,
            quartet_cost: Vec::new(),
            quad_count: Vec::new(),
        }
    }

    /// Append `params` as a new class and return the index quartet rows use.
    ///
    /// # Panics
    /// Panics if `params` does not carry this group's signature — merging a
    /// class under the wrong comptime parameters would silently evaluate it
    /// with another HRR branch or Rys order.
    pub fn push_class(&mut self, params: &TwoEClassParams) -> u32 {
        assert!(
            params.ibase == self.signature.ibase
                && params.kbase == self.signature.kbase
                && (self.signature.nroots == FUSED_NROOTS_BUCKET
                    || self.signature.nroots == params.nroots),
            "class does not belong to this launch group"
        );
        let index = self.class_factor.len() as u32;
        let idx_off = self.class_idx.len() as u32;
        self.class_shape.extend_from_slice(&[
            params.li,
            params.lj,
            params.lk,
            params.ll,
            params.di,
            params.dk,
            params.dl,
            params.dj,
            params.g_size,
            params.nmax,
            params.mmax,
            params.g2d_ijmax,
            params.g2d_klmax,
            idx_off,
            params.nroots,
        ]);
        self.class_factor.push(params.common_factor);
        // K1: the class's Cartesian index table, in the contraction's element
        // order — the same nest `contract_2e_cart` walks on the host.
        let (di, dk, dl, dj) = (params.di, params.dk, params.dl, params.dj);
        for &(lx, ly, lz) in &cart_comps(params.ll as u8) {
            for &(kx, ky, kz) in &cart_comps(params.lk as u8) {
                for &(jx, jy, jz) in &cart_comps(params.lj as u8) {
                    for &(ix, iy, iz) in &cart_comps(params.li as u8) {
                        self.class_idx.extend_from_slice(&[
                            u32::from(ix) * di
                                + u32::from(kx) * dk
                                + u32::from(lx) * dl
                                + u32::from(jx) * dj,
                            u32::from(iy) * di
                                + u32::from(ky) * dk
                                + u32::from(ly) * dl
                                + u32::from(jy) * dj,
                            u32::from(iz) * di
                                + u32::from(kz) * dk
                                + u32::from(lz) * dl
                                + u32::from(jz) * dj,
                        ]);
                    }
                }
            }
        }
        self.max_g_size = self.max_g_size.max(params.g_size);
        self.max_nroots = self.max_nroots.max(params.nroots);
        index
    }

    /// Number of quartets in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.quartets.len() / QUARTET_ROW_STRIDE
    }

    /// Is this group empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quartets.is_empty()
    }

    /// Number of angular-momentum classes merged into this dispatch.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_factor.len()
    }

    /// Bytes this group's quartet and class tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        (self.quartets.len() + self.class_shape.len() + self.class_idx.len())
            * std::mem::size_of::<u32>()
            + self.class_factor.len() * std::mem::size_of::<f64>()
    }

    /// Bytes this group's Cartesian output buffer costs — on the host and on
    /// the device, since `run_2e_batches` allocates exactly `out_len` `f64`s
    /// for it (M1, M6). One expression, so the pre-flight memory plan and the
    /// real allocation cannot drift apart.
    #[must_use]
    pub fn output_bytes(&self) -> usize {
        self.out_len * std::mem::size_of::<f64>()
    }

    /// Bytes this group's shared G-tensor scratch slab costs, sized to its
    /// widest merged class (M1, M6). One expression, shared by the pre-flight
    /// memory plan and the real slab allocation.
    #[must_use]
    pub fn g_slab_bytes(&self) -> usize {
        g_slab_stride(self.max_g_size as usize) * std::mem::size_of::<f64>()
    }

    /// Bytes of one slot's staged-contraction scratch for this group — `0`
    /// unless a quartet here is generally contracted. Shared by the
    /// pre-flight plan and the real allocation, like [`Self::g_slab_bytes`].
    #[must_use]
    pub fn ctr_slab_bytes(&self) -> usize {
        ctr_slab_stride(self.max_ctr_len as usize) * std::mem::size_of::<f64>()
    }
}

/// Widest device c2s-transform scratch length any group in `groups` needs
/// (M4.3), shared by [`run_2e_batches`] (which allocates it) and
/// [`plan_c2s_scratch_bytes`] (which pre-flights its byte cost) so the two
/// cannot drift apart independently.
fn c2s_scratch_widest_len<R: Runtime>(
    client: &ComputeClient<R>,
    groups: &[TwoELaunchGroup],
) -> usize {
    let mut len = 0_usize;
    for group in groups {
        if group.len() == 0 {
            continue;
        }
        // The transform's work items, not its quartets (§19). Sizing this from
        // `group.len()` while `launch_c2s` sizes its geometry from the item
        // count is the same disagreement §18.3 found between M4.1's shared G
        // slab and the split rows — except here the slab is *written*, so a
        // short one is not a lost allocation but an out-of-bounds store. It
        // showed up as `vendor|d|` of 3e-2 … 8e-1 on ROCm the moment the
        // transform started handing out `(quartet, quad)` pairs.
        let n_items: usize = group.quad_count.iter().map(|&q| q.max(1) as usize).sum();
        len = len.max(crate::kernels::c2s_device::c2s_scratch_len(
            client,
            n_items,
            group.max_block_len,
        ));
    }
    len
}

/// Dispatch every launch group of a batched 2e run on one backend client,
/// uploading the flattened basis **once** (Tasks 34-B / 34-C / 34-E / 35-M1).
///
/// Returns one concatenated Cartesian buffer per group, in `groups` order.
/// Each group costs exactly one kernel dispatch and one readback; the basis
/// costs one upload for the whole run.
///
/// Each block within a group buffer is laid out exactly as the single-quartet
/// path produced it: block `(ci,cj,ck,cl)` at
/// `(((ci*nctr_j+cj)*nctr_k+ck)*nctr_l+cl)*block_len`, `i` fastest.
fn run_2e_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &TwoEBasisHandles,
    pairs: &PairTableHandles,
    groups: &[TwoELaunchGroup],
    options: TwoEBatchOptions,
    on_group: &mut dyn FnMut(usize, Vec<f64>),
    c2s: Option<&C2sChunkPlan<'_>>,
) -> crate::memory_probe::DeviceMemoryProbe {
    if groups.is_empty() {
        return crate::memory_probe::DeviceMemoryProbe::new();
    }
    let probe = Arc::new(Mutex::new(crate::memory_probe::DeviceMemoryProbe::new()));
    probe
        .lock()
        .expect("device memory probe poisoned")
        .sample(client);

    // The basis buffers are already on the device — uploaded either by this
    // call's throwaway residency or by a [`ResidentTwoEBasis`] the caller keeps
    // across calls (Task 34-C). `Handle` is cheap to clone; the buffer it names
    // is shared by every dispatch below.
    let TwoEBasisHandles {
        exps: exps_h,
        coeffs: coeffs_h,
        centers: centers_h,
        shell_meta: meta_h,
        exps_len,
        coeffs_len,
        centers_len,
        shell_meta_len,
    } = basis;

    let rys_tables = crate::math::rys_wheeler::ext_rys_tables();
    // M4.2: one upload for the whole run. The kernel signature does not vary
    // with `nroots`, so every dispatch binds this buffer whether or not its
    // classes reach the extended entry — which is exactly why re-uploading it
    // per dispatch was pure waste (~4.7 KB x 24 dispatches on SO2/def2-TZVP).
    let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));
    probe
        .lock()
        .expect("device memory probe poisoned")
        .charge_tables(rys_tables.len() * std::mem::size_of::<f64>(), 1);

    // M4.1: one G-tensor scratch slab for the whole run, sized to the widest
    // heuristic geometry any group asks for, rather than one allocation inside
    // every `launch()`.
    //
    // The kernel only ever touches the leading `slots * g_stride` of what it is
    // given, and rebuilds every element it reads from the primitive quartet it
    // is on, so a slab carrying another group's leftovers is as good as a fresh
    // one — which is already true today across a group's own grid-stride walk.
    let mut shared_g_len = 0_usize;
    // The staged-contraction scratch (GTH plan, C1) is sized and reused the
    // same way, and stays a one-element placeholder for a segmented list.
    let mut shared_ctr_len = 0_usize;
    for group in groups {
        if group.len() == 0 {
            continue;
        }
        let g_size_u = group.max_g_size as usize;
        let ctr_len_u = group.max_ctr_len as usize;
        // The row count, not the quartet count: the ket-pair split expands the
        // rows a dispatch walks (G1), and the geometry below is the geometry
        // `launch()` will pick from those rows. Sizing this from `group.len()`
        // made the two disagree the moment anything split — the slab came out
        // too small, every launch fell through to its own allocation, and
        // M4.1's one-allocation-per-run quietly became one per launch.
        let n_rows: usize = kl_split_plan(group).iter().map(|&p| p as usize).sum();
        let cube_dim =
            two_e_cube_dim::<R>(client, group.max_block_len, n_rows, g_size_u, ctr_len_u);
        let n_cubes = two_e_cube_count::<R>(client, n_rows, g_size_u, ctr_len_u);
        let per_unit = two_e_per_unit::<R>(client);
        let slots = if per_unit {
            n_cubes as usize * cube_dim.num_elems() as usize
        } else {
            n_cubes as usize
        };
        // B1: a dispatch whose G tensor lives in shared memory allocates no
        // global slab for it at all — the memory the tiers save.
        if group.signature.tier == 0 {
            shared_g_len = shared_g_len.max(slots * g_slab_stride(g_size_u));
        }
        shared_ctr_len = shared_ctr_len.max(slots * ctr_slab_stride(ctr_len_u));
    }
    let shared_g_bytes = shared_g_len * std::mem::size_of::<f64>();
    let shared_g = client.empty(shared_g_bytes.max(1));
    let shared_ctr_bytes = shared_ctr_len * std::mem::size_of::<f64>();
    let shared_ctr = client.empty(shared_ctr_bytes.max(1));
    {
        let mut ledger = probe.lock().expect("device memory probe poisoned");
        ledger.charge_g_slab(shared_g_bytes);
        if shared_ctr_bytes > 0 {
            ledger.charge_g_slab(shared_ctr_bytes);
        }
    }

    // M4.3: one c2s transform scratch slab for the whole chunk, sized to the
    // widest group's need, instead of one allocation per group inside
    // `launch_c2s` — the same move M4.1 already made for the shared G-tensor
    // slab above. Only the device-transform path (`c2s.is_some()`) ever binds
    // this, so it costs nothing when the host transform is in use.
    let c2s_scratch = c2s.map(|_| {
        let len = c2s_scratch_widest_len(client, groups);
        client.empty(len.max(1) * std::mem::size_of::<f64>())
    });

    // S4: a memory budget means the caller wants the low peak, so the pipeline
    // is refused there regardless of the environment switch.
    let pipelined = options.memory_limit_bytes.is_none() && pipeline_overlap_enabled();
    let mut pending: Option<(usize, cubecl::server::Handle, usize)> = None;
    for (group_index, group) in groups.iter().enumerate() {
        let n_quartets = group.len();
        if n_quartets == 0 {
            on_group(group_index, Vec::new());
            continue;
        }
        // Sized to the widest class merged into this dispatch: every class
        // indexes only the leading `3 * g_size` of the slab, so a narrow class
        // touches exactly the elements it did when it launched alone.
        let g_size_u = group.max_g_size as usize;
        let per_unit = two_e_per_unit::<R>(client);
        // ── S3 / B1: the shared-memory tier is part of the signature ────────
        //
        // Decided when the groups were built (`TwoELaunchSignature::of`), from
        // the class and the backend's limit; `0` is the per-slot global slab,
        // and it is what every group carries under the per-unit decomposition.
        let shared_tier = group.signature.tier;
        debug_assert!(
            !per_unit || shared_tier == 0,
            "a per-unit dispatch carries a tier"
        );

        // ── G1: spread each cooperative quartet over several cubes ──────────
        //
        // A dispatch's work item lasts as long as its quartet's *serial* walk
        // over its primitive quartets — a cube on the cooperative shape (§10.5),
        // a row of K2's partition on the per-unit one, where a quartet is
        // indivisible and so bounds the dispatch below however good the
        // partition is (§17). Splitting a quartet's ket-pair range shortens
        // that walk by the same factor on both: part 0 accumulates into the
        // quartet's own output block, the rest into a compact partial region,
        // and one reduce kernel sums them in a fixed order.
        //
        // The split is sized **per quartet** (§16), because a fused dispatch is
        // heterogeneous: the `(dd|dd)` quartet that sets the critical path
        // takes the parts and the `(ss|ss)` quartets beside it take none — and
        // it is sized from the quartet *alone* (§18), so a memory budget
        // changes what a chunk holds but never what a quartet computes.
        // `CINTX_2E_KL_SPLIT=off` is the way back to an unsplit run.
        let trace = dispatch_trace_enabled();
        let mut t_phase = std::time::Instant::now();
        let mut t_plan = 0.0;
        let mut t_upload = 0.0;
        let splits: Vec<u32> = kl_split_plan(group);
        let any_split = splits.iter().any(|&s| s > 1);
        let split: Option<KlSplitLayout> = any_split.then(|| {
            let block_lens = quartet_block_lens(&group.quartets, group.out_len);
            expand_kl_split(
                &group.quartets,
                &splits,
                &block_lens,
                &group.quartet_cost,
                group.out_len,
            )
        });
        let (rows, reduce_table, row_cost, extra_len): (&[u32], &[u32], &[u64], usize) =
            match &split {
                Some(layout) => (
                    &layout.rows,
                    &layout.reduce_table,
                    &layout.row_cost,
                    layout.extra_len,
                ),
                None => (&group.quartets, &[], &group.quartet_cost, 0),
            };
        let n_rows = rows.len() / QUARTET_ROW_STRIDE;
        probe
            .lock()
            .expect("device memory probe poisoned")
            .note_kl_split(splits.iter().copied().max().unwrap_or(1));

        if trace {
            t_plan = trace_phase(client, &mut t_phase);
        }
        let quartets_h = client.create_from_slice(u32::as_bytes(rows));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        // The transform reads the same class shapes the evaluation does and the
        // *unsplit* quartet rows — one per quartet, pointing at the reduced
        // output — so it binds the same buffers where it can (M3).
        let quartets_h_for_c2s = if any_split {
            client.create_from_slice(u32::as_bytes(&group.quartets))
        } else {
            quartets_h.clone()
        };
        let shape_h_for_c2s = shape_h.clone();
        let factor_h = client.create_from_slice(f64::as_bytes(&group.class_factor));
        let idx_h = client.create_from_slice(u32::as_bytes(&group.class_idx));
        // The group's output, followed by the compact partial region the split
        // rows write into (G1, §16). `extra_len` is 0 when nothing is split,
        // and this is then exactly the buffer the unsplit path always had.
        let combined_len = group.out_len + extra_len;
        let out_h = client.empty(combined_len * std::mem::size_of::<f64>());
        {
            let mut ledger = probe.lock().expect("device memory probe poisoned");
            ledger.charge_tables(
                group.upload_bytes()
                    + (rows.len() - group.quartets.len()) * std::mem::size_of::<u32>(),
                4,
            );
            ledger.charge_output(combined_len * std::mem::size_of::<f64>());
        }

        let dispatch = TwoEGroupDispatch::<R> {
            client: client.clone(),
            exps: exps_h.clone(),
            coeffs: coeffs_h.clone(),
            centers: centers_h.clone(),
            shell_meta: meta_h.clone(),
            exps_len: *exps_len,
            coeffs_len: *coeffs_len,
            centers_len: *centers_len,
            shell_meta_len: *shell_meta_len,
            quartets: quartets_h,
            class_shape: shape_h,
            class_factor: factor_h,
            class_idx: idx_h,
            class_idx_len: group.class_idx.len(),
            rys_tables: rys_tab_h.clone(),
            shared_g: shared_g.clone(),
            shared_g_len,
            shared_ctr: shared_ctr.clone(),
            shared_ctr_len,
            ctr_len: group.max_ctr_len as usize,
            ctr_mode: contraction_mode(),
            coop_build: cooperative_build_mode(),
            // K2 partitions *rows*, so the costs it ranks must be the rows'
            // (§17): one entry per expanded row when the split is on, the
            // group's own vector when it is not.
            quartet_cost: Arc::new(row_cost.to_vec()),
            balance: balance_mode(),
            pair_data: pairs.data.clone(),
            pair_index: pairs.index.clone(),
            pair_offset: pairs.offset.clone(),
            out: out_h.clone(),
            quartets_len: rows.len(),
            class_shape_len: group.class_shape.len(),
            class_factor_len: group.class_factor.len(),
            pair_data_len: pairs.data_len,
            pair_index_len: pairs.index_len,
            pair_offset_len: pairs.offset_len,
            expcutoff: pairs.expcutoff,
            nbas: pairs.nbas,
            acc_slots_max: accumulator_slots_max(),
            shared_tier,
            out_len: combined_len,
            n_quartets: n_rows as u32,
            n_cubes: two_e_cube_count::<R>(client, n_rows, g_size_u, group.max_ctr_len as usize),
            g_size: g_size_u,
            per_unit,
            block_len: group.max_block_len,
            signature: group.signature,
            max_nroots: group.max_nroots,
            primitive_tolerance: options.primitive_tolerance,
            heuristic_cube_dim: two_e_cube_dim::<R>(
                client,
                group.max_block_len,
                n_rows,
                g_size_u,
                group.max_ctr_len as usize,
            ),
            probe: Arc::clone(&probe),
        };
        if trace {
            t_upload = trace_phase(client, &mut t_phase);
        }
        dispatch_2e_group(dispatch);
        let t_kernel = if trace {
            trace_phase(client, &mut t_phase)
        } else {
            0.0
        };
        // G1: fold the partial blocks onto part 0, in place, then hand the
        // consumer a view of just the group's output. Trimming the handle
        // rather than copying into a fresh buffer is what keeps the split free
        // for the quartets that did not take it (§17): an unsplit quartet is
        // neither summed nor moved.
        let out_h = if any_split {
            reduce_kl_partials_inplace::<R>(client, &out_h, combined_len, reduce_table, &probe);
            out_h.offset_end((extra_len * std::mem::size_of::<f64>()) as u64)
        } else {
            out_h
        };
        let t_reduce = if trace {
            trace_phase(client, &mut t_phase)
        } else {
            0.0
        };

        // ── S4: optionally keep one dispatch in flight ────────────────────
        //
        // CubeCL launches are lazy — only a read or a sync forces completion —
        // so draining the *previous* group here leaves this one's kernel
        // running while its predecessor is read back and transformed.
        //
        // Off by default, and deliberately so: the overlap keeps two groups'
        // output buffers alive at once, which is precisely the peak M1 just
        // turned from a sum into a maximum. A caller who has set a memory
        // budget is asking for the maximum, so the pipeline stays off there
        // whatever the environment says.
        // ── M3: transform on the device, and read back spherical ──────────
        //
        // The Cartesian buffer never leaves the device. Its blocks are contracted
        // against the `c2s` tables into the chunk's shared spherical buffer, and
        // the caller reads that back once for the whole chunk instead of once
        // per group. On SO2/def2-TZVP that is 99.8 MiB across the bus instead of
        // 177.9 MiB, and none of the transform on the host.
        if let Some(plan) = c2s {
            let offsets = &plan.sph_offsets[group_index];
            let offsets_h = client.create_from_slice(u32::as_bytes(offsets));
            probe
                .lock()
                .expect("device memory probe poisoned")
                .charge_tables(offsets.len() * std::mem::size_of::<u32>(), 1);
            // §19: one work item per `(quartet, contraction quad)`, built here
            // beside the offsets it travels with.
            let items = crate::kernels::c2s_device::c2s_work_items(&group.quad_count);
            let n_items = (items.len() / 2) as u32;
            let items_h = client.create_from_slice(u32::as_bytes(&items));
            probe
                .lock()
                .expect("device memory probe poisoned")
                .charge_tables(std::mem::size_of_val(items.as_slice()), 1);
            crate::kernels::c2s_device::launch_c2s(crate::kernels::c2s_device::C2sDispatch {
                client,
                cart: out_h,
                cart_len: group.out_len,
                items: items_h,
                items_len: items.len(),
                n_items,
                quartets: quartets_h_for_c2s,
                quartets_len: group.quartets.len(),
                sph_offsets: offsets_h,
                sph_offsets_len: offsets.len(),
                class_shape: shape_h_for_c2s,
                class_shape_len: group.class_shape.len(),
                shell_meta: meta_h.clone(),
                shell_meta_len: *shell_meta_len,
                tables: plan.tables,
                sph: plan.sph.clone(),
                sph_len: plan.sph_len,
                scratch_half: group.max_block_len,
                shape_stride: TWO_E_SHAPE_STRIDE as u32,
                scratch: c2s_scratch
                    .clone()
                    .expect("c2s_scratch is Some whenever a c2s plan is dispatched"),
            });
            probe
                .lock()
                .expect("device memory probe poisoned")
                .sample(client);
            if trace {
                let t_c2s = trace_phase(client, &mut t_phase);
                eprintln!(
                    "  trace group {group_index}: sig({},{},{},tier={}) rows={n_rows} cubes={} \
                     plan={t_plan:.2}ms upload={t_upload:.2}ms kernel={t_kernel:.2}ms \
                     reduce={t_reduce:.2}ms c2s={t_c2s:.2}ms",
                    group.signature.ibase,
                    group.signature.kbase,
                    group.signature.nroots,
                    group.signature.tier,
                    two_e_cube_count::<R>(client, n_rows, g_size_u, group.max_ctr_len as usize),
                );
            }
        } else if pipelined {
            if let Some((prev_index, prev_h, prev_len)) = pending.take() {
                drain_group(client, &probe, prev_index, prev_h, prev_len, on_group);
            }
            pending = Some((group_index, out_h, group.out_len));
        } else {
            drain_group(client, &probe, group_index, out_h, group.out_len, on_group);
        }
    }
    if let Some((index, handle, len)) = pending.take() {
        drain_group(client, &probe, index, handle, len, on_group);
    }
    *probe.lock().expect("device memory probe poisoned")
}

/// `CINTX_2E_TRACE=1`: time every phase of every dispatch in
/// [`run_2e_batches`], each one *synced* so the number is the phase's own and
/// not the enqueue cost of a lazy launch. It serialises the pipeline, so it is
/// an attribution aid and never a timing run (§22.5).
fn dispatch_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CINTX_2E_TRACE").is_ok_and(|v| v == "1"))
}

fn trace_phase<R: Runtime>(client: &ComputeClient<R>, since: &mut std::time::Instant) -> f64 {
    let _ = cubecl::future::block_on(client.sync());
    let ms = since.elapsed().as_secs_f64() * 1e3;
    *since = std::time::Instant::now();
    ms
}

/// Evaluate one chunk's groups and transform them on the device (M3).
///
/// Returns the chunk's **spherical** output, in the caller's quartet order.
/// Nothing Cartesian reaches the host: each group's Cartesian buffer is consumed
/// by the transform on the device and dropped there, and the single readback is
/// the spherical block the caller asked for.
fn dispatch_2e_batches_device_c2s(
    backend: &ResolvedBackend,
    basis: &TwoEBasisHandles,
    pairs: &PairTableHandles,
    groups: &[TwoELaunchGroup],
    options: TwoEBatchOptions,
    tables: &crate::kernels::c2s_device::C2sHandles,
    sph_offsets: &[Vec<u32>],
    sph_len: usize,
) -> Result<(crate::memory_probe::DeviceMemoryProbe, Vec<f64>), cintxRsError> {
    macro_rules! run {
        ($rt:ty, $client:expr) => {{
            let client = $client;
            let sph = client.empty(sph_len.max(1) * std::mem::size_of::<f64>());
            let plan = C2sChunkPlan {
                tables,
                sph: sph.clone(),
                sph_len,
                sph_offsets,
            };
            let probe = run_2e_batches::<$rt>(
                client,
                basis,
                pairs,
                groups,
                options,
                &mut |_, _| unreachable!("the device transform consumes every group"),
                Some(&plan),
            );
            let raw = client.read_one_unchecked(sph);
            let values = f64::from_bytes(&raw)[0..sph_len].to_vec();
            Ok((probe, values))
        }};
    }
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run!(cubecl::cpu::CpuRuntime, client),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run!(cubecl_wgpu::WgpuRuntime, client),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run!(cubecl_cuda::CudaRuntime, client),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run!(cubecl_hip::HipRuntime, client),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run!(cubecl_wgpu::WgpuRuntime, client),
    }
}

/// What one chunk's device-side transform binds (M3).
///
/// The spherical buffer is per *chunk*, not per group: every group writes its
/// quartets into their places in it, and the caller reads it back once. That is
/// also why the offsets are per group — a group's rows are ordered class by
/// class, and only the grouping knows which caller quartet each row belongs to.
pub(crate) struct C2sChunkPlan<'a> {
    pub(crate) tables: &'a crate::kernels::c2s_device::C2sHandles,
    pub(crate) sph: cubecl::server::Handle,
    pub(crate) sph_len: usize,
    pub(crate) sph_offsets: &'a [Vec<u32>],
}

/// Read one dispatch's Cartesian output back and hand it to the consumer.
///
/// Factored out of the dispatch loop so the pipelined and serial arms drain
/// through exactly the same path (S4) — the overlap must change *when* a group
/// is read, never *what* is read.
fn drain_group<R: Runtime>(
    client: &ComputeClient<R>,
    probe: &Arc<Mutex<crate::memory_probe::DeviceMemoryProbe>>,
    group_index: usize,
    out_h: cubecl::server::Handle,
    out_len: usize,
    on_group: &mut dyn FnMut(usize, Vec<f64>),
) {
    let raw = client.read_one_unchecked(out_h);
    let cart = f64::from_bytes(&raw)[0..out_len].to_vec();
    drop(raw);
    probe
        .lock()
        .expect("device memory probe poisoned")
        .sample(client);
    // The consumer transforms this group and drops the buffer, so the next
    // readback allocates into the space it just freed (M1).
    on_group(group_index, cart);
}

/// Does `CINTX_2E_PIPELINE` ask for the one-deep dispatch/readback overlap?
///
/// `async` turns it on, anything else (and unset) leaves it off. Off is the
/// default because the overlap trades memory for latency: two groups' output
/// buffers are live at once. Whether that trade pays is a backend question —
/// on a real GPU it hides the whole readback, on the CubeCL CPU runtime the
/// "device" is the same cores the transform runs on — so it is a measurement
/// switch rather than a policy.
fn pipeline_overlap_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CINTX_2E_PIPELINE").is_ok_and(|value| value.eq_ignore_ascii_case("async"))
    })
}

/// Everything one launch group's dispatch binds, in a form that can be cloned
/// and re-launched under a different cube width.
///
/// This is what the launch-geometry autotuner (Phase 6 of
/// `docs/design/cubecl_speed_optimization_plan.md`) benchmarks: each candidate
/// runs *this* dispatch, so a measurement is of the real kernel on the real
/// basis, not of a proxy. Every field but the cube width is held fixed, and the
/// G-tensor scratch is allocated inside [`TwoEGroupDispatch::launch`] because
/// its size is the one thing the width changes.
#[derive(Clone)]
struct TwoEGroupDispatch<R: Runtime> {
    client: ComputeClient<R>,
    exps: cubecl::server::Handle,
    coeffs: cubecl::server::Handle,
    centers: cubecl::server::Handle,
    shell_meta: cubecl::server::Handle,
    exps_len: usize,
    coeffs_len: usize,
    centers_len: usize,
    shell_meta_len: usize,
    quartets: cubecl::server::Handle,
    class_shape: cubecl::server::Handle,
    class_factor: cubecl::server::Handle,
    /// The group's Cartesian index tables (K1).
    class_idx: cubecl::server::Handle,
    class_idx_len: usize,
    rys_tables: cubecl::server::Handle,
    /// The run's shared G-tensor scratch (M4.1), used whenever this geometry's
    /// slab fits inside it.
    shared_g: cubecl::server::Handle,
    /// `f64` capacity of [`Self::shared_g`].
    shared_g_len: usize,
    /// The run's shared staged-contraction scratch (GTH plan, C1), sized and
    /// reused exactly as [`Self::shared_g`] is.
    shared_ctr: cubecl::server::Handle,
    /// `f64` capacity of [`Self::shared_ctr`].
    shared_ctr_len: usize,
    /// One slot's staged-contraction scratch length for this group
    /// (`TwoELaunchGroup::max_ctr_len`); `0` for a segmented group.
    ctr_len: usize,
    /// `1` for the staged general contraction, `0` for the naive fold.
    ctr_mode: u32,
    /// `1` splits the G build across the cube's lanes (S3), `0` builds it all
    /// on lane 0. Meaningless under the per-unit decomposition, where a
    /// cooperative group is one lane either way.
    coop_build: u32,
    /// Per-row cost estimates the per-unit partition is cut from (K2);
    /// shared, because the tuner clones this dispatch per candidate width.
    quartet_cost: Arc<Vec<u64>>,
    /// `1` cuts the per-unit partition by cost, `0` uniformly (K2).
    balance: u32,
    pair_data: cubecl::server::Handle,
    pair_index: cubecl::server::Handle,
    pair_offset: cubecl::server::Handle,
    out: cubecl::server::Handle,
    quartets_len: usize,
    class_shape_len: usize,
    class_factor_len: usize,
    pair_data_len: usize,
    pair_index_len: usize,
    pair_offset_len: usize,
    /// libcint's `expcutoff`; `+inf` is the unscreened A/B reference (S1).
    expcutoff: f64,
    /// Shell count the pair-table offsets are indexed by.
    nbas: u32,
    /// The S2 accumulator ceiling this dispatch runs under.
    acc_slots_max: u32,
    /// The shared-memory tier this dispatch's G tensor lives in (S3 / B1),
    /// `0` for the per-slot global slab.
    shared_tier: u32,
    out_len: usize,
    n_quartets: u32,
    n_cubes: u32,
    g_size: usize,
    per_unit: bool,
    block_len: u32,
    signature: TwoELaunchSignature,
    /// The kernel's comptime `nr_max` — this group's widest Rys order (F1, §15).
    max_nroots: u32,
    primitive_tolerance: f64,
    /// The geometry [`two_e_cube_dim`] picked — the safe default, and the
    /// candidate every tuned width has to beat.
    heuristic_cube_dim: CubeDim,
    /// The run's allocation ledger, charged from inside [`Self::launch`].
    ///
    /// Shared rather than owned because the tuner clones this dispatch once per
    /// candidate width and launches each clone: the G slab is allocated inside
    /// `launch`, so a per-dispatch counter would report one slab where a tuned
    /// group allocated a dozen. That over-allocation is the thing M4 removes,
    /// so the ledger has to see it (M6).
    probe: Arc<Mutex<crate::memory_probe::DeviceMemoryProbe>>,
}

impl<R: Runtime> TwoEGroupDispatch<R> {
    /// Launch this group at `cube_dim`.
    ///
    /// The kernel walks its quartet list grid-stride and splits each quartet's
    /// contraction block across the cube's lanes, so it covers the same index
    /// space and writes the same values at every geometry: `cube_dim` buys
    /// speed, never results.
    fn launch(&self, cube_dim: CubeDim) {
        // One private G slab per *slot*: a slot is a cube in the cooperative
        // decomposition and a unit in the per-unit one.
        let n_slots = if self.per_unit {
            self.n_cubes as usize * cube_dim.num_elems() as usize
        } else {
            self.n_cubes as usize
        };
        let g_stride = g_slab_stride(self.g_size);
        // B1: a dispatch running in a shared-memory tier never indexes the
        // global slab, and binds the run's (possibly empty) one unread.
        let g_len = if self.shared_tier > 0 {
            0
        } else {
            n_slots * g_stride
        };
        // The run's shared slab covers every heuristic geometry (M4.1). A tuning
        // candidate may ask for a wider cube than the heuristic did, and so for
        // more slots than the shared slab holds; that candidate — and only that
        // candidate — falls back to its own allocation.
        let (g_h, g_capacity) = if g_len <= self.shared_g_len {
            (self.shared_g.clone(), self.shared_g_len)
        } else {
            let g_bytes = g_len * std::mem::size_of::<f64>();
            self.probe
                .lock()
                .expect("device memory probe poisoned")
                .charge_g_slab(g_bytes);
            (self.client.empty(g_bytes), g_len)
        };
        // The contraction scratch follows the same rule; a segmented group
        // binds the one-element placeholder and never indexes it.
        let ctr_stride = ctr_slab_stride(self.ctr_len);
        let ctr_total = n_slots * ctr_stride;
        let (ctr_h, ctr_capacity) = if ctr_total <= self.shared_ctr_len {
            (self.shared_ctr.clone(), self.shared_ctr_len.max(1))
        } else {
            let ctr_bytes = ctr_total * std::mem::size_of::<f64>();
            self.probe
                .lock()
                .expect("device memory probe poisoned")
                .charge_g_slab(ctr_bytes);
            (self.client.empty(ctr_bytes), ctr_total)
        };

        // K2: the per-unit walk's row bounds, one range per slot. The
        // cooperative arm ignores them (it indexes `slot * punit == 0`) and is
        // handed a two-element placeholder. Recomputed per launch because the
        // slot count is the one thing the tuner's candidate widths change.
        let slot_bounds: Vec<u32> = if self.per_unit {
            per_unit_slot_bounds(
                &self.quartet_cost[..self.n_quartets as usize],
                n_slots,
                self.balance == 1,
            )
        } else {
            vec![0, self.n_quartets]
        };
        let bounds_h = self.client.create_from_slice(u32::as_bytes(&slot_bounds));
        // A real (tiny) upload per launch. It reaches `device_table_bytes_total`
        // through the ledger; `transfer_bytes` is summed from the groups'
        // `upload_bytes` before any width is chosen, so it does not carry the
        // `4 * (n_slots + 1)` bytes here.
        self.probe
            .lock()
            .expect("device memory probe poisoned")
            .charge_tables(slot_bounds.len() * std::mem::size_of::<u32>(), 1);

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_quartets`, by the class index in each quartet row (bounded by
        // `class_count`), by the per-shell `nprim`/`nctr` read from
        // `shell_meta`, by `n_slots + 1` for the partition bounds, and by the
        // per-class G-tensor extents — the same bounds the single-quartet path
        // has always satisfied.
        unsafe {
            two_electron_scalar_kernel::launch_unchecked::<f64, R>(
                &self.client,
                crate::plane::cube_count_1d(self.n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(self.exps.clone(), self.exps_len),
                ArrayArg::from_raw_parts(self.coeffs.clone(), self.coeffs_len),
                ArrayArg::from_raw_parts(self.centers.clone(), self.centers_len),
                ArrayArg::from_raw_parts(self.shell_meta.clone(), self.shell_meta_len),
                ArrayArg::from_raw_parts(self.quartets.clone(), self.quartets_len),
                ArrayArg::from_raw_parts(self.class_shape.clone(), self.class_shape_len),
                ArrayArg::from_raw_parts(self.class_factor.clone(), self.class_factor_len),
                ArrayArg::from_raw_parts(self.class_idx.clone(), self.class_idx_len),
                ArrayArg::from_raw_parts(self.rys_tables.clone(), EXT_TABLES_LEN),
                ArrayArg::from_raw_parts(self.pair_data.clone(), self.pair_data_len),
                ArrayArg::from_raw_parts(self.pair_index.clone(), self.pair_index_len),
                ArrayArg::from_raw_parts(self.pair_offset.clone(), self.pair_offset_len),
                ArrayArg::from_raw_parts(bounds_h, slot_bounds.len()),
                ArrayArg::from_raw_parts(g_h, g_capacity),
                ArrayArg::from_raw_parts(ctr_h, ctr_capacity),
                ArrayArg::from_raw_parts(self.out.clone(), self.out_len),
                PIE4,
                self.primitive_tolerance,
                self.expcutoff,
                self.nbas,
                self.acc_slots_max,
                self.n_quartets,
                self.n_cubes,
                g_stride as u32,
                ctr_stride as u32,
                self.ctr_mode,
                self.coop_build,
                self.signature.ibase,
                self.signature.kbase,
                self.max_nroots,
                u32::from(self.per_unit),
                self.shared_tier,
                QUARTET_ROW_STRIDE as u32,
            );
        }
    }

    /// The coarse device-and-workload key this dispatch's geometry is measured
    /// against.
    fn tuning_key(&self) -> crate::tuning::LaunchGeometryKey {
        crate::tuning::LaunchGeometryKey::new(
            crate::tuning::TunedFamily::TwoE,
            &crate::plane::launch_hardware(&self.client),
            if self.per_unit {
                crate::tuning::Decomposition::PerUnit
            } else {
                crate::tuning::Decomposition::Cooperative
            },
            self.max_nroots,
            self.n_quartets as usize,
            self.block_len,
            slot_scratch_bytes(self.g_size, self.ctr_len),
        )
    }

    /// The same dispatch over a bounded prefix of the quartet list, for the
    /// benchmark passes.
    ///
    /// The kernel reads its quartet count from a scalar, so this is the same
    /// program on the same shapes with a shorter list — same specialization,
    /// same G extents, same decomposition. Only the winning candidate is then
    /// re-executed on the full list, so the truncation can cost ranking
    /// accuracy and nothing else. `n_cubes` is re-derived because the grid is
    /// sized to the item count in the cooperative arm.
    ///
    /// The prefix length is work-aware: `g_size` is this group's per-quartet
    /// G-tensor cost, so a cheap `ssss` class is benchmarked over many quartets
    /// and an expensive `dddd` one over few, and each pass costs about the same.
    fn truncated_for_tuning(&self) -> Self {
        let n_quartets =
            crate::tuning::tune_sample_items(self.n_quartets as usize, self.g_size.max(1));
        let mut truncated = self.clone();
        truncated.n_quartets = n_quartets as u32;
        truncated.n_cubes =
            two_e_cube_count::<R>(&self.client, n_quartets, self.g_size, self.ctr_len);
        truncated
    }
}

/// The launch-geometry tuner for the batched 2e kernel.
///
/// One tuner for the whole crate-and-family: it is keyed by device inside, and
/// its persistent cache lives under that device's directory.
static TWO_E_GEOMETRY_TUNER: LocalTuner<crate::tuning::LaunchGeometryKey, String> =
    local_tuner!("2e-geometry");

/// The candidate cube widths for the 2e dispatch, with their viability
/// priorities.
///
/// The heuristic candidate is registered **without** a group, which is what
/// puts it in the first batch of the tuning plan alongside the highest-priority
/// group: the search always measures the geometry it is trying to beat. Every
/// width candidate is in one group whose intra-group priority prunes
/// (`-1`) the widths this device or this workload cannot honour, so the plan
/// skips them before compilation.
fn two_e_geometry_tunables<R: Runtime>()
-> TunableSet<crate::tuning::LaunchGeometryKey, TwoEGroupDispatch<R>, ()> {
    let mut set = TunableSet::new(
        |dispatch: &TwoEGroupDispatch<R>| dispatch.tuning_key(),
        |_key: &crate::tuning::LaunchGeometryKey, dispatch: &TwoEGroupDispatch<R>| {
            dispatch.truncated_for_tuning()
        },
    )
    .with(Tunable::new(
        "2e-geometry:heuristic",
        |dispatch: TwoEGroupDispatch<R>| {
            dispatch.launch(dispatch.heuristic_cube_dim);
            Ok::<(), String>(())
        },
    ));

    let viable = TuneGroup::<crate::tuning::LaunchGeometryKey>::new("viable-widths", |_| 1);
    for width in crate::tuning::CANDIDATE_CUBE_WIDTHS {
        set = set.with(
            Tunable::new(
                &format!("2e-geometry:width-{width}"),
                move |dispatch: TwoEGroupDispatch<R>| {
                    dispatch.launch(CubeDim::new_1d(width));
                    Ok::<(), String>(())
                },
            )
            .group(&viable, move |key| {
                crate::tuning::cube_width_priority(key, width, MAX_BATCH_SCRATCH_BYTES)
            }),
        );
    }
    set
}

/// Launch one group, tuning its cube width when the policy and the workload
/// both justify it.
///
/// Falls back to the heuristic geometry whenever tuning is off, the dispatch is
/// too small to pay for a benchmark, or the process has already tuned as many
/// distinct keys as it is allowed to — see [`crate::tuning`] for those bounds.
fn dispatch_2e_group<R: Runtime>(dispatch: TwoEGroupDispatch<R>) {
    // A pinned cube width is an instruction, not a hint: the tuner exists to
    // *choose* a width, so running it against an override both wastes the
    // benchmark and can launch at a width the caller did not ask for. Every
    // A/B that pins the geometry — `two_e_cooperative_arm`'s four-lane arm,
    // `gth_profile`'s unit-count curve — depends on the width it set being the
    // width that runs.
    //
    // F1 (§16) is how this surfaced. Fusing the Rys orders makes a cooperative
    // dispatch four times wider, which pushed the pinned four-lane arm over
    // `MIN_TUNE_ITEMS` for the first time; the tuner then benchmarked every
    // candidate width, and on the CubeCL CPU runtime each width is its own JIT
    // compilation of the whole kernel. A 1.5 s gate became a 20 minute one.
    if two_e_cube_dim_override().is_some() {
        dispatch.launch(dispatch.heuristic_cube_dim);
        return;
    }
    let key = dispatch.tuning_key();
    if !crate::tuning::should_tune(&key, dispatch.n_quartets as usize) {
        dispatch.launch(dispatch.heuristic_cube_dim);
        return;
    }
    let client = dispatch.client.clone();
    let device = crate::tuning::device_fingerprint(&client);
    let tunables = TWO_E_GEOMETRY_TUNER.init(two_e_geometry_tunables::<R>);
    TWO_E_GEOMETRY_TUNER.execute(&device, &client, tunables, dispatch);
}

/// Single-quartet dispatch — a one-class, one-quartet batch.
///
/// Kept as its own entry point because the per-tuple compatibility API
/// (`eval_raw`, `SessionRequest`) evaluates exactly one shell quartet and must
/// keep doing so. It marshals the four shells into the flattened form
/// [`run_2e_batches`] consumes, so both paths execute the *same* kernel and
/// every existing parity test covers the batched code at `n_quartets == 1`.
#[allow(clippy::too_many_arguments)]
fn run_2e_scalar_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    lk: u32,
    ll: u32,
    nprim_i: u32,
    nprim_j: u32,
    nprim_k: u32,
    nprim_l: u32,
    nctr_i: u32,
    nctr_j: u32,
    nctr_k: u32,
    nctr_l: u32,
    di: u32,
    dk: u32,
    dl: u32,
    dj: u32,
    g_size: u32,
    nmax: u32,
    mmax: u32,
    g2d_ijmax: u32,
    g2d_klmax: u32,
    ibase: u32,
    kbase: u32,
    nroots: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    exps_l: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
    coeff_l: &[f64],
    out_len: usize,
    expcutoff: f64,
) -> Vec<f64> {
    let params = TwoEClassParams {
        li,
        lj,
        lk,
        ll,
        di,
        dk,
        dl,
        dj,
        g_size,
        nmax,
        mmax,
        g2d_ijmax,
        g2d_klmax,
        ibase,
        kbase,
        nroots,
        common_factor,
    };

    let mut basis = TwoEFlatBasis::default();
    // One shell's contribution to a single-quartet batch: exponents,
    // coefficients, centre, primitive count, contraction count.
    type MarshalledShell<'a> = (&'a [f64], &'a [f64], [f64; 3], u32, u32);
    let shells: [MarshalledShell<'_>; 4] = [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_j, coeff_j, rj, nprim_j, nctr_j),
        (exps_k, coeff_k, rk, nprim_k, nctr_k),
        (exps_l, coeff_l, rl, nprim_l, nctr_l),
    ];
    for (exps, coeffs, center, nprim, nctr) in shells {
        basis.shell_meta.extend_from_slice(&[
            basis.exps.len() as u32,
            basis.coeffs.len() as u32,
            nprim,
            nctr,
        ]);
        basis.exps.extend_from_slice(exps);
        basis.coeffs.extend_from_slice(coeffs);
        basis.centers.extend_from_slice(&center);
    }

    // One class, one dispatch: the Rys-order fusion (F1) has nothing to merge
    // here, so this group is keyed on the class's own order.
    let mut group = TwoELaunchGroup::new(TwoELaunchSignature::of(&params, false, None));
    let class_index = group.push_class(&params);
    group.out_len = out_len;
    group.max_block_len = (out_len / ((nctr_i * nctr_j * nctr_k * nctr_l) as usize).max(1)) as u32;
    group.max_ctr_len = staged_ctr_len(
        [nctr_i, nctr_j, nctr_k, nctr_l],
        group.max_block_len as usize,
    ) as u32;

    // The per-tuple compatibility path builds a one-quartet residency, so its
    // pair table covers exactly these four shells, with `(0,1)` the bra and
    // `(2,3)` the ket.
    //
    // It screens at `expcutoff` (libcint's default unless the caller set
    // env[PTR_EXPCUTOFF] or `ExecutionOptions::expcutoff`, S1) like every other
    // entry point, and that is deliberate: `cint2e_sph` itself runs
    // `CINT2e_loop_nopt`, which applies the same three `expcutoff` tests.
    // Leaving this path unscreened would make it disagree with the vendor it
    // exists to reproduce *and* with the batched path that
    // `def2_2e_batch_parity` compares against it. What stays exact by contract
    // here is `prim_tol` — cintx's own extra screen, which the default
    // [`TwoEBatchOptions`] leaves at zero.
    let pair_shells: Vec<BatchShell> = shells
        .iter()
        .map(|&(exps, coeffs, center, nprim, nctr)| BatchShell {
            l: 0,
            nprim,
            nctr,
            exponents: exps.to_vec(),
            coefficients: coeffs.to_vec(),
            center,
        })
        .collect();
    // `l` above is a placeholder; the estimate needs the real angular momenta,
    // because `log_rr_ij` carries an `(li + lj) * ln(d + 1)` term.
    let pair_shells: Vec<BatchShell> = pair_shells
        .into_iter()
        .zip([li, lj, lk, ll])
        .map(|(shell, l)| BatchShell {
            l: l as u8,
            ..shell
        })
        .collect();
    let pairs = crate::kernels::pair_table::PairTable::build(
        &pair_shells,
        crate::kernels::pair_table::PairTableOptions { expcutoff },
    );
    // One row, whose ket range is the whole `(2,3)` span of that table.
    let kl_slot = (2 * pairs.nbas + 3) as usize;
    group.quartets.extend_from_slice(&[
        0,
        1,
        2,
        3,
        0,
        class_index,
        pairs.offset[kl_slot],
        pairs.offset[kl_slot + 1],
    ]);
    group.quartet_cost.push(1);
    let handles = upload_2e_basis::<R>(client, &basis);
    let pair_handles = upload_pair_table::<R>(client, &pairs);
    let mut cart = Vec::new();
    run_2e_batches::<R>(
        client,
        &handles,
        &pair_handles,
        std::slice::from_ref(&group),
        TwoEBatchOptions::default(),
        &mut |_, buffer| cart = buffer,
        None,
    );
    cart
}

/// `int2e_ip1` gradient launch — the ∇_A <ij|kl> two-electron force (GRAD-07).
///
/// Builds the plain Coulomb G-tensor with `li_ceil = li+1` headroom
/// ([`fill_g_tensor_2e`] via `rys_roots_host`), reuses
/// [`crate::kernels::f12::gout_ip1`] verbatim on it, and emits component-leading
/// `[3, nl, nk, nj, ni]` F-order matching pyscf-gto `layout_table.rs` (Risk R3).
///
/// Guards (fail-closed):
///   - `grad_shape.nroots > 12` → `UnsupportedApi` (Phase 25 FND-02 / T-25-03): the
///     host Rys engine (`rys_roots_host` → `rys_wheeler`) supports nroots 6..12, so the
///     li→li+1 raise routes Hessian-elevated quartets to the host `fill_g_tensor_2e`
///     path; only nroots>12 (vendor quadmath ceiling) is rejected.
///   - `Representation::Spinor` → `UnsupportedApi` (R5 / T-21-05-04).
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_ip1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Spinor gradient: not supported (R5 / D-03). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int2e_ip1 gradient".to_owned(),
        });
    }

    // li → li+1 headroom shape (D-06). gout_ip1's nabla1i_2e reads up to index li+1.
    //
    // D-PBC-24 P2-1: ω is part of the SHAPE, not only of the root evaluation —
    // short range doubles `nroots`, which sets `di`/`dk`/`g_size`. The raise and
    // the doubling compose: `rys_order = (li+1 + lj + lk + ll)/2 + 1`, then
    // `nroots = 2 * rys_order` under short range. `range_omega = None` leaves
    // this byte-identical to `build_2e_shape`.
    let range_omega = plan.operator_env_params.range_omega;
    let grad_shape = build_2e_shape_omega(
        li as usize + 1,
        lj as usize,
        lk as usize,
        ll as usize,
        range_omega,
    );

    // Phase 25 FND-02: this is the HOST gradient path (the loop below calls
    // `fill_g_tensor_2e` → `rys_roots_host`, NOT the device comptime kernel). The host
    // Rys engine now supports nroots 6..12 (rys_wheeler.rs), so the elevated-li Hessian
    // d-quartets that push nroots to 6 route here instead of returning UnsupportedApi.
    // The ceiling is the vendor-validated 12 (quadmath disabled); nroots>12 stays
    // fail-closed (T-25-03: typed error, never a panic).
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = 3 * block_len; // 3 components × Cartesian AO product

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    // Per-contraction-quad component-leading Cartesian accumulator: one
    // `3 * nfi*nfj*nfk*nfl` block per (ci,cj,ck,cl) quad (mirrors the scalar 2e
    // general-contraction layout). For all-nctr==1 this is a single block.
    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    // S1 follow-up (`def2_speed_memory_optimization_plan.md`): the same
    // `env[PTR_EXPCUTOFF]` cutoff the scalar/batched 2e path applies
    // (`crate::kernels::pair_table`) screens this derivative operator's
    // primitive quartets too, via the same `QuartetExpScreen` every
    // derivative launcher in this module builds.
    let crate::kernels::pair_table::QuartetExpScreen {
        expcutoff,
        bra_screen,
        ket_screen,
        log_maxc_i,
        log_maxc_j,
        log_maxc_k,
        log_maxc_l,
    } = crate::kernels::pair_table::QuartetExpScreen::new(
        plan,
        [li, lj, lk, ll],
        [n_prim_i, n_prim_j, n_prim_k, n_prim_l],
        [n_ctr_i, n_ctr_j, n_ctr_k, n_ctr_l],
        [
            &shell_i.exponents,
            &shell_j.exponents,
            &shell_k.exponents,
            &shell_l.exponents,
        ],
        [
            &shell_i.coefficients,
            &shell_j.coefficients,
            &shell_k.coefficients,
            &shell_l.coefficients,
        ],
        [ri, rj, rk, rl],
    );

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let cceij = bra_screen.cceij(ai, aj, log_maxc_i[pi], log_maxc_j[pj]);
            if !(cceij < expcutoff) {
                continue;
            }
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let ccekl = ket_screen.cceij(ak, al, log_maxc_k[pk], log_maxc_l[pl]);
                    if !(ccekl < expcutoff) {
                        continue;
                    }
                    let eijcutoff = expcutoff - ccekl;
                    if !(cceij <= eijcutoff) {
                        continue;
                    }
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    // Plain Coulomb G-tensor at the elevated li (li+1 headroom).
                    let Some(g) = fill_g_tensor_2e_range(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                        range_omega,
                    )?
                    else {
                        // Short range past EXPCUTOFF_SR: this primitive quartet
                        // contributes nothing (g2e.c:4460). Not zeros — nothing.
                        continue;
                    };

                    // Reuse gout_ip1 verbatim (f12.rs). It returns interleaved
                    // out[n*3+comp]; n walks [cl, ck, cj, ci] (ll slowest, li fastest).
                    // gout_ip1 is called at BASE li (the G-tensor carries li+1 headroom).
                    let gout = crate::kernels::f12::gout_ip1(
                        &g,
                        &grad_f12_shape,
                        li as usize,
                        lj as usize,
                        lk as usize,
                        ll as usize,
                        ai,
                    );

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
                                        * total_len;
                                    // TRANSPOSE interleaved gout[n*3+comp] into the
                                    // component-leading block: cart[comp*block + n].
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
        }
    }

    // Write component-leading `[3, nl, nk, nj, ni]` F-order to staging.
    // For each component, the per-quad block is the i-fastest `[nl][nk][nj][ni]`
    // Cartesian tensor — run the cart→sph transform per component for the sph rep.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
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
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Spinor => unreachable!("spinor int2e_ip1 rejected above"),
    }

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// int2e_ip2 gradient launcher (Phase 23 DRV1-01).
///
/// Sibling of [`launch_two_electron_ip1`] for the **ket** bra-center `k`
/// (`G2E_D_K`, libcint `CINTgout2e_int2e_ip2`, grad2.c:101). The only differences
/// vs ip1 are:
///   - headroom raised on `lk` (`build_2e_shape(li, lj, lk+1, ll)`) so
///     `nabla1k_2e` can read up to index `lk+1`;
///   - the single-side contraction uses [`crate::kernels::f12::gout_ipn`] with
///     `Nabla1Center::K` and the per-primitive **k-shell** exponent `ak`.
///     The s[0..2] mixing, the component-leading transpose, and the cart/sph output
///     path are identical to ip1.
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_ip2<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Spinor gradient: not supported (R5 / D-06). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int2e_ip2 gradient".to_owned(),
        });
    }

    // lk → lk+1 headroom shape (D-06). gout_ipn's nabla1k_2e reads up to index lk+1.
    // D-PBC-24 P2-1: see `launch_two_electron_ip1` — ω sizes the shape.
    let range_omega = plan.operator_env_params.range_omega;
    let grad_shape = build_2e_shape_omega(
        li as usize,
        lj as usize,
        lk as usize + 1,
        ll as usize,
        range_omega,
    );

    // Phase 25 FND-02: HOST gradient path (fill_g_tensor_2e → rys_roots_host). The host
    // Rys engine supports nroots 6..12 (rys_wheeler.rs); route Hessian-elevated quartets
    // here instead of UnsupportedApi. Ceiling = vendor-validated 12; nroots>12 fail-closed.
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = 3 * block_len; // 3 components × Cartesian AO product

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    // S1 follow-up (`def2_speed_memory_optimization_plan.md`): the same
    // `env[PTR_EXPCUTOFF]` cutoff the scalar/batched 2e path applies
    // (`crate::kernels::pair_table`) screens this derivative operator's
    // primitive quartets too, via the same `QuartetExpScreen` every
    // derivative launcher in this module builds.
    let crate::kernels::pair_table::QuartetExpScreen {
        expcutoff,
        bra_screen,
        ket_screen,
        log_maxc_i,
        log_maxc_j,
        log_maxc_k,
        log_maxc_l,
    } = crate::kernels::pair_table::QuartetExpScreen::new(
        plan,
        [li, lj, lk, ll],
        [n_prim_i, n_prim_j, n_prim_k, n_prim_l],
        [n_ctr_i, n_ctr_j, n_ctr_k, n_ctr_l],
        [
            &shell_i.exponents,
            &shell_j.exponents,
            &shell_k.exponents,
            &shell_l.exponents,
        ],
        [
            &shell_i.coefficients,
            &shell_j.coefficients,
            &shell_k.coefficients,
            &shell_l.coefficients,
        ],
        [ri, rj, rk, rl],
    );

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let cceij = bra_screen.cceij(ai, aj, log_maxc_i[pi], log_maxc_j[pj]);
            if !(cceij < expcutoff) {
                continue;
            }
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let ccekl = ket_screen.cceij(ak, al, log_maxc_k[pk], log_maxc_l[pl]);
                    if !(ccekl < expcutoff) {
                        continue;
                    }
                    let eijcutoff = expcutoff - ccekl;
                    if !(cceij <= eijcutoff) {
                        continue;
                    }
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    // Plain Coulomb G-tensor at the elevated lk (lk+1 headroom).
                    let Some(g) = fill_g_tensor_2e_range(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                        range_omega,
                    )?
                    else {
                        // Short range past EXPCUTOFF_SR: this primitive quartet
                        // contributes nothing (g2e.c:4460). Not zeros — nothing.
                        continue;
                    };

                    // ∇ on the ket bra-center k (Nabla1Center::K, exponent ak).
                    // gout_ipn is called at BASE lk (the G-tensor carries lk+1 headroom).
                    let gout = crate::kernels::f12::gout_ipn(
                        &g,
                        &grad_f12_shape,
                        li as usize,
                        lj as usize,
                        lk as usize,
                        ll as usize,
                        crate::kernels::f12::Nabla1Center::K,
                        ak,
                    );

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
                                        * total_len;
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
        }
    }

    // Component-leading `[3, nl, nk, nj, ni]` F-order write (identical to ip1).
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
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
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Spinor => unreachable!("spinor int2e_ip2 rejected above"),
    }

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// Which 2e Hessian family a [`launch_two_electron_hess2e`] call evaluates.
///
/// Each variant carries its own G-tensor headroom (`i_inc`/`j_inc`/`k_inc`) and
/// selects the matching verbatim-from-hess.c gout permutation. The host path
/// (`fill_g_tensor_2e` → `rys_roots_host`) is shared with the gradient families
/// so nroots≥6 Hessian-elevated d-quartets reach the FND-02 host Rys engine.
#[derive(Clone, Copy)]
enum Hess2eKind {
    /// int2e_ipip1 (∇²bra-i), rank 9, headroom i+2.
    Ipip1,
    /// int2e_ipvip1 (∇_i∇_j), rank 9, headroom i+1, j+1.
    Ipvip1,
    /// int2e_ip1ip2 (∇_i∇_k), rank 9, headroom i+1, k+1.
    Ip1ip2,
    /// int2e_ipip1ipip2 (∇²_i∇²_k), rank 81, headroom i+2, k+2.
    Ipip1ipip2,
    /// int2e_ipvip1ipvip2 (∇_i∇_j∇_k∇_l), rank 81, one derivative per center.
    Ipvip1ipvip2,
}

impl Hess2eKind {
    fn ncomp(self) -> usize {
        match self {
            Hess2eKind::Ipip1 | Hess2eKind::Ipvip1 | Hess2eKind::Ip1ip2 => 9,
            Hess2eKind::Ipip1ipip2 | Hess2eKind::Ipvip1ipvip2 => 81,
        }
    }
    /// (i_inc, j_inc, k_inc, l_inc) headroom raised on the plain G-tensor.
    fn headroom(self) -> (usize, usize, usize, usize) {
        match self {
            Hess2eKind::Ipip1 => (2, 0, 0, 0),
            Hess2eKind::Ipvip1 => (1, 1, 0, 0),
            Hess2eKind::Ip1ip2 => (1, 0, 1, 0),
            Hess2eKind::Ipip1ipip2 => (2, 0, 2, 0),
            Hess2eKind::Ipvip1ipvip2 => (1, 1, 1, 1),
        }
    }
}

/// Host-routed 2e Hessian launcher (Phase 25 HESS-02 / D-07).
///
/// Mirrors [`launch_two_electron_ip1`]/[`launch_two_electron_ip2`] but emits
/// `ncomp` (9 or 81) components via the verbatim-from-hess.c gout helpers in
/// `f12.rs` (`gout_ipip1`/`gout_ipvip1`/`gout_ip1ip2`/`gout_ipip1ipip2`). The plain
/// Coulomb G-tensor is built with the per-family headroom and the launcher routes
/// through the HOST `fill_g_tensor_2e` (→ `rys_roots_host`) so nroots 6..12
/// Hessian-elevated d-quartets hit the FND-02 host Rys engine, not the device
/// comptime kernel. This family was not flipped by task 33-03 — only the scalar
/// `int2e` path was — so its device ceiling stays at `BASE_DEVICE_NROOTS`.
/// Spinor → UnsupportedApi (D-11).
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_hess2e<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    kind: Hess2eKind,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Spinor Hessian: not supported (D-11). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor 2e Hessian".to_owned(),
        });
    }

    let ncomp = kind.ncomp();
    let (i_inc, j_inc, k_inc, l_inc) = kind.headroom();

    // Per-family headroom shape (D-09): raise the G-tensor angular momenta so the
    // gout's nabla compositions can read up to the elevated indices.
    // D-PBC-24 P2-1: ω sizes the shape as well as selecting the `CINTg0_2e` arm.
    // `Hess2eKind::headroom` and `cintx_runtime::range_omega::derivative_headroom`
    // must agree — the planner sizes the workspace from the latter.
    let range_omega = plan.operator_env_params.range_omega;
    let grad_shape = build_2e_shape_omega(
        li as usize + i_inc,
        lj as usize + j_inc,
        lk as usize + k_inc,
        ll as usize + l_inc,
        range_omega,
    );

    // FND-02 host Rys ceiling: nroots 6..12 route here; >12 stays fail-closed.
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = ncomp * block_len;

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    // S1 follow-up (`def2_speed_memory_optimization_plan.md`): the same
    // `env[PTR_EXPCUTOFF]` cutoff the scalar/batched 2e path applies
    // (`crate::kernels::pair_table`) screens this derivative operator's
    // primitive quartets too, via the same `QuartetExpScreen` every
    // derivative launcher in this module builds.
    let crate::kernels::pair_table::QuartetExpScreen {
        expcutoff,
        bra_screen,
        ket_screen,
        log_maxc_i,
        log_maxc_j,
        log_maxc_k,
        log_maxc_l,
    } = crate::kernels::pair_table::QuartetExpScreen::new(
        plan,
        [li, lj, lk, ll],
        [n_prim_i, n_prim_j, n_prim_k, n_prim_l],
        [n_ctr_i, n_ctr_j, n_ctr_k, n_ctr_l],
        [
            &shell_i.exponents,
            &shell_j.exponents,
            &shell_k.exponents,
            &shell_l.exponents,
        ],
        [
            &shell_i.coefficients,
            &shell_j.coefficients,
            &shell_k.coefficients,
            &shell_l.coefficients,
        ],
        [ri, rj, rk, rl],
    );

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let cceij = bra_screen.cceij(ai, aj, log_maxc_i[pi], log_maxc_j[pj]);
            if !(cceij < expcutoff) {
                continue;
            }
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let ccekl = ket_screen.cceij(ak, al, log_maxc_k[pk], log_maxc_l[pl]);
                    if !(ccekl < expcutoff) {
                        continue;
                    }
                    let eijcutoff = expcutoff - ccekl;
                    if !(cceij <= eijcutoff) {
                        continue;
                    }
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    // Plain Coulomb G-tensor at the elevated headroom.
                    let Some(g) = fill_g_tensor_2e_range(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                        range_omega,
                    )?
                    else {
                        // Short range past EXPCUTOFF_SR: this primitive quartet
                        // contributes nothing (g2e.c:4460). Not zeros — nothing.
                        continue;
                    };

                    // Reuse the verbatim hess.c gout permutation. gout is called at
                    // BASE (li,lj,lk,ll); the G-tensor carries the headroom. Returns
                    // interleaved out[n*ncomp+comp]; n walks [cl,ck,cj,ci].
                    let li_b = li as usize;
                    let lj_b = lj as usize;
                    let lk_b = lk as usize;
                    let ll_b = ll as usize;
                    let gout = match kind {
                        Hess2eKind::Ipip1 => crate::kernels::f12::gout_ipip1(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            ai,
                        ),
                        Hess2eKind::Ipvip1 => crate::kernels::f12::gout_ipvip1(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            ai,
                            aj,
                        ),
                        Hess2eKind::Ip1ip2 => crate::kernels::f12::gout_ip1ip2(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            ai,
                            ak,
                        ),
                        Hess2eKind::Ipip1ipip2 => crate::kernels::f12::gout_ipip1ipip2(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            ai,
                            ak,
                        ),
                        Hess2eKind::Ipvip1ipvip2 => crate::kernels::f12::gout_ipvip1ipvip2(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                    };

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
                                        * total_len;
                                    // TRANSPOSE interleaved gout[n*ncomp+comp] into
                                    // the component-leading block: cart[comp*block + n].
                                    for n in 0..block_len {
                                        for comp in 0..ncomp {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * ncomp + comp];
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

    // Component-leading `[ncomp, nl, nk, nj, ni]` F-order write.
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            for comp in 0..ncomp {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            for comp in 0..ncomp {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
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
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Spinor => unreachable!("spinor 2e Hessian rejected above"),
    }

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// Host-routed launcher for the `intor2.c` gauge / cross-product 2e families
/// (W4-06: `int2e_ip1v_r1`, `int2e_ip1v_rc1`, `int2e_ipvg1_xp1`, `int2e_ipvg2_xp1`).
///
/// Mirrors [`launch_two_electron_hess2e`]: plain Coulomb G-tensor at the family's
/// `ng[0..3]` headroom, verbatim-from-`intor2.c` gout, host Rys throughout. Every one
/// of these families exceeds `BASE_DEVICE_NROOTS` already at a `d` quartet
/// (`(3+4+2+2)/2 + 1 = 6`), so there is no device path to fall back from.
///
/// Unlike the Hessian families these are spin-free in BOTH electrons
/// (`ng[5] == ng[6] == 1`, spinor driver = `c2s_sf_2e1 + c2s_sf_2e2`), so cart, sph
/// and spinor are all served — the spinor arm folds each tensor component through the
/// same `cart_to_spinor_sf_4d` path the scalar `int2e_spinor` launcher uses.
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_gauge2e<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    kind: Gauge2eKind,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    let ncomp = kind.ncomp();
    let (i_inc, j_inc, k_inc, l_inc) = kind.headroom();
    // Structural guard for W4-06 risk R-05: the gauge origin is read ONLY for the one
    // family whose cascade actually uses it (`ip1v_rc1`, `G2E_RCJ`). The other three
    // raise about a basis centre or use a plain stride shift, and feeding them a
    // non-zero origin would be a silent wrong answer rather than an error.
    let common_orig = if kind.uses_common_origin() {
        plan.operator_env_params
            .common_orig
            .unwrap_or([0.0, 0.0, 0.0])
    } else {
        [0.0, 0.0, 0.0]
    };

    let grad_shape = build_2e_shape(
        li as usize + i_inc,
        lj as usize + j_inc,
        lk as usize + k_inc,
        ll as usize + l_inc,
    );
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = ncomp * block_len;

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];
    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);
    // libcint applies the per-family `envs.common_factor *= ...` BEFORE the quartet loop.
    let common_factor = common_factor * kind.common_factor_scale();

    // S1 follow-up (`def2_speed_memory_optimization_plan.md`): the same
    // `env[PTR_EXPCUTOFF]` cutoff the scalar/batched 2e path applies
    // (`crate::kernels::pair_table`) screens this derivative operator's
    // primitive quartets too, via the same `QuartetExpScreen` every
    // derivative launcher in this module builds.
    let crate::kernels::pair_table::QuartetExpScreen {
        expcutoff,
        bra_screen,
        ket_screen,
        log_maxc_i,
        log_maxc_j,
        log_maxc_k,
        log_maxc_l,
    } = crate::kernels::pair_table::QuartetExpScreen::new(
        plan,
        [li, lj, lk, ll],
        [n_prim_i, n_prim_j, n_prim_k, n_prim_l],
        [n_ctr_i, n_ctr_j, n_ctr_k, n_ctr_l],
        [
            &shell_i.exponents,
            &shell_j.exponents,
            &shell_k.exponents,
            &shell_l.exponents,
        ],
        [
            &shell_i.coefficients,
            &shell_j.coefficients,
            &shell_k.coefficients,
            &shell_l.coefficients,
        ],
        [ri, rj, rk, rl],
    );

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let cceij = bra_screen.cceij(ai, aj, log_maxc_i[pi], log_maxc_j[pj]);
            if !(cceij < expcutoff) {
                continue;
            }
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let ccekl = ket_screen.cceij(ak, al, log_maxc_k[pk], log_maxc_l[pl]);
                    if !(ccekl < expcutoff) {
                        continue;
                    }
                    let eijcutoff = expcutoff - ccekl;
                    if !(cceij <= eijcutoff) {
                        continue;
                    }
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    let g = fill_g_tensor_2e(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                    );

                    let gout = crate::kernels::f12::gout_gauge2e(
                        kind,
                        &g,
                        &grad_f12_shape,
                        li as usize,
                        lj as usize,
                        lk as usize,
                        ll as usize,
                        ai,
                        aj,
                        ak,
                        al,
                        ri,
                        rj,
                        rk,
                        rl,
                        common_orig,
                    );

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
                                        * total_len;
                                    for n in 0..block_len {
                                        for comp in 0..ncomp {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * ncomp + comp];
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

    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            if staging.len() < ncomp * sph_block {
                return Err(cintxRsError::BufferTooSmall {
                    required: ncomp * sph_block,
                    provided: staging.len(),
                });
            }
            for comp in 0..ncomp {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            if staging.len() < ncomp * cart_block {
                return Err(cintxRsError::BufferTooSmall {
                    required: ncomp * cart_block,
                    provided: staging.len(),
                });
            }
            for comp in 0..ncomp {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
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
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
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
        Representation::Spinor => {
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            let kappa_k = shell_k.kappa;
            let kappa_l = shell_l.kappa;
            let di = spinor_len(li, kappa_i as i32);
            let dj = spinor_len(lj, kappa_j as i32);
            let dk = spinor_len(lk, kappa_k as i32);
            let dl = spinor_len(ll, kappa_l as i32);
            let n2c_i = n_ctr_i * di;
            let n2c_j = n_ctr_j * dj;
            let n2c_k = n_ctr_k * dk;
            let n2c_l = n_ctr_l * dl;
            let spinor_block = n2c_i * n2c_j * n2c_k * n2c_l * 2;
            if staging.len() < ncomp * spinor_block {
                return Err(cintxRsError::BufferTooSmall {
                    required: ncomp * spinor_block,
                    provided: staging.len(),
                });
            }
            // Apply the imaginary-ket phase on the REAL cart blocks, before the
            // transform, so no `F`-space negation is needed.
            let phase = kind.spinor_phase();
            if phase != 1.0 {
                for v in cart_blocks.iter_mut() {
                    *v *= phase;
                }
            }
            let mut tmp = vec![F::from_f64_lossy(0.0); di * dj * dk * dl * 2];
            for comp in 0..ncomp {
                let staging_comp_base = comp * spinor_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                cart_to_spinor_sf_4d::<F>(
                                    &mut tmp,
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    kappa_i,
                                    lj,
                                    kappa_j,
                                    lk,
                                    kappa_k,
                                    ll,
                                    kappa_l,
                                )?;
                                for l_sp in 0..dl {
                                    let lidx = cl * dl + l_sp;
                                    for k_sp in 0..dk {
                                        let kidx = ck * dk + k_sp;
                                        for j_sp in 0..dj {
                                            let jidx = cj * dj + j_sp;
                                            for i_sp in 0..di {
                                                let iidx = ci * di + i_sp;
                                                let src = (((l_sp * dk + k_sp) * dj + j_sp) * di
                                                    + i_sp)
                                                    * 2;
                                                let dst = staging_comp_base
                                                    + (((lidx * n2c_k + kidx) * n2c_j + jidx)
                                                        * n2c_i
                                                        + iidx)
                                                        * 2;
                                                staging[dst] = tmp[src];
                                                staging[dst + 1] = tmp[src + 1];
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

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;
    let staging_bytes = std::mem::size_of_val(staging);
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

/// Which spin-free 2e GIAO family a [`launch_two_electron_giao2e`] call evaluates
/// (Phase 26 GIAO-02 / D-16). Each variant carries its component_rank, its plain
/// Coulomb G-tensor headroom, and the per-family libcint `common_factor` multiplier.
#[derive(Clone, Copy)]
enum Giao2eKind {
    /// int2e_g1 (gauge on e1, rank 3), headroom i+1, cf ×0.5.
    G1,
    /// int2e_ig1 (sign-flipped g1, rank 3), headroom i+1, cf ×0.5.
    Ig1,
    /// int2e_gg1 (2nd-order gauge on e1, rank 9), headroom i+2, cf ×0.25.
    Gg1,
    /// int2e_g1g2 (gauge on both e1+e2, rank 9), headroom i+2 & k+1, cf ×-0.25 (D-16).
    G1g2,
}

impl Giao2eKind {
    fn ncomp(self) -> usize {
        match self {
            Giao2eKind::G1 | Giao2eKind::Ig1 => 3,
            Giao2eKind::Gg1 | Giao2eKind::G1g2 => 9,
        }
    }
    /// (i_inc, j_inc, k_inc) headroom raised on the plain G-tensor (ll never raised).
    /// g1/ig1: a single R0I needs i+1. gg1: R0I(R0I(·,i+1)) needs i+2. g1g2:
    /// R0I(R0K(·,i+1)) needs i+2 on the i-side and k+1 for the R0K shift.
    fn headroom(self) -> (usize, usize, usize) {
        match self {
            Giao2eKind::G1 | Giao2eKind::Ig1 => (1, 0, 0),
            Giao2eKind::Gg1 => (2, 0, 0),
            Giao2eKind::G1g2 => (2, 0, 1),
        }
    }
    /// libcint per-family `common_factor` multiplier (intor4.c:1323 / intor2.c).
    fn common_factor_scale(self) -> f64 {
        match self {
            Giao2eKind::G1 | Giao2eKind::Ig1 => 0.5,
            Giao2eKind::Gg1 => 0.25,
            Giao2eKind::G1g2 => -0.25,
        }
    }
}

/// Host-routed spin-free 2e GIAO launcher (Phase 26 GIAO-02 / D-16).
///
/// Mirrors [`launch_two_electron_hess2e`] but emits COMPLEX-INTERLEAVED staging:
/// the GIAO families are purely imaginary, so the device emits the REAL magnitude
/// and the host materializes `[re=0, im=value]` pairs for the FND-03 `Complex<f64>`
/// view (D-15). gout combos are transcribed verbatim from libcint autocode via the
/// f12.rs `gout_g1`/`gout_ig1`/`gout_gg1`/`gout_g1g2` helpers (built on the
/// `r0i_2e`/`r0k_2e` position operators). Spinor → UnsupportedApi (D-11).
#[allow(clippy::too_many_arguments)]
fn launch_two_electron_giao2e<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    kind: Giao2eKind,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // Spinor GIAO: not supported (D-11). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor 2e GIAO".to_owned(),
        });
    }

    let ncomp = kind.ncomp();
    let (i_inc, j_inc, k_inc) = kind.headroom();
    let cf = common_factor * kind.common_factor_scale();

    // Per-family headroom shape (D-12: raise ket-side k via ng, not bra).
    let grad_shape = build_2e_shape(
        li as usize + i_inc,
        lj as usize + j_inc,
        lk as usize + k_inc,
        ll as usize,
    );

    // FND-02 host Rys ceiling: nroots 6..12 route here; >12 stays fail-closed.
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl;
    let total_len = ncomp * block_len;

    let nsi = nsph(li);
    let nsj = nsph(lj);
    let nsk = nsph(lk);
    let nsl = nsph(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    // S1 follow-up (`def2_speed_memory_optimization_plan.md`): the same
    // `env[PTR_EXPCUTOFF]` cutoff the scalar/batched 2e path applies
    // (`crate::kernels::pair_table`) screens this derivative operator's
    // primitive quartets too, via the same `QuartetExpScreen` every
    // derivative launcher in this module builds.
    let crate::kernels::pair_table::QuartetExpScreen {
        expcutoff,
        bra_screen,
        ket_screen,
        log_maxc_i,
        log_maxc_j,
        log_maxc_k,
        log_maxc_l,
    } = crate::kernels::pair_table::QuartetExpScreen::new(
        plan,
        [li, lj, lk, ll],
        [n_prim_i, n_prim_j, n_prim_k, n_prim_l],
        [n_ctr_i, n_ctr_j, n_ctr_k, n_ctr_l],
        [
            &shell_i.exponents,
            &shell_j.exponents,
            &shell_k.exponents,
            &shell_l.exponents,
        ],
        [
            &shell_i.coefficients,
            &shell_j.coefficients,
            &shell_k.coefficients,
            &shell_l.coefficients,
        ],
        [ri, rj, rk, rl],
    );

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pj in 0..n_prim_j {
            let aj = shell_j.exponents[pj];
            let cceij = bra_screen.cceij(ai, aj, log_maxc_i[pi], log_maxc_j[pj]);
            if !(cceij < expcutoff) {
                continue;
            }
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..n_prim_k {
                let ak = shell_k.exponents[pk];
                for pl in 0..n_prim_l {
                    let al = shell_l.exponents[pl];
                    let ccekl = ket_screen.cceij(ak, al, log_maxc_k[pk], log_maxc_l[pl]);
                    if !(ccekl < expcutoff) {
                        continue;
                    }
                    let eijcutoff = expcutoff - ccekl;
                    if !(cceij <= eijcutoff) {
                        continue;
                    }
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = cf * pdata_ij.fac * pdata_kl.fac;

                    // Plain Coulomb G-tensor at the elevated headroom.
                    let g = fill_g_tensor_2e(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                    );

                    let li_b = li as usize;
                    let lj_b = lj as usize;
                    let lk_b = lk as usize;
                    let ll_b = ll as usize;
                    // gout is called at BASE (li,lj,lk,ll); the G-tensor carries the
                    // headroom. Returns interleaved out[n*ncomp+comp]; n walks
                    // [cl,ck,cj,ci] (matching the Hess2e / ip1 convention).
                    let gout = match kind {
                        Giao2eKind::G1 => crate::kernels::f12::gout_g1(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            &ri,
                            &rj,
                        ),
                        Giao2eKind::Ig1 => crate::kernels::f12::gout_ig1(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            &ri,
                            &rj,
                        ),
                        Giao2eKind::Gg1 => crate::kernels::f12::gout_gg1(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            &ri,
                            &rj,
                        ),
                        Giao2eKind::G1g2 => crate::kernels::f12::gout_g1g2(
                            &g,
                            &grad_f12_shape,
                            li_b,
                            lj_b,
                            lk_b,
                            ll_b,
                            &ri,
                            &rj,
                            &rk,
                            &rl,
                        ),
                    };

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
                                        * total_len;
                                    // TRANSPOSE interleaved gout[n*ncomp+comp] into
                                    // the component-leading block: cart[comp*block + n].
                                    for n in 0..block_len {
                                        for comp in 0..ncomp {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * ncomp + comp];
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

    // COMPLEX-INTERLEAVED component-leading write: the GIAO families are purely
    // imaginary, so each real value `v` is materialized as `[re=0, im=v]` (D-15 /
    // FND-03). staging is sized 2 * ncomp * ni*nj*nk*nl (complex_output=true). Fail
    // closed on undersized staging (FND-06 / no silent partial write).
    let real_total = match plan.representation {
        Representation::Spheric => {
            ncomp * (n_ctr_i * nsi) * (n_ctr_j * nsj) * (n_ctr_k * nsk) * (n_ctr_l * nsl)
        }
        Representation::Cart => {
            ncomp * (n_ctr_i * nfi) * (n_ctr_j * nfj) * (n_ctr_k * nfk) * (n_ctr_l * nfl)
        }
        Representation::Spinor => unreachable!("spinor 2e GIAO rejected above"),
    };
    let needed = 2 * real_total;
    if staging.len() < needed {
        return Err(cintxRsError::BufferTooSmall {
            required: needed,
            provided: staging.len(),
        });
    }
    // Zero the interleaved buffer so the real (re) half is exactly 0.0 (D-07).
    for slot in staging.iter_mut().take(needed) {
        *slot = F::from_f64_lossy(0.0);
    }

    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dj = n_ctr_j * nsj;
            let dk = n_ctr_k * nsk;
            let dl = n_ctr_l * nsl;
            let sph_block = di * dj * dk * dl;
            for comp in 0..ncomp {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
                                let sph = cart_to_sph_2e(
                                    &cart_blocks[base..base + block_len],
                                    li,
                                    lj,
                                    lk,
                                    ll,
                                );
                                for ml in 0..nsl {
                                    let lidx = cl * nsl + ml;
                                    for mk in 0..nsk {
                                        let kidx = ck * nsk + mk;
                                        for mj in 0..nsj {
                                            let jidx = cj * nsj + mj;
                                            for mi in 0..nsi {
                                                let iidx = ci * nsi + mi;
                                                let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
                                                // [re=0, im=value] at 2*dst.
                                                staging[2 * dst + 1] = F::from_f64_lossy(sph[src]);
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
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dj = n_ctr_j * nfj;
            let dk = n_ctr_k * nfk;
            let dl = n_ctr_l * nfl;
            let cart_block = di * dj * dk * dl;
            for comp in 0..ncomp {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        for ck in 0..n_ctr_k {
                            for cl in 0..n_ctr_l {
                                let base = (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                    * total_len
                                    + comp * block_len;
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
                                                let dst = staging_comp_base
                                                    + iidx
                                                    + di * (jidx + dj * (kidx + dk * lidx));
                                                staging[2 * dst + 1] =
                                                    F::from_f64_lossy(block[src]);
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
        Representation::Spinor => unreachable!("spinor 2e GIAO rejected above"),
    }

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    // WR-04: GIAO output is [re=0, im=v] interleaved; count
    // the imaginary component only so not0 matches libcint's real double* semantics.
    let not0 = staging
        .chunks_exact(2)
        .filter(|c| c[1].abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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

/// Phase 29 Wave-2 (REL-03 / D-03 BLOCKING): plan-based wrapper for the
/// `int2e_spsp1` Spinor path. Extracts the four shells from `plan` and drives
/// [`launch_int2e_spsp1_spinor_quartet`].
#[allow(clippy::too_many_arguments)]
fn launch_int2e_spsp1_spinor<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let exps_i: Vec<f64> = shell_i.exponents[..shell_i.nprim as usize].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..shell_j.nprim as usize].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..shell_k.nprim as usize].to_vec();
    let exps_l: Vec<f64> = shell_l.exponents[..shell_l.nprim as usize].to_vec();
    let coeff_i: Vec<f64> =
        shell_i.coefficients[..shell_i.nprim as usize * shell_i.nctr as usize].to_vec();
    let coeff_j: Vec<f64> =
        shell_j.coefficients[..shell_j.nprim as usize * shell_j.nctr as usize].to_vec();
    let coeff_k: Vec<f64> =
        shell_k.coefficients[..shell_k.nprim as usize * shell_k.nctr as usize].to_vec();
    let coeff_l: Vec<f64> =
        shell_l.coefficients[..shell_l.nprim as usize * shell_l.nctr as usize].to_vec();

    launch_int2e_spsp1_spinor_quartet::<F>(
        li,
        shell_i.kappa,
        lj,
        shell_j.kappa,
        lk,
        shell_k.kappa,
        ll,
        shell_l.kappa,
        shell_i.nprim as usize,
        shell_j.nprim as usize,
        shell_k.nprim as usize,
        shell_l.nprim as usize,
        shell_i.nctr as usize,
        shell_j.nctr as usize,
        shell_k.nctr as usize,
        shell_l.nctr as usize,
        ri,
        rj,
        rk,
        rl,
        common_factor,
        &exps_i,
        &exps_j,
        &exps_k,
        &exps_l,
        &coeff_i,
        &coeff_j,
        &coeff_k,
        &coeff_l,
        staging,
    )?;

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;
    let staging_bytes = std::mem::size_of_val(staging);
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

/// Phase 29 Wave-2 (REL-03 / D-03 BLOCKING gate): drive the thinnest 2e σ family
/// `int2e_spsp1_spinor` through the brand-new 2e si/sf transform suite for one shell
/// quartet `(i, j, k, l)`.
///
/// The σ·p₁ operator `(σ·∇_i)(σ·∇_j)` acts on electron 1. The host σ·p assembler
/// ([`crate::kernels::f12::gout_spsp1`], = libcint `CINTgout2e_int2e_spsp1`) builds
/// the four component-leading cart blocks `gc_x/gc_y/gc_z/gc_1` (each a full 2e
/// `[ncl][nck][ncj][nci]` i-fastest KET-major block) per contraction quad. The
/// pairing is `c2s_si_2e1` (electron 1, real bra σ-mix) + `c2s_sf_2e2` (electron 2,
/// spin-free), exactly as `int2e_spsp1_spinor` selects in intor4.c:85.
///
/// # Layout
/// Output is the flat interleaved-complex spinor block via `zcopy_iklj` inside
/// `cart_to_spinor_sf_2e2`:
/// `staging[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2 + {0:re,1:im}]`, with each spinor
/// extent from `spinor_len` (kappa≠0 → 2l or 2l+2, NEVER 4l+2). Total length
/// `ni_sp*nj_sp*nk_sp*nl_sp*2`.
///
/// # nctr>1 (D-02 fixture rides shell-i nctr=2)
/// Loops the contraction quads; the electron-1 transform's `opij` and electron-2's
/// `zcopy_iklj` store carry the contraction-major spinor AO grid.
///
/// # Fail-closed (Phase-28 CR-01 / T-29-07)
/// A staging guard `required = ni_sp*nj_sp*nk_sp*nl_sp*2` rejects BEFORE any write
/// (OOM-safe stop, no partial writes) — this inline 2e arm bypasses any
/// `launch_*_pair` guard.
///
/// `coeff_*` are ROW-major `[ip*nctr + ic]` (the cintx `Shell` convention).
#[allow(clippy::too_many_arguments)]
pub fn launch_int2e_spsp1_spinor_quartet<F: CintFloat>(
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    lk: u8,
    kappa_k: i16,
    ll: u8,
    kappa_l: i16,
    nprim_i: usize,
    nprim_j: usize,
    nprim_k: usize,
    nprim_l: usize,
    nctr_i: usize,
    nctr_j: usize,
    nctr_k: usize,
    nctr_l: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    exps_l: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
    coeff_l: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl; // a single component's cart block
    const NGC: usize = 4; // gc_x, gc_y, gc_z, gc_1

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let dk = spinor_len(lk, kappa_k as i32);
    let dl = spinor_len(ll, kappa_l as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;
    let nk_sp = nctr_k * dk;
    let nl_sp = nctr_l * dl;

    // ── Fail-closed staging guard (T-29-07) BEFORE any write. ──
    let staging_required = ni_sp * nj_sp * nk_sp * nl_sp * 2;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // ── σ·p₁ assembler: 4 component-leading cart blocks per contraction quad. ──
    // headroom = Hess2eKind::Ipvip1 = (i_inc=1, j_inc=1, k_inc=0) so gout_spsp1's
    // nabla1j(g,li+1) + nabla1i compositions can read the elevated indices.
    let grad_shape = build_2e_shape(li as usize + 1, lj as usize + 1, lk as usize, ll as usize);
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }
    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    let total_len = NGC * block_len; // per-quad component-leading extent
    let mut cart_blocks = vec![0.0_f64; nctr_i * nctr_j * nctr_k * nctr_l * total_len];

    for pi in 0..nprim_i {
        let ai = exps_i[pi];
        for pj in 0..nprim_j {
            let aj = exps_j[pj];
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..nprim_k {
                let ak = exps_k[pk];
                for pl in 0..nprim_l {
                    let al = exps_l[pl];
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    let g = fill_g_tensor_2e(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                    );

                    // gout called at BASE (li,lj,lk,ll); G-tensor carries headroom.
                    // Returns interleaved out[n*4+comp]; n walks [cl,ck,cj,ci].
                    let gout = crate::kernels::f12::gout_spsp1(
                        &g,
                        &grad_f12_shape,
                        li as usize,
                        lj as usize,
                        lk as usize,
                        ll as usize,
                        ai,
                        aj,
                    );

                    for ci in 0..nctr_i {
                        let ci_coeff = coeff_i[pi * nctr_i + ci];
                        for cj in 0..nctr_j {
                            let cj_coeff = coeff_j[pj * nctr_j + cj];
                            for ck in 0..nctr_k {
                                let ck_coeff = coeff_k[pk * nctr_k + ck];
                                for cl in 0..nctr_l {
                                    let cl_coeff = coeff_l[pl * nctr_l + cl];
                                    let weight = ci_coeff * cj_coeff * ck_coeff * cl_coeff;
                                    let base = (((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl)
                                        * total_len;
                                    // TRANSPOSE interleaved gout[n*4+comp] into the
                                    // four contiguous component-leading cart blocks
                                    // cart[comp*block + n] (gc_x|gc_y|gc_z|gc_1).
                                    for n in 0..block_len {
                                        for comp in 0..NGC {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * NGC + comp];
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

    // ── Per contraction quad: electron-1 si transform → opij, then electron-2 sf
    //    transform → the (di×dj×dk×dl) spinor sub-block, scattered contraction-major. ──
    let ij_stride = di * dj;
    let opij_len = nfk * nfl * ij_stride * 2;
    let mut opij = vec![0.0_f64; opij_len];
    let mut sub = vec![F::from_f64_lossy(0.0); di * dj * dk * dl * 2];

    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            for ck in 0..nctr_k {
                for cl in 0..nctr_l {
                    let base = (((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl) * total_len;
                    let gc_x = &cart_blocks[base..base + block_len];
                    let gc_y = &cart_blocks[base + block_len..base + 2 * block_len];
                    let gc_z = &cart_blocks[base + 2 * block_len..base + 3 * block_len];
                    let gc_1 = &cart_blocks[base + 3 * block_len..base + 4 * block_len];

                    // Electron 1 (real bra σ-mix + ordinary ket) — owns KET→BRA transpose.
                    for v in opij.iter_mut() {
                        *v = 0.0;
                    }
                    cart_to_spinor_si_2e1(
                        &mut opij, gc_x, gc_y, gc_z, gc_1, li, kappa_i, lj, kappa_j, lk, ll,
                    )?;

                    // Electron 2 (spin-free) — apply_2d_spinor_zf + a_ket1 + zcopy_iklj.
                    for v in sub.iter_mut() {
                        *v = F::from_f64_lossy(0.0);
                    }
                    cart_to_spinor_sf_2e2::<F>(
                        &mut sub, &opij, li, kappa_i, lj, kappa_j, lk, kappa_k, ll, kappa_l,
                    )?;

                    // Scatter the (di×dj×dk×dl) sub-block into the contraction-major
                    // spinor AO grid. sub layout (zcopy_iklj):
                    //   sub[(((l*dk+k)*dj+j)*di+i)*2 + {re,im}].
                    for l in 0..dl {
                        let l_g = cl * dl + l;
                        for k in 0..dk {
                            let k_g = ck * dk + k;
                            for j in 0..dj {
                                let j_g = cj * dj + j;
                                for i in 0..di {
                                    let i_g = ci * di + i;
                                    let src = (((l * dk + k) * dj + j) * di + i) * 2;
                                    let dst =
                                        (((l_g * nk_sp + k_g) * nj_sp + j_g) * ni_sp + i_g) * 2;
                                    staging[dst] = sub[src];
                                    staging[dst + 1] = sub[src + 1];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Map a 2e Group-4 σ family operator name to its (gout, e1×e2 transform) pairing
/// (29-RESEARCH §Per-Family Map 2e, AUTHORITATIVE). Returns `None` for non-Group-4
/// operators (which fall through to the scalar/other dispatch). `spsp1` is handled
/// by its dedicated arm (the D-03 vehicle) and is intentionally NOT here.
pub fn rel2e_family_dispatch(name: &str) -> Option<(Rel2eGout, E1Transform, E2Transform)> {
    use E1Transform::*;
    use E2Transform as E2;
    match name {
        // REL-03 (intor4.c): 1-sided σ → si_2e1 + sf_2e2.
        "srsr1" => Some((Rel2eGout::Srsr1, Si, E2::Sf)),
        "spsp2" => Some((Rel2eGout::Spsp2, Sf, E2::Si)),
        // REL-03 (intor4.c): 2-sided σ → si_2e1 + si_2e2.
        "spsp1spsp2" => Some((Rel2eGout::Spsp1spsp2, Si, E2::Si)),
        "srsr1srsr2" => Some((Rel2eGout::Srsr1srsr2, Si, E2::Si)),
        "ipspsp1" => Some((Rel2eGout::IpSpsp1, Si, E2::Sf)),
        "ip1spsp2" => Some((Rel2eGout::Ip1Spsp2, Sf, E2::Si)),
        "ipspsp1spsp2" => Some((Rel2eGout::IpSpsp1Spsp2, Si, E2::Si)),
        "ipsrsr1" => Some((Rel2eGout::IpSrsr1, Si, E2::Sf)),
        "ip1srsr2" => Some((Rel2eGout::Ip1Srsr2, Sf, E2::Si)),
        "ipsrsr1srsr2" => Some((Rel2eGout::IpSrsr1Srsr2, Si, E2::Si)),
        // REL-04 (gaunt1.c): ssp/sps → si_2e1i + si_2e2i (BOTH imaginary).
        "ssp1ssp2" => Some((Rel2eGout::Ssp1ssp2, SiI, E2::SiI)),
        "ssp1sps2" => Some((Rel2eGout::Ssp1sps2, SiI, E2::SiI)),
        "sps1ssp2" => Some((Rel2eGout::Sps1ssp2, SiI, E2::SiI)),
        "sps1sps2" => Some((Rel2eGout::Sps1sps2, SiI, E2::SiI)),
        // REL-04 (dkb.c): 1-sided vsp/spv → si_2e1 + sf_2e2.
        "spv1" => Some((Rel2eGout::Spv1, Si, E2::Sf)),
        "vsp1" => Some((Rel2eGout::Vsp1, Si, E2::Sf)),
        // REL-04 (dkb.c): 2-sided spv/vsp → si_2e1 + si_2e2.
        "spv1spv2" => Some((Rel2eGout::Spv1spv2, Si, E2::Si)),
        "vsp1spv2" => Some((Rel2eGout::Vsp1spv2, Si, E2::Si)),
        "spv1vsp2" => Some((Rel2eGout::Spv1vsp2, Si, E2::Si)),
        "vsp1vsp2" => Some((Rel2eGout::Vsp1vsp2, Si, E2::Si)),
        "spv1spsp2" => Some((Rel2eGout::Spv1spsp2, Si, E2::Si)),
        "vsp1spsp2" => Some((Rel2eGout::Vsp1spsp2, Si, E2::Si)),
        _ => None,
    }
}

/// Plan-based wrapper for the generic REL-03/04 2e σ Spinor launcher: extracts the
/// four shells from `plan` and drives [`launch_rel2e_sigma_spinor_quartet`].
#[allow(clippy::too_many_arguments)]
fn launch_rel2e_sigma_spinor<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    gout_kind: Rel2eGout,
    e1: E1Transform,
    e2: E2Transform,
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_j = &shells[1];
    let shell_k = &shells[2];
    let shell_l = &shells[3];

    let exps_i: Vec<f64> = shell_i.exponents[..shell_i.nprim as usize].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..shell_j.nprim as usize].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..shell_k.nprim as usize].to_vec();
    let exps_l: Vec<f64> = shell_l.exponents[..shell_l.nprim as usize].to_vec();
    let coeff_i: Vec<f64> =
        shell_i.coefficients[..shell_i.nprim as usize * shell_i.nctr as usize].to_vec();
    let coeff_j: Vec<f64> =
        shell_j.coefficients[..shell_j.nprim as usize * shell_j.nctr as usize].to_vec();
    let coeff_k: Vec<f64> =
        shell_k.coefficients[..shell_k.nprim as usize * shell_k.nctr as usize].to_vec();
    let coeff_l: Vec<f64> =
        shell_l.coefficients[..shell_l.nprim as usize * shell_l.nctr as usize].to_vec();

    launch_rel2e_sigma_spinor_quartet::<F>(
        gout_kind,
        e1,
        e2,
        li,
        shell_i.kappa,
        lj,
        shell_j.kappa,
        lk,
        shell_k.kappa,
        ll,
        shell_l.kappa,
        shell_i.nprim as usize,
        shell_j.nprim as usize,
        shell_k.nprim as usize,
        shell_l.nprim as usize,
        shell_i.nctr as usize,
        shell_j.nctr as usize,
        shell_k.nctr as usize,
        shell_l.nctr as usize,
        ri,
        rj,
        rk,
        rl,
        common_factor,
        &exps_i,
        &exps_j,
        &exps_k,
        &exps_l,
        &coeff_i,
        &coeff_j,
        &coeff_k,
        &coeff_l,
        staging,
    )?;

    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;
    let staging_bytes = std::mem::size_of_val(staging);
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

/// Per-electron spinor transform selection for a 2e σ family (29-RESEARCH §2e map).
#[derive(Clone, Copy, PartialEq)]
pub enum E1Transform {
    /// `c2s_sf_2e1` — spin-free electron-1 transform.
    Sf,
    /// `c2s_si_2e1` — real bra σ-mix + ordinary ket.
    Si,
    /// `c2s_si_2e1i` — bra σ-mix + imaginary (×i) ket.
    SiI,
}

#[derive(Clone, Copy, PartialEq)]
pub enum E2Transform {
    /// `c2s_sf_2e2` — spin-free (single scalar e2 block).
    Sf,
    /// `c2s_si_2e2` — σ-mix on e2 (four e2 blocks ox/oy/oz/o1).
    Si,
    /// `c2s_si_2e2i` — σ-mix on e2, imaginary ket.
    SiI,
}

/// The σ·p / σ·r G-tensor "gout" a 2e Group-4 family emits, and the headroom its
/// derivative/shift composition needs. Each variant transcribes one libcint
/// `CINTgout2e_int2e_*` (intor4.c / gaunt1.c / dkb.c).
#[derive(Clone, Copy, PartialEq)]
pub enum Rel2eGout {
    // REL-03 (intor4.c)
    Spsp1,
    Spsp2,
    Srsr1,
    Spsp1spsp2,
    Srsr1srsr2,
    IpSpsp1,
    Ip1Spsp2,
    IpSpsp1Spsp2,
    IpSrsr1,
    Ip1Srsr2,
    IpSrsr1Srsr2,
    // REL-04 gaunt1.c (rank-9, ncomp 16)
    Ssp1ssp2,
    Ssp1sps2,
    Sps1ssp2,
    Sps1sps2,
    // REL-04 dkb.c rank-4 (ncomp 4, 1-sided σ·∇)
    Spv1,
    Vsp1,
    // REL-04 dkb.c rank-9 2-sided (ncomp 16)
    Spv1spv2,
    Vsp1spv2,
    Spv1vsp2,
    Vsp1vsp2,
    // REL-04 dkb.c rank-27 2-sided (ncomp 16)
    Spv1spsp2,
    Vsp1spsp2,
}

impl Rel2eGout {
    /// Output component count (4 for 1-sided families, 16 for 2-sided σ⊗σ).
    fn ncomp(self) -> usize {
        use Rel2eGout::*;
        match self {
            Spsp1 | Spsp2 | Srsr1 | Spv1 | Vsp1 => 4,
            IpSpsp1 | Ip1Spsp2 | IpSrsr1 | Ip1Srsr2 => 12,
            IpSpsp1Spsp2 | IpSrsr1Srsr2 => 48,
            _ => 16,
        }
    }
    fn visible_rank(self) -> usize {
        use Rel2eGout::*;
        match self {
            IpSpsp1 | Ip1Spsp2 | IpSpsp1Spsp2 | IpSrsr1 | Ip1Srsr2 | IpSrsr1Srsr2 => 3,
            _ => 1,
        }
    }
    /// Headroom (i_inc, j_inc, k_inc, l_inc) for the G-tensor build — the libcint
    /// `ng[0..3]` increments (read verbatim from each driver's optimizer ng).
    fn headroom(self) -> (usize, usize, usize, usize) {
        use Rel2eGout::*;
        match self {
            Spsp1 | Srsr1 => (1, 1, 0, 0),
            Spsp2 => (0, 0, 1, 1),
            Spsp1spsp2 | Srsr1srsr2 => (1, 1, 1, 1),
            IpSpsp1 | IpSrsr1 => (2, 1, 0, 0),
            Ip1Spsp2 | Ip1Srsr2 => (1, 0, 1, 1),
            IpSpsp1Spsp2 | IpSrsr1Srsr2 => (2, 1, 1, 1),
            Ssp1ssp2 => (0, 1, 0, 1),
            Ssp1sps2 => (0, 1, 1, 0),
            Sps1ssp2 => (1, 0, 0, 1),
            Sps1sps2 => (1, 0, 1, 0),
            Spv1 => (1, 0, 0, 0),
            Vsp1 => (0, 1, 0, 0),
            Spv1spv2 => (1, 0, 1, 0),
            Vsp1spv2 => (0, 1, 1, 0),
            Spv1vsp2 => (1, 0, 0, 1),
            Vsp1vsp2 => (0, 1, 0, 1),
            Spv1spsp2 => (1, 0, 1, 1),
            Vsp1spsp2 => (0, 1, 1, 1),
        }
    }
}

/// Generic 2e Group-4 σ Spinor launcher. Builds the family's cart σ-tensor blocks
/// per contraction quad via the family gout, then applies the per-electron transform
/// pair (`e1` × `e2`) and scatters the interleaved-complex spinor sub-blocks
/// contraction-major. Mirrors [`launch_int2e_spsp1_spinor_quartet`] but parameterized
/// over the family gout + transform pair (29-06 REL-03/04).
///
/// Fail-closed: a staging guard `required = ni_sp*nj_sp*nk_sp*nl_sp*2` rejects BEFORE
/// any write (Phase-28 CR-01 / T-29-11). `coeff_*` are ROW-major `[ip*nctr+ic]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_rel2e_sigma_spinor_quartet<F: CintFloat>(
    gout_kind: Rel2eGout,
    e1: E1Transform,
    e2: E2Transform,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    lk: u8,
    kappa_k: i16,
    ll: u8,
    kappa_l: i16,
    nprim_i: usize,
    nprim_j: usize,
    nprim_k: usize,
    nprim_l: usize,
    nctr_i: usize,
    nctr_j: usize,
    nctr_k: usize,
    nctr_l: usize,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    exps_k: &[f64],
    exps_l: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    coeff_k: &[f64],
    coeff_l: &[f64],
    staging: &mut [F],
) -> Result<(), cintxRsError> {
    let nfi = ncart(li);
    let nfj = ncart(lj);
    let nfk = ncart(lk);
    let nfl = ncart(ll);
    let block_len = nfi * nfj * nfk * nfl; // a single component's cart block
    let ngc = gout_kind.ncomp();
    let visible_rank = gout_kind.visible_rank();

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let dk = spinor_len(lk, kappa_k as i32);
    let dl = spinor_len(ll, kappa_l as i32);
    let ni_sp = nctr_i * di;
    let nj_sp = nctr_j * dj;
    let nk_sp = nctr_k * dk;
    let nl_sp = nctr_l * dl;

    // ── Fail-closed staging guard (T-29-11) BEFORE any write. ──
    let spinor_block_len = ni_sp * nj_sp * nk_sp * nl_sp * 2;
    let staging_required = visible_rank * spinor_block_len;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // ── G-tensor headroom per family. ──
    let (ii, ji, ki, li_inc) = gout_kind.headroom();
    let grad_shape = build_2e_shape(
        li as usize + ii,
        lj as usize + ji,
        lk as usize + ki,
        ll as usize + li_inc,
    );
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }
    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    let total_len = ngc * block_len; // per-quad component-leading extent
    let mut cart_blocks = vec![0.0_f64; nctr_i * nctr_j * nctr_k * nctr_l * total_len];

    for pi in 0..nprim_i {
        let ai = exps_i[pi];
        for pj in 0..nprim_j {
            let aj = exps_j[pj];
            let pdata_ij =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            for pk in 0..nprim_k {
                let ak = exps_k[pk];
                for pl in 0..nprim_l {
                    let al = exps_l[pl];
                    let pdata_kl = compute_pdata_host(
                        ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                    );
                    let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;

                    let g = fill_g_tensor_2e(
                        ai,
                        aj,
                        ak,
                        al,
                        &ri,
                        &rj,
                        &rk,
                        &rl,
                        grad_shape,
                        quartet_fac,
                    );

                    use crate::kernels::f12;
                    let (gli, glj, glk, gll) = (li as usize, lj as usize, lk as usize, ll as usize);
                    let gout = match gout_kind {
                        Rel2eGout::Spsp1 => {
                            f12::gout_spsp1(&g, &grad_f12_shape, gli, glj, glk, gll, ai, aj)
                        }
                        Rel2eGout::Spsp2 => {
                            f12::gout_spsp2(&g, &grad_f12_shape, gli, glj, glk, gll, ai, aj, ak, al)
                        }
                        Rel2eGout::Srsr1 => {
                            f12::gout_srsr1(&g, &grad_f12_shape, gli, glj, glk, gll)
                        }
                        Rel2eGout::Spsp1spsp2 => f12::gout_spsp1spsp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Srsr1srsr2 => {
                            f12::gout_srsr1srsr2(&g, &grad_f12_shape, gli, glj, glk, gll)
                        }
                        Rel2eGout::IpSpsp1 => f12::gout_ip_sigma(
                            0,
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Ip1Spsp2 => f12::gout_ip_sigma(
                            1,
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::IpSpsp1Spsp2 => f12::gout_ip_sigma(
                            2,
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::IpSrsr1 => f12::gout_ip_sigma(
                            3,
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Ip1Srsr2 => f12::gout_ip_sigma(
                            4,
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::IpSrsr1Srsr2 => f12::gout_ip_sigma(
                            5,
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Ssp1ssp2 => f12::gout_ssp1ssp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Ssp1sps2 => f12::gout_ssp1sps2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Sps1ssp2 => f12::gout_sps1ssp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Sps1sps2 => f12::gout_sps1sps2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Spv1 => {
                            f12::gout_spv1(&g, &grad_f12_shape, gli, glj, glk, gll, ai, aj, ak, al)
                        }
                        Rel2eGout::Vsp1 => {
                            f12::gout_vsp1(&g, &grad_f12_shape, gli, glj, glk, gll, ai, aj, ak, al)
                        }
                        Rel2eGout::Spv1spv2 => f12::gout_spv1spv2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Vsp1spv2 => f12::gout_vsp1spv2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Spv1vsp2 => f12::gout_spv1vsp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Vsp1vsp2 => f12::gout_vsp1vsp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Spv1spsp2 => f12::gout_spv1spsp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                        Rel2eGout::Vsp1spsp2 => f12::gout_vsp1spsp2(
                            &g,
                            &grad_f12_shape,
                            gli,
                            glj,
                            glk,
                            gll,
                            ai,
                            aj,
                            ak,
                            al,
                        ),
                    };

                    for ci in 0..nctr_i {
                        let ci_coeff = coeff_i[pi * nctr_i + ci];
                        for cj in 0..nctr_j {
                            let cj_coeff = coeff_j[pj * nctr_j + cj];
                            for ck in 0..nctr_k {
                                let ck_coeff = coeff_k[pk * nctr_k + ck];
                                for cl in 0..nctr_l {
                                    let cl_coeff = coeff_l[pl * nctr_l + cl];
                                    let weight = ci_coeff * cj_coeff * ck_coeff * cl_coeff;
                                    let base = (((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl)
                                        * total_len;
                                    for n in 0..block_len {
                                        for comp in 0..ngc {
                                            cart_blocks[base + comp * block_len + n] +=
                                                weight * gout[n * ngc + comp];
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

    // ── Per contraction quad: electron-1 transform → opij(s), then electron-2
    //    transform → the (di×dj×dk×dl) spinor sub-block, scattered contraction-major. ──
    let ij_stride = di * dj;
    let opij_len = nfk * nfl * ij_stride * 2;
    // For 2-sided e2, we need 4 opij arrays (ox/oy/oz/o1); for sf/1-sided, 1.
    let n_e2_blocks = if e2 == E2Transform::Sf { 1 } else { 4 };
    let mut opij_buf = vec![0.0_f64; n_e2_blocks * opij_len];
    let mut sub = vec![F::from_f64_lossy(0.0); di * dj * dk * dl * 2];

    // Run the electron-1 transform for one e2-group of 4 contiguous cart blocks
    // (e1-fast gout: comp = e2*4 + e1, see libcint CINT2e_spinor_drv gctr advance).
    let run_e1 = |opij_slot: &mut [f64],
                  cart_blocks: &[f64],
                  base: usize,
                  comp0: usize|
     -> Result<(), cintxRsError> {
        for v in opij_slot.iter_mut() {
            *v = 0.0;
        }
        let cb = |c: usize| &cart_blocks[base + c * block_len..base + (c + 1) * block_len];
        match e1 {
            E1Transform::Sf => {
                cart_to_spinor_sf_2e1(opij_slot, cb(comp0), li, kappa_i, lj, kappa_j, lk, ll)
            }
            E1Transform::Si => cart_to_spinor_si_2e1(
                opij_slot,
                cb(comp0),
                cb(comp0 + 1),
                cb(comp0 + 2),
                cb(comp0 + 3),
                li,
                kappa_i,
                lj,
                kappa_j,
                lk,
                ll,
            ),
            E1Transform::SiI => cart_to_spinor_si_2e1i(
                opij_slot,
                cb(comp0),
                cb(comp0 + 1),
                cb(comp0 + 2),
                cb(comp0 + 3),
                li,
                kappa_i,
                lj,
                kappa_j,
                lk,
                ll,
            ),
        }
    };

    for ci in 0..nctr_i {
        for cj in 0..nctr_j {
            for ck in 0..nctr_k {
                for cl in 0..nctr_l {
                    let base = (((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl) * total_len;

                    for group in 0..visible_rank {
                        let e1_components = if e1 == E1Transform::Sf { 1 } else { 4 };
                        let group_base = group * e1_components * n_e2_blocks;
                        match e2 {
                            E2Transform::Sf => {
                                // 1-sided: 4 e1 blocks (x,y,z,1) → one opij → sf_2e2.
                                let (slot, _) = opij_buf.split_at_mut(opij_len);
                                run_e1(slot, &cart_blocks, base, group_base)?;
                                for v in sub.iter_mut() {
                                    *v = F::from_f64_lossy(0.0);
                                }
                                cart_to_spinor_sf_2e2::<F>(
                                    &mut sub,
                                    &opij_buf[..opij_len],
                                    li,
                                    kappa_i,
                                    lj,
                                    kappa_j,
                                    lk,
                                    kappa_k,
                                    ll,
                                    kappa_l,
                                )?;
                            }
                            E2Transform::Si | E2Transform::SiI => {
                                // 2-sided: for each of the 4 e2 σ-components, run e1 on its
                                // 4 e1 blocks → opij[e2]; then feed ox/oy/oz/o1 to si_2e2(i).
                                for e2c in 0..4 {
                                    let mut scratch = vec![0.0_f64; opij_len];
                                    run_e1(
                                        &mut scratch,
                                        &cart_blocks,
                                        base,
                                        group_base + e2c * e1_components,
                                    )?;
                                    opij_buf[e2c * opij_len..(e2c + 1) * opij_len]
                                        .copy_from_slice(&scratch);
                                }
                                for v in sub.iter_mut() {
                                    *v = F::from_f64_lossy(0.0);
                                }
                                let (ox, rest) = opij_buf.split_at(opij_len);
                                let (oy, rest) = rest.split_at(opij_len);
                                let (oz, o1) = rest.split_at(opij_len);
                                let o1 = &o1[..opij_len];
                                if e2 == E2Transform::Si {
                                    cart_to_spinor_si_2e2::<F>(
                                        &mut sub, ox, oy, oz, o1, li, kappa_i, lj, kappa_j, lk,
                                        kappa_k, ll, kappa_l,
                                    )?;
                                } else {
                                    cart_to_spinor_si_2e2i::<F>(
                                        &mut sub, ox, oy, oz, o1, li, kappa_i, lj, kappa_j, lk,
                                        kappa_k, ll, kappa_l,
                                    )?;
                                }
                            }
                        }

                        // Scatter the (di×dj×dk×dl) sub-block into the contraction-major
                        // spinor AO grid: sub[(((l*dk+k)*dj+j)*di+i)*2 + {re,im}].
                        for l in 0..dl {
                            let l_g = cl * dl + l;
                            for k in 0..dk {
                                let k_g = ck * dk + k;
                                for j in 0..dj {
                                    let j_g = cj * dj + j;
                                    for i in 0..di {
                                        let i_g = ci * di + i;
                                        let src = (((l * dk + k) * dj + j) * di + i) * 2;
                                        let dst = group * spinor_block_len
                                            + (((l_g * nk_sp + k_g) * nj_sp + j_g) * ni_sp + i_g)
                                                * 2;
                                        staging[dst] = sub[src];
                                        staging[dst + 1] = sub[src + 1];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
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
    // `g2e.c:54-56` chains the four factors; see `int2e_common_factor`.
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
        * common_fac_sp(li)
        * common_fac_sp(lj)
        * common_fac_sp(lk)
        * common_fac_sp(ll);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    // ─────────────────────────────────────────────────────────────────────────
    // int2e_ip1 gradient path (Plan 21-05 / GRAD-07).
    //
    // The two-electron force ∇_A <ij|kl> — the highest-impact term in every
    // analytical HF/DFT/MP2/CCSD gradient. The first-derivative math is the
    // standard libcint `∂/∂A χ_l = -2α·χ_{l+1} + l·χ_{l-1}` (`CINTnabla1i_2e`),
    // reused VERBATIM from f12.rs (`gout_ip1` / `nabla1i_2e`, made pub(crate) in
    // Task 0). The only difference vs the F12 gradient is the G-tensor source:
    // here we feed `gout_ip1` the PLAIN Coulomb G-tensor from `fill_g_tensor_2e`
    // (`rys_roots_host`) instead of the F12 stg-roots tensor (D-04).
    //
    // The G-tensor is built with `li_ceil = li + 1` headroom so `nabla1i_2e` can
    // read up to index li+1; `gout_ip1` is then called at BASE li (the documented
    // headroom recipe from f12.rs:584-585 + 1163-1168).
    //
    // Output is component-leading `[3, nl, nk, nj, ni]` F-order: `gout_ip1`
    // returns interleaved `out[n*3+comp]` where `n` walks `[cl, ck, cj, ci]`
    // (ll slowest, li fastest); we TRANSPOSE to `staging[comp*block + n]`
    // matching pyscf-gto `layout_table.rs` (Risk R3, validated vs vendor in the
    // oracle test).
    if plan.descriptor.operator_name() == "ip1" {
        return launch_two_electron_ip1::<F>(
            plan,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // int2e_ip2 gradient path (Phase 23 DRV1-01): ∇ on the ket bra-center k.
    if plan.descriptor.operator_name() == "ip2" {
        return launch_two_electron_ip2::<F>(
            plan,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // Phase 25 HESS-02 (D-07): host-routed 2e Hessian families (rank 9 / 81).
    // int2e_ipip1/ipvip1/ip1ip2 (rank 9) + int2e_ipip1ipip2 (rank 81). All route
    // through fill_g_tensor_2e (FND-02 host Rys) so nroots≥6 d-quartets are served.
    if let Some(kind) = match plan.descriptor.operator_name() {
        "ipip1" => Some(Hess2eKind::Ipip1),
        "ipvip1" => Some(Hess2eKind::Ipvip1),
        "ip1ip2" => Some(Hess2eKind::Ip1ip2),
        "ipip1ipip2" => Some(Hess2eKind::Ipip1ipip2),
        "ipvip1ipvip2" => Some(Hess2eKind::Ipvip1ipvip2),
        _ => None,
    } {
        return launch_two_electron_hess2e::<F>(
            plan,
            kind,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // W4-06: host-routed intor2.c gauge / cross-product families (rank 9, cart+sph+
    // spinor). Keyed on operator_name (RULE 5), never on a positional OperatorId.
    if let Some(kind) = Gauge2eKind::from_operator_name(plan.descriptor.operator_name()) {
        return launch_two_electron_gauge2e::<F>(
            plan,
            kind,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // Phase 26 GIAO-02 (D-16): host-routed spin-free 2e GIAO families. int2e_g1/ig1
    // (rank 3) + int2e_gg1/g1g2 (rank 9). The device emits REAL components; the
    // FND-03 host materialization wraps them into the interleaved complex view.
    if let Some(kind) = match plan.descriptor.operator_name() {
        "g1" => Some(Giao2eKind::G1),
        "ig1" => Some(Giao2eKind::Ig1),
        "gg1" => Some(Giao2eKind::Gg1),
        "g1g2" => Some(Giao2eKind::G1g2),
        _ => None,
    } {
        return launch_two_electron_giao2e::<F>(
            plan,
            kind,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 29 Wave-2 (REL-03, D-03 BLOCKING gate): int2e_spsp1 Spinor path.
    //
    // The thinnest 2e σ family — the vehicle that proves the brand-new 2e si/sf
    // transform suite (cart_to_spinor_si_2e1 + cart_to_spinor_sf_2e2) byte-identical
    // to vendored libcint BEFORE any further 2e σ family wires onto it (29-06).
    //
    // σ·p₁ = (σ·∇_i)(σ·∇_j) acts on electron-1 (bra i, ket j); the host σ·p
    // assembler (gout_spsp1, = libcint CINTgout2e_int2e_spsp1) emits the four
    // component-leading gc_x/gc_y/gc_z/gc_1 cart blocks per quad. The c2s_si_2e1
    // transform (electron 1) folds them (owns the KET→BRA transpose), then
    // c2s_sf_2e2 (electron 2, spin-free) reorders into the interleaved-complex
    // spinor block. Spinor-only (the scalar/sph forms are not registered this
    // phase). nctr>1 is HANDLED (the kappa fixture rides shell-i nctr=2).
    // ─────────────────────────────────────────────────────────────────────────
    if plan.descriptor.operator_name() == "spsp1" {
        if plan.representation != Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: "int2e_spsp1 is Spinor-only (the D-03 2e transform proof \
                            vehicle); cart/spheric int2e_spsp1 is not registered this phase"
                    .to_owned(),
            });
        }
        return launch_int2e_spsp1_spinor::<F>(
            plan,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 29 Wave-3 (REL-03/04): the remaining 2e Group-4 σ Spinor families.
    // Each is Spinor-only; the per-family (gout, e1×e2 transform) pairing comes
    // from 29-RESEARCH §Per-Family Map 2e. The generic quartet launcher
    // (launch_rel2e_sigma_spinor_quartet) wires the family gout onto the proven
    // 2e si/sf transform suite with a per-arm fail-closed staging guard.
    // ─────────────────────────────────────────────────────────────────────────
    if let Some((gout_kind, e1, e2)) = rel2e_family_dispatch(plan.descriptor.operator_name()) {
        if plan.representation != Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "int2e_{} is Spinor-only (Group-4 relativistic σ); cart/spheric \
                     not registered",
                    plan.descriptor.operator_name()
                ),
            });
        }
        return launch_rel2e_sigma_spinor::<F>(
            plan,
            gout_kind,
            e1,
            e2,
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            staging,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar 2e device dispatch (quick task 260529-q4k).
    //
    // The per-(ci,cj,ck,cl) i-fastest Cartesian block accumulation
    // (`fill_g_tensor_2e` → `contract_2e_cart`, weighted + summed over every
    // primitive and contraction quad) now runs in the `two_electron_scalar_kernel`
    // `#[cube(launch)]` device kernel via `run_2e_scalar_device`, dispatched on
    // the resolved backend (CPU / Wgpu / Cuda / ROCm / Metal). The returned
    // `cart_blocks` has the IDENTICAL layout the host loop produced (block
    // (ci,cj,ck,cl) at offset (((ci*n_ctr_j+cj)*n_ctr_k+ck)*n_ctr_l+cl)*block_len,
    // i fastest within each block), so the representation-dispatch host scatter
    // below (cart_to_sph_2e / cart_to_spinor_sf_4d + contraction-major AO scatter)
    // consumes it unchanged — the host part of the honest host/device split.
    // ─────────────────────────────────────────────────────────────────────────

    // D-PBC-24: re-derive the shape under the range-separation parameter. Short
    // range doubles `nrys_roots` (g2e.c:76-79) and `nrys_roots` sets
    // `g_stride_i` / `g_stride_k` / `g_size`, so the omega has to reach the
    // stride metadata, not only the root evaluation. With `range_omega == None`
    // this is bit-for-bit the shape built above.
    //
    // Only the SCALAR int2e path is reached here — every derivative operator
    // returned above with its own `grad_shape`, and the validator refuses a set
    // omega on those rather than letting them run full range (D-PBC-24 §1).
    let range_omega = plan.operator_env_params.range_omega;
    let shape = build_2e_shape_omega(
        li as usize,
        lj as usize,
        lk as usize,
        ll as usize,
        range_omega,
    );

    // S1: libcint's own primitive-pair/quartet cutoff, from the raw API's
    // env[PTR_EXPCUTOFF] or the safe API's `ExecutionOptions::expcutoff`.
    // `None` (nothing set) falls back to libcint's own default, exactly as a
    // zero `env[PTR_EXPCUTOFF]` does in `g2e.c:57`.
    let expcutoff = plan
        .operator_env_params
        .expcutoff
        .unwrap_or(crate::kernels::pair_table::LIBCINT_EXPCUTOFF);

    // Fail-closed nroots guard BEFORE any dispatch: scalar
    // nroots = (li+lj+lk+ll)/2+1, and 12 is where the vendor itself would need
    // quadmath. Where the *device* stops inside that range is the family
    // ceiling resolved below, not this constant.
    if shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: format!(
                "2e kernel supports nroots<={HOST_RYS_NROOTS_CEILING}; \
                 got nroots={} for l=({li},{lj},{lk},{ll})",
                shape.nroots
            ),
        });
    }

    // Flatten the f64 per-shell exps/coeffs the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let exps_l: Vec<f64> = shell_l.exponents[..n_prim_l].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();
    let coeff_l: Vec<f64> = shell_l.coefficients[..n_prim_l * n_ctr_l].to_vec();

    let out_len = n_ctr_i * n_ctr_j * n_ctr_k * n_ctr_l * block_len;

    // Task 33-03: the boundary between the device kernel and the host
    // primitive loop is the family's ceiling, not a constant. With `int2e`
    // flipped onto the inline extended entry and this backend's FMA probe
    // passing, orders 6..=12 stay on the device; otherwise they fall to the
    // host loop below, exactly as before.
    let device_ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int2e,
    );
    // D-PBC-24 stage 4: the device `#[cube]` kernel selects `rys_root{1..5}` on a
    // comptime `nroots` at ONE argument; short range needs two evaluations plus a
    // root rescaling, and its doubled `nroots = 6` at `rys_order = 3` is above
    // `BASE_DEVICE_NROOTS` anyway. Until the device arms land, range separation
    // routes to the host engine — explicitly and logged, not incidentally.
    let route_host = cintx_runtime::range_omega::is_range_separated(range_omega);
    if route_host {
        tracing::debug!(
            family = "2e",
            omega = range_omega.unwrap_or(0.0),
            rys_order = shape.rys_order,
            nroots = shape.nroots,
            "range-separated 2e routed to the host Rys engine (D-PBC-24 stage 4)"
        );
    }
    let cart_blocks: Vec<f64> = if route_host || shape.nroots > device_ceiling {
        let mut cart_accum = vec![0.0f64; out_len];
        for pi in 0..n_prim_i {
            let ai = exps_i[pi];
            for pj in 0..n_prim_j {
                let aj = exps_j[pj];
                // `fill_g_tensor_2e` does NOT compute the bra/ket Gaussian-product
                // exponentials — the caller folds them into `fac_env`, exactly as
                // the device kernel folds `fac_ij * fac_kl` into `fac1` and as
                // `launch_two_electron_hess2e` already does here. This arm passed the
                // bare `common_factor`, dropping
                // `exp(-ai*aj/(ai+aj) * |Ri-Rj|^2) * exp(-ak*al/(ak+al) * |Rk-Rl|^2)`.
                //
                // Every single-centre quartet has both factors equal to 1, which is
                // why every fixture that reached this arm (and every def2-TZVP Rys-6
                // class whose shells all sit on one atom) was right anyway. The
                // multi-centre `(p,f|f,f)` classes were wrong by the reciprocal of the
                // missing factor — a uniform 5.37x on H-centred def2-TZVP water.
                let pdata_ij =
                    compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
                for pk in 0..n_prim_k {
                    let ak = exps_k[pk];
                    for pl in 0..n_prim_l {
                        let al = exps_l[pl];
                        let pdata_kl = compute_pdata_host(
                            ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0,
                        );
                        let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;
                        let Some(g) = fill_g_tensor_2e_range(
                            ai,
                            aj,
                            ak,
                            al,
                            &ri,
                            &rj,
                            &rk,
                            &rl,
                            shape,
                            quartet_fac,
                            range_omega,
                        )?
                        else {
                            // Short-range integrand past EXPCUTOFF_SR: this
                            // primitive quartet contributes nothing (g2e.c:4460).
                            continue;
                        };
                        let cart_prim = contract_2e_cart(&g, shape, li, lj, lk, ll);
                        // `Shell::coefficients` is PRIMITIVE-major (`coeff[p*nctr + c]`,
                        // WR-03 in `cintx_compat::raw`) — the same layout the device
                        // kernel reads. This host arm indexed it contraction-major,
                        // which is only harmless when `nctr == 1` or `nprim == 1`;
                        // for a general contraction above the device Rys ceiling it
                        // silently pulled the wrong coefficient. Found by the def2-TZVP
                        // class sweep on `(p,f|f,f)` (nroots 6, max |diff| 5.8e-1).
                        for ci in 0..n_ctr_i {
                            let ci_coeff = coeff_i[pi * n_ctr_i + ci];
                            for cj in 0..n_ctr_j {
                                let cj_coeff = coeff_j[pj * n_ctr_j + cj];
                                for ck in 0..n_ctr_k {
                                    let ck_coeff = coeff_k[pk * n_ctr_k + ck];
                                    for cl in 0..n_ctr_l {
                                        let cl_coeff = coeff_l[pl * n_ctr_l + cl];
                                        let quad_weight = ci_coeff * cj_coeff * ck_coeff * cl_coeff;
                                        let block_offset =
                                            (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl)
                                                * block_len;
                                        for idx in 0..block_len {
                                            cart_accum[block_offset + idx] +=
                                                quad_weight * cart_prim[idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        cart_accum
    } else {
        match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(client) => run_2e_scalar_device::<cubecl::cpu::CpuRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                ll as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_prim_l as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                n_ctr_l as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ri,
                rj,
                rk,
                rl,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &exps_l,
                &coeff_i,
                &coeff_j,
                &coeff_k,
                &coeff_l,
                out_len,
                expcutoff,
            ),
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(client, _) => run_2e_scalar_device::<cubecl_wgpu::WgpuRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                ll as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_prim_l as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                n_ctr_l as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ri,
                rj,
                rk,
                rl,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &exps_l,
                &coeff_i,
                &coeff_j,
                &coeff_k,
                &coeff_l,
                out_len,
                expcutoff,
            ),
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(client) => run_2e_scalar_device::<cubecl_cuda::CudaRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                ll as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_prim_l as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                n_ctr_l as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ri,
                rj,
                rk,
                rl,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &exps_l,
                &coeff_i,
                &coeff_j,
                &coeff_k,
                &coeff_l,
                out_len,
                expcutoff,
            ),
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(client) => run_2e_scalar_device::<cubecl_hip::HipRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                ll as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_prim_l as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                n_ctr_l as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ri,
                rj,
                rk,
                rl,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &exps_l,
                &coeff_i,
                &coeff_j,
                &coeff_k,
                &coeff_l,
                out_len,
                expcutoff,
            ),
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(client, _) => run_2e_scalar_device::<cubecl_wgpu::WgpuRuntime>(
                client,
                li as u32,
                lj as u32,
                lk as u32,
                ll as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_prim_k as u32,
                n_prim_l as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                n_ctr_k as u32,
                n_ctr_l as u32,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ri,
                rj,
                rk,
                rl,
                common_factor,
                &exps_i,
                &exps_j,
                &exps_k,
                &exps_l,
                &coeff_i,
                &coeff_j,
                &coeff_k,
                &coeff_l,
                out_len,
                expcutoff,
            ),
        }
    };

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
                            let base =
                                (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl) * block_len;
                            let sph = cart_to_sph_2e(
                                &cart_blocks[base..base + block_len],
                                li,
                                lj,
                                lk,
                                ll,
                            );
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
        Representation::Spinor => {
            // General-contraction (nctr>1) spin-free 2e cart→spinor (260601-aty).
            // The device scalar kernel already accumulated every (ci,cj,ck,cl) block
            // with ITS OWN per-column coefficients (out_len = nctr_i*…*nctr_l*block_len),
            // exactly as the Spheric/Cart arms above consume. This arm therefore only
            // transforms + scatters the already-contracted per-quad blocks — it does
            // NOT re-apply coefficients.
            //
            // cart_to_spinor_sf_4d reads i-fastest cart[((l*nck+k)*ncj+j)*nci+i]; the
            // device 4D block is emitted i-fastest block[ic + nfi*(jc + nfj*(kc +
            // nfk*lc))] (see the Cart arm below, which scatters it with NO transpose),
            // so the per-quad sub-block feeds sf_4d directly with NO transpose — only a
            // per-quad contraction-major scatter into the dense n2c^4 output.
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            let kappa_k = shell_k.kappa;
            let kappa_l = shell_l.kappa;
            let di = spinor_len(li, kappa_i as i32);
            let dj = spinor_len(lj, kappa_j as i32);
            let dk = spinor_len(lk, kappa_k as i32);
            let dl = spinor_len(ll, kappa_l as i32);
            let n2c_i = n_ctr_i * di; // dense bra1 spinor dim (contraction-major)
            let n2c_j = n_ctr_j * dj;
            let n2c_k = n_ctr_k * dk;
            let n2c_l = n_ctr_l * dl;

            // Fail-closed staging guard (T-aty-03, OOM-safe stop contract): refuse
            // before any write if the caller workspace cannot hold the full dense
            // interleaved-complex 4D spinor block. Prevents a partial write on nctr>1.
            let staging_required = n2c_i * n2c_j * n2c_k * n2c_l * 2;
            if staging.len() < staging_required {
                return Err(cintxRsError::BufferTooSmall {
                    required: staging_required,
                    provided: staging.len(),
                });
            }

            let mut tmp = vec![F::from_f64_lossy(0.0); di * dj * dk * dl * 2];
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    for ck in 0..n_ctr_k {
                        for cl in 0..n_ctr_l {
                            let base =
                                (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl) * block_len;
                            cart_to_spinor_sf_4d::<F>(
                                &mut tmp,
                                &cart_blocks[base..base + block_len],
                                li,
                                kappa_i,
                                lj,
                                kappa_j,
                                lk,
                                kappa_k,
                                ll,
                                kappa_l,
                            )?;
                            // tmp: staging[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2 +{re,im}].
                            // Scatter contraction-major in all four indices.
                            for l_sp in 0..dl {
                                let lidx = cl * dl + l_sp;
                                for k_sp in 0..dk {
                                    let kidx = ck * dk + k_sp;
                                    for j_sp in 0..dj {
                                        let jidx = cj * dj + j_sp;
                                        for i_sp in 0..di {
                                            let iidx = ci * di + i_sp;
                                            let src =
                                                (((l_sp * dk + k_sp) * dj + j_sp) * di + i_sp) * 2;
                                            let dst = (((lidx * n2c_k + kidx) * n2c_j + jidx)
                                                * n2c_i
                                                + iidx)
                                                * 2;
                                            staging[dst] = tmp[src];
                                            staging[dst + 1] = tmp[src + 1];
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
                            let base =
                                (((ci * n_ctr_j + cj) * n_ctr_k + ck) * n_ctr_l + cl) * block_len;
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

    // WR-06: precision-aware sentinel so f32 stale lanes (< f32 noise floor ~1e-7)
    // are not counted. The outer F32 arm already bounds staging to out_elems, so this
    // scan cannot touch stale upper-half lanes.
    let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
        1e-12
    } else {
        1e-18
    });
    let not0 = staging
        .iter()
        .filter(|&&v| v.abs() > nonzero_threshold)
        .count() as i32;

    let staging_bytes = std::mem::size_of_val(staging);
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
            launch_two_electron_typed::<f32>(
                backend,
                plan,
                specialization,
                &mut staging_f32[..out_elems],
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar 2e device-vs-host cross-check + f32 genericity (quick task 260529-q4k)
//
// The device kernel must reproduce the host
// `contract_2e_cart(fill_g_tensor_2e(...))` Cartesian buffer for a single
// primitive/contraction pair across all four HRR branches (ibase/kbase ∈
// {true,false} via li>lj / lk>ll), and must compile+launch for F = f32.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod device_tests {
    use super::*;

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host single-pair Cartesian reference: fill_g_tensor_2e → contract_2e_cart
    /// for one primitive and one contraction (coeff weights applied).
    #[allow(clippy::too_many_arguments)]
    fn host_cart_2e(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
        ai: f64,
        aj: f64,
        ak: f64,
        al: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        rk: [f64; 3],
        rl: [f64; 3],
        common_factor: f64,
        ci: f64,
        cj: f64,
        ck: f64,
        cl: f64,
    ) -> Vec<f64> {
        let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
        let pdata_ij =
            compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let pdata_kl =
            compute_pdata_host(ak, al, rk[0], rk[1], rk[2], rl[0], rl[1], rl[2], 1.0, 1.0);
        let quartet_fac = common_factor * pdata_ij.fac * pdata_kl.fac;
        let g = fill_g_tensor_2e(ai, aj, ak, al, &ri, &rj, &rk, &rl, shape, quartet_fac);
        let prim_cart = contract_2e_cart(&g, shape, li, lj, lk, ll);
        let weight = ci * cj * ck * cl;
        prim_cart.iter().map(|&v| v * weight).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_device_matches_host(
        li: u8,
        lj: u8,
        lk: u8,
        ll: u8,
        ai: f64,
        aj: f64,
        ak: f64,
        al: f64,
    ) {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 1.1];
        let rk = [0.7_f64, 0.0, 0.0];
        let rl = [0.0_f64, 0.9, 0.0];
        let ci = 0.9_f64;
        let cj = 1.1_f64;
        let ck = 0.8_f64;
        let cl = 1.2_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk)
            * common_fac_sp(ll);
        let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
        let nfi = ncart(li);
        let nfj = ncart(lj);
        let nfk = ncart(lk);
        let nfl = ncart(ll);
        let out_len = nfi * nfj * nfk * nfl;

        let host = host_cart_2e(
            li,
            lj,
            lk,
            ll,
            ai,
            aj,
            ak,
            al,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            ci,
            cj,
            ck,
            cl,
        );
        let dev = run_2e_scalar_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
            li as u32,
            lj as u32,
            lk as u32,
            ll as u32,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            shape.ibase as u32,
            shape.kbase as u32,
            shape.nroots as u32,
            ri,
            rj,
            rk,
            rl,
            common_factor,
            &[ai],
            &[aj],
            &[ak],
            &[al],
            &[ci],
            &[cj],
            &[ck],
            &[cl],
            out_len,
            crate::kernels::pair_table::LIBCINT_EXPCUTOFF,
        );

        assert_eq!(
            host.len(),
            dev.len(),
            "length mismatch for ({li},{lj},{lk},{ll})"
        );
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch ({li},{lj},{lk},{ll}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    // (s,s,s,s): nroots=1, ibase=false, kbase=false → lj2d branch.
    #[test]
    fn test_2e_device_ssss() {
        assert_device_matches_host(0, 0, 0, 0, 1.0, 0.8, 0.9, 1.1);
    }

    // (p,s,s,s): li>lj → ibase=true; lk==ll → kbase=false → il2d branch.
    #[test]
    fn test_2e_device_psss() {
        assert_device_matches_host(1, 0, 0, 0, 0.8, 1.0, 0.9, 1.1);
    }

    // (s,p,s,s): li<lj → ibase=false; kbase=false → lj2d branch.
    #[test]
    fn test_2e_device_spss() {
        assert_device_matches_host(0, 1, 0, 0, 1.0, 0.8, 0.9, 1.1);
    }

    // (p,p,s,s): li==lj → ibase=false, lj2d branch with i+k mixing.
    #[test]
    fn test_2e_device_ppss() {
        assert_device_matches_host(1, 1, 0, 0, 0.8, 0.9, 1.0, 1.1);
    }

    // (d,s,s,s): higher li, ibase=true → il2d branch, nroots=2.
    #[test]
    fn test_2e_device_dsss() {
        assert_device_matches_host(2, 0, 0, 0, 0.7, 1.0, 0.9, 1.1);
    }

    // (s,s,p,s): lk>ll → kbase=true, ibase=false → kj2d branch.
    #[test]
    fn test_2e_device_sspsk() {
        assert_device_matches_host(0, 0, 1, 0, 1.0, 0.8, 0.9, 1.1);
    }

    // ── kj2d regression guard (the `di`-vs-`dk` loop-bound bug) ──────────────
    //
    // `test_2e_device_sspsk` above exercises the kj2d branch, but with li=0 and
    // ll=0 — and that is exactly where the bug hid: `dk == nroots * (li + 1)`
    // collapses to `di == nroots` when li==0, and the second transfer loop does
    // not run at all when ll==0. Both conditions must be broken to see it, so
    // these cases carry li>=1 AND ll>=1 with ibase=false and kbase=true.
    //
    // Found by driving a full def2-SVP basis through a class-complete sweep;
    // the failing classes were (p,p,d,p), (p,d,d,p) and (d,d,d,p).

    // (p,p,d,p): ibase=false (li==lj), kbase=true (2>1), li>=1, ll>=1.
    #[test]
    fn test_2e_device_ppdp_kj2d_regression() {
        assert_device_matches_host(1, 1, 2, 1, 0.8, 1.0, 0.9, 1.1);
    }

    // (p,d,d,p): ibase=false (1<2), kbase=true, li>=1, ll>=1.
    #[test]
    fn test_2e_device_pddp_kj2d_regression() {
        assert_device_matches_host(1, 2, 2, 1, 0.7, 1.0, 0.9, 1.1);
    }

    // (d,d,d,p): the largest def2-SVP class that tripped the bug (max |diff|
    // was 1.17e1 before the fix).
    #[test]
    fn test_2e_device_dddp_kj2d_regression() {
        assert_device_matches_host(2, 2, 2, 1, 0.8, 0.9, 1.0, 1.1);
    }

    // (p,s,p,s): ibase=true, kbase=true → ik2d branch (the 4th HRR branch).
    #[test]
    fn test_2e_device_psps() {
        assert_device_matches_host(1, 0, 1, 0, 0.8, 1.0, 0.9, 1.1);
    }

    // (p,p,p,p): nroots=3, full b00 cross-coupling, lj2d branch.
    #[test]
    fn test_2e_device_pppp() {
        assert_device_matches_host(1, 1, 1, 1, 0.8, 0.9, 1.0, 1.1);
    }

    /// Genericity: the kernel compiles AND launches for F = f32. An s-s-s-s f32
    /// launch yields a finite result.
    #[test]
    fn test_two_electron_scalar_kernel_generic_f32() {
        let client = cpu_client();
        let shape = build_2e_shape(0, 0, 0, 0);
        let g_size = shape.g_size;
        let g_zero = vec![0.0_f32; 3 * g_size];
        let out_zero = [0.0_f32; 1];

        // Flattened four-shell basis: one primitive, one contraction each.
        let exps = [1.0_f32; 4];
        let coeffs = [1.0_f32; 4];
        let centers = [
            0.0_f32, 0.0, 0.0, // i
            0.0, 0.0, 1.1, // j
            0.7, 0.0, 0.0, // k
            0.0, 0.9, 0.0, // l
        ];
        let shell_meta: [u32; 16] = [
            0, 0, 1, 1, //
            1, 1, 1, 1, //
            2, 2, 1, 1, //
            3, 3, 1, 1, //
        ];
        // `[si, sj, sk, sl, out_off, class, kl_lo, kl_hi]` — one class, index
        // 0; the ket `(2,3)` is pair slot `2*4+3 = 11` of an unscreened table
        // of single-primitive shells, so its one row is row 11.
        let quartets: [u32; QUARTET_ROW_STRIDE] = [0, 1, 2, 3, 0, 0, 11, 12];
        let class_shape: [u32; TWO_E_SHAPE_STRIDE] = [
            0,
            0,
            0,
            0,
            shape.di as u32,
            shape.dk as u32,
            shape.dl as u32,
            shape.dj as u32,
            shape.g_size as u32,
            shape.nmax as u32,
            shape.mmax as u32,
            shape.g2d_ijmax as u32,
            shape.g2d_klmax as u32,
            // K1: the class's index table starts at 0.
            0,
            // F1: this class's Rys order, read per quartet since the fused
            // dispatch carries several.
            1,
        ];
        // One `(ss|ss)` Cartesian element, whose three G offsets are all 0.
        let class_idx: [u32; 3] = [0, 0, 0];
        // K2: one slot, one row; the cooperative arm reads index 0 only.
        let slot_bounds: [u32; 2] = [0, 1];

        let exps_h = client.create_from_slice(f32::as_bytes(&exps));
        let idx_h = client.create_from_slice(u32::as_bytes(&class_idx));
        let bounds_h = client.create_from_slice(u32::as_bytes(&slot_bounds));
        let coeffs_h = client.create_from_slice(f32::as_bytes(&coeffs));
        let centers_h = client.create_from_slice(f32::as_bytes(&centers));
        let meta_h = client.create_from_slice(u32::as_bytes(&shell_meta));
        let quartets_h = client.create_from_slice(u32::as_bytes(&quartets));
        let shape_h = client.create_from_slice(u32::as_bytes(&class_shape));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let class_factor = [((PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(0)
            * common_fac_sp(0)
            * common_fac_sp(0)
            * common_fac_sp(0)) as f32];
        let factor_h = client.create_from_slice(f32::as_bytes(&class_factor));
        // The extended-Rys tables are an unconditional kernel argument; an
        // `nroots = 1` smoke launch never reads them.
        let rys_tables = crate::math::rys_wheeler::ext_rys_tables();
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));

        // The primitive-pair table (S1): four single-primitive shells, so the
        // bra `(0,1)` and ket `(2,3)` each have exactly one surviving pair. The
        // rows are built the way `PairTable::build` would, but written out here
        // because this test's basis is four synthetic shells rather than a
        // `BatchShell` list — and because the values it asserts are the point.
        let pair_shells = [
            BatchShell {
                l: 0,
                nprim: 1,
                nctr: 1,
                exponents: vec![1.0],
                coefficients: vec![1.0],
                center: [0.0, 0.0, 0.0],
            },
            BatchShell {
                l: 0,
                nprim: 1,
                nctr: 1,
                exponents: vec![1.0],
                coefficients: vec![1.0],
                center: [0.0, 0.0, 1.1],
            },
            BatchShell {
                l: 0,
                nprim: 1,
                nctr: 1,
                exponents: vec![1.0],
                coefficients: vec![1.0],
                center: [0.7, 0.0, 0.0],
            },
            BatchShell {
                l: 0,
                nprim: 1,
                nctr: 1,
                exponents: vec![1.0],
                coefficients: vec![1.0],
                center: [0.0, 0.9, 0.0],
            },
        ];
        let pairs = crate::kernels::pair_table::PairTable::build(
            &pair_shells,
            crate::kernels::pair_table::PairTableOptions::unscreened(),
        );
        let pair_data: Vec<f32> = pairs.data.iter().map(|value| *value as f32).collect();
        let pair_data_h = client.create_from_slice(f32::as_bytes(&pair_data));
        let pair_index_h = client.create_from_slice(u32::as_bytes(&pairs.index));
        let pair_offset_h = client.create_from_slice(u32::as_bytes(&pairs.offset));
        let ctr_h = client.create_from_slice(f32::as_bytes(&[0.0_f32]));

        two_electron_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            crate::plane::cooperative_cube_dim::<cubecl::cpu::CpuRuntime>(&client, 1),
            unsafe { ArrayArg::from_raw_parts(exps_h, exps.len()) },
            unsafe { ArrayArg::from_raw_parts(coeffs_h, coeffs.len()) },
            unsafe { ArrayArg::from_raw_parts(centers_h, centers.len()) },
            unsafe { ArrayArg::from_raw_parts(meta_h, shell_meta.len()) },
            unsafe { ArrayArg::from_raw_parts(quartets_h, quartets.len()) },
            unsafe { ArrayArg::from_raw_parts(shape_h, class_shape.len()) },
            unsafe { ArrayArg::from_raw_parts(factor_h, class_factor.len()) },
            unsafe { ArrayArg::from_raw_parts(idx_h, class_idx.len()) },
            unsafe { ArrayArg::from_raw_parts(rys_tab_h, EXT_TABLES_LEN) },
            unsafe { ArrayArg::from_raw_parts(pair_data_h, pair_data.len()) },
            unsafe { ArrayArg::from_raw_parts(pair_index_h, pairs.index.len()) },
            unsafe { ArrayArg::from_raw_parts(pair_offset_h, pairs.offset.len()) },
            unsafe { ArrayArg::from_raw_parts(bounds_h, slot_bounds.len()) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size) },
            // Four segmented shells: the staged-contraction scratch is never
            // indexed, so the one-element placeholder is what a real dispatch
            // would bind too.
            unsafe { ArrayArg::from_raw_parts(ctr_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            PIE4 as f32,
            // No primitive screening: this test asserts the exact arithmetic.
            0.0_f32,
            // Unscreened: every pair the table holds is evaluated.
            f32::INFINITY,
            4u32,
            // A one-element `(ss|ss)` block, so the private accumulator applies.
            ACC_SLOTS_DEFAULT as u32,
            1u32,
            1u32,
            (3 * shape.g_size) as u32,
            0u32,
            1u32,
            // S3 split; with one lane it is the same build either way.
            1u32,
            shape.ibase as u32,
            shape.kbase as u32,
            // F1: `nr_max` — the widest Rys order the dispatch carries. One
            // here, so only the `rys_root1` arm is emitted and the vector VRR
            // arm is comptime-off.
            1u32,
            // Cooperative decomposition: one cube, one quartet, `cooperative_cube_dim`
            // lanes — the shape this test's single `3 * g_size` slab is sized for.
            0u32,
            // Global G slab: this test hands the kernel its own `3 * g_size`
            // buffer and asserts what lands in it.
            0u32,
            QUARTET_ROW_STRIDE as u32,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(
            out.is_finite(),
            "f32 scalar 2e kernel result must be finite"
        );
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

    // nroots ceiling (Phase 25 FND-02): the HOST gradient path now supports nroots 6..12
    // via the Wheeler engine. An all-f quartet has gradient nroots = (3+1+3+3+3)/2+1 = 7,
    // which previously returned UnsupportedApi but now routes to the host fill_g_tensor_2e
    // path. The fail-closed ceiling moves to nroots>12 (HOST_RYS_NROOTS_CEILING), e.g. an
    // all-i (l=6) quartet → gradient nroots = (6+1+6+6+6)/2+1 = 13 > 12 → UnsupportedApi.
    #[test]
    fn test_int2e_ip1_nroots_guard() {
        // (f,f,f,f) gradient nroots = 7 ∈ 6..=12 → now ALLOWED via the host Wheeler path.
        let (basis, shells, op) = build_ip1_plan_lll(3, 3, 3, 3);
        let ok = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "all-f int2e_ip1 quartet (nroots=7) must route to the host path (FND-02), got: {:?}",
            ok.err()
        );

        // (i,i,i,i) gradient nroots = (6+1+6+6+6)/2 + 1 = 13 > 12 → fail-closed (T-25-03).
        let (basis, shells, op) = build_ip1_plan_lll(6, 6, 6, 6);
        let result = run_ip1(&basis, shells, op, Representation::Spheric);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "all-i int2e_ip1 quartet (nroots=13 > 12) must return UnsupportedApi, got: {:?}",
            result.map(|(s, _)| s.len())
        );

        // (d,d,d,d) gradient nroots = (2+1+2+2+2)/2 + 1 = 5 → allowed.
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
        let q =
            query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
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

// ─────────────────────────────────────────────────────────────────────────────
// int2e_ip2 gradient tests (Phase 23 DRV1-01)
//
// ip2 is the ket-side (∇ on k) sibling of ip1. Same behavior contract:
//   - nroots guard: an all-f quartet (gradient nroots (li+lj+(lk+1)+ll)/2+1=7>5)
//     returns UnsupportedApi; an s/p/d quartet does not.
//   - component count: an (s,s,s,s) quartet → 3 outputs; (s,s,p,s) → 9.
//   - determinism: repeated evaluation is bit-identical.
//   - spinor: int2e_ip2 with Representation::Spinor returns UnsupportedApi.
//   - non-square sanity: an explicitly NON-SQUARE quartet (p on i, p on k in
//     different slots) is evaluated without panic and is nonzero (D-05 discipline).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod ip2_tests {
    use super::*;
    use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
    use crate::specialization::SpecializationKey;
    use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell, ShellTuple};
    use cintx_ops::resolver::Resolver;
    use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
    use std::sync::Arc;

    fn build_ip2_plan_lll(
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

        let op = Resolver::descriptor_by_symbol("int2e_ip2_sph")
            .expect("int2e_ip2_sph must be in manifest")
            .id;
        (basis, shells, op)
    }

    fn run_ip2(
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

        let out_elems = plan.output_layout.staging_elements;
        let mut staging = vec![0.0_f64; out_elems];
        let stats = launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok((staging, stats))
    }

    // nroots ceiling (Phase 25 FND-02): the HOST ip2 gradient path now supports nroots
    // 6..12 via the Wheeler engine. An all-f quartet (gradient nroots = (3+3+(3+1)+3)/2+1
    // = 7) routes to the host path; an all-i quartet ((6+6+(6+1)+6)/2+1 = 13 > 12) stays
    // fail-closed (T-25-03).
    #[test]
    fn test_int2e_ip2_nroots_guard() {
        // (f,f,f,f) gradient nroots = 7 ∈ 6..=12 → now ALLOWED via the host Wheeler path.
        let (basis, shells, op) = build_ip2_plan_lll(3, 3, 3, 3);
        let ok = run_ip2(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "all-f int2e_ip2 quartet (nroots=7) must route to the host path (FND-02), got: {:?}",
            ok.err()
        );

        // (i,i,i,i) gradient nroots = (6+6+(6+1)+6)/2 + 1 = 13 > 12 → fail-closed.
        let (basis, shells, op) = build_ip2_plan_lll(6, 6, 6, 6);
        let result = run_ip2(&basis, shells, op, Representation::Spheric);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "all-i int2e_ip2 quartet (nroots=13 > 12) must return UnsupportedApi, got: {:?}",
            result.map(|(s, _)| s.len())
        );

        let (basis, shells, op) = build_ip2_plan_lll(2, 2, 2, 2);
        let ok = run_ip2(&basis, shells, op, Representation::Spheric);
        assert!(
            ok.is_ok(),
            "(d,d,d,d) int2e_ip2 quartet (nroots=5) must be allowed, got: {:?}",
            ok.err()
        );
    }

    // Component count: (s,s,s,s) → 3; (s,s,p,s) → 3 * 1*1*3*1 = 9 (sph p on k).
    #[test]
    fn test_int2e_ip2_component_count() {
        let (basis, shells, op) = build_ip2_plan_lll(0, 0, 0, 0);
        let (staging, _stats) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(
            staging.len(),
            3,
            "(s,s,s,s) int2e_ip2 should produce 3 components"
        );

        let (basis, shells, op) = build_ip2_plan_lll(0, 0, 1, 0);
        let (staging, _stats) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(
            staging.len(),
            9,
            "(s,s,p,s) int2e_ip2 should produce 9 outputs"
        );
    }

    // Determinism: repeated evaluation is bit-identical on a NON-SQUARE quartet.
    #[test]
    fn test_int2e_ip2_determinism_nonsquare() {
        // p on i, p on k in different slots → non-square (ni=3, nk=3 but distinct
        // axes) and nonzero off-center.
        let (basis, shells, op) = build_ip2_plan_lll(1, 0, 1, 0);
        let (out1, _) = run_ip2(&basis, shells.clone(), op, Representation::Spheric).unwrap();
        let (out2, _) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out1.len(), out2.len());
        let any_nonzero = out1.iter().any(|v| v.abs() > 1e-14);
        assert!(
            any_nonzero,
            "int2e_ip2 (p,s,p,s) output is all-zero (regression)"
        );
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "int2e_ip2 output not bit-identical"
            );
        }
    }

    // Electron-exchange symmetry: int2e_ip2(i,j,k,l) must equal int2e_ip1(k,l,i,j)
    // (value multiset; the element ORDER differs because the AO indices permute).
    // This is the kernel-level guard that ip2's ∇_k reproduces ip1's proven ∇_i.
    #[test]
    fn test_int2e_ip2_matches_ip1_electron_swap() {
        // (p,s | s,p) on two atoms — distinct l's so a layout bug would show.
        let (li, lj, lk, ll) = (1u8, 0u8, 0u8, 1u8);
        let (basis, shells, op) = build_ip2_plan_lll(li, lj, lk, ll);
        let (ip2, _) = run_ip2(&basis, shells, op, Representation::Spheric).unwrap();

        // int2e_ip1 of the swapped quartet (k,l,i,j).
        let atom0 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom1 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom0, atom1].into_boxed_slice());
        let mk = |ai: u32, l: u8| {
            Arc::new(
                Shell::try_new(
                    ai,
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
        // swapped: i<-k(atom0), j<-l(atom1), k<-i(atom0), l<-j(atom1)
        let s0 = mk(0, lk);
        let s1 = mk(1, ll);
        let s2 = mk(0, li);
        let s3 = mk(1, lj);
        let all: Arc<[Arc<Shell>]> =
            Arc::from(vec![s0.clone(), s1.clone(), s2.clone(), s3.clone()].into_boxed_slice());
        let b2 = BasisSet::try_new(atoms, all).unwrap();
        let s2t = ShellTuple::try_from_iter([s0, s1, s2, s3]).unwrap();
        let op1 = Resolver::descriptor_by_symbol("int2e_ip1_sph").unwrap().id;

        let opts = ExecutionOptions::default();
        let q = query_workspace(op1, Representation::Spheric, &b2, s2t.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op1, Representation::Spheric, &b2, s2t, &q).unwrap();
        plan.precision = cintx_core::PrecisionKind::F64;
        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut ip1 = vec![0.0_f64; plan.output_layout.staging_elements];
        launch_two_electron_typed::<f64>(&backend, &plan, &spec, &mut ip1).unwrap();

        assert_eq!(ip2.len(), ip1.len());
        assert!(
            ip2.iter().any(|v| v.abs() > 1e-14),
            "ip2 swap-check is all-zero"
        );
        let round = |v: &f64| (v * 1e10).round() / 1e10;
        let mut a: Vec<f64> = ip2.iter().map(round).collect();
        let mut b: Vec<f64> = ip1.iter().map(round).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(
            a, b,
            "int2e_ip2 vs electron-swapped int2e_ip1 value multiset differs"
        );
    }

    // Spinor: int2e_ip2 with Representation::Spinor returns UnsupportedApi.
    #[test]
    fn test_int2e_ip2_spinor_unsupported() {
        let (basis, shells, op) = build_ip2_plan_lll(0, 0, 0, 0);
        let opts = ExecutionOptions::default();
        let q =
            query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
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
            "spinor int2e_ip2 should return UnsupportedApi, got: {:?}",
            result
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 34-A0 — cube-dimension A/B harness for the scalar 2e kernel.
//
// The 2e kernel distributes only its contraction block across the cube; the
// Rys roots and the entire VRR+HRR G-tensor build are serial on unit 0. This
// harness measures the steady-state cost of one shell quartet at a pinned
// `CINTX_2E_CUBE_DIM` so the parallel fraction can be bounded *before* the
// cooperative-G-tensor rewrite (34-A) is attempted.
//
// Run:
//   CINTX_2E_CUBE_DIM=1   cargo test --release -p cintx-cubecl --features cpu \
//     two_e_cube_dim_ab -- --ignored --nocapture
//   CINTX_2E_CUBE_DIM=256 ... (and 16/64)
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "cpu"))]
mod cube_dim_ab {
    use super::*;

    /// Representative def2-SVP-shaped quartets: (l-tuple, primitives per shell).
    const CASES: &[([u8; 4], usize)] = &[
        ([0, 0, 0, 0], 7),
        ([1, 1, 1, 1], 4),
        ([2, 2, 2, 2], 1),
        ([2, 2, 2, 2], 3),
    ];

    fn timed_quartet(l: [u8; 4], nprim: usize, reps: usize) -> f64 {
        let client = cubecl::cpu::CpuRuntime::client(&Default::default());
        let [li, lj, lk, ll] = l;
        let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
        let out_len = ncart(li) * ncart(lj) * ncart(lk) * ncart(ll);
        let exps: Vec<f64> = (0..nprim).map(|p| 0.8 * 2.5_f64.powi(p as i32)).collect();
        let coeffs: Vec<f64> = (0..nprim).map(|p| 0.4 + 0.05 * p as f64).collect();
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.0_f64, 0.0, 1.1];
        let rk = [0.7_f64, 0.0, 0.0];
        let rl = [0.0_f64, 0.9, 0.0];
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(li)
            * common_fac_sp(lj)
            * common_fac_sp(lk)
            * common_fac_sp(ll);

        let run = || {
            run_2e_scalar_device::<cubecl::cpu::CpuRuntime>(
                &client,
                li as u32,
                lj as u32,
                lk as u32,
                ll as u32,
                nprim as u32,
                nprim as u32,
                nprim as u32,
                nprim as u32,
                1,
                1,
                1,
                1,
                shape.di as u32,
                shape.dk as u32,
                shape.dl as u32,
                shape.dj as u32,
                shape.g_size as u32,
                shape.nmax as u32,
                shape.mmax as u32,
                shape.g2d_ijmax as u32,
                shape.g2d_klmax as u32,
                shape.ibase as u32,
                shape.kbase as u32,
                shape.nroots as u32,
                ri,
                rj,
                rk,
                rl,
                common_factor,
                &exps,
                &exps,
                &exps,
                &exps,
                &coeffs,
                &coeffs,
                &coeffs,
                &coeffs,
                out_len,
                crate::kernels::pair_table::LIBCINT_EXPCUTOFF,
            )
        };

        // Warm-up: pay the CubeCL specialization/JIT cost outside the timer.
        let _ = run();
        let start = std::time::Instant::now();
        for _ in 0..reps {
            let _ = run();
        }
        start.elapsed().as_secs_f64() * 1000.0 / reps as f64
    }

    #[test]
    #[ignore = "34-A0 measurement; run explicitly in release with --ignored"]
    fn two_e_cube_dim_ab() {
        let pinned = std::env::var("CINTX_2E_CUBE_DIM").unwrap_or_else(|_| "auto".to_owned());
        println!("\nCINTX_2E_CUBE_DIM={pinned}");
        println!(
            "{:<14} {:>6} {:>7} {:>7} {:>12}",
            "l-tuple", "nprim", "nroots", "block", "ms/quartet"
        );
        for &(l, nprim) in CASES {
            let shape = build_2e_shape(l[0] as usize, l[1] as usize, l[2] as usize, l[3] as usize);
            let block = ncart(l[0]) * ncart(l[1]) * ncart(l[2]) * ncart(l[3]);
            let reps = if nprim.pow(4) > 200 { 3 } else { 20 };
            let ms = timed_quartet(l, nprim, reps);
            println!(
                "{:<14} {:>6} {:>7} {:>7} {:>12.3}",
                format!("{l:?}"),
                nprim,
                shape.nroots,
                block,
                ms
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 34-B/34-E — batched shell-quartet evaluation.
//
// The per-tuple compatibility API (`eval_raw`) is the right *shape* for
// libcint compatibility and the wrong shape for throughput: every quartet pays
// a planner pass, twelve buffer allocations, a kernel dispatch and a blocking
// readback. Task 34-A0 removed the barrier cost from the kernel itself, which
// left that per-call overhead as the dominant term (~36 us/quartet against
// libcint's ~0.6 us on the same silicon).
//
// This entry point changes the unit of work to a *list* of quartets: the list
// is grouped into launch classes (which is what makes `nroots`, the HRR branch
// and the G-tensor shape comptime-constant within a dispatch), the basis is
// flattened and uploaded **once**, and each class is one dispatch and one
// readback.
// ─────────────────────────────────────────────────────────────────────────────

/// One shell in a batched 2e evaluation.
#[derive(Clone, Debug)]
pub struct BatchShell {
    /// Angular momentum.
    pub l: u8,
    /// Primitive count.
    pub nprim: u32,
    /// Contraction count.
    pub nctr: u32,
    /// `nprim` primitive exponents.
    pub exponents: Vec<f64>,
    /// `nprim * nctr` contraction coefficients, primitive-major
    /// (`coefficients[p * nctr + c]`) — the layout the scalar kernel has always
    /// consumed.
    pub coefficients: Vec<f64>,
    /// Shell center, in Bohr.
    pub center: [f64; 3],
}

impl BatchShell {
    /// Spherical AO count of this shell, including contraction.
    #[must_use]
    pub fn ao_len(&self) -> usize {
        nsph(self.l) * self.nctr as usize
    }
}

/// Tuning knobs for one batched evaluation.
///
/// The default is **exact**: every field's zero value reproduces the unscreened
/// arithmetic bit for bit, so a caller who does not opt in cannot lose accuracy
/// by accident.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TwoEBatchOptions {
    /// Primitive-quartet screening tolerance (Task 34-D).
    ///
    /// A primitive quartet whose G-tensor scale factor
    /// `sqrt(a0/a1^3) * common_factor * exp(-mu_ij R_ij^2) * exp(-mu_kl R_kl^2)`
    /// does not exceed this value is skipped entirely — no Rys roots, no VRR,
    /// no HRR, no contraction.
    ///
    /// `0.0` (the default) drops only quartets whose factor underflowed to
    /// exactly zero, so results are bit-identical to no screening at all. A
    /// positive value trades accuracy for work: the Rys weights and the
    /// recurrence coefficients are not bounded by one, so the factor is a proxy
    /// for the contribution rather than a bound on it.
    pub primitive_tolerance: f64,

    /// Ceiling on what this evaluation may hold at once, in bytes (M1).
    ///
    /// `None`, the default, means unbounded: one dispatch per launch signature,
    /// the whole work list in flight, exactly as before the memory plan existed.
    ///
    /// `Some(limit)` does two things. Group building caps each dispatch's
    /// Cartesian buffer, so a signature's quartets are split across as many
    /// dispatches as the budget requires; and a pre-flight plan refuses the
    /// batch with [`cintxRsError::MemoryLimitExceeded`] when even the chunked
    /// shape cannot fit — **before** the output buffer is allocated and before
    /// any launch, so a refusal leaves nothing partially written.
    ///
    /// Chunking changes no arithmetic. A chunk is a range of quartets, each
    /// quartet's evaluation is self-contained, and the transform writes rather
    /// than accumulates — so a chunked run is bit-identical to an unchunked one.
    pub memory_limit_bytes: Option<usize>,

    /// libcint's primitive-pair/quartet screening cutoff (S1), applied when the
    /// residency backing this batch is built.
    ///
    /// `None`, the default, uses [`crate::kernels::pair_table::LIBCINT_EXPCUTOFF`]
    /// — libcint's own `EXPCUTOFF = 60`. Unlike `primitive_tolerance` this is not
    /// an extra cintx screen: it is the same threshold `env[PTR_EXPCUTOFF]`
    /// controls under the raw API, so a batch caller who sets it here sees
    /// exactly the same primitive pairs a raw call with that `env[0]` would.
    pub expcutoff: Option<f64>,
}

/// [`TwoEBatchOptions`] under a family-neutral name.
///
/// The options block is one field — a primitive-screening tolerance — and it
/// means the same thing for every family that screens (Task 34-D2). The
/// concrete type's name predates the generalization, and renaming a public type
/// is not worth a break in the compatibility surface, so this alias carries the
/// general meaning instead. Same arrangement as
/// [`ResidentBasis`]/[`ResidentTwoEBasis`].
pub type BatchOptions = TwoEBatchOptions;

/// Auditable statistics for one batched evaluation.
///
/// A claimed speed-up is only credible if the launch and transfer counts that
/// produced it are visible, so these travel with the values rather than being
/// printed and discarded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchExecutionStats {
    /// Quartets evaluated.
    pub quartets: usize,
    /// Kernel dispatches — one per [`TwoELaunchSignature`] present in the list.
    pub kernel_launch_count: usize,
    /// Angular-momentum classes in the list.
    ///
    /// Before Task 35-M1 this was also the dispatch count. It is reported
    /// alongside [`Self::kernel_launch_count`] so the merge factor is visible
    /// rather than inferred: `launch_classes / kernel_launch_count` is what the
    /// merge actually bought on this work list.
    pub launch_classes: usize,
    /// Device-to-host readbacks (one per dispatch).
    pub readback_count: usize,
    /// Host-to-device bytes uploaded, basis included.
    ///
    /// Everything sized before a launch geometry is chosen: the basis, the
    /// pair table, the quartet rows, the class shapes and index tables, and
    /// the device transform's tables. The one upload it omits is the per-unit
    /// partition's row bounds (K2), `4 * (units + 1)` bytes per launch, which
    /// exist only once the cube width is known; those reach
    /// [`Self::device_table_bytes_total`].
    pub transfer_bytes: usize,
    /// The share of [`Self::transfer_bytes`] that was the basis upload.
    ///
    /// Zero on every call that reused a [`ResidentTwoEBasis`], which is how a
    /// device-resident basis is observed rather than assumed (Task 34-C).
    pub basis_upload_bytes: usize,
    /// Nanoseconds spent in backend dispatch: uploads, kernel launches and
    /// readbacks for every class.
    pub dispatch_ns: u64,
    /// Nanoseconds spent in the host cart-to-sph transform and scatter.
    ///
    /// Reported separately from [`Self::dispatch_ns`] because it is serial host
    /// work that no backend change can touch — keeping the two apart is what
    /// makes a throughput claim attributable.
    pub host_transform_ns: u64,
    /// Nanoseconds of [`Self::host_transform_ns`] spent allocating per-block
    /// buffers, **`0` unless `CINTX_HOST_TRANSFORM_PROFILE` is set**.
    ///
    /// See [`crate::transform::profile`]: the split is opt-in because the clock
    /// reads that produce it are not free against a 27-element block, and a
    /// profiled `host_transform_ns` carries that overhead too. Read the three
    /// as a ratio within one profiled run, never against an unprofiled one.
    pub host_transform_alloc_ns: u64,
    /// Nanoseconds of [`Self::host_transform_ns`] spent in the c2s arithmetic
    /// itself, **`0` unless `CINTX_HOST_TRANSFORM_PROFILE` is set**.
    pub host_transform_c2s_ns: u64,
    /// Nanoseconds of [`Self::host_transform_ns`] spent scattering spherical
    /// blocks into the caller's AO grid, **`0` unless
    /// `CINTX_HOST_TRANSFORM_PROFILE` is set**.
    ///
    /// This is the strided write, and it is the one of the three that no
    /// allocation removal or on-device transform can eliminate — an output
    /// materialized on the host has to be written somewhere.
    pub host_transform_scatter_ns: u64,
    /// Bytes of G-tensor scratch one slot owns in the widest dispatch.
    ///
    /// A merged dispatch sizes its slab to the widest `g_size` it carries, so a
    /// narrow class can be given more scratch than it needs. Reporting it keeps
    /// that cost observable — the merge is only free while this stays small
    /// against [`MAX_BATCH_SCRATCH_BYTES`].
    pub max_g_slab_bytes: usize,
    /// Bytes read back from the device (M3).
    ///
    /// Under the host transform this is the Cartesian intermediate, which is
    /// larger than the caller's output wherever `l > 1` — an f shell carries 10
    /// Cartesian components against 7 spherical. Under the device transform it
    /// is the spherical output itself. The gap between the two is what moving
    /// the transform buys on a backend where a readback is a real transfer.
    pub readback_bytes: usize,
    /// Work-list chunks this evaluation was split into (M1).
    ///
    /// `1` for a list whose Cartesian intermediate fits
    /// [`DEFAULT_CHUNK_CART_BYTES`] or the caller's budget; more when it does
    /// not. Each chunk is a run of consecutive quartets, dispatched, read back
    /// and transformed before the next one allocates.
    pub chunk_count: usize,

    // ── Memory accounting (`def2_speed_memory_optimization_plan.md` M6) ─────
    //
    // Every field below is planned bytes — computed from the same expressions
    // the dispatch allocates from — except the two named `device_*_peak` /
    // `device_allocs_added`, which are backend residency and stay `0` unless
    // `CINTX_BATCH_MEMORY_PROFILE` is set. See [`crate::memory_probe`].
    /// Bytes of the caller-visible spherical output this batch materialized.
    pub host_output_bytes: usize,
    /// Peak bytes of *Cartesian* device output held on the host at once.
    ///
    /// Today the whole run's Cartesian buffers are retained until the transform
    /// finishes, so this is their sum and the host peak is roughly
    /// `host_output_bytes + host_cart_bytes_peak`. M1's chunking and M3's
    /// device-side transform are measured by watching this fall.
    pub host_cart_bytes_peak: usize,
    /// Largest single Cartesian output buffer allocated on the device.
    pub device_out_bytes_peak: usize,
    /// G-tensor scratch summed over every dispatch of this run.
    pub device_g_slab_bytes_total: usize,
    /// Largest single G-tensor scratch allocation.
    pub device_g_slab_bytes_peak: usize,
    /// Quartet/shape/factor/Rys-table bytes uploaded, summed over dispatches.
    pub device_table_bytes_total: usize,
    /// Device allocations this run requested, counted from the plan.
    pub device_planned_allocs: u64,
    /// Peak backend `bytes_in_use`, `0` unless residency profiling is on.
    pub device_bytes_in_use_peak: u64,
    /// Backend allocation-count growth, `0` unless residency profiling is on.
    pub device_allocs_added: u64,
    /// Primitive quartets this work list contains: `Σ nprim_i·nprim_j·nprim_k·nprim_l`.
    ///
    /// The denominator of every arithmetic-per-primitive claim, and the
    /// baseline S1's pair cutoff is measured against.
    pub primitive_quartets_total: u64,
    /// Widest ket-pair split any dispatch of this run used (G1): `1` when no
    /// quartet was spread over more than one cube, which is always the case
    /// on the per-unit shape and under a memory budget.
    pub kl_split_max: u32,
    /// Primitive quartets the kernel is asked to evaluate.
    ///
    /// Equal to [`Self::primitive_quartets_total`] until S1's `expcutoff` pair
    /// screen lands; below it afterwards by exactly the pairs libcint's own
    /// `CINTset_pairdata` would have dropped. This counts what the *host* knows
    /// it dispatched, not a device-side tally: the cutoff is a function of pair
    /// data alone, so the host can count it exactly without an atomic in the
    /// hot loop.
    pub primitive_quartets_evaluated: u64,
}

/// Spherical AO blocks for a batch, plus the offsets that locate each quartet.
#[derive(Clone, Debug, Default)]
pub struct TwoEBatchOutput {
    /// Concatenated spherical AO blocks, in the caller's quartet order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where quartet `n`'s block starts in [`Self::values`].
    pub offsets: Vec<usize>,
    /// Execution statistics.
    pub stats: BatchExecutionStats,
}

/// Primitive quartets a work list contains: `Σ nprim_i·nprim_j·nprim_k·nprim_l`.
///
/// The denominator of every arithmetic-per-primitive claim (M6). It is the work
/// the kernel does today, and the baseline S1's `expcutoff` pair screen — the
/// one libcint's own `CINTset_pairdata` applies — is measured against.
///
/// `u64` because `nprim^4` at def2-TZVP sulfur is 2 401 per quartet and the
/// largest work list here is 181 070 quartets; `usize` would be fine on this
/// host and is not on a 32-bit one.
fn primitive_quartets_in(shells: &[BatchShell], quartets: &[[u32; 4]]) -> u64 {
    quartets
        .iter()
        .map(|quartet| {
            quartet
                .iter()
                .map(|&s| u64::from(shells[s as usize].nprim))
                .product::<u64>()
        })
        .sum()
}

/// Spherical AO block length of one quartet.
fn batch_sph_len(shells: &[BatchShell], quartet: [u32; 4]) -> usize {
    quartet
        .iter()
        .map(|&s| shells[s as usize].ao_len())
        .product()
}

/// Evaluate a list of shell quartets as `int2e_sph`, one dispatch per launch
/// class (Task 34-B).
///
/// Quartets are grouped by their angular-momentum class; within a class the
/// Rys order, HRR branch and G-tensor extents are constant, which is what lets
/// them stay comptime. The flattened basis is uploaded once per class rather
/// than once per quartet.
///
/// Output blocks are spherical, `i`-fastest, contraction-major — byte-identical
/// to what the per-quartet path writes for the same quartet.
///
/// # Errors
/// Returns [`cintxRsError::UnsupportedApi`] when a class needs more Rys roots
/// than the device kernel supports (`nroots > 5`); the batch is rejected as a
/// whole rather than silently returning zeros for part of it.
pub fn evaluate_2e_quartet_batch(
    backend: &ResolvedBackend,
    shells: &[BatchShell],
    quartets: &[[u32; 4]],
) -> Result<TwoEBatchOutput, cintxRsError> {
    let resident = ResidentTwoEBasis::new(backend, shells)?;
    evaluate_2e_quartet_batch_resident(backend, &resident, quartets)
}

/// Schwarz bounds `Q_ij = sqrt(max_ab |(ab|ab)|)` for every canonical shell pair
/// (S6).
///
/// # Why this belongs in the library
///
/// The Schwarz table is what turns an `O(nbas^4)` quartet list into the fraction
/// worth evaluating — 91% kept on SO2/def2-TZVP, so 9% of the work removed for
/// free. `cintx-driver` has always been able to *use* one, through a
/// `DiagonalEvaluator` the caller supplies, and the throughput benchmark
/// supplies vendored libcint.
///
/// A production caller has no vendored libcint. Without this they would build
/// the table one `(ij|ij)` at a time through the per-tuple path, which is the
/// 194x-slower shape the whole batch surface exists to replace — and the list
/// they then screened would not be the list the benchmark measured.
///
/// One batched dispatch over the diagonal quartets, then the same
/// block-maximum rule `build_schwarz_table` applies.
///
/// Returns `q[i * nbas + j]` for the full square, symmetric by construction
/// since `(ij|ij)` is; callers indexing a triangle can read either half.
///
/// # Errors
/// As [`evaluate_2e_quartet_batch_resident`].
pub fn schwarz_bounds(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    options: TwoEBatchOptions,
) -> Result<Vec<f64>, cintxRsError> {
    let nbas = resident.shells().len();
    let mut pairs = Vec::with_capacity(nbas * (nbas + 1) / 2);
    for i in 0..nbas as u32 {
        for j in 0..=i {
            pairs.push([i, j, i, j]);
        }
    }

    let shells = resident.shells();
    let mut bounds = vec![0.0_f64; nbas * nbas];
    let mut cursor = 0_usize;
    // Streamed rather than materialized: the diagonal list is `nbas^2 / 2`
    // quartets and its blocks are `(n_i n_j)^2`, which for a def2-TZVP sulfur
    // d-shell pair is already 1 225 elements. Reducing each block to one number
    // as it arrives keeps the table's cost proportional to the table, not to the
    // integrals it was derived from.
    stream_2e_quartet_batch(backend, resident, &pairs, options, &mut |chunk| {
        for n in 0..chunk.quartets {
            let [i, j, _, _] = pairs[chunk.first_quartet + n];
            let block = &chunk.values[chunk.offsets[n]..chunk.offsets[n + 1]];
            let ni = nsph(shells[i as usize].l) * shells[i as usize].nctr as usize;
            let nj = nsph(shells[j as usize].l) * shells[j as usize].nctr as usize;
            let side = ni * nj;
            // `Q_ij = sqrt(max_a |(aa|aa)|)` over the block's own diagonal — the
            // block maximum rather than a norm, so the per-quartet bound stays
            // valid element-wise. Verbatim `cintx_driver::build_schwarz_table`.
            let mut peak = 0.0_f64;
            for a in 0..side {
                let diagonal = block[a * side + a].abs();
                if diagonal > peak {
                    peak = diagonal;
                }
            }
            bounds[i as usize * nbas + j as usize] = peak.sqrt();
            bounds[j as usize * nbas + i as usize] = peak.sqrt();
        }
        cursor += chunk.quartets;
        Ok(())
    })?;
    debug_assert_eq!(cursor, pairs.len());
    Ok(bounds)
}

/// Evaluate a work list one chunk at a time, without materializing it (M2).
///
/// `on_chunk` sees each chunk's spherical blocks as they are produced, in the
/// caller's quartet order, and the buffer behind them is reused for the next
/// chunk. So the host holds one chunk of output rather than the whole list — the
/// difference between a work list that fits in memory and one that does not.
///
/// A 30-atom def2-TZVP system has a dense ERI tensor measured in terabytes. No
/// caller wants it materialized; a direct-SCF Fock build wants each block once,
/// contracted into a matrix and discarded. This is that shape.
///
/// [`TwoEBatchOptions::memory_limit_bytes`] sizes the chunks. Without it the
/// whole list is one chunk, which is correct but defeats the purpose — so a
/// streaming caller almost always sets one.
///
/// # Errors
/// Propagates a residency mismatch, a shell index out of range, a class above
/// the device Rys ceiling, a budget that cannot be met, and any error
/// `on_chunk` returns.
pub fn stream_2e_quartet_batch(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    options: TwoEBatchOptions,
    on_chunk: &mut dyn FnMut(ChunkView<'_>) -> Result<(), cintxRsError>,
) -> Result<BatchExecutionStats, cintxRsError> {
    // Validated up front so a bad work list is refused before the first chunk
    // reaches the consumer — a stream that fails halfway has already handed out
    // values the caller may have acted on. `plan_2e_stream` extends the same
    // guarantee to the memory budget: every chunk is grouped and checked before
    // this call ever invokes `on_chunk` (M1).
    batch_output_layout(backend, resident, quartets)?;
    let plan = plan_2e_stream(
        backend,
        resident,
        quartets,
        &options,
        StreamFootprint::Streaming,
    )?;
    let mut sink = StreamingSink {
        scratch: Vec::new(),
        on_chunk,
        peak_bytes: 0,
    };
    let mut stats = run_2e_stream(backend, resident, quartets, options, &mut sink, &plan)?;
    // What the stream actually held, rather than what a materialized run would
    // have: this is the number that makes streaming worth doing.
    stats.host_output_bytes = sink.peak_bytes;
    Ok(stats)
}

/// [`evaluate_2e_quartet_batch_resident`] into caller-owned storage (M2).
///
/// Identical arithmetic and layout to the allocating form; the difference is
/// that `values` is the caller's, so a repeated Fock build reuses one buffer
/// instead of allocating a fresh one per iteration.
///
/// # Errors
/// As [`stream_2e_quartet_batch`], plus [`cintxRsError::BufferTooSmall`] when
/// `values` cannot hold the work list's output.
pub fn evaluate_2e_quartet_batch_into(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    options: TwoEBatchOptions,
    values: &mut [f64],
) -> Result<(Vec<usize>, BatchExecutionStats), cintxRsError> {
    let (offsets, total) = batch_output_layout(backend, resident, quartets)?;
    if values.len() < total {
        return Err(cintxRsError::BufferTooSmall {
            required: total,
            provided: values.len(),
        });
    }
    // Every chunk is grouped and checked against the budget here, before the
    // first byte of the caller's buffer is written, so a refusal never leaves
    // it partially overwritten (M1).
    let plan = plan_2e_stream(
        backend,
        resident,
        quartets,
        &options,
        StreamFootprint::Whole,
    )?;
    let stats = {
        let mut sink = WholeListSink {
            values: &mut values[..total],
            offsets: &offsets,
            total,
        };
        run_2e_stream(backend, resident, quartets, options, &mut sink, &plan)?
    };
    Ok((offsets, stats))
}

/// [`evaluate_2e_quartet_batch_resident`] with explicit [`TwoEBatchOptions`].
pub fn evaluate_2e_quartet_batch_with(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    options: TwoEBatchOptions,
) -> Result<TwoEBatchOutput, cintxRsError> {
    evaluate_2e_batch_inner(backend, resident, quartets, options)
}

/// [`evaluate_2e_quartet_batch`] against a basis already on the device.
///
/// Task 34-C. Identical results; the difference is that the basis upload is the
/// caller's [`ResidentTwoEBasis`] rather than a throwaway one, so
/// [`BatchExecutionStats::transfer_bytes`] covers only this call's quartet
/// tables and [`BatchExecutionStats::basis_upload_bytes`] is zero.
pub fn evaluate_2e_quartet_batch_resident(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
) -> Result<TwoEBatchOutput, cintxRsError> {
    evaluate_2e_batch_inner(backend, resident, quartets, TwoEBatchOptions::default())
}

/// Where one angular-momentum class landed after launch-group merging.
///
/// The device dispatch is per [`TwoELaunchGroup`], but the host cart-to-sph
/// transform is per l-class, so each class records which group buffer holds its
/// Cartesian blocks and at what offsets.
struct TwoEClassPlacement {
    params: TwoEClassParams,
    /// Index into the group list — which dispatch's buffer holds these blocks.
    group: usize,
    /// Cartesian elements per contraction block for this class.
    cart_block: usize,
}

/// Where one quartet's Cartesian block landed (M1).
///
/// Per quartet rather than per class, because a class's members no longer share
/// a group: under a memory limit a signature's quartets are split across as many
/// dispatches as the budget requires, and a quartet's block is in whichever of
/// them took it.
#[derive(Clone, Copy, Debug)]
struct QuartetPlacement {
    /// Dispatch group holding this quartet's Cartesian block.
    group: usize,
    /// Index into `classes` for the shape of the transform.
    class: usize,
    /// Offset of the block within that group's Cartesian buffer.
    cart_offset: usize,
}

fn evaluate_2e_batch_inner(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    options: TwoEBatchOptions,
) -> Result<TwoEBatchOutput, cintxRsError> {
    // The plan validates every chunk's budget before anything is allocated: a
    // refusal here happens strictly before `total`'s output buffer exists,
    // rather than after it (M1).
    let plan = plan_2e_stream(
        backend,
        resident,
        quartets,
        &options,
        StreamFootprint::Whole,
    )?;
    let total = plan.total;
    let mut values = vec![0.0_f64; total];
    let stats = {
        let mut sink = WholeListSink {
            values: &mut values,
            offsets: &plan.offsets,
            total,
        };
        run_2e_stream(backend, resident, quartets, options, &mut sink, &plan)?
    };
    Ok(TwoEBatchOutput {
        values,
        offsets: plan.offsets,
        stats,
    })
}

/// Validate a work list and compute where each quartet's block will land.
///
/// Split out so the whole-list and streaming entry points agree on the layout by
/// construction rather than by both getting it right (M2).
///
/// # Errors
/// Rejects a shell index outside the residency's basis, and a residency built
/// for another backend.
fn batch_output_layout(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
) -> Result<(Vec<usize>, usize), cintxRsError> {
    resident.check(backend)?;
    let shells = resident.shells();
    let mut offsets = Vec::with_capacity(quartets.len());
    let mut total = 0_usize;
    for &quartet in quartets {
        for &s in &quartet {
            if s as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("2e-batch:shell-index-out-of-range:{s}"),
                });
            }
        }
        offsets.push(total);
        total += batch_sph_len(shells, quartet);
    }
    Ok((offsets, total))
}

/// How much of a run's output a sink holds live at once, for the pre-flight
/// memory plan (M1).
///
/// A whole-list sink's entire buffer is resident for the whole call, so its
/// footprint is the work list's `total`. A streaming sink only ever holds one
/// chunk at a time in its reusable scratch (M2), so its footprint is the
/// largest chunk the run will produce, not `total` — charging it the whole
/// list's output would refuse budgets sized for exactly what streaming exists
/// to bound.
#[derive(Clone, Copy)]
enum StreamFootprint {
    Whole,
    Streaming,
}

/// One chunk's dispatch grouping, computed once by [`plan_2e_stream`] and
/// reused by [`run_2e_stream`] so a chunk is never grouped twice.
struct TwoEChunkPlan {
    range: std::ops::Range<usize>,
    groups: Vec<TwoELaunchGroup>,
    classes: Vec<TwoEClassPlacement>,
    placement: Vec<QuartetPlacement>,
    row_owner: Vec<Vec<u32>>,
}

/// A validated plan for one chunked 2e batch run (M1, M2).
///
/// Building this plan is where a bad budget is refused: every chunk is
/// grouped and checked against `memory_limit_bytes` right here, before the
/// caller allocates an output buffer, writes to one, or receives a streamed
/// chunk — so a refusal is always clean, never a statement about a run that
/// has already started.
struct TwoEStreamPlan {
    lens: Vec<usize>,
    offsets: Vec<usize>,
    total: usize,
    device_transform: bool,
    chunks: Vec<TwoEChunkPlan>,
}

/// Validate a work list and its budget, chunk by chunk, before any of it runs.
///
/// The shared pre-flight of the whole-list and streaming entry points (M1,
/// M2): every chunk is grouped and checked here, in one pass, before
/// [`run_2e_stream`] dispatches or writes anything. That ordering is what
/// makes a refusal clean regardless of which chunk fails — the caller never
/// sees a partially written buffer or a stream that stopped partway with no
/// warning it was going to.
///
/// # Errors
/// Propagates a residency mismatch, a shell index out of range, a class above
/// the device Rys ceiling, and a budget that cannot be met by any chunk.
fn plan_2e_stream(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    options: &TwoEBatchOptions,
    footprint: StreamFootprint,
) -> Result<TwoEStreamPlan, cintxRsError> {
    let (offsets, total) = batch_output_layout(backend, resident, quartets)?;
    // M3 needs the residency to carry the `c2s` tables; a residency built before
    // the mode was switched on does not, and falls back to the host transform
    // rather than failing — the two produce the same values.
    let device_transform = crate::kernels::c2s_device::device_transform_enabled();
    if quartets.is_empty() {
        return Ok(TwoEStreamPlan {
            lens: Vec::new(),
            offsets,
            total,
            device_transform,
            chunks: Vec::new(),
        });
    }
    let shells = resident.shells();
    let lens: Vec<usize> = (0..quartets.len())
        .map(|index| offsets.get(index + 1).copied().unwrap_or(total) - offsets[index])
        .collect();

    let ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int2e,
    );
    // F1 (§15): one grouping decision for the whole plan, taken here so the
    // pre-flight budget below and the dispatch cannot disagree about it.
    let fuse = two_e_nroots_fusion();
    // B1 (§22): the shared-memory tier is part of the signature too, decided
    // once here for the same reason.
    let shared_limit = shared_tier_limit(backend);

    // ── M1: evaluate in chunks of consecutive quartets ────────────────────
    //
    // A chunk is a *range of the caller's work list*, not a subset of one
    // launch class, and that choice is load-bearing twice over.
    //
    // **Memory.** Only the chunk's Cartesian buffers are live at once, so the
    // host holds the spherical output plus one chunk's Cartesian intermediate
    // instead of the whole list's. On SO2/def2-TZVP that is 96.6 + 32 MiB
    // against 96.6 + 172.8.
    //
    // **Locality.** Consecutive quartets own a *contiguous* span of the output,
    // so each chunk's transform streams through its own span exactly once. An
    // earlier arrangement released memory per launch group instead, which
    // scattered every group's writes across the whole 96 MiB output and cost
    // 3.8x in the transform — the memory was won and the time given straight
    // back. Chunking by range wins both.
    // M1's Cartesian budget, and what happens when the first guess does not fit.
    //
    // `chunk_cart_budget` is a heuristic — half of what the limit leaves after
    // the caller's output — and a chunk it sizes can still plan over the limit,
    // now that the ket-pair split's partial blocks are charged to the same
    // ledger (§18). The answer is to *chunk harder*, not to refuse: halve the
    // budget and re-plan, down to `MIN_CHUNK_CART_BYTES`, which is one widest
    // Cartesian block and so the point past which chunking cannot help. Only
    // then is the request genuinely impossible, and the refusal carries the
    // numbers from the tightest arrangement tried rather than the first.
    //
    // The split itself is *not* narrowed to fit, because it is a pure function
    // of the quartet (§18): a budgeted run and an unbudgeted one must compute
    // the same values, and only the chunking may differ between them.
    let first_budget = options
        .memory_limit_bytes
        .map_or_else(default_chunk_cart_bytes, |limit| {
            chunk_cart_budget(limit, total * std::mem::size_of::<f64>())
        });
    let mut cart_budget = first_budget;
    let mut attempts = 0usize;
    let chunks = loop {
        attempts += 1;
        match plan_2e_chunks(
            backend,
            resident,
            quartets,
            shells,
            &lens,
            total,
            footprint,
            device_transform,
            ceiling,
            fuse,
            shared_limit,
            cart_budget,
            options.memory_limit_bytes,
        )? {
            Ok(chunks) => {
                if std::env::var("CINTX_2E_GROUPS").is_ok() {
                    eprintln!(
                        "  plan: {attempts} attempt(s), cart budget {cart_budget}                          (first {first_budget}), {} chunk(s)",
                        chunks.len()
                    );
                }
                break chunks;
            }
            Err(over) => {
                if cart_budget <= MIN_CHUNK_CART_BYTES {
                    return Err(over);
                }
                cart_budget = (cart_budget / 2).max(MIN_CHUNK_CART_BYTES);
            }
        }
    };

    Ok(TwoEStreamPlan {
        lens,
        offsets,
        total,
        device_transform,
        chunks,
    })
}

/// Plan one arrangement of chunks at `cart_budget` (M1).
///
/// The outer `Result` is a hard refusal — a class above the device Rys ceiling,
/// which no budget can fix. The inner one is "this arrangement does not fit":
/// the caller may retry at a smaller budget, and the error it carries is what
/// it would report if it does not.
#[allow(clippy::too_many_arguments)]
fn plan_2e_chunks(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    shells: &[BatchShell],
    lens: &[usize],
    total: usize,
    footprint: StreamFootprint,
    device_transform: bool,
    ceiling: usize,
    fuse: bool,
    shared_limit: Option<usize>,
    cart_budget: usize,
    limit: Option<usize>,
) -> Result<Result<Vec<TwoEChunkPlan>, cintxRsError>, cintxRsError> {
    let ranges = plan_quartet_chunks(quartets, shells, cart_budget);

    // What the sink will actually hold live across the run: the whole list
    // for a whole-list sink, or the largest single chunk for a streaming one
    // that reuses one scratch buffer (M2).
    let planned_output_len = match footprint {
        StreamFootprint::Whole => total,
        StreamFootprint::Streaming => ranges
            .iter()
            .map(|range| lens[range.clone()].iter().sum::<usize>())
            .max()
            .unwrap_or(0),
    };

    let mut chunks = Vec::with_capacity(ranges.len());
    for range in ranges {
        let sub = &quartets[range.clone()];
        let (groups, classes, placement, row_owner) =
            build_launch_groups(sub, shells, &resident.pairs, ceiling, fuse, shared_limit)?;

        // `CINTX_2E_GROUPS=1` prints the grouping this chunk will dispatch:
        // per signature, the quartet count, the merged class count, the widest
        // block and G tensor, the summed `quartet_cost_estimate` and the
        // costliest single quartet. It is the attribution F1 (§15) was chosen
        // from: `Σ_groups max(cost / units, max quartet cost)` is the floor the
        // per-unit partition can reach, and comparing it against
        // `max(Σ cost / units, max quartet cost)` says what fusing the
        // dispatches is worth *before* the kernel is touched. It costs nothing
        // when the variable is unset and it is the only way to see the
        // imbalance from outside — `launches=` counts dispatches but not what
        // is in them.
        if std::env::var("CINTX_2E_GROUPS").is_ok() {
            let cost_total: u64 = groups.iter().flat_map(|g| g.quartet_cost.iter()).sum();
            eprintln!("  chunk: {} groups, total cost {cost_total}", groups.len());
            for g in &groups {
                let cost: u64 = g.quartet_cost.iter().sum();
                eprintln!(
                    "    sig(ibase={},kbase={},nroots={},tier={}) quartets={} classes={} \
                     block={} g_size={} nr_max={} cost={} ({:.1}%) maxq={} split={}",
                    g.signature.ibase,
                    g.signature.kbase,
                    g.signature.nroots,
                    g.signature.tier,
                    g.len(),
                    g.class_count(),
                    g.max_block_len,
                    g.max_g_size,
                    g.max_nroots,
                    cost,
                    100.0 * cost as f64 / cost_total.max(1) as f64,
                    g.quartet_cost.iter().max().copied().unwrap_or(0),
                    kl_split_plan(g).iter().max().copied().unwrap_or(1),
                );
            }
        }

        // The pre-flight plan (M1): every term comes from the expression the
        // dispatch allocates from, so a refusal here is a statement about this
        // chunk's real footprint. Every chunk is checked in this loop before
        // `run_2e_stream` dispatches or writes any of them, so a refusal here
        // leaves nothing partially written or streamed, regardless of which
        // chunk in the list it was.
        if let Some(limit) = limit {
            // M4.3: the pre-flight budget must charge the same c2s scratch
            // buffer `run_2e_batches` allocates when the device transform is
            // on, or a chunk near the caller's limit can be approved here and
            // still exceed it at dispatch time.
            let c2s_scratch_bytes = if device_transform {
                plan_c2s_scratch_bytes(backend, &groups)
            } else {
                0
            };
            let plan = plan_batch_bytes(
                &groups,
                planned_output_len,
                c2s_scratch_bytes,
                device_transform,
            );
            if plan.peak_bytes > limit {
                return Ok(Err(cintxRsError::MemoryLimitExceeded {
                    requested: plan.peak_bytes,
                    limit,
                }));
            }
        }

        chunks.push(TwoEChunkPlan {
            range,
            groups,
            classes,
            placement,
            row_owner,
        });
    }
    Ok(Ok(chunks))
}

/// Execute a plan built by [`plan_2e_stream`], writing each chunk into `sink`.
///
/// The shared core of the whole-list and streaming entry points (M1, M2).
/// Every chunk's grouping and budget were already validated when `plan` was
/// built, so nothing here can refuse partway through a run.
///
/// # Errors
/// Propagates any refusal from the sink.
fn run_2e_stream(
    backend: &ResolvedBackend,
    resident: &ResidentTwoEBasis,
    quartets: &[[u32; 4]],
    options: TwoEBatchOptions,
    sink: &mut dyn ChunkSink,
    plan: &TwoEStreamPlan,
) -> Result<BatchExecutionStats, cintxRsError> {
    let shells = resident.shells();
    let mut stats = BatchExecutionStats {
        quartets: quartets.len(),
        ..BatchExecutionStats::default()
    };
    if quartets.is_empty() {
        return Ok(stats);
    }
    let device_transform = plan.device_transform;
    let lens = &plan.lens;
    let offsets = &plan.offsets;
    let total = plan.total;

    let mut transform_ns = 0_u64;
    let mut dispatch_ns = 0_u64;
    let mut host_cart_peak = 0_usize;
    let mut readback_bytes = 0_usize;
    let mut device_memory = crate::memory_probe::DeviceMemoryProbe::new();
    let mut profile_totals = crate::transform::profile::HostTransformProfile::new();
    let mut launch_count = 0_usize;
    let mut launch_classes = 0_usize;
    let mut chunk_offsets: Vec<usize> = Vec::new();
    let mut table_bytes = 0_usize;
    let mut max_g_slab_bytes = 0_usize;

    for chunk_plan in &plan.chunks {
        let chunk = chunk_plan.range.clone();
        let sub = &quartets[chunk.clone()];
        let groups = &chunk_plan.groups;
        let classes = &chunk_plan.classes;
        let placement = &chunk_plan.placement;
        let row_owner = &chunk_plan.row_owner;

        launch_count += groups.len();
        launch_classes += classes.len();
        table_bytes += groups
            .iter()
            .map(TwoELaunchGroup::upload_bytes)
            .sum::<usize>();
        max_g_slab_bytes = max_g_slab_bytes.max(
            groups
                .iter()
                .map(TwoELaunchGroup::g_slab_bytes)
                .max()
                .unwrap_or(0),
        );

        // The chunk's output span is contiguous, because its quartets are — so
        // a whole-list sink hands back a slice of the caller's buffer and a
        // streaming one hands back its reusable scratch, and neither transform
        // below can tell which it got (M2).
        let chunk_lens = &lens[chunk.clone()];
        let chunk_len: usize = chunk_lens.iter().sum();
        let chunk_base = offsets[chunk.start];

        if let Some(tables) = resident.c2s_handles.as_ref().filter(|_| device_transform) {
            // ── M3: the device transforms; the host only reads the result ──
            let sph_offsets: Vec<Vec<u32>> = row_owner
                .iter()
                .map(|rows| {
                    rows.iter()
                        .map(|&caller| (offsets[chunk.start + caller as usize] - chunk_base) as u32)
                        .collect()
                })
                .collect();
            // The transform's own uploads are transfers like any other: one
            // destination index per quartet, plus the residency's `c2s` tables
            // counted once in `basis_upload_bytes`.
            table_bytes += sph_offsets
                .iter()
                .map(|rows| rows.len() * std::mem::size_of::<u32>())
                .sum::<usize>();
            let dispatch_start = std::time::Instant::now();
            let (chunk_memory, sph) = dispatch_2e_batches_device_c2s(
                backend,
                &resident.handles,
                &resident.pair_handles,
                groups,
                options,
                tables,
                &sph_offsets,
                chunk_len,
            )?;
            dispatch_ns += dispatch_start.elapsed().as_nanos() as u64;
            device_memory.merge(&chunk_memory);
            // Nothing Cartesian was ever on the host.
            readback_bytes += sph.len() * std::mem::size_of::<f64>();
            sink.borrow(chunk.start, sub.len(), chunk_len)
                .copy_from_slice(&sph);
        } else {
            let mut carts: Vec<Vec<f64>> = vec![Vec::new(); groups.len()];
            let dispatch_start = std::time::Instant::now();
            let chunk_memory = dispatch_2e_batches(
                backend,
                &resident.handles,
                &resident.pair_handles,
                groups,
                options,
                &mut |group_index: usize, cart: Vec<f64>| carts[group_index] = cart,
                None,
            )?;
            dispatch_ns += dispatch_start.elapsed().as_nanos() as u64;
            device_memory.merge(&chunk_memory);
            let chunk_cart_bytes =
                carts.iter().map(Vec::len).sum::<usize>() * std::mem::size_of::<f64>();
            host_cart_peak = host_cart_peak.max(chunk_cart_bytes);
            readback_bytes += chunk_cart_bytes;

            let transform_start = std::time::Instant::now();
            let jobs: Vec<(usize, &mut [f64])> = crate::transform::host_batch::split_output_blocks(
                sink.borrow(chunk.start, sub.len(), chunk_len),
                chunk_lens,
            )
            .into_iter()
            .enumerate()
            .collect();
            transform_chunk(
                jobs,
                &carts,
                placement,
                classes,
                sub,
                shells,
                &mut profile_totals,
            );
            transform_ns += transform_start.elapsed().as_nanos() as u64;
        }

        // Chunk-relative offsets, with a trailing total so a consumer can bound
        // the last quartet without knowing the shell shapes.
        chunk_offsets.clear();
        let mut running = 0_usize;
        for &len in chunk_lens {
            chunk_offsets.push(running);
            running += len;
        }
        chunk_offsets.push(running);
        sink.commit(chunk.start, sub.len(), &chunk_offsets)?;
        // `carts` is dropped here: the next chunk allocates into the space it
        // just freed.
    }

    stats.dispatch_ns = dispatch_ns;
    stats.host_transform_ns = transform_ns;
    stats.chunk_count = plan.chunks.len();
    profile_totals.store_into(&mut stats);

    // The basis was uploaded when the residency was built. Count it against the
    // *first* evaluation only, so a repeated Fock build shows the quartet tables
    // alone and the amortization is visible rather than asserted.
    let first_use = resident.take_first_use();
    stats.basis_upload_bytes = if first_use { resident.upload_bytes } else { 0 };
    stats.kernel_launch_count = launch_count;
    stats.readback_count = launch_count;
    stats.launch_classes = launch_classes;
    stats.max_g_slab_bytes = max_g_slab_bytes;
    stats.transfer_bytes = stats.basis_upload_bytes + table_bytes;

    // ── Memory accounting (M6) ────────────────────────────────────────────
    // `host_cart_bytes_peak` is a maximum over *chunks*, not a sum over the
    // whole list: each chunk's Cartesian buffers were dropped before the next
    // chunk allocated (M1).
    device_memory.store_into(&mut stats);
    stats.host_output_bytes = total * std::mem::size_of::<f64>();
    stats.host_cart_bytes_peak = host_cart_peak;
    stats.readback_bytes = readback_bytes;
    stats.primitive_quartets_total = primitive_quartets_in(shells, quartets);
    // What the kernel was actually asked for, after S1's pair-level screen. The
    // quartet-level test (`cceij > expcutoff - ccekl`) removes more still, but
    // it is a function of two rows rather than one and is applied in the kernel,
    // so this is the exact dispatched count and an upper bound on the evaluated
    // one. Counting it on the host keeps an atomic out of the inner loop.
    stats.primitive_quartets_evaluated = resident.pairs.primitive_quartets_in(quartets);

    Ok(stats)
}

/// Ceiling on one chunk's Cartesian intermediate when no budget is set.
///
/// Unlimited, and that is a measured decision rather than an omission.
///
/// Chunking is not free. On SO2/def2-TZVP, in one process against the same work
/// list (`CINTX_2E_CHUNK_MIB`, best of 9):
///
/// | chunk ceiling | dispatches | dispatch | transform | Cartesian peak |
/// |---|---|---|---|---|
/// | unlimited | 24 | 194.7 ms | 29.3 ms | 172.8 MiB |
/// | 32 MiB | 115 | 225.6 ms | 44.8 ms | 32 MiB |
/// | 8 MiB | 337 | 231.7 ms | 77.7 ms | 8 MiB |
///
/// A 32 MiB ceiling costs 21% of wall time to save 140 MiB. That is a good
/// trade for a caller who needs the memory and a bad one for a caller who does
/// not, and only the caller knows which they are — so the default keeps the
/// speed and `memory_limit_bytes` buys the bound.
pub const DEFAULT_CHUNK_CART_BYTES: usize = usize::MAX;

/// The default chunk ceiling, with `CINTX_2E_CHUNK_MIB` applied.
///
/// A measurement switch, like `CINTX_2E_PER_UNIT` and `CINTX_2E_CUBE_DIM`
/// beside it: `0` means unlimited (one chunk, whatever the list's size), any
/// other value is a mebibyte ceiling. Unset uses [`DEFAULT_CHUNK_CART_BYTES`].
/// It exists so the memory/speed trade the default embodies can be measured
/// rather than asserted.
fn default_chunk_cart_bytes() -> usize {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    let pinned = *OVERRIDE.get_or_init(|| {
        std::env::var("CINTX_2E_CHUNK_MIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    });
    match pinned {
        Some(0) => usize::MAX,
        Some(mib) => mib * 1024 * 1024,
        None => DEFAULT_CHUNK_CART_BYTES,
    }
}

/// Largest Cartesian intermediate one chunk may hold, under an explicit budget.
///
/// The caller's spherical output is live for the whole call and is not the
/// dispatcher's to shrink, so it is subtracted first: what can be chunked is the
/// *headroom* above it. That headroom holds the chunk's Cartesian buffers on the
/// host and their twins on the device, hence the half; the tables and scratch
/// come out of the remainder, which is why the pre-flight plan still has the
/// final say.
fn chunk_cart_budget(limit_bytes: usize, output_bytes: usize) -> usize {
    (limit_bytes.saturating_sub(output_bytes) / 2).max(MIN_CHUNK_CART_BYTES)
}

/// Floor on a chunk's Cartesian budget.
///
/// One `(ff|ff)` contraction block is 10 000 Cartesian elements and has to be
/// placeable, or chunk planning would emit an empty chunk and make no progress.
const MIN_CHUNK_CART_BYTES: usize = 10_000 * std::mem::size_of::<f64>();

/// Split a work list into runs of consecutive quartets, each within `budget`.
///
/// Consecutive is the whole point: a run of adjacent quartets owns a contiguous
/// span of the caller's output, so its transform streams through that span once
/// rather than scattering writes across the whole buffer.
///
/// A single quartet larger than the budget gets its own chunk rather than being
/// refused here — the pre-flight plan is where a genuinely impossible request is
/// turned away, with the numbers to say why.
fn plan_quartet_chunks(
    quartets: &[[u32; 4]],
    shells: &[BatchShell],
    budget: usize,
) -> Vec<std::ops::Range<usize>> {
    plan_quartet_chunks_capped(quartets, shells, budget, chunk_quartet_cap())
}

/// Ceiling on quartets per chunk, from `CINTX_2E_CHUNK_QUARTETS`; `None`
/// (unset or `0`) leaves chunking to the byte budget alone.
///
/// This exists for one reason, and it is not memory. On a GPU that is also the
/// display device, a dispatch is a compute job the compositor's frame work
/// queues behind; a dispatch that runs longer than the driver's job timeout
/// (amdgpu: seconds) resets the GPU and takes the desktop session down with it.
/// A dispatch's length scales with the quartets it carries, so capping them per
/// chunk — and so per dispatch — bounds it independently of the byte budget,
/// which on a small generally contracted list never splits at all
/// (`gth_molopt_speed_memory_plan.md` §8.6). The cost is the launch count the
/// def2 plan measured for chunking; the arithmetic is unchanged.
fn chunk_quartet_cap() -> Option<usize> {
    use std::sync::OnceLock;
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CINTX_2E_CHUNK_QUARTETS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
    })
}

/// [`plan_quartet_chunks`] with an explicit quartet cap, so the cap is testable
/// without the environment.
fn plan_quartet_chunks_capped(
    quartets: &[[u32; 4]],
    shells: &[BatchShell],
    budget: usize,
    max_quartets: Option<usize>,
) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0_usize;
    let mut bytes = 0_usize;
    for (index, quartet) in quartets.iter().enumerate() {
        let block = quartet
            .iter()
            .map(|&s| {
                let shell = &shells[s as usize];
                ncart(shell.l) * shell.nctr as usize
            })
            .product::<usize>()
            * std::mem::size_of::<f64>();
        let over_count = max_quartets.is_some_and(|cap| index - start >= cap);
        if index > start && (bytes + block > budget || over_count) {
            chunks.push(start..index);
            start = index;
            bytes = 0;
        }
        bytes += block;
    }
    if start < quartets.len() {
        chunks.push(start..quartets.len());
    }
    chunks
}

/// What one call's grouping produced.
///
/// The fourth element is the inverse of the third: per group, the caller-order
/// index of each of its quartet rows, in row order. The placement answers "where
/// did quartet `n` go"; this answers "whose is row `r`", which is what the
/// device transform needs to know where to write (M3). Rows are appended class
/// by class, so neither ordering derives from the other without recording it.
type LaunchGrouping = (
    Vec<TwoELaunchGroup>,
    Vec<TwoEClassPlacement>,
    Vec<QuartetPlacement>,
    Vec<Vec<u32>>,
);

/// Group a work list into dispatches, capping each dispatch's Cartesian buffer.
///
/// A pure function of `(quartets, shells, ceiling, chunk_out_bytes)`, which is
/// what lets the caller search for a budget that fits by calling it more than
/// once (M1). `chunk_out_bytes = None` is the unbounded shape: one dispatch per
/// launch signature, exactly as before the memory plan existed.
///
/// # Errors
/// Refuses a class whose Rys order is above the backend's device ceiling, before
/// anything is dispatched.
fn build_launch_groups(
    quartets: &[[u32; 4]],
    shells: &[BatchShell],
    pairs: &crate::kernels::pair_table::PairTable,
    ceiling: usize,
    fuse: bool,
    shared_limit: Option<usize>,
) -> Result<LaunchGrouping, cintxRsError> {
    // Group by launch class, preserving the caller's order within a class.
    let mut grouped: std::collections::BTreeMap<[u8; 4], Vec<usize>> = Default::default();
    for (index, &quartet) in quartets.iter().enumerate() {
        let key = [
            shells[quartet[0] as usize].l,
            shells[quartet[1] as usize].l,
            shells[quartet[2] as usize].l,
            shells[quartet[3] as usize].l,
        ];
        grouped.entry(key).or_default().push(index);
    }

    // Build every class's quartet rows before dispatching anything, so a class
    // above the device Rys ceiling rejects the batch without having launched.
    //
    // Classes are then merged into dispatch **groups** keyed on the kernel's
    // comptime signature (Task 35-M1). The `(li,lj,lk,ll)` grouping survives as
    // the sub-grouping that drives the host cart-to-sph below, because that
    // transform genuinely is per l-class; only the *launch* is merged.
    let mut groups: Vec<TwoELaunchGroup> = Vec::new();
    let mut group_of: std::collections::BTreeMap<TwoELaunchSignature, usize> = Default::default();
    let mut classes: Vec<TwoEClassPlacement> = Vec::with_capacity(grouped.len());
    let mut placement = vec![
        QuartetPlacement {
            group: 0,
            class: 0,
            cart_offset: 0,
        };
        quartets.len()
    ];
    let mut row_owner: Vec<Vec<u32>> = Vec::new();
    for (class, members) in grouped {
        let [li, lj, lk, ll] = class;
        let params = TwoEClassParams::new(li, lj, lk, ll);
        // Per-backend ceiling (task 33-05): the base value everywhere, raised
        // only on a backend whose FMA-fusion probe passed and only with the
        // `extended-device-rys` opt-in. See `crate::device_rys_ceiling`.
        if params.nroots as usize > ceiling {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "2e-batch:nroots={} exceeds device ceiling {ceiling} \
                     for l=({li},{lj},{lk},{ll})",
                    params.nroots
                ),
            });
        }

        let signature = TwoELaunchSignature::of(&params, fuse, shared_limit);
        let cart_block = ncart(li) * ncart(lj) * ncart(lk) * ncart(ll);
        let class_index = classes.len();
        classes.push(TwoEClassPlacement {
            params,
            group: usize::MAX,
            cart_block,
        });

        let group_index = match group_of.get(&signature) {
            Some(&existing) => existing,
            None => {
                groups.push(TwoELaunchGroup::new(signature));
                row_owner.push(Vec::new());
                let fresh = groups.len() - 1;
                group_of.insert(signature, fresh);
                fresh
            }
        };
        let slot_in_group = groups[group_index].push_class(&params);

        for &index in &members {
            let q = quartets[index];
            let nctr = [
                shells[q[0] as usize].nctr,
                shells[q[1] as usize].nctr,
                shells[q[2] as usize].nctr,
                shells[q[3] as usize].nctr,
            ];
            let nctr_product: usize = nctr.iter().map(|&n| n as usize).product();
            let block = nctr_product * cart_block;

            row_owner[group_index].push(index as u32);
            let group = &mut groups[group_index];
            group.max_block_len = group.max_block_len.max(cart_block as u32);
            group.max_ctr_len = group
                .max_ctr_len
                .max(staged_ctr_len(nctr, cart_block) as u32);
            placement[index] = QuartetPlacement {
                group: group_index,
                class: class_index,
                cart_offset: group.out_len,
            };
            let kl_slot = (q[2] * pairs.nbas + q[3]) as usize;
            group.quartets.extend_from_slice(&[
                q[0],
                q[1],
                q[2],
                q[3],
                group.out_len as u32,
                slot_in_group,
                pairs.offset[kl_slot],
                pairs.offset[kl_slot + 1],
            ]);
            let prim =
                u64::from(pairs.pair_count(q[0], q[1])) * u64::from(pairs.pair_count(q[2], q[3]));
            group
                .quartet_cost
                .push(quartet_cost_estimate(prim, &params, nctr[0]));
            group.quad_count.push(nctr_product as u32);
            group.out_len += block;
        }
        classes[class_index].group = group_index;
    }
    Ok((groups, classes, placement, row_owner))
}

/// Where a chunk's spherical blocks are written (M2).
///
/// The chunked evaluation loop does not know whether its caller wants the whole
/// work list in one buffer or one chunk at a time, and it should not have to:
/// both want exactly the same arithmetic written to a `&mut [f64]`. The
/// difference is only *whose* memory that is, and what happens when the chunk is
/// done.
///
/// Both implementations are zero-copy. The whole-list sink hands out a slice of
/// the caller's output and commits nothing; the streaming sink hands out a
/// reusable scratch buffer and commits it to the consumer, then reuses it. That
/// is what lets a work list whose output does not fit in memory be evaluated at
/// all, and it is why this is a trait rather than a flag.
pub trait ChunkSink {
    /// Borrow `len` elements to write chunk `[start, start + count)` into.
    ///
    /// `start` is the caller-order index of the chunk's first quartet and `len`
    /// the total spherical elements the chunk produces.
    fn borrow(&mut self, start: usize, count: usize, len: usize) -> &mut [f64];

    /// The chunk is written. `offsets` locate each quartet within the borrow,
    /// relative to its start.
    ///
    /// # Errors
    /// A sink may refuse — a consumer that fails should stop the evaluation
    /// rather than have its refusal swallowed.
    fn commit(&mut self, start: usize, count: usize, offsets: &[usize])
    -> Result<(), cintxRsError>;
}

/// A sink that writes every chunk into one contiguous output buffer.
struct WholeListSink<'a> {
    values: &'a mut [f64],
    /// Absolute offset of each quartet's block, in caller order.
    offsets: &'a [usize],
    /// Total elements, for the last chunk's extent.
    total: usize,
}

impl ChunkSink for WholeListSink<'_> {
    fn borrow(&mut self, start: usize, count: usize, _len: usize) -> &mut [f64] {
        let from = self.offsets[start];
        let to = self
            .offsets
            .get(start + count)
            .copied()
            .unwrap_or(self.total);
        &mut self.values[from..to]
    }

    fn commit(
        &mut self,
        _start: usize,
        _count: usize,
        _offsets: &[usize],
    ) -> Result<(), cintxRsError> {
        // Already in place: `borrow` handed out the caller's own memory.
        Ok(())
    }
}

/// A sink that hands each chunk to a consumer and reuses one scratch buffer.
struct StreamingSink<'a> {
    scratch: Vec<f64>,
    on_chunk: &'a mut dyn FnMut(ChunkView<'_>) -> Result<(), cintxRsError>,
    /// Largest scratch this sink has held, for the memory ledger.
    peak_bytes: usize,
}

impl ChunkSink for StreamingSink<'_> {
    fn borrow(&mut self, _start: usize, _count: usize, len: usize) -> &mut [f64] {
        // `resize` keeps the allocation across chunks, so a steady-state stream
        // allocates once rather than once per chunk.
        if self.scratch.len() < len {
            self.scratch.resize(len, 0.0);
        }
        self.peak_bytes = self
            .peak_bytes
            .max(self.scratch.capacity() * std::mem::size_of::<f64>());
        &mut self.scratch[..len]
    }

    fn commit(
        &mut self,
        start: usize,
        count: usize,
        offsets: &[usize],
    ) -> Result<(), cintxRsError> {
        let len = offsets
            .last()
            .map_or(0, |_| offsets[offsets.len() - 1])
            .max(0);
        let _ = len;
        (self.on_chunk)(ChunkView {
            first_quartet: start,
            quartets: count,
            offsets,
            values: &self.scratch[..offsets.last().copied().unwrap_or(0)],
        })
    }
}

/// One chunk of a streamed evaluation (M2).
///
/// Borrowed, not owned: the buffer behind it is reused for the next chunk, so a
/// consumer that needs the values past its callback must copy them. That is the
/// whole point — a consumer that contracts them into a Fock matrix does not
/// need to, and so never materializes the work list.
#[derive(Debug)]
pub struct ChunkView<'a> {
    /// Caller-order index of this chunk's first quartet.
    pub first_quartet: usize,
    /// Quartets in this chunk.
    pub quartets: usize,
    /// `offsets[n]` locates quartet `first_quartet + n` within [`Self::values`].
    ///
    /// One entry per quartet plus a final total, so quartet `n` occupies
    /// `offsets[n]..offsets[n + 1]`.
    pub offsets: &'a [usize],
    /// The chunk's spherical AO blocks, concatenated in caller order.
    pub values: &'a [f64],
}

/// What one batched run will hold at its peak (M1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BatchMemoryPlan {
    /// The output the sink holds live across the run: the whole work list's
    /// spherical output for a whole-list sink, or the largest chunk's for a
    /// streaming sink that reuses one scratch buffer ([`StreamFootprint`]).
    host_output_bytes: usize,
    /// Largest single group's Cartesian block, on the host and on the device.
    group_cart_bytes: usize,
    /// Largest single group's ket-pair split partial region (G1, §18).
    ///
    /// Counted **once**, unlike the Cartesian block: it lives on the device
    /// only. The reduce folds it onto part 0 in place and the caller is handed
    /// a handle trimmed to the output, so the partial blocks never reach the
    /// host and never appear in a readback.
    group_split_bytes: usize,
    /// Quartet, shape and factor tables, summed over groups.
    table_bytes: usize,
    /// The shared G-tensor scratch slab.
    scratch_bytes: usize,
    /// The device-side c2s transform's ping-pong scratch, sized to the
    /// widest group exactly as [`run_2e_batches`]'s own `c2s_scratch`
    /// allocation is. `0` when the device transform is not in use.
    c2s_scratch_bytes: usize,
    /// The sum of the above — what the run needs at once.
    peak_bytes: usize,
}

/// Predict a batched run's peak from the same expressions it allocates from.
///
/// Deliberately not an estimate: [`TwoELaunchGroup::output_bytes`] and
/// [`TwoELaunchGroup::g_slab_bytes`] are the exact methods `run_2e_batches`
/// calls to size the buffer and the scratch slab it allocates, and
/// [`TwoELaunchGroup::upload_bytes`] is what it uploads — one set of
/// expressions rather than two, so this plan and the real allocations cannot
/// drift apart independently.
///
/// Under the host transform the Cartesian block is counted twice — once for the
/// device buffer and once for the host `Vec` its readback lands in — because
/// both are live at the moment the readback returns. On a unified-memory backend
/// they are the same physical bytes and this over-counts by one block;
/// over-counting a limit is the safe direction. The device transform (§19) has
/// no host `Vec` at all, and is charged once.
///
/// `output_len` is elements, not bytes, and is the caller's chosen
/// [`StreamFootprint`] applied to this run — the whole list for a whole-list
/// sink, the largest chunk for a streaming one.
///
/// `c2s_scratch_bytes` is the caller's pre-computed
/// [`plan_c2s_scratch_bytes`] result (`0` when the device transform is not in
/// use) — computing it here would need a concrete backend client this
/// function does not have, so the caller supplies it instead of this plan
/// silently omitting a buffer `run_2e_batches` actually allocates.
fn plan_batch_bytes(
    groups: &[TwoELaunchGroup],
    output_len: usize,
    c2s_scratch_bytes: usize,
    device_transform: bool,
) -> BatchMemoryPlan {
    let group_cart_bytes = groups
        .iter()
        .map(TwoELaunchGroup::output_bytes)
        .max()
        .unwrap_or(0);
    // The dispatch's Cartesian buffer is its output *followed by* the partial
    // region its ket-pair split writes into (G1, §18). `kl_split_plan` is a
    // pure function of the group, so this predicts exactly what
    // `run_2e_batches` will ask for — which is what lets the split run under a
    // caller's `memory_limit_bytes` instead of being switched off there and
    // making a budgeted run compute something different from an unbudgeted one.
    let group_split_bytes = groups
        .iter()
        .map(|group| kl_split_extra_len(group, &kl_split_plan(group)) * std::mem::size_of::<f64>())
        .max()
        .unwrap_or(0);
    let table_bytes = groups.iter().map(TwoELaunchGroup::upload_bytes).sum();
    // One slot's G slab plus its contraction slab, at the widest group; the
    // run allocates both once and reuses them across groups (M4.1, C1).
    let scratch_bytes = groups
        .iter()
        .map(TwoELaunchGroup::g_slab_bytes)
        .max()
        .unwrap_or(0)
        + groups
            .iter()
            .map(TwoELaunchGroup::ctr_slab_bytes)
            .max()
            .unwrap_or(0);
    let host_output_bytes = output_len * std::mem::size_of::<f64>();
    BatchMemoryPlan {
        host_output_bytes,
        group_cart_bytes,
        group_split_bytes,
        table_bytes,
        scratch_bytes,
        c2s_scratch_bytes,
        peak_bytes: host_output_bytes
            // The Cartesian block is live once on the device and, under the
            // host transform, again in the `Vec` its readback lands in. The
            // device transform (§19) has no such `Vec` — the block is consumed
            // where it was written and only spherical comes back — so charging
            // it twice there would refuse budgets the run can honour.
            + group_cart_bytes * if device_transform { 1 } else { 2 }
            + group_split_bytes
            + table_bytes
            + scratch_bytes
            + c2s_scratch_bytes,
    }
}

/// Predict the device c2s transform's scratch buffer for the widest group in
/// `groups`, matching [`run_2e_batches`]'s own `c2s_scratch` allocation
/// (M4.3) via the same [`c2s_scratch_widest_len`], so this plan and the real
/// allocation cannot drift apart independently. Only meaningful — and only
/// called by [`plan_2e_stream`] — when the device transform is enabled; the
/// host transform allocates no such buffer.
fn plan_c2s_scratch_bytes(backend: &ResolvedBackend, groups: &[TwoELaunchGroup]) -> usize {
    let len = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            c2s_scratch_widest_len::<cubecl::cpu::CpuRuntime>(client, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            c2s_scratch_widest_len::<cubecl_wgpu::WgpuRuntime>(client, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            c2s_scratch_widest_len::<cubecl_cuda::CudaRuntime>(client, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            c2s_scratch_widest_len::<cubecl_hip::HipRuntime>(client, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            c2s_scratch_widest_len::<cubecl_wgpu::WgpuRuntime>(client, groups)
        }
    };
    len.max(1) * std::mem::size_of::<f64>()
}

/// Transform one dispatch group's Cartesian buffer into the caller's blocks.
///
/// Split out of [`evaluate_2e_batch_inner`] so it can run inside the dispatch
/// loop, against a buffer that is dropped as soon as it returns (M1). The
/// arithmetic is unchanged from the whole-list loop it replaces: each job owns a
/// disjoint `&mut [f64]`, each output element is written by exactly one quartet,
/// and no summation is reordered — so the result is bit-identical to the
/// serial, whole-list form by construction rather than by tolerance.
fn transform_chunk(
    jobs: Vec<(usize, &mut [f64])>,
    carts: &[Vec<f64>],
    placement: &[QuartetPlacement],
    classes: &[TwoEClassPlacement],
    quartets: &[[u32; 4]],
    shells: &[BatchShell],
    profile: &mut crate::transform::profile::HostTransformProfile,
) {
    if jobs.is_empty() {
        return;
    }
    let states = crate::transform::host_batch::for_each_block(
        jobs,
        || {
            (
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                crate::transform::profile::HostTransformProfile::new(),
            )
        },
        |(sph_block, sph_scratch, worker), (index, block)| {
            let spot = placement[index];
            let class = &classes[spot.class];
            let cart = &carts[spot.group];
            let (li, lj, lk, ll) = (
                class.params.li as u8,
                class.params.lj as u8,
                class.params.lk as u8,
                class.params.ll as u8,
            );
            let cart_block = class.cart_block;
            let (nsi, nsj, nsk, nsl) = (nsph(li), nsph(lj), nsph(lk), nsph(ll));

            worker.start();
            sph_block.clear();
            sph_block.resize(nsi * nsj * nsk * nsl, 0.0);
            worker.charge_alloc();

            let q = quartets[index];
            let (nci, ncj, nck, ncl) = (
                shells[q[0] as usize].nctr as usize,
                shells[q[1] as usize].nctr as usize,
                shells[q[2] as usize].nctr as usize,
                shells[q[3] as usize].nctr as usize,
            );
            let (di, dj, dk) = (nci * nsi, ncj * nsj, nck * nsk);
            let src_base = spot.cart_offset;
            for ci in 0..nci {
                for cj in 0..ncj {
                    for ck in 0..nck {
                        for cl in 0..ncl {
                            let base =
                                src_base + (((ci * ncj + cj) * nck + ck) * ncl + cl) * cart_block;
                            crate::transform::c2s::cart_to_sph_2e_into(
                                &cart[base..base + cart_block],
                                li,
                                lj,
                                lk,
                                ll,
                                sph_block,
                                sph_scratch,
                            );
                            worker.charge_transform();
                            let sph = &sph_block[..];
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
                                            block[dst] = sph[src];
                                        }
                                    }
                                }
                            }
                            worker.charge_scatter();
                        }
                    }
                }
            }
            worker.pause();
        },
    );
    for (_, _, worker) in &states {
        profile.merge(worker);
    }
}

/// The four device buffers a flattened basis occupies.
///
/// Held apart from [`TwoEFlatBasis`] because the host-side arrays are needed
/// only to *build* the upload, while the handles are what a
/// [`ResidentTwoEBasis`] keeps alive across calls.
#[derive(Clone, Debug)]
pub(crate) struct TwoEBasisHandles {
    pub(crate) exps: cubecl::server::Handle,
    pub(crate) coeffs: cubecl::server::Handle,
    pub(crate) centers: cubecl::server::Handle,
    pub(crate) shell_meta: cubecl::server::Handle,
    pub(crate) exps_len: usize,
    pub(crate) coeffs_len: usize,
    pub(crate) centers_len: usize,
    pub(crate) shell_meta_len: usize,
}

/// The primitive-pair table on the device (S1).
///
/// Deliberately *not* part of [`TwoEBasisHandles`]: the 1e, 2c2e, 3c1e and 3c2e
/// batch paths all share that upload and none of them walks a pair table, so
/// folding these handles in would charge four families for a fifth's data.
#[derive(Debug)]
pub(crate) struct PairTableHandles {
    pub(crate) data: cubecl::server::Handle,
    pub(crate) index: cubecl::server::Handle,
    pub(crate) offset: cubecl::server::Handle,
    pub(crate) data_len: usize,
    pub(crate) index_len: usize,
    pub(crate) offset_len: usize,
    /// The threshold the table was compacted at, handed to the kernel unchanged
    /// so the table and the two in-kernel tests can never disagree.
    pub(crate) expcutoff: f64,
    pub(crate) nbas: u32,
}

pub(crate) fn upload_pair_table<R: Runtime>(
    client: &ComputeClient<R>,
    pairs: &crate::kernels::pair_table::PairTable,
) -> PairTableHandles {
    PairTableHandles {
        data: client.create_from_slice(f64::as_bytes(&pairs.data)),
        index: client.create_from_slice(u32::as_bytes(&pairs.index)),
        offset: client.create_from_slice(u32::as_bytes(&pairs.offset)),
        data_len: pairs.data.len(),
        index_len: pairs.index.len(),
        offset_len: pairs.offset.len(),
        expcutoff: pairs.expcutoff,
        nbas: pairs.nbas,
    }
}

pub(crate) fn upload_2e_basis<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &TwoEFlatBasis,
) -> TwoEBasisHandles {
    TwoEBasisHandles {
        exps: client.create_from_slice(f64::as_bytes(&basis.exps)),
        coeffs: client.create_from_slice(f64::as_bytes(&basis.coeffs)),
        centers: client.create_from_slice(f64::as_bytes(&basis.centers)),
        shell_meta: client.create_from_slice(u32::as_bytes(&basis.shell_meta)),
        exps_len: basis.exps.len(),
        coeffs_len: basis.coeffs.len(),
        centers_len: basis.centers.len(),
        shell_meta_len: basis.shell_meta.len(),
    }
}

/// Which `ResolvedBackend` arm a [`ResidentTwoEBasis`] was uploaded through.
///
/// A device handle is only meaningful to the server that produced it, so a
/// residency carries the arm it came from and refuses a mismatched backend
/// rather than indexing another device's memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Each non-CPU variant is constructed only under its backend's feature, so a
// single-feature build sees the rest as unconstructed. `name()` must still
// cover them all for the mismatch diagnostic.
#[allow(dead_code)]
enum ResidentBackendTag {
    Cpu,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
}

impl ResidentBackendTag {
    fn of(backend: &ResolvedBackend) -> Self {
        match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(_) => Self::Cpu,
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(_, _) => Self::Wgpu,
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(_) => Self::Cuda,
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(_) => Self::Rocm,
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(_, _) => Self::Metal,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
        }
    }
}

/// A shell basis held on the device, shared by every batched family.
///
/// The flattened form is family-independent — exponents, coefficients, centres
/// and a shell table — so `int2e`, `int3c2e` and `int2c2e` all read the same
/// four buffers. The alias exists because the concrete type's name predates
/// that generalization (Task 34-C2) and renaming a public type is not worth a
/// break in the compatibility surface.
pub type ResidentBasis = ResidentTwoEBasis;

/// A shell basis uploaded once and kept on the device across calls (Task 34-C).
///
/// Despite the name it is **not 2e-specific**: the flattened form
/// (`exps` / `coeffs` / `centers` / `shell_meta`) is the same for every batched
/// family, so `int3c2e` and `int2c2e` read the same four buffers rather than
/// uploading their own copies (Task 34-C2). See [`ResidentBasis`].
///
/// [`evaluate_2e_quartet_batch`] already uploads the flattened basis once per
/// *run* rather than once per launch class. This type extends that to once per
/// *basis*: a Fock build that evaluates the same work list every SCF iteration
/// uploads the exponents, coefficients, centres and shell table exactly once,
/// and each later call transfers only its quartet tables.
///
/// The saving is proportional to upload cost, which on the CPU backend is a
/// `memcpy` — the transfer counters in [`BatchExecutionStats`] are what make the
/// effect observable there. It is worth real wall-clock on a discrete GPU.
///
/// A residency is bound to the backend arm it was created on; passing it to a
/// different one returns [`cintxRsError::UnsupportedApi`] instead of reading
/// another device's memory.
#[derive(Debug)]
pub struct ResidentTwoEBasis {
    shells: Vec<BatchShell>,
    handles: TwoEBasisHandles,
    /// The device-side primitive-pair table (S1).
    pair_handles: PairTableHandles,
    /// The `c2s` coefficient tables, uploaded only when the device transform is
    /// selected (M3). `None` otherwise, so a run that does not use them does not
    /// carry 154 KB it will never read.
    c2s_handles: Option<crate::kernels::c2s_device::C2sHandles>,
    /// The same table on the host, so a batch can count the primitive quartets
    /// it is about to dispatch without reading anything back from the device.
    pairs: crate::kernels::pair_table::PairTable,
    tag: ResidentBackendTag,
    upload_bytes: usize,
    reuses: std::sync::atomic::AtomicUsize,
}

impl ResidentTwoEBasis {
    /// Flatten `shells` and upload them to `backend`, keeping the buffers alive
    /// for the lifetime of the returned value.
    ///
    /// The primitive-pair table is built at libcint's own `expcutoff` (S1). Use
    /// [`Self::new_with`] for the unscreened A/B reference.
    pub fn new(backend: &ResolvedBackend, shells: &[BatchShell]) -> Result<Self, cintxRsError> {
        Self::new_with(
            backend,
            shells,
            crate::kernels::pair_table::PairTableOptions::default(),
        )
    }

    /// [`Self::new`] with an explicit primitive-pair screening threshold.
    ///
    /// The threshold is a property of the *residency*, not of an individual
    /// evaluation: the table is compacted at it and the kernel is handed the
    /// same value, so a residency cannot be used under a cutoff it was not
    /// built for. A caller who wants both settings holds two residencies.
    pub fn new_with(
        backend: &ResolvedBackend,
        shells: &[BatchShell],
        pair_options: crate::kernels::pair_table::PairTableOptions,
    ) -> Result<Self, cintxRsError> {
        let flat = flatten_2e_basis(shells);
        let pairs = crate::kernels::pair_table::PairTable::build(shells, pair_options);
        let mut upload_bytes = flat.upload_bytes() + pairs.upload_bytes();
        let wants_c2s = crate::kernels::c2s_device::device_transform_enabled();
        let (handles, pair_handles, c2s_handles) = match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(client) => (
                upload_2e_basis::<cubecl::cpu::CpuRuntime>(client, &flat),
                upload_pair_table::<cubecl::cpu::CpuRuntime>(client, &pairs),
                wants_c2s.then(|| {
                    crate::kernels::c2s_device::upload_c2s_tables::<cubecl::cpu::CpuRuntime>(client)
                }),
            ),
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(client, _) => (
                upload_2e_basis::<cubecl_wgpu::WgpuRuntime>(client, &flat),
                upload_pair_table::<cubecl_wgpu::WgpuRuntime>(client, &pairs),
                wants_c2s.then(|| {
                    crate::kernels::c2s_device::upload_c2s_tables::<cubecl_wgpu::WgpuRuntime>(
                        client,
                    )
                }),
            ),
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(client) => (
                upload_2e_basis::<cubecl_cuda::CudaRuntime>(client, &flat),
                upload_pair_table::<cubecl_cuda::CudaRuntime>(client, &pairs),
                wants_c2s.then(|| {
                    crate::kernels::c2s_device::upload_c2s_tables::<cubecl_cuda::CudaRuntime>(
                        client,
                    )
                }),
            ),
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(client) => (
                upload_2e_basis::<cubecl_hip::HipRuntime>(client, &flat),
                upload_pair_table::<cubecl_hip::HipRuntime>(client, &pairs),
                wants_c2s.then(|| {
                    crate::kernels::c2s_device::upload_c2s_tables::<cubecl_hip::HipRuntime>(client)
                }),
            ),
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(client, _) => (
                upload_2e_basis::<cubecl_wgpu::WgpuRuntime>(client, &flat),
                upload_pair_table::<cubecl_wgpu::WgpuRuntime>(client, &pairs),
                wants_c2s.then(|| {
                    crate::kernels::c2s_device::upload_c2s_tables::<cubecl_wgpu::WgpuRuntime>(
                        client,
                    )
                }),
            ),
        };
        if let Some(tables) = c2s_handles.as_ref() {
            upload_bytes += tables.upload_bytes();
        }
        Ok(Self {
            shells: shells.to_vec(),
            handles,
            pair_handles,
            c2s_handles,
            pairs,
            tag: ResidentBackendTag::of(backend),
            upload_bytes,
            reuses: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// The shells this residency holds, in the order their indices refer to.
    #[must_use]
    pub fn shells(&self) -> &[BatchShell] {
        &self.shells
    }

    /// Bytes the one-time basis upload cost.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        self.upload_bytes
    }

    /// How many evaluations have reused this residency instead of re-uploading.
    #[must_use]
    pub fn reuse_count(&self) -> usize {
        self.reuses.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn check(&self, backend: &ResolvedBackend) -> Result<(), cintxRsError> {
        self.check_for("2e-batch", backend)
    }

    /// [`Self::check`] with the caller's family in the diagnostic.
    ///
    /// A residency is a set of device handles, and a handle is only meaningful
    /// to the server that produced it — which is a property of the *backend*,
    /// not of the integral family reading it. The other batched families
    /// therefore share this residency rather than each keeping their own copy
    /// of the same four buffers (Task 34-C2); `family` only labels the error.
    pub(crate) fn check_for(
        &self,
        family: &str,
        backend: &ResolvedBackend,
    ) -> Result<(), cintxRsError> {
        let tag = ResidentBackendTag::of(backend);
        if tag == self.tag {
            return Ok(());
        }
        Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "{family}:resident-basis-backend-mismatch:uploaded-on-{}:used-on-{}",
                self.tag.name(),
                tag.name()
            ),
        })
    }

    /// The device handles this residency keeps alive.
    pub(crate) fn handles(&self) -> &TwoEBasisHandles {
        &self.handles
    }

    /// Record one evaluation and report whether it was the *first*.
    ///
    /// The basis upload is charged to the first evaluation only, so a repeated
    /// Fock build shows the per-call tables alone and the amortization is
    /// observable rather than asserted.
    pub(crate) fn take_first_use(&self) -> bool {
        self.reuses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            == 0
    }
}

/// Backend dispatch for a whole batched run.
///
/// One match for the entire run rather than one per launch class, so the
/// resident basis handles are bound once on the selected client.
fn dispatch_2e_batches(
    backend: &ResolvedBackend,
    basis: &TwoEBasisHandles,
    pairs: &PairTableHandles,
    groups: &[TwoELaunchGroup],
    options: TwoEBatchOptions,
    on_group: &mut dyn FnMut(usize, Vec<f64>),
    c2s: Option<&C2sChunkPlan<'_>>,
) -> Result<crate::memory_probe::DeviceMemoryProbe, cintxRsError> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => Ok(run_2e_batches::<cubecl::cpu::CpuRuntime>(
            client, basis, pairs, groups, options, on_group, c2s,
        )),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => Ok(run_2e_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, pairs, groups, options, on_group, c2s,
        )),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => Ok(run_2e_batches::<cubecl_cuda::CudaRuntime>(
            client, basis, pairs, groups, options, on_group, c2s,
        )),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => Ok(run_2e_batches::<cubecl_hip::HipRuntime>(
            client, basis, pairs, groups, options, on_group, c2s,
        )),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => Ok(run_2e_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, pairs, groups, options, on_group, c2s,
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Specialization prewarm (def2 plan D2.1)
// ─────────────────────────────────────────────────────────────────────────────

/// What a prewarm pass did.
///
/// Reported rather than swallowed because the cost it moves is real and has to
/// land somewhere in a benchmark's books: a def2-TZVP class set measured ~6.8 s
/// of first-call JIT on the dev host's CPU runtime, and a "warm" number that
/// quietly included it would be the general plan's cold/warm gate violated in a
/// new place.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PrewarmReport {
    /// Distinct angular-momentum classes the shell set can produce.
    pub classes: usize,
    /// Distinct [`TwoELaunchSignature`]s among them — the number of kernel
    /// programs the JIT actually has to build, since Task 35-M1 merges every
    /// class sharing a signature into one dispatch.
    pub signatures: usize,
    /// Dispatches the prewarm issued.
    pub launches: usize,
    /// Quartets per class in the warm-up list.
    pub items_per_class: usize,
    /// Wall time the pass took, including compilation.
    pub elapsed: std::time::Duration,
    /// Classes the backend refused, by `(class, reason)`. A refusal here is not
    /// an error: a shell set may contain a class above the device ceiling, and
    /// the right response is to leave it for the caller's own error handling
    /// rather than to fail the warm-up.
    pub refused: Vec<([u8; 4], String)>,
}

impl PrewarmReport {
    /// Milliseconds of compilation amortized per signature — the number to
    /// quote when saying what a prewarm buys.
    #[must_use]
    pub fn ms_per_signature(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1000.0 / self.signatures.max(1) as f64
    }
}

/// Quartets per class in a prewarm list, on this backend.
///
/// The compiled identity of a dispatch includes its `CubeDim`, and on the
/// per-unit (CPU) decomposition that width is
/// `min(parallel_units, n_quartets, memory_cap)`. A one-quartet warm-up would
/// therefore compile a *one-lane* kernel and leave the real batch to compile its
/// own — the warm-up would cost time and buy nothing. Saturating `n_quartets`
/// past `parallel_units` is what makes the compiled program the same one the
/// real batch asks for.
fn prewarm_items_per_class(backend: &ResolvedBackend) -> usize {
    fn width<R: Runtime>(client: &ComputeClient<R>) -> usize {
        crate::plane::launch_hardware(client).parallel_units as usize
    }
    let units = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => width(client),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => width(client),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => width(client),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => width(client),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => width(client),
    };
    // A small floor so a backend that reports one unit still warms a list wide
    // enough to exercise the grid-stride walk rather than a single slot.
    units.max(8)
}

/// JIT-compile exactly the kernel programs `quartets` will dispatch, before the
/// first timed batch (`def2_speed_precision_plan.md` D2.1).
///
/// # Why this takes a work list rather than just a basis
///
/// A backend specializes a program per `(nroots, ibase, kbase, per_unit,
/// cube_dim)`, and on the per-unit (CPU) decomposition `cube_dim` is
/// `min(parallel_units, quartets_in_this_group, memory_cap)`. The middle term is
/// the problem: **the compiled identity depends on how much work the group
/// holds**, so warming the class set at one width leaves every group smaller
/// than `parallel_units` to compile its own program inside the caller's timing.
///
/// Measured on the dev host (16 units, H2O/def2-SVP): after warming all 16
/// signatures at 16 items each, a batch of 16 quartets was already warm (1.4x
/// its steady state) while batches of 1, 8, 32, 64 and 3081 still cost 780x,
/// 500x, 830x, 1130x and 173x — one compilation per newly reached width.
///
/// So the prewarm reduces the caller's *own* list instead of guessing: it groups
/// the quartets exactly as [`evaluate_2e_quartet_batch`] will, then keeps
/// `min(group_size, parallel_units)` of each group — one quartet per
/// angular-momentum class first, so the group's widest `g_size` (and hence its
/// memory cap) is preserved, then padded. That reproduces every group's
/// `cube_dim` exactly while evaluating a small fraction of the arithmetic.
///
/// # What it costs
///
/// One dispatch per launch signature over a few dozen quartets each, plus the
/// compilation itself — which is the whole point, and is reported in
/// [`PrewarmReport::elapsed`] so a benchmark can put it in the cold column
/// rather than the warm one.
///
/// A class the backend refuses (above the device Rys ceiling, say) is recorded
/// in [`PrewarmReport::refused`] and skipped. Refusing to warm is not refusing
/// to run: the caller's own batch will get the same typed error from the same
/// check, and failing the warm-up would turn an optional optimization into a new
/// way for a program not to start.
///
/// # Errors
/// Only for a malformed input — an empty shell set, or one the basis upload
/// rejects. A per-class refusal is reported, not returned.
pub fn prewarm_2e_work_list(
    backend: &ResolvedBackend,
    shells: &[BatchShell],
    quartets: &[[u32; 4]],
) -> Result<PrewarmReport, cintxRsError> {
    let start = std::time::Instant::now();
    if shells.is_empty() {
        return Err(cintxRsError::UnsupportedApi {
            requested: "prewarm:empty-shell-set".to_owned(),
        });
    }
    let units = prewarm_items_per_class(backend);
    // F1 (§15): warm the programs the real batch will dispatch, which means
    // grouping the representatives the way `plan_2e_stream` will group the
    // work list.
    let fuse = two_e_nroots_fusion();
    let mut report = PrewarmReport {
        items_per_class: units,
        ..Default::default()
    };
    if quartets.is_empty() {
        report.elapsed = start.elapsed();
        return Ok(report);
    }

    // Group by the caller's `(li, lj, lk, ll)`, the same key
    // `evaluate_2e_batch_inner` uses — it applies no canonical swap, so mirroring
    // it is a matter of reading the same four `l` values.
    let mut by_class: std::collections::BTreeMap<[u8; 4], (usize, [u32; 4])> = Default::default();
    for &quartet in quartets {
        for &s in &quartet {
            if s as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("prewarm:shell-index-out-of-range:{s}"),
                });
            }
        }
        let key = [
            shells[quartet[0] as usize].l,
            shells[quartet[1] as usize].l,
            shells[quartet[2] as usize].l,
            shells[quartet[3] as usize].l,
        ];
        let entry = by_class.entry(key).or_insert((0, quartet));
        entry.0 += 1;
    }
    report.classes = by_class.len();

    // Collect each signature's classes, widest `g_size` first: a truncation must
    // never drop the class that sets the group's memory cap.
    let mut by_signature: std::collections::BTreeMap<
        (u32, u32, u32),
        (usize, Vec<(u32, [u32; 4])>),
    > = Default::default();
    for (class, (count, representative)) in by_class {
        let [li, lj, lk, ll] = class;
        let params = TwoEClassParams::new(li, lj, lk, ll);
        let signature = TwoELaunchSignature::of(&params, fuse, None);
        let entry = by_signature
            .entry((signature.ibase, signature.kbase, signature.nroots))
            .or_default();
        entry.0 += count;
        entry.1.push((params.g_size, representative));
    }
    report.signatures = by_signature.len();

    let mut list: Vec<[u32; 4]> = Vec::new();
    for (_, (count, mut classes)) in by_signature {
        classes.sort_by(|a, b| b.0.cmp(&a.0));
        let target = count.min(units);
        let widest = classes[0].1;
        for (_, quartet) in &classes {
            list.push(*quartet);
        }
        // `target` can sit below the class count only when the group already
        // holds at least `parallel_units` quartets, in which case the width has
        // saturated and the extra representatives cost nothing. Otherwise pad
        // up to it so the width matches the real group's exactly.
        let sublist_start = list.len() - classes.len();
        while list.len() - sublist_start < target {
            list.push(widest);
        }
    }

    let resident = ResidentTwoEBasis::new(backend, shells)?;
    match evaluate_2e_quartet_batch_resident(backend, &resident, &list) {
        Ok(output) => report.launches = output.stats.kernel_launch_count,
        Err(_) => {
            // The whole-list call is refused if *any* class is, so warm class by
            // class and record which ones the backend will not take. Slower, but
            // it is the path that produces the diagnosis.
            for quartet in &list {
                let one: Vec<[u32; 4]> = std::iter::repeat_n(*quartet, units).collect();
                match evaluate_2e_quartet_batch_resident(backend, &resident, &one) {
                    Ok(output) => report.launches += output.stats.kernel_launch_count,
                    Err(error) => report.refused.push((
                        [
                            shells[quartet[0] as usize].l,
                            shells[quartet[1] as usize].l,
                            shells[quartet[2] as usize].l,
                            shells[quartet[3] as usize].l,
                        ],
                        error.to_string(),
                    )),
                }
            }
            report.refused.sort();
            report.refused.dedup();
        }
    }

    report.elapsed = start.elapsed();
    tracing::debug!(
        classes = report.classes,
        signatures = report.signatures,
        launches = report.launches,
        refused = report.refused.len(),
        elapsed_ms = report.elapsed.as_millis(),
        "2e work-list prewarm complete"
    );
    Ok(report)
}

/// JIT-compile every launch class a *shell set* can produce, at the saturated
/// launch width.
///
/// The basis-only prewarm: it enumerates the angular-momentum combinations the
/// shell set admits and warms each signature at `parallel_units` quartets, which
/// is the width every group of a steady-state batch reaches.
///
/// **It does not cover a group smaller than `parallel_units`.** `cube_dim` is
/// part of a program's compiled identity and shrinks with the group, so a work
/// list with a thin class still pays one compilation for it. Use
/// [`prewarm_2e_work_list`] when the list is known — it is exact, and costs
/// less. This entry point is for the case where it is not: a driver that has a
/// basis at start-up and will not see its first work list until later.
///
/// # Errors
/// Only for an empty shell set, or one the basis upload rejects.
pub fn prewarm_2e_quartet_classes(
    backend: &ResolvedBackend,
    shells: &[BatchShell],
) -> Result<PrewarmReport, cintxRsError> {
    if shells.is_empty() {
        return Err(cintxRsError::UnsupportedApi {
            requested: "prewarm:empty-shell-set".to_owned(),
        });
    }
    // One representative shell index per distinct angular momentum. `nprim` and
    // `nctr` do not enter the compiled identity — only `l` does, through the
    // G-tensor extents — so the first shell of each `l` is as good as any.
    let mut representative: std::collections::BTreeMap<u8, u32> = Default::default();
    for (index, shell) in shells.iter().enumerate() {
        representative.entry(shell.l).or_insert(index as u32);
    }
    let momenta: Vec<u8> = representative.keys().copied().collect();
    let units = prewarm_items_per_class(backend);

    let mut list = Vec::new();
    for &li in &momenta {
        for &lj in &momenta {
            for &lk in &momenta {
                for &ll in &momenta {
                    let quartet = [
                        representative[&li],
                        representative[&lj],
                        representative[&lk],
                        representative[&ll],
                    ];
                    for _ in 0..units {
                        list.push(quartet);
                    }
                }
            }
        }
    }
    prewarm_2e_work_list(backend, shells, &list)
}

#[cfg(test)]
mod chunk_cap_tests {
    use super::*;

    fn s_shell() -> BatchShell {
        BatchShell {
            l: 0,
            nprim: 1,
            nctr: 1,
            exponents: vec![1.0],
            coefficients: vec![1.0],
            center: [0.0; 3],
        }
    }

    /// The quartet cap splits a list the byte budget would keep whole, and the
    /// chunks tile the list exactly.
    #[test]
    fn quartet_cap_bounds_every_chunk() {
        let shells = vec![s_shell()];
        let quartets = vec![[0_u32, 0, 0, 0]; 10];
        let whole = plan_quartet_chunks_capped(&quartets, &shells, usize::MAX, None);
        assert_eq!(whole, vec![0..10]);
        let capped = plan_quartet_chunks_capped(&quartets, &shells, usize::MAX, Some(4));
        assert_eq!(capped, vec![0..4, 4..8, 8..10]);
        assert!(capped.iter().all(|r| r.len() <= 4));
        // The byte budget still applies underneath the cap.
        let tight = plan_quartet_chunks_capped(&quartets, &shells, 8, Some(4));
        assert_eq!(tight.len(), 10);
    }
}

#[cfg(test)]
mod kl_split_tests {
    use super::{KL_REDUCE_ROW_STRIDE, QUARTET_ROW_STRIDE, expand_kl_split, quartet_block_lens};

    /// Two rows: a 7-row ket range writing a 40-element block at offset 0, and
    /// a 2-row range writing a 60-element block at offset 40. Group output 100.
    const ROWS: [u32; 2 * QUARTET_ROW_STRIDE] = [
        0, 1, 2, 3, 0, 5, 10, 17, //
        0, 1, 3, 3, 40, 6, 30, 32,
    ];
    const COSTS: [u64; 2] = [900, 60];

    #[test]
    fn block_lengths_come_from_the_gaps_between_output_offsets() {
        assert_eq!(quartet_block_lens(&ROWS, 100), vec![40, 60]);
    }

    #[test]
    fn parts_tile_the_ket_range_in_order_and_offset_their_output() {
        let out = expand_kl_split(&ROWS, &[3, 3], &[40, 60], &COSTS, 100);
        assert_eq!(out.rows.len(), 2 * 3 * QUARTET_ROW_STRIDE);
        let parts: Vec<&[u32]> = out.rows.chunks_exact(QUARTET_ROW_STRIDE).collect();
        // First quartet: 7 rows over 3 parts → 2, 2, 3, contiguous, in order.
        assert_eq!(&parts[0][6..], &[10, 12]);
        assert_eq!(&parts[1][6..], &[12, 14]);
        assert_eq!(&parts[2][6..], &[14, 17]);
        // Part 0 writes the quartet's own block; the rest take consecutive
        // blocks of the partial region, which starts at `out_len`.
        assert_eq!(parts[0][4], 0);
        assert_eq!(parts[1][4], 100);
        assert_eq!(parts[2][4], 140);
        // Second quartet: 2 rows over 3 parts → one part is empty.
        assert_eq!(&parts[3][6..], &[30, 30]);
        assert_eq!(&parts[4][6..], &[30, 31]);
        assert_eq!(&parts[5][6..], &[31, 32]);
        assert_eq!(parts[3][4], 40);
        assert_eq!(parts[4][4], 100 + 80);
        assert_eq!(parts[5][4], 100 + 80 + 60);
        // Shells and class carry over unchanged.
        assert_eq!(&parts[4][..4], &[0, 1, 3, 3]);
        assert_eq!(parts[4][5], 6);
        // The partial region holds exactly the non-zeroth parts.
        assert_eq!(out.extra_len, 2 * 40 + 2 * 60);
        assert_eq!(
            out.reduce_table,
            vec![
                100, 2, 0, 40, //
                180, 2, 40, 60,
            ]
        );
        assert_eq!(out.reduce_table.len(), 2 * KL_REDUCE_ROW_STRIDE);
    }

    #[test]
    fn every_row_carries_its_own_share_of_its_quartets_cost() {
        // K2 ranks rows, so a split quartet's rows must each carry a part's
        // cost, not the whole quartet's — otherwise the partition believes the
        // dispatch is `parts` times more expensive than it is and cuts wrong.
        let out = expand_kl_split(&ROWS, &[3, 1], &[40, 60], &COSTS, 100);
        assert_eq!(out.row_cost.len(), out.rows.len() / QUARTET_ROW_STRIDE);
        assert_eq!(out.row_cost, vec![300, 300, 300, 60]);
    }

    #[test]
    fn an_unsplit_quartet_costs_no_partial_block_beside_a_split_one() {
        // The shape a fused dispatch produces: one expensive quartet takes
        // parts, its cheap neighbour takes none and keeps writing in place.
        let out = expand_kl_split(&ROWS, &[4, 1], &[40, 60], &COSTS, 100);
        assert_eq!(out.rows.len(), 5 * QUARTET_ROW_STRIDE);
        let parts: Vec<&[u32]> = out.rows.chunks_exact(QUARTET_ROW_STRIDE).collect();
        assert_eq!(parts[0][4], 0);
        assert_eq!(parts[1][4], 100);
        assert_eq!(parts[2][4], 140);
        assert_eq!(parts[3][4], 180);
        // The unsplit quartet keeps its whole ket range and its own offset.
        assert_eq!(&parts[4][6..], &[30, 32]);
        assert_eq!(parts[4][4], 40);
        // Only the split quartet's extra parts are charged.
        assert_eq!(out.extra_len, 3 * 40);
        // The unsplit quartet is absent from the table: nothing to reduce.
        assert_eq!(out.reduce_table, vec![100, 3, 0, 40]);
    }

    #[test]
    fn a_split_of_one_is_the_identity() {
        let out = expand_kl_split(&ROWS, &[1, 1], &[40, 60], &COSTS, 100);
        assert_eq!(out.rows, ROWS.to_vec());
        assert_eq!(out.extra_len, 0);
        assert_eq!(out.row_cost, COSTS.to_vec());
        // Nothing split, so there is nothing for the reduce to do at all.
        assert!(out.reduce_table.is_empty());
    }
}

#[cfg(test)]
mod partition_tests {
    use super::{TwoEClassParams, per_unit_slot_bounds, quartet_cost_estimate};

    fn covers_every_row_once(bounds: &[u32], n: usize, n_slots: usize) {
        assert_eq!(bounds.len(), n_slots + 1);
        assert_eq!(bounds[0], 0);
        assert_eq!(*bounds.last().unwrap() as usize, n);
        assert!(
            bounds.windows(2).all(|w| w[0] <= w[1]),
            "monotone: {bounds:?}"
        );
    }

    #[test]
    fn uniform_bounds_reproduce_the_blocked_walk() {
        // `ceil(n / n_slots)` rows per slot, the last slots possibly empty —
        // the shape the kernel computed for itself before K2.
        let cost = vec![1_u64; 10];
        let bounds = per_unit_slot_bounds(&cost, 4, false);
        assert_eq!(bounds, vec![0, 3, 6, 9, 10]);
        let bounds = per_unit_slot_bounds(&cost, 16, false);
        covers_every_row_once(&bounds, 10, 16);
        assert_eq!(&bounds[..11], &(0..=10).collect::<Vec<u32>>()[..]);
    }

    #[test]
    fn balanced_bounds_give_the_expensive_tail_fewer_rows() {
        // Rows appended class by class: nine cheap rows then three that each
        // cost as much as all the cheap ones together.
        let mut cost = vec![1_u64; 9];
        cost.extend_from_slice(&[9, 9, 9]);
        let bounds = per_unit_slot_bounds(&cost, 4, true);
        covers_every_row_once(&bounds, 12, 4);
        // Total 36, a quarter is 9: the cheap rows form one slot, and each
        // expensive row gets a slot of its own.
        assert_eq!(bounds, vec![0, 9, 10, 11, 12]);
    }

    #[test]
    fn balanced_bounds_handle_degenerate_shapes() {
        covers_every_row_once(&per_unit_slot_bounds(&[], 4, true), 0, 4);
        covers_every_row_once(&per_unit_slot_bounds(&[5], 4, true), 1, 4);
        covers_every_row_once(&per_unit_slot_bounds(&[1, 2, 3], 1, true), 3, 1);
        // A slot count above the row count leaves slots empty, never
        // double-booked — and pins the cuts, not just their shape. Consecutive
        // targets are closer together than one row is wide here, so a cut that
        // steps past its target is still the nearest one for the slots after
        // it; the repeated bounds below are that, and every one of them is a
        // slot that gets no rows.
        //
        // The exact vector is the assertion that matters: a `target - prefix`
        // that wrapped instead of measuring a distance also satisfies
        // `covers_every_row_once`, and only differs here.
        let bounds = per_unit_slot_bounds(&[3, 1, 4, 1, 5], 16, true);
        covers_every_row_once(&bounds, 5, 16);
        assert_eq!(
            bounds,
            vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        );
    }

    /// Every cut must be the row boundary nearest its slot's share, out of the
    /// one before it and the one after — the property the greedy exists to
    /// deliver, checked against the bounds themselves rather than by replaying
    /// the loop that produced them.
    ///
    /// Swept exhaustively over the small shapes, which is where the arithmetic
    /// goes wrong: an overshooting cut is only reachable when slots outnumber
    /// rows, and `balanced_bounds_give_the_expensive_tail_fewer_rows` never
    /// gets there.
    #[test]
    fn balanced_cuts_are_the_nearest_row_boundary() {
        fn check(cost: &[u64], n_slots: usize) {
            let bounds = per_unit_slot_bounds(cost, n_slots, true);
            covers_every_row_once(&bounds, cost.len(), n_slots);

            let total: u128 = cost.iter().map(|&c| u128::from(c)).sum();
            let prefix: Vec<u128> = cost
                .iter()
                .scan(0_u128, |acc, &c| {
                    *acc += u128::from(c);
                    Some(*acc)
                })
                .collect();
            let at = |cut: usize| if cut == 0 { 0 } else { prefix[cut - 1] };

            for (s, &cut) in bounds.iter().enumerate().take(n_slots).skip(1) {
                let cut = cut as usize;
                let target = total * s as u128 / n_slots as u128;
                let here = at(cut).abs_diff(target);
                if cut > 0 {
                    assert!(
                        here <= at(cut - 1).abs_diff(target),
                        "cost={cost:?} n_slots={n_slots} slot {s}: cut {cut} is further \
                         from {target} than {} is ({bounds:?})",
                        cut - 1
                    );
                }
                if cut < cost.len() {
                    assert!(
                        here <= at(cut + 1).abs_diff(target),
                        "cost={cost:?} n_slots={n_slots} slot {s}: cut {cut} is further \
                         from {target} than {} is ({bounds:?})",
                        cut + 1
                    );
                }
            }
        }

        // Every cost vector over {1, 2, 3} up to four rows, against every slot
        // count up to eight — 968 shapes, all of the slots-outnumber-rows kind
        // among them.
        let mut cost = Vec::new();
        for len in 0..=4usize {
            for encoded in 0..3usize.pow(len as u32) {
                cost.clear();
                let mut rest = encoded;
                for _ in 0..len {
                    cost.push((rest % 3 + 1) as u64);
                    rest /= 3;
                }
                for n_slots in 1..=8usize {
                    check(&cost, n_slots);
                }
            }
        }
    }

    /// The flat term of [`quartet_cost_estimate`] is what decides how far the
    /// ket-pair split reaches into the *narrow* classes, so it is pinned here
    /// against a silent change (§23.2 swept it and kept `100`).
    #[test]
    fn the_fixed_term_is_what_sets_the_narrow_to_wide_ratio() {
        let ssss = TwoEClassParams::new(0, 0, 0, 0);
        let pppp = TwoEClassParams::new(1, 1, 1, 1);
        // At the default the ranking is the one §22.6 questioned: an `(ss|ss)`
        // primitive quartet is charged about a twentieth of a `(pp|pp)` one.
        let narrow = quartet_cost_estimate(1, &ssss, 3);
        let wide = quartet_cost_estimate(1, &pppp, 3);
        assert_eq!(narrow, 9 + 10 + super::PRIM_FIXED_COST);
        assert_eq!(wide, 81 * 15 + 10 * 108 + super::PRIM_FIXED_COST);
        assert!(wide > 15 * narrow && wide < 25 * narrow);
        // The term is flat: raising it closes the ratio without moving the
        // wide class, which is the shape §23.2 tested and why it is not the
        // same knob as `KL_SPLIT_TARGET_PART_COST`.
        let lift = 1_100_u64 - super::PRIM_FIXED_COST;
        assert!((wide + lift) < 4 * (narrow + lift));
        assert!((wide + lift) < wide * 3 / 2);
    }

    #[test]
    fn cost_estimate_ranks_by_primitives_and_block() {
        let ssss = TwoEClassParams::new(0, 0, 0, 0);
        let pppp = TwoEClassParams::new(1, 1, 1, 1);
        // Same primitive count: the `(pp|pp)` row is far dearer.
        assert!(quartet_cost_estimate(2401, &pppp, 3) > 20 * quartet_cost_estimate(2401, &ssss, 3));
        // Same class: cost grows with the screened primitive count.
        assert!(quartet_cost_estimate(2401, &pppp, 3) > quartet_cost_estimate(625, &pppp, 3));
        // Never zero, so an empty pair list still takes a row's worth.
        assert!(quartet_cost_estimate(0, &ssss, 1) > 0);
    }
}
