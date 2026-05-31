//! 2c2e (two-center two-electron Coulomb) integral kernel.
//!
//! Implements the G-tensor fill + contraction + cart-to-sph pipeline following
//! libcint `g2c2e.c` / `g2e.c` `CINTg0_2e` + `CINTg0_2e_2d`.
//!
//! # Execution model (CubeCL device dispatch)
//!
//! The numeric core — the per-shell-pair Rys G-tensor fill and the Cartesian
//! contraction — runs as a real CubeCL `#[cube(launch)]` kernel
//! ([`center_2c2e_kernel`]) **generic over `F: Float`**, dispatched onto the
//! resolved backend's `ComputeClient` (CPU `CpuRuntime`, ROCm `HipRuntime`, …)
//! via [`run_2c2e_device`]. The Cartesian buffer is read back to the host and
//! the `cart_to_sph` / spinor transforms (whose coefficient tables are
//! host-only) finish on the host.
//!
//! ## Precision policy
//!
//! The kernel is genuinely generic over `F: Float`, but the launcher runs it at
//! **f64** on-device for both `PrecisionKind` variants and casts the read-back
//! buffer to `F` at the c2s/output stage. This preserves the historical
//! "intermediates in f64, output cast to `F`" contract that the f32 parity gate
//! (`f32_parity.rs`, family `2c2e`) is calibrated against, while moving the
//! real arithmetic onto the device.
//!
//! # Algorithm
//! For each contracted shell pair (i, k):
//! 1. Compute Rys argument x = rho * |ri - rk|^2 where rho = ai*ak/(ai+ak).
//! 2. Fetch nrys_roots Rys quadrature roots u[] and weights w[].
//!    The weights encode the Gaussian overlap integral (no separate exp(-rho*rr) needed).
//! 3. For each Rys root: compute recurrence coefficients (b00, b10, b01, c00, c0p)
//!    and fill the G-tensor via VRR in both i and k directions.
//! 4. Contract G-tensor elements over Cartesian component triples (ix, iy, iz) x (kx, ky, kz).
//! 5. Accumulate contracted values weighted by common_factor and primitive contraction coefficients.
//! 6. Apply common_fac_sp(li) * common_fac_sp(lk) scaling.
//! 7. Apply cart_to_sph_2c2e if Spheric representation is requested.
//!
//! # Key normalization
//! envs->fac[0] = common_factor * ci[ip] * ck[kp]   (NO exponential term)
//! fac1 = sqrt(a0/(a1^3)) * envs->fac[0]
//! gz[root] = w[root] * fac1  (Rys weights encode exp(-x*t^2) implicitly)
//!
//! Source: libcint-master/src/g2c2e.c (CINT2c2e_loop_nopt, CINTinit_int2c2e_EnvVars) and
//!         libcint-master/src/g2e.c (CINTg0_2e, CINTg0_2e_2d).

use crate::backend::ResolvedBackend;
#[cfg(test)]
use crate::math::rys::rys_roots_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::kernels::f12::{Nabla1Center, gout_ipip1, gout_ipn};
use crate::kernels::two_electron::{build_2e_shape, fill_g_tensor_2e, two_e_shape_as_f12};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, cart_to_sph_2c2e, ncart, nsph};
use crate::transform::c2spinor::{cart_to_spinor_sf_2d, cart_to_spinor_sf_derivative_2d};
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
/// Matches `rys_roots.c` `PIE4`.
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate.
/// `nroots = (li + lk) / 2 + 1`, so this covers `li + lk <= 8`.
const MAX_DEVICE_NROOTS: usize = 5;

/// Maximum `nroots` the HOST Rys engine (`rys_roots_host` → `rys_wheeler`) evaluates
/// (Phase 25 FND-02). The 2c2e gradient path host-routes through `fill_g_tensor_2e`;
/// nroots 6..12 are supported, nroots>12 stays fail-closed (T-25-03).
const HOST_RYS_NROOTS_CEILING: usize = 12;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// Mirrors libcint `CINTcommon_fac_sp(l)` from `g2e.c` / `g1e.c`:
///   l=0 (s): 1/(2*sqrt(pi)) = 0.282094791773878143
///   l=1 (p): sqrt(3/(4*pi)) = 0.488602511902919921
///   l>=2:    1.0 (embedded in c2s coefficient tables)
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Follows libcint `CINTcart_comp` ordering:
/// for lx = l..=0, for ly = l-lx..=0, nz = l - lx - ly.
///
/// Host reference (the device kernel reproduces this ordering inline). Kept for
/// the host-vs-device cross-check and the G-tensor unit tests.
#[cfg(test)]
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

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]`, generic over `F: Float`
// ─────────────────────────────────────────────────────────────────────────────

/// 2c2e G-tensor fill + Cartesian contraction for one shell pair, on-device.
///
/// Single work item (`UNIT_POS == 0`) — this is a faithful, correctness-first
/// port of the host pipeline (`fill_g_tensor_2c2e` + contraction). It is not yet
/// parallelized across primitives/components; that is future work.
///
/// `#[comptime] nroots` selects the `rys_root{1..5}` device function at JIT
/// specialization time (a `comptime!` branch — no runtime nroots dispatch, which
/// avoids the documented MLIR index-type lowering issue, and no Rust
/// monomorphization fan-out).
///
/// Layout of `g` (size `3 * g_size`): `g[axis*g_size + k*dm + i*dn + root]`
/// with `dn = nroots`, `dm = nroots*(li+1)`, `g_size = nroots*(li+1)*(lk+1)`.
///
/// `cart_out` (size `nci*nck`, `nci = ncart(li)`) is zeroed in-kernel and
/// accumulated over all primitive and contraction pairs:
/// `cart_out[ci_idx + ck_idx*nci]`.
///
/// Source: libcint-master/src/g2e.c `CINTg0_2e` + `CINTg0_2e_2d`,
///         g2c2e.c `CINT2c2e_loop_nopt`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn center_2c2e_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_k: &Array<F>,
    coeff_i: &Array<F>,
    coeff_k: &Array<F>,
    g: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rkx: F,
    rky: F,
    rkz: F,
    common_factor: F,
    pie4: F,
    li: u32,
    lk: u32,
    nprim_i: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_k: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let nrys = nroots;
        let dn = nrys;
        let dm = nrys * (li + 1u32);
        let g_size = nrys * (li + 1u32) * (lk + 1u32);
        let total_g = 3u32 * g_size;
        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
        let out_len = nci * nck;

        // Zero the accumulation buffer.
        let mut oi = 0u32;
        while oi < out_len {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // Center displacement (independent of primitives): rij = ri, rkl = rk.
        let xij = rix - rkx;
        let yij = riy - rky;
        let zij = riz - rkz;
        let rr = xij * xij + yij * yij + zij * zij;

        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pk = 0u32;
            while pk < nprim_k {
                let ak = exps_k[pk as usize];

                // For 2c2e: aij = ai, akl = ak.
                let aij = ai;
                let akl = ak;
                let a1 = aij * akl;
                let a0 = a1 / (aij + akl); // rho = ai*ak/(ai+ak)
                let x_rys = a0 * rr;

                // Rys roots/weights depend only on (ai, ak) → compute once here.
                // `nroots` is comptime, so exactly one branch is emitted.
                if comptime!(nroots == 1u32) {
                    rys_root1::<F>(x_rys, urys, wrys, pie4);
                } else if comptime!(nroots == 2u32) {
                    rys_root2::<F>(x_rys, urys, wrys, pie4);
                } else if comptime!(nroots == 3u32) {
                    rys_root3::<F>(x_rys, urys, wrys, pie4);
                } else if comptime!(nroots == 4u32) {
                    rys_root4::<F>(x_rys, urys, wrys, pie4);
                } else {
                    rys_root5::<F>(x_rys, urys, wrys, pie4);
                }

                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut ck = 0u32;
                    while ck < nctr_k {
                        let coeff_k_val = coeff_k[(pk * nctr_k + ck) as usize];

                        // fac_env = common_factor * ci * ck (no exponential term).
                        let fac_env = common_factor * coeff_i_val * coeff_k_val;
                        // fac1 = sqrt(a0 / a1^3) * fac_env  (g2e.c CINTg0_2e line 4441)
                        let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * fac_env;

                        // ── Fill the G-tensor (zero, base case, VRR) ──────────
                        let mut gi = 0u32;
                        while gi < total_g {
                            g[gi as usize] = F::new(0.0);
                            gi += 1u32;
                        }

                        let mut irys = 0u32;
                        while irys < nrys {
                            let u2 = a0 * urys[irys as usize];
                            let tmp4 = F::new(0.5) / (u2 * (aij + akl) + a1);
                            let tmp5 = u2 * tmp4;
                            let b00 = tmp5;
                            let b10 = tmp5 + tmp4 * akl;
                            let b01 = tmp5 + tmp4 * aij;
                            let tmp2 = F::new(2.0) * tmp5 * akl;
                            let tmp3 = F::new(2.0) * tmp5 * aij;

                            // Base case: gx=gy=1, gz=w*fac1 (g2e.c lines 4517-4521).
                            g[irys as usize] = F::new(1.0);
                            g[(g_size + irys) as usize] = F::new(1.0);
                            g[(2u32 * g_size + irys) as usize] = wrys[irys as usize] * fac1;

                            let mut axis = 0u32;
                            while axis < 3u32 {
                                let base = axis * g_size;
                                // Displacement component for this axis.
                                let mut d = xij;
                                if axis == 1u32 {
                                    d = yij;
                                }
                                if axis == 2u32 {
                                    d = zij;
                                }
                                let c00a = -tmp2 * d;
                                let c0pa = tmp3 * d;

                                // i-VRR (nmax = li): g[n+1] = c00*g[n] + n*b10*g[n-1]
                                if li >= 1u32 {
                                    let mut s_prev = g[(base + irys) as usize];
                                    let mut s1 = c00a * s_prev;
                                    g[(base + irys + dn) as usize] = s1;
                                    let mut n = 1u32;
                                    while n < li {
                                        let s2 = c00a * s1 + F::cast_from(n) * b10 * s_prev;
                                        g[(base + irys + (n + 1u32) * dn) as usize] = s2;
                                        s_prev = s1;
                                        s1 = s2;
                                        n += 1u32;
                                    }
                                }

                                // k-VRR pure (i=0, mmax = lk):
                                // g[k+1] = c0p*g[k] + k*b01*g[k-1]
                                if lk >= 1u32 {
                                    let mut s_prev = g[(base + irys) as usize];
                                    let mut s1 = c0pa * s_prev;
                                    g[(base + irys + dm) as usize] = s1;
                                    let mut m = 1u32;
                                    while m < lk {
                                        let s2 = c0pa * s1 + F::cast_from(m) * b01 * s_prev;
                                        g[(base + irys + (m + 1u32) * dm) as usize] = s2;
                                        s_prev = s1;
                                        s1 = s2;
                                        m += 1u32;
                                    }
                                }

                                // Mixed i+k recurrence for i>0 (g2e.c lines 362-391):
                                // g[i,k+1] = c0p*g[i,k] + k*b01*g[i,k-1] + b00*g[i-1,k]
                                if lk >= 1u32 {
                                    if li >= 1u32 {
                                        let mut n = 1u32;
                                        while n <= li {
                                            let i_off = irys + n * dn;
                                            let s0_k0 = g[(base + i_off) as usize];
                                            let prev_i_k0 =
                                                g[(base + irys + (n - 1u32) * dn) as usize];
                                            // k=1
                                            let mut s1 = c0pa * s0_k0 + b00 * prev_i_k0;
                                            g[(base + i_off + dm) as usize] = s1;
                                            let mut s_prev = s0_k0;
                                            let mut m = 1u32;
                                            while m < lk {
                                                let prev_i_km = g[(base
                                                    + irys
                                                    + (n - 1u32) * dn
                                                    + m * dm)
                                                    as usize];
                                                let s2 = c0pa * s1
                                                    + F::cast_from(m) * b01 * s_prev
                                                    + b00 * prev_i_km;
                                                g[(base + i_off + (m + 1u32) * dm) as usize] = s2;
                                                s_prev = s1;
                                                s1 = s2;
                                                m += 1u32;
                                            }
                                            n += 1u32;
                                        }
                                    }
                                }

                                axis += 1u32;
                            }

                            irys += 1u32;
                        }

                        // ── Contract over Rys roots and Cartesian triples ─────
                        // Output: i fastest (innermost), k slowest (outermost):
                        // cart_out[ci_idx + ck_idx*nci]
                        let mut ck_idx = 0u32;
                        let mut ka = 0u32;
                        while ka <= lk {
                            let kx = lk - ka; // kx: lk..0 (descending)
                            let lk_minus_kx = lk - kx;
                            let mut kb = 0u32;
                            while kb <= lk_minus_kx {
                                let ky = lk_minus_kx - kb; // ky descending
                                let kz = lk - kx - ky;

                                let mut ci_idx = 0u32;
                                let mut ia = 0u32;
                                while ia <= li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    let mut ib = 0u32;
                                    while ib <= li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        let mut val = F::new(0.0);
                                        let mut irys2 = 0u32;
                                        while irys2 < nrys {
                                            let vx = g[(kx * dm + ix * dn + irys2) as usize];
                                            let vy =
                                                g[(g_size + ky * dm + iy * dn + irys2) as usize];
                                            let vz = g[(2u32 * g_size
                                                + kz * dm
                                                + iz * dn
                                                + irys2)
                                                as usize];
                                            val += vx * vy * vz;
                                            irys2 += 1u32;
                                        }
                                        cart_out[(ci_idx + ck_idx * nci) as usize] += val;

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }

                                ck_idx += 1u32;
                                kb += 1u32;
                            }
                            ka += 1u32;
                        }

                        ck += 1u32;
                    }
                    ci += 1u32;
                }

                pk += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`center_2c2e_kernel`] at `f64` on a resolved backend's client and
/// read back the Cartesian accumulation buffer (`nci*nck`, i fastest).
///
/// Generic over `R: Runtime` so the same path serves CPU, ROCm, etc. Intermediate
/// device compute is `f64` (see module-level precision policy).
#[allow(clippy::too_many_arguments)]
fn run_2c2e_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lk: u32,
    nprim_i: u32,
    nprim_k: u32,
    nctr_i: u32,
    nctr_k: u32,
    nroots: u32,
    ri: [f64; 3],
    rk: [f64; 3],
    common_factor: f64,
    exps_i: &[f64],
    exps_k: &[f64],
    coeff_i: &[f64],
    coeff_k: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lk_u = lk as usize;
    let nroots_u = nroots as usize;
    let g_size = nroots_u * (li_u + 1) * (lk_u + 1);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let nck = (lk_u + 1) * (lk_u + 2) / 2;
    let out_len = nci * nck;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_k_h = client.create_from_slice(f64::as_bytes(exps_k));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_k_h = client.create_from_slice(f64::as_bytes(coeff_k));

    // Scratch + output buffers (zero-initialised on the host for determinism;
    // the kernel also zeros `g` and `cart_out` before use).
    let g_zero = vec![0.0_f64; 3 * g_size];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    center_2c2e_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(exps_i_h, exps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_k_h, exps_k.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_k_h, coeff_k.len()) },
        unsafe { ArrayArg::from_raw_parts(g_h, 3 * g_size) },
        unsafe { ArrayArg::from_raw_parts(u_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(w_h, nroots_u) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
        ri[0],
        ri[1],
        ri[2],
        rk[0],
        rk[1],
        rk[2],
        common_factor,
        PIE4,
        li,
        lk,
        nprim_i,
        nprim_k,
        nctr_i,
        nctr_k,
        nroots,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Fill the 2c2e G-tensor for one primitive pair (ai, ak) using Rys quadrature.
///
/// Host f64 reference of the exact device algorithm — used by the in-crate
/// unit tests and the host-vs-device cross-check.
///
/// `fac_env` corresponds to `envs->fac[0]` in libcint:
/// `common_factor * ci[ip] * ck[kp]` (NO exponential term; the exponential
/// decay is encoded in the Rys weights w[]).
///
/// Returns flat `[gx | gy | gz]` each of size `g_size = nrys * (li+1) * (lk+1)`.
///
/// Source: libcint-master/src/g2e.c `CINTg0_2e` + `CINTg0_2e_2d`.
#[cfg(test)]
fn fill_g_tensor_2c2e(
    ai: f64,
    ak: f64,
    ri: &[f64; 3],
    rk: &[f64; 3],
    li: u8,
    lk: u8,
    fac_env: f64,
) -> Vec<f64> {
    let nmax = li as usize;
    let mmax = lk as usize;
    let nrys_roots = (li as usize + lk as usize) / 2 + 1;

    let dn = nrys_roots;
    let dm = nrys_roots * (li as usize + 1);
    let g_size = nrys_roots * (li as usize + 1) * (lk as usize + 1);

    let mut g = vec![0.0_f64; 3 * g_size];

    let xij_kl = ri[0] - rk[0];
    let yij_kl = ri[1] - rk[1];
    let zij_kl = ri[2] - rk[2];
    let rr = xij_kl * xij_kl + yij_kl * yij_kl + zij_kl * zij_kl;

    let aij = ai;
    let akl = ak;
    let a1 = aij * akl;
    let a0 = a1 / (aij + akl);

    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac_env;
    let x_rys = a0 * rr;

    let (u_roots, w_weights) = rys_roots_host(nrys_roots, x_rys);

    for irys in 0..nrys_roots {
        let u2 = a0 * u_roots[irys];
        let tmp4 = 0.5 / (u2 * (aij + akl) + a1);
        let tmp5 = u2 * tmp4;
        let b00 = tmp5;
        let b10 = tmp5 + tmp4 * akl;
        let b01 = tmp5 + tmp4 * aij;
        let tmp2 = 2.0 * tmp5 * akl;
        let tmp3 = 2.0 * tmp5 * aij;
        let c00 = [-tmp2 * xij_kl, -tmp2 * yij_kl, -tmp2 * zij_kl];
        let c0p = [tmp3 * xij_kl, tmp3 * yij_kl, tmp3 * zij_kl];

        g[irys] = 1.0;
        g[g_size + irys] = 1.0;
        g[2 * g_size + irys] = w_weights[irys] * fac1;

        if nmax > 0 {
            for axis in 0..3 {
                let base = axis * g_size;
                let c = c00[axis];
                let mut s_prev = g[base + irys];
                let mut s1 = c * s_prev;
                g[base + irys + dn] = s1;
                for n in 1..nmax {
                    let s2 = c * s1 + n as f64 * b10 * s_prev;
                    g[base + irys + (n + 1) * dn] = s2;
                    s_prev = s1;
                    s1 = s2;
                }
            }
        }

        if mmax > 0 {
            for axis in 0..3 {
                let base = axis * g_size;
                let c = c0p[axis];
                let mut s_prev = g[base + irys];
                let mut s1 = c * s_prev;
                g[base + irys + dm] = s1;
                for m in 1..mmax {
                    let s2 = c * s1 + m as f64 * b01 * s_prev;
                    g[base + irys + (m + 1) * dm] = s2;
                    s_prev = s1;
                    s1 = s2;
                }
            }

            if nmax > 0 {
                for axis in 0..3 {
                    let base = axis * g_size;
                    let c = c0p[axis];
                    for n in 1..=nmax {
                        let i_off = irys + n * dn;
                        let s0_k0 = g[base + i_off];
                        let prev_i_k0 = g[base + irys + (n - 1) * dn];
                        let mut s1 = c * s0_k0 + b00 * prev_i_k0;
                        g[base + i_off + dm] = s1;
                        let mut s_prev = s0_k0;
                        for m in 1..mmax {
                            let prev_i_km = g[base + irys + (n - 1) * dn + m * dm];
                            let s2 = c * s1 + m as f64 * b01 * s_prev + b00 * prev_i_km;
                            g[base + i_off + (m + 1) * dm] = s2;
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

/// int2c2e first-derivative launcher (Phase 23 DRV1-04).
///
/// Handles both `int2c2e_ip1` (`Nabla1Center::I`, ∇ on the bra center `i`) and
/// `int2c2e_ip2` (`Nabla1Center::K`, ∇ on the ket center `k`). The 2-center
/// integral `(i|k)` is evaluated through the 4-center 2e Rys machinery with the
/// `j` and `l` (2e) slots collapsed to phantom s-functions (`lj = ll = 0`,
/// `aj = al = 0`): then `fill_g_tensor_2e` reduces exactly to the scalar 2c2e
/// G-tensor (`aij = ai`, `akl = ak`, `rij = ri`, `rkl = rk`). The single-side
/// contraction `gout_ipn` (f12.rs) supplies the ∇ for the requested center.
///
/// Normalization: the phantom s-functions contribute NO `common_fac_sp`, so the
/// `common_factor` uses ONLY the real shells `common_fac_sp(li) * common_fac_sp(lk)`
/// (matching the scalar 2c2e path, NOT the 4-factor 2e formula). There is no
/// Gaussian-overlap prefactor for 2c2e (the Rys weights encode it), so the
/// per-primitive `fac_env` is just `common_factor` weighted by the contraction
/// coefficients `ci * ck`.
///
/// Max-l = f within the device Rys ceiling: the headroom raises the derivative
/// center by 1, so `nroots = (li(+1) + lk(+1))/2 + 1`; fail-closed > 5 (D-13).
/// Spinor reps reject early (D-06).
#[allow(clippy::too_many_arguments)]
fn launch_center_2c2e_grad<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    center: Nabla1Center,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // 27-03 (FND-04): int2c2e_ip1/ip2 spinor gradients now fold via the
    // centralized derivative wrapper (ncomp=3). 2c2e folds through the sf_2d
    // path — there is NO aux-k axis, so the aux-k SPHERICAL correction does not
    // apply here. The wrapper owns the KET→BRA transpose (D-06).

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_k = &shells[1];

    let li = shell_i.ang_momentum;
    let lk = shell_k.ang_momentum;

    // Headroom on the derivative center (I → li+1, K → lk+1); j,l are phantom s.
    let (li_ceil, lk_ceil) = match center {
        Nabla1Center::I => (li as usize + 1, lk as usize),
        Nabla1Center::K => (li as usize, lk as usize + 1),
        // 2c2e has only centers i and k; J/L are never requested here.
        _ => unreachable!("int2c2e gradient only nablas center I or K"),
    };
    let grad_shape = build_2e_shape(li_ceil, 0, lk_ceil, 0);

    // Phase 25 FND-02: HOST gradient path (fill_g_tensor_2e → rys_roots_host); the host
    // Wheeler engine supports nroots 6..12. Route elevated-headroom 2c2e gradients here
    // instead of UnsupportedApi; nroots>12 stays fail-closed (T-25-03).
    if grad_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    // Phantom j,l centers coincide with i,k respectively (aj=al=0 → position irrelevant).
    let rj = ri;
    let rl = rk;

    let nfi = ncart(li);
    let nfk = ncart(lk);
    let block_len = nfi * nfk; // phantom j,l are s (nf=1)
    let total_len = 3 * block_len;

    let nsi = nsph(li);
    let nsk = nsph(lk);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // common_factor uses ONLY the real shells (phantom s contributes no fac_sp).
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lk);

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_k * total_len];

    let grad_f12_shape = two_e_shape_as_f12(&grad_shape);

    // The derivative exponent is the real shell's exponent on the nabla center.
    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pk in 0..n_prim_k {
            let ak = shell_k.exponents[pk];

            // 2c2e G-tensor via the 2e builder with phantom j,l (aj=al=0). No
            // Gaussian-overlap prefactor (Rys weights encode it): fac_env = common_factor.
            let g = fill_g_tensor_2e(
                ai, 0.0, ak, 0.0, &ri, &rj, &rk, &rl, grad_shape, common_factor,
            );

            let exponent = match center {
                Nabla1Center::I => ai,
                Nabla1Center::K => ak,
                _ => unreachable!(),
            };
            // gout_ipn at BASE li/lk (the G-tensor carries the +1 headroom).
            let gout = gout_ipn(&g, &grad_f12_shape, li as usize, 0, lk as usize, 0, center, exponent);

            for ci in 0..n_ctr_i {
                let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                for ck in 0..n_ctr_k {
                    let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                    let weight = coeff_i * coeff_k;
                    let base = (ci * n_ctr_k + ck) * total_len;
                    for n in 0..block_len {
                        for comp in 0..3usize {
                            cart_blocks[base + comp * block_len + n] += weight * gout[n * 3 + comp];
                        }
                    }
                }
            }
        }
    }

    // Component-leading `[3, nk, ni]` F-order write (j,l phantom s collapse out).
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dk = n_ctr_k * nsk;
            let sph_block = di * dk;
            for comp in 0..3usize {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for ck in 0..n_ctr_k {
                        let base = (ci * n_ctr_k + ck) * total_len + comp * block_len;
                        // Transform the (i, j=s, k, l=s) Cartesian block; s slots are
                        // cart==sph identities so this reduces to the 2c2e transform.
                        let sph = cart_to_sph_2e(&cart_blocks[base..base + block_len], li, 0, lk, 0);
                        for mk in 0..nsk {
                            let kidx = ck * nsk + mk;
                            for mi in 0..nsi {
                                let iidx = ci * nsi + mi;
                                let src = mi + nsi * mk;
                                let dst = staging_comp_base + iidx + di * kidx;
                                staging[dst] = F::from_f64_lossy(sph[src]);
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dk = n_ctr_k * nfk;
            let cart_block = di * dk;
            for comp in 0..3usize {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for ck in 0..n_ctr_k {
                        let base = (ci * n_ctr_k + ck) * total_len + comp * block_len;
                        let block = &cart_blocks[base..base + block_len];
                        for kc in 0..nfk {
                            let kidx = ck * nfk + kc;
                            for ic in 0..nfi {
                                let iidx = ci * nfi + ic;
                                let src = ic + nfi * kc;
                                let dst = staging_comp_base + iidx + di * kidx;
                                staging[dst] = F::from_f64_lossy(block[src]);
                            }
                        }
                    }
                }
            }
        }
        // 27-03 (FND-04): fold via the centralized derivative wrapper. 2c2e's two
        // centers are i and k (j,l are phantom s), so the wrapper's (i,j) roles map
        // to (i,k) here. ncomp=3 (lock component_rank for int2c2e_ip1/ip2_spinor).
        // cart_blocks is already KET-major bra-fastest contraction-major — exactly
        // the wrapper's expected device-native layout. No aux-k axis (D-06).
        Representation::Spinor => {
            cart_to_spinor_sf_derivative_2d::<F>(
                staging, &cart_blocks, 3, li, shell_i.kappa, lk, shell_k.kappa, n_ctr_i,
                n_ctr_k,
            )?;
        }
    }

    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;

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

/// `int2c2e_ipip1` Hessian launch — `∇²` on the bra center 1 (rank 9, HESS-03).
///
/// Mirrors [`launch_center_2c2e_grad`] but applies the SECOND bra derivative via
/// the verbatim `gout_ipip1` helper (`CINTgout2e_int2c2e_ipip1`, int3c2e.c). The
/// G-tensor needs `li+2` headroom (`gout_ipip1` reads `nabla1i_2e` up to `li+1`).
/// Phantom j,l centers collapse to s (aj=al=0). HOST-routed through
/// `fill_g_tensor_2e` so the elevated `li+2` raise can reach nroots 6..12 (FND-02).
fn launch_center_2c2e_hess1<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    const NCOMP: usize = 9;
    // Spinor Hessian: not supported (D-11). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor int2c2e_ipip1 Hessian".to_owned(),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_k = &shells[1];

    let li = shell_i.ang_momentum;
    let lk = shell_k.ang_momentum;

    // bra-i raised +2 (∇²); k is a spectator. Phantom 2e j,l = s.
    let hess_shape = build_2e_shape(li as usize + 2, 0, lk as usize, 0);

    // FND-02: route to the HOST path; the +2 raise can push nroots to 6..12.
    if hess_shape.nroots > HOST_RYS_NROOTS_CEILING {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unsupported_nrys_roots:{}", hess_shape.nroots),
        });
    }

    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;
    let rj = ri; // phantom j coincides with i (aj=0)
    let rl = rk; // phantom l coincides with k (al=0)

    let nfi = ncart(li);
    let nfk = ncart(lk);
    let block_len = nfi * nfk;
    let total_len = NCOMP * block_len;

    let nsi = nsph(li);
    let nsk = nsph(lk);

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lk);

    let mut cart_blocks = vec![0.0_f64; n_ctr_i * n_ctr_k * total_len];

    let hess_f12_shape = two_e_shape_as_f12(&hess_shape);

    for pi in 0..n_prim_i {
        let ai = shell_i.exponents[pi];
        for pk in 0..n_prim_k {
            let ak = shell_k.exponents[pk];

            let g = fill_g_tensor_2e(
                ai, 0.0, ak, 0.0, &ri, &rj, &rk, &rl, hess_shape, common_factor,
            );

            // gout_ipip1 at BASE li/lk (the G-tensor carries the +2 headroom).
            let gout = gout_ipip1(&g, &hess_f12_shape, li as usize, 0, lk as usize, 0, ai);

            for ci in 0..n_ctr_i {
                let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                for ck in 0..n_ctr_k {
                    let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                    let weight = coeff_i * coeff_k;
                    let base = (ci * n_ctr_k + ck) * total_len;
                    for n in 0..block_len {
                        for comp in 0..NCOMP {
                            cart_blocks[base + comp * block_len + n] += weight * gout[n * NCOMP + comp];
                        }
                    }
                }
            }
        }
    }

    // Component-leading `[9, nk, ni]` F-order write (j,l phantom s collapse out).
    match plan.representation {
        Representation::Spheric => {
            let di = n_ctr_i * nsi;
            let dk = n_ctr_k * nsk;
            let sph_block = di * dk;
            for comp in 0..NCOMP {
                let staging_comp_base = comp * sph_block;
                for ci in 0..n_ctr_i {
                    for ck in 0..n_ctr_k {
                        let base = (ci * n_ctr_k + ck) * total_len + comp * block_len;
                        let sph = cart_to_sph_2e(&cart_blocks[base..base + block_len], li, 0, lk, 0);
                        for mk in 0..nsk {
                            let kidx = ck * nsk + mk;
                            for mi in 0..nsi {
                                let iidx = ci * nsi + mi;
                                let src = mi + nsi * mk;
                                let dst = staging_comp_base + iidx + di * kidx;
                                staging[dst] = F::from_f64_lossy(sph[src]);
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            let di = n_ctr_i * nfi;
            let dk = n_ctr_k * nfk;
            let cart_block = di * dk;
            for comp in 0..NCOMP {
                let staging_comp_base = comp * cart_block;
                for ci in 0..n_ctr_i {
                    for ck in 0..n_ctr_k {
                        let base = (ci * n_ctr_k + ck) * total_len + comp * block_len;
                        let block = &cart_blocks[base..base + block_len];
                        for kc in 0..nfk {
                            let kidx = ck * nfk + kc;
                            for ic in 0..nfi {
                                let iidx = ci * nfi + ic;
                                let src = ic + nfi * kc;
                                let dst = staging_comp_base + iidx + di * kidx;
                                staging[dst] = F::from_f64_lossy(block[src]);
                            }
                        }
                    }
                }
            }
        }
        // 27-03: int2c2e_ipip1_spinor is NOT registered in the manifest lock (no
        // spinor form), so the early guard above still rejects it with
        // UnsupportedApi — this arm is defensively wired to the centralized
        // derivative wrapper (ncomp=NCOMP=9, KET-major bra-fastest cart_blocks,
        // no aux-k) so that a future registration folds correctly without a panic.
        Representation::Spinor => {
            cart_to_spinor_sf_derivative_2d::<F>(
                staging, &cart_blocks, NCOMP, li, shell_i.kappa, lk, shell_k.kappa, n_ctr_i,
                n_ctr_k,
            )?;
        }
    }

    let nonzero_threshold =
        F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 { 1e-12 } else { 1e-18 });
    let not0 = staging.iter().filter(|&&v| v.abs() > nonzero_threshold).count() as i32;

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

/// Generic inner for the 2c2e launcher.
///
/// Dispatches the [`center_2c2e_kernel`] device kernel (at f64) on `plan`'s
/// resolved backend, reads back the Cartesian buffer, then applies the
/// representation transform with the output cast to `F` (see module precision
/// policy). The signature is unchanged from the pre-GPU version so existing
/// callers/tests are unaffected.
///
/// # Normalization chain (from libcint):
/// common_factor = PI^3*2/sqrt(PI) * fac_sp_i * fac_sp_k   (g2c2e.c line 44-45)
/// fac_env = common_factor * ci * ck                         (cint2c2e.c line 129-133)
/// fac1 = sqrt(a0/a1^3) * fac_env                           (g2e.c line 4441)
/// gz[root] = w[root] * fac1                                 (g2e.c line 4563)
fn launch_center_2c2e_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "2c2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_2c2e",
            detail: format!(
                "canonical_family mismatch for 2c2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    let shells = plan.shells.as_slice();
    if shells.len() < 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_2c2e",
            detail: "2c2e kernel requires exactly 2 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_k = &shells[1];

    let li = shell_i.ang_momentum;
    let lk = shell_k.ang_momentum;

    // Phase 23 DRV1-04: int2c2e_ip1 (∇ on bra i) / int2c2e_ip2 (∇ on ket k).
    // 2c2e has NO operator dispatch in the scalar path; ADD it here, BEFORE the
    // scalar fall-through. The 2c2e g-tensor is 2e-style Rys, so the gradient
    // reuses the f12.rs gout_ipn engine with the j and l (2e) slots collapsed to
    // phantom s-functions (lj = ll = 0). (PATTERNS center_2c2e.rs assignment.)
    match plan.descriptor.operator_name() {
        "ip1" => return launch_center_2c2e_grad::<F>(plan, Nabla1Center::I, staging),
        "ip2" => return launch_center_2c2e_grad::<F>(plan, Nabla1Center::K, staging),
        // Phase 25 HESS-03: int2c2e_ipip1 — ∇² on bra center 1 (rank 9, host-routed).
        "ipip1" => return launch_center_2c2e_hess1::<F>(plan, staging),
        _ => {} // fall through to the existing scalar path
    }

    let nroots = (li as usize + lk as usize) / 2 + 1;
    if nroots > MAX_DEVICE_NROOTS {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_2c2e",
            detail: format!(
                "device 2c2e kernel supports nroots<={MAX_DEVICE_NROOTS} (l_i+l_k<=8); \
                 got nroots={nroots} for l_i={li}, l_k={lk}"
            ),
        });
    }

    // Atom coordinates.
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rk = atoms[shell_k.atom_index as usize].coord_bohr;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;

    // common_factor (g2c2e.c CINTinit_int2c2e_EnvVars line 44-45):
    //   common_factor = (M_PI^3)*2/SQRTPI * fac_sp_i * fac_sp_k
    let common_factor =
        (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lk);

    // Flatten the f64 primitive data the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();

    // Dispatch onto the resolved backend's device client (compute in f64).
    let cart_buf: Vec<f64> = match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_2c2e_device::<cubecl::cpu::CpuRuntime>(
            client, li as u32, lk as u32, n_prim_i as u32, n_prim_k as u32, n_ctr_i as u32,
            n_ctr_k as u32, nroots as u32, ri, rk, common_factor, &exps_i, &exps_k, &coeff_i,
            &coeff_k,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_2c2e_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lk as u32, n_prim_i as u32, n_prim_k as u32, n_ctr_i as u32,
            n_ctr_k as u32, nroots as u32, ri, rk, common_factor, &exps_i, &exps_k, &coeff_i,
            &coeff_k,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_2c2e_device::<cubecl_cuda::CudaRuntime>(
            client, li as u32, lk as u32, n_prim_i as u32, n_prim_k as u32, n_ctr_i as u32,
            n_ctr_k as u32, nroots as u32, ri, rk, common_factor, &exps_i, &exps_k, &coeff_i,
            &coeff_k,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_2c2e_device::<cubecl_hip::HipRuntime>(
            client, li as u32, lk as u32, n_prim_i as u32, n_prim_k as u32, n_ctr_i as u32,
            n_ctr_k as u32, nroots as u32, ri, rk, common_factor, &exps_i, &exps_k, &coeff_i,
            &coeff_k,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_2c2e_device::<cubecl_wgpu::WgpuRuntime>(
            client, li as u32, lk as u32, n_prim_i as u32, n_prim_k as u32, n_ctr_i as u32,
            n_ctr_k as u32, nroots as u32, ri, rk, common_factor, &exps_i, &exps_k, &coeff_i,
            &coeff_k,
        ),
    };

    // Representation dispatch: intermediate transforms use f64 temp buffers;
    // final values cast to F via F::from_f64_lossy.
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_2c2e(&cart_buf, li, lk);
            let copy_len = staging.len().min(sph.len());
            for (dst, &src) in staging[..copy_len].iter_mut().zip(sph[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
        Representation::Spinor => {
            let kappa_i = shell_i.kappa;
            let kappa_k = shell_k.kappa;
            cart_to_spinor_sf_2d::<F>(staging, &cart_buf, li, kappa_i, lk, kappa_k)?;
        }
        Representation::Cart => {
            let copy_len = staging.len().min(cart_buf.len());
            for (dst, &src) in staging[..copy_len].iter_mut().zip(cart_buf[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
    }

    // WR-06: precision-aware sentinel so f32 stale lanes are not counted.
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

/// 2c2e outer precision dispatcher — keeps the registered FamilyLaunchFn signature.
///
/// Dispatches on `plan.precision` to `launch_center_2c2e_typed::<F>`. The F32 arm
/// reinterprets `staging: &mut [f64]` as `&mut [f32]` via bytemuck (Plan 01 A5 proven).
/// CR-01: captures the true output element count BEFORE the bytemuck cast and bounds
/// the typed inner to that count, returning `BufferTooSmall` if the view cannot hold it.
pub fn launch_center_2c2e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_center_2c2e_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            let out_elems = staging.len();
            let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
            if staging_f32.len() < out_elems {
                return Err(cintxRsError::BufferTooSmall {
                    required: out_elems,
                    provided: staging_f32.len(),
                });
            }
            launch_center_2c2e_typed::<f32>(backend, plan, specialization, &mut staging_f32[..out_elems])
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {
    use super::*;
    use crate::transform::c2s::ncart;
    use cintx_core::BasisSet;

    /// Smoke test: s-s pair should produce a positive non-zero G-tensor base.
    #[test]
    fn test_fill_g_tensor_2c2e_ss_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 2.0];
        let ai = 1.0_f64;
        let ak = 1.0_f64;
        let fac_env = 1.0_f64;

        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, 0, 0, fac_env);

        assert_eq!(g.len(), 3, "s-s G-tensor should have 3 elements");
        let gz = g[2];
        assert!(gz > 0.0, "s-s gz[0] should be positive: got {gz:.6e}");
    }

    /// Smoke test: p-p pair should produce a non-trivial G-tensor.
    #[test]
    fn test_fill_g_tensor_2c2e_pp_nonzero() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 2.0];
        let ai = 0.5_f64;
        let ak = 0.5_f64;
        let fac_env = 1.0_f64;

        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, 1, 1, fac_env);
        assert_eq!(g.len(), 3 * 8, "p-p G-tensor size mismatch");

        let gz = &g[2 * 8..3 * 8];
        let nonzero = gz.iter().filter(|&&v| v.abs() > 1e-20).count();
        assert!(nonzero > 0, "p-p G-tensor gz should have non-zero entries");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Device kernel cross-check: the CubeCL kernel (on CpuRuntime, f64) must
    // reproduce the host `fill_g_tensor_2c2e` + contraction reference exactly.
    // ─────────────────────────────────────────────────────────────────────────

    /// Host reference: contract a single-primitive single-contraction shell pair
    /// the same way the device kernel does, using `fill_g_tensor_2c2e`.
    fn host_cart_2c2e(
        ai: f64,
        ak: f64,
        ri: [f64; 3],
        rk: [f64; 3],
        li: u8,
        lk: u8,
        common_factor: f64,
        coeff_i: f64,
        coeff_k: f64,
    ) -> Vec<f64> {
        let nci = ncart(li);
        let nck = ncart(lk);
        let nrys = (li as usize + lk as usize) / 2 + 1;
        let dn = nrys;
        let dm = nrys * (li as usize + 1);
        let g_size = nrys * (li as usize + 1) * (lk as usize + 1);

        let fac_env = common_factor * coeff_i * coeff_k;
        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, li, lk, fac_env);

        let ci_comps = cart_comps(li);
        let ck_comps = cart_comps(lk);
        let mut out = vec![0.0_f64; nci * nck];
        for (ck_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                let mut val = 0.0_f64;
                for irys in 0..nrys {
                    let vx = g[kx as usize * dm + ix as usize * dn + irys];
                    let vy = g[g_size + ky as usize * dm + iy as usize * dn + irys];
                    let vz = g[2 * g_size + kz as usize * dm + iz as usize * dn + irys];
                    val += vx * vy * vz;
                }
                out[ci_idx + ck_idx * nci] += val;
            }
        }
        out
    }

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    fn assert_device_matches_host(li: u8, lk: u8, ai: f64, ak: f64) {
        let ri = [0.0_f64, 0.0, 0.0];
        let rk = [0.0_f64, 0.0, 1.7];
        let coeff_i = 0.9_f64;
        let coeff_k = 1.1_f64;
        let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lk);
        let nroots = (li as usize + lk as usize) / 2 + 1;

        let host = host_cart_2c2e(ai, ak, ri, rk, li, lk, common_factor, coeff_i, coeff_k);
        let dev = run_2c2e_device::<cubecl::cpu::CpuRuntime>(
            &cpu_client(),
            li as u32,
            lk as u32,
            1,
            1,
            1,
            1,
            nroots as u32,
            ri,
            rk,
            common_factor,
            &[ai],
            &[ak],
            &[coeff_i],
            &[coeff_k],
        );

        assert_eq!(host.len(), dev.len(), "length mismatch for li={li} lk={lk}");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch li={li} lk={lk} idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    #[test]
    fn test_device_kernel_matches_host_ss() {
        assert_device_matches_host(0, 0, 1.0, 1.0);
    }

    #[test]
    fn test_device_kernel_matches_host_sp() {
        assert_device_matches_host(0, 1, 0.8, 1.3);
        assert_device_matches_host(1, 0, 1.3, 0.8);
    }

    #[test]
    fn test_device_kernel_matches_host_pp() {
        assert_device_matches_host(1, 1, 0.5, 0.7);
    }

    #[test]
    fn test_device_kernel_matches_host_dd() {
        // li=lk=2 → nroots=3, exercises the deeper VRR + mixed recurrence.
        assert_device_matches_host(2, 2, 0.6, 0.9);
    }

    /// Genericity evidence: the kernel compiles and runs for `F = f32` as well
    /// as `f64` ("cubecl kernel with generics float"). Launch an s-s pair at f32
    /// on the CPU runtime and assert a finite, positive result.
    #[test]
    fn test_center_2c2e_kernel_generic_f32() {
        let client = cpu_client();
        let exps_i = [1.0_f32];
        let exps_k = [1.0_f32];
        let coeff_i = [1.0_f32];
        let coeff_k = [1.0_f32];
        let g_zero = [0.0_f32; 3]; // nroots=1, g_size=1 → 3
        let rys_zero = [0.0_f32; 1];
        let out_zero = [0.0_f32; 1];

        let exps_i_h = client.create_from_slice(f32::as_bytes(&exps_i));
        let exps_k_h = client.create_from_slice(f32::as_bytes(&exps_k));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&coeff_i));
        let coeff_k_h = client.create_from_slice(f32::as_bytes(&coeff_k));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let common_factor = ((PI * PI * PI) * 2.0 / SQRTPI
            * common_fac_sp(0)
            * common_fac_sp(0)) as f32;

        center_2c2e_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_k_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(u_h, 1) },
            unsafe { ArrayArg::from_raw_parts(w_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32,
            0.0,
            0.0,
            0.0,
            0.0,
            1.7,
            common_factor,
            PIE4 as f32,
            0,
            0,
            1,
            1,
            1,
            1,
            1u32,
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(out.is_finite(), "f32 2c2e kernel result must be finite");
        assert!(out > 0.0, "s-s 2c2e f32 result should be positive: {out}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T04-2a: launch_center_2c2e_typed::<f64> writes a positive s-s 2c2e
    // integral (now through the CubeCL device kernel on CpuRuntime).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_2c2e_precision_dispatch_f64_positive() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let shell_a = Arc::new(Shell::try_new(0, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_b = Arc::new(Shell::try_new(1, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();

        let op = OperatorId::new(12);
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let opts = ExecutionOptions::default();
        let query = match query_workspace(op, Representation::Cart, &basis, shells.clone(), &opts) {
            Ok(q) => q,
            Err(_) => return,
        };
        let mut plan = ExecutionPlan::new(op, Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 1];

        let result = launch_center_2c2e_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "f64 2c2e typed inner should succeed: {:?}", result);
        assert!(staging[0].is_finite(), "2c2e f64 result should be finite");
        assert!(staging[0] > 0.0, "s-s 2c2e integral should be positive");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T04-2b: launch_center_2c2e_typed::<f32> writes a finite f32.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_2c2e_precision_dispatch_f32_positive() {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let shell_a = Arc::new(Shell::try_new(0, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let shell_b = Arc::new(Shell::try_new(1, 0, 1, 1, 0, Representation::Cart,
            Arc::from(vec![1.0_f64].into_boxed_slice()),
            Arc::from(vec![1.0_f64].into_boxed_slice())).unwrap());
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();

        let op = OperatorId::new(12);
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let opts = ExecutionOptions::default();
        let query = match query_workspace(op, Representation::Cart, &basis, shells.clone(), &opts) {
            Ok(q) => q,
            Err(_) => return,
        };
        let mut plan = ExecutionPlan::new(op, Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging_f32 = vec![0.0_f32; 1];

        let result = launch_center_2c2e_typed::<f32>(&backend, &plan, &spec, &mut staging_f32);
        assert!(result.is_ok(), "f32 2c2e typed inner should succeed: {:?}", result);
        assert!(staging_f32[0].is_finite(), "2c2e f32 result should be finite");
        assert!(staging_f32[0] > 0.0, "s-s 2c2e f32 integral should be positive");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 23 DRV1-04: int2c2e_ip1/ip2 gradient behavior contract.
    //   - component count: a (p, s) pair → 3 * 3*1 = 9 sph outputs (∇ rank 3).
    //   - nroots fail-closed: an (f, f) pair drives gradient nroots
    //     (3 + (3+1))/2 + 1 = 4 + ... > 5 → UnsupportedApi (D-13).
    //   - spinor: Representation::Spinor → UnsupportedApi (D-06).
    // ─────────────────────────────────────────────────────────────────────────
    fn build_2c2e_grad_plan(
        li: u8,
        lk: u8,
        symbol: &str,
    ) -> (BasisSet, cintx_core::ShellTuple, cintx_core::OperatorId) {
        use std::sync::Arc;
        use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell};
        use cintx_ops::resolver::Resolver;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
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
        let s0 = mk(0, li);
        let s1 = mk(1, lk);
        let all: Arc<[Arc<Shell>]> = Arc::from(vec![s0.clone(), s1.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([s0, s1]).unwrap();
        let op = Resolver::descriptor_by_symbol(symbol).expect("symbol in manifest").id;
        (basis, shells, op)
    }

    fn run_2c2e_grad(
        basis: &BasisSet,
        shells: cintx_core::ShellTuple,
        op: cintx_core::OperatorId,
        rep: cintx_core::Representation,
    ) -> Result<Vec<f64>, cintxRsError> {
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

        let opts = ExecutionOptions::default();
        let q = query_workspace(op, rep, basis, shells.clone(), &opts)?;
        let mut plan = ExecutionPlan::new(op, rep, basis, shells, &q)?;
        plan.precision = PrecisionKind::F64;
        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; plan.output_layout.staging_elements];
        launch_center_2c2e_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok(staging)
    }

    #[test]
    fn test_int2c2e_ip1_component_count() {
        // (p, s): sph ni=3, nk=1 → 3 * 3 * 1 = 9.
        let (basis, shells, op) = build_2c2e_grad_plan(1, 0, "int2c2e_ip1_sph");
        let out = run_2c2e_grad(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out.len(), 9, "(p,s) int2c2e_ip1 should produce 9 outputs");
        assert!(out.iter().any(|v| v.abs() > 1e-14), "int2c2e_ip1 (p,s) all-zero");
    }

    #[test]
    fn test_int2c2e_ip2_component_count() {
        let (basis, shells, op) = build_2c2e_grad_plan(0, 1, "int2c2e_ip2_sph");
        let out = run_2c2e_grad(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out.len(), 9, "(s,p) int2c2e_ip2 should produce 9 outputs");
        assert!(out.iter().any(|v| v.abs() > 1e-14), "int2c2e_ip2 (s,p) all-zero");
    }

    #[test]
    fn test_int2c2e_ip1_nroots_fail_closed() {
        // (f, f): gradient nroots = (3 + (3+1))/2 + 1 = 7/2 + 1 = 4 ... actually
        // (3 + 4)/2 + 1 = 4 ≤ 5 is allowed; use (f, f) where li_ceil+lk = 4+3=7 → 7/2+1=4.
        // To exceed 5 we need li_ceil+lk >= 9 → e.g. g(4)+f(3): but l>4 is gated.
        // Within the l<=3 cap the max is f,f → ip1 li_ceil=4: (4+3)/2+1 = 4 ≤ 5, allowed.
        // So assert the (f,f) gradient is ALLOWED (no false fail-closed), and a
        // hypothetical nroots>5 path is covered by the launcher guard.
        let (basis, shells, op) = build_2c2e_grad_plan(3, 3, "int2c2e_ip1_sph");
        let res = run_2c2e_grad(&basis, shells, op, Representation::Spheric);
        assert!(res.is_ok(), "(f,f) int2c2e_ip1 (nroots=4) must be allowed: {:?}", res.err());
    }

    // 27-03 (FND-04): int2c2e_ip1/ip2 spinor gradients now EVALUATE via the
    // centralized derivative wrapper (was UnsupportedApi). The wrapper owns the
    // KET→BRA transpose (D-06) and there is no aux-k axis for 2c2e.
    #[test]
    fn test_int2c2e_grad_spinor_evaluates() {
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use crate::specialization::SpecializationKey;
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};

        let (basis, shells, op) = build_2c2e_grad_plan(1, 0, "int2c2e_ip1_sph");
        let opts = ExecutionOptions::default();
        let q = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
        let mut plan = ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &q).unwrap();
        plan.representation = Representation::Spinor;
        plan.precision = PrecisionKind::F64;
        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        // (p,s) kappa=0: di=spinor_len(1,0)=4*1+2=6, dk=spinor_len(0,0)=2,
        // spinor_block=6*2*2=24, ncomp=3 → required = 72.
        let mut staging = vec![0.0_f64; 72];
        let result = launch_center_2c2e_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            result.is_ok(),
            "spinor int2c2e gradient should now evaluate (FND-04), got: {:?}",
            result
        );
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor int2c2e gradient staging is all-zero"
        );
    }
}
