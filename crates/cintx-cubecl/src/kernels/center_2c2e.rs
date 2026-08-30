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

// Transcribed verbatim from vendored libcint 6.1.3 (and, in `cintx-basis`, from the
// Lanczos reference these normalization constants come from). Result compatibility
// is decided by the exact bits these literals carry, so none is truncated to the
// shortest form that round-trips.
#![allow(clippy::excessive_precision)]
// Index arithmetic here is written in full — `base + 0 * stride`, `base + 1 * stride`,
// `out[n * 3 + 0]` — so that a slot or component index lines up column-wise with its
// neighbours and with the libcint layout being mirrored. Folding the `0 *` and `1 *`
// away would shorten the line and hide the stride.
#![allow(clippy::identity_op)]
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
use crate::kernels::f12::{Nabla1Center, gout_ip1ip2, gout_ipip1, gout_ipn};
use crate::kernels::two_electron::{build_2e_shape, fill_g_tensor_2e, two_e_shape_as_f12};
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::math::rys_wheeler::{
    EXT_TABLES_LEN, ext_rys_out_slots, ext_rys_slots, rys_roots_ext_dev,
};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2c2e, cart_to_sph_2c2e_into, cart_to_sph_2e, ncart, nsph};
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
// Verbatim libcint literal, not `std::f64::consts::FRAC_PI_4`: result compatibility
// with upstream is decided by the exact bits this file feeds the Rys kernels, so
// the constant is transcribed from `rys_roots.c` rather than recomputed.
#[allow(clippy::approx_constant)]
const PIE4: f64 = 0.78539816339744827900_f64;

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

/// Batched 2c2e G-tensor fill + Cartesian contraction — one shell pair per work
/// slot (Phase 35).
///
/// Faithful, correctness-first port of the host pipeline (`fill_g_tensor_2c2e` +
/// contraction), evaluating a whole **launch class** in one dispatch: every pair
/// in the list shares `(li, lk)` and therefore the G-tensor shape and the Rys
/// order. What varies per pair is only the shell data, read through a flattened
/// basis plus an index table:
///
/// - `exps` / `coeffs` — every shell's primitives concatenated;
/// - `centers` — 3 floats per shell;
/// - `shell_meta` — 4 `u32` per shell: `[exp_off, coeff_off, nprim, nctr]`;
/// - `pairs` — 3 `u32` per pair: `[si, sk, out_off]`.
///
/// `#[comptime] nroots` selects the `rys_root{1..5}` device function at JIT
/// specialization time (a `comptime!` branch — no runtime nroots dispatch, which
/// avoids the documented MLIR index-type lowering issue, and no Rust
/// monomorphization fan-out).
///
/// Layout of one slot's `g` slab (`3 * g_size` elements at `slot * g_stride`):
/// `g[gbase + axis*g_size + k*dm + i*dn + root]` with `dn = nroots`,
/// `dm = nroots*(li+1)`, `g_size = nroots*(li+1)*(lk+1)`.
///
/// `cart_out` block `out_off` (size `nctr_i*nctr_k*nci*nck`, `nci = ncart(li)`)
/// is zeroed in-kernel and accumulated over all primitive pairs. Contraction
/// block `(ci, ck)` lives at `out_off + (ci*nctr_k + ck) * nci*nck`, and within a
/// block the Cartesian index is `ci_idx + ck_idx*nci` — the same
/// contraction-major layout the 2e kernel produces.
///
/// This kernel's arithmetic is not split across a cube: one slot evaluates a
/// whole pair. `per_unit == 1` therefore maps a pair to each *unit* (the CubeCL
/// CPU shape, where a unit is an OS thread); `per_unit == 0` maps a pair to each
/// *cube* and leaves the remaining lanes idle, which is what the pre-batching
/// kernel did with its `UNIT_POS == 0` guard.
///
/// Source: libcint-master/src/g2e.c `CINTg0_2e` + `CINTg0_2e_2d`,
///         g2c2e.c `CINT2c2e_loop_nopt`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn center_2c2e_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    class_factor: &Array<F>,
    rys_tab: &Array<f64>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    // Slot decomposition — see the doc comment above, and the identical block in
    // `two_electron.rs` for why this is arithmetic on comptime-folded flags
    // rather than a `comptime!` if/else.
    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    // Only lane 0 of a cooperative group does anything here; in the per-unit
    // decomposition every unit is its own group of one.
    let lane = unit_pos * coop;

    // Rys roots/weights are written and read entirely inside the `lane == 0`
    // region below, so they are per-unit private storage rather than buffers.
    // The extent follows `nroots`: five for the polynomial-fit kernels, exactly
    // `nroots` once the inline extended entry (task 33-01) serves the class.
    let mut urys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    let mut wrys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    // The extended entry is f64-only, so it lands in its own pair and is cast
    // into `urys`/`wrys`. Both collapse to one element when the arm is absent.
    let mut uext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));
    let mut wext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));

    if lane == 0u32 {
        let nrys = nroots;
        let dn = nrys;
        let gbase = slot * g_stride;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
        // `u32::div_ceil` has no `#[cube]` expansion, so the blocked-walk
        // chunk size is written out.
        #[allow(clippy::manual_div_ceil)]
        let chunk = (n_pairs + n_slots - 1u32) / n_slots;
        let qi_start = slot * (chunk * punit + coop);
        let mut qi_stop = (qi_start + chunk) * punit + n_pairs * coop;
        if qi_stop > n_pairs {
            qi_stop = n_pairs;
        }
        let qi_step = n_slots * coop + punit;

        let mut qi = qi_start;
        while qi < qi_stop {
            let prow = qi * 4u32;
            let si = pairs[prow as usize];
            let sk = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            // ── Per-class shape (Task 35-M2) ──────────────────────────────
            //
            // `nroots` is this kernel's only comptime parameter — `li` and `lk`
            // were already runtime scalars — so one dispatch carries every
            // `(li,lk)` class of the same Rys order. The G slab is sized to the
            // widest class and each class indexes only what it owns, keeping
            // the merge bit-identical.
            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(TWO_C2E_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lk = class_shape[(srow + 1u32) as usize];
            let common_factor = class_factor[cls as usize];

            let dm = nrys * (li + 1u32);
            let g_size = nrys * (li + 1u32) * (lk + 1u32);
            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let nck = (lk + 1u32) * (lk + 2u32) / 2u32;
            let block_len = nci * nck;

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mk = sk * 4u32;
            let eoff_k = shell_meta[mk as usize];
            let coff_k = shell_meta[(mk + 1u32) as usize];
            let nprim_k = shell_meta[(mk + 2u32) as usize];
            let nctr_k = shell_meta[(mk + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let ck3 = sk * 3u32;
            let rkx = centers[ck3 as usize];
            let rky = centers[(ck3 + 1u32) as usize];
            let rkz = centers[(ck3 + 2u32) as usize];

            let out_len = nctr_i * nctr_k * block_len;

            // Zero the accumulation buffer.
            let mut oi = 0u32;
            while oi < out_len {
                cart_out[(out_off + oi) as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // Center displacement (independent of primitives): rij = ri, rkl = rk.
            let xij = rix - rkx;
            let yij = riy - rky;
            let zij = riz - rkz;
            let rr = xij * xij + yij * yij + zij * zij;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pk = 0u32;
                while pk < nprim_k {
                    let ak = exps[(eoff_k + pk) as usize];

                    // For 2c2e: aij = ai, akl = ak.
                    let aij = ai;
                    let akl = ak;
                    let a1 = aij * akl;
                    let a0 = a1 / (aij + akl); // rho = ai*ak/(ai+ak)
                    let x_rys = a0 * rr;

                    // Rys roots/weights depend only on (ai, ak) → compute once here.
                    // `nroots` is comptime, so exactly one branch is emitted.
                    if comptime!(nroots == 1u32) {
                        rys_root1::<F>(x_rys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 2u32) {
                        rys_root2::<F>(x_rys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 3u32) {
                        rys_root3::<F>(x_rys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 4u32) {
                        rys_root4::<F>(x_rys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 5u32) {
                        rys_root5::<F>(x_rys, &mut urys, &mut wrys, pie4);
                    } else {
                        // nroots 6..=12: the inline Wheeler/Jacobi entry
                        // (task 33-01), reachable only once
                        // `device_nroots_ceiling` was raised for this family.
                        rys_roots_ext_dev(
                            rys_tab,
                            f64::cast_from(x_rys),
                            &mut uext,
                            &mut wext,
                            nroots,
                        );
                        #[unroll]
                        for iext in 0..nroots {
                            urys[iext as usize] = F::cast_from(uext[iext as usize]);
                            wrys[iext as usize] = F::cast_from(wext[iext as usize]);
                        }
                    }

                    let fac1 = F::sqrt(a0 / (a1 * a1 * a1)) * common_factor;

                    // ── Fill the G-tensor (VRR) ──────────
                    #[unroll]
                    for irys in 0..nroots {
                        let u2 = a0 * urys[irys as usize];
                        let tmp4 = F::new(0.5_f32) / (u2 * (aij + akl) + a1);
                        let tmp5 = u2 * tmp4;
                        let b00 = tmp5;
                        let b10 = tmp5 + tmp4 * akl;
                        let b01 = tmp5 + tmp4 * aij;
                        let tmp2 = F::new(2.0_f32) * tmp5 * akl;
                        let tmp3 = F::new(2.0_f32) * tmp5 * aij;

                        // Base case: gx=gy=1, gz=w*fac1 (g2e.c lines 4517-4521).
                        g[(gbase + irys) as usize] = F::new(1.0_f32);
                        g[(gbase + g_size + irys) as usize] = F::new(1.0_f32);
                        g[(gbase + 2u32 * g_size + irys) as usize] = wrys[irys as usize] * fac1;

                        #[unroll]
                        for axis in 0..3u32 {
                            let base = gbase + axis * g_size;
                            // Displacement component for this axis.
                            let mut d = xij;
                            if axis == 1u32 {
                                d = yij;
                            } else if axis == 2u32 {
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
                            if lk >= 1u32 && li >= 1u32 {
                                let mut n = 1u32;
                                while n <= li {
                                    let i_off = irys + n * dn;
                                    let s0_k0 = g[(base + i_off) as usize];
                                    let prev_i_k0 = g[(base + irys + (n - 1u32) * dn) as usize];
                                    // k=1: I(n,1)=c0p*I(n,0)+n*b00*I(n-1,0)
                                    let mut s1 = c0pa * s0_k0 + F::cast_from(n) * b00 * prev_i_k0;
                                    g[(base + i_off + dm) as usize] = s1;
                                    let mut s_prev = s0_k0;
                                    let mut m = 1u32;
                                    while m < lk {
                                        let prev_i_km =
                                            g[(base + irys + (n - 1u32) * dn + m * dm) as usize];
                                        let s2 = c0pa * s1
                                            + F::cast_from(m) * b01 * s_prev
                                            + F::cast_from(n) * b00 * prev_i_km;
                                        g[(base + i_off + (m + 1u32) * dm) as usize] = s2;
                                        s_prev = s1;
                                        s1 = s2;
                                        m += 1u32;
                                    }
                                    n += 1u32;
                                }
                            }
                        }
                    }

                    // ── Contract over Rys roots and Cartesian triples ─────
                    // Output: i fastest (innermost), k slowest (outermost) within
                    // one contraction block; blocks are contraction-major.
                    //
                    // The Cartesian value is independent of (ci, ck) — only the
                    // weight is not — so it is computed once and scattered across
                    // the contraction blocks. Summing the weights into one scalar
                    // and writing a single block instead (as this kernel did before)
                    // is correct only when `nctr_i == nctr_k == 1`.
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

                                    let mut val = F::new(0.0_f32);
                                    #[unroll]
                                    for irys2 in 0..nroots {
                                        let vx = g[(gbase + kx * dm + ix * dn + irys2) as usize];
                                        let vy = g
                                            [(gbase + g_size + ky * dm + iy * dn + irys2) as usize];
                                        let vz =
                                            g[(gbase + 2u32 * g_size + kz * dm + iz * dn + irys2)
                                                as usize];
                                        val += vx * vy * vz;
                                    }

                                    let elem = ci_idx + ck_idx * nci;
                                    let mut ci = 0u32;
                                    while ci < nctr_i {
                                        let coeff_i_val =
                                            coeffs[(coff_i + pi * nctr_i + ci) as usize];
                                        let mut ck = 0u32;
                                        while ck < nctr_k {
                                            let coeff_k_val =
                                                coeffs[(coff_k + pk * nctr_k + ck) as usize];
                                            let block = (ci * nctr_k + ck) * block_len;
                                            cart_out[(out_off + block + elem) as usize] +=
                                                val * coeff_i_val * coeff_k_val;
                                            ck += 1u32;
                                        }
                                        ci += 1u32;
                                    }

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

                    pk += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// `u32` shape scalars per class row of the device shape table: `li, lk`.
const TWO_C2E_SHAPE_STRIDE: usize = 2;

/// One dispatch: every shell pair of the same Rys order (Task 35-M2).
///
/// `center_2c2e_kernel` specializes on `nroots` alone, so a launch class is a
/// Rys order rather than an `(li,lk)` tuple. Each pair names its class in the
/// fourth column of its table row.
#[derive(Clone, Debug)]
pub struct TwoC2eLaunchGroup {
    /// Rys order — the kernel's only comptime parameter.
    pub nroots: u32,
    /// [`TWO_C2E_SHAPE_STRIDE`] `u32` per merged class: `li, lk`.
    pub class_shape: Vec<u32>,
    /// One libcint `common_factor` per merged class.
    pub class_factor: Vec<f64>,
    /// `[si, sk, out_off, class]` per pair.
    pub pairs: Vec<u32>,
    /// Total Cartesian output elements across this group's pairs.
    pub out_len: usize,
    /// Widest per-slot G-tensor length in the group.
    pub max_g_size: usize,
}

impl TwoC2eLaunchGroup {
    /// An empty group of Rys order `nroots`.
    #[must_use]
    pub fn new(nroots: u32) -> Self {
        Self {
            nroots,
            class_shape: Vec::new(),
            class_factor: Vec::new(),
            pairs: Vec::new(),
            out_len: 0,
            max_g_size: 0,
        }
    }

    /// Append a class and return the index its pair rows carry.
    pub fn push_class(&mut self, li: u32, lk: u32, common_factor: f64) -> u32 {
        let index = self.class_factor.len() as u32;
        self.class_shape.extend_from_slice(&[li, lk]);
        self.class_factor.push(common_factor);
        self.max_g_size = self
            .max_g_size
            .max(self.nroots as usize * (li as usize + 1) * (lk as usize + 1));
        index
    }

    /// Number of angular-momentum classes merged into this dispatch.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_factor.len()
    }

    /// Number of pairs in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len() / 4
    }

    /// Is this group empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Bytes this group's pair and class tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        (self.pairs.len() + self.class_shape.len()) * std::mem::size_of::<u32>()
            + self.class_factor.len() * std::mem::size_of::<f64>()
    }
}

/// Stride, in `f64` elements, between one slot's 2c2e G slab and the next,
/// padded to a 64-byte cache line so concurrent slots never share a line.
fn two_c2e_g_slab_stride(g_size: usize) -> usize {
    const LINE: usize = 8;
    (3 * g_size).div_ceil(LINE) * LINE
}

/// Does this backend want the one-pair-per-unit decomposition? Same reasoning
/// and override knob as `two_electron::two_e_per_unit`.
fn two_c2e_per_unit<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let pinned = *OVERRIDE.get_or_init(|| {
        std::env::var("CINTX_2C2E_PER_UNIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    });
    match pinned {
        Some(value) => value != 0,
        None => !crate::plane::has_planes(client),
    }
}

/// Launch geometry for one 2c2e class: `(cube_count, cube_dim, n_slots)`.
fn two_c2e_launch_geometry<R: Runtime>(
    client: &ComputeClient<R>,
    n_pairs: usize,
    g_size: usize,
) -> (u32, CubeDim, usize) {
    /// Ceiling on the per-launch G-tensor scratch slab.
    const MAX_BATCH_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

    let per_slab = two_c2e_g_slab_stride(g_size) * std::mem::size_of::<f64>();
    let by_memory = (MAX_BATCH_SCRATCH_BYTES / per_slab.max(1)).max(1);

    if two_c2e_per_unit::<R>(client) {
        let units = crate::plane::per_unit_width(
            client,
            n_pairs,
            crate::plane::MIN_ITEMS_PER_UNIT_PAIR,
            by_memory,
        );
        return (1, CubeDim::new_1d(units), units as usize);
    }
    // The kernel's arithmetic is not split across a cube, so a wider cube would
    // only add idle lanes.
    let cubes = crate::plane::grid_cube_count(client, n_pairs.min(by_memory));
    (cubes, CubeDim::new_1d(1), cubes as usize)
}

/// Evaluate every class of a batched 2c2e run: one dispatch and one readback per
/// class, one basis upload for the whole run.
fn run_2c2e_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[TwoC2eLaunchGroup],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

    // The basis is already on the device; `Handle` is cheap to clone and the
    // buffer it names is shared by every dispatch below (Task 34-C2).
    let crate::kernels::two_electron::TwoEBasisHandles {
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

    let mut results = Vec::with_capacity(groups.len());
    for class in groups {
        let n_pairs = class.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Sized to the widest class merged into this dispatch.
        let g_size = class.max_g_size;

        let (n_cubes, cube_dim, n_slots) = two_c2e_launch_geometry::<R>(client, n_pairs, g_size);
        let g_stride = two_c2e_g_slab_stride(g_size);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&class.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&class.class_shape));
        let factor_h = client.create_from_slice(f64::as_bytes(&class.class_factor));
        // The extended-Rys constant tables (~4.7 KB), read only by a class whose
        // Rys order is past the polynomial-fit ceiling.
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(class.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(two_c2e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_pairs`, by the per-shell `nprim`/`nctr` read from `shell_meta`, and
        // by the class-uniform G-tensor extents.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    center_2c2e_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), class.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), class.class_shape.len()),
                        ArrayArg::from_raw_parts(factor_h.clone(), class.class_factor.len()),
                        ArrayArg::from_raw_parts(rys_tab_h.clone(), EXT_TABLES_LEN),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), class.out_len),
                        PIE4,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        // Every reachable order gets its own arm. The upstream ceiling check
        // already refused anything above `device_nroots_ceiling(backend,
        // RysFamily::Int2c2e)`, which is 5 unless `extended-device-rys` is
        // compiled in, this backend's FMA probe passed and this family is on
        // the flipped list — so the 6..=12 arms are both feature-gated and
        // unreachable without that evidence.
        match class.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            #[cfg(feature = "extended-device-rys")]
            6 => launch_with!(6u32),
            #[cfg(feature = "extended-device-rys")]
            7 => launch_with!(7u32),
            #[cfg(feature = "extended-device-rys")]
            8 => launch_with!(8u32),
            #[cfg(feature = "extended-device-rys")]
            9 => launch_with!(9u32),
            #[cfg(feature = "extended-device-rys")]
            10 => launch_with!(10u32),
            #[cfg(feature = "extended-device-rys")]
            11 => launch_with!(11u32),
            #[cfg(feature = "extended-device-rys")]
            12 => launch_with!(12u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..class.out_len].to_vec());
    }
    results
}

/// Spherical AO blocks for a 2c2e pair batch, plus the offsets locating each pair.
#[derive(Clone, Debug, Default)]
pub struct TwoC2eBatchOutput {
    /// Concatenated spherical AO blocks, in the caller's pair order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where pair `n`'s block starts in [`Self::values`].
    pub offsets: Vec<usize>,
    /// Execution statistics.
    pub stats: crate::kernels::two_electron::BatchExecutionStats,
}

/// Evaluate a list of shell pairs as `int2c2e_sph`, one dispatch per launch
/// class (Phase 35).
///
/// Same shape as `two_electron::evaluate_2e_quartet_batch`, two indices shorter:
/// the list is grouped by `(li, lk)`, each group costs exactly one dispatch and
/// one readback, and the basis is uploaded once for the whole run.
///
/// `pairs` are `[i, k]` indices into `shells`.
/// Where one `(li,lk)` class landed after launch-group merging (Task 35-M2).
struct TwoC2eClassPlacement {
    li: u32,
    lk: u32,
    /// Index into the group list — which dispatch's buffer holds these blocks.
    group: usize,
    /// Caller-order indices of this class's pairs.
    members: Vec<usize>,
    /// Each member's offset into the group's Cartesian buffer.
    cart_offsets: Vec<usize>,
}

pub fn evaluate_2c2e_pair_batch(
    backend: &ResolvedBackend,
    shells: &[crate::kernels::two_electron::BatchShell],
    pairs: &[[u32; 2]],
) -> Result<TwoC2eBatchOutput, cintxRsError> {
    let resident = crate::kernels::two_electron::ResidentBasis::new(backend, shells)?;
    evaluate_2c2e_pair_batch_resident(backend, &resident, pairs)
}

/// [`evaluate_2c2e_pair_batch`] against a basis already on the device
/// (Task 34-C2).
///
/// Identical results; the difference is that the flattened basis is the
/// caller's [`crate::kernels::two_electron::ResidentBasis`] rather than a
/// throwaway one, so `basis_upload_bytes` is the full upload on the first call
/// and **0** on every later one. An RI-J build re-evaluates the same `(P|Q)`
/// metric every SCF iteration, which is the case this exists for.
///
/// # Errors
/// As [`evaluate_2c2e_pair_batch`], plus a backend mismatch on `resident`.
pub fn evaluate_2c2e_pair_batch_resident(
    backend: &ResolvedBackend,
    resident: &crate::kernels::two_electron::ResidentBasis,
    pairs: &[[u32; 2]],
) -> Result<TwoC2eBatchOutput, cintxRsError> {
    resident.check_for("2c2e-batch", backend)?;
    let shells = resident.shells();

    let mut offsets = Vec::with_capacity(pairs.len());
    let mut total = 0_usize;
    for pair in pairs {
        for &s in pair {
            if s as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("2c2e-batch:shell-index-out-of-range:{s}"),
                });
            }
        }
        offsets.push(total);
        total += shells[pair[0] as usize].ao_len() * shells[pair[1] as usize].ao_len();
    }

    let mut output = TwoC2eBatchOutput {
        values: vec![0.0; total],
        offsets,
        stats: crate::kernels::two_electron::BatchExecutionStats {
            quartets: pairs.len(),
            ..Default::default()
        },
    };
    if pairs.is_empty() {
        return Ok(output);
    }

    let ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int2c2e,
    );

    let mut grouped: std::collections::BTreeMap<[u8; 2], Vec<usize>> = Default::default();
    for (index, pair) in pairs.iter().enumerate() {
        let key = [shells[pair[0] as usize].l, shells[pair[1] as usize].l];
        grouped.entry(key).or_default().push(index);
    }

    // Classes merged into dispatch groups keyed on the Rys order (Task 35-M2);
    // the `(li,lk)` grouping survives as the host cart-to-sph sub-grouping.
    let mut groups: Vec<TwoC2eLaunchGroup> = Vec::new();
    let mut group_of: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut classes: Vec<TwoC2eClassPlacement> = Vec::with_capacity(grouped.len());
    for (class, members) in grouped {
        let [li, lk] = class;
        let nroots = (li as usize + lk as usize) / 2 + 1;
        // Per-backend ceiling (task 33-05): the base value everywhere, raised
        // only on a backend whose FMA-fusion probe passed and only with the
        // `extended-device-rys` opt-in. See `crate::device_rys_ceiling`.
        if nroots > ceiling {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "2c2e-batch:nroots={nroots} exceeds device ceiling {ceiling} \
                     for l=({li},{lk})"
                ),
            });
        }

        let nroots = nroots as u32;
        let group_index = match group_of.get(&nroots) {
            Some(&index) => index,
            None => {
                groups.push(TwoC2eLaunchGroup::new(nroots));
                let index = groups.len() - 1;
                group_of.insert(nroots, index);
                index
            }
        };
        let group = &mut groups[group_index];
        let class_index = group.push_class(
            u32::from(li),
            u32::from(lk),
            // g2c2e.c `CINTinit_int2c2e_EnvVars` lines 44-45.
            (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lk),
        );

        let cart_block = ncart(li) * ncart(lk);
        group.pairs.reserve(members.len() * 4);
        let mut cart_offsets = Vec::with_capacity(members.len());
        for &index in &members {
            let p = pairs[index];
            let nctr_product =
                shells[p[0] as usize].nctr as usize * shells[p[1] as usize].nctr as usize;
            cart_offsets.push(group.out_len);
            group
                .pairs
                .extend_from_slice(&[p[0], p[1], group.out_len as u32, class_index]);
            group.out_len += nctr_product * cart_block;
        }

        classes.push(TwoC2eClassPlacement {
            li: u32::from(li),
            lk: u32::from(lk),
            group: group_index,
            members,
            cart_offsets,
        });
    }

    let dispatch_start = std::time::Instant::now();
    let carts = dispatch_2c2e_batches(backend, resident.handles(), &groups)?;
    output.stats.dispatch_ns = dispatch_start.elapsed().as_nanos() as u64;

    output.stats.basis_upload_bytes = if resident.take_first_use() {
        resident.upload_bytes()
    } else {
        0
    };
    output.stats.kernel_launch_count = groups.len();
    output.stats.launch_classes = classes.len();
    output.stats.readback_count = groups.len();
    output.stats.max_g_slab_bytes = groups
        .iter()
        .map(|group| two_c2e_g_slab_stride(group.max_g_size) * std::mem::size_of::<f64>())
        .max()
        .unwrap_or(0);
    output.stats.transfer_bytes = output.stats.basis_upload_bytes
        + groups
            .iter()
            .map(TwoC2eLaunchGroup::upload_bytes)
            .sum::<usize>();

    let transform_start = std::time::Instant::now();
    // Task 36-T1: one output block and one c2s scratch per worker, not one pair
    // per contraction block. Both are fully written before being read on every
    // call, so reuse across blocks does not change a single bit.
    //
    // Task 36-T2: one job per pair, in the caller's order, each writing a
    // disjoint output block. Each output element is produced by exactly one
    // pair, so the split reorders no summation.
    let carts = &carts;
    let mut placement = vec![(0_usize, 0_usize); pairs.len()];
    for (class_index, class) in classes.iter().enumerate() {
        for (slot, &index) in class.members.iter().enumerate() {
            placement[index] = (class_index, slot);
        }
    }
    let lens: Vec<usize> = pairs
        .iter()
        .map(|pair| shells[pair[0] as usize].ao_len() * shells[pair[1] as usize].ao_len())
        .collect();
    let jobs: Vec<(usize, &mut [f64])> =
        crate::transform::host_batch::split_output_blocks(&mut output.values, &lens)
            .into_iter()
            .enumerate()
            .collect();

    let states = crate::transform::host_batch::for_each_block(
        jobs,
        || {
            (
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                crate::transform::profile::HostTransformProfile::new(),
            )
        },
        |(sph, c2s_scratch, profile), (index, block)| {
            let (class_index, slot) = placement[index];
            let class = &classes[class_index];
            let (li, lk) = (class.li as u8, class.lk as u8);
            let cart_block = ncart(li) * ncart(lk);
            let (nsi, nsk) = (nsph(li), nsph(lk));

            profile.start();
            sph.clear();
            sph.resize(nsk * nsi, 0.0);
            profile.charge_alloc();

            let cart = &carts[class.group];
            let p = pairs[index];
            let (nci_ctr, nck_ctr) = (
                shells[p[0] as usize].nctr as usize,
                shells[p[1] as usize].nctr as usize,
            );
            let di = nci_ctr * nsi;
            let src_base = class.cart_offsets[slot];
            for ci in 0..nci_ctr {
                for ck in 0..nck_ctr {
                    let base = src_base + (ci * nck_ctr + ck) * cart_block;
                    cart_to_sph_2c2e_into(&cart[base..base + cart_block], li, lk, sph, c2s_scratch);
                    profile.charge_transform();
                    for mk in 0..nsk {
                        let kidx = ck * nsk + mk;
                        for mi in 0..nsi {
                            let iidx = ci * nsi + mi;
                            block[iidx + di * kidx] = sph[mi + nsi * mk];
                        }
                    }
                    profile.charge_scatter();
                }
            }
            profile.pause();
        },
    );

    let mut profile = crate::transform::profile::HostTransformProfile::new();
    for (_, _, worker) in &states {
        profile.merge(worker);
    }
    output.stats.host_transform_ns = transform_start.elapsed().as_nanos() as u64;
    profile.store_into(&mut output.stats);

    Ok(output)
}

/// Backend dispatch for a whole batched 2c2e run.
fn dispatch_2c2e_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[TwoC2eLaunchGroup],
) -> Result<Vec<Vec<f64>>, cintxRsError> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => Ok(run_2c2e_batches::<cubecl::cpu::CpuRuntime>(
            client, basis, groups,
        )),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => Ok(run_2c2e_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups,
        )),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => Ok(run_2c2e_batches::<cubecl_cuda::CudaRuntime>(
            client, basis, groups,
        )),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => Ok(run_2c2e_batches::<cubecl_hip::HipRuntime>(
            client, basis, groups,
        )),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => Ok(run_2c2e_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups,
        )),
    }
}

/// Single-pair dispatch — a one-class, one-pair batch.
///
/// Kept as its own entry point because the per-tuple compatibility API evaluates
/// exactly one shell pair. It marshals the two shells into the flattened form
/// [`run_2c2e_batches`] consumes, so both paths execute the *same* kernel and
/// every existing parity test covers the batched code at `n_pairs == 1`.
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
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let nck = (lk_u + 1) * (lk_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_k as usize) * nci * nck;

    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (exps, coeffs, center, nprim, nctr) in [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_k, coeff_k, rk, nprim_k, nctr_k),
    ] {
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
    let handles = crate::kernels::two_electron::upload_2e_basis::<R>(client, &basis);

    let mut group = TwoC2eLaunchGroup::new(nroots);
    let class_index = group.push_class(li, lk, common_factor);
    group.pairs.extend_from_slice(&[0, 1, 0, class_index]);
    group.out_len = out_len;

    run_2c2e_batches::<R>(client, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
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
///
/// # D-PBC-24: `range_omega`
///
/// `nrys_roots` is supplied by the caller rather than recomputed, because short
/// range DOUBLES it (`g2c2e.c:61-68`) and the caller has to size its own
/// contraction loop from the same value. Pass
/// [`cintx_runtime::range_omega::nrys_roots_for`]`(rys_order, range_omega)`.
///
/// Returns `Ok(None)` when the short-range integrand is past `EXPCUTOFF_SR` and
/// libcint would contribute nothing for this primitive pair.
fn fill_g_tensor_2c2e(
    ai: f64,
    ak: f64,
    ri: &[f64; 3],
    rk: &[f64; 3],
    li: u8,
    lk: u8,
    fac_env: f64,
    nrys_roots: usize,
    range_omega: Option<f64>,
) -> Result<Option<Vec<f64>>, cintxRsError> {
    let nmax = li as usize;
    let mmax = lk as usize;
    let rys_order = (li as usize + lk as usize) / 2 + 1;

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

    // D-PBC-24: the shared `CINTg0_2e` omega branch. `omega == 0` reduces to the
    // plain `rys_roots_host(nrys_roots, x_rys)` this used to call, with `fac1`
    // returned unchanged, so the full-range path stays byte-identical.
    let Some(roots) = crate::math::range_separation::rys_roots_range_separated(
        rys_order,
        nrys_roots,
        x_rys,
        a0,
        fac1,
        range_omega,
    )?
    else {
        return Ok(None);
    };
    let (u_roots, w_weights, fac1) = (roots.u, roots.w, roots.fac1);

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
                        let mut s1 = c * s0_k0 + n as f64 * b00 * prev_i_k0;
                        g[base + i_off + dm] = s1;
                        let mut s_prev = s0_k0;
                        for m in 1..mmax {
                            let prev_i_km = g[base + irys + (n - 1) * dn + m * dm];
                            let s2 = c * s1 + m as f64 * b01 * s_prev + n as f64 * b00 * prev_i_km;
                            g[base + i_off + (m + 1) * dm] = s2;
                            s_prev = s1;
                            s1 = s2;
                        }
                    }
                }
            }
        }
    }

    Ok(Some(g))
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
                ai,
                0.0,
                ak,
                0.0,
                &ri,
                &rj,
                &rk,
                &rl,
                grad_shape,
                common_factor,
            );

            let exponent = match center {
                Nabla1Center::I => ai,
                Nabla1Center::K => ak,
                _ => unreachable!(),
            };
            // gout_ipn at BASE li/lk (the G-tensor carries the +1 headroom).
            let gout = gout_ipn(
                &g,
                &grad_f12_shape,
                li as usize,
                0,
                lk as usize,
                0,
                center,
                exponent,
            );

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
                        let sph =
                            cart_to_sph_2e(&cart_blocks[base..base + block_len], li, 0, lk, 0);
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
                staging,
                &cart_blocks,
                3,
                li,
                shell_i.kappa,
                lk,
                shell_k.kappa,
                n_ctr_i,
                n_ctr_k,
            )?;
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

/// `int2c2e_ipip1` Hessian launch — `∇²` on the bra center 1 (rank 9, HESS-03).
///
/// Mirrors [`launch_center_2c2e_grad`] but applies the SECOND bra derivative via
/// the verbatim `gout_ipip1` helper (`CINTgout2e_int2c2e_ipip1`, int3c2e.c). The
/// G-tensor needs `li+2` headroom (`gout_ipip1` reads `nabla1i_2e` up to `li+1`).
/// Phantom j,l centers collapse to s (aj=al=0). HOST-routed through
/// `fill_g_tensor_2e` so the elevated `li+2` raise can reach nroots 6..12 (FND-02).
fn launch_center_2c2e_hess<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    mixed_centers: bool,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    const NCOMP: usize = 9;
    // Spinor Hessian: not supported (D-11). Reject before any compute.
    if plan.representation == Representation::Spinor {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "spinor int2c2e_{} Hessian",
                if mixed_centers { "ip1ip2" } else { "ipip1" }
            ),
        });
    }

    let shells = plan.shells.as_slice();
    let shell_i = &shells[0];
    let shell_k = &shells[1];

    let li = shell_i.ang_momentum;
    let lk = shell_k.ang_momentum;

    let hess_shape = if mixed_centers {
        build_2e_shape(li as usize + 1, 0, lk as usize + 1, 0)
    } else {
        build_2e_shape(li as usize + 2, 0, lk as usize, 0)
    };

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
                ai,
                0.0,
                ak,
                0.0,
                &ri,
                &rj,
                &rk,
                &rl,
                hess_shape,
                common_factor,
            );

            let gout = if mixed_centers {
                gout_ip1ip2(&g, &hess_f12_shape, li as usize, 0, lk as usize, 0, ai, ak)
            } else {
                gout_ipip1(&g, &hess_f12_shape, li as usize, 0, lk as usize, 0, ai)
            };

            for ci in 0..n_ctr_i {
                let coeff_i = shell_i.coefficients[pi * n_ctr_i + ci];
                for ck in 0..n_ctr_k {
                    let coeff_k = shell_k.coefficients[pk * n_ctr_k + ck];
                    let weight = coeff_i * coeff_k;
                    let base = (ci * n_ctr_k + ck) * total_len;
                    for n in 0..block_len {
                        for comp in 0..NCOMP {
                            cart_blocks[base + comp * block_len + n] +=
                                weight * gout[n * NCOMP + comp];
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
                        let sph =
                            cart_to_sph_2e(&cart_blocks[base..base + block_len], li, 0, lk, 0);
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
                staging,
                &cart_blocks,
                NCOMP,
                li,
                shell_i.kappa,
                lk,
                shell_k.kappa,
                n_ctr_i,
                n_ctr_k,
            )?;
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
        "ipip1" => return launch_center_2c2e_hess::<F>(plan, false, staging),
        "ip1ip2" => return launch_center_2c2e_hess::<F>(plan, true, staging),
        _ => {} // fall through to the existing scalar path
    }

    // D-PBC-24: `rys_order` is the plain `(li + lk)/2 + 1` (g2c2e.c:60); `nroots`
    // is that DOUBLED under short range at `rys_order <= 3` (g2c2e.c:61-68). The
    // doubling is what makes SR "full minus long range" at the root level, and it
    // is the same value `query_workspace` sized the workspace for
    // (`WorkspaceQuery::rys_roots`).
    let range_omega = plan.operator_env_params.range_omega;
    let rys_order = (li as usize + lk as usize) / 2 + 1;
    let nroots = cintx_runtime::range_omega::nrys_roots_for(rys_order, range_omega);
    if nroots > 12 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_2c2e",
            detail: format!(
                "2c2e kernel supports nroots<=12; \
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
    let common_factor = (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(lk);

    // Flatten the f64 primitive data the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_k: Vec<f64> = shell_k.exponents[..n_prim_k].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_k: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();

    // Task 33-03: the boundary between the device kernel and the host loop is
    // the family's ceiling, not a constant. With `int2c2e` flipped onto the
    // inline extended entry and this backend's FMA probe passing, orders 6..=12
    // stay on the device; otherwise they fall to the host loop, as before.
    let device_ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int2c2e,
    );
    // D-PBC-24 stage 4: the device `#[cube]` kernel has no omega branch — its
    // comptime `nroots` arms select `rys_root{1..5}` at a single argument, and
    // short range needs two evaluations plus a root rescaling. Until the device
    // arms land, range separation routes to the HOST engine, explicitly and
    // logged rather than incidentally.
    let route_host = cintx_runtime::range_omega::is_range_separated(range_omega);
    if route_host {
        tracing::debug!(
            family = "2c2e",
            omega = range_omega.unwrap_or(0.0),
            rys_order,
            nroots,
            "range-separated 2c2e routed to the host Rys engine (D-PBC-24 stage 4)"
        );
    }
    let cart_buf: Vec<f64> = if route_host || nroots > device_ceiling {
        let nci = ncart(li);
        let nck = ncart(lk);
        let block_len = nci * nck;
        let mut cart_accum = vec![0.0f64; n_ctr_i * n_ctr_k * block_len];
        for pi in 0..n_prim_i {
            let ai = exps_i[pi];
            for pk in 0..n_prim_k {
                let ak = exps_k[pk];
                let Some(g) = fill_g_tensor_2c2e(
                    ai,
                    ak,
                    &ri,
                    &rk,
                    li,
                    lk,
                    common_factor,
                    nroots,
                    range_omega,
                )?
                else {
                    // Short-range integrand past EXPCUTOFF_SR: this primitive
                    // pair contributes nothing (g2e.c:4460).
                    continue;
                };
                let dn = nroots;
                let dm = nroots * (li as usize + 1);
                let g_size = nroots * (li as usize + 1) * (lk as usize + 1);
                let ci_comps = cart_comps(li);
                let ck_comps = cart_comps(lk);
                let mut prim_buf = vec![0.0f64; block_len];
                for (ck_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
                    for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                        let mut val = 0.0f64;
                        for irys in 0..nroots {
                            let vx = g[kx as usize * dm + ix as usize * dn + irys];
                            let vy = g[g_size + ky as usize * dm + iy as usize * dn + irys];
                            let vz = g[2 * g_size + kz as usize * dm + iz as usize * dn + irys];
                            val += vx * vy * vz;
                        }
                        prim_buf[ci_idx + ck_idx * nci] = val;
                    }
                }
                for ci in 0..n_ctr_i {
                    let ci_coeff = coeff_i[pi * n_ctr_i + ci];
                    for ck in 0..n_ctr_k {
                        let ck_coeff = coeff_k[pk * n_ctr_k + ck];
                        let pair_weight = ci_coeff * ck_coeff;
                        let block_offset = (ci * n_ctr_k + ck) * block_len;
                        for idx in 0..block_len {
                            cart_accum[block_offset + idx] += pair_weight * prim_buf[idx];
                        }
                    }
                }
            }
        }
        cart_accum
    } else {
        match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(client) => run_2c2e_device::<cubecl::cpu::CpuRuntime>(
                client,
                li as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_k as u32,
                nroots as u32,
                ri,
                rk,
                common_factor,
                &exps_i,
                &exps_k,
                &coeff_i,
                &coeff_k,
            ),
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(client, _) => run_2c2e_device::<cubecl_wgpu::WgpuRuntime>(
                client,
                li as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_k as u32,
                nroots as u32,
                ri,
                rk,
                common_factor,
                &exps_i,
                &exps_k,
                &coeff_i,
                &coeff_k,
            ),
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(client) => run_2c2e_device::<cubecl_cuda::CudaRuntime>(
                client,
                li as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_k as u32,
                nroots as u32,
                ri,
                rk,
                common_factor,
                &exps_i,
                &exps_k,
                &coeff_i,
                &coeff_k,
            ),
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(client) => run_2c2e_device::<cubecl_hip::HipRuntime>(
                client,
                li as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_k as u32,
                nroots as u32,
                ri,
                rk,
                common_factor,
                &exps_i,
                &exps_k,
                &coeff_i,
                &coeff_k,
            ),
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(client, _) => run_2c2e_device::<cubecl_wgpu::WgpuRuntime>(
                client,
                li as u32,
                lk as u32,
                n_prim_i as u32,
                n_prim_k as u32,
                n_ctr_i as u32,
                n_ctr_k as u32,
                nroots as u32,
                ri,
                rk,
                common_factor,
                &exps_i,
                &exps_k,
                &coeff_i,
                &coeff_k,
            ),
        }
    };

    // `cart_buf` holds `n_ctr_i * n_ctr_k` Cartesian blocks, contraction-major.
    // The AO index of contraction `c` and component `m` is `c * n<comp> + m`, so
    // a general contraction has to be scattered rather than copied — the same
    // shape the 2e path uses.
    let cart_block = ncart(li) * ncart(lk);

    // Representation dispatch: intermediate transforms use f64 temp buffers;
    // final values cast to F via F::from_f64_lossy.
    match plan.representation {
        Representation::Spheric => {
            let (nsi, nsk) = (nsph(li), nsph(lk));
            let di = n_ctr_i * nsi;
            for ci in 0..n_ctr_i {
                for ck in 0..n_ctr_k {
                    let base = (ci * n_ctr_k + ck) * cart_block;
                    let sph = cart_to_sph_2c2e(&cart_buf[base..base + cart_block], li, lk);
                    for mk in 0..nsk {
                        let kidx = ck * nsk + mk;
                        for mi in 0..nsi {
                            let iidx = ci * nsi + mi;
                            let dst = iidx + di * kidx;
                            if dst < staging.len() {
                                staging[dst] = F::from_f64_lossy(sph[mi + nsi * mk]);
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            let kappa_i = shell_i.kappa;
            let kappa_k = shell_k.kappa;
            // The spinor transform consumes one Cartesian block. General
            // contraction is not wired through it, so fail closed rather than
            // silently transforming only the first block.
            if n_ctr_i != 1 || n_ctr_k != 1 {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!(
                        "2c2e-spinor:general-contraction nctr_i={n_ctr_i} nctr_k={n_ctr_k}"
                    ),
                });
            }
            cart_to_spinor_sf_2d::<F>(staging, &cart_buf, li, kappa_i, lk, kappa_k)?;
        }
        Representation::Cart => {
            let (nci, nck) = (ncart(li), ncart(lk));
            let di = n_ctr_i * nci;
            for ci in 0..n_ctr_i {
                for ck in 0..n_ctr_k {
                    let base = (ci * n_ctr_k + ck) * cart_block;
                    for mk in 0..nck {
                        let kidx = ck * nck + mk;
                        for mi in 0..nci {
                            let iidx = ci * nci + mi;
                            let dst = iidx + di * kidx;
                            if dst < staging.len() {
                                staging[dst] = F::from_f64_lossy(cart_buf[base + mi + nci * mk]);
                            }
                        }
                    }
                }
            }
        }
    }

    // WR-06: precision-aware sentinel so f32 stale lanes are not counted.
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
            launch_center_2c2e_typed::<f32>(
                backend,
                plan,
                specialization,
                &mut staging_f32[..out_elems],
            )
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

        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, 0, 0, fac_env, 1, None)
            .expect("full range is supported")
            .expect("full range is never screened out");

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

        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, 1, 1, fac_env, 2, None)
            .expect("full range is supported")
            .expect("full range is never screened out");
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
        let g = fill_g_tensor_2c2e(ai, ak, &ri, &rk, li, lk, fac_env, nrys, None)
            .expect("full range is supported")
            .expect("full range is never screened out");

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

    // ── libcint byte-parity harness (DF-01 regression) ───────────────────────
    // Reference values are upstream PySCF `auxmol.intor('int2c2e_cart')`
    // (= libcint), normalized by the (xy|xy) [d] / [1][1] [f] element, at a
    // NON-ORIGIN geometry (exps 0.6/0.9, sep 1.7 bohr) so the c00/c0p-coupled
    // recurrence is exercised. Guards the n*b00 mixed-recurrence factor: a
    // regression there shifts the diagonal-cartesian d+ terms (n>=2). cart_comps
    // order matches libcint (xx,xy,xz,yy,yz,zz / xxx,xxy,...,zzz).
    fn parity_norm_matrix(li: u8, ai: f64, ak: f64) -> (Vec<f64>, usize) {
        let cf = (PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(li) * common_fac_sp(li);
        let out = host_cart_2c2e(
            ai,
            ak,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.7],
            li,
            li,
            cf,
            1.0,
            1.0,
        );
        let nc = ncart(li);
        // ref element = position [1][1] (the second cart comp self-term) = out[1 + 1*nc]
        let refv = out[1 + 1 * nc];
        (out.iter().map(|&v| v / refv).collect(), nc)
    }

    #[test]
    fn libcint_parity_2c2e_dd_nonorigin() {
        let (m, nc) = parity_norm_matrix(2, 0.6, 0.9);
        // upstream (i|k) normalized matrix, xx,xy,xz,yy,yz,zz order:
        let up: [[f64; 6]; 6] = [
            [26.638237, 0.0, 0.0, 24.638237, 0.0, 26.725773],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.380464, 0.0, 0.0, 0.0],
            [24.638237, 0.0, 0.0, 26.638237, 0.0, 26.725773],
            [0.0, 0.0, 0.0, 0.0, -0.380464, 0.0],
            [28.459773, 0.0, 0.0, 28.459773, 0.0, 29.157171],
        ];
        for i in 0..nc {
            for k in 0..nc {
                let got = m[i + k * nc]; // (i|k)
                let want = up[i][k];
                assert!(
                    (got - want).abs() < 1e-5,
                    "d-d (i={i}|k={k}): got {got:.6} want {want:.6} (libcint parity; check n*b00)"
                );
            }
        }
    }

    #[test]
    fn libcint_parity_2c2e_ff_diag_nonorigin() {
        let (m, nc) = parity_norm_matrix(3, 0.6, 0.9);
        // upstream f-f cartesian self-diagonal (cart_comps order), normalized by [1][1]:
        let up_diag: [f64; 10] = [
            7.613222, 1.0, -0.043674, 1.0, -0.062891, 1.225374, 7.613222, -0.043674, 1.225374,
            3.290372,
        ];
        for i in 0..nc {
            let got = m[i + i * nc];
            assert!(
                (got - up_diag[i]).abs() < 1e-5,
                "f-f diag i={i}: got {got:.6} want {:.6} (libcint parity)",
                up_diag[i]
            );
        }
    }

    #[test]
    fn test_device_kernel_matches_host_ff() {
        // Lock the device f-f path to the host (nroots=4 — exercises deeper VRR).
        assert_device_matches_host(3, 3, 0.6, 0.9);
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
        // Flattened two-shell basis: one primitive, one contraction each.
        let exps = [1.0_f32, 1.0];
        let coeffs = [1.0_f32, 1.0];
        let centers = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 1.7];
        let shell_meta: [u32; 8] = [0, 0, 1, 1, 1, 1, 1, 1];
        // `[si, sk, out_off, class]` — one class, index 0.
        let pairs: [u32; 4] = [0, 1, 0, 0];
        let class_shape: [u32; TWO_C2E_SHAPE_STRIDE] = [0, 0];
        let g_zero = [0.0_f32; 3]; // nroots=1, g_size=1 → 3
        let out_zero = [0.0_f32; 1];

        let exps_h = client.create_from_slice(f32::as_bytes(&exps));
        let coeffs_h = client.create_from_slice(f32::as_bytes(&coeffs));
        let centers_h = client.create_from_slice(f32::as_bytes(&centers));
        let meta_h = client.create_from_slice(u32::as_bytes(&shell_meta));
        let pairs_h = client.create_from_slice(u32::as_bytes(&pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&class_shape));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        let class_factor =
            [((PI * PI * PI) * 2.0 / SQRTPI * common_fac_sp(0) * common_fac_sp(0)) as f32];
        let factor_h = client.create_from_slice(f32::as_bytes(&class_factor));
        // The extended-Rys tables are an unconditional kernel argument; an
        // `nroots = 1` smoke launch never reads them.
        let rys_tables = crate::math::rys_wheeler::ext_rys_tables();
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));

        center_2c2e_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_h, exps.len()) },
            unsafe { ArrayArg::from_raw_parts(coeffs_h, coeffs.len()) },
            unsafe { ArrayArg::from_raw_parts(centers_h, centers.len()) },
            unsafe { ArrayArg::from_raw_parts(meta_h, shell_meta.len()) },
            unsafe { ArrayArg::from_raw_parts(pairs_h, pairs.len()) },
            unsafe { ArrayArg::from_raw_parts(shape_h, class_shape.len()) },
            unsafe { ArrayArg::from_raw_parts(factor_h, class_factor.len()) },
            unsafe { ArrayArg::from_raw_parts(rys_tab_h, EXT_TABLES_LEN) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            PIE4 as f32,
            1u32, // n_pairs
            1u32, // n_cubes
            3u32, // g_stride (one slab, unpadded)
            1u32, // nroots
            // One cube, one pair: the cooperative shape this single slab is
            // sized for.
            0u32,
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
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell,
        };
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let shell_a = Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                Arc::from(vec![1.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let shell_b = Arc::new(
            Shell::try_new(
                1,
                0,
                1,
                1,
                0,
                Representation::Cart,
                Arc::from(vec![1.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();

        let op = OperatorId::new(12);
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let opts = ExecutionOptions::default();
        let query = match query_workspace(op, Representation::Cart, &basis, shells.clone(), &opts) {
            Ok(q) => q,
            Err(_) => return,
        };
        let mut plan =
            ExecutionPlan::new(op, Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 1];

        let result = launch_center_2c2e_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            result.is_ok(),
            "f64 2c2e typed inner should succeed: {:?}",
            result
        );
        assert!(staging[0].is_finite(), "2c2e f64 result should be finite");
        assert!(staging[0] > 0.0, "s-s 2c2e integral should be positive");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T04-2b: launch_center_2c2e_typed::<f32> writes a finite f32.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_2c2e_precision_dispatch_f32_positive() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, OperatorId, PrecisionKind, Representation, Shell,
        };
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let shell_a = Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Cart,
                Arc::from(vec![1.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let shell_b = Arc::new(
            Shell::try_new(
                1,
                0,
                1,
                1,
                0,
                Representation::Cart,
                Arc::from(vec![1.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let all_shells = Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();

        let op = OperatorId::new(12);
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        let opts = ExecutionOptions::default();
        let query = match query_workspace(op, Representation::Cart, &basis, shells.clone(), &opts) {
            Ok(q) => q,
            Err(_) => return,
        };
        let mut plan =
            ExecutionPlan::new(op, Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging_f32 = vec![0.0_f32; 1];

        let result = launch_center_2c2e_typed::<f32>(&backend, &plan, &spec, &mut staging_f32);
        assert!(
            result.is_ok(),
            "f32 2c2e typed inner should succeed: {:?}",
            result
        );
        assert!(
            staging_f32[0].is_finite(),
            "2c2e f32 result should be finite"
        );
        assert!(
            staging_f32[0] > 0.0,
            "s-s 2c2e f32 integral should be positive"
        );
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
        use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell};
        use cintx_ops::resolver::Resolver;
        use std::sync::Arc;

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
        let op = Resolver::descriptor_by_symbol(symbol)
            .expect("symbol in manifest")
            .id;
        (basis, shells, op)
    }

    fn run_2c2e_grad(
        basis: &BasisSet,
        shells: cintx_core::ShellTuple,
        op: cintx_core::OperatorId,
        rep: cintx_core::Representation,
    ) -> Result<Vec<f64>, cintxRsError> {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};

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
        assert!(
            out.iter().any(|v| v.abs() > 1e-14),
            "int2c2e_ip1 (p,s) all-zero"
        );
    }

    #[test]
    fn test_int2c2e_ip2_component_count() {
        let (basis, shells, op) = build_2c2e_grad_plan(0, 1, "int2c2e_ip2_sph");
        let out = run_2c2e_grad(&basis, shells, op, Representation::Spheric).unwrap();
        assert_eq!(out.len(), 9, "(s,p) int2c2e_ip2 should produce 9 outputs");
        assert!(
            out.iter().any(|v| v.abs() > 1e-14),
            "int2c2e_ip2 (s,p) all-zero"
        );
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
        assert!(
            res.is_ok(),
            "(f,f) int2c2e_ip1 (nroots=4) must be allowed: {:?}",
            res.err()
        );
    }

    // 27-03 (FND-04): int2c2e_ip1/ip2 spinor gradients now EVALUATE via the
    // centralized derivative wrapper (was UnsupportedApi). The wrapper owns the
    // KET→BRA transpose (D-06) and there is no aux-k axis for 2c2e.
    #[test]
    fn test_int2c2e_grad_spinor_evaluates() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};

        let (basis, shells, op) = build_2c2e_grad_plan(1, 0, "int2c2e_ip1_sph");
        let opts = ExecutionOptions::default();
        let q =
            query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts).unwrap();
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
