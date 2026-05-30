//! Host-side 1e integral kernel: overlap, kinetic, and nuclear attraction.
//!
//! Implements the G-tensor fill + operator post-processing pipeline from
//! libcint `g1e.c` lines 125-320, `intor1.c` lines 18-46, and `cint1e.c` lines 284-436.
//!
//! # Algorithm
//! For each contracted shell pair (i, j):
//! 1. Compute pair data (PairData) from exponents and centers.
//! 2. Fill G-tensor via VRR + HRR for each Cartesian axis.
//! 3. Contract Cartesian components for the operator (overlap, kinetic, or nuclear).
//! 4. Accumulate over primitives weighted by contraction coefficients.
//! 5. Apply cart-to-sph transform if representation is Spheric.

use crate::backend::ResolvedBackend;
#[cfg(test)]
use crate::math::obara_saika::{hrr_step_host, vrr_step_host};
#[cfg(test)]
use crate::math::pdata::compute_pdata_host;
#[cfg(test)]
use crate::math::rys::rys_roots_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use crate::transform::c2spinor::{cart_to_spinor_sf_2d, spinor_len};
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// sqrt(pi) constant — used in G-tensor base case normalization.
/// Matches libcint `g1e.c` `SQRTPI = sqrt(M_PI)`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Rys `PIE4 = pi/4` constant passed into the device `rys_root{1..5}` kernels.
/// Matches `rys_roots.c` `PIE4`. Used by the on-device nuclear-attraction arm.
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate for the
/// on-device nuclear-attraction arm. `nrys = (li + lj) / 2 + 1`, so this covers
/// `li + lj <= 8`. Same `MAX_DEVICE_NROOTS` guard the 2c2e device kernel uses.
const MAX_DEVICE_NROOTS: usize = 5;

/// Spherical harmonic normalization prefactor for s and p shells.
///
/// In libcint's `cart2sph.c` and `g1e.c`, the `CINTcommon_fac_sp(l)` function
/// returns the normalization factor that is incorporated into the primitive loop
/// rather than the cart-to-sph transform tables. The c2s tables for s and p use
/// coefficient 1.0, and `CINTcommon_fac_sp` carries the actual normalization:
///   - l=0 (s): 0.282094791773878 = 1/(2*sqrt(pi)) = Y_0^0
///   - l=1 (p): 0.488602511902920 = sqrt(3/(4*pi))
///   - l>=2:    1.0 (normalization is embedded in c2s coefficients)
///
/// This function must be applied as a post-processing scale factor to the
/// accumulated Cartesian buffer before (or after) the cart-to-sph transform.
/// Without it, s/p-type integrals are off by a factor of 4*pi relative to libcint.
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64, // 1/(2*sqrt(pi))
        1 => 0.488602511902919921_f64, // sqrt(3/(4*pi))
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
///
/// Follows libcint `CINTcart_comp` ordering:
/// for lx = l..=0, for ly = l-lx..=0, nz = l - lx - ly.
///
/// Host helper — the live device kernels enumerate the Cartesian triples inline;
/// the only remaining callers are the host cross-check / contract references.
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

/// Compute the G-tensor elements for one primitive pair (overlap base).
///
/// Returns a flat array of size `3 * g_per_axis` where g_per_axis = (nmax+1)*(lj+1).
/// The layout is [gx | gy | gz], each of size g_per_axis.
/// After VRR+HRR, element `g[axis * g_per_axis + j * (nmax+1) + i]` gives
/// the i-th VRR index for the j-th HRR transfer along that axis.
///
/// Parameters:
/// - `pd`: pair data from `compute_pdata_host`
/// - `ri`: bra center [x,y,z]
/// - `rj`: ket center [x,y,z]
/// - `nmax`: total angular momentum nmax = li + lj (VRR max)
/// - `lj`: ket angular momentum (HRR target)
///
/// Host reference helper — the live scalar/gradient paths now build the G-tensor
/// in-kernel; the only remaining callers are the device-vs-host cross-check tests.
#[cfg(test)]
fn fill_g_tensor_overlap(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    nmax: u32,
    lj: u32,
) -> Vec<f64> {
    // g_per_axis = (nmax+1) entries per j-level; we need lj+1 j-levels
    let g_per_axis = ((nmax + 1) * (lj + 1)) as usize;
    let mut g = vec![0.0_f64; 3 * g_per_axis];

    // Base case: gz[0] carries the full normalization constant.
    // gx[0] = 1.0, gy[0] = 1.0, gz[0] = fac * SQRTPI * PI / (zeta * sqrt(zeta))
    // Source: g1e.c lines 127-135, 139-145
    let aij = pd.zeta_ab;
    let gz0 = pd.fac * SQRTPI * std::f64::consts::PI / (aij * aij.sqrt());

    let gx = 0;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;

    g[gx] = 1.0;
    g[gy] = 1.0;
    g[gz] = gz0;

    // VRR: fill angular momentum on bra center (center i: P - Ri).
    // rijrx = P - Ri (if li >= lj, VRR on bra; otherwise on ket).
    // For simplicity we always VRR on bra (center i) then HRR to ket.
    // Source: g1e.c lines 164-172
    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
    let rijrx = [rp[0] - ri[0], rp[1] - ri[1], rp[2] - ri[2]];

    if nmax >= 1 {
        // VRR for gx, gy, gz separately
        // stride = 1 within each axis block
        vrr_step_host(&mut g[gx..gx + g_per_axis], rijrx[0], pd.aij2, nmax, 1);
        vrr_step_host(&mut g[gy..gy + g_per_axis], rijrx[1], pd.aij2, nmax, 1);
        vrr_step_host(&mut g[gz..gz + g_per_axis], rijrx[2], pd.aij2, nmax, 1);
    }

    // HRR: shift angular momentum to ket center (j).
    // rirj = Ri - Rj.
    // Source: g1e.c lines 175-182
    // di = 1 (i-stride), dj = nmax+1 (j-stride within axis block)
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    if lj >= 1 {
        let di = 1u32;
        let dj = nmax + 1;
        let li_max = nmax; // VRR built up to nmax on bra
        hrr_step_host(&mut g[gx..gx + g_per_axis], rirj[0], di, dj, li_max, lj);
        hrr_step_host(&mut g[gy..gy + g_per_axis], rirj[1], di, dj, li_max, lj);
        hrr_step_host(&mut g[gz..gz + g_per_axis], rirj[2], di, dj, li_max, lj);
    }

    g
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]`, generic over `F: Float`
//
//  Ports the SCALAR 1e arms (overlap / kinetic / nuclear-attraction) of
//  `launch_one_electron_typed` onto the CubeCL device, following the proven
//  `center_2c2e.rs::center_2c2e_kernel` / `run_2c2e_device` template EXACTLY.
//
//  `#[comptime] op_kind` selects the operator branch (0=overlap, 1=kinetic,
//  2=nuclear) at JIT specialization time — no runtime operator dispatch inside
//  the hot path, mirroring 2c2e's comptime `nroots`. `#[comptime] nroots`
//  selects the `rys_root{1..5}` device fn for the nuclear arm; for
//  overlap/kinetic it is fixed at 1 (no Rys quadrature, fixed-center VRR).
//
//  The kernel computes pair data (zeta, fac, aij2, P) IN-KERNEL in `F` so the
//  whole arithmetic core is genuinely generic over `F` (the host `compute_pdata`
//  returns f64-typed `PairData`, which would force f64; the 2c2e kernel likewise
//  recomputes its products inline to stay generic). The vrr_step / vrr_2e_step /
//  hrr_step / rys_rootN device fns are all `#[cube]` and called directly — this
//  is allowed (they are NOT plain-fn calls).
// ─────────────────────────────────────────────────────────────────────────────

/// On-device scalar 1e G-tensor fill + contraction for ONE shell pair.
///
/// Single work item (`UNIT_POS == 0`), faithful correctness-first port of the
/// host scalar pipeline (`fill_g_tensor_overlap` + `contract_overlap` /
/// `contract_kinetic` / `contract_nuclear`). Iterates the primitive pairs
/// (pi,pj) and contraction pairs (ci,cj) in-kernel and accumulates ONE
/// `nci*ncj` Cartesian block per (ci,cj) into `cart_out`, laid out
/// contraction-major / bra-fastest exactly as the host scalar path does:
/// block base `(ci*nctr_j + cj) * (nci*ncj)`, element `out[cj_idx*nci + ci_idx]`.
///
/// Scratch buffers `g` (overlap/kinetic G-tensor or per-root nuclear G-tensor),
/// `urys`/`wrys` (Rys roots, nuclear only) and the `cart_out` accumulator are
/// passed as `&mut Array<F>` and zeroed in-kernel before use, exactly like 2c2e.
///
/// Source: libcint-master/src/g1e.c `CINTg1e_ovlp` / `CINTg1e_nuc`,
///         autocode/intor1.c `CINTgout1e_int1e_kin`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_scalar_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    atom_coords: &Array<F>,
    atom_charges: &Array<F>,
    g: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    pie4: F,
    sqrtpi: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    natm: u32,
    #[comptime] op_kind: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        // Bind the comptime nroots to a runtime u32 (mirrors 2c2e `let nrys = nroots`).
        let nrys = nroots;
        // ── G-tensor sizing (mirrors the host scalar path) ───────────────────
        // overlap : nmax = li+lj      , lj_ext = lj
        // kinetic : nmax = li+lj+2    , lj_ext = lj+2  (D_j^2 needs jx+2 access)
        // nuclear : nmax = li+lj      , lj_ext = lj
        let mut nmax = li + lj;
        let mut lj_ext = lj;
        if comptime!(op_kind == 1u32) {
            nmax = li + lj + 2u32;
            lj_ext = lj + 2u32;
        }
        let dj = nmax + 1u32; // stride between consecutive j-levels within an axis block
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let out_total = nctr_i * nctr_j * block_len;

        // Zero the full accumulation buffer.
        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // `pi_const` is passed in as a runtime scalar so the f64 path keeps full
        // PI precision (F::new only accepts f32).

        // ── Primitive loop ───────────────────────────────────────────────────
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                // Pair data, computed in-kernel in F (norm_i = norm_j = 1.0).
                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // ── Build the per-primitive Cartesian block into a temporary
                //    region of `cart_out`? No — accumulate directly per (ci,cj)
                //    with the contraction weight, mirroring the host scatter.
                //
                // To avoid recomputing the (operator-specific) G-tensor for every
                // contraction pair, build it ONCE per primitive pair into `g`
                // (and for nuclear, accumulate the per-atom/per-root contributions
                // into a primitive Cartesian buffer stored in the FIRST block of
                // `cart_out`'s scratch — but cart_out is the live accumulator, so
                // instead we re-contract per (ci,cj) from the shared `g`).
                //
                // Overlap / kinetic: a single `g` is shared by all (ci,cj).
                // Nuclear: the per-root accumulation is folded into `g`-derived
                //          contraction below, recomputed per primitive (atoms loop
                //          inside), independent of (ci,cj). We therefore build a
                //          per-primitive Cartesian block `prim` on the fly inside
                //          the (ci,cj) loop by reading `g` (ovlp/kin) or by the
                //          nuclear root loop. For determinism + simplicity we
                //          compute the primitive Cartesian block ONCE here for
                //          ovlp/kin (store in `g`-derived reads) and for nuclear we
                //          recompute the root accumulation inside the contraction.

                if comptime!(op_kind == 0u32) {
                    // ===== OVERLAP G-tensor (fixed-center VRR + HRR) =====
                    let mut gi = 0u32;
                    while gi < total_g {
                        g[gi as usize] = F::new(0.0);
                        gi += 1u32;
                    }
                    // Base case: gx[0]=1, gy[0]=1, gz[0]=fac*SQRTPI*PI/(zeta*sqrt(zeta))
                    g[gx as usize] = F::new(1.0);
                    g[gy as usize] = F::new(1.0);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    // VRR on bra (center i): rijrx = P - Ri, per axis sub-block.
                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    // HRR to ket center on all 3 axes: rirj = Ri - Rj.
                    if lj >= 1u32 {
                        one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj);
                        one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj);
                        one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj);
                    }
                } else if comptime!(op_kind == 1u32) {
                    // ===== KINETIC: overlap G-tensor with lj+2 HRR levels =====
                    let mut gi = 0u32;
                    while gi < total_g {
                        g[gi as usize] = F::new(0.0);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0);
                    g[gy as usize] = F::new(1.0);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    // HRR to lj_ext = lj+2 levels.
                    if lj_ext >= 1u32 {
                        one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                    }
                }
                // (Nuclear builds its G-tensor per-root inside the contraction below.)

                // ── Contract into every (ci,cj) contraction block ────────────
                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;
                        let base = (ci * nctr_j + cj) * block_len;

                        // Iterate Cartesian component triples (cj outer, ci inner),
                        // matching the host cart_comps ordering (lx descending).
                        let mut cj_idx = 0u32;
                        let mut ja = 0u32;
                        while ja <= lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            let mut jb = 0u32;
                            while jb <= lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

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

                                        if comptime!(op_kind == 0u32) {
                                            // Overlap: vx*vy*vz from shared g.
                                            let vx = g[(gx + jx * dj + ix) as usize];
                                            let vy = g[(gy + jy * dj + iy) as usize];
                                            let vz = g[(gz + jz * dj + iz) as usize];
                                            val = vx * vy * vz;
                                        } else if comptime!(op_kind == 1u32) {
                                            // Kinetic: T = -0.5*(g3x*g0y*g0z + ...)
                                            let nx = jx * dj + ix;
                                            let ny = jy * dj + iy;
                                            let nz = jz * dj + iz;
                                            let vx0 = g[(gx + nx) as usize];
                                            let vy0 = g[(gy + ny) as usize];
                                            let vz0 = g[(gz + nz) as usize];

                                            let g3x =
                                                one_electron_kin_d2::<F>(g, gx, nx, dj, jx, aj);
                                            let g3y =
                                                one_electron_kin_d2::<F>(g, gy, ny, dj, jy, aj);
                                            let g3z =
                                                one_electron_kin_d2::<F>(g, gz, nz, dj, jz, aj);
                                            val = F::new(-0.5)
                                                * (g3x * vy0 * vz0
                                                    + vx0 * g3y * vz0
                                                    + vx0 * vy0 * g3z);
                                        } else {
                                            // Nuclear: sum over atoms and Rys roots.
                                            let mut atom = 0u32;
                                            while atom < natm {
                                                let z_c = atom_charges[atom as usize];
                                                let rcx = atom_coords[(atom * 3u32) as usize];
                                                let rcy =
                                                    atom_coords[(atom * 3u32 + 1u32) as usize];
                                                let rcz =
                                                    atom_coords[(atom * 3u32 + 2u32) as usize];

                                                // crij = C - P
                                                let crijx = rcx - px;
                                                let crijy = rcy - py;
                                                let crijz = rcz - pz;
                                                let x_boys = zeta
                                                    * (crijx * crijx
                                                        + crijy * crijy
                                                        + crijz * crijz);

                                                // Rys roots/weights (comptime nroots).
                                                if comptime!(nroots == 1u32) {
                                                    rys_root1::<F>(x_boys, urys, wrys, pie4);
                                                } else if comptime!(nroots == 2u32) {
                                                    rys_root2::<F>(x_boys, urys, wrys, pie4);
                                                } else if comptime!(nroots == 3u32) {
                                                    rys_root3::<F>(x_boys, urys, wrys, pie4);
                                                } else if comptime!(nroots == 4u32) {
                                                    rys_root4::<F>(x_boys, urys, wrys, pie4);
                                                } else {
                                                    rys_root5::<F>(x_boys, urys, wrys, pie4);
                                                }

                                                // fac1 = 2*PI*(-Z_C)*fac/zeta
                                                let neg_z = F::new(0.0) - z_c;
                                                let fac1 =
                                                    F::new(2.0) * pi_const * neg_z * fac / zeta;

                                                let mut irys: u32 = 0u32;
                                                while irys < nrys {
                                                    let u_n = urys[irys as usize];
                                                    let w_n = wrys[irys as usize];
                                                    let tau = u_n / (F::new(1.0) + u_n);
                                                    let rt = aij2 * (F::new(1.0) - tau);

                                                    let c00x = (px - rix) + tau * crijx;
                                                    let c00y = (py - riy) + tau * crijy;
                                                    let c00z = (pz - riz) + tau * crijz;

                                                    // Build per-root G-tensor in `g`.
                                                    let mut gi2 = 0u32;
                                                    while gi2 < total_g {
                                                        g[gi2 as usize] = F::new(0.0);
                                                        gi2 += 1u32;
                                                    }
                                                    g[gx as usize] = F::new(1.0);
                                                    g[gy as usize] = F::new(1.0);
                                                    g[gz as usize] = fac1 * w_n;

                                                    one_electron_vrr2e_axis::<F>(
                                                        g, gx, c00x, rt, nmax,
                                                    );
                                                    one_electron_vrr2e_axis::<F>(
                                                        g, gy, c00y, rt, nmax,
                                                    );
                                                    one_electron_vrr2e_axis::<F>(
                                                        g, gz, c00z, rt, nmax,
                                                    );
                                                    if lj >= 1u32 {
                                                        one_electron_hrr_axis::<F>(
                                                            g, gx, rirjx, dj, nmax, lj,
                                                        );
                                                        one_electron_hrr_axis::<F>(
                                                            g, gy, rirjy, dj, nmax, lj,
                                                        );
                                                        one_electron_hrr_axis::<F>(
                                                            g, gz, rirjz, dj, nmax, lj,
                                                        );
                                                    }

                                                    let vx = g[(gx + jx * dj + ix) as usize];
                                                    let vy = g[(gy + jy * dj + iy) as usize];
                                                    let vz = g[(gz + jz * dj + iz) as usize];
                                                    val += vx * vy * vz;

                                                    irys += 1u32;
                                                }
                                                atom += 1u32;
                                            }
                                        }

                                        cart_out[(base + cj_idx * nci + ci_idx) as usize] +=
                                            weight * val;

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }

                                cj_idx += 1u32;
                                jb += 1u32;
                            }
                            ja += 1u32;
                        }

                        cj += 1u32;
                    }
                    ci += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Per-axis 1e overlap VRR into the `g` sub-block starting at `base`.
///
/// Writes `g[base + n]` for `n = 1..=nmax` using the fixed-center recurrence,
/// reproducing `vrr_step` on a sub-block (the `#[cube]` `vrr_step` only operates
/// from index 0, so the multi-axis case needs an explicit base offset).
#[cube]
fn one_electron_vrr_axis<F: Float>(g: &mut Array<F>, base: u32, rijrx: F, aij2: F, nmax: u32) {
    if nmax >= 1u32 {
        g[(base + 1u32) as usize] = rijrx * g[base as usize];
        let mut n = 1u32;
        while n < nmax {
            g[(base + n + 1u32) as usize] = F::cast_from(n) * aij2 * g[(base + n - 1u32) as usize]
                + rijrx * g[(base + n) as usize];
            n += 1u32;
        }
    }
}

/// Per-axis 2e (root-dependent) VRR into the `g` sub-block starting at `base`.
///
/// Writes `g[base + n]` for `n = 1..=nmax` using `c00` / `b10 = rt` — the nuclear
/// attraction root recurrence (`vrr_2e_step` on a sub-block with explicit base).
#[cube]
fn one_electron_vrr2e_axis<F: Float>(g: &mut Array<F>, base: u32, c00: F, b10: F, nmax: u32) {
    if nmax >= 1u32 {
        g[(base + 1u32) as usize] = c00 * g[base as usize];
        let mut n = 1u32;
        while n < nmax {
            g[(base + n + 1u32) as usize] = F::cast_from(n) * b10 * g[(base + n - 1u32) as usize]
                + c00 * g[(base + n) as usize];
            n += 1u32;
        }
    }
}

/// Per-axis HRR into the `g` sub-block starting at `base` (i-stride = 1).
///
/// Shifts angular momentum to the ket center, building j-levels `1..=lj`,
/// reproducing `hrr_step` on a sub-block with explicit base offset.
#[cube]
fn one_electron_hrr_axis<F: Float>(
    g: &mut Array<F>,
    base: u32,
    rirj: F,
    dj: u32,
    li_max: u32,
    lj: u32,
) {
    let mut j = 1u32;
    while j <= lj {
        let i_max = li_max - j;
        let mut i = 0u32;
        while i <= i_max {
            let idx_out = base + j * dj + i;
            let idx_hi = base + (j - 1u32) * dj + (i + 1u32);
            let idx_lo = base + (j - 1u32) * dj + i;
            g[idx_out as usize] = g[idx_hi as usize] + rirj * g[idx_lo as usize];
            i += 1u32;
        }
        j += 1u32;
    }
}

/// Second ket-derivative `D_j^2(g0)[j, i]` on one axis (kinetic operator).
///
/// `g3 = jx*(jx-1)*g0[jx-2] - 2*aj*(2*jx+1)*g0[jx] + 4*aj^2*g0[jx+2]`, stepping
/// `±2` j-levels (`±2*dj` in the flat index). `nx = jx*dj + ix` is the base flat
/// offset within the axis sub-block at `base`. Matches `contract_kinetic`.
#[cube]
fn one_electron_kin_d2<F: Float>(g: &Array<F>, base: u32, nx: u32, dj: u32, jx: u32, aj: F) -> F {
    let g_hi = g[(base + nx + 2u32 * dj) as usize];
    let v0 = g[(base + nx) as usize];
    let jxf = F::cast_from(jx);
    let mut lo = F::new(0.0);
    if jx >= 2u32 {
        lo = g[(base + nx - 2u32 * dj) as usize];
    }
    F::new(4.0) * aj * aj * g_hi - F::new(2.0) * aj * (F::new(2.0) * jxf + F::new(1.0)) * v0
        + jxf * (jxf - F::new(1.0)) * lo
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — bra-nabla GRADIENT operators
//
//  Ports the host `contract_grad_1e_bra` (ipovlp, op_kind=0) and `contract_ipkin`
//  (ipkin, op_kind=1) onto the CubeCL device, cloning `one_electron_scalar_kernel`'s
//  structure EXACTLY (single UNIT_POS==0 work item; in-kernel pdata recompute in F;
//  zeroed scratch + cart_out; (pi,pj) primitive loop × (ci,cj) contraction loop).
//
//  The output is 3-component COMPONENT-LEADING per (ci,cj) block: base
//  `(ci*nctr_j+cj)*total_len`, then `comp*block_len + cj_idx*nci + ci_idx`
//  (total_len = 3*nci*ncj), matching the host `out[comp*nci*ncj + cj*nci + ci]`
//  layout and the live launcher's `cart_3comp` staging.
//
//  G-tensor sizing per `#[comptime] op_kind`:
//    ipovlp: nmax = li+lj+1, lj_ext = lj   (one extra bra level for the ix+1 nabla read)
//    ipkin : nmax = li+lj+3, lj_ext = lj+2 (kinetic D_j^2 needs jx+2 AND nabla ix+1)
//  Both build the OVERLAP base G-tensor (vrr_step + hrr_step), then apply the bra
//  nabla inline into `g1`. ipkin additionally builds `d2g0`/`d2g1` (D_j^2 of g0/g1)
//  over the base lj range, verbatim from `contract_ipkin`.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device bra-nabla 1e gradient (`ipovlp` op_kind=0 / `ipkin` op_kind=1).
///
/// Single work item; faithful port of `contract_grad_1e_bra` / `contract_ipkin`.
/// Scratch: `g` (overlap base G-tensor), `g1` (bra-nabla of g), `d2g0`/`d2g1`
/// (kinetic second ket-derivatives; only written for op_kind=1, but always sized).
/// `cart_out` is the 3-component component-leading accumulator.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_grad_bra_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    d2g0: &mut Array<F>,
    d2g1: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    sqrtpi: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    #[comptime] op_kind: u32,
) {
    if UNIT_POS == 0u32 {
        // G-tensor sizing (mirrors host gradient path).
        //   ipovlp: nmax = li+lj+1, lj_ext = lj
        //   ipkin : nmax = li+lj+3, lj_ext = lj+2
        let mut nmax = li + lj + 1u32;
        let mut lj_ext = lj;
        if comptime!(op_kind == 1u32) {
            nmax = li + lj + 3u32;
            lj_ext = lj + 2u32;
        }
        let dj = nmax + 1u32; // stride between consecutive j-levels within an axis block
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_len = 3u32 * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        // Zero the full accumulation buffer.
        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // ── Primitive loop ───────────────────────────────────────────────────
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                // Pair data, computed in-kernel in F (norm_i = norm_j = 1.0).
                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // ── Build OVERLAP base G-tensor in `g` (fixed-center VRR + HRR) ──
                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                if lj_ext >= 1u32 {
                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                }

                // ── Bra nabla1i → g1 over 0..=lj_ext j-levels, all 3 axes. ───────
                let mut g1i = 0u32;
                while g1i < total_g {
                    g1[g1i as usize] = F::new(0.0);
                    g1i += 1u32;
                }
                let ai2 = F::new(-2.0) * ai;
                let mut axisn = 0u32;
                while axisn < 3u32 {
                    let off = axisn * g_per_axis;
                    let mut jn = 0u32;
                    while jn <= lj_ext {
                        let jbase = jn * dj;
                        // ix = 0: f = -2*ai * g[ix+1]
                        g1[(off + jbase) as usize] = ai2 * g[(off + jbase + 1u32) as usize];
                        // ix >= 1: f = ix * g[ix-1] + (-2*ai) * g[ix+1]
                        let mut ix = 1u32;
                        while ix <= li {
                            g1[(off + jbase + ix) as usize] = F::cast_from(ix)
                                * g[(off + jbase + ix - 1u32) as usize]
                                + ai2 * g[(off + jbase + ix + 1u32) as usize];
                            ix += 1u32;
                        }
                        jn += 1u32;
                    }
                    axisn += 1u32;
                }

                // ── Kinetic: D_j^2 of BOTH g0 and g1 over the base lj range. ─────
                if comptime!(op_kind == 1u32) {
                    let mut d2axis = 0u32;
                    while d2axis < 3u32 {
                        let off = d2axis * g_per_axis;
                        let mut jd = 0u32;
                        while jd <= lj {
                            let jf = F::cast_from(jd);
                            let mut id = 0u32;
                            while id <= li {
                                let nx = jd * dj + id;
                                // lo term only valid for j>=2.
                                let mut g0_lo = F::new(0.0);
                                let mut g1_lo = F::new(0.0);
                                if jd >= 2u32 {
                                    g0_lo = g[(off + nx - 2u32 * dj) as usize];
                                    g1_lo = g1[(off + nx - 2u32 * dj) as usize];
                                }
                                let coef_mid =
                                    F::new(2.0) * aj * (F::new(2.0) * jf + F::new(1.0));
                                let coef_hi = F::new(4.0) * aj * aj;
                                let coef_lo = jf * (jf - F::new(1.0));
                                d2g0[(off + nx) as usize] =
                                    coef_hi * g[(off + nx + 2u32 * dj) as usize]
                                        - coef_mid * g[(off + nx) as usize]
                                        + coef_lo * g0_lo;
                                d2g1[(off + nx) as usize] =
                                    coef_hi * g1[(off + nx + 2u32 * dj) as usize]
                                        - coef_mid * g1[(off + nx) as usize]
                                        + coef_lo * g1_lo;
                                id += 1u32;
                            }
                            jd += 1u32;
                        }
                        d2axis += 1u32;
                    }
                }

                // ── Contract into every (ci,cj) contraction block ────────────
                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;
                        let base = (ci * nctr_j + cj) * total_len;

                        // cart_comps order: cj outer (lx descending), ci inner.
                        let mut cj_idx = 0u32;
                        let mut ja = 0u32;
                        while ja <= lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            let mut jb = 0u32;
                            while jb <= lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

                                let mut ci_idx = 0u32;
                                let mut ia = 0u32;
                                while ia <= li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    let mut ib = 0u32;
                                    while ib <= li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        let nx = jx * dj + ix;
                                        let ny = jy * dj + iy;
                                        let nz = jz * dj + iz;

                                        let g0x = g[(gx + nx) as usize];
                                        let g0y = g[(gy + ny) as usize];
                                        let g0z = g[(gz + nz) as usize];
                                        let g1x = g1[(gx + nx) as usize];
                                        let g1y = g1[(gy + ny) as usize];
                                        let g1z = g1[(gz + nz) as usize];

                                        // Default = ipovlp mixing (s0=g1x*g0y*g0z, etc.).
                                        // The init IS the op_kind==0 result; the kinetic
                                        // branch overwrites it.
                                        let mut s0 = g1x * g0y * g0z;
                                        let mut s1 = g0x * g1y * g0z;
                                        let mut s2 = g0x * g0y * g1z;

                                        if comptime!(op_kind == 1u32) {
                                            // ipkin: D_j^2 mixing (verbatim contract_ipkin).
                                            let d2x0 = d2g0[(gx + nx) as usize];
                                            let d2y0 = d2g0[(gy + ny) as usize];
                                            let d2z0 = d2g0[(gz + nz) as usize];
                                            let d2x1 = d2g1[(gx + nx) as usize];
                                            let d2y1 = d2g1[(gy + ny) as usize];
                                            let d2z1 = d2g1[(gz + nz) as usize];
                                            s0 = F::new(-0.5)
                                                * (d2x1 * g0y * g0z
                                                    + g1x * d2y0 * g0z
                                                    + g1x * g0y * d2z0);
                                            s1 = F::new(-0.5)
                                                * (d2x0 * g1y * g0z
                                                    + g0x * d2y1 * g0z
                                                    + g0x * g1y * d2z0);
                                            s2 = F::new(-0.5)
                                                * (d2x0 * g0y * g1z
                                                    + g0x * d2y0 * g1z
                                                    + g0x * g0y * d2z1);
                                        }

                                        let elem = cj_idx * nci + ci_idx;
                                        cart_out[(base + elem) as usize] += weight * s0;
                                        cart_out[(base + block_len + elem) as usize] +=
                                            weight * s1;
                                        cart_out
                                            [(base + 2u32 * block_len + elem) as usize] +=
                                            weight * s2;

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }

                                cj_idx += 1u32;
                                jb += 1u32;
                            }
                            ja += 1u32;
                        }

                        cj += 1u32;
                    }
                    ci += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`one_electron_grad_bra_kernel`] at `f64` on a resolved backend's
/// client. Returns the 3-component component-leading accumulator of length
/// `3 * nci * ncj * nctr_i * nctr_j`. Buffer creation clones
/// [`run_1e_scalar_device`]; `op_kind` (0=ipovlp, 1=ipkin) selects the
/// monomorphization at the `launch::<f64, R>` call site.
#[allow(clippy::too_many_arguments)]
fn run_1e_grad_bra_device<R: Runtime>(
    client: &ComputeClient<R>,
    op_kind: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let (nmax_u, lj_ext_u) = if op_kind == 1 {
        (li_u + lj_u + 3, lj_u + 2)
    } else {
        (li_u + lj_u + 1, lj_u)
    };
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * 3 * nci * ncj;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    // Scratch + output buffers.
    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let d2g0_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let d2g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($op:expr) => {
            one_electron_grad_bra_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(g1_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(d2g0_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(d2g1_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                SQRTPI,
                std::f64::consts::PI,
                li,
                lj,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                $op,
            )
        };
    }

    if op_kind == 0 {
        launch_with!(0u32);
    } else {
        launch_with!(1u32);
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_grad_bra_device`] (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_1e_grad_bra_on_backend(
    backend: &ResolvedBackend,
    op_kind: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_grad_bra_device::<cubecl::cpu::CpuRuntime>(
            client, op_kind, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_grad_bra_device::<cubecl_wgpu::WgpuRuntime>(
            client, op_kind, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_grad_bra_device::<cubecl_cuda::CudaRuntime>(
            client, op_kind, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_grad_bra_device::<cubecl_hip::HipRuntime>(
            client, op_kind, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_grad_bra_device::<cubecl_wgpu::WgpuRuntime>(
            client, op_kind, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — both-side rank-9 OVERLAP gradient
//
//  Implements `int1e_ipovlpip` = <NABLA i | OVLP | NABLA j> (libcint
//  `src/autocode/hess.c` `CINTgout1e_int1e_ipovlpip`). Extends
//  `one_electron_grad_bra_kernel`: builds the overlap base G-tensor g0 with
//  BOTH-side headroom (nmax = li+lj+2, lj_ext = lj+1), then three derivative
//  tensors —
//    g1 = D_j(g0)        ket nabla   (filled over i..=li+1, j..=lj)
//    g2 = D_i(g0)        bra nabla   (filled over i..=li,   j..=lj)
//    g3 = D_i(g1)        mixed both  (filled over i..=li,   j..=lj)
//  matching libcint's `G1E_D_J(g1,g0,i_l+1,j_l)`, `G1E_D_I(g2,g0,i_l,j_l)`,
//  `G1E_D_I(g3,g1,i_l,j_l)`. The 9 = 3×3 components use libcint's bra-major,
//  DIRECT (un-permuted) order `comp = bra_axis*3 + ket_axis`:
//    s0=g3x·g0y·g0z  s1=g2x·g1y·g0z  s2=g2x·g0y·g1z
//    s3=g1x·g2y·g0z  s4=g0x·g3y·g0z  s5=g0x·g2y·g1z
//    s6=g1x·g0y·g2z  s7=g0x·g1y·g2z  s8=g0x·g0y·g3z
//  Output is component-leading per (ci,cj):
//    out[(ci*nctr_j+cj)*9*block + comp*block + cj_idx*nci + ci_idx].
// ─────────────────────────────────────────────────────────────────────────────

/// On-device both-side rank-9 overlap gradient (`int1e_ipovlpip`).
///
/// Single work item; generic over `F: Float`. Faithful device port of libcint's
/// `CINTgout1e_int1e_ipovlpip`. Scratch buffers `g`/`g1`/`g2`/`g3` hold
/// g0, D_j(g0), D_i(g0), D_i(D_j(g0)). `cart_out` is the 9-component
/// component-leading accumulator of length `9 * nci * ncj * nctr_i * nctr_j`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_grad_both_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    g3: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    sqrtpi: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
) {
    if UNIT_POS == 0u32 {
        // Both-side overlap headroom: g0 must span i..=li+1 AND j..=lj+1.
        let nmax = li + lj + 2u32;
        let lj_ext = lj + 1u32;
        let dj = nmax + 1u32; // stride between consecutive j-levels within an axis block
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_len = 9u32 * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        // Zero the full accumulation buffer.
        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        // ── Primitive loop ───────────────────────────────────────────────────
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                // Pair data, computed in-kernel in F (norm_i = norm_j = 1.0).
                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // ── Build OVERLAP base G-tensor in `g` (fixed-center VRR + HRR) ──
                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                // Zero the three derivative tensors.
                let mut zi = 0u32;
                while zi < total_g {
                    g1[zi as usize] = F::new(0.0);
                    g2[zi as usize] = F::new(0.0);
                    g3[zi as usize] = F::new(0.0);
                    zi += 1u32;
                }

                let ai2 = F::new(-2.0) * ai;
                let aj2 = F::new(-2.0) * aj;
                let li1 = li + 1u32;

                // ── g1 = D_j(g0): ket nabla, over i..=li+1, j..=lj. ─────────────
                //   D_j[j=0] = -2*aj * g0[j=1]
                //   D_j[j>0] = j * g0[j-1] + (-2*aj) * g0[j+1]
                let mut a1 = 0u32;
                while a1 < 3u32 {
                    let off = a1 * g_per_axis;
                    let mut jn = 0u32;
                    while jn <= lj {
                        let jbase = jn * dj;
                        let jhi = (jn + 1u32) * dj;
                        let mut ii = 0u32;
                        while ii <= li1 {
                            let mut val = aj2 * g[(off + jhi + ii) as usize];
                            if jn >= 1u32 {
                                let jlo = (jn - 1u32) * dj;
                                val = F::cast_from(jn) * g[(off + jlo + ii) as usize] + val;
                            }
                            g1[(off + jbase + ii) as usize] = val;
                            ii += 1u32;
                        }
                        jn += 1u32;
                    }
                    a1 += 1u32;
                }

                // ── g2 = D_i(g0) and g3 = D_i(g1): bra nabla, over j..=lj, i..=li. ─
                //   D_i[i=0] = -2*ai * src[i=1]
                //   D_i[i>0] = i * src[i-1] + (-2*ai) * src[i+1]
                let mut a2 = 0u32;
                while a2 < 3u32 {
                    let off = a2 * g_per_axis;
                    let mut jn = 0u32;
                    while jn <= lj {
                        let jbase = jn * dj;
                        g2[(off + jbase) as usize] = ai2 * g[(off + jbase + 1u32) as usize];
                        g3[(off + jbase) as usize] = ai2 * g1[(off + jbase + 1u32) as usize];
                        let mut ii = 1u32;
                        while ii <= li {
                            g2[(off + jbase + ii) as usize] = F::cast_from(ii)
                                * g[(off + jbase + ii - 1u32) as usize]
                                + ai2 * g[(off + jbase + ii + 1u32) as usize];
                            g3[(off + jbase + ii) as usize] = F::cast_from(ii)
                                * g1[(off + jbase + ii - 1u32) as usize]
                                + ai2 * g1[(off + jbase + ii + 1u32) as usize];
                            ii += 1u32;
                        }
                        jn += 1u32;
                    }
                    a2 += 1u32;
                }

                // ── Contract into every (ci,cj) block — 9 components. ───────────
                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;
                        let base = (ci * nctr_j + cj) * total_len;

                        // cart_comps order: cj outer (lx descending), ci inner.
                        let mut cj_idx = 0u32;
                        let mut ja = 0u32;
                        while ja <= lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            let mut jb = 0u32;
                            while jb <= lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

                                let mut ci_idx = 0u32;
                                let mut ia = 0u32;
                                while ia <= li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    let mut ib = 0u32;
                                    while ib <= li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        let nx = jx * dj + ix;
                                        let ny = jy * dj + iy;
                                        let nz = jz * dj + iz;

                                        let g0x = g[(gx + nx) as usize];
                                        let g0y = g[(gy + ny) as usize];
                                        let g0z = g[(gz + nz) as usize];
                                        let g1x = g1[(gx + nx) as usize];
                                        let g1y = g1[(gy + ny) as usize];
                                        let g1z = g1[(gz + nz) as usize];
                                        let g2x = g2[(gx + nx) as usize];
                                        let g2y = g2[(gy + ny) as usize];
                                        let g2z = g2[(gz + nz) as usize];
                                        let g3x = g3[(gx + nx) as usize];
                                        let g3y = g3[(gy + ny) as usize];
                                        let g3z = g3[(gz + nz) as usize];

                                        // libcint hess.c CINTgout1e_int1e_ipovlpip order.
                                        let s0 = g3x * g0y * g0z;
                                        let s1 = g2x * g1y * g0z;
                                        let s2 = g2x * g0y * g1z;
                                        let s3 = g1x * g2y * g0z;
                                        let s4 = g0x * g3y * g0z;
                                        let s5 = g0x * g2y * g1z;
                                        let s6 = g1x * g0y * g2z;
                                        let s7 = g0x * g1y * g2z;
                                        let s8 = g0x * g0y * g3z;

                                        let elem = cj_idx * nci + ci_idx;
                                        cart_out[(base + elem) as usize] += weight * s0;
                                        cart_out[(base + block_len + elem) as usize] +=
                                            weight * s1;
                                        cart_out[(base + 2u32 * block_len + elem) as usize] +=
                                            weight * s2;
                                        cart_out[(base + 3u32 * block_len + elem) as usize] +=
                                            weight * s3;
                                        cart_out[(base + 4u32 * block_len + elem) as usize] +=
                                            weight * s4;
                                        cart_out[(base + 5u32 * block_len + elem) as usize] +=
                                            weight * s5;
                                        cart_out[(base + 6u32 * block_len + elem) as usize] +=
                                            weight * s6;
                                        cart_out[(base + 7u32 * block_len + elem) as usize] +=
                                            weight * s7;
                                        cart_out[(base + 8u32 * block_len + elem) as usize] +=
                                            weight * s8;

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }

                                cj_idx += 1u32;
                                jb += 1u32;
                            }
                            ja += 1u32;
                        }

                        cj += 1u32;
                    }
                    ci += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`one_electron_grad_both_kernel`] at `f64` on a resolved backend's
/// client. Returns the 9-component component-leading cart buffer of length
/// `9 * nci * ncj * nctr_i * nctr_j`. Buffer creation clones
/// [`run_1e_grad_bra_device`].
#[allow(clippy::too_many_arguments)]
fn run_1e_grad_both_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax_u = li_u + lj_u + 2;
    let lj_ext_u = lj_u + 1;
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * 9 * nci * ncj;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    // Scratch + output buffers.
    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g2_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g3_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    one_electron_grad_both_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
        unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(g1_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(g2_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(g3_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
        ri[0],
        ri[1],
        ri[2],
        rj[0],
        rj[1],
        rj[2],
        SQRTPI,
        std::f64::consts::PI,
        li,
        lj,
        nprim_i,
        nprim_j,
        nctr_i,
        nctr_j,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_grad_both_device`] (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_1e_grad_both_on_backend(
    backend: &ResolvedBackend,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_grad_both_device::<cubecl::cpu::CpuRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_grad_both_device::<cubecl_wgpu::WgpuRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_grad_both_device::<cubecl_cuda::CudaRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_grad_both_device::<cubecl_hip::HipRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_grad_both_device::<cubecl_wgpu::WgpuRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
    }
}

/// `#[cube]` helper: ket-direction nabla `dst = D_j(src)` for a 3-axis 1e
/// G-tensor, filled over j∈0..=jmax, i∈0..=imax. Stride `dj` between j-levels.
///   D_j[j=0] = -2*aj * src[j=1];  D_j[j>0] = j*src[j-1] + (-2*aj)*src[j+1].
#[cube]
#[allow(clippy::too_many_arguments)]
fn d_j_1e_into<F: Float + CubeElement>(
    dst: &mut Array<F>,
    src: &Array<F>,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    aj2: F,
) {
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            let jhi = (jn + 1u32) * dj;
            let mut ii = 0u32;
            while ii <= imax {
                let mut val = aj2 * src[(off + jhi + ii) as usize];
                if jn >= 1u32 {
                    val = F::cast_from(jn) * src[(off + (jn - 1u32) * dj + ii) as usize] + val;
                }
                dst[(off + jbase + ii) as usize] = val;
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `#[cube]` helper: bra-direction nabla `dst = D_i(src)` for a 3-axis 1e
/// G-tensor, filled over j∈0..=jmax, i∈0..=imax. Stride `dj` between j-levels.
///   D_i[i=0] = -2*ai * src[i=1];  D_i[i>0] = i*src[i-1] + (-2*ai)*src[i+1].
#[cube]
#[allow(clippy::too_many_arguments)]
fn d_i_1e_into<F: Float + CubeElement>(
    dst: &mut Array<F>,
    src: &Array<F>,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    ai2: F,
) {
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            dst[(off + jbase) as usize] = ai2 * src[(off + jbase + 1u32) as usize];
            let mut ii = 1u32;
            while ii <= imax {
                dst[(off + jbase + ii) as usize] = F::cast_from(ii)
                    * src[(off + jbase + ii - 1u32) as usize]
                    + ai2 * src[(off + jbase + ii + 1u32) as usize];
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — both-side rank-9 KINETIC gradient
//
//  Implements `int1e_ipkinip` = <NABLA i | P DOT P | NABLA j> (libcint
//  `src/autocode/hess.c` `CINTgout1e_int1e_ipkinip`). The kinetic operator's
//  Laplacian + the outer ket ip require up to a THIRD ket derivative, so libcint
//  builds g0..g15. Those reduce to 8 DISTINCT tensors:
//    dj0=g0, dj1=D_j(g0), dj2=D_j²(g0), dj3=D_j³(g0)         (ket orders 0..3)
//    di0=D_i(dj0), di1=D_i(dj1), di2=D_i(dj2), di3=D_i(dj3)   (+ one bra nabla)
//  (libcint's g1=g2=g4=dj1, g3=g5=g6=dj2, g7=dj3, g8=di0, g9=g10=g12=di1,
//   g11=g13=g14=di2, g15=di3). The 9 components are libcint's exact 27-term
//   recipe; the kinetic 0.5 (folded into `common_factor`) is applied as -0.5.
//  Headroom: g0 must span i..=li+1, j..=lj+3 → nmax=li+lj+4, lj_ext=lj+3.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device both-side rank-9 kinetic gradient (`int1e_ipkinip`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_grad_kin_both_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    dj1: &mut Array<F>,
    dj2: &mut Array<F>,
    dj3: &mut Array<F>,
    di0: &mut Array<F>,
    di1: &mut Array<F>,
    di2: &mut Array<F>,
    di3: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    sqrtpi: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
) {
    if UNIT_POS == 0u32 {
        // Kinetic both-side headroom: g0 spans i..=li+1, j..=lj+3.
        let nmax = li + lj + 4u32;
        let lj_ext = lj + 3u32;
        let dj = nmax + 1u32;
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_len = 9u32 * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let li1 = li + 1u32;
        let neg_half = F::new(-0.5);

        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // Build overlap base G-tensor g0 (= dj0) in `g`.
                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);
                one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                // Zero derivative tensors.
                let mut zi = 0u32;
                while zi < total_g {
                    dj1[zi as usize] = F::new(0.0);
                    dj2[zi as usize] = F::new(0.0);
                    dj3[zi as usize] = F::new(0.0);
                    di0[zi as usize] = F::new(0.0);
                    di1[zi as usize] = F::new(0.0);
                    di2[zi as usize] = F::new(0.0);
                    di3[zi as usize] = F::new(0.0);
                    zi += 1u32;
                }

                let ai2 = F::new(-2.0) * ai;
                let aj2 = F::new(-2.0) * aj;

                // Ket j-derivative chain (all at i..=li+1).
                d_j_1e_into::<F>(dj1, g, g_per_axis, dj, lj + 2u32, li1, aj2);
                d_j_1e_into::<F>(dj2, dj1, g_per_axis, dj, lj + 1u32, li1, aj2);
                d_j_1e_into::<F>(dj3, dj2, g_per_axis, dj, lj, li1, aj2);

                // Bra nabla of each ket-derivative tensor (at i..=li, j..=lj).
                d_i_1e_into::<F>(di0, g, g_per_axis, dj, lj, li, ai2);
                d_i_1e_into::<F>(di1, dj1, g_per_axis, dj, lj, li, ai2);
                d_i_1e_into::<F>(di2, dj2, g_per_axis, dj, lj, li, ai2);
                d_i_1e_into::<F>(di3, dj3, g_per_axis, dj, lj, li, ai2);

                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = neg_half * coeff_i_val * coeff_j_val;
                        let base = (ci * nctr_j + cj) * total_len;

                        let mut cj_idx = 0u32;
                        let mut ja = 0u32;
                        while ja <= lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            let mut jb = 0u32;
                            while jb <= lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

                                let mut ci_idx = 0u32;
                                let mut ia = 0u32;
                                while ia <= li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    let mut ib = 0u32;
                                    while ib <= li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        let nx = jx * dj + ix;
                                        let ny = jy * dj + iy;
                                        let nz = jz * dj + iz;

                                        // Per-axis tensor reads (ket order 0..3 = dj*, +bra = di*).
                                        let a0x = g[(gx + nx) as usize];
                                        let a0y = g[(gy + ny) as usize];
                                        let a0z = g[(gz + nz) as usize];
                                        let b1x = dj1[(gx + nx) as usize];
                                        let b1y = dj1[(gy + ny) as usize];
                                        let b1z = dj1[(gz + nz) as usize];
                                        let b2x = dj2[(gx + nx) as usize];
                                        let b2y = dj2[(gy + ny) as usize];
                                        let b2z = dj2[(gz + nz) as usize];
                                        let b3x = dj3[(gx + nx) as usize];
                                        let b3y = dj3[(gy + ny) as usize];
                                        let b3z = dj3[(gz + nz) as usize];
                                        let c0x = di0[(gx + nx) as usize];
                                        let c0y = di0[(gy + ny) as usize];
                                        let c0z = di0[(gz + nz) as usize];
                                        let c1x = di1[(gx + nx) as usize];
                                        let c1y = di1[(gy + ny) as usize];
                                        let c1z = di1[(gz + nz) as usize];
                                        let c2x = di2[(gx + nx) as usize];
                                        let c2y = di2[(gy + ny) as usize];
                                        let c2z = di2[(gz + nz) as usize];
                                        let c3x = di3[(gx + nx) as usize];
                                        let c3y = di3[(gy + ny) as usize];
                                        let c3z = di3[(gz + nz) as usize];

                                        // libcint hess.c CINTgout1e_int1e_ipkinip
                                        // (27 used terms; -0.5 folded into `weight`).
                                        let s0 = c3x * a0y * a0z + c1x * b2y * a0z + c1x * a0y * b2z;
                                        let s1 = c2x * b1y * a0z + c0x * b3y * a0z + c0x * b1y * b2z;
                                        let s2 = c2x * a0y * b1z + c0x * b2y * b1z + c0x * a0y * b3z;
                                        let s3 = b3x * c0y * a0z + b1x * c2y * a0z + b1x * c0y * b2z;
                                        let s4 = b2x * c1y * a0z + a0x * c3y * a0z + a0x * c1y * b2z;
                                        let s5 = b2x * c0y * b1z + a0x * c2y * b1z + a0x * c0y * b3z;
                                        let s6 = b3x * a0y * c0z + b1x * b2y * c0z + b1x * a0y * c2z;
                                        let s7 = b2x * b1y * c0z + a0x * b3y * c0z + a0x * b1y * c2z;
                                        let s8 = b2x * a0y * c1z + a0x * b2y * c1z + a0x * a0y * c3z;

                                        let elem = cj_idx * nci + ci_idx;
                                        cart_out[(base + elem) as usize] += weight * s0;
                                        cart_out[(base + block_len + elem) as usize] +=
                                            weight * s1;
                                        cart_out[(base + 2u32 * block_len + elem) as usize] +=
                                            weight * s2;
                                        cart_out[(base + 3u32 * block_len + elem) as usize] +=
                                            weight * s3;
                                        cart_out[(base + 4u32 * block_len + elem) as usize] +=
                                            weight * s4;
                                        cart_out[(base + 5u32 * block_len + elem) as usize] +=
                                            weight * s5;
                                        cart_out[(base + 6u32 * block_len + elem) as usize] +=
                                            weight * s6;
                                        cart_out[(base + 7u32 * block_len + elem) as usize] +=
                                            weight * s7;
                                        cart_out[(base + 8u32 * block_len + elem) as usize] +=
                                            weight * s8;

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }
                                cj_idx += 1u32;
                                jb += 1u32;
                            }
                            ja += 1u32;
                        }
                        cj += 1u32;
                    }
                    ci += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`one_electron_grad_kin_both_kernel`] at `f64` on a backend client.
#[allow(clippy::too_many_arguments)]
fn run_1e_grad_kin_both_device<R: Runtime>(
    client: &ComputeClient<R>,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax_u = li_u + lj_u + 4;
    let lj_ext_u = lj_u + 3;
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * 9 * nci * ncj;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let dj1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let dj2_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let dj3_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let di0_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let di1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let di2_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let di3_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    one_electron_grad_kin_both_kernel::launch::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
        unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
        unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(dj1_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(dj2_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(dj3_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(di0_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(di1_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(di2_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(di3_h.clone(), total_g) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
        ri[0],
        ri[1],
        ri[2],
        rj[0],
        rj[1],
        rj[2],
        SQRTPI,
        std::f64::consts::PI,
        li,
        lj,
        nprim_i,
        nprim_j,
        nctr_i,
        nctr_j,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_grad_kin_both_device`].
#[allow(clippy::too_many_arguments)]
fn run_1e_grad_kin_both_on_backend(
    backend: &ResolvedBackend,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_grad_kin_both_device::<cubecl::cpu::CpuRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_grad_kin_both_device::<cubecl_wgpu::WgpuRuntime>(
                client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
                coeff_j,
            )
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_grad_kin_both_device::<cubecl_cuda::CudaRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_grad_kin_both_device::<cubecl_hip::HipRuntime>(
            client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
            coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_grad_kin_both_device::<cubecl_wgpu::WgpuRuntime>(
                client, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j, coeff_i,
                coeff_j,
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — nuclear-attraction GRADIENT
//
//  Ports the host `contract_nuclear_grad` (ipnuc + iprinv share ONE kernel,
//  parameterized by an origins/charges array) onto the CubeCL device. Clones the
//  scalar nuclear arm's structure (in-kernel pdata recompute; `#[comptime] nroots`
//  selecting rys_root1..5; urys/wrys scratch) and adds the bra nabla1i + the
//  3-component component-leading output (verbatim from `contract_nuclear_grad`).
//
//  Origins are passed as two Arrays: `origin_coords` (norig*3 F, x/y/z per origin)
//  and `origin_charges` (norig F, the charge factor: -Z_C for ipnuc, +1.0 for
//  iprinv), plus `norig: u32`. Origins are accumulated low→high for bit-stable
//  reduction (D-10). nmax = li+lj+1 (+1 bra headroom), lj_ext = lj.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device nuclear-attraction gradient (`ipnuc` / `iprinv`, origins-parameterized).
///
/// Single work item; faithful port of `contract_nuclear_grad`. Scratch: `g`
/// (per-root nuclear G-tensor with +1 bra headroom), `g1` (bra-nabla of g),
/// `urys`/`wrys` (Rys roots). `cart_out` is the 3-component component-leading
/// accumulator (`(ci*nctr_j+cj)*total_len + comp*block_len + cj_idx*nci+ci_idx`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_nuc_grad_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    pie4: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    norig: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let nrys = nroots;
        // nmax = li+lj+1 (+1 bra headroom), lj_ext = lj.
        let nmax = li + lj + 1u32;
        let dj = nmax + 1u32;
        let g_per_axis = (nmax + 1u32) * (lj + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_len = 3u32 * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        // Zero the full accumulation buffer.
        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                let ai2 = F::new(-2.0) * ai;

                // Ordered accumulation over origins (low→high for ipnuc, single for iprinv).
                let mut orig = 0u32;
                while orig < norig {
                    let charge_factor = origin_charges[orig as usize];
                    let rcx = origin_coords[(orig * 3u32) as usize];
                    let rcy = origin_coords[(orig * 3u32 + 1u32) as usize];
                    let rcz = origin_coords[(orig * 3u32 + 2u32) as usize];

                    // crij = C - P.
                    let crijx = rcx - px;
                    let crijy = rcy - py;
                    let crijz = rcz - pz;
                    let x_boys =
                        zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                    // Rys roots/weights (comptime nroots).
                    if comptime!(nroots == 1u32) {
                        rys_root1::<F>(x_boys, urys, wrys, pie4);
                    } else if comptime!(nroots == 2u32) {
                        rys_root2::<F>(x_boys, urys, wrys, pie4);
                    } else if comptime!(nroots == 3u32) {
                        rys_root3::<F>(x_boys, urys, wrys, pie4);
                    } else if comptime!(nroots == 4u32) {
                        rys_root4::<F>(x_boys, urys, wrys, pie4);
                    } else {
                        rys_root5::<F>(x_boys, urys, wrys, pie4);
                    }

                    // fac1 = 2*PI * charge_factor * fac / zeta.
                    let fac1 = F::new(2.0) * pi_const * charge_factor * fac / zeta;

                    let mut irys: u32 = 0u32;
                    while irys < nrys {
                        let u_n = urys[irys as usize];
                        let w_n = wrys[irys as usize];
                        let tau = u_n / (F::new(1.0) + u_n);
                        let rt = aij2 * (F::new(1.0) - tau);

                        let c00x = (px - rix) + tau * crijx;
                        let c00y = (py - riy) + tau * crijy;
                        let c00z = (pz - riz) + tau * crijz;

                        // Build per-root G-tensor in `g` (root-dependent c00, b10=rt).
                        let mut gi = 0u32;
                        while gi < total_g {
                            g[gi as usize] = F::new(0.0);
                            gi += 1u32;
                        }
                        g[gx as usize] = F::new(1.0);
                        g[gy as usize] = F::new(1.0);
                        g[gz as usize] = fac1 * w_n;

                        one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                        if lj >= 1u32 {
                            one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj);
                            one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj);
                            one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj);
                        }

                        // Bra nabla1i → g1 over 0..=lj j-levels, all 3 axes.
                        let mut g1i = 0u32;
                        while g1i < total_g {
                            g1[g1i as usize] = F::new(0.0);
                            g1i += 1u32;
                        }
                        let mut axisn = 0u32;
                        while axisn < 3u32 {
                            let off = axisn * g_per_axis;
                            let mut jn = 0u32;
                            while jn <= lj {
                                let jbase = jn * dj;
                                g1[(off + jbase) as usize] =
                                    ai2 * g[(off + jbase + 1u32) as usize];
                                let mut ix = 1u32;
                                while ix <= li {
                                    g1[(off + jbase + ix) as usize] = F::cast_from(ix)
                                        * g[(off + jbase + ix - 1u32) as usize]
                                        + ai2 * g[(off + jbase + ix + 1u32) as usize];
                                    ix += 1u32;
                                }
                                jn += 1u32;
                            }
                            axisn += 1u32;
                        }

                        // Accumulate this root's 3-component contribution into every (ci,cj).
                        let mut ci = 0u32;
                        while ci < nctr_i {
                            let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                            let mut cj = 0u32;
                            while cj < nctr_j {
                                let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                                let weight = coeff_i_val * coeff_j_val;
                                let base = (ci * nctr_j + cj) * total_len;

                                let mut cj_idx = 0u32;
                                let mut ja = 0u32;
                                while ja <= lj {
                                    let jx = lj - ja;
                                    let lj_minus_jx = lj - jx;
                                    let mut jb = 0u32;
                                    while jb <= lj_minus_jx {
                                        let jy = lj_minus_jx - jb;
                                        let jz = lj - jx - jy;

                                        let mut ci_idx = 0u32;
                                        let mut ia = 0u32;
                                        while ia <= li {
                                            let ix = li - ia;
                                            let li_minus_ix = li - ix;
                                            let mut ib = 0u32;
                                            while ib <= li_minus_ix {
                                                let iy = li_minus_ix - ib;
                                                let iz = li - ix - iy;

                                                let nx = jx * dj + ix;
                                                let ny = jy * dj + iy;
                                                let nz = jz * dj + iz;

                                                let g0x = g[(gx + nx) as usize];
                                                let g0y = g[(gy + ny) as usize];
                                                let g0z = g[(gz + nz) as usize];
                                                let g1x = g1[(gx + nx) as usize];
                                                let g1y = g1[(gy + ny) as usize];
                                                let g1z = g1[(gz + nz) as usize];

                                                let elem = cj_idx * nci + ci_idx;
                                                cart_out[(base + elem) as usize] +=
                                                    weight * g1x * g0y * g0z;
                                                cart_out
                                                    [(base + block_len + elem) as usize] +=
                                                    weight * g0x * g1y * g0z;
                                                cart_out[(base
                                                    + 2u32 * block_len
                                                    + elem)
                                                    as usize] +=
                                                    weight * g0x * g0y * g1z;

                                                ci_idx += 1u32;
                                                ib += 1u32;
                                            }
                                            ia += 1u32;
                                        }
                                        cj_idx += 1u32;
                                        jb += 1u32;
                                    }
                                    ja += 1u32;
                                }
                                cj += 1u32;
                            }
                            ci += 1u32;
                        }

                        irys += 1u32;
                    }
                    orig += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`one_electron_nuc_grad_kernel`] at `f64` on a resolved backend's
/// client. Returns the 3-component component-leading accumulator of length
/// `3 * nci * ncj * nctr_i * nctr_j`. `nroots` (1..=5) selects the `rys_rootN`
/// monomorphization at the `launch::<f64, R>` call site.
#[allow(clippy::too_many_arguments)]
fn run_1e_nuc_grad_device<R: Runtime>(
    client: &ComputeClient<R>,
    nroots: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax_u = li_u + lj_u + 1;
    let g_per_axis = (nmax_u + 1) * (lj_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * 3 * nci * ncj;
    let nroots_u = nroots as usize;
    let norig = origin_charges.len() as u32;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    // origins are never empty (ipnuc >= 1 nucleus, iprinv = 1 origin), but guard
    // the len>0 CubeCL Array contract anyway (T-j7d-04).
    let coords_src = if origin_coords.is_empty() {
        &[0.0_f64][..]
    } else {
        origin_coords
    };
    let charges_src = if origin_charges.is_empty() {
        &[0.0_f64][..]
    } else {
        origin_charges
    };
    let coords_h = client.create_from_slice(f64::as_bytes(coords_src));
    let charges_h = client.create_from_slice(f64::as_bytes(charges_src));

    // Scratch + output buffers.
    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($nr:expr) => {
            one_electron_nuc_grad_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coords_h.clone(), coords_src.len()) },
                unsafe { ArrayArg::from_raw_parts(charges_h.clone(), charges_src.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(g1_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(u_h.clone(), nroots_u) },
                unsafe { ArrayArg::from_raw_parts(w_h.clone(), nroots_u) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                PIE4,
                std::f64::consts::PI,
                li,
                lj,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                norig,
                $nr,
            )
        };
    }

    match nroots {
        1 => launch_with!(1u32),
        2 => launch_with!(2u32),
        3 => launch_with!(3u32),
        4 => launch_with!(4u32),
        _ => launch_with!(5u32),
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_nuc_grad_device`] (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_1e_nuc_grad_on_backend(
    backend: &ResolvedBackend,
    nroots: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_nuc_grad_device::<cubecl::cpu::CpuRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_nuc_grad_device::<cubecl_wgpu::WgpuRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_nuc_grad_device::<cubecl_cuda::CudaRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_nuc_grad_device::<cubecl_hip::HipRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_nuc_grad_device::<cubecl_wgpu::WgpuRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — both-side rank-9 NUCLEAR gradient
//
//  Implements `int1e_ipnucip` = <NABLA i | NUC | NABLA j> (libcint
//  `src/autocode/hess.c` `CINTgout1e_int1e_ipnucip`). Structurally identical to
//  the overlap both-side kernel (g1=D_j(g0), g2=D_i(g0), g3=D_i(g1); 9 comps in
//  bra-major DIRECT order), but the base g0 is the NUCLEAR Rys G-tensor and the
//  9 components are summed over Rys roots and origins (with -Z_C / +1 folded into
//  g0 via `charge_factor`). Headroom mirrors the overlap both-side kernel:
//  nmax = li+lj+2, lj_ext = lj+1. nroots = nmax/2 + 1 (one extra root vs the
//  single-side nuclear gradient for the added ket headroom).
// ─────────────────────────────────────────────────────────────────────────────

/// On-device both-side rank-9 nuclear-attraction gradient (`int1e_ipnucip`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_nuc_grad_both_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    g3: &mut Array<F>,
    urys: &mut Array<F>,
    wrys: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    pie4: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    norig: u32,
    #[comptime] nroots: u32,
) {
    if UNIT_POS == 0u32 {
        let nrys = nroots;
        // Both-side headroom: g0 spans i..=li+1, j..=lj+1.
        let nmax = li + lj + 2u32;
        let lj_ext = lj + 1u32;
        let dj = nmax + 1u32;
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_len = 9u32 * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let li1 = li + 1u32;

        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                let ai2 = F::new(-2.0) * ai;
                let aj2 = F::new(-2.0) * aj;

                let mut orig = 0u32;
                while orig < norig {
                    let charge_factor = origin_charges[orig as usize];
                    let rcx = origin_coords[(orig * 3u32) as usize];
                    let rcy = origin_coords[(orig * 3u32 + 1u32) as usize];
                    let rcz = origin_coords[(orig * 3u32 + 2u32) as usize];

                    let crijx = rcx - px;
                    let crijy = rcy - py;
                    let crijz = rcz - pz;
                    let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                    if comptime!(nroots == 1u32) {
                        rys_root1::<F>(x_boys, urys, wrys, pie4);
                    } else if comptime!(nroots == 2u32) {
                        rys_root2::<F>(x_boys, urys, wrys, pie4);
                    } else if comptime!(nroots == 3u32) {
                        rys_root3::<F>(x_boys, urys, wrys, pie4);
                    } else if comptime!(nroots == 4u32) {
                        rys_root4::<F>(x_boys, urys, wrys, pie4);
                    } else {
                        rys_root5::<F>(x_boys, urys, wrys, pie4);
                    }

                    let fac1 = F::new(2.0) * pi_const * charge_factor * fac / zeta;

                    let mut irys: u32 = 0u32;
                    while irys < nrys {
                        let u_n = urys[irys as usize];
                        let w_n = wrys[irys as usize];
                        let tau = u_n / (F::new(1.0) + u_n);
                        let rt = aij2 * (F::new(1.0) - tau);

                        let c00x = (px - rix) + tau * crijx;
                        let c00y = (py - riy) + tau * crijy;
                        let c00z = (pz - riz) + tau * crijz;

                        // Per-root nuclear G-tensor in `g`.
                        let mut gi = 0u32;
                        while gi < total_g {
                            g[gi as usize] = F::new(0.0);
                            gi += 1u32;
                        }
                        g[gx as usize] = F::new(1.0);
                        g[gy as usize] = F::new(1.0);
                        g[gz as usize] = fac1 * w_n;

                        one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                        one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                        // Zero + build the three both-side derivatives.
                        let mut zi = 0u32;
                        while zi < total_g {
                            g1[zi as usize] = F::new(0.0);
                            g2[zi as usize] = F::new(0.0);
                            g3[zi as usize] = F::new(0.0);
                            zi += 1u32;
                        }
                        d_j_1e_into::<F>(g1, g, g_per_axis, dj, lj, li1, aj2);
                        d_i_1e_into::<F>(g2, g, g_per_axis, dj, lj, li, ai2);
                        d_i_1e_into::<F>(g3, g1, g_per_axis, dj, lj, li, ai2);

                        // Accumulate this root's 9-component contribution.
                        let mut ci = 0u32;
                        while ci < nctr_i {
                            let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                            let mut cj = 0u32;
                            while cj < nctr_j {
                                let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                                let weight = coeff_i_val * coeff_j_val;
                                let base = (ci * nctr_j + cj) * total_len;

                                let mut cj_idx = 0u32;
                                let mut ja = 0u32;
                                while ja <= lj {
                                    let jx = lj - ja;
                                    let lj_minus_jx = lj - jx;
                                    let mut jb = 0u32;
                                    while jb <= lj_minus_jx {
                                        let jy = lj_minus_jx - jb;
                                        let jz = lj - jx - jy;

                                        let mut ci_idx = 0u32;
                                        let mut ia = 0u32;
                                        while ia <= li {
                                            let ix = li - ia;
                                            let li_minus_ix = li - ix;
                                            let mut ib = 0u32;
                                            while ib <= li_minus_ix {
                                                let iy = li_minus_ix - ib;
                                                let iz = li - ix - iy;

                                                let nx = jx * dj + ix;
                                                let ny = jy * dj + iy;
                                                let nz = jz * dj + iz;

                                                let g0x = g[(gx + nx) as usize];
                                                let g0y = g[(gy + ny) as usize];
                                                let g0z = g[(gz + nz) as usize];
                                                let g1x = g1[(gx + nx) as usize];
                                                let g1y = g1[(gy + ny) as usize];
                                                let g1z = g1[(gz + nz) as usize];
                                                let g2x = g2[(gx + nx) as usize];
                                                let g2y = g2[(gy + ny) as usize];
                                                let g2z = g2[(gz + nz) as usize];
                                                let g3x = g3[(gx + nx) as usize];
                                                let g3y = g3[(gy + ny) as usize];
                                                let g3z = g3[(gz + nz) as usize];

                                                let s0 = g3x * g0y * g0z;
                                                let s1 = g2x * g1y * g0z;
                                                let s2 = g2x * g0y * g1z;
                                                let s3 = g1x * g2y * g0z;
                                                let s4 = g0x * g3y * g0z;
                                                let s5 = g0x * g2y * g1z;
                                                let s6 = g1x * g0y * g2z;
                                                let s7 = g0x * g1y * g2z;
                                                let s8 = g0x * g0y * g3z;

                                                let elem = cj_idx * nci + ci_idx;
                                                cart_out[(base + elem) as usize] += weight * s0;
                                                cart_out[(base + block_len + elem) as usize] +=
                                                    weight * s1;
                                                cart_out
                                                    [(base + 2u32 * block_len + elem) as usize] +=
                                                    weight * s2;
                                                cart_out
                                                    [(base + 3u32 * block_len + elem) as usize] +=
                                                    weight * s3;
                                                cart_out
                                                    [(base + 4u32 * block_len + elem) as usize] +=
                                                    weight * s4;
                                                cart_out
                                                    [(base + 5u32 * block_len + elem) as usize] +=
                                                    weight * s5;
                                                cart_out
                                                    [(base + 6u32 * block_len + elem) as usize] +=
                                                    weight * s6;
                                                cart_out
                                                    [(base + 7u32 * block_len + elem) as usize] +=
                                                    weight * s7;
                                                cart_out
                                                    [(base + 8u32 * block_len + elem) as usize] +=
                                                    weight * s8;

                                                ci_idx += 1u32;
                                                ib += 1u32;
                                            }
                                            ia += 1u32;
                                        }
                                        cj_idx += 1u32;
                                        jb += 1u32;
                                    }
                                    ja += 1u32;
                                }
                                cj += 1u32;
                            }
                            ci += 1u32;
                        }

                        irys += 1u32;
                    }
                    orig += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Dispatch [`one_electron_nuc_grad_both_kernel`] at `f64` on a backend client.
/// Returns the 9-component component-leading accumulator. `nroots` (1..=5)
/// selects the `rys_rootN` monomorphization at the `launch::<f64, R>` site.
#[allow(clippy::too_many_arguments)]
fn run_1e_nuc_grad_both_device<R: Runtime>(
    client: &ComputeClient<R>,
    nroots: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nmax_u = li_u + lj_u + 2;
    let lj_ext_u = lj_u + 1;
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * 9 * nci * ncj;
    let norig = origin_charges.len();

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let oc_h = client.create_from_slice(f64::as_bytes(origin_coords));
    let och_h = client.create_from_slice(f64::as_bytes(origin_charges));

    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g1_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g2_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let g3_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let urys_zero = vec![0.0_f64; nroots as usize];
    let urys_h = client.create_from_slice(f64::as_bytes(&urys_zero));
    let wrys_h = client.create_from_slice(f64::as_bytes(&urys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($nr:expr) => {
            one_electron_nuc_grad_both_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(oc_h.clone(), origin_coords.len()) },
                unsafe { ArrayArg::from_raw_parts(och_h.clone(), norig) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(g1_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(g2_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(g3_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(urys_h.clone(), nroots as usize) },
                unsafe { ArrayArg::from_raw_parts(wrys_h.clone(), nroots as usize) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                PIE4,
                std::f64::consts::PI,
                li,
                lj,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                norig as u32,
                $nr,
            )
        };
    }

    match nroots {
        1 => launch_with!(1u32),
        2 => launch_with!(2u32),
        3 => launch_with!(3u32),
        4 => launch_with!(4u32),
        _ => launch_with!(5u32),
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_nuc_grad_both_device`].
#[allow(clippy::too_many_arguments)]
fn run_1e_nuc_grad_both_on_backend(
    backend: &ResolvedBackend,
    nroots: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_nuc_grad_both_device::<cubecl::cpu::CpuRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_nuc_grad_both_device::<cubecl_wgpu::WgpuRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_nuc_grad_both_device::<cubecl_cuda::CudaRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_nuc_grad_both_device::<cubecl_hip::HipRuntime>(
            client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
            coeff_i, coeff_j, origin_coords, origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_nuc_grad_both_device::<cubecl_wgpu::WgpuRuntime>(
                client, nroots, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri, rj, exps_i, exps_j,
                coeff_i, coeff_j, origin_coords, origin_charges,
            )
        }
    }
}

/// Dispatch [`one_electron_scalar_kernel`] at `f64` on a resolved backend's
/// client and read back the contraction-major Cartesian accumulator.
///
/// Generic over `R: Runtime` so the same path serves CPU, ROCm, etc. Intermediate
/// device compute is `f64` (module-level precision policy, mirroring 2c2e). The
/// `op_kind` / `nroots` comptime args are selected at the `launch::<f64, R>` call
/// site by a small host-side match (CubeCL cannot pass comptime args dynamically).
#[allow(clippy::too_many_arguments)]
fn run_1e_scalar_device<R: Runtime>(
    client: &ComputeClient<R>,
    op_kind: u32,
    nroots: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    atom_coords: &[f64],
    atom_charges: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let (nmax_u, lj_ext_u) = if op_kind == 1 {
        (li_u + lj_u + 2, lj_u + 2)
    } else {
        (li_u + lj_u, lj_u)
    };
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * nci * ncj;
    let nroots_u = nroots as usize;
    let natm = atom_charges.len() as u32;

    // Input buffers.
    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    // atom_coords / atom_charges must be non-empty (CubeCL Array len > 0).
    let coords_src = if atom_coords.is_empty() {
        &[0.0_f64][..]
    } else {
        atom_coords
    };
    let charges_src = if atom_charges.is_empty() {
        &[0.0_f64][..]
    } else {
        atom_charges
    };
    let coords_h = client.create_from_slice(f64::as_bytes(coords_src));
    let charges_h = client.create_from_slice(f64::as_bytes(charges_src));

    // Scratch + output buffers.
    let g_zero = vec![0.0_f64; 3 * g_per_axis];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let rys_zero = vec![0.0_f64; nroots_u];
    let u_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let w_h = client.create_from_slice(f64::as_bytes(&rys_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    // Comptime op_kind / nroots: select the monomorphization at the call site.
    macro_rules! launch_with {
        ($op:expr, $nr:expr) => {
            one_electron_scalar_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coords_h.clone(), coords_src.len()) },
                unsafe { ArrayArg::from_raw_parts(charges_h.clone(), charges_src.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), 3 * g_per_axis) },
                unsafe { ArrayArg::from_raw_parts(u_h.clone(), nroots_u) },
                unsafe { ArrayArg::from_raw_parts(w_h.clone(), nroots_u) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                PIE4,
                SQRTPI,
                std::f64::consts::PI,
                li,
                lj,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                natm,
                $op,
                $nr,
            )
        };
    }

    // overlap (op_kind=0) / kinetic (op_kind=1) use nroots=1 (no Rys).
    // nuclear (op_kind=2) selects rys_rootN for nroots in 1..=5.
    if op_kind == 0 {
        launch_with!(0u32, 1u32);
    } else if op_kind == 1 {
        launch_with!(1u32, 1u32);
    } else {
        match nroots {
            1 => launch_with!(2u32, 1u32),
            2 => launch_with!(2u32, 2u32),
            3 => launch_with!(2u32, 3u32),
            4 => launch_with!(2u32, 4u32),
            _ => launch_with!(2u32, 5u32),
        }
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_scalar_device`] — copies the exact arm
/// set used by `center_2c2e.rs` / `f12.rs` (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_1e_scalar_on_backend(
    backend: &ResolvedBackend,
    op_kind: u32,
    nroots: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    atom_coords: &[f64],
    atom_charges: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_scalar_device::<cubecl::cpu::CpuRuntime>(
            client,
            op_kind,
            nroots,
            li,
            lj,
            nprim_i,
            nprim_j,
            nctr_i,
            nctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            atom_coords,
            atom_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client,
            op_kind,
            nroots,
            li,
            lj,
            nprim_i,
            nprim_j,
            nctr_i,
            nctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            atom_coords,
            atom_charges,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_scalar_device::<cubecl_cuda::CudaRuntime>(
            client,
            op_kind,
            nroots,
            li,
            lj,
            nprim_i,
            nprim_j,
            nctr_i,
            nctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            atom_coords,
            atom_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_scalar_device::<cubecl_hip::HipRuntime>(
            client,
            op_kind,
            nroots,
            li,
            lj,
            nprim_i,
            nprim_j,
            nctr_i,
            nctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            atom_coords,
            atom_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_scalar_device::<cubecl_wgpu::WgpuRuntime>(
            client,
            op_kind,
            nroots,
            li,
            lj,
            nprim_i,
            nprim_j,
            nctr_i,
            nctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            atom_coords,
            atom_charges,
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 24 Cluster A — parameterized overlap-derived moment kernel.
//
// One #[cube] kernel evaluates ALL of r / rr / rrr / rrrr / r2 / r4 / z / zz and
// their _origj variants. It builds the overlap G-tensor to `lj_ext = lj +
// moment_order` ket levels (ket headroom raised on ng[1], NOT the bra — D-07),
// then forms per-axis moment-power values via the closed-form binomial expansion
// of libcint's repeated `CINTx1j_1e` (g1e.c:453):
//
//   m_p[jx] = Σ_{t=0..p} C(p,t) · drj^(p-t) · overlap[jx+t]
//
// where `drj = rj - origin`. The origin-source branch (D-02) is realized purely
// host-side: the launcher passes `drjx/drjy/drjz = rj - origin`, with
//   origin = common_orig (base family, G1E_RCJ)  |  rj (_origj, G1E_R_J → drj=0).
//
// Components are emitted in the VERBATIM libcint gout order (intor1.c). Because
// each `s[k]` in libcint factorizes as (moment-power on x)·(power on y)·(power on
// z) and the value at the final ket position depends only on the per-axis power
// (the higher `j_l+n` build levels in libcint coincide at the emit index), the
// per-axis-power product reproduces libcint's gout byte-for-byte. The component
// ORDER per op_mode is the canonical nesting libcint emits:
//   r    (mode 0, rank 3):  x, y, z
//   rr   (mode 1, rank 9):  3×3 row-major (k = a*3 + b)
//   rrr  (mode 2, rank 27): 3×3×3 (k = a*9 + b*3 + c)
//   rrrr (mode 3, rank 81): 3×3×3×3 (k = a*27 + b*9 + c*3 + d)
//   r2   (mode 4, rank 1):  trace m2x + m2y + m2z (= s0+s4+s8 of rr)
//   r4   (mode 5, rank 1):  m4x+2 m2x m2y+2 m2x m2z+m4y+2 m2y m2z+m4z
//                           (= s0+2s4+2s8+s40+2s44+s80 of rrrr)
//   z    (mode 6, rank 1):  m1z              (= s[2] of r)
//   zz   (mode 7, rank 1):  m2z              (= s[8] of rr)
// ─────────────────────────────────────────────────────────────────────────────

/// On-device parameterized Cluster-A moment kernel (overlap × position-power).
///
/// `op_mode` selects the family/component layout; `moment_order` (1..=4) is the
/// ket headroom and the depth of the per-axis moment ladder; `rank` is the
/// output component count (3/9/27/81/1). `cart_out` is component-leading:
/// `cart_out[(ci*nctr_j+cj)*rank*block_len + comp*block_len + cj_idx*nci+ci_idx]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn one_electron_moment_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    drjx: F,
    drjy: F,
    drjz: F,
    sqrtpi: F,
    pi_const: F,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    #[comptime] op_mode: u32,
    #[comptime] moment_order: u32,
    #[comptime] rank: u32,
) {
    if UNIT_POS == 0u32 {
        // Ket headroom: overlap G-tensor must span j..=lj+moment_order so the
        // per-axis moment ladder can read overlap[jx + t] for t up to moment_order.
        let nmax = li + lj + moment_order;
        let lj_ext = lj + moment_order;
        let dj = nmax + 1u32;
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let total_g = 3u32 * g_per_axis;
        let gx = 0u32;
        let gy = g_per_axis;
        let gz = 2u32 * g_per_axis;

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;
        let total_len = rank * block_len;
        let out_total = nctr_i * nctr_j * total_len;

        let mut oi = 0u32;
        while oi < out_total {
            cart_out[oi as usize] = F::new(0.0);
            oi += 1u32;
        }

        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps_i[pi as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps_j[pj as usize];

                let zeta = ai + aj;
                let aij2 = F::new(0.5) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                // Build the OVERLAP base G-tensor (fixed-center VRR + HRR to lj_ext).
                let mut gi = 0u32;
                while gi < total_g {
                    g[gi as usize] = F::new(0.0);
                    gi += 1u32;
                }
                g[gx as usize] = F::new(1.0);
                g[gy as usize] = F::new(1.0);
                g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                if lj_ext >= 1u32 {
                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                }

                let mut ci = 0u32;
                while ci < nctr_i {
                    let coeff_i_val = coeff_i[(pi * nctr_i + ci) as usize];
                    let mut cj = 0u32;
                    while cj < nctr_j {
                        let coeff_j_val = coeff_j[(pj * nctr_j + cj) as usize];
                        let weight = coeff_i_val * coeff_j_val;
                        let base = (ci * nctr_j + cj) * total_len;

                        let mut cj_idx = 0u32;
                        let mut ja = 0u32;
                        while ja <= lj {
                            let jx = lj - ja;
                            let lj_minus_jx = lj - jx;
                            let mut jb = 0u32;
                            while jb <= lj_minus_jx {
                                let jy = lj_minus_jx - jb;
                                let jz = lj - jx - jy;

                                let mut ci_idx = 0u32;
                                let mut ia = 0u32;
                                while ia <= li {
                                    let ix = li - ia;
                                    let li_minus_ix = li - ix;
                                    let mut ib = 0u32;
                                    while ib <= li_minus_ix {
                                        let iy = li_minus_ix - ib;
                                        let iz = li - ix - iy;

                                        // Per-axis moment-power ladders m0..m4
                                        // (only m0..m{moment_order} are meaningful;
                                        // higher entries stay zero and are unused).
                                        let mx0 = F::new(0.0);
                                        let mx1 = F::new(0.0);
                                        let mx2 = F::new(0.0);
                                        let mx3 = F::new(0.0);
                                        let mx4 = F::new(0.0);
                                        let my0 = F::new(0.0);
                                        let my1 = F::new(0.0);
                                        let my2 = F::new(0.0);
                                        let my3 = F::new(0.0);
                                        let my4 = F::new(0.0);
                                        let mz0 = F::new(0.0);
                                        let mz1 = F::new(0.0);
                                        let mz2 = F::new(0.0);
                                        let mz3 = F::new(0.0);
                                        let mz4 = F::new(0.0);
                                        let mut mx0 = mx0;
                                        let mut mx1 = mx1;
                                        let mut mx2 = mx2;
                                        let mut mx3 = mx3;
                                        let mut mx4 = mx4;
                                        let mut my0 = my0;
                                        let mut my1 = my1;
                                        let mut my2 = my2;
                                        let mut my3 = my3;
                                        let mut my4 = my4;
                                        let mut mz0 = mz0;
                                        let mut mz1 = mz1;
                                        let mut mz2 = mz2;
                                        let mut mz3 = mz3;
                                        let mut mz4 = mz4;

                                        // x axis
                                        moment_axis_ladder::<F>(
                                            g, gx, jx, dj, ix, drjx, moment_order, &mut mx0,
                                            &mut mx1, &mut mx2, &mut mx3, &mut mx4,
                                        );
                                        moment_axis_ladder::<F>(
                                            g, gy, jy, dj, iy, drjy, moment_order, &mut my0,
                                            &mut my1, &mut my2, &mut my3, &mut my4,
                                        );
                                        moment_axis_ladder::<F>(
                                            g, gz, jz, dj, iz, drjz, moment_order, &mut mz0,
                                            &mut mz1, &mut mz2, &mut mz3, &mut mz4,
                                        );

                                        let elem = cj_idx * nci + ci_idx;

                                        if comptime!(op_mode == 6u32) {
                                            // z: s[2] of r = m0x·m0y·m1z
                                            cart_out[(base + elem) as usize] +=
                                                weight * (mx0 * my0 * mz1);
                                        } else if comptime!(op_mode == 7u32) {
                                            // zz: s[8] of rr = m0x·m0y·m2z
                                            cart_out[(base + elem) as usize] +=
                                                weight * (mx0 * my0 * mz2);
                                        } else if comptime!(op_mode == 4u32) {
                                            // r2 trace: m2x + m2y + m2z (s0+s4+s8 of rr)
                                            let t = mx2 * my0 * mz0
                                                + mx0 * my2 * mz0
                                                + mx0 * my0 * mz2;
                                            cart_out[(base + elem) as usize] += weight * t;
                                        } else if comptime!(op_mode == 5u32) {
                                            // r4: s0+2s4+2s8+s40+2s44+s80 of rrrr
                                            let t = mx4 * my0 * mz0
                                                + F::new(2.0) * (mx2 * my2 * mz0)
                                                + F::new(2.0) * (mx2 * my0 * mz2)
                                                + mx0 * my4 * mz0
                                                + F::new(2.0) * (mx0 * my2 * mz2)
                                                + mx0 * my0 * mz4;
                                            cart_out[(base + elem) as usize] += weight * t;
                                        } else {
                                            // Full tensor families r/rr/rrr/rrrr:
                                            // emit rank components in canonical nesting
                                            // order via per-axis power digits.
                                            let mut comp = 0u32;
                                            while comp < rank {
                                                // decompose comp into moment_order
                                                // base-3 digits → per-axis powers.
                                                let rem = comp;
                                                let mut px_pow = 0u32;
                                                let mut py_pow = 0u32;
                                                let mut pz_pow = 0u32;
                                                let mut d = 0u32;
                                                while d < moment_order {
                                                    // most-significant digit first:
                                                    // weight = 3^(moment_order-1-d)
                                                    let mut w = 1u32;
                                                    let mut e = 0u32;
                                                    let exp = moment_order - 1u32 - d;
                                                    while e < exp {
                                                        w *= 3u32;
                                                        e += 1u32;
                                                    }
                                                    let digit = (rem / w) % 3u32;
                                                    if digit == 0u32 {
                                                        px_pow += 1u32;
                                                    } else if digit == 1u32 {
                                                        py_pow += 1u32;
                                                    } else {
                                                        pz_pow += 1u32;
                                                    }
                                                    d += 1u32;
                                                }

                                                let vx = moment_pick::<F>(
                                                    px_pow, mx0, mx1, mx2, mx3, mx4,
                                                );
                                                let vy = moment_pick::<F>(
                                                    py_pow, my0, my1, my2, my3, my4,
                                                );
                                                let vz = moment_pick::<F>(
                                                    pz_pow, mz0, mz1, mz2, mz3, mz4,
                                                );

                                                cart_out
                                                    [(base + comp * block_len + elem) as usize] +=
                                                    weight * (vx * vy * vz);
                                                comp += 1u32;
                                            }
                                        }

                                        ci_idx += 1u32;
                                        ib += 1u32;
                                    }
                                    ia += 1u32;
                                }

                                cj_idx += 1u32;
                                jb += 1u32;
                            }
                            ja += 1u32;
                        }

                        cj += 1u32;
                    }
                    ci += 1u32;
                }

                pj += 1u32;
            }
            pi += 1u32;
        }
    }
}

/// Per-axis moment-power ladder for the moment kernel.
///
/// Computes `m_p = Σ_{t=0..p} C(p,t) · drj^(p-t) · overlap[jx+t]` for p = 0..=4
/// (only p ≤ `moment_order` are filled; higher remain at their input value),
/// where `overlap[jx+t] = g[off + (jx+t)*dj + i]`. This is the closed-form of
/// libcint's repeated `CINTx1j_1e` (g1e.c:453). For `_origj`, `drj = 0` so the
/// ladder collapses to a pure ket-level shift (libcint's `G1E_R_J`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn moment_axis_ladder<F: Float>(
    g: &Array<F>,
    off: u32,
    jx: u32,
    dj: u32,
    i: u32,
    drj: F,
    #[comptime] moment_order: u32,
    m0: &mut F,
    m1: &mut F,
    m2: &mut F,
    m3: &mut F,
    m4: &mut F,
) {
    let ov0 = g[(off + jx * dj + i) as usize];
    *m0 = ov0;
    if comptime!(moment_order >= 1u32) {
        let ov1 = g[(off + (jx + 1u32) * dj + i) as usize];
        *m1 = ov1 + drj * ov0;
    }
    if comptime!(moment_order >= 2u32) {
        let ov1 = g[(off + (jx + 1u32) * dj + i) as usize];
        let ov2 = g[(off + (jx + 2u32) * dj + i) as usize];
        *m2 = ov2 + F::new(2.0) * drj * ov1 + drj * drj * ov0;
    }
    if comptime!(moment_order >= 3u32) {
        let ov1 = g[(off + (jx + 1u32) * dj + i) as usize];
        let ov2 = g[(off + (jx + 2u32) * dj + i) as usize];
        let ov3 = g[(off + (jx + 3u32) * dj + i) as usize];
        let d2 = drj * drj;
        *m3 = ov3
            + F::new(3.0) * drj * ov2
            + F::new(3.0) * d2 * ov1
            + d2 * drj * ov0;
    }
    if comptime!(moment_order >= 4u32) {
        let ov1 = g[(off + (jx + 1u32) * dj + i) as usize];
        let ov2 = g[(off + (jx + 2u32) * dj + i) as usize];
        let ov3 = g[(off + (jx + 3u32) * dj + i) as usize];
        let ov4 = g[(off + (jx + 4u32) * dj + i) as usize];
        let d2 = drj * drj;
        let d3 = d2 * drj;
        let d4 = d2 * d2;
        *m4 = ov4
            + F::new(4.0) * drj * ov3
            + F::new(6.0) * d2 * ov2
            + F::new(4.0) * d3 * ov1
            + d4 * ov0;
    }
}

/// Select the moment-power value `m{power}` from the per-axis ladder (power 0..4).
#[cube]
fn moment_pick<F: Float>(power: u32, m0: F, m1: F, m2: F, m3: F, m4: F) -> F {
    let mut out = m0;
    if power == 1u32 {
        out = m1;
    } else if power == 2u32 {
        out = m2;
    } else if power == 3u32 {
        out = m3;
    } else if power == 4u32 {
        out = m4;
    }
    out
}

/// Dispatch [`one_electron_moment_kernel`] at `f64` on a backend client. Returns
/// the component-leading cart buffer of length `rank * nci * ncj * nctr_i*nctr_j`.
#[allow(clippy::too_many_arguments)]
fn run_1e_moment_device<R: Runtime>(
    client: &ComputeClient<R>,
    op_mode: u32,
    moment_order: u32,
    rank: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    drj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let li_u = li as usize;
    let lj_u = lj as usize;
    let mo = moment_order as usize;
    let nmax_u = li_u + lj_u + mo;
    let lj_ext_u = lj_u + mo;
    let g_per_axis = (nmax_u + 1) * (lj_ext_u + 1);
    let total_g = 3 * g_per_axis;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * (rank as usize) * nci * ncj;

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));

    let g_zero = vec![0.0_f64; total_g];
    let g_h = client.create_from_slice(f64::as_bytes(&g_zero));
    let out_zero = vec![0.0_f64; out_len];
    let out_h = client.create_from_slice(f64::as_bytes(&out_zero));

    macro_rules! launch_with {
        ($mode:expr, $order:expr, $rank:expr) => {
            one_electron_moment_kernel::launch::<f64, R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), exps_i.len()) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), exps_j.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_i_h.clone(), coeff_i.len()) },
                unsafe { ArrayArg::from_raw_parts(coeff_j_h.clone(), coeff_j.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), total_g) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                drj[0],
                drj[1],
                drj[2],
                SQRTPI,
                std::f64::consts::PI,
                li,
                lj,
                nprim_i,
                nprim_j,
                nctr_i,
                nctr_j,
                $mode,
                $order,
                $rank,
            )
        };
    }

    // Comptime (op_mode, moment_order, rank) selected via a host match. The valid
    // Cluster-A combinations are enumerated explicitly (CubeCL cannot pass comptime
    // args dynamically).
    match op_mode {
        0u32 => launch_with!(0u32, 1u32, 3u32),  // r
        1u32 => launch_with!(1u32, 2u32, 9u32),  // rr
        2u32 => launch_with!(2u32, 3u32, 27u32), // rrr
        3u32 => launch_with!(3u32, 4u32, 81u32), // rrrr
        4u32 => launch_with!(4u32, 2u32, 1u32),  // r2
        5u32 => launch_with!(5u32, 4u32, 1u32),  // r4
        6u32 => launch_with!(6u32, 1u32, 1u32),  // z
        _ => launch_with!(7u32, 2u32, 1u32),     // zz
    }

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// 5-arm backend dispatch for [`run_1e_moment_device`] (Cpu/Wgpu/Cuda/Rocm/Metal).
#[allow(clippy::too_many_arguments)]
fn run_1e_moment_on_backend(
    backend: &ResolvedBackend,
    op_mode: u32,
    moment_order: u32,
    rank: u32,
    li: u32,
    lj: u32,
    nprim_i: u32,
    nprim_j: u32,
    nctr_i: u32,
    nctr_j: u32,
    ri: [f64; 3],
    rj: [f64; 3],
    drj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_moment_device::<cubecl::cpu::CpuRuntime>(
            client, op_mode, moment_order, rank, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri,
            rj, drj, exps_i, exps_j, coeff_i, coeff_j,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_moment_device::<cubecl_wgpu::WgpuRuntime>(
            client, op_mode, moment_order, rank, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri,
            rj, drj, exps_i, exps_j, coeff_i, coeff_j,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_moment_device::<cubecl_cuda::CudaRuntime>(
            client, op_mode, moment_order, rank, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri,
            rj, drj, exps_i, exps_j, coeff_i, coeff_j,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_moment_device::<cubecl_hip::HipRuntime>(
            client, op_mode, moment_order, rank, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri,
            rj, drj, exps_i, exps_j, coeff_i, coeff_j,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_moment_device::<cubecl_wgpu::WgpuRuntime>(
            client, op_mode, moment_order, rank, li, lj, nprim_i, nprim_j, nctr_i, nctr_j, ri,
            rj, drj, exps_i, exps_j, coeff_i, coeff_j,
        ),
    }
}

/// Contract G-tensor elements for the overlap operator.
///
/// Loops over all (ix+jx, iy+jy, iz+jz) Cartesian products and returns the
/// flat cartesian integral buffer of size ncart(li) * ncart(lj).
///
/// Host f64 reference — used by the device-vs-host cross-check and unit tests
/// (the live scalar path now computes via [`one_electron_scalar_kernel`]).
#[cfg(test)]
fn contract_overlap(g: &[f64], li: u8, lj: u8, nmax: u32) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let mut out = vec![0.0_f64; nci * ncj];

    let gx = 0;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            // g[axis + j*dj + i]
            let vx = g[gx + jx as usize * dj + ix as usize];
            let vy = g[gy + jy as usize * dj + iy as usize];
            let vz = g[gz + jz as usize * dj + iz as usize];
            // Column-major (bra fastest): out[ket*nci + bra]. This is the layout
            // cart_to_sph_1e reads (cart_buf[j*nci+ci]) and pyscf-rs stitches
            // (block[ii+jj*ni]). Row-major here silently transposed cross-l blocks
            // (li!=lj, both>0: p-d/p-f/d-g) since only those have nci,ncj both >1.
            out[cj_idx * nci + ci_idx] += vx * vy * vz;
        }
    }

    out
}

/// Contract G-tensor elements for the kinetic operator.
///
/// Implements `CINTgout1e_int1e_kin` from `autocode/intor1.c` (lines 18-46).
///
/// Libcint builds three derivative G-tensors via `CINTnabla1j_1e` (derivative in j,
/// i.e., ket direction):
///   g1 = D_j(g0)  with lj levels (used for cross terms s[1]..s[8])
///   g2 = D_j(g0)  with lj+1 levels (intermediate for second derivative)
///   g3 = D_j(g2)  with lj levels (second derivative = D_j^2(g0))
///
/// `CINTnabla1j_1e` formula (stepping in j-direction with stride dj):
///   D_j[g][j=0, i] = -2*aj * g[j=1, i]
///   D_j[g][j>0, i] = j * g[j-1, i] + (-2*aj) * g[j+1, i]
///
/// So `g3[jx, ix] = D_j^2(g0)[jx, ix]`:
///   g2[jx] = D_j(g0)[jx], computed with lj+1 coverage: g2[0..lj+1]
///   g3[jx] = D_j(g2)[jx] = jx*g2[jx-1] - 2*aj*g2[jx+1], for jx=0..lj
///
/// Expanding g2:
///   g3[jx] = jx*(jx-1)*g0[jx-2] - 2*aj*(2*jx+1)*g0[jx] + 4*aj^2*g0[jx+2]
///
/// Note: the derivative steps ±2 levels in j (i.e., ±2*dj in the flat index), NOT ±1.
/// g0[jx+2] requires g2 to have lj+2 j-levels, which means HRR must be built to lj+2.
///
/// (where j-level steps by stride `dj = nmax+1`; ix is the bra index unchanged)
///
/// The kinetic kernel output:
///   gout[n] = -(g3x*g0y*g0z + g0x*g3y*g0z + g0x*g0y*g3z)
/// and `int1e_kin_sph` applies `common_factor *= 0.5`, giving T = -0.5 * (...).
///
/// Requires G-tensor built with `lj_ext = lj + 2` HRR j-levels so that `g0[jx+2]`
/// (accessed via `jx*dj + 2*dj`) is valid. `nmax = li + lj + 2` ensures the VRR
/// bra has enough levels for the HRR to shift two extra quanta to the ket.
///
/// Host f64 reference — used by the device-vs-host cross-check and unit tests
/// (the live scalar path now computes via [`one_electron_scalar_kernel`]).
#[cfg(test)]
fn contract_kinetic(g: &[f64], li: u8, lj: u8, nmax: u32, aj: f64) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    // G-tensor was built with lj+2 HRR j-levels to allow jx+2 access.
    // g_per_axis = (nmax+1) * (lj+2+1) = (nmax+1) * (lj+3)
    let lj_ext = lj as u32 + 2;
    let g_per_axis = ((nmax + 1) * (lj_ext + 1)) as usize;
    let dj = (nmax + 1) as usize; // stride between j-levels within each axis block

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let mut out = vec![0.0_f64; nci * ncj];

    let gx = 0;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            // Index into G-tensor: base index for g0[jx, ix] = jx*dj + ix
            let nx = jx as usize * dj + ix as usize;
            let ny = jy as usize * dj + iy as usize;
            let nz = jz as usize * dj + iz as usize;

            let vx0 = g[gx + nx];
            let vy0 = g[gy + ny];
            let vz0 = g[gz + nz];

            // Second j-derivative of g0 at (jx, ix) (derived from two D_j applications):
            //   g3x = jx*(jx-1)*g0[jx-2, ix] - 2*aj*(2*jx+1)*g0[jx, ix] + 4*aj^2*g0[jx+2, ix]
            // Stepping in j-direction uses stride dj; "+2 levels" = +2*dj, "-2 levels" = -2*dj.
            // g0[jx+2, ix] = g[gx + (jx+2)*dj + ix] = g[gx + nx + 2*dj]  (valid since lj_ext=lj+2)
            // g0[jx-2, ix] = g[gx + (jx-2)*dj + ix] = g[gx + nx - 2*dj]  (valid only when jx >= 2)
            let jxf = jx as f64;
            let g3x = 4.0 * aj * aj * g[gx + nx + 2 * dj] - 2.0 * aj * (2.0 * jxf + 1.0) * vx0
                + jxf * (jxf - 1.0) * if jx >= 2 { g[gx + nx - 2 * dj] } else { 0.0 };

            let jyf = jy as f64;
            let g3y = 4.0 * aj * aj * g[gy + ny + 2 * dj] - 2.0 * aj * (2.0 * jyf + 1.0) * vy0
                + jyf * (jyf - 1.0) * if jy >= 2 { g[gy + ny - 2 * dj] } else { 0.0 };

            let jzf = jz as f64;
            let g3z = 4.0 * aj * aj * g[gz + nz + 2 * dj] - 2.0 * aj * (2.0 * jzf + 1.0) * vz0
                + jzf * (jzf - 1.0) * if jz >= 2 { g[gz + nz - 2 * dj] } else { 0.0 };

            // T = -0.5 * (g3x*g0y*g0z + g0x*g3y*g0z + g0x*g0y*g3z)
            // The 0.5 factor comes from int1e_kin_sph common_factor *= 0.5.
            let kinetic = -0.5 * (g3x * vy0 * vz0 + vx0 * g3y * vz0 + vx0 * vy0 * g3z);
            // Column-major (bra fastest): out[ket*nci + bra] — see contract_overlap.
            out[cj_idx * nci + ci_idx] += kinetic;
        }
    }

    out
}

/// Apply the bra-center nabla (`∂/∂Ai`) to the 1e overlap G-tensor.
///
/// Corresponds to `CINTnabla1i_1e` in libcint `g1e.c` (the bra-side derivative).
/// For `int1e_ipovlp`, the overlap G-tensor is built with `nmax = li + lj + 1`
/// (one extra bra level so that `g[jx*dj + ix+1]` is valid for the nabla formula).
///
/// Formula per axis (nabla on axis a):
///   ix == 0: g1[jx*dj + 0] = -2*ai * g[jx*dj + 1]
///   ix >= 1: g1[jx*dj + ix] = ix * g[jx*dj + (ix-1)] + (-2*ai) * g[jx*dj + (ix+1)]
///
/// Component mixing rule (standard ip1 formula):
///   s[0] = g1x[jx,ix] * g0y[jy,iy] * g0z[jz,iz]   (∂/∂Ax)
///   s[1] = g0x[jx,ix] * g1y[jy,iy] * g0z[jz,iz]   (∂/∂Ay)
///   s[2] = g0x[jx,ix] * g0y[jy,iy] * g1z[jz,iz]   (∂/∂Az)
///
/// Returns `Vec<f64>` of length `3 * nci * ncj` in component-leading layout:
/// `out[comp * nci*ncj + cj_idx * nci + ci_idx]`.
///
/// # Arguments
/// - `g`: the overlap G-tensor built with `nmax` (including +1 bra headroom)
/// - `li`, `lj`: bra/ket angular momenta (base values)
/// - `nmax`: the VRR max used when building g (= li + lj + 1 for ipovlp)
/// - `ai`: bra Gaussian exponent
///
/// Host f64 reference — used by the device-vs-host cross-check and unit tests
/// (the live ipovlp path now computes via [`one_electron_grad_bra_kernel`]).
#[cfg(test)]
fn contract_grad_1e_bra(g: &[f64], li: u8, lj: u8, nmax: u32, ai: f64) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    // G-tensor shape: g_per_axis = (nmax+1) * (lj+1)
    // dj = nmax+1 (stride between consecutive j-levels within one axis block)
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    // Apply nabla1i to all 3 axes: g1[axis*g_per_axis + jx*dj + ix]
    let ai2 = -2.0 * ai;
    let mut g1 = vec![0.0_f64; 3 * g_per_axis];

    for axis in 0..3usize {
        let off = axis * g_per_axis;
        for j in 0..=(lj as usize) {
            let jbase = j * dj;
            // ix = 0: f = -2*ai * g[ix+1]
            g1[off + jbase] = ai2 * g[off + jbase + 1];
            // ix >= 1: f = ix * g[ix-1] + (-2*ai) * g[ix+1]
            for ix in 1..=(li as usize) {
                g1[off + jbase + ix] =
                    ix as f64 * g[off + jbase + ix - 1] + ai2 * g[off + jbase + ix + 1];
            }
        }
    }

    // Build 3-component output using axis-mixing rule:
    //   comp 0 (∂/∂Ax): g1x * g0y * g0z
    //   comp 1 (∂/∂Ay): g0x * g1y * g0z
    //   comp 2 (∂/∂Az): g0x * g0y * g1z
    let block_len = nci * ncj;
    let mut out = vec![0.0_f64; 3 * block_len];

    let gx = 0usize;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let nx = jx as usize * dj + ix as usize;
            let ny = jy as usize * dj + iy as usize;
            let nz = jz as usize * dj + iz as usize;

            let g0x = g[gx + nx];
            let g0y = g[gy + ny];
            let g0z = g[gz + nz];
            let g1x = g1[gx + nx];
            let g1y = g1[gy + ny];
            let g1z = g1[gz + nz];

            let n = cj_idx * nci + ci_idx;
            out[0 * block_len + n] += g1x * g0y * g0z;
            out[1 * block_len + n] += g0x * g1y * g0z;
            out[2 * block_len + n] += g0x * g0y * g1z;
        }
    }

    out
}

/// Compute the bra-derivative of the kinetic integral (`int1e_ipkin`).
///
/// For `int1e_ipkin = ∂/∂Ai of (-0.5 * ∇^2)`, the formula is derived by applying
/// the product rule. The kinetic integrand is:
///   T_ij = -0.5 * (d2x*g0y*g0z + g0x*d2y*g0z + g0x*g0y*d2z)
/// where d2a = D_j^2(g0a) is the second ket-derivative on axis a.
///
/// Since ∂/∂Ax_i commutes with D_j^2 (they act on different indices), and ∂g0y/∂Ax = 0
/// (the y-axis g-tensor is not affected by Ax), the bra-derivative simplifies to:
///   ∂T/∂Ax = -0.5 * ( D_j^2(g1x)*g0y*g0z + g1x*d2y*g0z + g1x*g0y*d2z )
///   ∂T/∂Ay = -0.5 * ( d2x*g1y*g0z + g0x*D_j^2(g1y)*g0z + g0x*g1y*d2z )
///   ∂T/∂Az = -0.5 * ( d2x*g0y*g1z + g0x*d2y*g1z + g0x*g0y*D_j^2(g1z) )
///
/// where g1a = nabla1i(g0a) (bra-derivative on axis a).
///
/// The G-tensor is built with `lj_ext = lj + 2` (kinetic headroom) and
/// `nmax = li + lj + 3` (kinetic +2 AND nabla +1 extra bra level).
///
/// Returns `Vec<f64>` of length `3 * nci * ncj` in component-leading layout.
///
/// Host f64 reference — used by the device-vs-host cross-check and unit tests
/// (the live ipkin path now computes via [`one_electron_grad_bra_kernel`]).
#[cfg(test)]
fn contract_ipkin(g: &[f64], li: u8, lj: u8, nmax: u32, ai: f64, aj: f64) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    // G-tensor was built with lj_ext = lj+2, nmax = li+lj+3
    let lj_ext = lj as u32 + 2;
    let g_per_axis = ((nmax + 1) * (lj_ext + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    // Step 1: Apply nabla1i on bra i-index across all 3 axes (over full lj_ext j-range).
    let ai2 = -2.0 * ai;
    let mut g1 = vec![0.0_f64; 3 * g_per_axis];

    for axis in 0..3usize {
        let off = axis * g_per_axis;
        for j in 0..=(lj_ext as usize) {
            let jbase = j * dj;
            g1[off + jbase] = ai2 * g[off + jbase + 1];
            for ix in 1..=(li as usize) {
                g1[off + jbase + ix] =
                    ix as f64 * g[off + jbase + ix - 1] + ai2 * g[off + jbase + ix + 1];
            }
        }
    }

    // Step 2: Apply D_j^2 (ket kinetic) to both g0 and g1 over the base lj range.
    // D_j^2(f)[j, i] = jf*(jf-1)*f[j-2,i] - 2*aj*(2*jf+1)*f[j,i] + 4*aj^2*f[j+2,i]
    let mut d2g0 = vec![0.0_f64; 3 * g_per_axis]; // D_j^2(g0)
    let mut d2g1 = vec![0.0_f64; 3 * g_per_axis]; // D_j^2(g1)

    for axis in 0..3usize {
        let off = axis * g_per_axis;
        for j in 0..=(lj as usize) {
            let jf = j as f64;
            for i_idx in 0..=(li as usize) {
                let nx = j * dj + i_idx;
                let g0_lo = if j >= 2 { g[off + nx - 2 * dj] } else { 0.0 };
                let g1_lo = if j >= 2 { g1[off + nx - 2 * dj] } else { 0.0 };
                d2g0[off + nx] = 4.0 * aj * aj * g[off + nx + 2 * dj]
                    - 2.0 * aj * (2.0 * jf + 1.0) * g[off + nx]
                    + jf * (jf - 1.0) * g0_lo;
                d2g1[off + nx] = 4.0 * aj * aj * g1[off + nx + 2 * dj]
                    - 2.0 * aj * (2.0 * jf + 1.0) * g1[off + nx]
                    + jf * (jf - 1.0) * g1_lo;
            }
        }
    }

    // Step 3: Build 3-component output using the ipkin axis-mixing formula.
    let block_len = nci * ncj;
    let mut out = vec![0.0_f64; 3 * block_len];

    let gx = 0usize;
    let gy = g_per_axis;
    let gz = 2 * g_per_axis;

    for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
        for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
            let nx = jx as usize * dj + ix as usize;
            let ny = jy as usize * dj + iy as usize;
            let nz = jz as usize * dj + iz as usize;

            let g0x = g[gx + nx];
            let g0y = g[gy + ny];
            let g0z = g[gz + nz];
            let g1x = g1[gx + nx];
            let g1y = g1[gy + ny];
            let g1z = g1[gz + nz];
            let d2x0 = d2g0[gx + nx];
            let d2y0 = d2g0[gy + ny];
            let d2z0 = d2g0[gz + nz];
            let d2x1 = d2g1[gx + nx];
            let d2y1 = d2g1[gy + ny];
            let d2z1 = d2g1[gz + nz];

            // ∂T/∂Ax = -0.5*(D_j^2(g1x)*g0y*g0z + g1x*d2y0*g0z + g1x*g0y*d2z0)
            let s0 = -0.5 * (d2x1 * g0y * g0z + g1x * d2y0 * g0z + g1x * g0y * d2z0);
            // ∂T/∂Ay = -0.5*(d2x0*g1y*g0z + g0x*D_j^2(g1y)*g0z + g0x*g1y*d2z0)
            let s1 = -0.5 * (d2x0 * g1y * g0z + g0x * d2y1 * g0z + g0x * g1y * d2z0);
            // ∂T/∂Az = -0.5*(d2x0*g0y*g1z + g0x*d2y0*g1z + g0x*g0y*D_j^2(g1z))
            let s2 = -0.5 * (d2x0 * g0y * g1z + g0x * d2y0 * g1z + g0x * g0y * d2z1);

            let n = cj_idx * nci + ci_idx;
            out[0 * block_len + n] += s0;
            out[1 * block_len + n] += s1;
            out[2 * block_len + n] += s2;
        }
    }

    out
}

/// Compute nuclear attraction integrals for one primitive pair, all atoms.
///
/// Uses Rys quadrature with Boys-weighted VRR.
/// Reference: g1e.c lines 208-320 (CINTg1e_nuc).
///
/// Host f64 reference — used by the device-vs-host cross-check and unit tests
/// (the live scalar path now computes via [`one_electron_scalar_kernel`]).
#[cfg(test)]
fn contract_nuclear(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    li: u8,
    lj: u8,
    atoms: &[cintx_core::Atom],
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let mut out = vec![0.0_f64; nci * ncj];

    let nmax = (li + lj) as u32;
    let nrys_roots = (li + lj) as u32 / 2 + 1;

    // g_per_axis for nuclear: one VRR per Rys root, same HRR layout
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];

    for atom in atoms {
        let z_c = atom.atomic_number as f64;
        let rc = atom.coord_bohr;

        // Vector from C to P: crij[d] = rc[d] - P[d] (Note: g1e.c uses C - P)
        let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];

        // Boys argument x = zeta * |P - C|^2
        let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

        // Get Rys roots and weights. Dispatch through the general nroots=1..5
        // host quadrature so high-l nuclear attraction (li+lj>=4 → nrys_roots>=3,
        // e.g. d|d on cc-pVDZ heavy atoms) is supported, not just the old
        // hardcoded 2-root branch (DI-02-11-CINTX-NUC-HIGHL). Bit-identical for
        // nrys_roots∈{1,2} (same rys_root1/2_host); rys_roots_host panics >5.
        let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);

        // Nuclear prefactor: fac1 = 2*PI * (-Z_C) * fac / zeta
        // Source: g1e.c line 218-221
        let fac1 = 2.0 * std::f64::consts::PI * (-z_c) * pd.fac / pd.zeta_ab;

        // For each Rys root
        for n in 0..nrys_roots as usize {
            let u_n = u_arr[n];
            let w_n = w_arr[n];

            // tau = u_n / (1 + u_n) transforms root to [0,1] interval
            let tau = u_n / (1.0 + u_n);

            // Modified recurrence coefficient b10 = aij2 * (1 - tau) = aij2 - aij2*tau
            // Source: g1e.c line 229
            let rt = pd.aij2 * (1.0 - tau);

            // Modified center displacement: r0[d] = (P[d] - ri[d]) + tau * crij[d]
            // Note: crij[d] = rc[d] - rp[d], and for nuc VRR the displacement is
            // modified from P-Ri to account for the Rys root.
            // From g1e.c: the VRR c00 = rp[d] - ri[d] + tau*(rc[d] - rp[d])
            //           = (rp[d] - ri[d]) + tau*crij_d
            // where crij_d here is rc[d] - rp[d] = -crij[d] from our definition
            // Actually: crij[d] = rc[d] - rp[d] above, so tau*(rc[d]-rp[d]) = tau*crij[d]
            // and c00 = (P-Ri)[d] + tau*(rc[d] - rp[d])
            let c00 = [
                (rp[0] - ri[0]) + tau * crij[0],
                (rp[1] - ri[1]) + tau * crij[1],
                (rp[2] - ri[2]) + tau * crij[2],
            ];

            // gz base = fac1 * w_n for this root
            let gz0_root = fac1 * w_n;

            // Build per-root G-tensor using VRR 2e pattern (root-dependent c00 and b10)
            let mut g_root = vec![0.0_f64; 3 * g_per_axis];

            let gx_off = 0;
            let gy_off = g_per_axis;
            let gz_off = 2 * g_per_axis;

            g_root[gx_off] = 1.0;
            g_root[gy_off] = 1.0;
            g_root[gz_off] = gz0_root;

            // Nuclear VRR uses modified c00 (root-dependent), b10 = rt
            // vrr_2e_step_host signature: (g, c00, b10, nmax, stride)
            if nmax >= 1 {
                crate::math::obara_saika::vrr_2e_step_host(
                    &mut g_root[gx_off..gx_off + g_per_axis],
                    c00[0],
                    rt,
                    nmax,
                    1,
                );
                crate::math::obara_saika::vrr_2e_step_host(
                    &mut g_root[gy_off..gy_off + g_per_axis],
                    c00[1],
                    rt,
                    nmax,
                    1,
                );
                crate::math::obara_saika::vrr_2e_step_host(
                    &mut g_root[gz_off..gz_off + g_per_axis],
                    c00[2],
                    rt,
                    nmax,
                    1,
                );
            }

            // HRR to shift to ket center
            let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            if lj >= 1 {
                let di = 1u32;
                let dj_stride = nmax + 1;
                hrr_step_host(
                    &mut g_root[gx_off..gx_off + g_per_axis],
                    rirj[0],
                    di,
                    dj_stride,
                    nmax,
                    lj as u32,
                );
                hrr_step_host(
                    &mut g_root[gy_off..gy_off + g_per_axis],
                    rirj[1],
                    di,
                    dj_stride,
                    nmax,
                    lj as u32,
                );
                hrr_step_host(
                    &mut g_root[gz_off..gz_off + g_per_axis],
                    rirj[2],
                    di,
                    dj_stride,
                    nmax,
                    lj as u32,
                );
            }

            // Contract this root's contribution
            for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let vx = g_root[gx_off + jx as usize * dj + ix as usize];
                    let vy = g_root[gy_off + jy as usize * dj + iy as usize];
                    let vz = g_root[gz_off + jz as usize * dj + iz as usize];
                    // Column-major (bra fastest): out[ket*nci + bra]. This is the layout
                    // cart_to_sph_1e reads (cart_buf[j*nci+ci]) and pyscf-rs stitches
                    // (block[ii+jj*ni]). Row-major here silently transposed cross-l blocks
                    // (li!=lj, both>0: p-d/p-f/d-g) since only those have nci,ncj both >1.
                    out[cj_idx * nci + ci_idx] += vx * vy * vz;
                }
            }
        }
    }

    out
}

/// Compute the bra-derivative of the nuclear-attraction integral for one primitive pair.
///
/// Shared math for `int1e_ipnuc` (sum of `∑_C (-Z_C)·∇` over all nuclei) and
/// `int1e_iprinv` (single rinv origin, factor `+1.0`, no `-Z_C`). The two callers
/// differ ONLY in the `origins` list passed here:
///   - ipnuc: `[(atoms[c].coord_bohr, -(Z_C as f64)) for each nucleus c]`
///   - iprinv: `[(rinv_orig, 1.0)]` (single entry, no charge)
///
/// Reference: `g1e.c` `CINTg1e_nuc` (lines 208-320) for the base nuclear G-tensor,
/// `grad1.c` `CINTgout1e_int1e_ipnuc`/`int1e_iprinv` for the `∂/∂Ai` (bra) derivative.
/// The bra nabla identity is `∂/∂Ai χ_l = l·χ_{l-1} - 2·ai·χ_{l+1}`
/// (`CINTnabla1i_1e`, identical to the overlap-gradient mixing in `contract_grad_1e_bra`).
///
/// The base nuclear prefactor `fac1 = 2*PI * charge_factor * fac / aij` matches
/// libcint exactly: for ipnuc the charge factor is `-Z_C` (point nucleus,
/// `CINTnuc_mod`→`tau=1.0`), for iprinv it is `+1.0` (g1e.c lines 226-234).
///
/// The G-tensor is built with `nmax = li + lj + 1` (one extra bra level so the
/// `ix+1` nabla access is valid), via the root-dependent VRR `vrr_2e_step_host`
/// (Phase 09 nuclear-attraction decision — NOT the fixed-center `vrr_step_host`).
///
/// Atoms are accumulated in the order they appear in `origins` (ipnuc passes the
/// low→high atom index order) for bit-stable reduction (D-10, T-21-04-04).
///
/// Returns `Vec<f64>` of length `3 * nci * ncj` in component-leading layout:
/// `out[comp * nci*ncj + cj_idx * nci + ci_idx]`.
///
/// Host f64 reference — used by the device-vs-host cross-check and unit tests
/// (the live ipnuc/iprinv path now computes via [`one_electron_nuc_grad_kernel`]).
#[cfg(test)]
fn contract_nuclear_grad(
    pd: &crate::math::pdata::PairData,
    ri: [f64; 3],
    rj: [f64; 3],
    li: u8,
    lj: u8,
    ai: f64,
    origins: &[([f64; 3], f64)],
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);

    // +1 bra headroom so the nabla1i ix+1 access is valid.
    let nmax = (li + lj) as u32 + 1;
    let nrys_roots = ((li + lj) as u32 + 1) / 2 + 1;

    // G-tensor per-axis layout: (nmax+1) entries per j-level, (lj+1) j-levels.
    let g_per_axis = ((nmax + 1) * (lj as u32 + 1)) as usize;
    let dj = (nmax + 1) as usize;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];

    let block_len = nci * ncj;
    let mut out = vec![0.0_f64; 3 * block_len];

    // Ordered accumulation over origins (low→high atom index for ipnuc; single for iprinv).
    for &(rc, charge_factor) in origins {
        // Vector from C to P: crij[d] = rc[d] - P[d] (g1e.c uses C - P).
        let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];

        // Boys argument x = zeta * |P - C|^2.
        let x_boys = pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);

        let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);

        // Nuclear prefactor (point nucleus, tau=1): fac1 = 2*PI * charge_factor * fac / aij.
        // ipnuc: charge_factor = -Z_C; iprinv: charge_factor = +1.0 (g1e.c 226-234).
        let fac1 = 2.0 * std::f64::consts::PI * charge_factor * pd.fac / pd.zeta_ab;

        for n in 0..nrys_roots as usize {
            let u_n = u_arr[n];
            let w_n = w_arr[n];

            let tau = u_n / (1.0 + u_n);
            let rt = pd.aij2 * (1.0 - tau);

            let c00 = [
                (rp[0] - ri[0]) + tau * crij[0],
                (rp[1] - ri[1]) + tau * crij[1],
                (rp[2] - ri[2]) + tau * crij[2],
            ];

            let gz0_root = fac1 * w_n;

            // Build per-root G-tensor (root-dependent c00 and b10=rt) with +1 headroom.
            let mut g_root = vec![0.0_f64; 3 * g_per_axis];
            let gx_off = 0usize;
            let gy_off = g_per_axis;
            let gz_off = 2 * g_per_axis;

            g_root[gx_off] = 1.0;
            g_root[gy_off] = 1.0;
            g_root[gz_off] = gz0_root;

            if nmax >= 1 {
                crate::math::obara_saika::vrr_2e_step_host(
                    &mut g_root[gx_off..gx_off + g_per_axis],
                    c00[0],
                    rt,
                    nmax,
                    1,
                );
                crate::math::obara_saika::vrr_2e_step_host(
                    &mut g_root[gy_off..gy_off + g_per_axis],
                    c00[1],
                    rt,
                    nmax,
                    1,
                );
                crate::math::obara_saika::vrr_2e_step_host(
                    &mut g_root[gz_off..gz_off + g_per_axis],
                    c00[2],
                    rt,
                    nmax,
                    1,
                );
            }

            // HRR to shift to ket center.
            let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            if lj >= 1 {
                let di = 1u32;
                let dj_stride = nmax + 1;
                hrr_step_host(
                    &mut g_root[gx_off..gx_off + g_per_axis],
                    rirj[0],
                    di,
                    dj_stride,
                    nmax,
                    lj as u32,
                );
                hrr_step_host(
                    &mut g_root[gy_off..gy_off + g_per_axis],
                    rirj[1],
                    di,
                    dj_stride,
                    nmax,
                    lj as u32,
                );
                hrr_step_host(
                    &mut g_root[gz_off..gz_off + g_per_axis],
                    rirj[2],
                    di,
                    dj_stride,
                    nmax,
                    lj as u32,
                );
            }

            // Apply nabla1i (bra derivative) to all 3 axes of this root's G-tensor.
            // f[jx*dj+ix] = ix * g[jx*dj+(ix-1)] + (-2*ai) * g[jx*dj+(ix+1)].
            let ai2 = -2.0 * ai;
            let mut g1 = vec![0.0_f64; 3 * g_per_axis];
            for axis in 0..3usize {
                let off = axis * g_per_axis;
                for j in 0..=(lj as usize) {
                    let jbase = j * dj;
                    g1[off + jbase] = ai2 * g_root[off + jbase + 1];
                    for ix in 1..=(li as usize) {
                        g1[off + jbase + ix] = ix as f64 * g_root[off + jbase + ix - 1]
                            + ai2 * g_root[off + jbase + ix + 1];
                    }
                }
            }

            // Build this root's 3-component contribution and accumulate.
            let gx = 0usize;
            let gy = g_per_axis;
            let gz = 2 * g_per_axis;
            for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let nx = jx as usize * dj + ix as usize;
                    let ny = jy as usize * dj + iy as usize;
                    let nz = jz as usize * dj + iz as usize;

                    let g0x = g_root[gx + nx];
                    let g0y = g_root[gy + ny];
                    let g0z = g_root[gz + nz];
                    let g1x = g1[gx + nx];
                    let g1y = g1[gy + ny];
                    let g1z = g1[gz + nz];

                    let n_out = cj_idx * nci + ci_idx;
                    out[0 * block_len + n_out] += g1x * g0y * g0z;
                    out[1 * block_len + n_out] += g0x * g1y * g0z;
                    out[2 * block_len + n_out] += g0x * g0y * g1z;
                }
            }
        }
    }

    out
}

/// Generic inner for the 1e launcher.
///
/// Contains the full algorithm of `launch_one_electron` parameterized over the
/// output float type `F: CintFloat`. The staging buffer is typed `&mut [F]` so the
/// bytemuck-cast pattern at the outer boundary is sound (Plan 01 A5 proven).
///
/// Intermediate computations (G-tensor, cart_buf) remain `f64` throughout —
/// precision conversion happens only at the final staging write via
/// `F::from_f64_lossy`. For f64 this is a zero-cost identity; for f32 it truncates.
///
/// The outer public `launch_one_electron` is a thin dispatcher that matches on
/// `plan.precision`, binds the appropriate `&mut [F]` view (bytemuck cast for F32;
/// the existing slice for F64), and calls this inner.
fn launch_one_electron_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_1e",
            detail: format!(
                "canonical_family mismatch: expected 1e, got {}",
                specialization.canonical_family()
            ),
        });
    }

    // `backend` is consumed by the scalar device-kernel dispatch below
    // (`run_1e_scalar_on_backend`). The gradient + spinor arms remain host-side.

    let shells = plan.shells.as_slice();
    if shells.len() < 2 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_1e",
            detail: "1e kernel requires exactly 2 shells".to_owned(),
        });
    }

    let shell_i = &shells[0];
    let shell_j = &shells[1];

    let li = shell_i.ang_momentum;
    let lj = shell_j.ang_momentum;

    // Atom coordinates
    let atoms = plan.basis.atoms();
    let ri = atoms[shell_i.atom_index as usize].coord_bohr;
    let rj = atoms[shell_j.atom_index as usize].coord_bohr;

    // Operator dispatch
    let op_name = plan.descriptor.operator_name();
    let is_overlap = op_name == "overlap";
    let is_kinetic = op_name == "kinetic";
    let is_nuclear = op_name == "nuclear-attraction";
    let is_ipovlp = op_name == "ipovlp";
    let is_ipkin = op_name == "ipkin";
    let is_ipnuc = op_name == "ipnuc";
    let is_iprinv = op_name == "iprinv";
    // Both-side rank-9 family (Phase 23): <NABLA i | OP | NABLA j>.
    let is_ipovlpip = op_name == "ipovlpip";
    let is_ipkinip = op_name == "ipkinip";
    let is_ipnucip = op_name == "ipnucip";
    let is_rank9_both = is_ipovlpip || is_ipkinip || is_ipnucip;

    // Phase 24 Cluster A (MOM-01/02/03): overlap-derived position-tensor moment
    // families r/rr/rrr/rrrr/r2/r4/z/zz and their `_origj` variants. Each maps to
    // a (op_mode, moment_order, rank) tuple for the parameterized moment kernel.
    // `_origj` reuses the SAME op_mode/order/rank — only the origin source differs
    // (handled below via drj = rj - origin; for _origj origin = rj so drj = 0).
    let moment_dispatch: Option<(u32, u32, u32)> = match op_name {
        "r" | "r_origj" => Some((0, 1, 3)),
        "rr" | "rr_origj" => Some((1, 2, 9)),
        "rrr" => Some((2, 3, 27)),
        "rrrr" => Some((3, 4, 81)),
        "r2" | "r2_origj" => Some((4, 2, 1)),
        "r4" | "r4_origj" => Some((5, 4, 1)),
        "z" | "z_origj" => Some((6, 1, 1)),
        "zz" | "zz_origj" => Some((7, 2, 1)),
        _ => None,
    };
    let is_moment = moment_dispatch.is_some();
    // `_origj` variants read the ket basis center rj; base families read the gauge
    // origin env[PTR_COMMON_ORIG]. D-02: origin source is a kernel-side coordinate
    // choice, realized host-side as drj = rj - origin.
    let is_origj = op_name.ends_with("_origj");

    if !is_overlap
        && !is_kinetic
        && !is_nuclear
        && !is_ipovlp
        && !is_ipkin
        && !is_ipnuc
        && !is_iprinv
        && !is_rank9_both
        && !is_moment
    {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("1e operator '{}' is not supported", op_name),
        });
    }

    // Output sizes
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 24 Cluster A moment path (r/rr/rrr/rrrr/r2/r4/z/zz + _origj)
    // — rank ∈ {1,3,9,27,81} component-leading output
    // ─────────────────────────────────────────────────────────────────────────
    if let Some((op_mode, moment_order, rank)) = moment_dispatch {
        // Spinor moment reps are registered for surface completeness but not
        // implemented: fail typed, never partial (D-09).
        if plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name} moment"),
            });
        }

        // Origin-source branch (D-02). Base families read the gauge origin from
        // env[PTR_COMMON_ORIG] (defaulting to [0,0,0] when unset); `_origj`
        // variants use the ket basis center rj directly (drj = 0). This realizes
        // libcint's G1E_RCJ (drj = rj - common_orig) vs G1E_R_J (origin = rj).
        let origin: [f64; 3] = if is_origj {
            rj
        } else {
            plan.operator_env_params.common_orig.unwrap_or([0.0; 3])
        };
        let drj = [rj[0] - origin[0], rj[1] - origin[1], rj[2] - origin[2]];

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let block_len = nci * ncj;
        let total_len = (rank as usize) * block_len;

        // Internal G-tensor ceiling: nmax = li + lj + moment_order. Stays within
        // the engine's li+lj<=8 envelope on the gate corpus (STO-3G li,lj<=1 →
        // nmax <= 6 for rrrr/r4). Fail closed if a corpus shell would exceed it,
        // rather than silently truncating (RESEARCH A1).
        if li as u32 + lj as u32 + moment_order > 8 {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device 1e moment kernel supports l_i+l_j+order<=8; got \
                     l_i={li}, l_j={lj}, order={moment_order} for int1e_{op_name}"
                ),
            });
        }

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        let mut cart_comp = run_1e_moment_on_backend(
            backend, op_mode, moment_order, rank, li as u32, lj as u32, n_prim_i as u32,
            n_prim_j as u32, n_ctr_i as u32, n_ctr_j as u32, ri, rj, drj, &exps_i, &exps_j,
            &coeff_i, &coeff_j,
        );

        // Apply the libcint CINTcommon_fac_sp normalization to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        let rank_us = rank as usize;
        match plan.representation {
            Representation::Spheric => {
                let ni_sph = n_ctr_i * nsi;
                let nj_sph = n_ctr_j * nsj;
                let sph_block = ni_sph * nj_sph;
                for comp in 0..rank_us {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let cart_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let mut sph_tmp = vec![0.0_f64; nsi * nsj];
                            cart_to_sph_1e(
                                &cart_comp[cart_base..cart_base + block_len],
                                &mut sph_tmp,
                                li,
                                lj,
                            );
                            let staging_comp_base = comp * sph_block;
                            for mj in 0..nsj {
                                let jj = cj * nsj + mj;
                                for mi in 0..nsi {
                                    let ii = ci * nsi + mi;
                                    let dst = staging_comp_base + ii + jj * ni_sph;
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Representation::Cart => {
                let ni_cart = n_ctr_i * nci;
                let nj_cart = n_ctr_j * ncj;
                let cart_block = ni_cart * nj_cart;
                for comp in 0..rank_us {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let src_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let block = &cart_comp[src_base..src_base + block_len];
                            let staging_comp_base = comp * cart_block;
                            for jc in 0..ncj {
                                let jj = cj * ncj + jc;
                                for ic in 0..nci {
                                    let ii = ci * nci + ic;
                                    let dst = staging_comp_base + ii + jj * ni_cart;
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Representation::Spinor => unreachable!("spinor moment rejected above"),
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

        let staging_bytes = staging.len() * std::mem::size_of::<F>();
        return Ok(ExecutionStats {
            workspace_bytes: plan.workspace.bytes,
            required_workspace_bytes: plan.workspace.required_bytes,
            peak_workspace_bytes: staging_bytes,
            chunk_count: 1,
            planned_batches: 1,
            transfer_bytes: staging_bytes,
            not0,
            fallback_reason: plan.workspace.fallback_reason,
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Both-side rank-9 gradient path (`ipovlpip` / `ipkinip` / `ipnucip`)
    // — 9-component output
    // ─────────────────────────────────────────────────────────────────────────
    if is_rank9_both {
        // Spinor reps for the rank-9 both-side family are registered but not
        // implemented (Phase 23 D-06 / Phase 21 D-03): fail typed, never partial.
        if plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name} gradient"),
            });
        }

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let block_len = nci * ncj;
        let total_len = 9 * block_len;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        // ipnucip Rys nroots: one extra root vs single-side nuclear grad for the
        // added ket headroom. nmax = li+lj+2 → nroots = nmax/2 + 1. Fail closed.
        let nuc_nroots_both = (li as u32 + lj as u32 + 2) / 2 + 1;
        if is_ipnucip && nuc_nroots_both as usize > MAX_DEVICE_NROOTS {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_ipnucip kernel supports nroots<={MAX_DEVICE_NROOTS}; \
                     got nroots={nuc_nroots_both} for l_i={li}, l_j={lj}"
                ),
            });
        }

        let mut cart_9comp = if is_ipovlpip {
            run_1e_grad_both_on_backend(
                backend, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32, n_ctr_i as u32,
                n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
            )
        } else if is_ipkinip {
            run_1e_grad_kin_both_on_backend(
                backend, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32, n_ctr_i as u32,
                n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
            )
        } else {
            // ipnucip: ∑_C (-Z_C)·∇²-mixed over ALL nuclei, low→high (D-10).
            let mut origin_coords = Vec::with_capacity(atoms.len() * 3);
            let mut origin_charges = Vec::with_capacity(atoms.len());
            for atom in atoms.iter() {
                origin_coords.extend_from_slice(&atom.coord_bohr);
                origin_charges.push(-(atom.atomic_number as f64));
            }
            run_1e_nuc_grad_both_on_backend(
                backend, nuc_nroots_both, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32,
                n_ctr_i as u32, n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
                &origin_coords, &origin_charges,
            )
        };

        // Apply sp normalization scale to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_9comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        // Write to staging: component-leading layout staging[comp * ni*nj + n].
        match plan.representation {
            Representation::Spheric => {
                let ni_sph = n_ctr_i * nsi;
                let nj_sph = n_ctr_j * nsj;
                let sph_block = ni_sph * nj_sph;
                for comp in 0..9usize {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let cart_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let mut sph_tmp = vec![0.0_f64; nsi * nsj];
                            cart_to_sph_1e(
                                &cart_9comp[cart_base..cart_base + block_len],
                                &mut sph_tmp,
                                li,
                                lj,
                            );
                            let staging_comp_base = comp * sph_block;
                            for mj in 0..nsj {
                                let jj = cj * nsj + mj;
                                for mi in 0..nsi {
                                    let ii = ci * nsi + mi;
                                    let dst = staging_comp_base + ii + jj * ni_sph;
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Representation::Cart => {
                let ni_cart = n_ctr_i * nci;
                let nj_cart = n_ctr_j * ncj;
                let cart_block = ni_cart * nj_cart;
                for comp in 0..9usize {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let src_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let block = &cart_9comp[src_base..src_base + block_len];
                            let staging_comp_base = comp * cart_block;
                            for jc in 0..ncj {
                                let jj = cj * ncj + jc;
                                for ic in 0..nci {
                                    let ii = ci * nci + ic;
                                    let dst = staging_comp_base + ii + jj * ni_cart;
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Representation::Spinor => unreachable!("spinor rejected above"),
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

        let staging_bytes = staging.len() * std::mem::size_of::<F>();
        return Ok(ExecutionStats {
            workspace_bytes: plan.workspace.bytes,
            required_workspace_bytes: plan.workspace.required_bytes,
            peak_workspace_bytes: staging_bytes,
            chunk_count: 1,
            planned_batches: 1,
            transfer_bytes: staging_bytes,
            not0,
            fallback_reason: plan.workspace.fallback_reason,
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gradient path (ipovlp / ipkin / ipnuc / iprinv) — 3-component output
    // ─────────────────────────────────────────────────────────────────────────
    if is_ipovlp || is_ipkin || is_ipnuc || is_iprinv {
        // For iprinv, resolve the single rinv origin up front. The validator (21-01)
        // should have rejected a None origin before reaching the kernel, but we fail
        // typed (never panic, never read a garbage origin) — T-21-04-01 (defensive gate).
        let iprinv_origin: Option<[f64; 3]> = if is_iprinv {
            Some(
                plan.operator_env_params
                    .rinv_orig
                    .ok_or(cintxRsError::InvalidEnvParam {
                        param: "PTR_RINV_ORIG",
                        reason: "iprinv kernel reached with no rinv origin".to_owned(),
                    })?,
            )
        } else {
            None
        };

        // ipnuc origins: every nucleus coordinate with charge factor -Z_C, in
        // low→high atom-index order for bit-stable ordered reduction (D-10).
        let ipnuc_origins: Vec<([f64; 3], f64)> = if is_ipnuc {
            atoms
                .iter()
                .map(|atom| (atom.coord_bohr, -(atom.atomic_number as f64)))
                .collect()
        } else {
            Vec::new()
        };

        // Primitive / contraction counts.
        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        // 3-component cart accumulator: [3 * nci * ncj] per contraction pair,
        // component-leading (comp * block_len + cj_idx*nci + ci_idx). The device
        // kernels produce this exact layout (primitive × contraction accumulation
        // performed in-kernel like the scalar path).
        let block_len = nci * ncj;
        let total_len = 3 * block_len;

        // Flatten the f64 primitive data the device kernels read.
        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        // Nuclear-gradient nroots fail-closed guard BEFORE any device/Rys call.
        // nroots = (li+lj+1)/2 + 1 (the +1 bra headroom raises the VRR ceiling by
        // one root vs the scalar nuclear path). Mirror center_3c2e_ip1's guard and
        // the scalar nuclear guard. T-j7d-01.
        let nuc_nroots = ((li as u32 + lj as u32) + 1) / 2 + 1;
        if (is_ipnuc || is_iprinv) && nuc_nroots as usize > MAX_DEVICE_NROOTS {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device 1e nuclear-gradient kernel supports nroots<={MAX_DEVICE_NROOTS} \
                     (l_i+l_j<=8); got nroots={nuc_nroots} for l_i={li}, l_j={lj}"
                ),
            });
        }

        // Dispatch the appropriate device kernel for this gradient operator. Each
        // kernel performs the primitive × contraction accumulation internally and
        // returns the per-(ci,cj) 3-component component-leading cart buffer.
        let mut cart_3comp = if is_ipovlp {
            run_1e_grad_bra_on_backend(
                backend, 0, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32,
                n_ctr_i as u32, n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
            )
        } else if is_ipkin {
            run_1e_grad_bra_on_backend(
                backend, 1, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32,
                n_ctr_i as u32, n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
            )
        } else if is_ipnuc {
            // ipnuc: ∑_C (-Z_C)·∇ over ALL nuclei, ordered low→high (D-10).
            let mut origin_coords = Vec::with_capacity(ipnuc_origins.len() * 3);
            let mut origin_charges = Vec::with_capacity(ipnuc_origins.len());
            for (coord, charge) in &ipnuc_origins {
                origin_coords.extend_from_slice(coord);
                origin_charges.push(*charge);
            }
            run_1e_nuc_grad_on_backend(
                backend, nuc_nroots, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32,
                n_ctr_i as u32, n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
                &origin_coords, &origin_charges,
            )
        } else {
            // iprinv: single rinv origin, factor +1.0, no -Z_C (D-08).
            let origin = iprinv_origin.expect("iprinv origin resolved above");
            run_1e_nuc_grad_on_backend(
                backend, nuc_nroots, li as u32, lj as u32, n_prim_i as u32, n_prim_j as u32,
                n_ctr_i as u32, n_ctr_j as u32, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j,
                &[origin[0], origin[1], origin[2]], &[1.0],
            )
        };

        // Apply sp normalization scale to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_3comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        // Write to staging: component-leading layout staging[comp * ni*nj + n].
        // For sph: transform each component separately; for cart: copy directly.
        match plan.representation {
            Representation::Spheric => {
                let ni_sph = n_ctr_i * nsi;
                let nj_sph = n_ctr_j * nsj;
                let sph_block = ni_sph * nj_sph;
                for comp in 0..3usize {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let cart_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let mut sph_tmp = vec![0.0_f64; nsi * nsj];
                            cart_to_sph_1e(
                                &cart_3comp[cart_base..cart_base + block_len],
                                &mut sph_tmp,
                                li,
                                lj,
                            );
                            let staging_comp_base = comp * sph_block;
                            for mj in 0..nsj {
                                let jj = cj * nsj + mj;
                                for mi in 0..nsi {
                                    let ii = ci * nsi + mi;
                                    let dst = staging_comp_base + ii + jj * ni_sph;
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Representation::Cart => {
                let ni_cart = n_ctr_i * nci;
                let nj_cart = n_ctr_j * ncj;
                let cart_block = ni_cart * nj_cart;
                for comp in 0..3usize {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let src_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let block = &cart_3comp[src_base..src_base + block_len];
                            let staging_comp_base = comp * cart_block;
                            for jc in 0..ncj {
                                let jj = cj * ncj + jc;
                                for ic in 0..nci {
                                    let ii = ci * nci + ic;
                                    let dst = staging_comp_base + ii + jj * ni_cart;
                                    if dst < staging.len() {
                                        staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Representation::Spinor => {
                // Spin-free cart→spinor transform applied PER COMPONENT, mirroring the
                // SCALAR spinor 1e path (one_electron.rs ~line 2857). The 3-component
                // Cartesian gradient is already on-device; the c2s_sf_1e analogue stays
                // host-side per project convention. General contraction (nctr>1) is not
                // wired for the spinor transform — same guard as the scalar spinor path.
                if n_ctr_i != 1 || n_ctr_j != 1 {
                    return Err(cintxRsError::UnsupportedApi {
                        requested: "spinor 1e gradient with general contraction (nctr>1)"
                            .to_owned(),
                    });
                }
                // di/dj are the spinor component counts; each spinor block is
                // di*dj*2 interleaved-complex F elements (the per-component staging stride).
                let di = spinor_len(li, shell_i.kappa as i32);
                let dj = spinor_len(lj, shell_j.kappa as i32);
                let spinor_block = di * dj * 2;
                // nctr=1 → cart pair base is 0, total_len = 3 * block_len.
                for comp in 0..3usize {
                    let src_base = comp * block_len;
                    let block = &cart_3comp[src_base..src_base + block_len];
                    let staging_comp_base = comp * spinor_block;
                    // The device gradient kernels emit each per-component Cartesian
                    // block ket-major / bra-fastest (`block[cj_idx*nci + ci_idx]`),
                    // but `cart_to_spinor_sf_2d` expects its `cart` argument
                    // bra-major / ket-fastest (`cart[bra*ncj + ket]`, see
                    // c2spinor.rs apply_bra_block: `cart[n*ncj + j]`). Transpose each
                    // per-component block into bra-major before the spin-free
                    // cart→spinor transform so the bra/ket coefficient roles line up
                    // with libcint c2s_sf_1e. (For square symmetric blocks this is a
                    // no-op, which is why the asymmetric nuclear-gradient operators
                    // ipnuc/iprinv are the ones that surface the orientation.)
                    let mut block_bra_major = vec![0.0f64; block_len];
                    for ic in 0..nci {
                        for jc in 0..ncj {
                            block_bra_major[ic * ncj + jc] = block[jc * nci + ic];
                        }
                    }
                    cart_to_spinor_sf_2d::<F>(
                        &mut staging[staging_comp_base..staging_comp_base + spinor_block],
                        &block_bra_major,
                        li,
                        shell_i.kappa,
                        lj,
                        shell_j.kappa,
                    )?;
                }
            }
        }

        // Nonzero sentinel
        let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
            1e-12
        } else {
            1e-18
        });
        let not0 = staging
            .iter()
            .filter(|&&v| v.abs() > nonzero_threshold)
            .count() as i32;

        let staging_bytes = staging.len() * std::mem::size_of::<F>();
        return Ok(ExecutionStats {
            workspace_bytes: plan.workspace.bytes,
            required_workspace_bytes: plan.workspace.required_bytes,
            peak_workspace_bytes: staging_bytes,
            chunk_count: 1,
            planned_batches: 1,
            transfer_bytes: staging_bytes,
            not0,
            fallback_reason: plan.workspace.fallback_reason,
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar path (overlap / kinetic / nuclear)
    // ─────────────────────────────────────────────────────────────────────────

    // Scalar/gradient fork rationale (no silent narrowing):
    //   This task ports ONLY the three scalar operators (overlap/kinetic/nuclear)
    //   onto the CubeCL device kernel (`one_electron_scalar_kernel`). The gradient
    //   operators (ipovlp/ipkin/ipnuc/iprinv) — handled in the block above and
    //   returned before reaching here — and the spinor representation KEEP their
    //   existing host code paths unchanged. Each derivative is a distinct
    //   nabla1i / D_j^2 mixing pipeline with +1/+2 angular headroom; porting all
    //   four kernels plus the spinor transform on-device exceeds a single
    //   quick-task budget. On-device port of the gradient + spinor arms is
    //   deferred to a follow-up quick task — scalar-at-minimum scoping per the
    //   task constraint, matching how 3c2e/ECP staged their device-kernel ports.

    // Primitive / contraction counts.
    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;

    // Per-contraction-pair Cartesian accumulators. Generally-contracted shells
    // (nctr>1, e.g. ANO/ANO-RCC) emit ONE nci*ncj Cartesian block per (ci,cj)
    // contraction pair, contraction-major / bra-fastest. Block (ci,cj) starts at
    // flat offset (ci*n_ctr_j + cj) * block_len. The device kernel produces this
    // exact layout, so the readback is the same `cart_blocks` the host loop built.
    let block_len = nci * ncj;

    // Operator → comptime op_kind + nroots for the device kernel.
    //   overlap = 0 (nroots=1, no Rys)
    //   kinetic = 1 (nroots=1, no Rys)
    //   nuclear = 2 (nroots = (li+lj)/2 + 1, Rys quadrature)
    let (op_kind, nroots) = if is_overlap {
        (0u32, 1u32)
    } else if is_kinetic {
        (1u32, 1u32)
    } else {
        (2u32, (li as u32 + lj as u32) / 2 + 1)
    };

    // Device Rys kernels cover nroots<=5 (li+lj<=8). H2O/STO-3G stays well within.
    if op_kind == 2 && nroots as usize > MAX_DEVICE_NROOTS {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_1e",
            detail: format!(
                "device 1e nuclear kernel supports nroots<={MAX_DEVICE_NROOTS} (l_i+l_j<=8); \
                 got nroots={nroots} for l_i={li}, l_j={lj}"
            ),
        });
    }

    // Flatten the f64 primitive data the kernel reads.
    let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
    let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
    let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

    // Nuclear: flatten ALL atom coords + charges (-Z_C) in passed order for a
    // bit-stable reduction (mirrors `contract_nuclear`'s atom loop, D-10).
    let (atom_coords, atom_charges): (Vec<f64>, Vec<f64>) = if op_kind == 2 {
        let mut coords = Vec::with_capacity(atoms.len() * 3);
        let mut charges = Vec::with_capacity(atoms.len());
        for atom in atoms {
            coords.extend_from_slice(&atom.coord_bohr);
            charges.push(atom.atomic_number as f64);
        }
        (coords, charges)
    } else {
        (Vec::new(), Vec::new())
    };

    // Dispatch the scalar arm onto the resolved backend's device client (f64).
    let cart_blocks = run_1e_scalar_on_backend(
        backend,
        op_kind,
        nroots,
        li as u32,
        lj as u32,
        n_prim_i as u32,
        n_prim_j as u32,
        n_ctr_i as u32,
        n_ctr_j as u32,
        ri,
        rj,
        &exps_i,
        &exps_j,
        &coeff_i,
        &coeff_j,
        &atom_coords,
        &atom_charges,
    );
    let mut cart_blocks = cart_blocks;

    // Apply the libcint `CINTcommon_fac_sp` normalization scale to the
    // accumulated Cartesian buffer.  libcint moves the spherical normalization
    // for s (l=0) and p (l=1) shells out of the c2s tables and into the
    // primitive loop (`g1e.c` line 120: `common_factor * CINTcommon_fac_sp(i_l)
    // * CINTcommon_fac_sp(j_l)`). The c2s coefficient tables in `cart2sph.c`
    // therefore use 1.0 for s and p, and the cintx C2S_L0/C2S_L1 constants
    // match that convention. Without this scale factor, s/p-type integrals
    // are off by ~4*pi relative to vendored libcint output.
    let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
    if (sp_scale - 1.0).abs() > 1e-15 {
        for v in cart_blocks.iter_mut() {
            *v *= sp_scale;
        }
    }

    // Apply cart-to-sph or cart-to-spinor transform, or copy Cartesian to staging.
    // Intermediate transforms use a f64 temporary buffer; final values are cast to F
    // via F::from_f64_lossy. For f64 this is a zero-cost identity; for f32 it truncates.
    match plan.representation {
        Representation::Spheric => {
            // Per-contraction-pair cart→sph, scattered into the contraction-major
            // AO grid. di_sph = n_ctr_i * nsi = shell_i.ao_per_shell(). The staging
            // block is column-major in (bra,ket) with the bra index fastest
            // (staging[ii + jj*di_sph]) — byte-identical to the prior single-block
            // linear copy when n_ctr_i == n_ctr_j == 1 (di_sph == nsi).
            let di_sph = n_ctr_i * nsi;
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    let base = (ci * n_ctr_j + cj) * block_len;
                    let mut sph_tmp = vec![0.0_f64; nsi * nsj];
                    cart_to_sph_1e(&cart_blocks[base..base + block_len], &mut sph_tmp, li, lj);
                    for mj in 0..nsj {
                        let jj = cj * nsj + mj;
                        for mi in 0..nsi {
                            let ii = ci * nsi + mi;
                            let dst = ii + jj * di_sph;
                            if dst < staging.len() {
                                staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            // Single-contraction spinor preserves the exact prior behavior.
            // General contraction (nctr>1) is not wired for the spinor transform
            // (and is not exercised by the non-relativistic callers); return an
            // explicit error rather than silently truncating to the (0,0) block.
            if n_ctr_i != 1 || n_ctr_j != 1 {
                return Err(cintxRsError::UnsupportedApi {
                    requested: "spinor 1e with general contraction (nctr>1)".to_owned(),
                });
            }
            // cart_blocks is exactly one nci*ncj block here (nctr=1 enforced above).
            // The device scalar kernel emits it ket-major / bra-fastest
            // (block[cj*nci + ci]), but cart_to_spinor_sf_2d reads bra-major /
            // ket-fastest (cart[bra*ncj + ket], see c2spinor.rs apply_bra_block:
            // cart[n*ncj + j]). Transpose to bra-major before the spin-free
            // cart→spinor transform so the bra/ket coefficient roles line up with
            // libcint c2s_sf_1e — identical to the GRADIENT arm fix (260529-jtd).
            // For square symmetric blocks (an s side, or the intrinsically
            // transpose-symmetric overlap p×p block) this is a no-op, which is why a
            // NON-SQUARE asymmetric p×d cross block is the configuration that surfaces
            // the orientation.
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            let mut cart_bra_major = vec![0.0f64; nci * ncj];
            for ic in 0..nci {
                for jc in 0..ncj {
                    cart_bra_major[ic * ncj + jc] = cart_blocks[jc * nci + ic];
                }
            }
            cart_to_spinor_sf_2d::<F>(staging, &cart_bra_major, li, kappa_i, lj, kappa_j)?;
        }
        Representation::Cart => {
            // Each contraction block is column-major [nci, ncj] (bra fastest:
            // block[jc*nci + ic]); scatter it into the contraction-major AO grid
            // column-major (staging[ii + jj*di_cart], ii = ci*nci+ic) — matching
            // pyscf-rs's Cart stitch (block[ii + jj*ni]). For n_ctr_i==n_ctr_j==1
            // this is the single-block layout cart_to_sph/stitch already expect.
            let di_cart = n_ctr_i * nci;
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    let base = (ci * n_ctr_j + cj) * block_len;
                    let block = &cart_blocks[base..base + block_len];
                    for jc in 0..ncj {
                        let jj = cj * ncj + jc;
                        for ic in 0..nci {
                            let ii = ci * nci + ic;
                            let dst = ii + jj * di_cart;
                            if dst < staging.len() {
                                staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Per-symbol nonzero sentinel: count staging elements with |v| > threshold.
    // F: CintFloat includes num_traits::Float which provides .abs() via method syntax.
    // WR-06: use a precision-aware sentinel so f32 stale lanes (< f32 noise floor ~1e-7)
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

/// Real 1e integral host-side kernel for overlap, kinetic, and nuclear attraction.
///
/// Outer precision dispatcher: keeps the registered `FamilyLaunchFn` signature
/// (`fn(..., &mut [f64]) -> ...`) so the `as FamilyLaunchFn` cast in `kernels/mod.rs`
/// compiles unchanged. Internally matches on `plan.precision` and delegates to the
/// generic inner `launch_one_electron_typed::<F>`, reinterpreting the staging buffer
/// via `bytemuck::cast_slice_mut` for the F32 arm (A5 proven sound in Plan 01).
///
/// # Precision dispatch
/// - `PrecisionKind::F64` (default): passes `staging` directly to the `f64` inner.
///   Zero cost — no cast required, byte-identical to the pre-generic code.
/// - `PrecisionKind::F32`: reinterprets `staging: &mut [f64]` as `&mut [f32]` via
///   `bytemuck::cast_slice_mut` (8-byte aligned f64 buffer satisfies 4-byte f32 align;
///   A3 confirmed 2×M f32 lanes per M f64 slots), then calls the `f32` inner.
pub fn launch_one_electron(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            // F64 arm: staging is already &mut [f64] = &mut [F]; zero cast.
            launch_one_electron_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            // F32 arm: capture the true output element count BEFORE the bytemuck cast
            // (api.rs sizes Vec<f64> to chunk_len == the TRUE output element count;
            // after cast staging_f32.len() == chunk_len*2, so out_elems = staging.len() pre-cast).
            let out_elems = staging.len(); // f64 slice length == TRUE output element count
            let staging_f32: &mut [f32] = bytemuck::cast_slice_mut(staging);
            if staging_f32.len() < out_elems {
                return Err(cintxRsError::BufferTooSmall {
                    required: out_elems,
                    provided: staging_f32.len(),
                });
            }
            launch_one_electron_typed::<f32>(
                backend,
                plan,
                specialization,
                &mut staging_f32[..out_elems],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::boys::boys_gamma_inc_host;
    use crate::math::pdata::compute_pdata_host;
    use crate::math::rys::rys_root2_host;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: rys_root2_host weight-sum identity
    // The sum of Rys weights for nroots=2 at argument x should equal F_0(x)
    // (zeroth Boys function), because the Rys quadrature integrates exp(-x*t^2)
    // and sum(w_n) = F_0(x) = integral_0^1 exp(-x*t^2) dt = F_0(x).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_rys_root2_host_identity() {
        let x = 0.5_f64;
        let (_, w) = rys_root2_host(x);
        let w_sum = w[0] + w[1];
        // F_0(x) from Boys function
        let f0 = boys_gamma_inc_host(x, 0)[0];
        assert!(
            (w_sum - f0).abs() < 1e-8,
            "weight sum {w_sum} should equal F_0({x}) = {f0}, diff = {}",
            (w_sum - f0).abs()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: s-s overlap, same center — analytic result
    // For ai=aj=1, ri=rj=[0,0,0]: S = fac * sqrt(pi)^3 / (2*sqrt(2))
    // where zeta = ai+aj = 2, fac = 1 (exp(0)*1*1), gz0 = SQRTPI*PI/(zeta*sqrt(zeta))
    // S = gz0 * gx[0] * gy[0] = fac * SQRTPI * PI / (2 * sqrt(2))
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ovlp_ss_same_center() {
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let pd = compute_pdata_host(ai, aj, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);

        let nmax = 0u32;
        let g = fill_g_tensor_overlap(&pd, [0.0; 3], [0.0; 3], nmax, 0);

        // gz[0] = fac * SQRTPI * PI / (zeta * sqrt(zeta)) = 1 * SQRTPI * PI / (2 * sqrt(2))
        let gz0 = g[2]; // gz is at offset 2*g_per_axis, g_per_axis=1
        assert!(gz0 > 0.0, "gz0 should be positive");

        // Full s-s overlap: gx[0] * gy[0] * gz[0] = 1 * 1 * gz0
        let s_ss = g[0] * g[1] * gz0; // gx[0]*gy[0]*gz[0]
        let expected = SQRTPI * std::f64::consts::PI / (2.0 * 2.0_f64.sqrt());
        assert!(s_ss > 0.0, "s-s overlap should be positive");
        assert!(
            (s_ss - expected).abs() < 1e-10,
            "s-s overlap {s_ss} should equal {expected}, diff = {}",
            (s_ss - expected).abs()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: s-s overlap displaced — still positive, but less than same-center
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ovlp_ss_displaced() {
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4_f64, 0.0, 0.0];

        let pd_same = compute_pdata_host(ai, aj, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);
        let pd_disp =
            compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        let nmax = 0u32;
        let g_same = fill_g_tensor_overlap(&pd_same, [0.0; 3], [0.0; 3], nmax, 0);
        let g_disp = fill_g_tensor_overlap(&pd_disp, ri, rj, nmax, 0);

        let s_same = g_same[0] * g_same[1] * g_same[2];
        let s_disp = g_disp[0] * g_disp[1] * g_disp[2];

        assert!(s_disp > 0.0, "displaced s-s overlap should be positive");
        assert!(
            s_disp < s_same,
            "displaced overlap {s_disp} should be less than same-center {s_same}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: s-s kinetic integral — positive for same center
    // T_ss = 0.5 * d2z contribution from gz with jz=0:
    //   d2_jz=0 = 4*aj^2*gz[2] - 2*aj*1*gz[0] + 0
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_kinetic_ss_positive() {
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let pd = compute_pdata_host(ai, aj, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);

        // Kinetic uses nmax = li+lj+2 = 0+0+2 = 2
        let nmax = 2u32;
        let g = fill_g_tensor_overlap(&pd, [0.0; 3], [0.0; 3], nmax, 0);

        // For s-s: jx=jy=jz=0, ix=iy=iz=0
        // d2x = 4*aj^2*gx[2] - 2*aj*1*gx[0] + 0
        // g_per_axis = (nmax+1)*(lj+1) = 3*1 = 3
        // gx[0]=g[0], gx[2]=g[2], gy[0]=g[3], gy[2]=g[5], gz[0]=g[6], gz[2]=g[8]
        let gx = &g[0..3];
        let gy = &g[3..6];
        let gz_arr = &g[6..9];

        let d2x = 4.0 * aj * aj * gx[2] - 2.0 * aj * 1.0 * gx[0];
        let d2y = 4.0 * aj * aj * gy[2] - 2.0 * aj * 1.0 * gy[0];
        let d2z = 4.0 * aj * aj * gz_arr[2] - 2.0 * aj * 1.0 * gz_arr[0];

        // T = -0.5*(d2x*gy[0]*gz[0] + gx[0]*d2y*gz[0] + gx[0]*gy[0]*d2z)
        // The minus sign is needed because D_j^2 g < 0 for Gaussians.
        let t_ss = -0.5 * (d2x * gy[0] * gz_arr[0] + gx[0] * d2y * gz_arr[0] + gx[0] * gy[0] * d2z);
        assert!(
            t_ss > 0.0,
            "s-s kinetic integral should be positive, got {t_ss}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: s-s nuclear attraction — negative for a proton at origin
    // For a proton (Z=1) at the origin attracting s-type Gaussians, the integral
    // should be negative (attractive potential).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_nuclear_ss_negative() {
        use cintx_core::{Atom, NuclearModel};

        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let pd = compute_pdata_host(ai, aj, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);

        // A proton at origin
        let proton = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = [proton];

        let result = contract_nuclear(&pd, [0.0; 3], [0.0; 3], 0, 0, &atoms);
        assert!(result.len() == 1, "s-s integral should have 1 component");
        assert!(
            result[0] < 0.0,
            "s-s nuclear attraction should be negative for proton at origin, got {}",
            result[0]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T04-1a: launch_one_electron_typed::<f64> writes a positive finite
    // s-s overlap. This is the generic inner that the dispatcher delegates to.
    // RED: compile fails until launch_one_electron_typed is implemented.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_precision_dispatch_f64_inner_positive_overlap() {
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;
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
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let mut plan = ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 1];

        // Call the generic _typed inner directly (f64 monomorphization).
        // RED: compile fails until launch_one_electron_typed is defined.
        let result = launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "f64 inner should succeed: {:?}", result);
        assert!(
            staging[0].is_finite(),
            "f64 overlap should be finite, got {}",
            staging[0]
        );
        assert!(
            staging[0] > 0.0,
            "s-s overlap should be positive, got {}",
            staging[0]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T04-1b: launch_one_electron_typed::<f32> writes a positive finite
    // f32 s-s overlap. RED: compile fails until the typed inner exists; then
    // fails at runtime until f32 math is correctly wired.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_precision_dispatch_f32_inner_positive_overlap() {
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;
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
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();

        let opts = ExecutionOptions::default();
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let mut plan = ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        // Staging is 1 f64 = 8 bytes. The f32 inner writes 1 f32 (4 bytes) at index 0.
        let mut staging_f32 = vec![0.0_f32; 1];

        // Call the generic _typed inner directly (f32 monomorphization).
        // RED: compile fails until launch_one_electron_typed is defined.
        let result = launch_one_electron_typed::<f32>(&backend, &plan, &spec, &mut staging_f32);
        assert!(result.is_ok(), "f32 inner should succeed: {:?}", result);
        assert!(
            staging_f32[0].is_finite(),
            "f32 overlap should be finite, got {}",
            staging_f32[0]
        );
        assert!(
            staging_f32[0] > 0.0,
            "s-s overlap (f32) should be positive, got {}",
            staging_f32[0]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: rys_root2_host returns valid roots (0,1) and positive weights
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_rys_root2_host_valid_roots() {
        for x in [0.01, 0.5, 2.0, 5.0, 15.0, 35.0, 45.0] {
            let (u, w) = rys_root2_host(x);
            assert!(
                u[0] >= 0.0,
                "root u[0] should be non-negative for x={x}, got {}",
                u[0]
            );
            assert!(
                u[1] >= 0.0,
                "root u[1] should be non-negative for x={x}, got {}",
                u[1]
            );
            assert!(
                w[0] > 0.0,
                "weight w[0] should be positive for x={x}, got {}",
                w[0]
            );
            assert!(
                w[1] > 0.0,
                "weight w[1] should be positive for x={x}, got {}",
                w[1]
            );
            assert!(
                u[0] <= u[1],
                "roots should be ordered u[0] <= u[1] for x={x}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Device-vs-host cross-check (CpuRuntime, f64): the new
    // `run_1e_scalar_device::<CpuRuntime>` device kernel must reproduce the host
    // `contract_overlap` / `contract_kinetic` / `contract_nuclear` references
    // within atol=1e-12 + rtol=1e-10, for li,lj in
    // {(0,0),(0,1),(1,0),(1,1),(2,2)}. Modeled on
    // `center_2c2e.rs::assert_device_matches_host`.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    fn cpu_client_1e() -> cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::Runtime;
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Host overlap reference: single-primitive, single-contraction shell pair.
    #[cfg(feature = "cpu")]
    fn host_overlap_block(
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        li: u8,
        lj: u8,
    ) -> Vec<f64> {
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let nmax = (li + lj) as u32;
        let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32);
        contract_overlap(&g, li, lj, nmax)
    }

    /// Host kinetic reference.
    #[cfg(feature = "cpu")]
    fn host_kinetic_block(
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        li: u8,
        lj: u8,
    ) -> Vec<f64> {
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let nmax = (li + lj) as u32 + 2;
        let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32 + 2);
        contract_kinetic(&g, li, lj, nmax, aj)
    }

    /// Host nuclear reference (single-primitive) over an atom slab.
    #[cfg(feature = "cpu")]
    fn host_nuclear_block(
        ai: f64,
        aj: f64,
        ri: [f64; 3],
        rj: [f64; 3],
        li: u8,
        lj: u8,
        atoms: &[cintx_core::Atom],
    ) -> Vec<f64> {
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        contract_nuclear(&pd, ri, rj, li, lj, atoms)
    }

    #[cfg(feature = "cpu")]
    fn assert_close(host: &[f64], dev: &[f64], tag: &str) {
        assert_eq!(host.len(), dev.len(), "length mismatch ({tag})");
        for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
            let diff = (h - d).abs();
            let thr = 1e-12 + 1e-10 * h.abs();
            assert!(
                diff <= thr,
                "device/host mismatch ({tag}) idx={idx}: host={h:.15e} dev={d:.15e} diff={diff:.3e}"
            );
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_matches_host_overlap() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1), (2, 2)] {
            let host = host_overlap_block(ai, aj, ri, rj, li, lj);
            let dev = run_1e_scalar_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                0,
                1,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
                &[],
                &[],
            );
            assert_close(&host, &dev, &format!("overlap li={li} lj={lj}"));
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_matches_host_kinetic() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1), (2, 2)] {
            let host = host_kinetic_block(ai, aj, ri, rj, li, lj);
            let dev = run_1e_scalar_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                1,
                1,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
                &[],
                &[],
            );
            assert_close(&host, &dev, &format!("kinetic li={li} lj={lj}"));
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_matches_host_nuclear() {
        use cintx_core::{Atom, NuclearModel};
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        let atoms = [
            Atom::try_new(8, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
            Atom::try_new(1, [0.4, 0.3, 0.9], NuclearModel::Point, None, None).unwrap(),
        ];
        // Flatten atom slab for the device call (coords in passed order, +Z charges;
        // the kernel applies the -Z_C sign internally via fac1).
        let mut coords = Vec::new();
        let mut charges = Vec::new();
        for a in &atoms {
            coords.extend_from_slice(&a.coord_bohr);
            charges.push(a.atomic_number as f64);
        }
        // contract_nuclear only implements nrys<=2 (li+lj<=3) for the host ref, so
        // limit the cross-check pairs to li+lj<=3 (overlap/kinetic cover (2,2)).
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            let nroots = (li as u32 + lj as u32) / 2 + 1;
            let host = host_nuclear_block(ai, aj, ri, rj, li, lj, &atoms);
            let dev = run_1e_scalar_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                2,
                nroots,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
                &coords,
                &charges,
            );
            assert_close(&host, &dev, &format!("nuclear li={li} lj={lj}"));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Device-vs-host cross-checks for the 1e GRADIENT operators (quick-260529-j7d).
    // CpuRuntime, f64, atol=1e-12/rtol=1e-10, 3-component component-leading output.
    // ─────────────────────────────────────────────────────────────────────────

    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_matches_host_ipovlp() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1), (2, 2)] {
            // Host reference: overlap G-tensor with nmax = li+lj+1, then bra nabla.
            let pd =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nmax = (li + lj) as u32 + 1;
            let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32);
            let host = contract_grad_1e_bra(&g, li, lj, nmax, ai);
            let dev = run_1e_grad_bra_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                0,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
            );
            assert_close(&host, &dev, &format!("ipovlp li={li} lj={lj}"));
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_matches_host_ipkin() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1), (2, 2)] {
            // Host reference: G-tensor with lj_ext=lj+2 AND nmax=li+lj+3.
            let pd =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nmax = (li + lj) as u32 + 3;
            let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32 + 2);
            let host = contract_ipkin(&g, li, lj, nmax, ai, aj);
            let dev = run_1e_grad_bra_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                1,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
            );
            assert_close(&host, &dev, &format!("ipkin li={li} lj={lj}"));
        }
    }

    /// Both-side rank-9 `int1e_ipovlpip` device kernel vs an independent host
    /// recomputation of the SAME libcint `hess.c` composition (g1=D_j, g2=D_i,
    /// g3=D_iD_j; 9 components in bra-major direct order). Validates the CubeCL
    /// device lowering + index arithmetic. NON-SQUARE p×d / d×p blocks are
    /// included per Phase 23 D-05 — a square block is transpose-symmetric and
    /// would hide a component-ordering/layout bug.
    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_ipovlpip_matches_host_reference() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1), (1, 2), (2, 1)] {
            let li_u = li as usize;
            let lj_u = lj as usize;
            let nmax = (li + lj) as u32 + 2;
            let lj_ext = lj as u32 + 1;
            let dj = (nmax + 1) as usize;
            let g_per_axis = ((nmax + 1) * (lj_ext + 1)) as usize;

            let pd =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let g0 = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj_ext);

            // g1 = D_j(g0) over i..=li+1, j..=lj.
            // g2 = D_i(g0), g3 = D_i(g1) over i..=li, j..=lj.
            let mut g1 = vec![0.0f64; 3 * g_per_axis];
            let mut g2 = vec![0.0f64; 3 * g_per_axis];
            let mut g3 = vec![0.0f64; 3 * g_per_axis];
            let ai2 = -2.0 * ai;
            let aj2 = -2.0 * aj;
            for axis in 0..3usize {
                let off = axis * g_per_axis;
                for jn in 0..=lj_u {
                    let jbase = jn * dj;
                    let jhi = (jn + 1) * dj;
                    for ii in 0..=(li_u + 1) {
                        let mut val = aj2 * g0[off + jhi + ii];
                        if jn >= 1 {
                            let jlo = (jn - 1) * dj;
                            val += (jn as f64) * g0[off + jlo + ii];
                        }
                        g1[off + jbase + ii] = val;
                    }
                }
                for jn in 0..=lj_u {
                    let jbase = jn * dj;
                    g2[off + jbase] = ai2 * g0[off + jbase + 1];
                    g3[off + jbase] = ai2 * g1[off + jbase + 1];
                    for ii in 1..=li_u {
                        g2[off + jbase + ii] =
                            (ii as f64) * g0[off + jbase + ii - 1] + ai2 * g0[off + jbase + ii + 1];
                        g3[off + jbase + ii] =
                            (ii as f64) * g1[off + jbase + ii - 1] + ai2 * g1[off + jbase + ii + 1];
                    }
                }
            }

            let nci = (li_u + 1) * (li_u + 2) / 2;
            let ncj = (lj_u + 1) * (lj_u + 2) / 2;
            let block_len = nci * ncj;
            let gx = 0usize;
            let gy = g_per_axis;
            let gz = 2 * g_per_axis;
            let mut host = vec![0.0f64; 9 * block_len];
            let mut cj_idx = 0usize;
            for ja in 0..=lj_u {
                let jx = lj_u - ja;
                for jb in 0..=(lj_u - jx) {
                    let jy = (lj_u - jx) - jb;
                    let jz = lj_u - jx - jy;
                    let mut ci_idx = 0usize;
                    for ia in 0..=li_u {
                        let ix = li_u - ia;
                        for ib in 0..=(li_u - ix) {
                            let iy = (li_u - ix) - ib;
                            let iz = li_u - ix - iy;
                            let nx = jx * dj + ix;
                            let ny = jy * dj + iy;
                            let nz = jz * dj + iz;
                            let g0x = g0[gx + nx];
                            let g0y = g0[gy + ny];
                            let g0z = g0[gz + nz];
                            let g1x = g1[gx + nx];
                            let g1y = g1[gy + ny];
                            let g1z = g1[gz + nz];
                            let g2x = g2[gx + nx];
                            let g2y = g2[gy + ny];
                            let g2z = g2[gz + nz];
                            let g3x = g3[gx + nx];
                            let g3y = g3[gy + ny];
                            let g3z = g3[gz + nz];
                            let s = [
                                g3x * g0y * g0z,
                                g2x * g1y * g0z,
                                g2x * g0y * g1z,
                                g1x * g2y * g0z,
                                g0x * g3y * g0z,
                                g0x * g2y * g1z,
                                g1x * g0y * g2z,
                                g0x * g1y * g2z,
                                g0x * g0y * g3z,
                            ];
                            let elem = cj_idx * nci + ci_idx;
                            for comp in 0..9 {
                                host[comp * block_len + elem] = s[comp];
                            }
                            ci_idx += 1;
                        }
                    }
                    cj_idx += 1;
                }
            }

            let dev = run_1e_grad_both_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
            );
            assert_close(&host, &dev, &format!("ipovlpip li={li} lj={lj}"));
        }
    }

    /// Host ket-nabla `D_j` of a 3-axis 1e G-tensor (test reference).
    fn host_dj_1e(src: &[f64], gpa: usize, dj: usize, jmax: usize, imax: usize, aj2: f64) -> Vec<f64> {
        let mut dst = vec![0.0f64; 3 * gpa];
        for axis in 0..3 {
            let off = axis * gpa;
            for jn in 0..=jmax {
                let jbase = jn * dj;
                let jhi = (jn + 1) * dj;
                for ii in 0..=imax {
                    let mut v = aj2 * src[off + jhi + ii];
                    if jn >= 1 {
                        v += (jn as f64) * src[off + (jn - 1) * dj + ii];
                    }
                    dst[off + jbase + ii] = v;
                }
            }
        }
        dst
    }

    /// Host bra-nabla `D_i` of a 3-axis 1e G-tensor (test reference).
    fn host_di_1e(src: &[f64], gpa: usize, dj: usize, jmax: usize, imax: usize, ai2: f64) -> Vec<f64> {
        let mut dst = vec![0.0f64; 3 * gpa];
        for axis in 0..3 {
            let off = axis * gpa;
            for jn in 0..=jmax {
                let jbase = jn * dj;
                dst[off + jbase] = ai2 * src[off + jbase + 1];
                for ii in 1..=imax {
                    dst[off + jbase + ii] = (ii as f64) * src[off + jbase + ii - 1]
                        + ai2 * src[off + jbase + ii + 1];
                }
            }
        }
        dst
    }

    /// `int1e_ipkinip` device kernel vs host recomputation of libcint `hess.c`
    /// `CINTgout1e_int1e_ipkinip` (8 distinct tensors, 27-term recipe, -0.5).
    /// Includes NON-SQUARE p×d / d×p blocks per Phase 23 D-05.
    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_ipkinip_matches_host_reference() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1), (1, 2), (2, 1)] {
            let li_u = li as usize;
            let lj_u = lj as usize;
            let nmax = (li + lj) as usize + 4;
            let lj_ext = lj_u + 3;
            let dj = nmax + 1;
            let gpa = (nmax + 1) * (lj_ext + 1);
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let g0 = fill_g_tensor_overlap(&pd, ri, rj, nmax as u32, lj_ext as u32);
            let aj2 = -2.0 * aj;
            let ai2 = -2.0 * ai;
            let dj1 = host_dj_1e(&g0, gpa, dj, lj_u + 2, li_u + 1, aj2);
            let dj2 = host_dj_1e(&dj1, gpa, dj, lj_u + 1, li_u + 1, aj2);
            let dj3 = host_dj_1e(&dj2, gpa, dj, lj_u, li_u + 1, aj2);
            let di0 = host_di_1e(&g0, gpa, dj, lj_u, li_u, ai2);
            let di1 = host_di_1e(&dj1, gpa, dj, lj_u, li_u, ai2);
            let di2 = host_di_1e(&dj2, gpa, dj, lj_u, li_u, ai2);
            let di3 = host_di_1e(&dj3, gpa, dj, lj_u, li_u, ai2);
            let nci = (li_u + 1) * (li_u + 2) / 2;
            let ncj = (lj_u + 1) * (lj_u + 2) / 2;
            let block = nci * ncj;
            let (gx, gy, gz) = (0usize, gpa, 2 * gpa);
            let mut host = vec![0.0f64; 9 * block];
            let mut cjx = 0usize;
            for ja in 0..=lj_u {
                let jxx = lj_u - ja;
                for jb in 0..=(lj_u - jxx) {
                    let jyy = (lj_u - jxx) - jb;
                    let jzz = lj_u - jxx - jyy;
                    let mut cix = 0usize;
                    for ia in 0..=li_u {
                        let ixx = li_u - ia;
                        for ib in 0..=(li_u - ixx) {
                            let iyy = (li_u - ixx) - ib;
                            let izz = li_u - ixx - iyy;
                            let nx = jxx * dj + ixx;
                            let ny = jyy * dj + iyy;
                            let nz = jzz * dj + izz;
                            let a0x = g0[gx + nx];
                            let a0y = g0[gy + ny];
                            let a0z = g0[gz + nz];
                            let b1x = dj1[gx + nx];
                            let b1y = dj1[gy + ny];
                            let b1z = dj1[gz + nz];
                            let b2x = dj2[gx + nx];
                            let b2y = dj2[gy + ny];
                            let b2z = dj2[gz + nz];
                            let b3x = dj3[gx + nx];
                            let b3y = dj3[gy + ny];
                            let b3z = dj3[gz + nz];
                            let c0x = di0[gx + nx];
                            let c0y = di0[gy + ny];
                            let c0z = di0[gz + nz];
                            let c1x = di1[gx + nx];
                            let c1y = di1[gy + ny];
                            let c1z = di1[gz + nz];
                            let c2x = di2[gx + nx];
                            let c2y = di2[gy + ny];
                            let c2z = di2[gz + nz];
                            let c3x = di3[gx + nx];
                            let c3y = di3[gy + ny];
                            let c3z = di3[gz + nz];
                            let s = [
                                c3x * a0y * a0z + c1x * b2y * a0z + c1x * a0y * b2z,
                                c2x * b1y * a0z + c0x * b3y * a0z + c0x * b1y * b2z,
                                c2x * a0y * b1z + c0x * b2y * b1z + c0x * a0y * b3z,
                                b3x * c0y * a0z + b1x * c2y * a0z + b1x * c0y * b2z,
                                b2x * c1y * a0z + a0x * c3y * a0z + a0x * c1y * b2z,
                                b2x * c0y * b1z + a0x * c2y * b1z + a0x * c0y * b3z,
                                b3x * a0y * c0z + b1x * b2y * c0z + b1x * a0y * c2z,
                                b2x * b1y * c0z + a0x * b3y * c0z + a0x * b1y * c2z,
                                b2x * a0y * c1z + a0x * b2y * c1z + a0x * a0y * c3z,
                            ];
                            let elem = cjx * nci + cix;
                            for comp in 0..9 {
                                host[comp * block + elem] = -0.5 * s[comp];
                            }
                            cix += 1;
                        }
                    }
                    cjx += 1;
                }
            }
            let dev = run_1e_grad_kin_both_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
            );
            assert_close(&host, &dev, &format!("ipkinip li={li} lj={lj}"));
        }
    }

    /// `int1e_ipnucip` device kernel vs host recomputation of libcint `hess.c`
    /// `CINTgout1e_int1e_ipnucip` (nuclear Rys g0 + both-side g1/g2/g3, summed
    /// over roots and origins). Includes NON-SQUARE blocks per Phase 23 D-05.
    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_ipnucip_matches_host_reference() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        // Two nuclei with -Z charge factors (low→high order).
        let origins: [([f64; 3], f64); 2] = [([0.1, 0.2, 0.3], -1.0), ([0.8, -0.4, 0.5], -6.0)];
        for &(li, lj) in &[(0u8, 0u8), (1, 0), (0, 1), (1, 1), (1, 2), (2, 1)] {
            let li_u = li as usize;
            let lj_u = lj as usize;
            let nmax = (li + lj) as usize + 2;
            let lj_ext = lj_u + 1;
            let dj = nmax + 1;
            let gpa = (nmax + 1) * (lj_ext + 1);
            let nroots = ((li + lj) as usize + 2) / 2 + 1;
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let rp = [pd.center_p_x, pd.center_p_y, pd.center_p_z];
            let aj2 = -2.0 * aj;
            let ai2 = -2.0 * ai;
            let nci = (li_u + 1) * (li_u + 2) / 2;
            let ncj = (lj_u + 1) * (lj_u + 2) / 2;
            let block = nci * ncj;
            let (gx, gy, gz) = (0usize, gpa, 2 * gpa);
            let mut host = vec![0.0f64; 9 * block];
            for &(rc, charge) in &origins {
                let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
                let x_boys =
                    pd.zeta_ab * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);
                let (u_arr, w_arr) = rys_roots_host(nroots, x_boys);
                let fac1 = 2.0 * std::f64::consts::PI * charge * pd.fac / pd.zeta_ab;
                for n in 0..nroots {
                    let u_n = u_arr[n];
                    let w_n = w_arr[n];
                    let tau = u_n / (1.0 + u_n);
                    let rt = pd.aij2 * (1.0 - tau);
                    let c00 = [
                        (rp[0] - ri[0]) + tau * crij[0],
                        (rp[1] - ri[1]) + tau * crij[1],
                        (rp[2] - ri[2]) + tau * crij[2],
                    ];
                    let mut g0 = vec![0.0f64; 3 * gpa];
                    g0[gx] = 1.0;
                    g0[gy] = 1.0;
                    g0[gz] = fac1 * w_n;
                    for (axoff, c) in [(gx, c00[0]), (gy, c00[1]), (gz, c00[2])] {
                        crate::math::obara_saika::vrr_2e_step_host(
                            &mut g0[axoff..axoff + gpa],
                            c,
                            rt,
                            nmax as u32,
                            1,
                        );
                    }
                    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
                    for (axoff, r) in [(gx, rirj[0]), (gy, rirj[1]), (gz, rirj[2])] {
                        crate::math::obara_saika::hrr_step_host(
                            &mut g0[axoff..axoff + gpa],
                            r,
                            1,
                            (nmax + 1) as u32,
                            nmax as u32,
                            lj_ext as u32,
                        );
                    }
                    let g1 = host_dj_1e(&g0, gpa, dj, lj_u, li_u + 1, aj2);
                    let g2 = host_di_1e(&g0, gpa, dj, lj_u, li_u, ai2);
                    let g3 = host_di_1e(&g1, gpa, dj, lj_u, li_u, ai2);
                    let mut cjx = 0usize;
                    for ja in 0..=lj_u {
                        let jxx = lj_u - ja;
                        for jb in 0..=(lj_u - jxx) {
                            let jyy = (lj_u - jxx) - jb;
                            let jzz = lj_u - jxx - jyy;
                            let mut cix = 0usize;
                            for ia in 0..=li_u {
                                let ixx = li_u - ia;
                                for ib in 0..=(li_u - ixx) {
                                    let iyy = (li_u - ixx) - ib;
                                    let izz = li_u - ixx - iyy;
                                    let nx = jxx * dj + ixx;
                                    let ny = jyy * dj + iyy;
                                    let nz = jzz * dj + izz;
                                    let g0x = g0[gx + nx];
                                    let g0y = g0[gy + ny];
                                    let g0z = g0[gz + nz];
                                    let g1x = g1[gx + nx];
                                    let g1y = g1[gy + ny];
                                    let g1z = g1[gz + nz];
                                    let g2x = g2[gx + nx];
                                    let g2y = g2[gy + ny];
                                    let g2z = g2[gz + nz];
                                    let g3x = g3[gx + nx];
                                    let g3y = g3[gy + ny];
                                    let g3z = g3[gz + nz];
                                    let s = [
                                        g3x * g0y * g0z,
                                        g2x * g1y * g0z,
                                        g2x * g0y * g1z,
                                        g1x * g2y * g0z,
                                        g0x * g3y * g0z,
                                        g0x * g2y * g1z,
                                        g1x * g0y * g2z,
                                        g0x * g1y * g2z,
                                        g0x * g0y * g3z,
                                    ];
                                    let elem = cjx * nci + cix;
                                    for comp in 0..9 {
                                        host[comp * block + elem] += s[comp];
                                    }
                                    cix += 1;
                                }
                            }
                            cjx += 1;
                        }
                    }
                }
            }
            let mut oc = Vec::new();
            let mut och = Vec::new();
            for &(rc, charge) in &origins {
                oc.extend_from_slice(&rc);
                och.push(charge);
            }
            let dev = run_1e_nuc_grad_both_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                nroots as u32,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
                &oc,
                &och,
            );
            assert_close(&host, &dev, &format!("ipnucip li={li} lj={lj}"));
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_device_matches_host_nuclear_grad() {
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.6_f64, 0.5, 0.7];
        let ai = 0.9_f64;
        let aj = 1.3_f64;
        // 2-atom origins list (coords + charge factor -Z): ipnuc shape.
        let origins: [([f64; 3], f64); 2] =
            [([0.0, 0.0, 0.0], -8.0), ([0.4, 0.3, 0.9], -1.0)];
        let origin_coords: Vec<f64> = origins
            .iter()
            .flat_map(|(c, _)| c.iter().copied())
            .collect();
        let origin_charges: Vec<f64> = origins.iter().map(|(_, q)| *q).collect();

        // Keep li+lj<=3 so nrys = (li+lj+1)/2+1 <= 2 (matches the scalar nuclear
        // cross-check bound; rys_roots_host covers nrys<=5 but the host reference
        // exercises the device-supported grid).
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            let pd =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nroots = ((li as u32 + lj as u32) + 1) / 2 + 1;
            let host = contract_nuclear_grad(&pd, ri, rj, li, lj, ai, &origins);
            let dev = run_1e_nuc_grad_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                nroots,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
                &origin_coords,
                &origin_charges,
            );
            assert_close(&host, &dev, &format!("ipnuc li={li} lj={lj}"));
        }

        // iprinv-style case: single origin, charge factor +1.0.
        let iprinv_origins: [([f64; 3], f64); 1] = [([0.1, 0.2, 0.3], 1.0)];
        let ip_coords: Vec<f64> = iprinv_origins[0].0.to_vec();
        let ip_charges: Vec<f64> = vec![iprinv_origins[0].1];
        for &(li, lj) in &[(0u8, 0u8), (1, 1)] {
            let pd =
                compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nroots = ((li as u32 + lj as u32) + 1) / 2 + 1;
            let host = contract_nuclear_grad(&pd, ri, rj, li, lj, ai, &iprinv_origins);
            let dev = run_1e_nuc_grad_device::<cubecl::cpu::CpuRuntime>(
                &cpu_client_1e(),
                nroots,
                li as u32,
                lj as u32,
                1,
                1,
                1,
                1,
                ri,
                rj,
                &[ai],
                &[aj],
                &[1.0],
                &[1.0],
                &ip_coords,
                &ip_charges,
            );
            assert_close(&host, &dev, &format!("iprinv li={li} lj={lj}"));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Genericity evidence: the kernel compiles and runs for F=f32 — an s-s
    // overlap on CpuRuntime returns a finite positive value. Same shape as
    // `center_2c2e.rs::test_center_2c2e_kernel_generic_f32`.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_one_electron_scalar_kernel_generic_f32() {
        let client = cpu_client_1e();
        let exps_i = [1.0_f32];
        let exps_j = [1.0_f32];
        let coeff_i = [1.0_f32];
        let coeff_j = [1.0_f32];
        let coords = [0.0_f32]; // unused for overlap, must be len>0
        let charges = [0.0_f32];
        // overlap s-s: nmax=0, lj_ext=0, g_per_axis=1 → 3 g elements; out_len=1.
        let g_zero = [0.0_f32; 3];
        let rys_zero = [0.0_f32; 1];
        let out_zero = [0.0_f32; 1];

        let exps_i_h = client.create_from_slice(f32::as_bytes(&exps_i));
        let exps_j_h = client.create_from_slice(f32::as_bytes(&exps_j));
        let coeff_i_h = client.create_from_slice(f32::as_bytes(&coeff_i));
        let coeff_j_h = client.create_from_slice(f32::as_bytes(&coeff_j));
        let coords_h = client.create_from_slice(f32::as_bytes(&coords));
        let charges_h = client.create_from_slice(f32::as_bytes(&charges));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let u_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let w_h = client.create_from_slice(f32::as_bytes(&rys_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

        one_electron_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { ArrayArg::from_raw_parts(exps_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(exps_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_i_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coeff_j_h, 1) },
            unsafe { ArrayArg::from_raw_parts(coords_h, 1) },
            unsafe { ArrayArg::from_raw_parts(charges_h, 1) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(u_h, 1) },
            unsafe { ArrayArg::from_raw_parts(w_h, 1) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            0.0_f32, // rix
            0.0,     // riy
            0.0,     // riz
            0.0,     // rjx
            0.0,     // rjy
            1.4,     // rjz
            PIE4 as f32,
            SQRTPI as f32,
            std::f64::consts::PI as f32,
            0,    // li
            0,    // lj
            1,    // nprim_i
            1,    // nprim_j
            1,    // nctr_i
            1,    // nctr_j
            0,    // natm (unused for overlap)
            0u32, // op_kind = overlap
            1u32, // nroots
        );

        let raw = client.read_one_unchecked(out_h);
        let out = f32::from_bytes(&raw)[0];
        assert!(
            out.is_finite(),
            "f32 1e overlap kernel result must be finite"
        );
        assert!(
            out > 0.0,
            "s-s 1e overlap f32 result should be positive: {out}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GC-1: general-contraction (nctr>1) parity for s-shells.
    //
    // A generally-contracted s shell (2 shared primitives, 2 contraction columns)
    // paired with itself must produce the FULL 2x2 Gram matrix of the two contracted
    // functions — not a single truncated value (the bug summed every (ci,cj) pair
    // into the (0,0) slot, leaving 3/4 of the block zero). Parity reference: assemble
    // the 2x2 from segmented (nctr=1) single-column launches. No vendored data.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_general_contraction_s_parity() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple,
        };
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        // Resolve the spherical-overlap operator id (int1e_ovlp_sph is its own
        // manifest entry; OperatorId::new(0) is the cart variant). This exercises
        // the kernel's Spheric branch — the path pyscf-rs uses for int1e_ovlp_sph.
        let op = Resolver::descriptor_by_symbol("int1e_ovlp_sph")
            .expect("int1e_ovlp_sph must be in the cintx manifest")
            .id;

        // Overlap block for a single shell pair, all shells on one center at origin.
        let overlap = |sa: Arc<Shell>, sb: Arc<Shell>| -> Vec<f64> {
            let n = sa.ao_per_shell() * sb.ao_per_shell();
            let atoms: Arc<[Atom]> = Arc::from(
                vec![Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap()]
                    .into_boxed_slice(),
            );
            let all: Arc<[Arc<Shell>]> = Arc::from(vec![sa.clone(), sb.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let shells = ShellTuple::try_from_iter([sa, sb]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts)
                .unwrap();
            let mut plan =
                ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
            plan.precision = PrecisionKind::F64;
            let spec = SpecializationKey::from_plan(&plan);
            let backend = ResolvedBackend::Cpu(resolve_cpu_client().unwrap());
            let mut staging = vec![0.0_f64; n];
            launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging).unwrap();
            staging
        };

        // Two shared primitives; two contraction columns. coefficients[pi*nctr + ci].
        let exps: Arc<[f64]> = Arc::from(vec![3.5_f64, 0.8].into_boxed_slice());
        let coeffs_gc: Arc<[f64]> = Arc::from(vec![0.6_f64, -0.3, 0.4, 0.9].into_boxed_slice());
        let coeffs_c0: Arc<[f64]> = Arc::from(vec![0.6_f64, 0.4].into_boxed_slice());
        let coeffs_c1: Arc<[f64]> = Arc::from(vec![-0.3_f64, 0.9].into_boxed_slice());

        let gc = Arc::new(
            Shell::try_new(
                0,
                0,
                2,
                2,
                0,
                Representation::Spheric,
                exps.clone(),
                coeffs_gc,
            )
            .unwrap(),
        );
        let c0 = Arc::new(
            Shell::try_new(
                0,
                0,
                2,
                1,
                0,
                Representation::Spheric,
                exps.clone(),
                coeffs_c0,
            )
            .unwrap(),
        );
        let c1 = Arc::new(
            Shell::try_new(0, 0, 2, 1, 0, Representation::Spheric, exps, coeffs_c1).unwrap(),
        );

        let block = overlap(gc.clone(), gc.clone());
        assert_eq!(
            block.len(),
            4,
            "gc s nctr=2 self-overlap must be a 2x2 block, got {}",
            block.len()
        );

        let s00 = overlap(c0.clone(), c0.clone())[0];
        let s01 = overlap(c0.clone(), c1.clone())[0];
        let s11 = overlap(c1.clone(), c1.clone())[0];

        // Contraction-major, bra-fastest (di_sph = 2, nsi = 1): block[ci + cj*2].
        assert!(
            (block[0] - s00).abs() < 1e-12,
            "S[0,0] {} vs {}",
            block[0],
            s00
        );
        assert!(
            (block[2] - s01).abs() < 1e-12,
            "S[0,1] {} vs {}",
            block[2],
            s01
        );
        assert!(
            (block[1] - s01).abs() < 1e-12,
            "S[1,0] {} vs {}",
            block[1],
            s01
        );
        assert!(
            (block[3] - s11).abs() < 1e-12,
            "S[1,1] {} vs {}",
            block[3],
            s11
        );
        assert!(
            block[0] > 0.0 && block[3] > 0.0,
            "diagonal must be positive (PSD)"
        );
        assert!(
            (block[1] - block[2]).abs() < 1e-12,
            "block must be symmetric"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GC-2: general-contraction parity for p-shells, DISPLACED centers.
    //
    // This is the test that disambiguates contraction-MAJOR (AO = ctr*(2l+1) + m,
    // libcint/PySCF order) from contraction-minor for l>0. A gc p-shell (nctr=2)
    // on atom 0 paired with a plain s-shell on atom 1 yields a 6x1 block; rows
    // [0..3) must equal the (p_c0 | s) segmented overlap and rows [3..6) the
    // (p_c1 | s) overlap.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_general_contraction_p_parity_contraction_major() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple,
        };
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        // Resolve the spherical-overlap operator id (int1e_ovlp_sph is its own
        // manifest entry; OperatorId::new(0) is the cart variant). This exercises
        // the kernel's Spheric branch — the path pyscf-rs uses for int1e_ovlp_sph.
        let op = Resolver::descriptor_by_symbol("int1e_ovlp_sph")
            .expect("int1e_ovlp_sph must be in the cintx manifest")
            .id;

        // bra shell on atom 0 (origin), ket shell on atom 1 (displaced along x).
        let overlap = |bra: Arc<Shell>, ket: Arc<Shell>| -> Vec<f64> {
            let n = bra.ao_per_shell() * ket.ao_per_shell();
            let atoms: Arc<[Atom]> = Arc::from(
                vec![
                    Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                    Atom::try_new(1, [1.3, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                ]
                .into_boxed_slice(),
            );
            let all: Arc<[Arc<Shell>]> =
                Arc::from(vec![bra.clone(), ket.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let shells = ShellTuple::try_from_iter([bra, ket]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts)
                .unwrap();
            let mut plan =
                ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
            plan.precision = PrecisionKind::F64;
            let spec = SpecializationKey::from_plan(&plan);
            let backend = ResolvedBackend::Cpu(resolve_cpu_client().unwrap());
            let mut staging = vec![0.0_f64; n];
            launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging).unwrap();
            staging
        };

        let p_exps: Arc<[f64]> = Arc::from(vec![1.2_f64, 0.4].into_boxed_slice());
        // coefficients[pi*nctr + ci]: ctr0 = (0.7, 0.3), ctr1 = (0.2, 0.8)
        let p_gc_co: Arc<[f64]> = Arc::from(vec![0.7_f64, 0.2, 0.3, 0.8].into_boxed_slice());
        let p_c0_co: Arc<[f64]> = Arc::from(vec![0.7_f64, 0.3].into_boxed_slice());
        let p_c1_co: Arc<[f64]> = Arc::from(vec![0.2_f64, 0.8].into_boxed_slice());

        let p_gc = Arc::new(
            Shell::try_new(
                0,
                1,
                2,
                2,
                0,
                Representation::Spheric,
                p_exps.clone(),
                p_gc_co,
            )
            .unwrap(),
        );
        let p_c0 = Arc::new(
            Shell::try_new(
                0,
                1,
                2,
                1,
                0,
                Representation::Spheric,
                p_exps.clone(),
                p_c0_co,
            )
            .unwrap(),
        );
        let p_c1 = Arc::new(
            Shell::try_new(0, 1, 2, 1, 0, Representation::Spheric, p_exps, p_c1_co).unwrap(),
        );
        let s_ket = Arc::new(
            Shell::try_new(
                1,
                0,
                1,
                1,
                0,
                Representation::Spheric,
                Arc::from(vec![0.9_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );

        let block = overlap(p_gc, s_ket.clone()); // 6x1: di_sph=2*3=6, dj=1
        assert_eq!(
            block.len(),
            6,
            "gc p nctr=2 vs s must be a 6x1 block, got {}",
            block.len()
        );

        let seg0 = overlap(p_c0, s_ket.clone()); // 3 components of contraction 0
        let seg1 = overlap(p_c1, s_ket); // 3 components of contraction 1
        assert_eq!(seg0.len(), 3);
        assert_eq!(seg1.len(), 3);

        // Contraction-major: rows [0..3) = contraction 0, rows [3..6) = contraction 1.
        for m in 0..3 {
            assert!(
                (block[m] - seg0[m]).abs() < 1e-12,
                "ctr0 comp {m}: {} vs {}",
                block[m],
                seg0[m]
            );
            assert!(
                (block[3 + m] - seg1[m]).abs() < 1e-12,
                "ctr1 comp {m}: {} vs {}",
                block[3 + m],
                seg1[m]
            );
        }
        // The gc block must NOT be truncated: contraction-1 rows are non-zero.
        assert!(
            block[3..6].iter().any(|v| v.abs() > 1e-12),
            "contraction-1 rows must be populated, not truncated to zero"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test XL-1: cross-angular-momentum overlap symmetry (li != lj, both > 0).
    //
    // The full overlap matrix is symmetric, so the (a,b) shell-pair block must be
    // the transpose of the (b,a) block: <a_i|b_j> == <b_j|a_i>. This exercises the
    // Cartesian buffer layout for li != lj with BOTH nci,ncj > 1 (p-d, p-f, d-g) —
    // the case where row-major vs column-major actually differ (vectors and
    // symmetric same-l blocks hide the bug). Single contraction; displaced centers.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_cross_l_overlap_is_symmetric() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple,
        };
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        // Integral block for an ORDERED shell pair drawn from a FIXED 2-shell mol
        // (la on atom 0 @ origin, lb on atom 1 @ displaced). `swapped=false` gives
        // <la|op|lb> (bra=la,ket=lb); `swapped=true` gives <lb|op|la> on the SAME
        // geometry. The transpose symmetry <la_i|op|lb_j> == <lb_j|op|la_i> holds for
        // every Hermitian 1e operator (ovlp/kin/nuc) — independent of l. Covers
        // contract_overlap, contract_kinetic, and contract_nuclear.
        let block = |op_sym: &str, la: u8, lb: u8, swapped: bool| -> Vec<f64> {
            let op = Resolver::descriptor_by_symbol(op_sym)
                .unwrap_or_else(|_| panic!("{op_sym} must be in the cintx manifest"))
                .id;
            let atoms: Arc<[Atom]> = Arc::from(
                vec![
                    Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                    Atom::try_new(1, [0.8, 0.5, -0.3], NuclearModel::Point, None, None).unwrap(),
                ]
                .into_boxed_slice(),
            );
            let s_la = Arc::new(
                Shell::try_new(
                    0,
                    la,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.9_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            );
            let s_lb = Arc::new(
                Shell::try_new(
                    1,
                    lb,
                    1,
                    1,
                    0,
                    Representation::Spheric,
                    Arc::from(vec![0.6_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            );
            let all: Arc<[Arc<Shell>]> =
                Arc::from(vec![s_la.clone(), s_lb.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let (bra, ket) = if swapped {
                (s_lb.clone(), s_la.clone())
            } else {
                (s_la.clone(), s_lb.clone())
            };
            let n = bra.ao_per_shell() * ket.ao_per_shell();
            let shells = ShellTuple::try_from_iter([bra, ket]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts)
                .unwrap();
            let mut plan =
                ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
            plan.precision = PrecisionKind::F64;
            let spec = SpecializationKey::from_plan(&plan);
            let backend = ResolvedBackend::Cpu(resolve_cpu_client().unwrap());
            let mut staging = vec![0.0_f64; n];
            launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging).unwrap();
            staging
        };

        // p-d (1,2), p-f (1,3), d-g (2,4): all have nci,ncj > 1. ovlp/kin use the
        // analytic g-tensor and handle any l; contract_nuclear only implements
        // <=2 Rys roots (li+lj<=3), so its arm is limited to p-d (a separate
        // pre-existing high-l-nuclear limitation, orthogonal to the cross-l layout
        // fixed here). All three share the column-major cross-l Cartesian layout.
        let cases: [(&str, &[(u8, u8)]); 3] = [
            ("int1e_ovlp_sph", &[(1, 2), (1, 3), (2, 4)]),
            ("int1e_kin_sph", &[(1, 2), (1, 3), (2, 4)]),
            ("int1e_nuc_sph", &[(1, 2)]),
        ];
        for (op_sym, pairs) in cases {
            for &(la, lb) in pairs {
                let nsa = nsph(la);
                let nsb = nsph(lb);
                let ab = block(op_sym, la, lb, false); // ab[i + j*nsa] = <la_i | lb_j>
                let ba = block(op_sym, la, lb, true); // ba[j + i*nsb] = <lb_j | la_i>
                let mut max_asym = 0.0_f64;
                for i in 0..nsa {
                    for j in 0..nsb {
                        max_asym = max_asym.max((ab[i + j * nsa] - ba[j + i * nsb]).abs());
                    }
                }
                assert!(
                    max_asym < 1e-12,
                    "{op_sym} cross-l l=({la},{lb}) not symmetric: max |M_ab - M_ba^T| = {max_asym}"
                );
                // Sanity: the block is not all-zero (real integral at this separation).
                assert!(
                    ab.iter().any(|v| v.abs() > 1e-10),
                    "{op_sym} l=({la},{lb}) block unexpectedly all-zero"
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test XL-2: general-contraction (nctr>1) AND high-l cross-block together —
    // a generally-contracted d-shell (l=2, nctr=2) vs a generally-contracted
    // f-shell (l=3, nctr=2). This is the exact combination the executor flagged
    // (DI-02-11-CINTX-NCTR-HIGHL: l>=3 nctr>1). The full block must be the
    // transpose of the swapped block: <d_gc_i | f_gc_j> == <f_gc_j | d_gc_i>.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_general_contraction_high_l_cross_block_is_symmetric() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{
            Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell, ShellTuple,
        };
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let op = Resolver::descriptor_by_symbol("int1e_ovlp_sph")
            .expect("int1e_ovlp_sph in manifest")
            .id;

        // Generally-contracted shells: 2 shared primitives, 2 contraction columns.
        // coefficients[ip*nctr + ic] (row-major canonical).
        let d_exps: Arc<[f64]> = Arc::from(vec![1.4_f64, 0.45].into_boxed_slice());
        let d_co: Arc<[f64]> = Arc::from(vec![0.5_f64, 0.2, 0.3, 0.7].into_boxed_slice());
        let f_exps: Arc<[f64]> = Arc::from(vec![1.1_f64, 0.35].into_boxed_slice());
        let f_co: Arc<[f64]> = Arc::from(vec![0.6_f64, 0.1, 0.25, 0.8].into_boxed_slice());

        let block = |swapped: bool| -> Vec<f64> {
            let atoms: Arc<[Atom]> = Arc::from(
                vec![
                    Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap(),
                    Atom::try_new(1, [0.7, -0.4, 0.6], NuclearModel::Point, None, None).unwrap(),
                ]
                .into_boxed_slice(),
            );
            let d_gc = Arc::new(
                Shell::try_new(
                    0,
                    2,
                    2,
                    2,
                    0,
                    Representation::Spheric,
                    d_exps.clone(),
                    d_co.clone(),
                )
                .unwrap(),
            );
            let f_gc = Arc::new(
                Shell::try_new(
                    1,
                    3,
                    2,
                    2,
                    0,
                    Representation::Spheric,
                    f_exps.clone(),
                    f_co.clone(),
                )
                .unwrap(),
            );
            let all: Arc<[Arc<Shell>]> =
                Arc::from(vec![d_gc.clone(), f_gc.clone()].into_boxed_slice());
            let basis = BasisSet::try_new(atoms, all).unwrap();
            let (bra, ket) = if swapped {
                (f_gc.clone(), d_gc.clone())
            } else {
                (d_gc.clone(), f_gc.clone())
            };
            let n = bra.ao_per_shell() * ket.ao_per_shell();
            let shells = ShellTuple::try_from_iter([bra, ket]).unwrap();
            let opts = ExecutionOptions::default();
            let query = query_workspace(op, Representation::Spheric, &basis, shells.clone(), &opts)
                .unwrap();
            let mut plan =
                ExecutionPlan::new(op, Representation::Spheric, &basis, shells, &query).unwrap();
            plan.precision = PrecisionKind::F64;
            let spec = SpecializationKey::from_plan(&plan);
            let backend = ResolvedBackend::Cpu(resolve_cpu_client().unwrap());
            let mut staging = vec![0.0_f64; n];
            launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging).unwrap();
            staging
        };

        let nd = 2 * nsph(2); // d_gc ao_per_shell = 2*5 = 10
        let nf = 2 * nsph(3); // f_gc ao_per_shell = 2*7 = 14
        let ab = block(false); // ab[i + j*nd] = <d_gc_i | f_gc_j>
        let ba = block(true); // ba[j + i*nf] = <f_gc_j | d_gc_i>
        assert_eq!(ab.len(), nd * nf);
        let mut max_asym = 0.0_f64;
        for i in 0..nd {
            for j in 0..nf {
                max_asym = max_asym.max((ab[i + j * nd] - ba[j + i * nf]).abs());
            }
        }
        assert!(
            max_asym < 1e-12,
            "generally-contracted d(nctr=2)-f(nctr=2) cross-block not symmetric: max |Δ| = {max_asym}"
        );
        assert!(
            ab.iter().any(|v| v.abs() > 1e-10),
            "d_gc-f_gc block unexpectedly all-zero"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-01: ipovlp s-s component count
    // For an s,s shell pair (li=lj=0, nci=ncj=1), ipovlp must return 3 elements
    // (3 components × 1×1).  For a p,s pair (li=1,lj=0, nci=3, ncj=1), 9 elements.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipovlp_component_count() {
        let ai = 1.0_f64;
        let aj = 0.8_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4, 0.0, 0.0];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        // s,s: expect 3 elements
        let nmax_ss = 0u32 + 1;
        let g_ss = fill_g_tensor_overlap(&pd, ri, rj, nmax_ss, 0);
        let out_ss = contract_grad_1e_bra(&g_ss, 0, 0, nmax_ss, ai);
        assert_eq!(out_ss.len(), 3, "s-s ipovlp should return 3 components");

        // p,s: expect 9 elements
        let li_p = 1u8;
        let nmax_ps = (li_p as u32) + 0 + 1;
        let g_ps = fill_g_tensor_overlap(&pd, ri, rj, nmax_ps, 0);
        let out_ps = contract_grad_1e_bra(&g_ps, li_p, 0, nmax_ps, ai);
        assert_eq!(
            out_ps.len(),
            3 * ncart(li_p) * 1,
            "p-s ipovlp should return 9 components"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-02: ipovlp determinism
    // Two evaluations of the same shell pair must be bit-identical.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipovlp_determinism() {
        let ai = 1.23_f64;
        let aj = 0.57_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4, 0.5, 0.0];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let nmax = 1u32 + 1; // s-p: li=0, lj=1
        let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, 1);

        let out1 = contract_grad_1e_bra(&g, 0, 1, nmax, ai);
        let out2 = contract_grad_1e_bra(&g, 0, 1, nmax, ai);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ipovlp output not bit-identical on two calls"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-03: ipovlp z-component non-zero for z-displaced s-s pair
    // For two s-functions displaced along z, the overlap derivative ∂S/∂Az ≠ 0
    // and the x/y components should be zero (by symmetry of the z-displacement).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipovlp_z_component_z_displacement() {
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let ri = [0.0_f64; 3];
        let rj = [0.0, 0.0, 1.4]; // pure z displacement
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let nmax = 0u32 + 1; // s-s with +1 headroom
        let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, 0);
        let out = contract_grad_1e_bra(&g, 0, 0, nmax, ai);
        // out = [s_x, s_y, s_z] (3 components, each is a single value for s-s)
        let sx = out[0]; // comp 0 * block_len 1 + 0
        let sy = out[1];
        let sz = out[2];
        assert!(
            sx.abs() < 1e-14,
            "x-component should be ~0 for z-displacement, got {sx:.3e}"
        );
        assert!(
            sy.abs() < 1e-14,
            "y-component should be ~0 for z-displacement, got {sy:.3e}"
        );
        assert!(
            sz.abs() > 1e-6,
            "z-component should be nonzero for z-displacement, got {sz:.3e}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test helper: build a forced-Spinor gradient plan for the given int1e_ip*_sph
    // operator symbol, on a 2-atom H2-like fixture with two `l` shells (kappa=0).
    // The sph OperatorId is used to query a valid workspace; we then force
    // Representation::Spinor on the plan to exercise the spinor gradient arm.
    // Returns (plan, basis-backed ShellTuple already inside the plan).
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    fn run_forced_spinor_grad(
        op_symbol: &str,
        l: u8,
        nctr: u16,
        rinv_orig: Option<[f64; 3]>,
        staging_len: usize,
    ) -> Result<Vec<f64>, cintxRsError> {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell};
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let op_sph = Resolver::descriptor_by_symbol(op_symbol)
            .unwrap_or_else(|e| panic!("{op_symbol} must be in manifest: {e:?}"))
            .id;

        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom2 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom, atom2].into_boxed_slice());

        let nctr_usize = nctr as usize;
        // 1 primitive, nctr contractions (coeff length = nprim * nctr).
        let exps: Vec<f64> = vec![1.0_f64];
        let coeffs: Vec<f64> = vec![1.0_f64; nctr_usize];

        let shell_a = Arc::new(
            Shell::try_new(
                0,
                l,
                1,
                nctr,
                0,
                Representation::Spheric,
                Arc::from(exps.clone().into_boxed_slice()),
                Arc::from(coeffs.clone().into_boxed_slice()),
            )
            .unwrap(),
        );
        let shell_b = Arc::new(
            Shell::try_new(
                1,
                l,
                1,
                nctr,
                0,
                Representation::Spheric,
                Arc::from(exps.into_boxed_slice()),
                Arc::from(coeffs.into_boxed_slice()),
            )
            .unwrap(),
        );
        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();

        let opts = ExecutionOptions::default();
        let q = query_workspace(op_sph, Representation::Spheric, &basis, shells.clone(), &opts)
            .unwrap();
        let mut plan =
            ExecutionPlan::new(op_sph, Representation::Spheric, &basis, shells, &q).unwrap();
        // Force spinor representation on the plan to exercise the spinor gradient arm.
        plan.representation = Representation::Spinor;
        plan.precision = cintx_core::PrecisionKind::F64;
        if let Some(o) = rinv_orig {
            plan.operator_env_params.rinv_orig = Some(o);
        }

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; staging_len];
        launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging)?;
        Ok(staging)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-04: ipovlp spinor (nctr=1) evaluates and writes a 3-component
    // interleaved-complex staging of length 3 * di * dj * 2 (di=dj=spinor_len(0,0)=2).
    // Replaces the prior UnsupportedApi rejection (Risk R5 / D-03), now implemented.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_ipovlp_spinor_grad_evaluates() {
        // di = dj = spinor_len(0, 0) = 2; spinor_block = 2*2*2 = 8; total = 24.
        let result = run_forced_spinor_grad("int1e_ipovlp_sph", 0, 1, None, 24);
        let staging = result.expect("spinor ipovlp gradient (nctr=1) should evaluate");
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor ipovlp gradient staging is all-zero"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-04b: ipovlp spinor gradient with general contraction (nctr>1)
    // still returns UnsupportedApi (guard preserved).
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_ipovlp_spinor_grad_nctr_gt1_returns_unsupported() {
        // Oversize staging so the nctr>1 guard (not a bounds issue) is what fires.
        let result = run_forced_spinor_grad("int1e_ipovlp_sph", 0, 2, None, 256);
        assert!(
            matches!(result, Err(cintxRsError::UnsupportedApi { .. })),
            "spinor ipovlp gradient with nctr>1 should return UnsupportedApi, got: {result:?}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-05: ipkin s-s component count (3 elements)
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipkin_component_count() {
        let ai = 1.0_f64;
        let aj = 0.8_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4, 0.0, 0.0];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        // s-s: nmax = li + lj + 3 = 3
        let nmax = 3u32;
        let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, 0u32 + 2); // lj_ext = lj+2 = 2
        let out = contract_ipkin(&g, 0, 0, nmax, ai, aj);
        assert_eq!(out.len(), 3, "s-s ipkin should return 3 components");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-06: ipkin determinism
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipkin_determinism() {
        let ai = 1.23_f64;
        let aj = 0.57_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4, 0.5, 0.0];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        let li = 0u8;
        let lj = 0u8;
        let nmax = (li as u32) + (lj as u32) + 3;
        let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32 + 2);

        let out1 = contract_ipkin(&g, li, lj, nmax, ai, aj);
        let out2 = contract_ipkin(&g, li, lj, nmax, ai, aj);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ipkin output not bit-identical on two calls"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-07: ipnuc/iprinv component count
    // For an s,s pair (li=lj=0, nci=ncj=1) contract_nuclear_grad returns 3 elements;
    // for a p,s pair (li=1, nci=3) returns 9. Single origin reused for both ipnuc
    // (one nucleus) and iprinv (one origin) — same helper, same shape.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipnuc_component_count() {
        let ai = 1.0_f64;
        let aj = 0.8_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4, 0.0, 0.0];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        // s,s: 3 elements (one nucleus, charge -1)
        let origins_ss = [([0.0_f64, 0.0, 0.7], -1.0_f64)];
        let out_ss = contract_nuclear_grad(&pd, ri, rj, 0, 0, ai, &origins_ss);
        assert_eq!(out_ss.len(), 3, "s-s ipnuc should return 3 components");

        // p,s: 9 elements
        let out_ps = contract_nuclear_grad(&pd, ri, rj, 1, 0, ai, &origins_ss);
        assert_eq!(
            out_ps.len(),
            3 * ncart(1) * 1,
            "p-s ipnuc should return 9 components"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-08: ipnuc determinism — ordered atom-loop reduction (D-10)
    // Repeated evaluation with a multi-nucleus origin list is bit-identical.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_ipnuc_determinism() {
        let ai = 1.23_f64;
        let aj = 0.57_f64;
        let ri = [0.0_f64; 3];
        let rj = [1.4, 0.5, 0.0];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
        // Three nuclei (sum over all, charge -Z_C), low→high index order.
        let origins = [
            ([0.0_f64, 0.0, 0.0], -8.0_f64),
            ([0.0_f64, 1.4307, 1.1078], -1.0_f64),
            ([0.0_f64, -1.4307, 1.1078], -1.0_f64),
        ];
        let out1 = contract_nuclear_grad(&pd, ri, rj, 0, 1, ai, &origins);
        let out2 = contract_nuclear_grad(&pd, ri, rj, 0, 1, ai, &origins);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ipnuc output not bit-identical on two calls"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-09: iprinv single-origin sensitivity (the key proof)
    // contract_nuclear_grad with one origin must consume that origin: two DIFFERENT
    // origins must produce DIFFERENT output (proves iprinv is NOT origin-blind).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn test_iprinv_origin_sensitivity() {
        let ai = 1.0_f64;
        let aj = 1.0_f64;
        let ri = [0.0_f64; 3];
        let rj = [0.0, 0.0, 1.4];
        let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);

        // iprinv uses factor +1.0, single origin.
        let out_a = contract_nuclear_grad(&pd, ri, rj, 0, 0, ai, &[([0.0_f64, 0.0, 1.4], 1.0)]);
        let out_b = contract_nuclear_grad(&pd, ri, rj, 0, 0, ai, &[([0.7_f64, 0.3, 0.2], 1.0)]);

        // Nonzero output for both.
        assert!(
            out_a.iter().any(|v| v.abs() > 1e-12),
            "iprinv output should be nonzero"
        );
        // Different origins must produce a different result.
        let any_diff = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            any_diff,
            "different rinv origins must produce different output (origin consumed)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-10: iprinv with None origin returns a typed error (no panic)
    // Dispatch iprinv through launch_one_electron_typed with rinv_orig == None.
    // The kernel must return InvalidEnvParam, never panic (T-21-04-01 defensive gate).
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_iprinv_none_origin_returns_typed_error() {
        use crate::backend::{ResolvedBackend, cpu_backend::resolve_cpu_client};
        use crate::specialization::SpecializationKey;
        use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell};
        use cintx_ops::resolver::Resolver;
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let op_sph = Resolver::descriptor_by_symbol("int1e_iprinv_sph")
            .expect("int1e_iprinv_sph must be in manifest")
            .id;

        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom2 = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms: Arc<[Atom]> = Arc::from(vec![atom, atom2].into_boxed_slice());
        let shell_a = Arc::new(
            Shell::try_new(
                0,
                0,
                1,
                1,
                0,
                Representation::Spheric,
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
                Representation::Spheric,
                Arc::from(vec![1.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
            .unwrap(),
        );
        let all_shells: Arc<[Arc<Shell>]> =
            Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice());
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells = cintx_core::ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();

        let opts = ExecutionOptions::default();
        let q = query_workspace(
            op_sph,
            Representation::Spheric,
            &basis,
            shells.clone(),
            &opts,
        )
        .unwrap();
        let mut plan =
            ExecutionPlan::new(op_sph, Representation::Spheric, &basis, shells, &q).unwrap();
        // Leave plan.operator_env_params.rinv_orig as None (the defensive case).
        plan.precision = cintx_core::PrecisionKind::F64;
        assert!(
            plan.operator_env_params.rinv_orig.is_none(),
            "test precondition: origin must be None"
        );

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);
        let mut staging = vec![0.0_f64; 3];

        let result = launch_one_electron_typed::<f64>(&backend, &plan, &spec, &mut staging);
        assert!(
            matches!(
                result,
                Err(cintxRsError::InvalidEnvParam {
                    param: "PTR_RINV_ORIG",
                    ..
                })
            ),
            "iprinv with None origin should return InvalidEnvParam, got: {:?}",
            result
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-04c: ipkin spinor (nctr=1) evaluates and writes a 3-component
    // interleaved-complex staging of length 3 * di * dj * 2 (di=dj=2 for s/s).
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_ipkin_spinor_grad_evaluates() {
        let result = run_forced_spinor_grad("int1e_ipkin_sph", 0, 1, None, 24);
        let staging = result.expect("spinor ipkin gradient (nctr=1) should evaluate");
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor ipkin gradient staging is all-zero"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-11: ipnuc spinor (nctr=1) evaluates (replaces the old
    // UnsupportedApi rejection — Risk R5 / D-03 closed).
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_ipnuc_spinor_grad_evaluates() {
        let result = run_forced_spinor_grad("int1e_ipnuc_sph", 0, 1, None, 24);
        let staging = result.expect("spinor ipnuc gradient (nctr=1) should evaluate");
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor ipnuc gradient staging is all-zero"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test GRAD-12: iprinv spinor (nctr=1) evaluates with a valid rinv origin.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_iprinv_spinor_grad_evaluates() {
        // iprinv needs a resolved rinv origin (env[PTR_RINV_ORIG]).
        let result =
            run_forced_spinor_grad("int1e_iprinv_sph", 0, 1, Some([0.0, 0.0, 0.0]), 24);
        let staging = result.expect("spinor iprinv gradient (nctr=1) should evaluate");
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor iprinv gradient staging is all-zero"
        );
    }
}
