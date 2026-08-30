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
use crate::math::obara_saika::hrr_step_host;
// `vrr_step_host`'s only remaining caller is `fill_g_tensor_overlap`, which is
// itself `#[cfg(test)]`: the live scalar and gradient paths build the G tensor
// in-kernel. `compute_pdata_host` is likewise reached only through its full path
// (one in-kernel call site) and by the test module's own import.
#[cfg(test)]
use crate::math::obara_saika::vrr_step_host;
use crate::math::rys::rys_roots_host;
use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
use crate::math::rys_wheeler::{
    EXT_TABLES_LEN, ext_rys_out_slots, ext_rys_slots, rys_roots_ext_dev,
};
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_1e, ncart, nsph};
use crate::transform::c2spinor::{
    cart_to_spinor_sf_2d, cart_to_spinor_sf_derivative_2d, cart_to_spinor_si_2d, spinor_len,
};
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
// Verbatim libcint literal, not `std::f64::consts::FRAC_PI_4`: result compatibility
// with upstream is decided by the exact bits this file feeds the Rys kernels, so
// the constant is transcribed from `rys_roots.c` rather than recomputed.
#[allow(clippy::approx_constant)]
const PIE4: f64 = 0.78539816339744827900_f64;

/// Maximum `nroots` the device Rys kernels (`rys_root1..5`) can evaluate for the
/// on-device nuclear-attraction arm. `nrys = (li + lj) / 2 + 1`, so this covers
/// `li + lj <= 8`. Same `MAX_DEVICE_NROOTS` guard the 2c2e device kernel uses.
const MAX_DEVICE_NROOTS: usize = 5;

/// Host Rys nroots ceiling for the 1e nuclear/rinv path (FND-02). The host
/// Wheeler/Jacobi `rys_roots_host` serves nroots 6..12; above 12 the vendor
/// reference itself caps (quadmath disabled), so the HESS-04 deriv34 families
/// fail closed (typed `UnsupportedApi`, no partial write) beyond it.
const HOST_RYS_NROOTS_CEILING_1E: usize = 12;

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

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// Batched scalar 1e kernel — one shell pair per work slot (Phase 35).
///
/// Faithful correctness-first port of the host scalar pipeline
/// (`fill_g_tensor_overlap` + `contract_overlap` / `contract_kinetic` /
/// `contract_nuclear`), evaluating a whole **launch class** in one dispatch:
/// every pair in the list shares `(li, lj)` and therefore the G-tensor shape,
/// the Rys order and the HRR branch. What varies per pair is only the shell
/// data, read through a flattened basis plus an index table:
///
/// - `exps` / `coeffs` — every shell's primitives concatenated;
/// - `centers` — 3 floats per shell;
/// - `shell_meta` — 4 `u32` per shell: `[exp_off, coeff_off, nprim, nctr]`;
/// - `pairs` — 3 `u32` per pair: `[si, sj, out_off]`.
///
/// For each pair it iterates the primitive pairs (pi,pj) and accumulates one
/// `nci*ncj` Cartesian block per contraction pair (ci,cj) into `cart_out`,
/// contraction-major / bra-fastest exactly as the host scalar path does:
/// block base `out_off + (ci*nctr_j + cj) * (nci*ncj)`, element
/// `out[cj_idx*nci + ci_idx]`.
///
/// `g` is a per-slot slab of `g_stride >= 3 * g_per_axis`; the Rys roots and
/// weights are kernel-local arrays, since every read of them sits inside the
/// same `lane == 0` region that writes them.
///
/// The slot/lane decomposition and the `per_unit` comptime flag are the same
/// ones [`crate::kernels::two_electron`] documents: `per_unit == 1` gives each
/// unit a whole pair with no reachable barrier (the CubeCL CPU shape),
/// `per_unit == 0` gives each cube a pair with the cube cooperating on its
/// contraction (the GPU shape).
///
/// Source: libcint-master/src/g1e.c `CINTg1e_ovlp` / `CINTg1e_nuc`,
///         autocode/intor1.c `CINTgout1e_int1e_kin`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_scalar_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    atom_coords: &Array<F>,
    atom_charges: &Array<F>,
    rys_tab: &Array<f64>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    prim_tol: F,
    sqrtpi: F,
    pi_const: F,
    natm: u32,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] op_kind: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    // Slot / lane decomposition — see the doc comment above and the identical
    // block in `two_electron.rs` for why this is arithmetic on comptime-folded
    // flags rather than a `comptime!` if/else.
    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;
    let lanes = (cube_dim - 1u32) * coop + 1u32;

    // Rys roots/weights are written and read entirely inside `lane == 0`
    // regions, so they are per-unit private storage rather than buffers. The
    // extent follows `nroots`: five for the polynomial-fit kernels, exactly
    // `nroots` once the inline extended entry (task 33-01) serves the class.
    let mut urys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    let mut wrys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    // The extended entry is f64-only, so it lands in its own pair and is cast
    // into `urys`/`wrys`. Both collapse to one element when the arm is absent.
    let mut uext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));
    let mut wext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));

    let gbase = slot * g_stride;

    // Blocked walk under `per_unit == 1`, grid-stride otherwise — neighbouring
    // pairs write neighbouring `cart_out` blocks, so an interleaved assignment
    // would put every unit's accumulation on the same cache lines.
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
        let sj = pairs[(prow + 1u32) as usize];
        let out_off = pairs[(prow + 2u32) as usize];

        // ── Per-class shape (Task 35-M2) ──────────────────────────────────
        //
        // `op_kind` and `nroots` are this kernel's comptime parameters, and
        // `op_kind` is fixed by the caller's operator, so one dispatch carries
        // every `(li,lj)` class of the same Rys order. The G slab is sized to
        // the widest class in the dispatch and each class indexes only the
        // leading `3 * g_per_axis` it owns, keeping the merge bit-identical.
        let cls = pairs[(prow + 3u32) as usize];
        let srow = cls * comptime!(ONE_E_SHAPE_STRIDE as u32);
        let li = class_shape[srow as usize];
        let lj = class_shape[(srow + 1u32) as usize];

        let nci = (li + 1u32) * (li + 2u32) / 2u32;
        let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
        let block_len = nci * ncj;

        // ── G-tensor sizing (mirrors the host scalar path) ────────────────
        let mut nmax = li + lj;
        let mut lj_ext = lj;
        if comptime!(op_kind == 1u32) {
            nmax = li + lj + 2u32;
            lj_ext = lj + 2u32;
        }
        // Stride between consecutive j-levels within an axis block.
        let dj = nmax + 1u32;
        let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
        let gx = gbase;
        let gy = gbase + g_per_axis;
        let gz = gbase + 2u32 * g_per_axis;

        let mi = si * 4u32;
        let eoff_i = shell_meta[mi as usize];
        let coff_i = shell_meta[(mi + 1u32) as usize];
        let nprim_i = shell_meta[(mi + 2u32) as usize];
        let nctr_i = shell_meta[(mi + 3u32) as usize];
        let mj = sj * 4u32;
        let eoff_j = shell_meta[mj as usize];
        let coff_j = shell_meta[(mj + 1u32) as usize];
        let nprim_j = shell_meta[(mj + 2u32) as usize];
        let nctr_j = shell_meta[(mj + 3u32) as usize];

        let ci3 = si * 3u32;
        let rix = centers[ci3 as usize];
        let riy = centers[(ci3 + 1u32) as usize];
        let riz = centers[(ci3 + 2u32) as usize];
        let cj3 = sj * 3u32;
        let rjx = centers[cj3 as usize];
        let rjy = centers[(cj3 + 1u32) as usize];
        let rjz = centers[(cj3 + 2u32) as usize];

        let out_total = nctr_i * nctr_j * block_len;

        // Zero this pair's accumulation block across the slot's lanes.
        let mut oi = lane;
        while oi < out_total {
            cart_out[(out_off + oi) as usize] = F::new(0.0_f32);
            oi += lanes;
        }
        if comptime!(per_unit == 0u32) {
            sync_cube();
        }

        let is_uncontracted_1e = (nctr_i == 1u32) && (nctr_j == 1u32);

        // ── Primitive loop ───────────────────────────────────────────────────
        let mut pi = 0u32;
        while pi < nprim_i {
            let ai = exps[(eoff_i + pi) as usize];
            let mut pj = 0u32;
            while pj < nprim_j {
                let aj = exps[(eoff_j + pj) as usize];

                // Pair data, computed in-kernel in F (norm_i = norm_j = 1.0).
                let zeta = ai + aj;
                let aij2 = F::new(0.5_f32) / zeta;
                let rirjx = rix - rjx;
                let rirjy = riy - rjy;
                let rirjz = riz - rjz;
                let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                let fac = F::exp(-ai * aj / zeta * rr);
                let px = (ai * rix + aj * rjx) / zeta;
                let py = (ai * riy + aj * rjy) / zeta;
                let pz = (ai * riz + aj * rjz) / zeta;

                let prim_weight_1e = if is_uncontracted_1e {
                    coeffs[(coff_i + pi) as usize] * coeffs[(coff_j + pj) as usize]
                } else {
                    F::new(0.0_f32)
                };

                if comptime!(op_kind == 0u32 || op_kind == 1u32) {
                    if lane == 0u32 {
                        if comptime!(op_kind == 0u32) {
                            // ===== OVERLAP G-tensor (fixed-center VRR + HRR) =====
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
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
                        } else {
                            // ===== KINETIC: overlap G-tensor with lj+2 HRR levels =====
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
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
                    }
                    if comptime!(per_unit == 0u32) {
                        sync_cube();
                    }

                    // ── Contract into every (ci,cj) contraction block cooperatively across lanes ────────────
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

                                    let elem_idx = cj_idx * nci + ci_idx;
                                    if ((elem_idx as u32) % lanes) == lane {
                                        let mut val = F::new(0.0_f32);
                                        if comptime!(op_kind == 0u32) {
                                            // Overlap: vx*vy*vz from shared g.
                                            let vx = g[(gx + jx * dj + ix) as usize];
                                            let vy = g[(gy + jy * dj + iy) as usize];
                                            let vz = g[(gz + jz * dj + iz) as usize];
                                            val = vx * vy * vz;
                                        } else {
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
                                            val = F::new(-0.5_f32)
                                                * (g3x * vy0 * vz0
                                                    + vx0 * g3y * vz0
                                                    + vx0 * vy0 * g3z);
                                        }

                                        if is_uncontracted_1e {
                                            cart_out[(out_off + cj_idx * nci + ci_idx) as usize] +=
                                                prim_weight_1e * val;
                                        } else {
                                            let mut ci = 0u32;
                                            while ci < nctr_i {
                                                let coeff_i_val =
                                                    coeffs[(coff_i + pi * nctr_i + ci) as usize];
                                                let mut cj = 0u32;
                                                while cj < nctr_j {
                                                    let coeff_j_val = coeffs
                                                        [(coff_j + pj * nctr_j + cj) as usize];
                                                    let base = (ci * nctr_j + cj) * block_len;
                                                    cart_out[(out_off
                                                        + base
                                                        + cj_idx * nci
                                                        + ci_idx)
                                                        as usize] +=
                                                        coeff_i_val * coeff_j_val * val;
                                                    cj += 1u32;
                                                }
                                                ci += 1u32;
                                            }
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
                    if comptime!(per_unit == 0u32) {
                        sync_cube();
                    }
                } else {
                    // Nuclear: sum over atoms and Rys roots FIRST, building G-tensor once per (atom, root)
                    let mut atom = 0u32;
                    while atom < natm {
                        let z_c = atom_charges[atom as usize];
                        let rcx = atom_coords[(atom * 3u32) as usize];
                        let rcy = atom_coords[(atom * 3u32 + 1u32) as usize];
                        let rcz = atom_coords[(atom * 3u32 + 2u32) as usize];

                        // crij = C - P
                        let crijx = rcx - px;
                        let crijy = rcy - py;
                        let crijz = rcz - pz;
                        let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                        // fac1 = 2*PI*(-Z_C)*fac/zeta
                        let neg_z = F::new(0.0_f32) - z_c;
                        let fac1 = F::new(2.0_f32) * pi_const * neg_z * fac / zeta;

                        // Primitive screening for the nuclear arm (Task 34-D2).
                        //
                        // `fac1` is the scalar the whole (primitive pair, atom)
                        // contribution is built from — `gz` starts at
                        // `w_n * fac1` and `gx`/`gy` at 1 — so it plays the same
                        // role the 2e/3c2e `fac1` does. Two differences matter:
                        //
                        // - it is **negative** for a nuclear attraction
                        //   (`-Z_C`), so the test is on its magnitude;
                        // - the branch is on values uniform across the cube
                        //   (same shell pair, same atom for every lane), so the
                        //   `sync_cube` barriers inside it are still reached by
                        //   every lane or by none.
                        //
                        // At `prim_tol == 0` (the default) the only terms
                        // dropped are those whose `fac1` underflowed to exactly
                        // zero, which contribute exactly zero — the
                        // tolerance-zero identity gate.
                        if F::abs(fac1) > prim_tol {
                            if lane == 0u32 {
                                // Rys roots/weights (comptime nroots).
                                if comptime!(nroots == 1u32) {
                                    rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                                } else if comptime!(nroots == 2u32) {
                                    rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                                } else if comptime!(nroots == 3u32) {
                                    rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                                } else if comptime!(nroots == 4u32) {
                                    rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                                } else if comptime!(nroots == 5u32) {
                                    rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                                } else {
                                    // nroots 6..=12: the inline Wheeler/Jacobi
                                    // entry (task 33-01), reachable only once
                                    // `device_nroots_ceiling` was raised for
                                    // this family.
                                    rys_roots_ext_dev(
                                        rys_tab,
                                        f64::cast_from(x_boys),
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
                            }
                            if comptime!(per_unit == 0u32) {
                                sync_cube();
                            }

                            #[unroll]
                            for irys in 0..nroots {
                                if lane == 0u32 {
                                    let u_n = urys[irys as usize];
                                    let w_n = wrys[irys as usize];
                                    let tau = u_n / (F::new(1.0_f32) + u_n);
                                    let rt = aij2 * (F::new(1.0_f32) - tau);

                                    let c00x = (px - rix) + tau * crijx;
                                    let c00y = (py - riy) + tau * crijy;
                                    let c00z = (pz - riz) + tau * crijz;

                                    // Base case: gx=1, gy=1, gz=fac1*w_n
                                    g[gx as usize] = F::new(1.0_f32);
                                    g[gy as usize] = F::new(1.0_f32);
                                    g[gz as usize] = fac1 * w_n;

                                    one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                                    one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                                    one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                                    if lj >= 1u32 {
                                        one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj);
                                        one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj);
                                        one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj);
                                    }
                                }
                                if comptime!(per_unit == 0u32) {
                                    sync_cube();
                                }

                                // Accumulate over Cartesian triples and contractions cooperatively
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

                                                let elem_idx = cj_idx * nci + ci_idx;
                                                if ((elem_idx as u32) % lanes) == lane {
                                                    let vx = g[(gx + jx * dj + ix) as usize];
                                                    let vy = g[(gy + jy * dj + iy) as usize];
                                                    let vz = g[(gz + jz * dj + iz) as usize];
                                                    let val = vx * vy * vz;

                                                    let mut ci = 0u32;
                                                    while ci < nctr_i {
                                                        let coeff_i_val = coeffs
                                                            [(coff_i + pi * nctr_i + ci) as usize];
                                                        let mut cj = 0u32;
                                                        while cj < nctr_j {
                                                            let coeff_j_val =
                                                                coeffs[(coff_j + pj * nctr_j + cj)
                                                                    as usize];
                                                            let base =
                                                                (ci * nctr_j + cj) * block_len;
                                                            cart_out[(out_off
                                                                + base
                                                                + cj_idx * nci
                                                                + ci_idx)
                                                                as usize] +=
                                                                coeff_i_val * coeff_j_val * val;
                                                            cj += 1u32;
                                                        }
                                                        ci += 1u32;
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
                                if comptime!(per_unit == 0u32) {
                                    sync_cube();
                                }
                            }
                        }

                        atom += 1u32;
                    }
                }

                pj += 1u32;
            }
            pi += 1u32;
        }

        qi += qi_step;
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
    let mut lo = F::new(0.0_f32);
    if jx >= 2u32 {
        lo = g[(base + nx - 2u32 * dj) as usize];
    }
    F::new(4.0_f32) * aj * aj * g_hi
        - F::new(2.0_f32) * aj * (F::new(2.0_f32) * jxf + F::new(1.0_f32)) * v0
        + jxf * (jxf - F::new(1.0_f32)) * lo
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
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_grad_bra_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    d2g0: &mut Array<F>,
    d2g1: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] op_kind: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All four scratch slabs share a stride and a slot, because the
        // contraction reads them at identical relative offsets.
        let gbase = slot * g_stride;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            // ── Per-class shape (Task 35-D) ───────────────────────────────
            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

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
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 3u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            // Zero the full accumulation buffer.
            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // ── Primitive loop ───────────────────────────────────────────────────
            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    // Pair data, computed in-kernel in F (norm_i = norm_j = 1.0).
                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // ── Build OVERLAP base G-tensor in `g` (fixed-center VRR + HRR) ──
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
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
                    let mut g1i = gbase;
                    while g1i < gbase + total_g {
                        g1[g1i as usize] = F::new(0.0_f32);
                        g1i += 1u32;
                    }
                    let ai2 = F::new(-2.0_f32) * ai;
                    let mut axisn = 0u32;
                    while axisn < 3u32 {
                        let off = gbase + axisn * g_per_axis;
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
                            let off = gbase + d2axis * g_per_axis;
                            let mut jd = 0u32;
                            while jd <= lj {
                                let jf = F::cast_from(jd);
                                let mut id = 0u32;
                                while id <= li {
                                    let nx = jd * dj + id;
                                    // lo term only valid for j>=2.
                                    let mut g0_lo = F::new(0.0_f32);
                                    let mut g1_lo = F::new(0.0_f32);
                                    if jd >= 2u32 {
                                        g0_lo = g[(off + nx - 2u32 * dj) as usize];
                                        g1_lo = g1[(off + nx - 2u32 * dj) as usize];
                                    }
                                    let coef_mid = F::new(2.0_f32)
                                        * aj
                                        * (F::new(2.0_f32) * jf + F::new(1.0_f32));
                                    let coef_hi = F::new(4.0_f32) * aj * aj;
                                    let coef_lo = jf * (jf - F::new(1.0_f32));
                                    d2g0[(off + nx) as usize] = coef_hi
                                        * g[(off + nx + 2u32 * dj) as usize]
                                        - coef_mid * g[(off + nx) as usize]
                                        + coef_lo * g0_lo;
                                    d2g1[(off + nx) as usize] = coef_hi
                                        * g1[(off + nx + 2u32 * dj) as usize]
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
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                                s0 = F::new(-0.5_f32)
                                                    * (d2x1 * g0y * g0z
                                                        + g1x * d2y0 * g0z
                                                        + g1x * g0y * d2z0);
                                                s1 = F::new(-0.5_f32)
                                                    * (d2x0 * g1y * g0z
                                                        + g0x * d2y1 * g0z
                                                        + g0x * g1y * d2z0);
                                                s2 = F::new(-0.5_f32)
                                                    * (d2x0 * g0y * g1z
                                                        + g0x * d2y0 * g1z
                                                        + g0x * g0y * d2z1);
                                            }

                                            let elem = cj_idx * nci + ci_idx;
                                            cart_out[(base + elem) as usize] += weight * s0;
                                            cart_out[(base + block_len + elem) as usize] +=
                                                weight * s1;
                                            cart_out[(base + 2u32 * block_len + elem) as usize] +=
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

            qi += qi_step;
        }
    }
}

/// `u32` shape scalars per class row of the 1e **derivative** shape table.
///
/// The same `li, lj` the scalar 1e table carries: every derivative kernel
/// rederives its own G-tensor extents in-kernel from those two plus its
/// comptime `op_kind`, because the headroom differs per family and duplicating
/// that arithmetic on the host would put the two out of step.
pub(crate) const ONE_E_DERIV_SHAPE_STRIDE: usize = ONE_E_SHAPE_STRIDE;

/// One dispatch of a 1e derivative family: a set of shell pairs sharing the
/// kernel's comptime parameters (Task 35-D).
///
/// Unlike the Rys families there is usually nothing left to key on — a
/// derivative kernel's `op_kind` is fixed by the caller's operator — so a whole
/// work list normally collapses to a **single** dispatch. `nroots` is carried
/// for the nuclear arm, which does specialize on it.
#[derive(Clone, Debug, Default)]
pub struct OneEDerivLaunchGroup {
    /// Rys order, for the families that specialize on it; 1 otherwise.
    pub nroots: u32,
    /// [`ONE_E_DERIV_SHAPE_STRIDE`] `u32` per merged class: `li, lj`.
    pub class_shape: Vec<u32>,
    /// `[si, sj, out_off, class]` per pair.
    pub pairs: Vec<u32>,
    /// Total Cartesian output elements across this group's pairs.
    pub out_len: usize,
    /// Widest per-slot `g_per_axis` in the group.
    pub max_g_per_axis: usize,
}

impl OneEDerivLaunchGroup {
    /// An empty group of Rys order `nroots` (pass 1 for the non-Rys families).
    #[must_use]
    pub fn new(nroots: u32) -> Self {
        Self {
            nroots,
            ..Self::default()
        }
    }

    /// Append a class and return the index its pair rows carry.
    ///
    /// `g_per_axis` is the caller's, because only the caller knows its family's
    /// headroom — the same reason the kernel rederives it rather than reading it
    /// from the shape row.
    pub fn push_class(&mut self, li: u32, lj: u32, g_per_axis: usize) -> u32 {
        let index = (self.class_shape.len() / ONE_E_DERIV_SHAPE_STRIDE) as u32;
        self.class_shape.extend_from_slice(&[li, lj]);
        self.max_g_per_axis = self.max_g_per_axis.max(g_per_axis);
        index
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

    /// Number of angular-momentum classes merged into this dispatch.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_shape.len() / ONE_E_DERIV_SHAPE_STRIDE
    }

    /// Bytes this group's pair and class tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        (self.pairs.len() + self.class_shape.len()) * std::mem::size_of::<u32>()
    }
}

/// Evaluate every launch group of a batched `int1e_ipovlp` / `int1e_ipkin` run.
///
/// `op_kind` selects the family (0 = ipovlp, 1 = ipkin) and is the kernel's only
/// comptime parameter, so a whole work list is one dispatch.
fn run_1e_grad_bra_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Four scratch slabs, all `3 * g_per_axis`, sharing a stride and a slot:
        // the contraction reads them at identical relative offsets.
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let d2g0_h = client.empty(g_len * std::mem::size_of::<f64>());
        let d2g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_pairs`, by the class index in each pair row, by the per-shell
        // `nprim`/`nctr` read from `shell_meta`, and by the per-class extents.
        macro_rules! launch_with {
            ($op:expr) => {
                unsafe {
                    one_electron_grad_bra_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(d2g0_h.clone(), g_len),
                        ArrayArg::from_raw_parts(d2g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        SQRTPI,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $op,
                        per_unit,
                    );
                }
            };
        }

        if op_kind == 0 {
            launch_with!(0u32);
        } else {
            launch_with!(1u32);
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipovlp` / `int1e_ipkin` run.
fn dispatch_1e_grad_bra_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_grad_bra_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, op_kind)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_grad_bra_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, op_kind)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_grad_bra_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, op_kind)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_grad_bra_batches::<cubecl_hip::HipRuntime>(client, basis, groups, op_kind)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_grad_bra_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, op_kind)
        }
    }
}

/// `g_per_axis` for one `int1e_ipovlp` / `int1e_ipkin` class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn one_e_grad_bra_g_per_axis(op_kind: u32, li: usize, lj: usize) -> usize {
    let (nmax, lj_ext) = if op_kind == 1 {
        (li + lj + 3, lj + 2)
    } else {
        (li + lj + 1, lj)
    };
    (nmax + 1) * (lj_ext + 1)
}

/// One shell pair through the batched `int1e_ipovlp` / `int1e_ipkin` path.
///
/// The per-tuple compatibility API evaluates exactly one pair and must keep
/// doing so, but it goes through the *same* kernel as a wide batch — a one-pair
/// group. That is what makes every existing parity test a test of the batched
/// kernel too (Task 35-D's acceptance bar).
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
    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (exps, coeffs, center, nprim, nctr) in [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_j, coeff_j, rj, nprim_j, nctr_j),
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

    let (li_u, lj_u) = (li as usize, lj as usize);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = nctr_i as usize * nctr_j as usize * 3 * nci * ncj;

    let mut group = OneEDerivLaunchGroup::new(1);
    let class_index = group.push_class(li, lj, one_e_grad_bra_g_per_axis(op_kind, li_u, lj_u));
    group.pairs.extend_from_slice(&[0, 1, 0, class_index]);
    group.out_len = out_len;

    let handles = upload_flat_basis(backend, &basis);
    dispatch_1e_grad_bra_batches(backend, &handles, std::slice::from_ref(&group), op_kind)
        .pop()
        .unwrap_or_default()
}

/// Build the one-pair launch group and basis residency a per-tuple 1e
/// derivative call needs (Task 35-D, waves 3-5).
///
/// Every converted derivative family keeps its per-tuple compatibility entry
/// point, and every one of them needs the same four things: the two shells
/// flattened into the batch basis form, a group of Rys order `nroots` holding
/// one class, one pair row `[0, 1, 0, class]`, and an `out_len` of
/// `rank * nctr_i * nctr_j * nci * ncj`. Doing it once here is what keeps the
/// conversions mechanical — the alternative is the same twenty lines in each of
/// twenty-seven launchers, differing only in `rank`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn one_e_deriv_single_pair_group(
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
    rank: usize,
    g_per_axis: usize,
    nroots: u32,
) -> (
    OneEDerivLaunchGroup,
    crate::kernels::two_electron::TwoEBasisHandles,
) {
    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (exps, coeffs, center, nprim, nctr) in [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_j, coeff_j, rj, nprim_j, nctr_j),
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

    let (li_u, lj_u) = (li as usize, lj as usize);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;

    let mut group = OneEDerivLaunchGroup::new(nroots);
    let class_index = group.push_class(li, lj, g_per_axis);
    group.pairs.extend_from_slice(&[0, 1, 0, class_index]);
    group.out_len = nctr_i as usize * nctr_j as usize * rank * nci * ncj;

    let handles = upload_flat_basis(backend, &basis);
    (group, handles)
}

/// Upload a flattened basis through whichever backend arm `backend` is.
pub(crate) fn upload_flat_basis(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEFlatBasis,
) -> crate::kernels::two_electron::TwoEBasisHandles {
    use crate::kernels::two_electron::upload_2e_basis;
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => upload_2e_basis::<cubecl::cpu::CpuRuntime>(client, basis),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            upload_2e_basis::<cubecl_wgpu::WgpuRuntime>(client, basis)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => upload_2e_basis::<cubecl_cuda::CudaRuntime>(client, basis),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => upload_2e_basis::<cubecl_hip::HipRuntime>(client, basis),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            upload_2e_basis::<cubecl_wgpu::WgpuRuntime>(client, basis)
        }
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
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_grad_both_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    g3: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All four scratch slabs share a stride and a slot, because the
        // contraction reads them at identical relative offsets.
        let gbase = slot * g_stride;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            // ── Per-class shape (Task 35-D) ───────────────────────────────
            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // Both-side overlap headroom: g0 must span i..=li+1 AND j..=lj+1.
            let nmax = li + lj + 2u32;
            let lj_ext = lj + 1u32;
            let dj = nmax + 1u32; // stride between consecutive j-levels within an axis block
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 9u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            // Zero the full accumulation buffer.
            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            // ── Primitive loop ───────────────────────────────────────────────
            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    // Pair data, computed in-kernel in F (norm_i = norm_j = 1.0).
                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // ── Build OVERLAP base G-tensor in `g` (fixed-center VRR + HRR) ──
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                    // Zero the three derivative tensors.
                    let mut zi = gbase;
                    while zi < gbase + total_g {
                        g1[zi as usize] = F::new(0.0_f32);
                        g2[zi as usize] = F::new(0.0_f32);
                        g3[zi as usize] = F::new(0.0_f32);
                        zi += 1u32;
                    }

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;
                    let li1 = li + 1u32;

                    // ── g1 = D_j(g0): ket nabla, over i..=li+1, j..=lj. ─────────────
                    //   D_j[j=0] = -2*aj * g0[j=1]
                    //   D_j[j>0] = j * g0[j-1] + (-2*aj) * g0[j+1]
                    let mut a1 = 0u32;
                    while a1 < 3u32 {
                        let off = gbase + a1 * g_per_axis;
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
                        let off = gbase + a2 * g_per_axis;
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
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_ipovlpip` class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn one_e_grad_both_g_per_axis(li: usize, lj: usize) -> usize {
    // Both-side headroom: nmax = li+lj+2, lj_ext = lj+1.
    (li + lj + 3) * (lj + 2)
}

/// Evaluate every launch group of a batched `int1e_ipovlpip` run.
///
/// The kernel has no comptime shape parameter left once `li`/`lj` are per-pair,
/// so a whole work list is one dispatch (Task 35-D, wave 3).
fn run_1e_grad_both_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Four scratch slabs, all `3 * g_per_axis`, sharing a stride and a slot:
        // the contraction reads them at identical relative offsets.
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g3_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_pairs`, by the class index in each pair row, by the per-shell
        // `nprim`/`nctr` read from `shell_meta`, and by the per-class extents.
        unsafe {
            one_electron_grad_both_kernel::launch_unchecked::<f64, R>(
                client,
                crate::plane::cube_count_1d(n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                ArrayArg::from_raw_parts(g_h.clone(), g_len),
                ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                ArrayArg::from_raw_parts(g2_h.clone(), g_len),
                ArrayArg::from_raw_parts(g3_h.clone(), g_len),
                ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                SQRTPI,
                std::f64::consts::PI,
                n_pairs as u32,
                n_cubes,
                g_stride as u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipovlpip` run.
fn dispatch_1e_grad_both_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_grad_both_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_grad_both_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_grad_both_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_grad_both_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_grad_both_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell pair through the batched `int1e_ipovlpip` path.
///
/// The per-tuple compatibility API evaluates exactly one pair and must keep
/// doing so, but it goes through the *same* kernel as a wide batch — a one-pair
/// group. That is what makes every existing parity test a test of the batched
/// kernel too (Task 35-D's acceptance bar).
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        9,
        one_e_grad_both_g_per_axis(li as usize, lj as usize),
        1,
    );
    dispatch_1e_grad_both_batches(backend, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — `int1e_ipipovlp` (∇²bra OVERLAP, rank 9)
//
//  Faithful port of libcint `src/autocode/hess.c` `CINTgout1e_int1e_ipipovlp`:
//    g1 = D_I(g0, i_l+1);  g2 = D_I(g0, i_l+0);  g3 = D_I(g1, i_l+0)
//    gout[n*9 + {0,1,2,3,4,5,6,7,8}] = s[{0,3,6,1,4,7,2,5,8}]  (column-major 3×3)
//      s0=g3x·g0y·g0z  s1=g2x·g1y·g0z  s2=g2x·g0y·g1z
//      s3=g1x·g2y·g0z  s4=g0x·g3y·g0z  s5=g0x·g2y·g1z
//      s6=g1x·g0y·g2z  s7=g0x·g1y·g2z  s8=g0x·g0y·g3z
//  Bra-only headroom (ng={2,0,0,0,...}): nmax = li+lj+2, lj_ext = lj (NO ket +2).
//  ipipovlp uses the no-Rys overlap base; ipipnuc/ipiprinv reuse this gout
//  permutation atop the nuclear Rys base (separate kernel below).
// ─────────────────────────────────────────────────────────────────────────────

/// On-device bra-only rank-9 overlap Hessian (`int1e_ipipovlp`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_gradgrad_bra_ovlp_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    g3: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All four scratch slabs share a stride and a slot, because the
        // contraction reads them at identical relative offsets.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // Bra-only second derivative: g0 spans i..=li+2, j..=lj (no ket
            // headroom).
            let nmax = li + lj + 2u32;
            let lj_ext = lj;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 9u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let li1 = li + 1u32;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);

                    if lj_ext >= 1u32 {
                        one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                    }

                    let mut zi = gbase;
                    while zi < gbase + total_g {
                        g1[zi as usize] = F::new(0.0_f32);
                        g2[zi as usize] = F::new(0.0_f32);
                        g3[zi as usize] = F::new(0.0_f32);
                        zi += 1u32;
                    }
                    let ai2 = F::new(-2.0_f32) * ai;
                    // g1 = D_I(g0) over i..=li+1 (one extra bra level); g2 = D_I(g0)
                    // over i..=li; g3 = D_I(g1) over i..=li.
                    d_i_1e_into::<F>(g1, g, gbase, g_per_axis, dj, lj, li1, ai2);
                    d_i_1e_into::<F>(g2, g, gbase, g_per_axis, dj, lj, li, ai2);
                    d_i_1e_into::<F>(g3, g1, gbase, g_per_axis, dj, lj, li, ai2);

                    gradgrad_bra_contract::<F>(
                        g, g1, g2, g3, cart_out, coeffs, coff_i, coff_j, out_off, gx, gy, gz, dj,
                        nci, block_len, total_len, li, lj, nctr_i, nctr_j, pi, pj,
                    );

                    pj += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// `#[cube]` helper: contract the bra-only ∇² g0/g1/g2/g3 tensors into the
/// 9-component component-leading `cart_out` using libcint's `hess.c` ipip gout
/// permutation `gout[{0..8}] = s[{0,3,6,1,4,7,2,5,8}]`. Shared by the overlap
/// and nuclear bra-only Hessian kernels (the gout block is identical for
/// ipipovlp / ipipnuc / ipiprinv).
#[cube]
#[allow(clippy::too_many_arguments)]
fn gradgrad_bra_contract<F: Float + CubeElement>(
    g: &Array<F>,
    g1: &Array<F>,
    g2: &Array<F>,
    g3: &Array<F>,
    cart_out: &mut Array<F>,
    coeffs: &Array<F>,
    coff_i: u32,
    coff_j: u32,
    out_off: u32,
    gx: u32,
    gy: u32,
    gz: u32,
    dj: u32,
    nci: u32,
    block_len: u32,
    total_len: u32,
    li: u32,
    lj: u32,
    nctr_i: u32,
    nctr_j: u32,
    pi: u32,
    pj: u32,
) {
    let mut ci = 0u32;
    while ci < nctr_i {
        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
        let mut cj = 0u32;
        while cj < nctr_j {
            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
            let weight = coeff_i_val * coeff_j_val;
            let base = out_off + (ci * nctr_j + cj) * total_len;

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

                            // libcint hess.c ipip s-tensor (bra second derivative).
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
                            // gout permutation: [s0,s3,s6,s1,s4,s7,s2,s5,s8].
                            cart_out[(base + elem) as usize] += weight * s0;
                            cart_out[(base + block_len + elem) as usize] += weight * s3;
                            cart_out[(base + 2u32 * block_len + elem) as usize] += weight * s6;
                            cart_out[(base + 3u32 * block_len + elem) as usize] += weight * s1;
                            cart_out[(base + 4u32 * block_len + elem) as usize] += weight * s4;
                            cart_out[(base + 5u32 * block_len + elem) as usize] += weight * s7;
                            cart_out[(base + 6u32 * block_len + elem) as usize] += weight * s2;
                            cart_out[(base + 7u32 * block_len + elem) as usize] += weight * s5;
                            cart_out[(base + 8u32 * block_len + elem) as usize] += weight * s8;

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
}

/// `g_per_axis` for one `int1e_ipipovlp` class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn one_e_gradgrad_bra_ovlp_g_per_axis(li: usize, lj: usize) -> usize {
    // Bra-only second derivative: nmax = li+lj+2, lj_ext = lj.
    (li + lj + 3) * (lj + 1)
}

/// Evaluate every launch group of a batched `int1e_ipipovlp` run (Task 35-D,
/// wave 3). One dispatch per group; the kernel has no comptime shape parameter.
fn run_1e_gradgrad_bra_ovlp_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Four scratch slabs, all `3 * g_per_axis`, sharing a stride and a slot.
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g3_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        unsafe {
            one_electron_gradgrad_bra_ovlp_kernel::launch_unchecked::<f64, R>(
                client,
                crate::plane::cube_count_1d(n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                ArrayArg::from_raw_parts(g_h.clone(), g_len),
                ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                ArrayArg::from_raw_parts(g2_h.clone(), g_len),
                ArrayArg::from_raw_parts(g3_h.clone(), g_len),
                ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                SQRTPI,
                std::f64::consts::PI,
                n_pairs as u32,
                n_cubes,
                g_stride as u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipipovlp` run.
fn dispatch_1e_gradgrad_bra_ovlp_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_gradgrad_bra_ovlp_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_gradgrad_bra_ovlp_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_gradgrad_bra_ovlp_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_gradgrad_bra_ovlp_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_gradgrad_bra_ovlp_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell pair through the batched `int1e_ipipovlp` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_gradgrad_bra_ovlp_on_backend(
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        9,
        one_e_gradgrad_bra_ovlp_g_per_axis(li as usize, lj as usize),
        1,
    );
    dispatch_1e_gradgrad_bra_ovlp_batches(backend, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
}

/// `#[cube]` helper: ket-direction nabla `dst = D_j(src)` for a 3-axis 1e
/// G-tensor, filled over j∈0..=jmax, i∈0..=imax. Stride `dj` between j-levels.
///   D_j[j=0] = -2*aj * src[j=1];  D_j[j>0] = j*src[j-1] + (-2*aj)*src[j+1].
#[cube]
#[allow(clippy::too_many_arguments)]
fn d_j_1e_into<F: Float + CubeElement>(
    dst: &mut Array<F>,
    src: &Array<F>,
    gbase: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    aj2: F,
) {
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = gbase + axisn * g_per_axis;
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
    gbase: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    ai2: F,
) {
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = gbase + axisn * g_per_axis;
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
//  Device kernel — `#[cube(launch)]` — `int1e_p4` (∇⁴, rank 1)
//
//  Implements `int1e_p4` = <i | ∇⁴ | j> (libcint `src/autocode/intor1.c`
//  `CINTgout1e_int1e_p4`, lines 2413-2537). p4 is the Laplacian-of-Laplacian on
//  the OVERLAP G-tensor (no Rys), raising angular-momentum headroom on BOTH the
//  bra and the ket (ng = {2,2,0,0,4,1,1,1} — Pitfall 4): i→i_l+2, j→j_l+2.
//
//  libcint builds g0..g15 but the final rank-1 contraction
//      out = s0 + 2·s4 + 2·s8 + s40 + 2·s44 + s80
//  references ONLY four distinct G-tensors:
//      g0  = overlap base
//      g3  = D_J²(g0)           (libcint G1E_D_J twice; "dj2" below)
//      g12 = D_I²(g0)           (libcint G1E_D_I twice; "di2" below)
//      g15 = D_I²(D_J²(g0))     (libcint G1E_D_I twice on g3; "di2dj2" below)
//  with the six surviving s-terms:
//      s0  = g15·g0 ·g0  = di2dj2_x · g0_y     · g0_z
//      s4  = g12·g3 ·g0  = di2_x    · dj2_y    · g0_z
//      s8  = g12·g0 ·g3  = di2_x    · g0_y     · dj2_z
//      s40 = g0 ·g15·g0  = g0_x     · di2dj2_y · g0_z
//      s44 = g0 ·g12·g3  = g0_x     · di2_y    · dj2_z
//      s80 = g0 ·g0 ·g15 = g0_x     · g0_y     · di2dj2_z
//  This contraction is the Laplacian² (∂²/∂x² + ∂²/∂y² + ∂²/∂z²)² and is copied
//  VERBATIM from intor1.c:2534 — NOT re-derived.
//
//  Headroom: g0 must span i..=li+2, j..=lj+2 → nmax = li+lj+4, lj_ext = lj+2.
//  (The bra-and-ket +2 distinguishes p4 from the ket-only moment families.)
// ─────────────────────────────────────────────────────────────────────────────

/// On-device `int1e_p4` (∇⁴, rank 1 — Laplacian-of-Laplacian on the overlap
/// G-tensor, no Rys). Scratch tensors:
///   `g` = overlap base; `dj1`/`dj2` = D_J/D_J² of g; `di1`/`di2` = D_I/D_I² of g;
///   `t1` = D_I(dj2); `di2dj2` = D_I(t1) = D_I²(D_J²(g)). Output is the single
/// rank-1 component per AO pair (component-leading, blocks of `nci*ncj`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_p4_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    dj1: &mut Array<F>,
    dj2: &mut Array<F>,
    di1: &mut Array<F>,
    di2: &mut Array<F>,
    t1: &mut Array<F>,
    di2dj2: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All seven scratch slabs share a stride and a slot.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // BOTH-side headroom: g0 spans i..=li+2, j..=lj+2 (ng={2,2,...}).
            let nmax = li + lj + 4u32;
            let lj_ext = lj + 2u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            // rank 1: total_len == block_len.
            let out_total = nctr_i * nctr_j * block_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let li2 = li + 2u32;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Overlap base G-tensor g0 in `g`.
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);
                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                    // Zero derivative scratch tensors.
                    let mut zi = gbase;
                    while zi < gbase + total_g {
                        dj1[zi as usize] = F::new(0.0_f32);
                        dj2[zi as usize] = F::new(0.0_f32);
                        di1[zi as usize] = F::new(0.0_f32);
                        di2[zi as usize] = F::new(0.0_f32);
                        t1[zi as usize] = F::new(0.0_f32);
                        di2dj2[zi as usize] = F::new(0.0_f32);
                        zi += 1u32;
                    }

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;

                    // Ket Laplacian chain: dj2 = D_J²(g0), kept at i..=li+2.
                    //   dj1 = D_J(g0)  over j..=lj+1, i..=li+2
                    //   dj2 = D_J(dj1) over j..=lj  , i..=li+2
                    d_j_1e_into::<F>(dj1, g, gbase, g_per_axis, dj, lj + 1u32, li2, aj2);
                    d_j_1e_into::<F>(dj2, dj1, gbase, g_per_axis, dj, lj, li2, aj2);

                    // Bra Laplacian chain: di2 = D_I²(g0), at i..=li, j..=lj.
                    //   di1 = D_I(g0)  over i..=li+1, j..=lj
                    //   di2 = D_I(di1) over i..=li  , j..=lj
                    d_i_1e_into::<F>(di1, g, gbase, g_per_axis, dj, lj, li + 1u32, ai2);
                    d_i_1e_into::<F>(di2, di1, gbase, g_per_axis, dj, lj, li, ai2);

                    // Mixed Laplacian²: di2dj2 = D_I²(dj2), at i..=li, j..=lj.
                    //   t1     = D_I(dj2) over i..=li+1
                    //   di2dj2 = D_I(t1)  over i..=li
                    d_i_1e_into::<F>(t1, dj2, gbase, g_per_axis, dj, lj, li + 1u32, ai2);
                    d_i_1e_into::<F>(di2dj2, t1, gbase, g_per_axis, dj, lj, li, ai2);

                    let mut ci = 0u32;
                    while ci < nctr_i {
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * block_len;

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

                                            // g0 (=a0), dj2 (=b2), di2 (=c2), di2dj2 (=d2).
                                            // Only the axes that survive the 6-term
                                            // Laplacian² contraction are read (b2x and
                                            // c2z never appear — intor1.c:2534).
                                            let a0x = g[(gx + nx) as usize];
                                            let a0y = g[(gy + ny) as usize];
                                            let a0z = g[(gz + nz) as usize];
                                            let b2y = dj2[(gy + ny) as usize];
                                            let b2z = dj2[(gz + nz) as usize];
                                            let c2x = di2[(gx + nx) as usize];
                                            let c2y = di2[(gy + ny) as usize];
                                            let d2x = di2dj2[(gx + nx) as usize];
                                            let d2y = di2dj2[(gy + ny) as usize];
                                            let d2z = di2dj2[(gz + nz) as usize];

                                            // intor1.c:2534 verbatim:
                                            //   s0  = g15·g0·g0 = d2x·a0y·a0z
                                            //   s4  = g12·g3·g0 = c2x·b2y·a0z
                                            //   s8  = g12·g0·g3 = c2x·a0y·b2z
                                            //   s40 = g0·g15·g0 = a0x·d2y·a0z
                                            //   s44 = g0·g12·g3 = a0x·c2y·b2z
                                            //   s80 = g0·g0·g15 = a0x·a0y·d2z
                                            // out = s0 + 2·s4 + 2·s8 + s40 + 2·s44 + s80
                                            let s0 = d2x * a0y * a0z;
                                            let s4 = c2x * b2y * a0z;
                                            let s8 = c2x * a0y * b2z;
                                            let s40 = a0x * d2y * a0z;
                                            let s44 = a0x * c2y * b2z;
                                            let s80 = a0x * a0y * d2z;
                                            let two = F::new(2.0_f32);
                                            let val =
                                                s0 + two * s4 + two * s8 + s40 + two * s44 + s80;

                                            let elem = cj_idx * nci + ci_idx;
                                            cart_out[(base + elem) as usize] += weight * val;

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

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_p4` class.
fn one_e_p4_g_per_axis(li: usize, lj: usize) -> usize {
    // Both-side headroom: nmax = li+lj+4, lj_ext = lj+2.
    (li + lj + 5) * (lj + 3)
}

/// Evaluate every launch group of a batched `int1e_p4` run (Task 35-D,
/// wave 4). One dispatch per group; the kernel has no comptime shape parameter.
fn run_1e_p4_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let dj1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let dj2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let di1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let di2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let t1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let di2dj2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        unsafe {
            one_electron_p4_kernel::launch_unchecked::<f64, R>(
                client,
                crate::plane::cube_count_1d(n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                ArrayArg::from_raw_parts(g_h.clone(), g_len),
                ArrayArg::from_raw_parts(dj1_h.clone(), g_len),
                ArrayArg::from_raw_parts(dj2_h.clone(), g_len),
                ArrayArg::from_raw_parts(di1_h.clone(), g_len),
                ArrayArg::from_raw_parts(di2_h.clone(), g_len),
                ArrayArg::from_raw_parts(t1_h.clone(), g_len),
                ArrayArg::from_raw_parts(di2dj2_h.clone(), g_len),
                ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                SQRTPI,
                std::f64::consts::PI,
                n_pairs as u32,
                n_cubes,
                g_stride as u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_p4` run.
fn dispatch_1e_p4_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_p4_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_p4_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_p4_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_p4_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_p4_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell pair through the batched `int1e_p4` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_p4_on_backend(
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        1,
        one_e_p4_g_per_axis(li as usize, lj as usize),
        1,
    );
    dispatch_1e_p4_batches(backend, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — `int1e_irp` (i·r×∇, rank 9)
//
//  Implements `int1e_irp` = <i | (∇ ⊗ r) | j> (libcint `src/autocode/intor1.c`
//  `CINTgout1e_int1e_irp`, lines 2781-2816). irp is a gauge-origin family: the
//  `r` part reads PTR_COMMON_ORIG via libcint's G1E_RCJ (drj = rj - common_orig).
//  It runs on the overlap-derivative engine (NO Rys), ket headroom +2 (ng[1]=2).
//
//  libcint builds:
//      g0 = overlap base
//      g1 = G1E_D_J(g0)        (∇ on ket;        "dj1" below)
//      g2 = G1E_RCJ(g0)        (r on ket, drj;   "rcj" below)
//      g3 = G1E_D_J(g2)        (∇ on the r-block; "djrcj" below)
//  then emits the 3×3 (∇-axis ⊗ r-axis) tensor s[0..8], copied VERBATIM from
//  intor1.c:2788-2816 (the per-axis g-table for each of the 9 components):
//      s0 = g3·g0·g0   s1 = g2·g1·g0   s2 = g2·g0·g1
//      s3 = g1·g2·g0   s4 = g0·g3·g0   s5 = g0·g2·g1
//      s6 = g1·g0·g2   s7 = g0·g1·g2   s8 = g0·g0·g3
//  (A permuted/transposed order passes a square block but fails a non-square one.)
//
//  Headroom: g3 = D_J(g2) reads g2 at j+1, and g2 = RCJ(g0) reaches level lj+1
//  reading g0 at j and j+1, so g2 must span j..=lj+1 → g0 must span j..=lj+2.
//  Thus lj_ext = lj+2, nmax = li+lj+2.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device `int1e_irp` (i·r×∇, rank 9 — 3×3 ∇⊗r tensor on the overlap-derivative
/// engine, no Rys, reads the gauge origin via `drj`). Scratch tensors:
///   `g` = overlap base g0; `dj1` = D_J(g0); `rcj` = RCJ(g0) (r on ket); `djrcj`
///   = D_J(rcj). Output is the 9 components per AO pair (component-leading, blocks
/// of `nci*ncj`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn one_electron_irp_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    pair_drj: &Array<F>,
    g: &mut Array<F>,
    dj1: &mut Array<F>,
    rcj: &mut Array<F>,
    djrcj: &mut Array<F>,
    di2: &mut Array<F>,
    di2rcj: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] op_kind: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All six scratch slabs share a stride and a slot.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // `drj = rj - origin` is per pair, not per class: the base families
            // measure from a common origin and the `_origj` variants from `rj`
            // itself (making it zero). The host already resolves that choice, so
            // the batch carries the resolved vector rather than re-deriving it.
            let d3 = qi * 3u32;
            let drjx = pair_drj[d3 as usize];
            let drjy = pair_drj[(d3 + 1u32) as usize];
            let drjz = pair_drj[(d3 + 2u32) as usize];

            // op_kind=0: irp, ng={0,2}; op_kind=1: ipipr, ng={2,1}.
            let mut nmax = li + lj + 2u32;
            let mut lj_ext = lj + 2u32;
            if op_kind == 1u32 {
                nmax = li + lj + 3u32;
                lj_ext = lj + 1u32;
            }
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let mut rank = 9u32;
            if op_kind == 1u32 {
                rank = 27u32;
            }
            let total_len = rank * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Overlap base G-tensor g0 in `g`.
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);
                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                    // Zero derivative scratch tensors.
                    let mut zi = gbase;
                    while zi < gbase + total_g {
                        dj1[zi as usize] = F::new(0.0_f32);
                        rcj[zi as usize] = F::new(0.0_f32);
                        djrcj[zi as usize] = F::new(0.0_f32);
                        di2[zi as usize] = F::new(0.0_f32);
                        di2rcj[zi as usize] = F::new(0.0_f32);
                        zi += 1u32;
                    }

                    if op_kind == 0u32 {
                        let aj2 = F::new(-2.0_f32) * aj;
                        d_j_1e_into::<F>(dj1, g, gbase, g_per_axis, dj, lj, li, aj2);
                        rcj_1e_into::<F>(
                            rcj,
                            g,
                            gbase,
                            g_per_axis,
                            dj,
                            lj + 1u32,
                            li,
                            drjx,
                            drjy,
                            drjz,
                        );
                        d_j_1e_into::<F>(djrcj, rcj, gbase, g_per_axis, dj, lj, li, aj2);
                    } else {
                        // hess.c CINTgout1e_int1e_ipipr. g4/g5 are the lower-range
                        // portions of g2/g3 and therefore share those buffers.
                        let ai2 = F::new(-2.0_f32) * ai;
                        rcj_1e_into::<F>(
                            rcj,
                            g,
                            gbase,
                            g_per_axis,
                            dj,
                            lj,
                            li + 2u32,
                            drjx,
                            drjy,
                            drjz,
                        );
                        d_i_1e_into::<F>(dj1, g, gbase, g_per_axis, dj, lj, li + 1u32, ai2);
                        d_i_1e_into::<F>(djrcj, rcj, gbase, g_per_axis, dj, lj, li + 1u32, ai2);
                        d_i_1e_into::<F>(di2, dj1, gbase, g_per_axis, dj, lj, li, ai2);
                        d_i_1e_into::<F>(di2rcj, djrcj, gbase, g_per_axis, dj, lj, li, ai2);
                    }

                    let mut ci = 0u32;
                    while ci < nctr_i {
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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

                                            // Per-axis g-table values (g0/g1/g2/g3).
                                            let g0x = g[(gx + nx) as usize];
                                            let g0y = g[(gy + ny) as usize];
                                            let g0z = g[(gz + nz) as usize];
                                            let g1x = dj1[(gx + nx) as usize];
                                            let g1y = dj1[(gy + ny) as usize];
                                            let g1z = dj1[(gz + nz) as usize];
                                            let g2x = rcj[(gx + nx) as usize];
                                            let g2y = rcj[(gy + ny) as usize];
                                            let g2z = rcj[(gz + nz) as usize];
                                            let g3x = djrcj[(gx + nx) as usize];
                                            let g3y = djrcj[(gy + ny) as usize];
                                            let g3z = djrcj[(gz + nz) as usize];

                                            let elem = cj_idx * nci + ci_idx;
                                            if op_kind == 0u32 {
                                                let s0 = g3x * g0y * g0z;
                                                let s1 = g2x * g1y * g0z;
                                                let s2 = g2x * g0y * g1z;
                                                let s3 = g1x * g2y * g0z;
                                                let s4 = g0x * g3y * g0z;
                                                let s5 = g0x * g2y * g1z;
                                                let s6 = g1x * g0y * g2z;
                                                let s7 = g0x * g1y * g2z;
                                                let s8 = g0x * g0y * g3z;
                                                cart_out
                                                    [(base + 0u32 * block_len + elem) as usize] +=
                                                    weight * s0;
                                                cart_out
                                                    [(base + 1u32 * block_len + elem) as usize] +=
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
                                            } else {
                                                let g6x = di2[(gx + nx) as usize];
                                                let g6y = di2[(gy + ny) as usize];
                                                let g6z = di2[(gz + nz) as usize];
                                                let g7x = di2rcj[(gx + nx) as usize];
                                                let g7y = di2rcj[(gy + ny) as usize];
                                                let g7z = di2rcj[(gz + nz) as usize];
                                                // hess.c:840-953. g4==g2 and g5==g3 at
                                                // the contracted indices; the repeated
                                                // values below preserve libcint's output
                                                // permutation exactly.
                                                let s0 = g7x * g0y * g0z;
                                                let s1 = g6x * g2y * g0z;
                                                let s2 = g6x * g0y * g2z;
                                                let s3 = g3x * g1y * g0z;
                                                let s4 = g1x * g3y * g0z;
                                                let s5 = g1x * g1y * g2z;
                                                let s6 = g3x * g0y * g1z;
                                                let s7 = g1x * g2y * g1z;
                                                let s8 = g1x * g0y * g3z;
                                                let s12 = g2x * g6y * g0z;
                                                let s13 = g0x * g7y * g0z;
                                                let s14 = g0x * g6y * g2z;
                                                let s15 = g2x * g1y * g1z;
                                                let s16 = g0x * g3y * g1z;
                                                let s17 = g0x * g1y * g3z;
                                                let s24 = g2x * g0y * g6z;
                                                let s25 = g0x * g2y * g6z;
                                                let s26 = g0x * g0y * g7z;
                                                let mut comp = 0u32;
                                                while comp < 27u32 {
                                                    let mut value = s0;
                                                    if comp == 1u32 {
                                                        value = s1;
                                                    } else if comp == 2u32 {
                                                        value = s2;
                                                    } else if comp == 3u32 || comp == 9u32 {
                                                        value = s3;
                                                    } else if comp == 4u32 || comp == 10u32 {
                                                        value = s4;
                                                    } else if comp == 5u32 || comp == 11u32 {
                                                        value = s5;
                                                    } else if comp == 6u32 || comp == 18u32 {
                                                        value = s6;
                                                    } else if comp == 7u32 || comp == 19u32 {
                                                        value = s7;
                                                    } else if comp == 8u32 || comp == 20u32 {
                                                        value = s8;
                                                    } else if comp == 12u32 {
                                                        value = s12;
                                                    } else if comp == 13u32 {
                                                        value = s13;
                                                    } else if comp == 14u32 {
                                                        value = s14;
                                                    } else if comp == 15u32 || comp == 21u32 {
                                                        value = s15;
                                                    } else if comp == 16u32 || comp == 22u32 {
                                                        value = s16;
                                                    } else if comp == 17u32 || comp == 23u32 {
                                                        value = s17;
                                                    } else if comp == 24u32 {
                                                        value = s24;
                                                    } else if comp == 25u32 {
                                                        value = s25;
                                                    } else if comp == 26u32 {
                                                        value = s26;
                                                    }
                                                    cart_out[(base + comp * block_len + elem)
                                                        as usize] += weight * value;
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

            qi += qi_step;
        }
    }
}

/// `#[cube]` helper: ket-direction position multiply `dst = RCJ(src)` for a 3-axis
/// 1e G-tensor, filled over j∈0..=jmax, i∈0..=imax. This is libcint's `CINTx1j_1e`
/// (`G1E_RCJ`): each axis shifts the ket level up by one and adds `drj` times the
/// same level —   RCJ[j][i] = src[j+1][i] + drj_axis * src[j][i].
/// `drj = rj - env[PTR_COMMON_ORIG]` is the per-axis gauge-origin offset.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rcj_1e_into<F: Float + CubeElement>(
    dst: &mut Array<F>,
    src: &Array<F>,
    gbase: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    drjx: F,
    drjy: F,
    drjz: F,
) {
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = gbase + axisn * g_per_axis;
        let mut drj = drjx;
        if axisn == 1u32 {
            drj = drjy;
        } else if axisn == 2u32 {
            drj = drjz;
        }
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            let jhi = (jn + 1u32) * dj;
            let mut ii = 0u32;
            while ii <= imax {
                dst[(off + jbase + ii) as usize] =
                    src[(off + jhi + ii) as usize] + drj * src[(off + jbase + ii) as usize];
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `#[cube]` helper: bra-direction position multiply `dst = R0I(src)` for a 3-axis
/// 1e G-tensor, filled over j∈0..=jmax, i∈0..=imax. This is libcint's `CINTx1i_1e`
/// (`G1E_R0I`): each axis shifts the BRA level up by one and adds `ri` times the
/// same level —   R0I[j][i] = src[j][i+1] + ri_axis * src[j][i].
/// `ri` is the ABSOLUTE bra basis-center coordinate (g1e.c:446, `gx[i+1]+ri*gx[i]`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn r0i_1e_into<F: Float + CubeElement>(
    dst: &mut Array<F>,
    src: &Array<F>,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    rix: F,
    riy: F,
    riz: F,
) {
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut ri = rix;
        if axisn == 1u32 {
            ri = riy;
        } else if axisn == 2u32 {
            ri = riz;
        }
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            let mut ii = 0u32;
            while ii <= imax {
                dst[(off + jbase + ii) as usize] =
                    src[(off + jbase + ii + 1u32) as usize] + ri * src[(off + jbase + ii) as usize];
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `g_per_axis` for one `int1e_irp` / `int1e_ipipr` class.
fn one_e_irp_g_per_axis(op_kind: u32, li: usize, lj: usize) -> usize {
    // op_kind=0: nmax = li+lj+2, lj_ext = lj+2; op_kind=1: +3 / +1.
    let (nmax, lj_ext) = if op_kind == 1 {
        (li + lj + 3, lj + 1)
    } else {
        (li + lj + 2, lj + 2)
    };
    (nmax + 1) * (lj_ext + 1)
}

/// Evaluate every launch group of a batched `int1e_irp` / `int1e_ipipr` run
/// (Task 35-D, wave 4).
///
/// `op_kind` is fixed by the caller's operator and is the kernel's only
/// comptime parameter, so a whole work list is one dispatch. `pair_drj` carries
/// the resolved `rj - origin` per pair — see the kernel for why that is per-pair
/// rather than per-class.
fn run_1e_irp_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let drj_h = client.create_from_slice(f64::as_bytes(&pair_drj[index]));
        let slab = || client.empty(g_len * std::mem::size_of::<f64>());
        let g_h = slab();
        let dj1_h = slab();
        let rcj_h = slab();
        let djrcj_h = slab();
        let di2_h = slab();
        let di2rcj_h = slab();
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($op:expr) => {
                unsafe {
                    one_electron_irp_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(drj_h.clone(), pair_drj[index].len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(dj1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(rcj_h.clone(), g_len),
                        ArrayArg::from_raw_parts(djrcj_h.clone(), g_len),
                        ArrayArg::from_raw_parts(di2_h.clone(), g_len),
                        ArrayArg::from_raw_parts(di2rcj_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        SQRTPI,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $op,
                        per_unit,
                    );
                }
            };
        }

        if op_kind == 1 {
            launch_with!(1u32);
        } else {
            launch_with!(0u32);
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_irp` / `int1e_ipipr` run.
fn dispatch_1e_irp_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_irp_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, pair_drj, op_kind)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_irp_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, pair_drj, op_kind)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_irp_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, pair_drj, op_kind)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_irp_batches::<cubecl_hip::HipRuntime>(client, basis, groups, pair_drj, op_kind)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_irp_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, pair_drj, op_kind)
        }
    }
}

/// One shell pair through the batched `int1e_irp` / `int1e_ipipr` path — a
/// one-pair group through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_irp_on_backend(
    backend: &ResolvedBackend,
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
    op_kind: u32,
) -> Vec<f64> {
    let rank = if op_kind == 1 { 27 } else { 9 };
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        rank,
        one_e_irp_g_per_axis(op_kind, li as usize, lj as usize),
        1,
    );
    dispatch_1e_irp_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        std::slice::from_ref(&drj.to_vec()),
        op_kind,
    )
    .pop()
    .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — 1e GIAO OVERLAP-engine families (rank 3)
//
//  Implements the spin-free 1e GIAO/CG families that ride the no-Rys overlap
//  G-tensor engine (Phase 26 GIAO-01). op_kind selects the family:
//    0 = int1e_govlp  (<G i|OVLP|j>, intor3.c CINTgout1e_int1e_govlp, factor 0.5)
//    1 = int1e_igovlp (<G i|OVLP|j>, intor1.c CINTgout1e_int1e_igovlp, factor 0.5,
//                       sign-flipped gout vs govlp)
//    2 = int1e_cg_irxp   (<i|OVLP|RC CROSS P j>, intor1.c, gauge-relative drj)
//    3 = int1e_giao_irjxp(<i|OVLP|R CROSS P j>, intor1.c, ket-center rj)
//    4 = int1e_igkin  (<G i|P DOT P|j>, intor1.c CINTgout1e_int1e_igkin, factor 0.25)
//
//  The gout `c[]·s[]` combos are transcribed VERBATIM from the cited libcint gout
//  functions (intor1.c / intor3.c). c = ri - rj (the bra-minus-ket displacement).
//
//  g-tensor recipe per family (built into the named scratch tensors):
//    govlp/igovlp: g1 = R0I(g0)   (bra position multiply; G1E_R0I)
//    cg_irxp:      g1 = D_J(g0),   g2 = RCJ(g0)[j+1] (gauge-relative), g3 = D_J(g2)
//    giao_irjxp:   g1 = D_J(g0),   g2 = R_J(g0)[j+1] (ket-center rj),  g3 = D_J(g2)
//    igkin:        g1 = D_J(g0)[i+1], g2 = D_J(D_J(g0))[i+1], g3 = D_J(g2)[i+1]
//                  then g4..g7 = R0I(g0..g3)  (27-s kinetic-of-r table)
//
//  Headroom: govlp/igovlp need bra+1 (R0I reads i+1) → nmax=li+lj+1, lj_ext=lj.
//  cg_irxp/giao_irjxp need ket+2 → nmax=li+lj+2, lj_ext=lj+2.
//  igkin needs bra+2 (R0I after D_J), ket+1 → nmax=li+lj+3, lj_ext=lj+1.
//  We size for the maximum (nmax=li+lj+3, lj_ext=lj+2) at comptime per op_kind.
// ─────────────────────────────────────────────────────────────────────────────

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// On-device 1e GIAO overlap-engine kernel (rank 3). `op_kind` selects the family
/// (0=govlp 1=igovlp 2=cg_irxp 3=giao_irjxp 4=igkin). Emits REAL components
/// (component-leading; the host materializes the complex re=0/im=value view).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn one_electron_giao_ovlp_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    pair_drj: &Array<F>,
    g: &mut Array<F>,
    t1: &mut Array<F>,
    t2: &mut Array<F>,
    t3: &mut Array<F>,
    t4: &mut Array<F>,
    t5: &mut Array<F>,
    t6: &mut Array<F>,
    t7: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] op_kind: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All eight scratch slabs share a stride and a slot.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // `drj = rj - origin` is per pair; the host resolves the origin.
            let d3 = qi * 3u32;
            let drjx = pair_drj[d3 as usize];
            let drjy = pair_drj[(d3 + 1u32) as usize];
            let drjz = pair_drj[(d3 + 2u32) as usize];

            // Comptime headroom (sized for the max over all op_kinds so one buffer
            // shape serves every family; the host runner sizes scratch identically).
            let nmax = li + lj + 3u32;
            let lj_ext = lj + 2u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 3u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            // common_factor per family: govlp/igovlp *= 0.5, igkin *= 0.25, others *= 1.
            // One arm per family, even where two families share a factor: each arm
            // corresponds to a single `envs.common_factor *=` line in libcint's
            // `intor1.c`, and merging the equal ones would break that correspondence.
            #[allow(clippy::if_same_then_else)]
            let mut fam_factor = F::new(1.0_f32);
            #[allow(clippy::if_same_then_else)]
            if comptime!(op_kind == 0u32) {
                fam_factor = F::new(0.5_f32);
            } else if comptime!(op_kind == 1u32) {
                fam_factor = F::new(0.5_f32);
            } else if comptime!(op_kind == 4u32) {
                fam_factor = F::new(0.25_f32);
            }

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Overlap base G-tensor g0.
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        t1[gi as usize] = F::new(0.0_f32);
                        t2[gi as usize] = F::new(0.0_f32);
                        t3[gi as usize] = F::new(0.0_f32);
                        t4[gi as usize] = F::new(0.0_f32);
                        t5[gi as usize] = F::new(0.0_f32);
                        t6[gi as usize] = F::new(0.0_f32);
                        t7[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);
                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                    let aj2 = F::new(-2.0_f32) * aj;

                    // Build the family-specific decorating g-tensors.
                    if comptime!(op_kind == 0u32) {
                        // govlp / igovlp share R0I(g0).
                        r0i_1e_into::<F>(t1, g, g_per_axis, dj, lj, li, rix, riy, riz);
                    } else if comptime!(op_kind == 1u32) {
                        r0i_1e_into::<F>(t1, g, g_per_axis, dj, lj, li, rix, riy, riz);
                    } else if comptime!(op_kind == 2u32) {
                        // cg_irxp: g1=D_J(g0); g2=RCJ(g0)[j+1]; g3=D_J(g2).
                        d_j_1e_into::<F>(t1, g, gbase, g_per_axis, dj, lj, li, aj2);
                        rcj_1e_into::<F>(
                            t2,
                            g,
                            gbase,
                            g_per_axis,
                            dj,
                            lj + 1u32,
                            li,
                            drjx,
                            drjy,
                            drjz,
                        );
                        d_j_1e_into::<F>(t3, t2, gbase, g_per_axis, dj, lj, li, aj2);
                    } else if comptime!(op_kind == 3u32) {
                        // giao_irjxp: g1=D_J(g0); g2=R_J(g0)[j+1] (ket-center rj); g3=D_J(g2).
                        d_j_1e_into::<F>(t1, g, gbase, g_per_axis, dj, lj, li, aj2);
                        rcj_1e_into::<F>(
                            t2,
                            g,
                            gbase,
                            g_per_axis,
                            dj,
                            lj + 1u32,
                            li,
                            rjx,
                            rjy,
                            rjz,
                        );
                        d_j_1e_into::<F>(t3, t2, gbase, g_per_axis, dj, lj, li, aj2);
                    } else {
                        // igkin (intor1.c): g1=D_J(g0,j_l), g2=D_J(g0,j_l+1),
                        // g3=D_J(g2,j_l) = D_J²(g0); then g4..g7 = R0I(g0,g1,g2,g3).
                        // The gout only references g0, g3, g4=R0I(g0), g7=R0I(g3), so
                        // g3 MUST be the SECOND derivative (not the third).
                        d_j_1e_into::<F>(t1, g, gbase, g_per_axis, dj, lj, li + 1u32, aj2);
                        d_j_1e_into::<F>(t2, g, gbase, g_per_axis, dj, lj + 1u32, li + 1u32, aj2);
                        d_j_1e_into::<F>(t3, t2, gbase, g_per_axis, dj, lj, li + 1u32, aj2);
                        r0i_1e_into::<F>(t4, g, g_per_axis, dj, lj, li, rix, riy, riz);
                        r0i_1e_into::<F>(t5, t1, g_per_axis, dj, lj, li, rix, riy, riz);
                        r0i_1e_into::<F>(t6, t2, g_per_axis, dj, lj, li, rix, riy, riz);
                        r0i_1e_into::<F>(t7, t3, g_per_axis, dj, lj, li, rix, riy, riz);
                    }

                    let cx = rirjx;
                    let cy = rirjy;
                    let cz = rirjz;

                    let mut ci = 0u32;
                    while ci < nctr_i {
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = fam_factor * coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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

                                            let mut o0 = F::new(0.0_f32);
                                            let mut o1 = F::new(0.0_f32);
                                            let mut o2 = F::new(0.0_f32);

                                            if comptime!(op_kind <= 1u32) {
                                                // govlp / igovlp: s = R0I components.
                                                let g1x = t1[(gx + nx) as usize];
                                                let g1y = t1[(gy + ny) as usize];
                                                let g1z = t1[(gz + nz) as usize];
                                                let s0 = g1x * g0y * g0z;
                                                let s1 = g0x * g1y * g0z;
                                                let s2 = g0x * g0y * g1z;
                                                // govlp: + c1 s2 - c2 s1, etc.
                                                o0 = cy * s2 - cz * s1;
                                                o1 = cz * s0 - cx * s2;
                                                o2 = cx * s1 - cy * s0;
                                                if comptime!(op_kind == 1u32) {
                                                    // igovlp = -govlp gout.
                                                    o0 = F::new(0.0_f32) - o0;
                                                    o1 = F::new(0.0_f32) - o1;
                                                    o2 = F::new(0.0_f32) - o2;
                                                }
                                            } else if comptime!(op_kind <= 3u32) {
                                                // cg_irxp / giao_irjxp: 9-s curl table.
                                                let g1x = t1[(gx + nx) as usize];
                                                let g1y = t1[(gy + ny) as usize];
                                                let g1z = t1[(gz + nz) as usize];
                                                let g2x = t2[(gx + nx) as usize];
                                                let g2y = t2[(gy + ny) as usize];
                                                let g2z = t2[(gz + nz) as usize];
                                                let g3x = t3[(gx + nx) as usize];
                                                let g3y = t3[(gy + ny) as usize];
                                                let g3z = t3[(gz + nz) as usize];
                                                let s1 = g2x * g1y * g0z;
                                                let s2 = g2x * g0y * g1z;
                                                let s3 = g1x * g2y * g0z;
                                                let s5 = g0x * g2y * g1z;
                                                let s6 = g1x * g0y * g2z;
                                                let s7 = g0x * g1y * g2z;
                                                // gout: s5-s7, s6-s2, s1-s3.
                                                o0 = s5 - s7;
                                                o1 = s6 - s2;
                                                o2 = s1 - s3;
                                                let _ = g3x;
                                                let _ = g3y;
                                                let _ = g3z;
                                            } else {
                                                // igkin: 27-s kinetic-of-r table.
                                                let g0x4 = t4[(gx + nx) as usize];
                                                let g0y4 = t4[(gy + ny) as usize];
                                                let g0z4 = t4[(gz + nz) as usize];
                                                let g1x = t1[(gx + nx) as usize];
                                                let g1y = t1[(gy + ny) as usize];
                                                let g1z = t1[(gz + nz) as usize];
                                                let g1x5 = t5[(gx + nx) as usize];
                                                let g1y5 = t5[(gy + ny) as usize];
                                                let g1z5 = t5[(gz + nz) as usize];
                                                let g2x = t2[(gx + nx) as usize];
                                                let g2y = t2[(gy + ny) as usize];
                                                let g2z = t2[(gz + nz) as usize];
                                                let g2x6 = t6[(gx + nx) as usize];
                                                let g2y6 = t6[(gy + ny) as usize];
                                                let g2z6 = t6[(gz + nz) as usize];
                                                let g3x = t3[(gx + nx) as usize];
                                                let g3y = t3[(gy + ny) as usize];
                                                let g3z = t3[(gz + nz) as usize];
                                                let g3x7 = t7[(gx + nx) as usize];
                                                let g3y7 = t7[(gy + ny) as usize];
                                                let g3z7 = t7[(gz + nz) as usize];
                                                // s[] table (intor1.c igkin): g0..g3 = D_J chain,
                                                // g4..g7 = R0I(g0..g3). Index map matches the
                                                // 27-entry s[] of CINTgout1e_int1e_igkin.
                                                let s0 = g3x7 * g0y * g0z;
                                                let s4 = g0x4 * g3y * g0z;
                                                let s8 = g0x4 * g0y * g3z;
                                                let s9 = g3x * g0y4 * g0z;
                                                let s13 = g0x * g3y7 * g0z;
                                                let s17 = g0x * g0y4 * g3z;
                                                let s18 = g3x * g0y * g0z4;
                                                let s22 = g0x * g3y * g0z4;
                                                let s26 = g0x * g0y * g3z7;
                                                let _ = g1x;
                                                let _ = g1y;
                                                let _ = g1z;
                                                let _ = g1x5;
                                                let _ = g1y5;
                                                let _ = g1z5;
                                                let _ = g2x;
                                                let _ = g2y;
                                                let _ = g2z;
                                                let _ = g2x6;
                                                let _ = g2y6;
                                                let _ = g2z6;
                                                // gout (intor1.c igkin):
                                                //  [0]=c1 s18 - c2 s9 + c1 s22 - c2 s13 + c1 s26 - c2 s17
                                                //  [1]=c2 s0 - c0 s18 + c2 s4 - c0 s22 + c2 s8 - c0 s26
                                                //  [2]=c0 s9 - c1 s0 + c0 s13 - c1 s4 + c0 s17 - c1 s8
                                                o0 = cy * s18 - cz * s9 + cy * s22 - cz * s13
                                                    + cy * s26
                                                    - cz * s17;
                                                o1 = cz * s0 - cx * s18 + cz * s4 - cx * s22
                                                    + cz * s8
                                                    - cx * s26;
                                                o2 = cx * s9 - cy * s0 + cx * s13 - cy * s4
                                                    + cx * s17
                                                    - cy * s8;
                                            }

                                            let elem = cj_idx * nci + ci_idx;
                                            cart_out[(base + 0u32 * block_len + elem) as usize] +=
                                                weight * o0;
                                            cart_out[(base + 1u32 * block_len + elem) as usize] +=
                                                weight * o1;
                                            cart_out[(base + 2u32 * block_len + elem) as usize] +=
                                                weight * o2;

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

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one GIAO overlap-engine class.
fn one_e_giao_ovlp_g_per_axis(li: usize, lj: usize) -> usize {
    // Sized for the max over all op_kinds: nmax = li+lj+3, lj_ext = lj+2.
    (li + lj + 4) * (lj + 3)
}

/// Evaluate every launch group of a batched GIAO overlap-engine run (Task 35-D,
/// wave 4). `op_kind` is fixed by the caller's operator, so a whole work list is
/// one dispatch.
fn run_1e_giao_ovlp_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let drj_h = client.create_from_slice(f64::as_bytes(&pair_drj[index]));
        let slab = || client.empty(g_len * std::mem::size_of::<f64>());
        let g_h = slab();
        let t1_h = slab();
        let t2_h = slab();
        let t3_h = slab();
        let t4_h = slab();
        let t5_h = slab();
        let t6_h = slab();
        let t7_h = slab();
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($kind:expr) => {
                unsafe {
                    one_electron_giao_ovlp_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(drj_h.clone(), pair_drj[index].len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t2_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t3_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t4_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t5_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t6_h.clone(), g_len),
                        ArrayArg::from_raw_parts(t7_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        SQRTPI,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $kind,
                        per_unit,
                    );
                }
            };
        }

        match op_kind {
            0u32 => launch_with!(0u32),
            1u32 => launch_with!(1u32),
            2u32 => launch_with!(2u32),
            3u32 => launch_with!(3u32),
            4u32 => launch_with!(4u32),
            _ => unreachable!("invalid GIAO overlap op_kind {op_kind} (must be 0..=4)"),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched GIAO overlap-engine run.
fn dispatch_1e_giao_ovlp_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_giao_ovlp_batches::<cubecl::cpu::CpuRuntime>(
            client, basis, groups, pair_drj, op_kind,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_giao_ovlp_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups, pair_drj, op_kind,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_giao_ovlp_batches::<cubecl_cuda::CudaRuntime>(
            client, basis, groups, pair_drj, op_kind,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_giao_ovlp_batches::<cubecl_hip::HipRuntime>(
            client, basis, groups, pair_drj, op_kind,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_giao_ovlp_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups, pair_drj, op_kind,
        ),
    }
}

/// One shell pair through the batched GIAO overlap-engine path — a one-pair
/// group through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_giao_ovlp_on_backend(
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
    drj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        3,
        one_e_giao_ovlp_g_per_axis(li as usize, lj as usize),
        1,
    );
    dispatch_1e_giao_ovlp_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        std::slice::from_ref(&drj.to_vec()),
        op_kind,
    )
    .pop()
    .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — 1e GIAO NUCLEAR-engine families
//
//  Implements the spin-free 1e GIAO/CG families that ride the nuclear/Rys 1e path
//  (Phase 26 GIAO-01). op_kind selects the family:
//    0 = int1e_gnuc  (<G i|NUC|j>,  rank 3, intor3.c, factor 0.5)
//    1 = int1e_ignuc (<G i|NUC|j>,  rank 3, intor1.c, factor 0.5, sign-flip)
//    2 = int1e_ia01p (<i|NABLA-RINV|CROSS P j>, rank 3, intor1.c)
//    3 = int1e_a01gp (<G i|NABLA-RINV CROSS P|j>, rank 9, intor1.c, factor 0.5)
//    4 = int1e_cg_a11part   (<i|NABLA-RINV|RC j>, rank 9, intor1.c, factor -0.5)
//    5 = int1e_giao_a11part (<i|NABLA-RINV|R  j>, rank 9, intor1.c, factor -0.5)
//
//  All emit REAL components → host complex re=0/im=value materialization (D-15).
//  gout `c[]·s[]` combos transcribed VERBATIM from the cited libcint gout fns.
//  Flat gbuf holds 8 tensors (g0 + t1..t7); decorations via the flat helpers.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device 1e GIAO nuclear-engine kernel. `op_kind` selects the family;
/// `rank` (comptime) is 3 (gnuc/ignuc/ia01p) or 9 (a01gp/cg_a11part/giao_a11part).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_giao_nuc_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    pair_drj: &Array<F>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    gbuf: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    pi_const: F,
    norig: u32,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] op_kind: u32,
    #[comptime] rank: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Written and read entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;
        // One slab per slot holding all tensors back to back; the stride must
        // cover the widest class's whole tensor set.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // `drj = rj - origin` is per pair; the host resolves the origin.
            let d3 = qi * 3u32;
            let drjx = pair_drj[d3 as usize];
            let drjy = pair_drj[(d3 + 1u32) as usize];
            let drjz = pair_drj[(d3 + 2u32) as usize];

            // Headroom sized for the max over families (a01gp: bra i+3, ket j+2).
            let nmax = li + lj + 5u32;
            let lj_ext = lj + 2u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = rank * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            // One arm per family, even where two families share a factor: each arm
            // corresponds to a single `envs.common_factor *=` line in libcint's
            // `intor1.c`, and merging the equal ones would break that correspondence.
            #[allow(clippy::if_same_then_else)]
            let mut fam_factor = F::new(1.0_f32);
            #[allow(clippy::if_same_then_else)]
            if comptime!(op_kind == 0u32) {
                fam_factor = F::new(0.5_f32);
            } else if comptime!(op_kind == 1u32) {
                fam_factor = F::new(0.5_f32);
            } else if comptime!(op_kind == 3u32) {
                // a01gp: libcint applies `envs.common_factor *= 0.5`
                // (intor1.c:551,572 int1e_a01gp_{cart,sph}). The earlier kernel
                // left this at 1.0, producing a uniform ~2x on every output
                // component (component 0 vanishes on the H1xO test block, masking
                // the factor there). This is the 26-02 ket-derivative "double-count".
                fam_factor = F::new(0.5_f32);
            } else if comptime!(op_kind == 4u32) {
                fam_factor = F::new(-0.5_f32);
            } else if comptime!(op_kind == 5u32) {
                fam_factor = F::new(-0.5_f32);
            }

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let cx = rix - rjx;
            let cy = riy - rjy;
            let cz = riz - rjz;

            let li1 = li + 1u32;
            let li2 = li + 2u32;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;

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
                            rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 2u32) {
                            rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 3u32) {
                            rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 4u32) {
                            rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else {
                            rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        }

                        let fac1 = F::new(2.0_f32) * pi_const * charge_factor * fac / zeta;

                        let mut irys: u32 = 0u32;
                        while irys < nrys {
                            let u_n = urys[irys as usize];
                            let w_n = wrys[irys as usize];
                            let tau = u_n / (F::new(1.0_f32) + u_n);
                            let rt = aij2 * (F::new(1.0_f32) - tau);

                            let c00x = (px - rix) + tau * crijx;
                            let c00y = (py - riy) + tau * crijy;
                            let c00z = (pz - riz) + tau * crijz;

                            // Zero the whole flat buffer (g0 + t1..t7 = 8 tensors).
                            let mut zi = gbase;
                            while zi < gbase + 8u32 * total_g {
                                gbuf[zi as usize] = F::new(0.0_f32);
                                zi += 1u32;
                            }
                            // Per-root nuclear base G-tensor g0 (tensor slot 0).
                            gbuf[gx as usize] = F::new(1.0_f32);
                            gbuf[gy as usize] = F::new(1.0_f32);
                            gbuf[gz as usize] = fac1 * w_n;
                            one_electron_vrr2e_axis::<F>(gbuf, gx, c00x, rt, nmax);
                            one_electron_vrr2e_axis::<F>(gbuf, gy, c00y, rt, nmax);
                            one_electron_vrr2e_axis::<F>(gbuf, gz, c00z, rt, nmax);
                            one_electron_hrr_axis::<F>(gbuf, gx, rirjx, dj, nmax, lj_ext);
                            one_electron_hrr_axis::<F>(gbuf, gy, rirjy, dj, nmax, lj_ext);
                            one_electron_hrr_axis::<F>(gbuf, gz, rirjz, dj, nmax, lj_ext);

                            // Build the family-specific decoration tensors.
                            if comptime!(op_kind <= 1u32) {
                                // gnuc / ignuc: t1 = R0I(g0).
                                r0i_1e_flat::<F>(
                                    gbuf, gbase, 1u32, 0u32, total_g, g_per_axis, dj, lj, li, rix,
                                    riy, riz,
                                );
                            } else if comptime!(op_kind == 2u32) {
                                // ia01p: t1=D_J(g0); t2=D_J(g0,j+1); t3=D_I(g0,j+1);
                                // t2+=t3; t3=D_J(t2).
                                d_j_1e_flat::<F>(
                                    gbuf, gbase, 1u32, 0u32, total_g, g_per_axis, dj, lj, li, aj2,
                                );
                                d_j_1e_flat::<F>(
                                    gbuf,
                                    gbase,
                                    2u32,
                                    0u32,
                                    total_g,
                                    g_per_axis,
                                    dj,
                                    lj + 1u32,
                                    li,
                                    aj2,
                                );
                                d_i_1e_flat::<F>(
                                    gbuf,
                                    gbase,
                                    3u32,
                                    0u32,
                                    total_g,
                                    g_per_axis,
                                    dj,
                                    lj + 1u32,
                                    li,
                                    ai2,
                                );
                                add_tensor_flat::<F>(gbuf, gbase, 2u32, 3u32, total_g);
                                d_j_1e_flat::<F>(
                                    gbuf, gbase, 3u32, 2u32, total_g, g_per_axis, dj, lj, li, aj2,
                                );
                            } else if comptime!(op_kind == 3u32) {
                                // a01gp: t1=D_J(g0,i+2); t2=D_J(g0,i+1,j+1);
                                // t3=D_I(g0,i+1,j+1); t2+=t3; t3=D_J(t2,i+2);
                                // t4=R0I(g0); t5=R0I(t1); t6=R0I(t2); t7=R0I(t3).
                                d_j_1e_flat::<F>(
                                    gbuf, gbase, 1u32, 0u32, total_g, g_per_axis, dj, lj, li2, aj2,
                                );
                                d_j_1e_flat::<F>(
                                    gbuf,
                                    gbase,
                                    2u32,
                                    0u32,
                                    total_g,
                                    g_per_axis,
                                    dj,
                                    lj + 1u32,
                                    li1,
                                    aj2,
                                );
                                d_i_1e_flat::<F>(
                                    gbuf,
                                    gbase,
                                    3u32,
                                    0u32,
                                    total_g,
                                    g_per_axis,
                                    dj,
                                    lj + 1u32,
                                    li1,
                                    ai2,
                                );
                                add_tensor_flat::<F>(gbuf, gbase, 2u32, 3u32, total_g);
                                d_j_1e_flat::<F>(
                                    gbuf, gbase, 3u32, 2u32, total_g, g_per_axis, dj, lj, li2, aj2,
                                );
                                r0i_1e_flat::<F>(
                                    gbuf, gbase, 4u32, 0u32, total_g, g_per_axis, dj, lj, li, rix,
                                    riy, riz,
                                );
                                r0i_1e_flat::<F>(
                                    gbuf, gbase, 5u32, 1u32, total_g, g_per_axis, dj, lj, li, rix,
                                    riy, riz,
                                );
                                r0i_1e_flat::<F>(
                                    gbuf, gbase, 6u32, 2u32, total_g, g_per_axis, dj, lj, li, rix,
                                    riy, riz,
                                );
                                r0i_1e_flat::<F>(
                                    gbuf, gbase, 7u32, 3u32, total_g, g_per_axis, dj, lj, li, rix,
                                    riy, riz,
                                );
                            } else {
                                // cg_a11part(4) / giao_a11part(5): t1=RCJ/R_J(g0);
                                // t2=D_J(g0,j+1); t3=D_I(g0,j+1); t2+=t3; t3=RCJ/R_J(t2).
                                // RCJ uses drj (gauge-relative); R_J uses rj (ket center).
                                let mut ox = drjx;
                                let mut oy = drjy;
                                let mut oz = drjz;
                                if comptime!(op_kind == 5u32) {
                                    ox = rjx;
                                    oy = rjy;
                                    oz = rjz;
                                }
                                rcj_1e_flat::<F>(
                                    gbuf, gbase, 1u32, 0u32, total_g, g_per_axis, dj, lj, li, ox,
                                    oy, oz,
                                );
                                d_j_1e_flat::<F>(
                                    gbuf,
                                    gbase,
                                    2u32,
                                    0u32,
                                    total_g,
                                    g_per_axis,
                                    dj,
                                    lj + 1u32,
                                    li,
                                    aj2,
                                );
                                d_i_1e_flat::<F>(
                                    gbuf,
                                    gbase,
                                    3u32,
                                    0u32,
                                    total_g,
                                    g_per_axis,
                                    dj,
                                    lj + 1u32,
                                    li,
                                    ai2,
                                );
                                add_tensor_flat::<F>(gbuf, gbase, 2u32, 3u32, total_g);
                                rcj_1e_flat::<F>(
                                    gbuf, gbase, 3u32, 2u32, total_g, g_per_axis, dj, lj, li, ox,
                                    oy, oz,
                                );
                            }

                            // Contract + accumulate the s-table into cart_out.
                            let mut ci = 0u32;
                            while ci < nctr_i {
                                let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                                let mut cj = 0u32;
                                while cj < nctr_j {
                                    let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                                    let weight = fam_factor * coeff_i_val * coeff_j_val;
                                    let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                                    let elem = cj_idx * nci + ci_idx;

                                                    giao_nuc_accumulate::<F>(
                                                        gbuf, cart_out, base, block_len, total_g,
                                                        g_per_axis, gx, gy, gz, nx, ny, nz, elem,
                                                        weight, cx, cy, cz, op_kind,
                                                    );

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

            qi += qi_step;
        }
    }
}

/// `#[cube]` helper: accumulate the GIAO nuclear-family s-table for one AO pair.
/// Reads g0 (slot 0) + decoration tensors (slots 1..7) from `gbuf` and scatters
/// the rank-3 or rank-9 component combo into `cart_out` (component-leading). The
/// gout combos are transcribed verbatim from the cited libcint gout functions.
#[cube]
#[allow(clippy::too_many_arguments)]
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
fn giao_nuc_accumulate<F: Float + CubeElement>(
    gbuf: &Array<F>,
    cart_out: &mut Array<F>,
    base: u32,
    block_len: u32,
    total_g: u32,
    g_per_axis: u32,
    gx: u32,
    gy: u32,
    gz: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    elem: u32,
    weight: F,
    cx: F,
    cy: F,
    cz: F,
    #[comptime] op_kind: u32,
) {
    // Slot helpers: tensor t, axis offset a, index n → gbuf[t*total_g + a + n].
    let t0x = gbuf[(0u32 * total_g + gx + nx) as usize];
    let t0y = gbuf[(0u32 * total_g + gy + ny) as usize];
    let t0z = gbuf[(0u32 * total_g + gz + nz) as usize];

    if comptime!(op_kind <= 1u32) {
        // gnuc / ignuc: s = R0I (slot 1).
        let t1x = gbuf[(1u32 * total_g + gx + nx) as usize];
        let t1y = gbuf[(1u32 * total_g + gy + ny) as usize];
        let t1z = gbuf[(1u32 * total_g + gz + nz) as usize];
        let s0 = t1x * t0y * t0z;
        let s1 = t0x * t1y * t0z;
        let s2 = t0x * t0y * t1z;
        let mut o0 = cy * s2 - cz * s1;
        let mut o1 = cz * s0 - cx * s2;
        let mut o2 = cx * s1 - cy * s0;
        if comptime!(op_kind == 1u32) {
            o0 = F::new(0.0_f32) - o0;
            o1 = F::new(0.0_f32) - o1;
            o2 = F::new(0.0_f32) - o2;
        }
        cart_out[(base + 0u32 * block_len + elem) as usize] += weight * o0;
        cart_out[(base + 1u32 * block_len + elem) as usize] += weight * o1;
        cart_out[(base + 2u32 * block_len + elem) as usize] += weight * o2;
    } else if comptime!(op_kind == 2u32) {
        // ia01p: 9-s curl table (g0,t1=g1,t2=g2,t3=g3). gout: s5-s7, s6-s2, s1-s3.
        let g1x = gbuf[(1u32 * total_g + gx + nx) as usize];
        let g1y = gbuf[(1u32 * total_g + gy + ny) as usize];
        let g1z = gbuf[(1u32 * total_g + gz + nz) as usize];
        let g2x = gbuf[(2u32 * total_g + gx + nx) as usize];
        let g2y = gbuf[(2u32 * total_g + gy + ny) as usize];
        let g2z = gbuf[(2u32 * total_g + gz + nz) as usize];
        let s1 = g2x * g1y * t0z;
        let s2 = g2x * t0y * g1z;
        let s3 = g1x * g2y * t0z;
        let s5 = t0x * g2y * g1z;
        let s6 = g1x * t0y * g2z;
        let s7 = t0x * g1y * g2z;
        cart_out[(base + 0u32 * block_len + elem) as usize] += weight * (s5 - s7);
        cart_out[(base + 1u32 * block_len + elem) as usize] += weight * (s6 - s2);
        cart_out[(base + 2u32 * block_len + elem) as usize] += weight * (s1 - s3);
    } else if comptime!(op_kind == 3u32) {
        // a01gp: 27-s table (g0..g3 = slots 0..3, g4..g7 = slots 4..7).
        // s[k] index map matches CINTgout1e_int1e_a01gp; 9-component gout uses c.
        let g0x = t0x;
        let g0y = t0y;
        let g0z = t0z;
        let g1x = gbuf[(1u32 * total_g + gx + nx) as usize];
        let g1y = gbuf[(1u32 * total_g + gy + ny) as usize];
        let g1z = gbuf[(1u32 * total_g + gz + nz) as usize];
        let g2x = gbuf[(2u32 * total_g + gx + nx) as usize];
        let g2y = gbuf[(2u32 * total_g + gy + ny) as usize];
        let g2z = gbuf[(2u32 * total_g + gz + nz) as usize];
        let g3x = gbuf[(3u32 * total_g + gx + nx) as usize];
        let g3y = gbuf[(3u32 * total_g + gy + ny) as usize];
        let g3z = gbuf[(3u32 * total_g + gz + nz) as usize];
        let g4x = gbuf[(4u32 * total_g + gx + nx) as usize];
        let g4y = gbuf[(4u32 * total_g + gy + ny) as usize];
        let g4z = gbuf[(4u32 * total_g + gz + nz) as usize];
        let g5x = gbuf[(5u32 * total_g + gx + nx) as usize];
        let g5y = gbuf[(5u32 * total_g + gy + ny) as usize];
        let g5z = gbuf[(5u32 * total_g + gz + nz) as usize];
        let g6x = gbuf[(6u32 * total_g + gx + nx) as usize];
        let g6y = gbuf[(6u32 * total_g + gy + ny) as usize];
        let g6z = gbuf[(6u32 * total_g + gz + nz) as usize];
        let g7x = gbuf[(7u32 * total_g + gx + nx) as usize];
        let g7y = gbuf[(7u32 * total_g + gy + ny) as usize];
        let g7z = gbuf[(7u32 * total_g + gz + nz) as usize];
        // 27 s[] (intor1.c CINTgout1e_int1e_a01gp).
        let s1 = g6x * g1y * g0z;
        let s2 = g6x * g0y * g1z;
        let s3 = g5x * g2y * g0z;
        let s5 = g4x * g2y * g1z;
        let s6 = g5x * g0y * g2z;
        let s7 = g4x * g1y * g2z;
        let s10 = g2x * g5y * g0z;
        let s11 = g2x * g4y * g1z;
        let s12 = g1x * g6y * g0z;
        let s14 = g0x * g6y * g1z;
        let s15 = g1x * g4y * g2z;
        let s16 = g0x * g5y * g2z;
        let s19 = g2x * g1y * g4z;
        let s20 = g2x * g0y * g5z;
        let s21 = g1x * g2y * g4z;
        let s23 = g0x * g2y * g5z;
        let s24 = g1x * g0y * g6z;
        let s25 = g0x * g1y * g6z;
        let _ = g3x;
        let _ = g3y;
        let _ = g3z;
        let _ = g7x;
        let _ = g7y;
        let _ = g7z;
        let _ = g0x;
        let _ = g0y;
        let _ = g0z;
        // gout (intor1.c), c = ri - rj.
        let o0 = cy * s23 - cz * s14 - cy * s25 + cz * s16;
        let o1 = cy * s24 - cz * s15 - cy * s20 + cz * s11;
        let o2 = cy * s19 - cz * s10 - cy * s21 + cz * s12;
        let o3 = cz * s5 - cx * s23 - cz * s7 + cx * s25;
        let o4 = cz * s6 - cx * s24 - cz * s2 + cx * s20;
        let o5 = cz * s1 - cx * s19 - cz * s3 + cx * s21;
        let o6 = cx * s14 - cy * s5 - cx * s16 + cy * s7;
        let o7 = cx * s15 - cy * s6 - cx * s11 + cy * s2;
        let o8 = cx * s10 - cy * s1 - cx * s12 + cy * s3;
        cart_out[(base + 0u32 * block_len + elem) as usize] += weight * o0;
        cart_out[(base + 1u32 * block_len + elem) as usize] += weight * o1;
        cart_out[(base + 2u32 * block_len + elem) as usize] += weight * o2;
        cart_out[(base + 3u32 * block_len + elem) as usize] += weight * o3;
        cart_out[(base + 4u32 * block_len + elem) as usize] += weight * o4;
        cart_out[(base + 5u32 * block_len + elem) as usize] += weight * o5;
        cart_out[(base + 6u32 * block_len + elem) as usize] += weight * o6;
        cart_out[(base + 7u32 * block_len + elem) as usize] += weight * o7;
        cart_out[(base + 8u32 * block_len + elem) as usize] += weight * o8;
    } else {
        // cg_a11part / giao_a11part: direct gout[k] = s[k], 9-s table.
        let g1x = gbuf[(1u32 * total_g + gx + nx) as usize];
        let g1y = gbuf[(1u32 * total_g + gy + ny) as usize];
        let g1z = gbuf[(1u32 * total_g + gz + nz) as usize];
        let g2x = gbuf[(2u32 * total_g + gx + nx) as usize];
        let g2y = gbuf[(2u32 * total_g + gy + ny) as usize];
        let g2z = gbuf[(2u32 * total_g + gz + nz) as usize];
        let g3x = gbuf[(3u32 * total_g + gx + nx) as usize];
        let g3y = gbuf[(3u32 * total_g + gy + ny) as usize];
        let g3z = gbuf[(3u32 * total_g + gz + nz) as usize];
        let s0 = g3x * t0y * t0z;
        let s1 = g2x * g1y * t0z;
        let s2 = g2x * t0y * g1z;
        let s3 = g1x * g2y * t0z;
        let s4 = t0x * g3y * t0z;
        let s5 = t0x * g2y * g1z;
        let s6 = g1x * t0y * g2z;
        let s7 = t0x * g1y * g2z;
        let s8 = t0x * t0y * g3z;
        cart_out[(base + 0u32 * block_len + elem) as usize] += weight * s0;
        cart_out[(base + 1u32 * block_len + elem) as usize] += weight * s1;
        cart_out[(base + 2u32 * block_len + elem) as usize] += weight * s2;
        cart_out[(base + 3u32 * block_len + elem) as usize] += weight * s3;
        cart_out[(base + 4u32 * block_len + elem) as usize] += weight * s4;
        cart_out[(base + 5u32 * block_len + elem) as usize] += weight * s5;
        cart_out[(base + 6u32 * block_len + elem) as usize] += weight * s6;
        cart_out[(base + 7u32 * block_len + elem) as usize] += weight * s7;
        cart_out[(base + 8u32 * block_len + elem) as usize] += weight * s8;
    }
    let _ = g_per_axis;
}

/// Per-slot slab units for one GIAO nuclear-engine class.
///
/// This kernel keeps **eight** tensors in one buffer, so the slot stride has to
/// cover all of them; expressing that as `8 * g_per_axis` lets
/// [`one_e_g_slab_stride`] size it the way every other family's is sized.
fn one_e_giao_nuc_slab_units(li: usize, lj: usize) -> usize {
    // Headroom sized for the max over families: nmax = li+lj+5, lj_ext = lj+2.
    8 * (li + lj + 6) * (lj + 3)
}

/// Evaluate every launch group of a batched GIAO nuclear-engine run
/// (Task 35-D, wave 4).
///
/// Three comptime parameters, one of which is a shape selector: `op_kind` and
/// `rank` are both fixed by the caller's operator (`rank` is a function of
/// `op_kind`), leaving `nroots` as the merge key — one dispatch per Rys order.
fn run_1e_giao_nuc_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    origin_coords: &[f64],
    origin_charges: &[f64],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    // CubeCL `Array`s must be non-empty; a molecule with no origins never
    // reads these.
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
    let oc_h = client.create_from_slice(f64::as_bytes(coords_src));
    let och_h = client.create_from_slice(f64::as_bytes(charges_src));
    let norig = origin_charges.len() as u32;

    let mut results = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let drj_h = client.create_from_slice(f64::as_bytes(&pair_drj[index]));
        let gbuf_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($kind:expr, $rank:expr, $nr:expr) => {
                unsafe {
                    one_electron_giao_nuc_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(drj_h.clone(), pair_drj[index].len()),
                        ArrayArg::from_raw_parts(oc_h.clone(), coords_src.len()),
                        ArrayArg::from_raw_parts(och_h.clone(), charges_src.len()),
                        ArrayArg::from_raw_parts(gbuf_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        PIE4,
                        std::f64::consts::PI,
                        norig,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $kind,
                        $rank,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        // Comptime (op_kind, rank, nroots). nroots ∈ 1..=5 (MAX_DEVICE_NROOTS),
        // rank is fixed per op_kind. Enumerate the valid combinations
        // explicitly.
        macro_rules! by_nroots {
            ($kind:expr, $rank:expr) => {
                match group.nroots {
                    1u32 => launch_with!($kind, $rank, 1u32),
                    2u32 => launch_with!($kind, $rank, 2u32),
                    3u32 => launch_with!($kind, $rank, 3u32),
                    4u32 => launch_with!($kind, $rank, 4u32),
                    _ => launch_with!($kind, $rank, 5u32),
                }
            };
        }
        match op_kind {
            0u32 => by_nroots!(0u32, 3u32),
            1u32 => by_nroots!(1u32, 3u32),
            2u32 => by_nroots!(2u32, 3u32),
            3u32 => by_nroots!(3u32, 9u32),
            4u32 => by_nroots!(4u32, 9u32),
            5u32 => by_nroots!(5u32, 9u32),
            _ => unreachable!("invalid GIAO nuclear op_kind {op_kind} (must be 0..=5)"),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched GIAO nuclear-engine run.
#[allow(clippy::too_many_arguments)]
fn dispatch_1e_giao_nuc_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    origin_coords: &[f64],
    origin_charges: &[f64],
    op_kind: u32,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_giao_nuc_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            pair_drj,
            origin_coords,
            origin_charges,
            op_kind,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_giao_nuc_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            pair_drj,
            origin_coords,
            origin_charges,
            op_kind,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_giao_nuc_batches::<cubecl_cuda::CudaRuntime>(
            client,
            basis,
            groups,
            pair_drj,
            origin_coords,
            origin_charges,
            op_kind,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_giao_nuc_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            pair_drj,
            origin_coords,
            origin_charges,
            op_kind,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_giao_nuc_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            pair_drj,
            origin_coords,
            origin_charges,
            op_kind,
        ),
    }
}

/// One shell pair through the batched GIAO nuclear-engine path — a one-pair
/// group through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_giao_nuc_on_backend(
    backend: &ResolvedBackend,
    op_kind: u32,
    rank: u32,
    nroots: u32,
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
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<f64> {
    // `rank` is fixed by `op_kind` (3 for the gnuc/ignuc/ia01p arms, 9 for
    // a01gp/a11part/g_ssa10ssp); the caller passes the value it already derived
    // so the two cannot drift.
    debug_assert_eq!(rank, if op_kind >= 3 { 9 } else { 3 });
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        rank as usize,
        one_e_giao_nuc_slab_units(li as usize, lj as usize),
        nroots,
    );
    dispatch_1e_giao_nuc_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        std::slice::from_ref(&drj.to_vec()),
        origin_coords,
        origin_charges,
        op_kind,
    )
    .pop()
    .unwrap_or_default()
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
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_grad_kin_both_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    dj1: &mut Array<F>,
    dj2: &mut Array<F>,
    dj3: &mut Array<F>,
    di0: &mut Array<F>,
    di1: &mut Array<F>,
    di2: &mut Array<F>,
    di3: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // All eight scratch slabs share a stride and a slot, because the
        // contraction reads them at identical relative offsets.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // Kinetic both-side headroom: g0 spans i..=li+1, j..=lj+3.
            let nmax = li + lj + 4u32;
            let lj_ext = lj + 3u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 9u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let li1 = li + 1u32;
            let neg_half = F::new(-0.5_f32);

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Build overlap base G-tensor g0 (= dj0) in `g`.
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
                    g[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));

                    one_electron_vrr_axis::<F>(g, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(g, gz, pz - riz, aij2, nmax);
                    one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                    // Zero derivative tensors.
                    let mut zi = gbase;
                    while zi < gbase + total_g {
                        dj1[zi as usize] = F::new(0.0_f32);
                        dj2[zi as usize] = F::new(0.0_f32);
                        dj3[zi as usize] = F::new(0.0_f32);
                        di0[zi as usize] = F::new(0.0_f32);
                        di1[zi as usize] = F::new(0.0_f32);
                        di2[zi as usize] = F::new(0.0_f32);
                        di3[zi as usize] = F::new(0.0_f32);
                        zi += 1u32;
                    }

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;

                    // Ket j-derivative chain (all at i..=li+1).
                    d_j_1e_into::<F>(dj1, g, gbase, g_per_axis, dj, lj + 2u32, li1, aj2);
                    d_j_1e_into::<F>(dj2, dj1, gbase, g_per_axis, dj, lj + 1u32, li1, aj2);
                    d_j_1e_into::<F>(dj3, dj2, gbase, g_per_axis, dj, lj, li1, aj2);

                    // Bra nabla of each ket-derivative tensor (at i..=li, j..=lj).
                    d_i_1e_into::<F>(di0, g, gbase, g_per_axis, dj, lj, li, ai2);
                    d_i_1e_into::<F>(di1, dj1, gbase, g_per_axis, dj, lj, li, ai2);
                    d_i_1e_into::<F>(di2, dj2, gbase, g_per_axis, dj, lj, li, ai2);
                    d_i_1e_into::<F>(di3, dj3, gbase, g_per_axis, dj, lj, li, ai2);

                    let mut ci = 0u32;
                    while ci < nctr_i {
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = neg_half * coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                            let s0 =
                                                c3x * a0y * a0z + c1x * b2y * a0z + c1x * a0y * b2z;
                                            let s1 =
                                                c2x * b1y * a0z + c0x * b3y * a0z + c0x * b1y * b2z;
                                            let s2 =
                                                c2x * a0y * b1z + c0x * b2y * b1z + c0x * a0y * b3z;
                                            let s3 =
                                                b3x * c0y * a0z + b1x * c2y * a0z + b1x * c0y * b2z;
                                            let s4 =
                                                b2x * c1y * a0z + a0x * c3y * a0z + a0x * c1y * b2z;
                                            let s5 =
                                                b2x * c0y * b1z + a0x * c2y * b1z + a0x * c0y * b3z;
                                            let s6 =
                                                b3x * a0y * c0z + b1x * b2y * c0z + b1x * a0y * c2z;
                                            let s7 =
                                                b2x * b1y * c0z + a0x * b3y * c0z + a0x * b1y * c2z;
                                            let s8 =
                                                b2x * a0y * c1z + a0x * b2y * c1z + a0x * a0y * c3z;

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

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_ipkinip` class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn one_e_grad_kin_both_g_per_axis(li: usize, lj: usize) -> usize {
    // Kinetic both-side headroom: nmax = li+lj+4, lj_ext = lj+3.
    (li + lj + 5) * (lj + 4)
}

/// Evaluate every launch group of a batched `int1e_ipkinip` run (Task 35-D,
/// wave 3). One dispatch per group; the kernel has no comptime shape parameter.
fn run_1e_grad_kin_both_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Eight scratch slabs, all `3 * g_per_axis`, sharing a stride and a
        // slot: the contraction reads them at identical relative offsets.
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let slab = |()| client.empty(g_len * std::mem::size_of::<f64>());
        let g_h = slab(());
        let dj1_h = slab(());
        let dj2_h = slab(());
        let dj3_h = slab(());
        let di0_h = slab(());
        let di1_h = slab(());
        let di2_h = slab(());
        let di3_h = slab(());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_pairs`, by the class index in each pair row, by the per-shell
        // `nprim`/`nctr` read from `shell_meta`, and by the per-class extents.
        unsafe {
            one_electron_grad_kin_both_kernel::launch_unchecked::<f64, R>(
                client,
                crate::plane::cube_count_1d(n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                ArrayArg::from_raw_parts(g_h.clone(), g_len),
                ArrayArg::from_raw_parts(dj1_h.clone(), g_len),
                ArrayArg::from_raw_parts(dj2_h.clone(), g_len),
                ArrayArg::from_raw_parts(dj3_h.clone(), g_len),
                ArrayArg::from_raw_parts(di0_h.clone(), g_len),
                ArrayArg::from_raw_parts(di1_h.clone(), g_len),
                ArrayArg::from_raw_parts(di2_h.clone(), g_len),
                ArrayArg::from_raw_parts(di3_h.clone(), g_len),
                ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                SQRTPI,
                std::f64::consts::PI,
                n_pairs as u32,
                n_cubes,
                g_stride as u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipkinip` run.
fn dispatch_1e_grad_kin_both_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_grad_kin_both_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_grad_kin_both_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_grad_kin_both_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_grad_kin_both_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_grad_kin_both_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell pair through the batched `int1e_ipkinip` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        9,
        one_e_grad_kin_both_g_per_axis(li as usize, lj as usize),
        1,
    );
    dispatch_1e_grad_kin_both_batches(backend, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
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
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_nuc_grad_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    pi_const: F,
    norig: u32,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Written and read entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;
        let gbase = slot * g_stride;

        // Blocked walk under `per_unit == 1`, grid-stride otherwise.
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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            // ── Per-class shape (Task 35-D) ───────────────────────────────
            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // nmax = li+lj+1 (+1 bra headroom), lj_ext = lj.
            let nmax = li + lj + 1u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 3u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            // Zero the full accumulation buffer.
            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    let ai2 = F::new(-2.0_f32) * ai;

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
                        let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                        // Rys roots/weights (comptime nroots).
                        if comptime!(nroots == 1u32) {
                            rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 2u32) {
                            rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 3u32) {
                            rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 4u32) {
                            rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else {
                            rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        }

                        // fac1 = 2*PI * charge_factor * fac / zeta.
                        let fac1 = F::new(2.0_f32) * pi_const * charge_factor * fac / zeta;

                        let mut irys: u32 = 0u32;
                        while irys < nrys {
                            let u_n = urys[irys as usize];
                            let w_n = wrys[irys as usize];
                            let tau = u_n / (F::new(1.0_f32) + u_n);
                            let rt = aij2 * (F::new(1.0_f32) - tau);

                            let c00x = (px - rix) + tau * crijx;
                            let c00y = (py - riy) + tau * crijy;
                            let c00z = (pz - riz) + tau * crijz;

                            // Build per-root G-tensor in `g` (root-dependent c00, b10=rt).
                            let mut gi = gbase;
                            while gi < gbase + total_g {
                                g[gi as usize] = F::new(0.0_f32);
                                gi += 1u32;
                            }
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
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
                            let mut g1i = gbase;
                            while g1i < gbase + total_g {
                                g1[g1i as usize] = F::new(0.0_f32);
                                g1i += 1u32;
                            }
                            let mut axisn = 0u32;
                            while axisn < 3u32 {
                                let off = gbase + axisn * g_per_axis;
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
                                let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                                let mut cj = 0u32;
                                while cj < nctr_j {
                                    let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                                    let weight = coeff_i_val * coeff_j_val;
                                    let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                                    cart_out[(base + block_len + elem) as usize] +=
                                                        weight * g0x * g1y * g0z;
                                                    cart_out[(base + 2u32 * block_len + elem)
                                                        as usize] += weight * g0x * g0y * g1z;

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

            qi += qi_step;
        }
    }
}

/// Evaluate every launch group of a batched `int1e_ipnuc` run.
///
/// The nuclear arm is a Rys quadrature, so unlike the overlap/kinetic gradient
/// it does specialize on `nroots` — one dispatch per distinct Rys order rather
/// than one for the whole list.
fn run_1e_nuc_grad_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    // CubeCL `Array`s must be non-empty; a molecule with no origins never
    // reads these.
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
    let norig = origin_charges.len() as u32;

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    one_electron_nuc_grad_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(coords_h.clone(), coords_src.len()),
                        ArrayArg::from_raw_parts(charges_h.clone(), charges_src.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        PIE4,
                        std::f64::consts::PI,
                        norig,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        match group.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipnuc` run.
fn dispatch_1e_nuc_grad_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_nuc_grad_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_nuc_grad_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_nuc_grad_batches::<cubecl_cuda::CudaRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_nuc_grad_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_nuc_grad_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
    }
}

/// `g_per_axis` for one `int1e_ipnuc` class; mirrors the kernel's own sizing.
fn one_e_nuc_grad_g_per_axis(li: usize, lj: usize) -> usize {
    (li + lj + 2) * (lj + 1)
}

/// One shell pair through the batched `int1e_ipnuc` path — a one-pair group, so
/// the per-tuple API and a wide batch execute the same kernel (Task 35-D).
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
    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (exps, coeffs, center, nprim, nctr) in [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_j, coeff_j, rj, nprim_j, nctr_j),
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

    let (li_u, lj_u) = (li as usize, lj as usize);
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let out_len = nctr_i as usize * nctr_j as usize * 3 * nci * ncj;

    let mut group = OneEDerivLaunchGroup::new(nroots);
    let class_index = group.push_class(li, lj, one_e_nuc_grad_g_per_axis(li_u, lj_u));
    group.pairs.extend_from_slice(&[0, 1, 0, class_index]);
    group.out_len = out_len;

    let handles = upload_flat_basis(backend, &basis);
    dispatch_1e_nuc_grad_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        origin_coords,
        origin_charges,
    )
    .pop()
    .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Batched 1e gradient families (Task 35-D)
// ─────────────────────────────────────────────────────────────────────────────

/// The 1e gradient operator a batched run evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneEDerivOperator {
    /// `int1e_ipovlp` — `\nabla` on the bra, overlap kernel.
    IpOvlp,
    /// `int1e_ipkin` — `\nabla` on the bra, kinetic kernel.
    IpKin,
    /// `int1e_ipnuc` — `\nabla` on the bra, nuclear-attraction kernel.
    IpNuc,
}

impl OneEDerivOperator {
    /// libcint symbol this operator corresponds to.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::IpOvlp => "int1e_ipovlp_sph",
            Self::IpKin => "int1e_ipkin_sph",
            Self::IpNuc => "int1e_ipnuc_sph",
        }
    }

    /// `g_per_axis` for one class under this operator's headroom.
    fn g_per_axis(self, li: usize, lj: usize) -> usize {
        match self {
            Self::IpOvlp => one_e_grad_bra_g_per_axis(0, li, lj),
            Self::IpKin => one_e_grad_bra_g_per_axis(1, li, lj),
            Self::IpNuc => one_e_nuc_grad_g_per_axis(li, lj),
        }
    }

    /// Rys order for one class; 1 for the two non-Rys operators.
    fn nroots(self, li: usize, lj: usize) -> usize {
        match self {
            Self::IpNuc => (li + lj).div_ceil(2) + 1,
            _ => 1,
        }
    }
}

/// Spherical AO blocks for a batched 1e gradient run.
///
/// Each pair's block is **component-leading**: three consecutive `[nj][ni]`
/// spherical matrices, `i` fastest, contraction-major on both axes — the same
/// layout the per-pair path writes to staging.
#[derive(Clone, Debug, Default)]
pub struct OneEDerivBatchOutput {
    /// Concatenated spherical AO blocks, in the caller's pair order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where pair `n`'s 3-component block starts.
    pub offsets: Vec<usize>,
    /// Execution statistics.
    pub stats: crate::kernels::two_electron::BatchExecutionStats,
}

/// Where one `(li,lj)` class landed after launch-group merging.
struct OneEDerivPlacement {
    li: u8,
    lj: u8,
    /// Index into the group list — which dispatch's buffer holds these blocks.
    group: usize,
    /// Caller-order indices of this class's pairs.
    members: Vec<usize>,
    /// Each member's offset into the group's Cartesian buffer.
    cart_offsets: Vec<usize>,
    /// Half-open range of the group's Cartesian buffer this class owns.
    cart_span: (usize, usize),
}

/// Evaluate a list of shell pairs as `int1e_ipovlp`, `int1e_ipkin` or
/// `int1e_ipnuc`, one dispatch per Rys order (Task 35-D).
///
/// The two non-Rys operators collapse to a **single** dispatch for the whole
/// work list, because `op_kind` is fixed by the caller and nothing else is
/// comptime.
///
/// # Errors
/// Returns [`cintxRsError::UnsupportedApi`] on an out-of-range shell index, or
/// when a class needs more Rys roots than the device serves. The batch is
/// rejected as a whole rather than partly evaluated.
pub fn evaluate_1e_deriv_pair_batch(
    backend: &ResolvedBackend,
    operator: OneEDerivOperator,
    shells: &[crate::kernels::two_electron::BatchShell],
    atoms: &[BatchAtom],
    pairs: &[[u32; 2]],
) -> Result<OneEDerivBatchOutput, cintxRsError> {
    let resident = crate::kernels::two_electron::ResidentBasis::new(backend, shells)?;
    evaluate_1e_deriv_pair_batch_resident(backend, operator, &resident, atoms, pairs)
}

/// [`evaluate_1e_deriv_pair_batch`] against a basis already on the device
/// (Task 34-C2).
///
/// Identical results; the difference is that the flattened basis is the
/// caller's [`crate::kernels::two_electron::ResidentBasis`] rather than a
/// throwaway one, so `basis_upload_bytes` is the full upload on the first call
/// and **0** on every later one. A gradient build that walks the same pair list
/// every step is exactly the case this exists for.
///
/// # Errors
/// As [`evaluate_1e_deriv_pair_batch`], plus a backend mismatch on `resident`.
pub fn evaluate_1e_deriv_pair_batch_resident(
    backend: &ResolvedBackend,
    operator: OneEDerivOperator,
    resident: &crate::kernels::two_electron::ResidentBasis,
    atoms: &[BatchAtom],
    pairs: &[[u32; 2]],
) -> Result<OneEDerivBatchOutput, cintxRsError> {
    use crate::transform::c2s::cart_to_sph_1e_into;

    resident.check_for(operator.symbol(), backend)?;
    let shells = resident.shells();

    let mut offsets = Vec::with_capacity(pairs.len());
    let mut total = 0_usize;
    for pair in pairs {
        for &shell in pair {
            if shell as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("{}:shell-index-out-of-range:{shell}", operator.symbol()),
                });
            }
        }
        offsets.push(total);
        total += 3 * shells[pair[0] as usize].ao_len() * shells[pair[1] as usize].ao_len();
    }

    let mut output = OneEDerivBatchOutput {
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

    let mut origin_coords = Vec::with_capacity(atoms.len() * 3);
    let mut origin_charges = Vec::with_capacity(atoms.len());
    for atom in atoms {
        origin_coords.extend_from_slice(&atom.center);
        // libcint's nuclear attraction carries `-Z_C`; the per-pair path applies
        // the same sign through the same kernel argument.
        origin_charges.push(-atom.charge);
    }

    let mut grouped: std::collections::BTreeMap<[u8; 2], Vec<usize>> = Default::default();
    for (index, pair) in pairs.iter().enumerate() {
        grouped
            .entry([shells[pair[0] as usize].l, shells[pair[1] as usize].l])
            .or_default()
            .push(index);
    }

    let ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int1eDeriv,
    );
    let mut groups: Vec<OneEDerivLaunchGroup> = Vec::new();
    let mut group_of: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut classes: Vec<OneEDerivPlacement> = Vec::with_capacity(grouped.len());

    for (class, members) in grouped {
        let [li, lj] = class;
        let nroots = operator.nroots(li as usize, lj as usize);
        if nroots > ceiling {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "{}:nroots={nroots} exceeds device ceiling {ceiling} for l=({li},{lj})",
                    operator.symbol()
                ),
            });
        }
        let nroots = nroots as u32;

        let group_index = match group_of.get(&nroots) {
            Some(&index) => index,
            None => {
                groups.push(OneEDerivLaunchGroup::new(nroots));
                let index = groups.len() - 1;
                group_of.insert(nroots, index);
                index
            }
        };
        let group = &mut groups[group_index];
        let class_index = group.push_class(
            u32::from(li),
            u32::from(lj),
            operator.g_per_axis(li as usize, lj as usize),
        );

        // Three components per contraction block.
        let cart_block = 3 * ncart(li) * ncart(lj);
        group.pairs.reserve(members.len() * 4);
        let mut cart_offsets = Vec::with_capacity(members.len());
        let cart_span_start = group.out_len;
        for &index in &members {
            let pair = pairs[index];
            let nctr_product =
                shells[pair[0] as usize].nctr as usize * shells[pair[1] as usize].nctr as usize;
            cart_offsets.push(group.out_len);
            group
                .pairs
                .extend_from_slice(&[pair[0], pair[1], group.out_len as u32, class_index]);
            group.out_len += nctr_product * cart_block;
        }

        classes.push(OneEDerivPlacement {
            li,
            lj,
            group: group_index,
            members,
            cart_offsets,
            // Members were appended contiguously, so the class owns exactly this
            // range. The s/p normalization below scales it and no other class's.
            cart_span: (cart_span_start, group.out_len),
        });
    }

    let handles = resident.handles();
    let dispatch_start = std::time::Instant::now();
    let mut carts = match operator {
        OneEDerivOperator::IpOvlp => dispatch_1e_grad_bra_batches(backend, handles, &groups, 0),
        OneEDerivOperator::IpKin => dispatch_1e_grad_bra_batches(backend, handles, &groups, 1),
        OneEDerivOperator::IpNuc => {
            dispatch_1e_nuc_grad_batches(backend, handles, &groups, &origin_coords, &origin_charges)
        }
    };
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
        .map(|group| one_e_g_slab_stride(group.max_g_per_axis) * std::mem::size_of::<f64>())
        .max()
        .unwrap_or(0);
    output.stats.transfer_bytes = output.stats.basis_upload_bytes
        + groups
            .iter()
            .map(OneEDerivLaunchGroup::upload_bytes)
            .sum::<usize>();

    let transform_start = std::time::Instant::now();

    // libcint moves the s/p spherical normalization into the primitive loop,
    // so the c2s tables carry 1.0 there; applied to the *Cartesian* buffer,
    // before the transform, exactly where the per-pair path applies it.
    // Scoped to this class's own span: a dispatch carries several classes.
    //
    // This prepass mutates `carts`, so it stays serial and runs to completion
    // before the transform below reads any of it (Task 36-T2).
    for class in &classes {
        let sp_scale = common_fac_sp(class.li) * common_fac_sp(class.lj);
        let (span_start, span_end) = class.cart_span;
        if (sp_scale - 1.0).abs() > 1e-15 {
            for value in carts[class.group][span_start..span_end].iter_mut() {
                *value *= sp_scale;
            }
        }
    }
    let carts = &carts;

    // Task 36-T2: one job per pair, in the caller's order, each writing a
    // disjoint output block.
    let mut placement = vec![(0_usize, 0_usize); pairs.len()];
    for (class_index, class) in classes.iter().enumerate() {
        for (slot, &index) in class.members.iter().enumerate() {
            placement[index] = (class_index, slot);
        }
    }
    let lens: Vec<usize> = pairs
        .iter()
        .map(|pair| 3 * shells[pair[0] as usize].ao_len() * shells[pair[1] as usize].ao_len())
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
            let (li, lj) = (class.li, class.lj);
            let (nci, ncj) = (ncart(li), ncart(lj));
            let (nsi, nsj) = (nsph(li), nsph(lj));
            let block_len = nci * ncj;
            let total_len = 3 * block_len;

            profile.start();
            sph.clear();
            sph.resize(nsi * nsj, 0.0);
            profile.charge_alloc();

            let cart = &carts[class.group];
            let pair = pairs[index];
            let (nctr_i, nctr_j) = (
                shells[pair[0] as usize].nctr as usize,
                shells[pair[1] as usize].nctr as usize,
            );
            let (ni_sph, nj_sph) = (nctr_i * nsi, nctr_j * nsj);
            let sph_block = ni_sph * nj_sph;
            let src_base = class.cart_offsets[slot];

            for comp in 0..3usize {
                let dst_comp = comp * sph_block;
                for ci in 0..nctr_i {
                    for cj in 0..nctr_j {
                        let base = src_base + (ci * nctr_j + cj) * total_len + comp * block_len;
                        cart_to_sph_1e_into(
                            &cart[base..base + block_len],
                            sph,
                            li,
                            lj,
                            c2s_scratch,
                        );
                        profile.charge_transform();
                        for mj in 0..nsj {
                            let jj = cj * nsj + mj;
                            for mi in 0..nsi {
                                let ii = ci * nsi + mi;
                                block[dst_comp + ii + jj * ni_sph] = sph[mj * nsi + mi];
                            }
                        }
                        profile.charge_scatter();
                    }
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

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — plain `int1e_rinv` (rank 1, Cluster B)
//
//  Single-center 1/r Coulomb potential evaluated at the rinv center (env[4..6],
//  PTR_RINV_ORIG — D-04/OQ-1), charge=+1, NO atom-sum. Verbatim port of the scalar
//  nuclear Rys arm (one_electron_scalar_kernel op_kind=2) with the atom-loop
//  dropped to a SINGLE origin and the charge factor passed as +1 (no -Z_C). libcint
//  g1e.c:226-228 (nuc_id<0): fac1 = 2*PI * fac * tau / aij; cr = env+PTR_RINV_ORIG.
//  For a point center tau=1, so x_boys = aij * SQUARE(crij) — identical to the
//  existing scalar nuclear arm. nmax = li+lj (ng={0,0,0,0,0,1,0,1}, no headroom).
//  gout (intor1.c:3627-3638): s += g0[ix+i]*g0[iy+i]*g0[iz+i] over nrys roots.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device single-center 1/r Coulomb potential (`int1e_rinv`, rank 1).
///
/// Single work item. Scratch: `g` (per-root rinv G-tensor), `urys`/`wrys` (Rys
/// roots). `cart_out` is the rank-1 contraction-major accumulator
/// (`(ci*nctr_j+cj)*block_len + cj_idx*nci+ci_idx`). `charge` is the +1 factor.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_rinv_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
    rcx: F,
    rcy: F,
    rcz: F,
    charge: F,
    pie4: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Written and read entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // nmax = li+lj (no headroom; rinv ng = {0,0,0,0,0,1,0,1}).
            let nmax = li + lj;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let out_total = nctr_i * nctr_j * block_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Single center C = rinv origin. crij = C - P.
                    let crijx = rcx - px;
                    let crijy = rcy - py;
                    let crijz = rcz - pz;
                    let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                    if comptime!(nroots == 1u32) {
                        rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 2u32) {
                        rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 3u32) {
                        rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 4u32) {
                        rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else {
                        rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    }

                    // fac1 = 2*PI * charge * fac / zeta  (charge=+1, NO -Z_C).
                    let fac1 = F::new(2.0_f32) * pi_const * charge * fac / zeta;

                    let mut irys: u32 = 0u32;
                    while irys < nrys {
                        let u_n = urys[irys as usize];
                        let w_n = wrys[irys as usize];
                        let tau = u_n / (F::new(1.0_f32) + u_n);
                        let rt = aij2 * (F::new(1.0_f32) - tau);

                        let c00x = (px - rix) + tau * crijx;
                        let c00y = (py - riy) + tau * crijy;
                        let c00z = (pz - riz) + tau * crijz;

                        let mut gi = gbase;
                        while gi < gbase + total_g {
                            g[gi as usize] = F::new(0.0_f32);
                            gi += 1u32;
                        }
                        g[gx as usize] = F::new(1.0_f32);
                        g[gy as usize] = F::new(1.0_f32);
                        g[gz as usize] = fac1 * w_n;

                        one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                        if lj >= 1u32 {
                            one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj);
                            one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj);
                            one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj);
                        }

                        let mut ci = 0u32;
                        while ci < nctr_i {
                            let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                            let mut cj = 0u32;
                            while cj < nctr_j {
                                let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                                let weight = coeff_i_val * coeff_j_val;
                                let base = out_off + (ci * nctr_j + cj) * block_len;

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

                                                let elem = cj_idx * nci + ci_idx;
                                                cart_out[(base + elem) as usize] +=
                                                    weight * g0x * g0y * g0z;

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

                    pj += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_rinv` class.
fn one_e_rinv_g_per_axis(li: usize, lj: usize) -> usize {
    // nmax = li+lj, no headroom.
    (li + lj + 1) * (lj + 1)
}

/// Evaluate every launch group of a batched `int1e_rinv` run (Task 35-D,
/// wave 4). One dispatch per Rys order.
///
/// The rinv origin and its charge are per-*call* scalars, not per-pair: the
/// operator names a single centre, so they stay kernel arguments rather than
/// joining the pair table.
#[allow(clippy::too_many_arguments)]
fn run_1e_rinv_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rc: [f64; 3],
    charge: f64,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    one_electron_rinv_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        rc[0],
                        rc[1],
                        rc[2],
                        charge,
                        PIE4,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        match group.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_rinv` run.
fn dispatch_1e_rinv_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rc: [f64; 3],
    charge: f64,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_rinv_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_rinv_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_rinv_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_rinv_batches::<cubecl_hip::HipRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_rinv_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rc, charge)
        }
    }
}

/// One shell pair through the batched `int1e_rinv` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_rinv_on_backend(
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
    rc: [f64; 3],
    charge: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        1,
        one_e_rinv_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    dispatch_1e_rinv_batches(backend, &handles, std::slice::from_ref(&group), rc, charge)
        .pop()
        .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — `int1e_drinv` (rank 3, Cluster B)
//
//  Gradient of the single-center 1/r potential wrt the rinv center C. libcint
//  intor1.c:3671-3700 (CINTgout1e_int1e_drinv): builds the rinv G-tensor g0
//  (ng = {1,1,0,0,1,1,0,3} → bra i_l+1 AND ket j_l+1 headroom), then
//    g1 = G2E_D_J(g0); g2 = G2E_D_I(g0); g1 += g2
//  and emits the rank-3 per-axis tensor
//    s[0] += g1[ix]*g0[iy]*g0[iz];  s[1] += g0[ix]*g1[iy]*g0[iz];
//    s[2] += g0[ix]*g0[iy]*g1[iz]   (summed over nrys roots).
//  g1 = (D_I + D_J)(g0) is the translational-invariance derivative −(∂_i+∂_j) wrt
//  C — distinct from iprinv's ∂/∂A_bra. nmax = li+lj+2, lj_ext = lj+1. Single
//  center, charge=+1, no atom-sum.
// ─────────────────────────────────────────────────────────────────────────────

/// On-device gradient of the single-center 1/r potential (`int1e_drinv`, rank 3).
///
/// Single work item. Scratch: `g` (rinv G-tensor with bra+1 / ket+1 headroom),
/// `g1` (D_J), `g2` (D_I), `urys`/`wrys`. `cart_out` is the rank-3 component-leading
/// accumulator (`(ci*nctr_j+cj)*total_len + comp*block_len + cj_idx*nci+ci_idx`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_drinv_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    cart_out: &mut Array<F>,
    rcx: F,
    rcy: F,
    rcz: F,
    charge: F,
    pie4: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Written and read entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;
        // All three scratch slabs share a stride and a slot.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // Headroom: g0 spans i..=li+1, j..=lj+1 (ng = {1,1,...}).
            let nmax = li + lj + 2u32;
            let lj_ext = lj + 1u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 3u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;

                    let crijx = rcx - px;
                    let crijy = rcy - py;
                    let crijz = rcz - pz;
                    let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

                    if comptime!(nroots == 1u32) {
                        rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 2u32) {
                        rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 3u32) {
                        rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else if comptime!(nroots == 4u32) {
                        rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    } else {
                        rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                    }

                    let fac1 = F::new(2.0_f32) * pi_const * charge * fac / zeta;

                    let mut irys: u32 = 0u32;
                    while irys < nrys {
                        let u_n = urys[irys as usize];
                        let w_n = wrys[irys as usize];
                        let tau = u_n / (F::new(1.0_f32) + u_n);
                        let rt = aij2 * (F::new(1.0_f32) - tau);

                        let c00x = (px - rix) + tau * crijx;
                        let c00y = (py - riy) + tau * crijy;
                        let c00z = (pz - riz) + tau * crijz;

                        // Per-root rinv G-tensor in `g` (bra+1 / ket+1 headroom).
                        let mut gi = gbase;
                        while gi < gbase + total_g {
                            g[gi as usize] = F::new(0.0_f32);
                            gi += 1u32;
                        }
                        g[gx as usize] = F::new(1.0_f32);
                        g[gy as usize] = F::new(1.0_f32);
                        g[gz as usize] = fac1 * w_n;

                        one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                        one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                        one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                        one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                        // g1 = D_J(g0), g2 = D_I(g0), then g1 += g2 over the FINAL
                        // i∈0..=li, j∈0..=lj range. D_J reads j+1 (lj_ext headroom),
                        // D_I reads i+1 (li+1 headroom). libcint intor1.c:3678-3680.
                        let mut zi = gbase;
                        while zi < gbase + total_g {
                            g1[zi as usize] = F::new(0.0_f32);
                            g2[zi as usize] = F::new(0.0_f32);
                            zi += 1u32;
                        }
                        d_j_1e_into::<F>(g1, g, gbase, g_per_axis, dj, lj, li, aj2);
                        d_i_1e_into::<F>(g2, g, gbase, g_per_axis, dj, lj, li, ai2);
                        let mut si = gbase;
                        while si < gbase + total_g {
                            g1[si as usize] = g1[si as usize] + g2[si as usize];
                            si += 1u32;
                        }

                        let mut ci = 0u32;
                        while ci < nctr_i {
                            let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                            let mut cj = 0u32;
                            while cj < nctr_j {
                                let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                                let weight = coeff_i_val * coeff_j_val;
                                let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                                cart_out[(base + block_len + elem) as usize] +=
                                                    weight * g0x * g1y * g0z;
                                                cart_out
                                                    [(base + 2u32 * block_len + elem) as usize] +=
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

                    pj += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_drinv` class.
fn one_e_drinv_g_per_axis(li: usize, lj: usize) -> usize {
    // Headroom: nmax = li+lj+2, lj_ext = lj+1.
    (li + lj + 3) * (lj + 2)
}

/// Evaluate every launch group of a batched `int1e_drinv` run (Task 35-D,
/// wave 4). One dispatch per Rys order; the rinv origin and charge stay
/// per-call scalars, as for `int1e_rinv`.
#[allow(clippy::too_many_arguments)]
fn run_1e_drinv_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rc: [f64; 3],
    charge: f64,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    one_electron_drinv_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g2_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        rc[0],
                        rc[1],
                        rc[2],
                        charge,
                        PIE4,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        match group.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_drinv` run.
fn dispatch_1e_drinv_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    rc: [f64; 3],
    charge: f64,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_drinv_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_drinv_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_drinv_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_drinv_batches::<cubecl_hip::HipRuntime>(client, basis, groups, rc, charge)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_drinv_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups, rc, charge)
        }
    }
}

/// One shell pair through the batched `int1e_drinv` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_drinv_on_backend(
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
    rc: [f64; 3],
    charge: f64,
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
) -> Vec<f64> {
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        3,
        one_e_drinv_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    dispatch_1e_drinv_batches(backend, &handles, std::slice::from_ref(&group), rc, charge)
        .pop()
        .unwrap_or_default()
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
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_nuc_grad_both_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    g3: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    pi_const: F,
    norig: u32,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Written and read entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;
        // All four scratch slabs share a stride and a slot, because the
        // contraction reads them at identical relative offsets.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // Both-side headroom: g0 spans i..=li+1, j..=lj+1.
            let nmax = li + lj + 2u32;
            let lj_ext = lj + 1u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 9u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let li1 = li + 1u32;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;

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
                            rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 2u32) {
                            rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 3u32) {
                            rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 4u32) {
                            rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else {
                            rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        }

                        let fac1 = F::new(2.0_f32) * pi_const * charge_factor * fac / zeta;

                        let mut irys: u32 = 0u32;
                        while irys < nrys {
                            let u_n = urys[irys as usize];
                            let w_n = wrys[irys as usize];
                            let tau = u_n / (F::new(1.0_f32) + u_n);
                            let rt = aij2 * (F::new(1.0_f32) - tau);

                            let c00x = (px - rix) + tau * crijx;
                            let c00y = (py - riy) + tau * crijy;
                            let c00z = (pz - riz) + tau * crijz;

                            // Per-root nuclear G-tensor in `g`.
                            let mut gi = gbase;
                            while gi < gbase + total_g {
                                g[gi as usize] = F::new(0.0_f32);
                                gi += 1u32;
                            }
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
                            g[gz as usize] = fac1 * w_n;

                            one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                            one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                            one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                            one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                            one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                            one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);

                            // Zero + build the three both-side derivatives.
                            let mut zi = gbase;
                            while zi < gbase + total_g {
                                g1[zi as usize] = F::new(0.0_f32);
                                g2[zi as usize] = F::new(0.0_f32);
                                g3[zi as usize] = F::new(0.0_f32);
                                zi += 1u32;
                            }
                            d_j_1e_into::<F>(g1, g, gbase, g_per_axis, dj, lj, li1, aj2);
                            d_i_1e_into::<F>(g2, g, gbase, g_per_axis, dj, lj, li, ai2);
                            d_i_1e_into::<F>(g3, g1, gbase, g_per_axis, dj, lj, li, ai2);

                            // Accumulate this root's 9-component contribution.
                            let mut ci = 0u32;
                            while ci < nctr_i {
                                let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                                let mut cj = 0u32;
                                while cj < nctr_j {
                                    let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                                    let weight = coeff_i_val * coeff_j_val;
                                    let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                                    cart_out[(base + 2u32 * block_len + elem)
                                                        as usize] += weight * s2;
                                                    cart_out[(base + 3u32 * block_len + elem)
                                                        as usize] += weight * s3;
                                                    cart_out[(base + 4u32 * block_len + elem)
                                                        as usize] += weight * s4;
                                                    cart_out[(base + 5u32 * block_len + elem)
                                                        as usize] += weight * s5;
                                                    cart_out[(base + 6u32 * block_len + elem)
                                                        as usize] += weight * s6;
                                                    cart_out[(base + 7u32 * block_len + elem)
                                                        as usize] += weight * s7;
                                                    cart_out[(base + 8u32 * block_len + elem)
                                                        as usize] += weight * s8;

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

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_ipnucip` class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn one_e_nuc_grad_both_g_per_axis(li: usize, lj: usize) -> usize {
    // Both-side headroom: nmax = li+lj+2, lj_ext = lj+1.
    (li + lj + 3) * (lj + 2)
}

/// Evaluate every launch group of a batched `int1e_ipnucip` run (Task 35-D,
/// wave 3). One dispatch per Rys order — the kernel's only comptime shape.
fn run_1e_nuc_grad_both_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    // CubeCL `Array`s must be non-empty; a molecule with no origins never
    // reads these.
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
    let norig = origin_charges.len() as u32;

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Four scratch slabs, all `3 * g_per_axis`, sharing a stride and a slot.
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g3_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    one_electron_nuc_grad_both_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(coords_h.clone(), coords_src.len()),
                        ArrayArg::from_raw_parts(charges_h.clone(), charges_src.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g2_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g3_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        PIE4,
                        std::f64::consts::PI,
                        norig,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        match group.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipnucip` run.
fn dispatch_1e_nuc_grad_both_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_nuc_grad_both_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_nuc_grad_both_batches::<cubecl_wgpu::WgpuRuntime>(
                client,
                basis,
                groups,
                origin_coords,
                origin_charges,
            )
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_nuc_grad_both_batches::<cubecl_cuda::CudaRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_nuc_grad_both_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_nuc_grad_both_batches::<cubecl_wgpu::WgpuRuntime>(
                client,
                basis,
                groups,
                origin_coords,
                origin_charges,
            )
        }
    }
}

/// One shell pair through the batched `int1e_ipnucip` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        9,
        one_e_nuc_grad_both_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    dispatch_1e_nuc_grad_both_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        origin_coords,
        origin_charges,
    )
    .pop()
    .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — `int1e_ipipnuc` / `int1e_ipiprinv`
//  (∇²bra NUCLEAR / RINV, rank 9)
//
//  Faithful port of libcint `hess.c` `CINTgout1e_int1e_ipipnuc` (lines 520-568)
//  and `CINTgout1e_int1e_ipiprinv` (lines 680-728) — identical s-tensor + gout:
//    g1 = G2E_D_I(g0, i_l+1);  g2 = G2E_D_I(g0, i_l+0);  g3 = G2E_D_I(g1, i_l+0)
//    s/gout permutation = [s0,s3,s6,s1,s4,s7,s2,s5,s8]  (column-major 3×3)
//  Base g0 is the per-root nuclear Rys G-tensor (mirrors the both-side nuclear
//  kernel) but bra-only headroom: nmax = li+lj+2, lj_ext = lj (NO ket +2).
//  ipipnuc sums over ALL nuclei (charge -Z_C); ipiprinv is a single origin
//  (charge +1) — both flow through `origin_charges`.
//  nroots = (li+lj+2)/2 + 1 (the +2 bra headroom raises the VRR ceiling).
// ─────────────────────────────────────────────────────────────────────────────

/// On-device bra-only rank-9 nuclear/rinv Hessian (`int1e_ipipnuc`/`ipiprinv`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_nuc_gradgrad_bra_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    origin_coords: &Array<F>,
    origin_charges: &Array<F>,
    g: &mut Array<F>,
    g1: &mut Array<F>,
    g2: &mut Array<F>,
    g3: &mut Array<F>,
    cart_out: &mut Array<F>,
    pie4: F,
    pi_const: F,
    norig: u32,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    // Written and read entirely inside the `lane == 0` region, so per-unit
    // private storage rather than buffers.
    let mut urys = Array::<F>::new(5usize);
    let mut wrys = Array::<F>::new(5usize);

    if lane == 0u32 {
        let nrys = nroots;
        // All four scratch slabs share a stride and a slot, because the
        // contraction reads them at identical relative offsets.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // Bra-only headroom: g0 spans i..=li+2, j..=lj (no ket extension).
            let nmax = li + lj + 2u32;
            let lj_ext = lj;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 9u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let li1 = li + 1u32;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    let ai2 = F::new(-2.0_f32) * ai;

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
                            rys_root1::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 2u32) {
                            rys_root2::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 3u32) {
                            rys_root3::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else if comptime!(nroots == 4u32) {
                            rys_root4::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        } else {
                            rys_root5::<F>(x_boys, &mut urys, &mut wrys, pie4);
                        }

                        let fac1 = F::new(2.0_f32) * pi_const * charge_factor * fac / zeta;

                        let mut irys: u32 = 0u32;
                        while irys < nrys {
                            let u_n = urys[irys as usize];
                            let w_n = wrys[irys as usize];
                            let tau = u_n / (F::new(1.0_f32) + u_n);
                            let rt = aij2 * (F::new(1.0_f32) - tau);

                            let c00x = (px - rix) + tau * crijx;
                            let c00y = (py - riy) + tau * crijy;
                            let c00z = (pz - riz) + tau * crijz;

                            let mut gi = gbase;
                            while gi < gbase + total_g {
                                g[gi as usize] = F::new(0.0_f32);
                                gi += 1u32;
                            }
                            g[gx as usize] = F::new(1.0_f32);
                            g[gy as usize] = F::new(1.0_f32);
                            g[gz as usize] = fac1 * w_n;

                            one_electron_vrr2e_axis::<F>(g, gx, c00x, rt, nmax);
                            one_electron_vrr2e_axis::<F>(g, gy, c00y, rt, nmax);
                            one_electron_vrr2e_axis::<F>(g, gz, c00z, rt, nmax);
                            if lj_ext >= 1u32 {
                                one_electron_hrr_axis::<F>(g, gx, rirjx, dj, nmax, lj_ext);
                                one_electron_hrr_axis::<F>(g, gy, rirjy, dj, nmax, lj_ext);
                                one_electron_hrr_axis::<F>(g, gz, rirjz, dj, nmax, lj_ext);
                            }

                            let mut zi = gbase;
                            while zi < gbase + total_g {
                                g1[zi as usize] = F::new(0.0_f32);
                                g2[zi as usize] = F::new(0.0_f32);
                                g3[zi as usize] = F::new(0.0_f32);
                                zi += 1u32;
                            }
                            // Bra-only ∇²: g1 = D_I(g0,i+1), g2 = D_I(g0,i), g3 = D_I(g1,i).
                            d_i_1e_into::<F>(g1, g, gbase, g_per_axis, dj, lj, li1, ai2);
                            d_i_1e_into::<F>(g2, g, gbase, g_per_axis, dj, lj, li, ai2);
                            d_i_1e_into::<F>(g3, g1, gbase, g_per_axis, dj, lj, li, ai2);

                            gradgrad_bra_contract::<F>(
                                g, g1, g2, g3, cart_out, coeffs, coff_i, coff_j, out_off, gx, gy,
                                gz, dj, nci, block_len, total_len, li, lj, nctr_i, nctr_j, pi, pj,
                            );

                            irys += 1u32;
                        }
                        orig += 1u32;
                    }

                    pj += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// `g_per_axis` for one `int1e_ipipnuc` / `int1e_ipiprinv` class.
///
/// Mirrors the kernel's own sizing; the two must agree or the slab is too small.
fn one_e_nuc_gradgrad_bra_g_per_axis(li: usize, lj: usize) -> usize {
    // Bra-only headroom: nmax = li+lj+2, lj_ext = lj.
    (li + lj + 3) * (lj + 1)
}

/// Evaluate every launch group of a batched `int1e_ipipnuc` / `int1e_ipiprinv`
/// run (Task 35-D, wave 3). One dispatch per Rys order.
fn run_1e_nuc_gradgrad_bra_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    // CubeCL `Array`s must be non-empty; a molecule with no origins never
    // reads these.
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
    let norig = origin_charges.len() as u32;

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Four scratch slabs, all `3 * g_per_axis`, sharing a stride and a slot.
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g1_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g2_h = client.empty(g_len * std::mem::size_of::<f64>());
        let g3_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, `norig`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($nr:expr) => {
                unsafe {
                    one_electron_nuc_gradgrad_bra_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(coords_h.clone(), coords_src.len()),
                        ArrayArg::from_raw_parts(charges_h.clone(), charges_src.len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g1_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g2_h.clone(), g_len),
                        ArrayArg::from_raw_parts(g3_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        PIE4,
                        std::f64::consts::PI,
                        norig,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        match group.nroots {
            1 => launch_with!(1u32),
            2 => launch_with!(2u32),
            3 => launch_with!(3u32),
            4 => launch_with!(4u32),
            _ => launch_with!(5u32),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipipnuc` / `int1e_ipiprinv` run.
fn dispatch_1e_nuc_gradgrad_bra_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    origin_coords: &[f64],
    origin_charges: &[f64],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_nuc_gradgrad_bra_batches::<cubecl::cpu::CpuRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_nuc_gradgrad_bra_batches::<
            cubecl_wgpu::WgpuRuntime,
        >(
            client, basis, groups, origin_coords, origin_charges
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_nuc_gradgrad_bra_batches::<cubecl_cuda::CudaRuntime>(
                client,
                basis,
                groups,
                origin_coords,
                origin_charges,
            )
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_nuc_gradgrad_bra_batches::<cubecl_hip::HipRuntime>(
            client,
            basis,
            groups,
            origin_coords,
            origin_charges,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_nuc_gradgrad_bra_batches::<
            cubecl_wgpu::WgpuRuntime,
        >(
            client, basis, groups, origin_coords, origin_charges
        ),
    }
}

/// One shell pair through the batched `int1e_ipipnuc` / `int1e_ipiprinv` path —
/// a one-pair group through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_nuc_gradgrad_bra_on_backend(
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        9,
        one_e_nuc_gradgrad_bra_g_per_axis(li as usize, lj as usize),
        nroots,
    );
    dispatch_1e_nuc_gradgrad_bra_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        origin_coords,
        origin_charges,
    )
    .pop()
    .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]` — `int1e_ipipkin` (∇²bra KINETIC, rank 9)
//
//  Faithful port of libcint `hess.c` `CINTgout1e_int1e_ipipkin` (lines 170-310).
//  16 g-tensors (g0..g15) live in one flat `gbuf` (tensor t at offset t*total_g):
//    g1 = D_J(g0, i+2, j+0)   g2 = D_J(g0, i+2, j+1)   g3 = D_J(g2, i+2, j+0)
//    g4 = D_I(g0, i+1)        g5 = D_I(g1, i+1)        g6 = D_I(g2, i+1)
//    g7 = D_I(g3, i+1)        g8 = D_I(g0, i+0)        g9 = D_I(g1, i+0)
//    g10 = D_I(g2, i+0)       g11 = D_I(g3, i+0)       g12 = D_I(g4, i+0)
//    g13 = D_I(g5, i+0)       g14 = D_I(g6, i+0)       g15 = D_I(g7, i+0)
//  The 81-term s[] table and the 9-component gout = -(s_a + s_b + s_c) triples
//  are copied verbatim (the kinetic -½∇²_j folded into the gout sign). Headroom
//  ng={2,2,...}: nmax = li+lj+4, lj_ext = lj+2.
// ─────────────────────────────────────────────────────────────────────────────

/// `#[cube]` helper: bra-direction nabla into a flat 16-tensor buffer, reading
/// tensor `src_t` and writing tensor `dst_t` (disjoint regions of `gbuf`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn d_i_1e_flat<F: Float + CubeElement>(
    gbuf: &mut Array<F>,
    gbase: u32,
    dst_t: u32,
    src_t: u32,
    total_g: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    ai2: F,
) {
    let dbase = gbase + dst_t * total_g;
    let sbase = gbase + src_t * total_g;
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            gbuf[(dbase + off + jbase) as usize] =
                ai2 * gbuf[(sbase + off + jbase + 1u32) as usize];
            let mut ii = 1u32;
            while ii <= imax {
                gbuf[(dbase + off + jbase + ii) as usize] = F::cast_from(ii)
                    * gbuf[(sbase + off + jbase + ii - 1u32) as usize]
                    + ai2 * gbuf[(sbase + off + jbase + ii + 1u32) as usize];
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `#[cube]` helper: ket-direction nabla into a flat 16-tensor buffer.
#[cube]
#[allow(clippy::too_many_arguments)]
fn d_j_1e_flat<F: Float + CubeElement>(
    gbuf: &mut Array<F>,
    gbase: u32,
    dst_t: u32,
    src_t: u32,
    total_g: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    aj2: F,
) {
    let dbase = gbase + dst_t * total_g;
    let sbase = gbase + src_t * total_g;
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            let jhi = (jn + 1u32) * dj;
            let mut ii = 0u32;
            while ii <= imax {
                let mut val = aj2 * gbuf[(sbase + off + jhi + ii) as usize];
                if jn >= 1u32 {
                    val = F::cast_from(jn) * gbuf[(sbase + off + (jn - 1u32) * dj + ii) as usize]
                        + val;
                }
                gbuf[(dbase + off + jbase + ii) as usize] = val;
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `#[cube]` helper: bra-direction position multiply `dst = R0I(src)` into a flat
/// tensor buffer (G2E_R0I / G1E_R0I: `f[i] = g[i+1] + ri*g[i]`, i-direction).
#[cube]
#[allow(clippy::too_many_arguments)]
fn r0i_1e_flat<F: Float + CubeElement>(
    gbuf: &mut Array<F>,
    gbase: u32,
    dst_t: u32,
    src_t: u32,
    total_g: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    rix: F,
    riy: F,
    riz: F,
) {
    let dbase = gbase + dst_t * total_g;
    let sbase = gbase + src_t * total_g;
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut ri = rix;
        if axisn == 1u32 {
            ri = riy;
        } else if axisn == 2u32 {
            ri = riz;
        }
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            let mut ii = 0u32;
            while ii <= imax {
                gbuf[(dbase + off + jbase + ii) as usize] = gbuf
                    [(sbase + off + jbase + ii + 1u32) as usize]
                    + ri * gbuf[(sbase + off + jbase + ii) as usize];
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `#[cube]` helper: ket-direction position multiply `dst = RCJ(src)` into a flat
/// tensor buffer (G2E_RCJ / G1E_RCJ: `f[i] = g[i+dj] + drj*g[i]`, j-direction).
#[cube]
#[allow(clippy::too_many_arguments)]
fn rcj_1e_flat<F: Float + CubeElement>(
    gbuf: &mut Array<F>,
    gbase: u32,
    dst_t: u32,
    src_t: u32,
    total_g: u32,
    g_per_axis: u32,
    dj: u32,
    jmax: u32,
    imax: u32,
    drjx: F,
    drjy: F,
    drjz: F,
) {
    let dbase = gbase + dst_t * total_g;
    let sbase = gbase + src_t * total_g;
    let mut axisn = 0u32;
    while axisn < 3u32 {
        let off = axisn * g_per_axis;
        let mut drj = drjx;
        if axisn == 1u32 {
            drj = drjy;
        } else if axisn == 2u32 {
            drj = drjz;
        }
        let mut jn = 0u32;
        while jn <= jmax {
            let jbase = jn * dj;
            let jhi = (jn + 1u32) * dj;
            let mut ii = 0u32;
            while ii <= imax {
                gbuf[(dbase + off + jbase + ii) as usize] = gbuf[(sbase + off + jhi + ii) as usize]
                    + drj * gbuf[(sbase + off + jbase + ii) as usize];
                ii += 1u32;
            }
            jn += 1u32;
        }
        axisn += 1u32;
    }
}

/// `#[cube]` helper: in-place add of one flat tensor into another (`dst += src`),
/// over the full tensor span (used for the ia01p/a01gp/a11part `g2 += g3` step).
#[cube]
fn add_tensor_flat<F: Float + CubeElement>(
    gbuf: &mut Array<F>,
    gbase: u32,
    dst_t: u32,
    src_t: u32,
    total_g: u32,
) {
    let dbase = gbase + dst_t * total_g;
    let sbase = gbase + src_t * total_g;
    let mut ix = 0u32;
    while ix < total_g {
        gbuf[(dbase + ix) as usize] = gbuf[(dbase + ix) as usize] + gbuf[(sbase + ix) as usize];
        ix += 1u32;
    }
}

/// On-device bra-only rank-9 kinetic Hessian (`int1e_ipipkin`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_gradgrad_bra_kin_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    gbuf: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] per_unit: u32,
) {
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        // One slab per slot, holding all sixteen tensors back to back: the
        // contraction reads them at `t * total_g` from the slot base, so the
        // stride must cover `16 * 3 * g_per_axis` of the widest class.
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // ng={2,2,...}: bra +2 AND ket +2 headroom.
            let nmax = li + lj + 4u32;
            let lj_ext = lj + 2u32;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = 9u32 * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let li1 = li + 1u32;
            let li2 = li + 2u32;
            let ljp1 = lj + 1u32;

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Zero the whole 16-tensor buffer.
                    let buf_len = 16u32 * total_g;
                    let mut zi = gbase;
                    while zi < gbase + buf_len {
                        gbuf[zi as usize] = F::new(0.0_f32);
                        zi += 1u32;
                    }

                    // Build g0 (overlap base) in tensor slot 0.
                    gbuf[gx as usize] = F::new(1.0_f32);
                    gbuf[gy as usize] = F::new(1.0_f32);
                    gbuf[gz as usize] = fac * sqrtpi * pi_const / (zeta * F::sqrt(zeta));
                    one_electron_vrr_axis::<F>(gbuf, gx, px - rix, aij2, nmax);
                    one_electron_vrr_axis::<F>(gbuf, gy, py - riy, aij2, nmax);
                    one_electron_vrr_axis::<F>(gbuf, gz, pz - riz, aij2, nmax);
                    one_electron_hrr_axis::<F>(gbuf, gx, rirjx, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(gbuf, gy, rirjy, dj, nmax, lj_ext);
                    one_electron_hrr_axis::<F>(gbuf, gz, rirjz, dj, nmax, lj_ext);

                    let ai2 = F::new(-2.0_f32) * ai;
                    let aj2 = F::new(-2.0_f32) * aj;

                    // Kinetic ket-derivative chain (imax = li+2):
                    d_j_1e_flat::<F>(
                        gbuf, gbase, 1u32, 0u32, total_g, g_per_axis, dj, lj, li2, aj2,
                    );
                    d_j_1e_flat::<F>(
                        gbuf, gbase, 2u32, 0u32, total_g, g_per_axis, dj, ljp1, li2, aj2,
                    );
                    d_j_1e_flat::<F>(
                        gbuf, gbase, 3u32, 2u32, total_g, g_per_axis, dj, lj, li2, aj2,
                    );
                    // First bra-derivative (imax = li+1) of g0..g3 → g4..g7:
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 4u32, 0u32, total_g, g_per_axis, dj, lj, li1, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 5u32, 1u32, total_g, g_per_axis, dj, lj, li1, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 6u32, 2u32, total_g, g_per_axis, dj, lj, li1, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 7u32, 3u32, total_g, g_per_axis, dj, lj, li1, ai2,
                    );
                    // Bra-derivative (imax = li) of g0..g3 → g8..g11:
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 8u32, 0u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 9u32, 1u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 10u32, 2u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 11u32, 3u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    // Second bra-derivative (imax = li) of g4..g7 → g12..g15:
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 12u32, 4u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 13u32, 5u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 14u32, 6u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );
                    d_i_1e_flat::<F>(
                        gbuf, gbase, 15u32, 7u32, total_g, g_per_axis, dj, lj, li, ai2,
                    );

                    let t4 = 4u32 * total_g;
                    let t7 = 7u32 * total_g;
                    let t8 = 8u32 * total_g;
                    let t11 = 11u32 * total_g;
                    let t12 = 12u32 * total_g;
                    let t15 = 15u32 * total_g;

                    let mut ci = 0u32;
                    while ci < nctr_i {
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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

                                            // Per-axis g-values for the tensors the
                                            // gout s[] table references (tensors 1, 2,
                                            // 5, 6, 9, 10, 13, 14 are intermediates only).
                                            let g0x = gbuf[(gx + nx) as usize];
                                            let g0y = gbuf[(gy + ny) as usize];
                                            let g0z = gbuf[(gz + nz) as usize];
                                            let g3x = gbuf[(3u32 * total_g + gx + nx) as usize];
                                            let g3y = gbuf[(3u32 * total_g + gy + ny) as usize];
                                            let g3z = gbuf[(3u32 * total_g + gz + nz) as usize];
                                            let g4x = gbuf[(t4 + gx + nx) as usize];
                                            let g4y = gbuf[(t4 + gy + ny) as usize];
                                            let g4z = gbuf[(t4 + gz + nz) as usize];
                                            let g7x = gbuf[(t7 + gx + nx) as usize];
                                            let g7y = gbuf[(t7 + gy + ny) as usize];
                                            let g7z = gbuf[(t7 + gz + nz) as usize];
                                            let g8x = gbuf[(t8 + gx + nx) as usize];
                                            let g8y = gbuf[(t8 + gy + ny) as usize];
                                            let g8z = gbuf[(t8 + gz + nz) as usize];
                                            let g11x = gbuf[(t11 + gx + nx) as usize];
                                            let g11y = gbuf[(t11 + gy + ny) as usize];
                                            let g11z = gbuf[(t11 + gz + nz) as usize];
                                            let g12x = gbuf[(t12 + gx + nx) as usize];
                                            let g12y = gbuf[(t12 + gy + ny) as usize];
                                            let g12z = gbuf[(t12 + gz + nz) as usize];
                                            let g15x = gbuf[(t15 + gx + nx) as usize];
                                            let g15y = gbuf[(t15 + gy + ny) as usize];
                                            let g15z = gbuf[(t15 + gz + nz) as usize];

                                            // s[] table verbatim from hess.c ipipkin.
                                            let s0 = g15x * g0y * g0z;
                                            let s4 = g12x * g3y * g0z;
                                            let s8 = g12x * g0y * g3z;
                                            let s9 = g11x * g4y * g0z;
                                            let s13 = g8x * g7y * g0z;
                                            let s17 = g8x * g4y * g3z;
                                            let s18 = g11x * g0y * g4z;
                                            let s22 = g8x * g3y * g4z;
                                            let s26 = g8x * g0y * g7z;
                                            let s27 = g7x * g8y * g0z;
                                            let s31 = g4x * g11y * g0z;
                                            let s35 = g4x * g8y * g3z;
                                            let s36 = g3x * g12y * g0z;
                                            let s40 = g0x * g15y * g0z;
                                            let s44 = g0x * g12y * g3z;
                                            let s45 = g3x * g8y * g4z;
                                            let s49 = g0x * g11y * g4z;
                                            let s53 = g0x * g8y * g7z;
                                            let s54 = g7x * g0y * g8z;
                                            let s58 = g4x * g3y * g8z;
                                            let s62 = g4x * g0y * g11z;
                                            let s63 = g3x * g4y * g8z;
                                            let s67 = g0x * g7y * g8z;
                                            let s71 = g0x * g4y * g11z;
                                            let s72 = g3x * g0y * g12z;
                                            let s76 = g0x * g3y * g12z;
                                            let s80 = g0x * g0y * g15z;

                                            // gout = -½·(triple): libcint's ipipkin gout
                                            // emits -(s) where s carries the full ∇²_j
                                            // (4aj²·g[j+2] - 2aj(2j+1)·g[j] + j(j-1)·g[j-2]);
                                            // the kinetic operator is T = -½∇²_j, so the
                                            // ½ is folded here (cintx contracts s directly,
                                            // unlike libcint which scales in CINT1e_drv).
                                            let half = F::new(-0.5_f32);
                                            let go0 = half * (s0 + s4 + s8);
                                            let go1 = half * (s27 + s31 + s35);
                                            let go2 = half * (s54 + s58 + s62);
                                            let go3 = half * (s9 + s13 + s17);
                                            let go4 = half * (s36 + s40 + s44);
                                            let go5 = half * (s63 + s67 + s71);
                                            let go6 = half * (s18 + s22 + s26);
                                            let go7 = half * (s45 + s49 + s53);
                                            let go8 = half * (s72 + s76 + s80);

                                            let elem = cj_idx * nci + ci_idx;
                                            cart_out[(base + elem) as usize] += weight * go0;
                                            cart_out[(base + block_len + elem) as usize] +=
                                                weight * go1;
                                            cart_out[(base + 2u32 * block_len + elem) as usize] +=
                                                weight * go2;
                                            cart_out[(base + 3u32 * block_len + elem) as usize] +=
                                                weight * go3;
                                            cart_out[(base + 4u32 * block_len + elem) as usize] +=
                                                weight * go4;
                                            cart_out[(base + 5u32 * block_len + elem) as usize] +=
                                                weight * go5;
                                            cart_out[(base + 6u32 * block_len + elem) as usize] +=
                                                weight * go6;
                                            cart_out[(base + 7u32 * block_len + elem) as usize] +=
                                                weight * go7;
                                            cart_out[(base + 8u32 * block_len + elem) as usize] +=
                                                weight * go8;

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

            qi += qi_step;
        }
    }
}

/// Per-slot slab units for one `int1e_ipipkin` class.
///
/// This kernel keeps **sixteen** tensors in one buffer rather than one per
/// scratch argument, so the slot stride has to cover all of them. Expressing
/// that as `16 * g_per_axis` lets [`one_e_g_slab_stride`] size it the same way
/// every other family's is sized — the `3 *` for the three axes is already in
/// that function.
fn one_e_gradgrad_bra_kin_slab_units(li: usize, lj: usize) -> usize {
    // ng={2,2,...}: nmax = li+lj+4, lj_ext = lj+2.
    16 * (li + lj + 5) * (lj + 3)
}

/// Evaluate every launch group of a batched `int1e_ipipkin` run (Task 35-D,
/// wave 3). One dispatch per group; the kernel has no comptime shape parameter.
fn run_1e_gradgrad_bra_kin_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for group in groups {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        unsafe {
            one_electron_gradgrad_bra_kin_kernel::launch_unchecked::<f64, R>(
                client,
                crate::plane::cube_count_1d(n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                ArrayArg::from_raw_parts(g_h.clone(), g_len),
                ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                SQRTPI,
                std::f64::consts::PI,
                n_pairs as u32,
                n_cubes,
                g_stride as u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched `int1e_ipipkin` run.
fn dispatch_1e_gradgrad_bra_kin_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_1e_gradgrad_bra_kin_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_1e_gradgrad_bra_kin_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_1e_gradgrad_bra_kin_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_1e_gradgrad_bra_kin_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_1e_gradgrad_bra_kin_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell pair through the batched `int1e_ipipkin` path — a one-pair group
/// through the same kernel a wide batch uses (Task 35-D).
#[allow(clippy::too_many_arguments)]
fn run_1e_gradgrad_bra_kin_on_backend(
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
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        9,
        one_e_gradgrad_bra_kin_slab_units(li as usize, lj as usize),
        1,
    );
    dispatch_1e_gradgrad_bra_kin_batches(backend, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
}

/// `u32` shape scalars per class row of the device shape table: `li, lj`.
pub(crate) const ONE_E_SHAPE_STRIDE: usize = 2;

/// One dispatch: every shell pair of the same Rys order (Task 35-M2).
///
/// `one_electron_scalar_kernel` specializes on `op_kind` — fixed by the
/// caller's operator — and `nroots`. `li` and `lj` are runtime scalars, so a
/// launch class is a Rys order rather than an `(li,lj)` tuple. Each pair names
/// its class in the fourth column of its table row. Overlap and kinetic use
/// `nroots = 1` throughout, so both collapse to a single dispatch.
#[derive(Clone, Debug)]
pub struct OneELaunchGroup {
    /// Rys order — with `op_kind`, the kernel's comptime specialization.
    pub nroots: u32,
    /// [`ONE_E_SHAPE_STRIDE`] `u32` per merged class: `li, lj`.
    pub class_shape: Vec<u32>,
    /// `[si, sj, out_off, class]` per pair.
    pub pairs: Vec<u32>,
    /// Total Cartesian output elements across this group's pairs.
    pub out_len: usize,
    /// Widest per-slot `g_per_axis` in the group (`op_kind`-dependent).
    pub max_g_per_axis: usize,
    /// Widest Cartesian contraction block — the cooperative parallel width.
    pub max_block_len: u32,
}

impl OneELaunchGroup {
    /// An empty group of Rys order `nroots`.
    #[must_use]
    pub fn new(nroots: u32) -> Self {
        Self {
            nroots,
            class_shape: Vec::new(),
            pairs: Vec::new(),
            out_len: 0,
            max_g_per_axis: 0,
            max_block_len: 0,
        }
    }

    /// Append a class and return the index its pair rows carry.
    ///
    /// `kinetic` selects the `op_kind == 1` G-tensor headroom, matching the
    /// kernel's comptime branch — the slab must be sized for the same shape the
    /// kernel will index.
    pub fn push_class(&mut self, li: u32, lj: u32, kinetic: bool) -> u32 {
        let index = (self.class_shape.len() / ONE_E_SHAPE_STRIDE) as u32;
        self.class_shape.extend_from_slice(&[li, lj]);
        let (li_u, lj_u) = (li as usize, lj as usize);
        let (nmax, lj_ext) = if kinetic {
            (li_u + lj_u + 2, lj_u + 2)
        } else {
            (li_u + lj_u, lj_u)
        };
        self.max_g_per_axis = self.max_g_per_axis.max((nmax + 1) * (lj_ext + 1));
        index
    }

    /// Number of angular-momentum classes merged into this dispatch.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_shape.len() / ONE_E_SHAPE_STRIDE
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
    }
}

/// Stride, in `f64` elements, between one slot's 1e G slab and the next.
///
/// Padded to a 64-byte cache line so that concurrent slots — OS threads, in the
/// per-unit decomposition — never share a line while writing the G tensor.
pub(crate) fn one_e_g_slab_stride(g_per_axis: usize) -> usize {
    const LINE: usize = 8;
    (3 * g_per_axis).div_ceil(LINE) * LINE
}

/// Does this backend want the one-pair-per-unit decomposition?
///
/// Same reasoning, and the same override knob, as
/// `two_electron::two_e_per_unit`: on the CubeCL CPU runtime a unit is an OS
/// thread and `cube_count` lowers to a sequential loop, so the cube is the only
/// parallelism axis; on GPU backends the grid is.
pub(crate) fn one_e_per_unit<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let pinned = *OVERRIDE.get_or_init(|| {
        std::env::var("CINTX_1E_PER_UNIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    });
    match pinned {
        Some(value) => value != 0,
        None => !crate::plane::has_planes(client),
    }
}

/// Launch geometry for one 1e class: `(cube_count, cube_dim, n_slots)`.
pub(crate) fn one_e_launch_geometry<R: Runtime>(
    client: &ComputeClient<R>,
    n_pairs: usize,
    g_per_axis: usize,
    block_len: u32,
) -> (u32, CubeDim, usize) {
    /// Ceiling on the per-launch G-tensor scratch slab.
    const MAX_BATCH_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

    let per_slab = one_e_g_slab_stride(g_per_axis) * std::mem::size_of::<f64>();
    let by_memory = (MAX_BATCH_SCRATCH_BYTES / per_slab.max(1)).max(1);

    if one_e_per_unit::<R>(client) {
        let units = crate::plane::per_unit_width(
            client,
            n_pairs,
            crate::plane::MIN_ITEMS_PER_UNIT_PAIR,
            by_memory,
        );
        return (1, CubeDim::new_1d(units), units as usize);
    }
    let cubes = crate::plane::grid_cube_count(client, n_pairs.min(by_memory));
    (
        cubes,
        crate::plane::cooperative_cube_dim(client, block_len),
        cubes as usize,
    )
}

/// Evaluate every class of a batched 1e run: one dispatch and one readback per
/// class, one basis upload for the whole run.
///
/// Returns one Cartesian buffer per class, in `classes` order.
#[allow(clippy::too_many_arguments)]
fn run_1e_batches<R: Runtime>(
    client: &ComputeClient<R>,
    op_kind: u32,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneELaunchGroup],
    atom_coords: &[f64],
    atom_charges: &[f64],
    prim_tol: f64,
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
    let natm = atom_charges.len() as u32;

    let mut results = Vec::with_capacity(groups.len());
    let rys_tables = crate::math::rys_wheeler::ext_rys_tables();

    for class in groups {
        let n_pairs = class.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        // Sized to the widest class merged into this dispatch.
        let g_per_axis = class.max_g_per_axis;
        let nroots = class.nroots as usize;

        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, g_per_axis, class.max_block_len);
        let g_stride = one_e_g_slab_stride(g_per_axis);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&class.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&class.class_shape));
        // The extended-Rys constant tables (~4.7 KB), read only by a nuclear
        // class whose Rys order is past the polynomial-fit ceiling.
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(class.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_pairs`, by the per-shell `nprim`/`nctr` read from `shell_meta`,
        // and by the class-uniform G-tensor extents.
        macro_rules! launch_with {
            ($op:expr, $nr:expr) => {
                unsafe {
                    one_electron_scalar_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), class.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), class.class_shape.len()),
                        ArrayArg::from_raw_parts(coords_h.clone(), coords_src.len()),
                        ArrayArg::from_raw_parts(charges_h.clone(), charges_src.len()),
                        ArrayArg::from_raw_parts(rys_tab_h.clone(), EXT_TABLES_LEN),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), class.out_len),
                        PIE4,
                        prim_tol,
                        SQRTPI,
                        std::f64::consts::PI,
                        natm,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $op,
                        $nr,
                        per_unit,
                    );
                }
            };
        }

        // overlap (op_kind=0) / kinetic (op_kind=1) use nroots=1 (no Rys).
        // nuclear (op_kind=2) selects rys_rootN for nroots in 1..=5.
        if op_kind == 0 {
            launch_with!(0u32, 1u32);
        } else if op_kind == 1 {
            launch_with!(1u32, 1u32);
        } else {
            // Every reachable order gets its own arm. The upstream ceiling
            // check already refused anything above
            // `device_nroots_ceiling(backend, RysFamily::Int1e)`, which is 5
            // unless the feature, the backend's FMA probe and this family's
            // flip all agree.
            match nroots {
                1 => launch_with!(2u32, 1u32),
                2 => launch_with!(2u32, 2u32),
                3 => launch_with!(2u32, 3u32),
                4 => launch_with!(2u32, 4u32),
                #[cfg(feature = "extended-device-rys")]
                6 => launch_with!(2u32, 6u32),
                #[cfg(feature = "extended-device-rys")]
                7 => launch_with!(2u32, 7u32),
                #[cfg(feature = "extended-device-rys")]
                8 => launch_with!(2u32, 8u32),
                #[cfg(feature = "extended-device-rys")]
                9 => launch_with!(2u32, 9u32),
                #[cfg(feature = "extended-device-rys")]
                10 => launch_with!(2u32, 10u32),
                #[cfg(feature = "extended-device-rys")]
                11 => launch_with!(2u32, 11u32),
                #[cfg(feature = "extended-device-rys")]
                12 => launch_with!(2u32, 12u32),
                _ => launch_with!(2u32, 5u32),
            }
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..class.out_len].to_vec());
    }
    results
}

/// The scalar 1e operator a batched run evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneEOperator {
    /// `int1e_ovlp`
    Overlap,
    /// `int1e_kin`
    Kinetic,
    /// `int1e_nuc`
    Nuclear,
}

impl OneEOperator {
    fn op_kind(self) -> u32 {
        match self {
            Self::Overlap => 0,
            Self::Kinetic => 1,
            Self::Nuclear => 2,
        }
    }

    /// libcint symbol this operator corresponds to, in the spherical form.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Overlap => "int1e_ovlp_sph",
            Self::Kinetic => "int1e_kin_sph",
            Self::Nuclear => "int1e_nuc_sph",
        }
    }
}

/// A point-charge nucleus for the [`OneEOperator::Nuclear`] arm.
#[derive(Clone, Copy, Debug)]
pub struct BatchAtom {
    /// Nuclear charge.
    pub charge: f64,
    /// Position, in Bohr.
    pub center: [f64; 3],
}

/// Spherical AO blocks for a 1e pair batch, plus the offsets that locate each pair.
#[derive(Clone, Debug, Default)]
pub struct OneEBatchOutput {
    /// Concatenated spherical AO blocks, in the caller's pair order.
    pub values: Vec<f64>,
    /// `offsets[n]` is where pair `n`'s block starts in [`Self::values`].
    pub offsets: Vec<usize>,
    /// Execution statistics.
    pub stats: crate::kernels::two_electron::BatchExecutionStats,
}

/// Evaluate a list of shell pairs as `int1e_{ovlp,kin,nuc}_sph`, one dispatch
/// per launch class (Phase 35).
///
/// Same shape as `two_electron::evaluate_2e_quartet_batch`, one index shorter:
/// the list is grouped by `(li, lj)`, each group costs exactly one dispatch and
/// one readback, and the basis is uploaded once for the whole run. `atoms` is
/// read only by [`OneEOperator::Nuclear`]; pass an empty slice otherwise.
///
/// `pairs` are `[i, j]` indices into `shells`.
/// Where one `(li,lj)` class landed after launch-group merging (Task 35-M2).
struct OneEClassPlacement {
    li: u32,
    lj: u32,
    /// Index into the group list — which dispatch's buffer holds these blocks.
    group: usize,
    /// Caller-order indices of this class's pairs.
    members: Vec<usize>,
    /// Each member's offset into the group's Cartesian buffer.
    cart_offsets: Vec<usize>,
    /// Cartesian elements per contraction block for this class.
    cart_block: usize,
    /// Half-open range of the group's Cartesian buffer this class owns.
    cart_span: (usize, usize),
}

pub fn evaluate_1e_pair_batch(
    backend: &ResolvedBackend,
    operator: OneEOperator,
    shells: &[crate::kernels::two_electron::BatchShell],
    atoms: &[BatchAtom],
    pairs: &[[u32; 2]],
) -> Result<OneEBatchOutput, cintxRsError> {
    evaluate_1e_pair_batch_with(
        backend,
        operator,
        shells,
        atoms,
        pairs,
        crate::kernels::two_electron::BatchOptions::default(),
    )
}

/// [`evaluate_1e_pair_batch`] with primitive screening (Task 34-D2).
///
/// Only the nuclear arm screens — the overlap and kinetic arms have no
/// per-atom inner loop to skip. `options.primitive_tolerance` is exact at its
/// default `0.0`: the only terms dropped there are the ones whose scale factor
/// underflowed to exactly zero, so the result is bit-identical to no screening
/// at all.
///
/// # Errors
/// As [`evaluate_1e_pair_batch`].
pub fn evaluate_1e_pair_batch_with(
    backend: &ResolvedBackend,
    operator: OneEOperator,
    shells: &[crate::kernels::two_electron::BatchShell],
    atoms: &[BatchAtom],
    pairs: &[[u32; 2]],
    options: crate::kernels::two_electron::BatchOptions,
) -> Result<OneEBatchOutput, cintxRsError> {
    let resident = crate::kernels::two_electron::ResidentBasis::new(backend, shells)?;
    evaluate_1e_pair_batch_resident_with(backend, operator, &resident, atoms, pairs, options)
}

/// [`evaluate_1e_pair_batch`] against a basis already on the device
/// (Task 34-C2).
///
/// Identical results; the difference is that the flattened basis is the
/// caller's [`crate::kernels::two_electron::ResidentBasis`] rather than a
/// throwaway one, so `basis_upload_bytes` is the full upload on the first call
/// and **0** on every later one. An SCF that rebuilds the same one-electron
/// matrices every iteration is the case this exists for.
///
/// # Errors
/// As [`evaluate_1e_pair_batch`], plus a backend mismatch on `resident`.
pub fn evaluate_1e_pair_batch_resident(
    backend: &ResolvedBackend,
    operator: OneEOperator,
    resident: &crate::kernels::two_electron::ResidentBasis,
    atoms: &[BatchAtom],
    pairs: &[[u32; 2]],
) -> Result<OneEBatchOutput, cintxRsError> {
    evaluate_1e_pair_batch_resident_with(
        backend,
        operator,
        resident,
        atoms,
        pairs,
        crate::kernels::two_electron::BatchOptions::default(),
    )
}

/// [`evaluate_1e_pair_batch_resident`] with primitive screening (Task 34-D2).
///
/// # Errors
/// As [`evaluate_1e_pair_batch_resident`].
pub fn evaluate_1e_pair_batch_resident_with(
    backend: &ResolvedBackend,
    operator: OneEOperator,
    resident: &crate::kernels::two_electron::ResidentBasis,
    atoms: &[BatchAtom],
    pairs: &[[u32; 2]],
    options: crate::kernels::two_electron::BatchOptions,
) -> Result<OneEBatchOutput, cintxRsError> {
    use crate::transform::c2s::cart_to_sph_1e_into;

    resident.check_for(operator.symbol(), backend)?;
    let shells = resident.shells();

    // Output offsets in the caller's order, computed before any dispatch so a
    // failure cannot leave a partially-sized buffer behind.
    let mut offsets = Vec::with_capacity(pairs.len());
    let mut total = 0_usize;
    for pair in pairs {
        for &s in pair {
            if s as usize >= shells.len() {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("1e-batch:shell-index-out-of-range:{s}"),
                });
            }
        }
        offsets.push(total);
        total += shells[pair[0] as usize].ao_len() * shells[pair[1] as usize].ao_len();
    }

    let mut output = OneEBatchOutput {
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

    let mut atom_coords = Vec::with_capacity(atoms.len() * 3);
    let mut atom_charges = Vec::with_capacity(atoms.len());
    for atom in atoms {
        atom_coords.extend_from_slice(&atom.center);
        atom_charges.push(atom.charge);
    }

    // Group by launch class, preserving the caller's order within a class.
    let mut grouped: std::collections::BTreeMap<[u8; 2], Vec<usize>> = Default::default();
    for (index, pair) in pairs.iter().enumerate() {
        let key = [shells[pair[0] as usize].l, shells[pair[1] as usize].l];
        grouped.entry(key).or_default().push(index);
    }

    // Build every class's pair table before dispatching anything, so a class
    // above the device Rys ceiling rejects the batch without having launched.
    //
    // Classes are merged into dispatch groups keyed on the Rys order (Task
    // 35-M2). `op_kind` is fixed by `operator`, so it needs no key: overlap and
    // kinetic are `nroots == 1` throughout and collapse to one dispatch.
    let kinetic = operator == OneEOperator::Kinetic;
    // Task 33-03: nuclear-attraction classes past the polynomial-fit ceiling
    // are accepted only where the feature, this backend's FMA probe and the
    // `int1e` flip all agree.
    let batch_ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int1e,
    );
    let mut groups: Vec<OneELaunchGroup> = Vec::new();
    let mut group_of: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut classes: Vec<OneEClassPlacement> = Vec::with_capacity(grouped.len());
    for (class, members) in grouped {
        let [li, lj] = class;
        let nroots = if operator == OneEOperator::Nuclear {
            let nroots = (li as usize + lj as usize) / 2 + 1;
            if nroots > batch_ceiling {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!(
                        "1e-batch:nroots={nroots} exceeds device ceiling {batch_ceiling} \
                         for l=({li},{lj})"
                    ),
                });
            }
            nroots as u32
        } else {
            // Overlap and kinetic are not Rys quadratures; the kernel is
            // specialized at `nroots == 1` for both.
            1
        };

        let group_index = match group_of.get(&nroots) {
            Some(&index) => index,
            None => {
                groups.push(OneELaunchGroup::new(nroots));
                let index = groups.len() - 1;
                group_of.insert(nroots, index);
                index
            }
        };
        let group = &mut groups[group_index];
        let class_index = group.push_class(u32::from(li), u32::from(lj), kinetic);

        let cart_block = ncart(li) * ncart(lj);
        group.max_block_len = group.max_block_len.max(cart_block as u32);
        group.pairs.reserve(members.len() * 4);
        let mut cart_offsets = Vec::with_capacity(members.len());
        let cart_span_start = group.out_len;
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

        classes.push(OneEClassPlacement {
            li: u32::from(li),
            lj: u32::from(lj),
            group: group_index,
            members,
            cart_offsets,
            cart_block,
            // Members were appended contiguously, so the class owns exactly
            // this half-open range of the group's Cartesian buffer. The s/p
            // normalization below scales that range and no other class's.
            cart_span: (cart_span_start, group.out_len),
        });
    }

    let dispatch_start = std::time::Instant::now();
    let mut carts = dispatch_1e_batches(
        backend,
        operator.op_kind(),
        resident.handles(),
        &groups,
        &atom_coords,
        &atom_charges,
        options.primitive_tolerance,
    )?;
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
        .map(|group| one_e_g_slab_stride(group.max_g_per_axis) * std::mem::size_of::<f64>())
        .max()
        .unwrap_or(0);
    output.stats.transfer_bytes = output.stats.basis_upload_bytes
        + groups
            .iter()
            .map(OneELaunchGroup::upload_bytes)
            .sum::<usize>();

    let transform_start = std::time::Instant::now();

    // libcint moves the spherical normalization for s (l=0) and p (l=1)
    // shells out of the c2s tables and into the primitive loop (`g1e.c`
    // line 120: `common_factor * CINTcommon_fac_sp(i_l) *
    // CINTcommon_fac_sp(j_l)`), so the c2s tables carry 1.0 there. Without
    // this factor s/p integrals come out ~4*pi too large.
    //
    // Applied to the *Cartesian* buffer, before the transform, exactly where
    // the per-pair launcher applies it: scaling the spherical result instead
    // would reorder the multiplication and cost a ULP against that path.
    //
    // Scoped to this class's own span of the group buffer: after Task 35-M2
    // a dispatch carries several classes, and each has its own `sp_scale`.
    //
    // This prepass mutates `carts`, so it stays serial and runs to completion
    // before the transform below reads any of it (Task 36-T2).
    for class in &classes {
        let (li, lj) = (class.li as u8, class.lj as u8);
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        let (span_start, span_end) = class.cart_span;
        if (sp_scale - 1.0).abs() > 1e-15 {
            for value in carts[class.group][span_start..span_end].iter_mut() {
                *value *= sp_scale;
            }
        }
    }
    let carts = &carts;

    // Task 36-T2: one job per pair, in the caller's order, each writing a
    // disjoint output block. Each output element is produced by exactly one
    // pair, so the split reorders no summation.
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
            let (li, lj) = (class.li as u8, class.lj as u8);
            let cart_block = class.cart_block;
            let (nsi, nsj) = (nsph(li), nsph(lj));

            profile.start();
            sph.clear();
            sph.resize(nsi * nsj, 0.0);
            profile.charge_alloc();

            let cart = &carts[class.group];
            let p = pairs[index];
            let (nci_ctr, ncj_ctr) = (
                shells[p[0] as usize].nctr as usize,
                shells[p[1] as usize].nctr as usize,
            );
            let di = nci_ctr * nsi;
            let src_base = class.cart_offsets[slot];
            for ci in 0..nci_ctr {
                for cj in 0..ncj_ctr {
                    let base = src_base + (ci * ncj_ctr + cj) * cart_block;
                    cart_to_sph_1e_into(&cart[base..base + cart_block], sph, li, lj, c2s_scratch);
                    profile.charge_transform();
                    for mj in 0..nsj {
                        let jidx = cj * nsj + mj;
                        for mi in 0..nsi {
                            let iidx = ci * nsi + mi;
                            block[iidx + di * jidx] = sph[mj * nsi + mi];
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

/// Backend dispatch for a whole batched 1e run.
fn dispatch_1e_batches(
    backend: &ResolvedBackend,
    op_kind: u32,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneELaunchGroup],
    atom_coords: &[f64],
    atom_charges: &[f64],
    prim_tol: f64,
) -> Result<Vec<Vec<f64>>, cintxRsError> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => Ok(run_1e_batches::<cubecl::cpu::CpuRuntime>(
            client,
            op_kind,
            basis,
            groups,
            atom_coords,
            atom_charges,
            prim_tol,
        )),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => Ok(run_1e_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            op_kind,
            basis,
            groups,
            atom_coords,
            atom_charges,
            prim_tol,
        )),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => Ok(run_1e_batches::<cubecl_cuda::CudaRuntime>(
            client,
            op_kind,
            basis,
            groups,
            atom_coords,
            atom_charges,
            prim_tol,
        )),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => Ok(run_1e_batches::<cubecl_hip::HipRuntime>(
            client,
            op_kind,
            basis,
            groups,
            atom_coords,
            atom_charges,
            prim_tol,
        )),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => Ok(run_1e_batches::<cubecl_wgpu::WgpuRuntime>(
            client,
            op_kind,
            basis,
            groups,
            atom_coords,
            atom_charges,
            prim_tol,
        )),
    }
}

/// Single-pair dispatch — a one-class, one-pair batch.
///
/// Kept as its own entry point because the per-tuple compatibility API
/// evaluates exactly one shell pair and must keep doing so. It marshals the two
/// shells into the flattened form [`run_1e_batches`] consumes, so both paths
/// execute the *same* kernel and every existing parity test covers the batched
/// code at `n_pairs == 1`.
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
    let _ = nroots; // derived per class inside `run_1e_batches`
    let li_u = li as usize;
    let lj_u = lj as usize;
    let nci = (li_u + 1) * (li_u + 2) / 2;
    let ncj = (lj_u + 1) * (lj_u + 2) / 2;
    let block_len = nci * ncj;
    let out_len = (nctr_i as usize) * (nctr_j as usize) * block_len;

    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (exps, coeffs, center, nprim, nctr) in [
        (exps_i, coeff_i, ri, nprim_i, nctr_i),
        (exps_j, coeff_j, rj, nprim_j, nctr_j),
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

    let mut group = OneELaunchGroup::new(nroots);
    let class_index = group.push_class(li, lj, op_kind == 1);
    group.pairs.extend_from_slice(&[0, 1, 0, class_index]);
    group.out_len = out_len;
    group.max_block_len = block_len as u32;

    // The per-tuple compatibility path never screens: it has no options.
    run_1e_batches::<R>(
        client,
        op_kind,
        &handles,
        std::slice::from_ref(&group),
        atom_coords,
        atom_charges,
        0.0,
    )
    .pop()
    .unwrap_or_default()
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
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn one_electron_moment_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    pairs: &Array<u32>,
    class_shape: &Array<u32>,
    pair_drj: &Array<F>,
    g: &mut Array<F>,
    cart_out: &mut Array<F>,
    sqrtpi: F,
    pi_const: F,
    n_pairs: u32,
    n_cubes: u32,
    g_stride: u32,
    #[comptime] op_mode: u32,
    #[comptime] moment_order: u32,
    #[comptime] rank: u32,
    #[comptime] per_unit: u32,
) {
    // WR-05: the inert comptime `complex_output` hint was removed. The moment
    // device path emits REAL components; the host materializes re=0/im=value for
    // the safe-API complex_values() view. There is no on-device consumer today,
    // and the future GIAO-on-device path (Phase 30 GIAO×σ) will introduce its own
    // output-convention plumbing when it lands — carrying a dead comptime arg
    // through three signatures until then only obscures the launcher contract.
    // Slot / lane decomposition — see `two_electron.rs` for why this is
    // arithmetic on comptime-folded flags rather than a `comptime!` if/else.
    let cube_pos = CUBE_POS as u32;
    let unit_pos = UNIT_POS as u32;
    let cube_dim = CUBE_DIM as u32;

    let coop = if comptime!(per_unit == 1u32) {
        0u32
    } else {
        1u32
    };
    let punit = 1u32 - coop;
    let slots_per_cube = cube_dim * punit + coop;
    let slot = cube_pos * slots_per_cube + unit_pos * punit;
    let n_slots = n_cubes * slots_per_cube;
    let lane = unit_pos * coop;

    if lane == 0u32 {
        let gbase = slot * g_stride;

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
            let sj = pairs[(prow + 1u32) as usize];
            let out_off = pairs[(prow + 2u32) as usize];

            let cls = pairs[(prow + 3u32) as usize];
            let srow = cls * comptime!(ONE_E_DERIV_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];

            let mi = si * 4u32;
            let eoff_i = shell_meta[mi as usize];
            let coff_i = shell_meta[(mi + 1u32) as usize];
            let nprim_i = shell_meta[(mi + 2u32) as usize];
            let nctr_i = shell_meta[(mi + 3u32) as usize];
            let mj = sj * 4u32;
            let eoff_j = shell_meta[mj as usize];
            let coff_j = shell_meta[(mj + 1u32) as usize];
            let nprim_j = shell_meta[(mj + 2u32) as usize];
            let nctr_j = shell_meta[(mj + 3u32) as usize];

            let ci3 = si * 3u32;
            let rix = centers[ci3 as usize];
            let riy = centers[(ci3 + 1u32) as usize];
            let riz = centers[(ci3 + 2u32) as usize];
            let cj3 = sj * 3u32;
            let rjx = centers[cj3 as usize];
            let rjy = centers[(cj3 + 1u32) as usize];
            let rjz = centers[(cj3 + 2u32) as usize];

            // `drj = rj - origin` is per pair: the base families measure from a
            // common origin and the `_origj` variants from `rj` itself. The host
            // resolves the choice; the batch carries the resolved vector.
            let d3 = qi * 3u32;
            let drjx = pair_drj[d3 as usize];
            let drjy = pair_drj[(d3 + 1u32) as usize];
            let drjz = pair_drj[(d3 + 2u32) as usize];

            // Ket headroom: overlap G-tensor must span j..=lj+moment_order so the
            // per-axis moment ladder can read overlap[jx + t] for t up to moment_order.
            let nmax = li + lj + moment_order;
            let lj_ext = lj + moment_order;
            let dj = nmax + 1u32;
            let g_per_axis = (nmax + 1u32) * (lj_ext + 1u32);
            let total_g = 3u32 * g_per_axis;
            let gx = gbase;
            let gy = gbase + g_per_axis;
            let gz = gbase + 2u32 * g_per_axis;

            let nci = (li + 1u32) * (li + 2u32) / 2u32;
            let ncj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let block_len = nci * ncj;
            let total_len = rank * block_len;
            let out_total = nctr_i * nctr_j * total_len;

            let mut oi = out_off;
            while oi < out_off + out_total {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];

                    let zeta = ai + aj;
                    let aij2 = F::new(0.5_f32) / zeta;
                    let rirjx = rix - rjx;
                    let rirjy = riy - rjy;
                    let rirjz = riz - rjz;
                    let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
                    let fac = F::exp(-ai * aj / zeta * rr);
                    let px = (ai * rix + aj * rjx) / zeta;
                    let py = (ai * riy + aj * rjy) / zeta;
                    let pz = (ai * riz + aj * rjz) / zeta;

                    // Build the OVERLAP base G-tensor (fixed-center VRR + HRR to lj_ext).
                    let mut gi = gbase;
                    while gi < gbase + total_g {
                        g[gi as usize] = F::new(0.0_f32);
                        gi += 1u32;
                    }
                    g[gx as usize] = F::new(1.0_f32);
                    g[gy as usize] = F::new(1.0_f32);
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
                        let coeff_i_val = coeffs[(coff_i + pi * nctr_i + ci) as usize];
                        let mut cj = 0u32;
                        while cj < nctr_j {
                            let coeff_j_val = coeffs[(coff_j + pj * nctr_j + cj) as usize];
                            let weight = coeff_i_val * coeff_j_val;
                            let base = out_off + (ci * nctr_j + cj) * total_len;

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
                                            let mx0 = F::new(0.0_f32);
                                            let mx1 = F::new(0.0_f32);
                                            let mx2 = F::new(0.0_f32);
                                            let mx3 = F::new(0.0_f32);
                                            let mx4 = F::new(0.0_f32);
                                            let my0 = F::new(0.0_f32);
                                            let my1 = F::new(0.0_f32);
                                            let my2 = F::new(0.0_f32);
                                            let my3 = F::new(0.0_f32);
                                            let my4 = F::new(0.0_f32);
                                            let mz0 = F::new(0.0_f32);
                                            let mz1 = F::new(0.0_f32);
                                            let mz2 = F::new(0.0_f32);
                                            let mz3 = F::new(0.0_f32);
                                            let mz4 = F::new(0.0_f32);
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
                                                g,
                                                gx,
                                                jx,
                                                dj,
                                                ix,
                                                drjx,
                                                moment_order,
                                                &mut mx0,
                                                &mut mx1,
                                                &mut mx2,
                                                &mut mx3,
                                                &mut mx4,
                                            );
                                            moment_axis_ladder::<F>(
                                                g,
                                                gy,
                                                jy,
                                                dj,
                                                iy,
                                                drjy,
                                                moment_order,
                                                &mut my0,
                                                &mut my1,
                                                &mut my2,
                                                &mut my3,
                                                &mut my4,
                                            );
                                            moment_axis_ladder::<F>(
                                                g,
                                                gz,
                                                jz,
                                                dj,
                                                iz,
                                                drjz,
                                                moment_order,
                                                &mut mz0,
                                                &mut mz1,
                                                &mut mz2,
                                                &mut mz3,
                                                &mut mz4,
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
                                                    + F::new(2.0_f32) * (mx2 * my2 * mz0)
                                                    + F::new(2.0_f32) * (mx2 * my0 * mz2)
                                                    + mx0 * my4 * mz0
                                                    + F::new(2.0_f32) * (mx0 * my2 * mz2)
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

                                                    cart_out[(base + comp * block_len + elem)
                                                        as usize] += weight * (vx * vy * vz);
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

            qi += qi_step;
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
        *m2 = ov2 + F::new(2.0_f32) * drj * ov1 + drj * drj * ov0;
    }
    if comptime!(moment_order >= 3u32) {
        let ov1 = g[(off + (jx + 1u32) * dj + i) as usize];
        let ov2 = g[(off + (jx + 2u32) * dj + i) as usize];
        let ov3 = g[(off + (jx + 3u32) * dj + i) as usize];
        let d2 = drj * drj;
        *m3 = ov3 + F::new(3.0_f32) * drj * ov2 + F::new(3.0_f32) * d2 * ov1 + d2 * drj * ov0;
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
            + F::new(4.0_f32) * drj * ov3
            + F::new(6.0_f32) * d2 * ov2
            + F::new(4.0_f32) * d3 * ov1
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

/// Single source of truth for the Cluster-A moment `(op_mode → moment_order, rank)`
/// mapping (WR-01). The dispatcher (`moment_dispatch`), the device runner's buffer
/// sizing (`out_len`/`nmax`), and the device comptime `match` MUST all agree; this
/// `const fn` is the one table they derive from so they can never silently drift.
///
/// Returns `(moment_order, rank)` for a valid `op_mode` in `0..=7`. `op_mode` is set
/// only by `moment_dispatch`, so an out-of-range value is a wiring bug — fail loudly.
///
/// Mode map: 0=r 1=rr 2=rrr 3=rrrr 4=r2 5=r4 6=z 7=zz (`_origj` variants reuse the
/// same `op_mode`; only the origin source differs, handled host-side via `drj`).
const fn moment_params(op_mode: u32) -> (u32, u32) {
    match op_mode {
        0 => (1, 3),  // r
        1 => (2, 9),  // rr
        2 => (3, 27), // rrr
        3 => (4, 81), // rrrr
        4 => (2, 1),  // r2
        5 => (4, 1),  // r4
        6 => (1, 1),  // z
        7 => (2, 1),  // zz
        _ => panic!("invalid moment op_mode (must be 0..=7)"),
    }
}

/// IN-03: single source of truth for the GIAO 1e per-engine VRR headroom. The host
/// fail-closed guard, the host nuclear Rys-nroots ceiling, and the host-side device
/// buffer sizing (`run_1e_giao_ovlp_device` / `run_1e_giao_nuc_device`) all derive
/// from these `const fn`s so the VRR envelope cannot drift between the guard and the
/// allocation. (D-13: under-sizing the device scratch silently truncates output, so
/// the guard and the sizing MUST agree by construction.) The `#[cube]` kernel bodies
/// recompute the same `nmax = li+lj+{3,5}` inline because CubeCL forbids plain-fn
/// calls inside `#[cube]` (D-08); these const fns govern the host envelope they read.
///
/// overlap engine: `nmax = li + lj + 3`, VRR envelope checked `nmax <= 8`.
/// nuclear engine: `nmax = li + lj + 5`, Rys `nroots = nmax / 2 + 1`.
const fn giao_ovlp_nmax(li: u32, lj: u32) -> u32 {
    li + lj + 3
}
const fn giao_nuc_nmax(li: u32, lj: u32) -> u32 {
    li + lj + 5
}
const fn giao_nuc_nroots(li: u32, lj: u32) -> u32 {
    giao_nuc_nmax(li, lj) / 2 + 1
}

/// `g_per_axis` for one moment class at a given `moment_order`.
fn one_e_moment_g_per_axis(moment_order: u32, li: usize, lj: usize) -> usize {
    let mo = moment_order as usize;
    (li + lj + mo + 1) * (lj + mo + 1)
}

/// Evaluate every launch group of a batched moment run (Task 35-D, wave 4).
///
/// Three comptime parameters, but only one of them is a *shape* selector:
/// `op_mode` picks the operator, and `moment_order`/`rank` are both functions of
/// it through `moment_params`. All three are therefore fixed by the caller, so a
/// whole work list is one dispatch.
fn run_1e_moment_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    op_mode: u32,
) -> Vec<Vec<f64>> {
    if groups.is_empty() {
        return Vec::new();
    }

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

    let mut results = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let n_pairs = group.len();
        if n_pairs == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = one_e_g_slab_stride(group.max_g_per_axis);
        let (n_cubes, cube_dim, n_slots) =
            one_e_launch_geometry::<R>(client, n_pairs, group.max_g_per_axis, 1);
        let g_len = n_slots * g_stride;

        let pairs_h = client.create_from_slice(u32::as_bytes(&group.pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let drj_h = client.create_from_slice(f64::as_bytes(&pair_drj[index]));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(one_e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`; in-kernel indices are bounded by
        // `n_pairs`, the class index, and the per-shell counts.
        macro_rules! launch_with {
            ($mode:expr, $order:expr, $rank:expr) => {
                unsafe {
                    one_electron_moment_kernel::launch_unchecked::<f64, R>(
                        client,
                        crate::plane::cube_count_1d(n_cubes),
                        cube_dim,
                        ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                        ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                        ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                        ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                        ArrayArg::from_raw_parts(pairs_h.clone(), group.pairs.len()),
                        ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                        ArrayArg::from_raw_parts(drj_h.clone(), pair_drj[index].len()),
                        ArrayArg::from_raw_parts(g_h.clone(), g_len),
                        ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                        SQRTPI,
                        std::f64::consts::PI,
                        n_pairs as u32,
                        n_cubes,
                        g_stride as u32,
                        $mode,
                        $order,
                        $rank,
                        per_unit,
                    );
                }
            };
        }

        // Comptime (op_mode, moment_order, rank) selected via a host match. The
        // valid Cluster-A combinations are enumerated explicitly (CubeCL cannot
        // pass comptime args dynamically). WR-01: the (order, rank) literals are
        // derived from the `moment_params` const fn — const-evaluated to
        // compile-time literals here — so this match can never drift from the
        // dispatcher's sizing.
        match op_mode {
            0u32 => launch_with!(0u32, moment_params(0).0, moment_params(0).1), // r
            1u32 => launch_with!(1u32, moment_params(1).0, moment_params(1).1), // rr
            2u32 => launch_with!(2u32, moment_params(2).0, moment_params(2).1), // rrr
            3u32 => launch_with!(3u32, moment_params(3).0, moment_params(3).1), // rrrr
            4u32 => launch_with!(4u32, moment_params(4).0, moment_params(4).1), // r2
            5u32 => launch_with!(5u32, moment_params(5).0, moment_params(5).1), // r4
            6u32 => launch_with!(6u32, moment_params(6).0, moment_params(6).1), // z
            7u32 => launch_with!(7u32, moment_params(7).0, moment_params(7).1), // zz
            // `op_mode` is set only by `moment_dispatch` (validated against
            // `moment_params`, which is 0..=7). An out-of-range value here is an
            // upstream wiring bug, not a runtime condition — surface it loudly
            // instead of silently computing `zz`.
            _ => unreachable!("invalid moment op_mode {op_mode} (must be 0..=7)"),
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Backend dispatch for a whole batched moment run.
fn dispatch_1e_moment_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[OneEDerivLaunchGroup],
    pair_drj: &[Vec<f64>],
    op_mode: u32,
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_1e_moment_batches::<cubecl::cpu::CpuRuntime>(
            client, basis, groups, pair_drj, op_mode,
        ),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_1e_moment_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups, pair_drj, op_mode,
        ),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_1e_moment_batches::<cubecl_cuda::CudaRuntime>(
            client, basis, groups, pair_drj, op_mode,
        ),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_1e_moment_batches::<cubecl_hip::HipRuntime>(
            client, basis, groups, pair_drj, op_mode,
        ),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_1e_moment_batches::<cubecl_wgpu::WgpuRuntime>(
            client, basis, groups, pair_drj, op_mode,
        ),
    }
}

/// One shell pair through the batched moment path — a one-pair group through
/// the same kernel a wide batch uses (Task 35-D).
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
    // WR-01: the passed (moment_order, rank) MUST match the comptime triple
    // selected from `op_mode` in the runner. Both derive from `moment_params`
    // (the single source of truth), so this only ever fires if the dispatcher
    // and the runner drift apart.
    debug_assert_eq!(
        (moment_order, rank),
        moment_params(op_mode),
        "moment (order, rank) ({moment_order}, {rank}) disagree with \
         moment_params(op_mode={op_mode})"
    );
    let (group, handles) = one_e_deriv_single_pair_group(
        backend,
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
        rank as usize,
        one_e_moment_g_per_axis(moment_order, li as usize, lj as usize),
        1,
    );
    dispatch_1e_moment_batches(
        backend,
        &handles,
        std::slice::from_ref(&group),
        std::slice::from_ref(&drj.to_vec()),
        op_mode,
    )
    .pop()
    .unwrap_or_default()
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
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
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
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
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
/// Host f64 reference — used by the device-vs-host cross-check, unit tests, and host fallback.
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
// `0u32 * block_len` is deliberate: these accumulations write a
// component-leading table (`0`, `1`, `2`, ... times `block_len`) and dropping the
// zero term would break the column alignment that makes the component index
// readable at a glance.
#[allow(clippy::erasing_op)]
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
    let nrys_roots = ((li + lj) as u32).div_ceil(2) + 1;

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

/// Shared component-leading staging writer for the Phase-24 1e families
/// (`moment`, `rinv`/`drinv`, `p4`, `irp`). IN-03: this collapses five
/// near-verbatim copy blocks (cart/sph contraction-blocked copy + fail-closed
/// `BufferTooSmall` check + `not0` non-zero count) into one host-side helper so a
/// future hardening of one path (e.g. the WR-02 fail-closed check) cannot be
/// forgotten in the others. Behavior is byte-identical to the inlined blocks.
///
/// `cart_comp` is the `f64` device output in component-leading,
/// contraction-blocked layout: element `(ci,cj,comp)` block starts at
/// `(ci*n_ctr_j + cj)*total_len + comp*block_len`, where `block_len == nci*ncj`.
/// `staging` receives the component-leading, contraction-blocked representation
/// output (spheric or cart per `rep`). Returns the `not0` non-zero element count.
///
/// This runs entirely host-side (it uses `Vec`, `cart_to_sph_1e`, and the typed
/// `staging` slice), so it is NOT `#[cube]` device code and is free to be a plain
/// generic fn over `F: CintFloat`.
#[allow(clippy::too_many_arguments)]
fn write_component_leading_staging<F: CintFloat>(
    rep: Representation,
    rank: usize,
    n_ctr_i: usize,
    n_ctr_j: usize,
    nci: usize,
    ncj: usize,
    nsi: usize,
    nsj: usize,
    li: u8,
    lj: u8,
    kappa_i: i16,
    kappa_j: i16,
    total_len: usize,
    block_len: usize,
    cart_comp: &[f64],
    staging: &mut [F],
) -> Result<i32, cintxRsError> {
    match rep {
        Representation::Spheric => {
            let ni_sph = n_ctr_i * nsi;
            let nj_sph = n_ctr_j * nsj;
            let sph_block = ni_sph * nj_sph;
            // WR-02: fail closed if staging cannot hold the full component rank —
            // never silently drop trailing components (OOM-safe no-partial-writes
            // contract). `needed` is the full contraction-blocked component span.
            let needed = rank * sph_block;
            if staging.len() < needed {
                return Err(cintxRsError::BufferTooSmall {
                    required: needed,
                    provided: staging.len(),
                });
            }
            for comp in 0..rank {
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
                                staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
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
            // WR-02: fail closed if staging cannot hold the full component rank.
            let needed = rank * cart_block;
            if staging.len() < needed {
                return Err(cintxRsError::BufferTooSmall {
                    required: needed,
                    provided: staging.len(),
                });
            }
            for comp in 0..rank {
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
                                staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            cart_to_spinor_sf_derivative_2d::<F>(
                staging, cart_comp, rank, li, kappa_i, lj, kappa_j, n_ctr_i, n_ctr_j,
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
    Ok(not0)
}

/// Phase 26 GIAO-01 (FND-03 / D-07): complex-interleaved staging writer.
///
/// The GIAO families are purely imaginary: libcint's cart/sph symbol returns the
/// REAL magnitude of the imaginary part, so the device emits REAL components. This
/// writer materializes the safe-API `Complex<f64>` view by writing each real value
/// `v` as the interleaved pair `[re=0.0, im=v]` into a `2×`-sized staging buffer
/// (the planner sized staging `2 * rank * block` because the manifest flag
/// `complex_output=true`). `complex_values()` then reads `chunks_exact(2)` → re=0,
/// im=v. The element ordering within each component block mirrors
/// `write_component_leading_staging` (column-major bra-fastest), so the imaginary
/// half is byte-identical (atol=1e-12) to the vendor's real `double*` output.
#[allow(clippy::too_many_arguments)]
fn write_giao_complex_staging<F: CintFloat>(
    plan: &ExecutionPlan<'_>,
    rank: usize,
    n_ctr_i: usize,
    n_ctr_j: usize,
    nci: usize,
    ncj: usize,
    nsi: usize,
    nsj: usize,
    li: u8,
    lj: u8,
    total_len: usize,
    block_len: usize,
    cart_comp: &[f64],
    staging: &mut [F],
) -> Result<i32, cintxRsError> {
    let rep = plan.representation;
    // Build the REAL component-leading block first into a scratch Vec, then splice
    // it into the interleaved staging as [0, v] pairs. The real block layout is
    // identical to `write_component_leading_staging`.
    let (ni, nj, real_block) = match rep {
        Representation::Spheric => (n_ctr_i * nsi, n_ctr_j * nsj, nsi * nsj),
        Representation::Cart => (n_ctr_i * nci, n_ctr_j * ncj, nci * ncj),
        Representation::Spinor => {
            return Err(cintxRsError::UnsupportedApi {
                requested: "spinor GIAO complex staging".to_owned(),
            });
        }
    };
    let real_total = rank * ni * nj;
    // complex_output=true → staging sized 2 * real_total. Fail closed otherwise
    // (never a silent partial write — FND-06 / D-04).
    let needed = 2 * real_total;
    if staging.len() < needed {
        return Err(cintxRsError::BufferTooSmall {
            required: needed,
            provided: staging.len(),
        });
    }

    let mut real = vec![0.0_f64; real_total];
    match rep {
        Representation::Spheric => {
            let ni_sph = ni;
            for comp in 0..rank {
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
                        let comp_base = comp * real_block;
                        for mj in 0..nsj {
                            let jj = cj * nsj + mj;
                            for mi in 0..nsi {
                                let ii = ci * nsi + mi;
                                real[comp_base + ii + jj * ni_sph] = sph_tmp[mj * nsi + mi];
                            }
                        }
                    }
                }
            }
        }
        Representation::Cart => {
            let ni_cart = ni;
            for comp in 0..rank {
                for ci in 0..n_ctr_i {
                    for cj in 0..n_ctr_j {
                        let src_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                        let block = &cart_comp[src_base..src_base + block_len];
                        let comp_base = comp * real_block;
                        for jc in 0..ncj {
                            let jj = cj * ncj + jc;
                            for ic in 0..nci {
                                let ii = ci * nci + ic;
                                real[comp_base + ii + jj * ni_cart] = block[jc * nci + ic];
                            }
                        }
                    }
                }
            }
        }
        Representation::Spinor => unreachable!(),
    }

    // Interleave: staging[2p] = 0 (re), staging[2p+1] = real[p] (im).
    let zero = F::from_f64_lossy(0.0);
    for (p, &v) in real.iter().enumerate() {
        staging[2 * p] = zero;
        staging[2 * p + 1] = F::from_f64_lossy(v);
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
    Ok(not0)
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
    // Phase 24 Cluster B (MOM-04): plain single-center 1/r Coulomb potential.
    // `rinv` (rank 1) is the int1e_nuc Rys kernel at the rinv center (env[4..6]),
    // charge=+1, NO atom-sum. `drinv` (rank 3) is the gradient wrt the rinv center C
    // (= D_I + D_J applied to the rinv G-tensor). Both single-center.
    let is_rinv = op_name == "rinv";
    let is_drinv = op_name == "drinv";
    // Both-side rank-9 family (Phase 23): <NABLA i | OP | NABLA j>.
    let is_ipovlpip = op_name == "ipovlpip";
    let is_ipkinip = op_name == "ipkinip";
    let is_ipnucip = op_name == "ipnucip";
    let is_iprinvip = op_name == "iprinvip";
    let is_rank9_both = is_ipovlpip || is_ipkinip || is_ipnucip || is_iprinvip;
    // Phase 25 HESS-01 bra-only rank-9 1e Hessian family (<∇∇ i | OP | j>).
    // ipipovlp/ipipkin ride the no-Rys overlap-deriv engine; ipipnuc/ipiprinv
    // ride the nuclear/Rys 1e path (FND-02 consumer at nroots≥6).
    let is_ipipovlp = op_name == "ipipovlp";
    let is_ipipkin = op_name == "ipipkin";
    let is_ipipnuc = op_name == "ipipnuc";
    let is_ipiprinv = op_name == "ipiprinv";
    let is_rank9_bra = is_ipipovlp || is_ipipkin || is_ipipnuc || is_ipiprinv;
    // Phase 24 Cluster C (MOM-04): `int1e_p4` = ∇⁴ (rank 1), the
    // Laplacian-of-Laplacian on the overlap G-tensor with BOTH-side +2 headroom
    // (ng={2,2,...}), no Rys.
    let is_p4 = op_name == "p4";
    // Phase 24 Cluster D (MOM-04): `int1e_irp` = i·r×∇ (rank 9), the 3×3 (∇-axis ⊗
    // r-axis) tensor on the overlap-derivative engine (no Rys). The `r` part reads
    // the gauge origin env[PTR_COMMON_ORIG] via G1E_RCJ (drj = rj - common_orig);
    // ket headroom +2 (ng={0,2,...}).
    let is_irp = op_name == "irp";
    let is_ipipr = op_name == "ipipr";

    // Phase 28 FND-05 (Gap B2): `int1e_sp` = σ·p on the bra only. Detected by
    // SYMBOL name (Pitfall 6 — never a positional OperatorId literal). This is the
    // proof VEHICLE for the spin-included si_2d transform + σ·p assembler; it stays
    // UnsupportedApi at the manifest level (D-01) and is driven only via the Spinor
    // representation. The σ·p assembler emits the four gc_x/gc_y/gc_z/gc_1 blocks the
    // host `cart_to_spinor_si_2d` consumes; nctr>1 is HANDLED (not rejected).
    let is_sp = op_name == "sp";
    let is_sigma_family = crate::kernels::sigma_1e::family_id(op_name).is_some();

    // Phase 26 GIAO-01: spin-free 1e GIAO/CG families (complex output). The
    // overlap-engine families (govlp/igovlp/cg_irxp/giao_irjxp/igkin) ride the
    // no-Rys overlap G-tensor; the nuclear-engine families (gnuc/ignuc/ia01p/
    // a01gp/cg_a11part/giao_a11part) ride the nuclear/Rys 1e path. Each maps to a
    // (op_kind, rank) tuple. The device emits REAL components; the host then
    // materializes the complex re=0/im=value interleaved view (FND-03 / D-15).
    let giao_ovlp_op: Option<u32> = match op_name {
        "govlp" => Some(0),
        "igovlp" => Some(1),
        "cg_irxp" => Some(2),
        "giao_irjxp" => Some(3),
        "igkin" => Some(4),
        _ => None,
    };
    let is_giao_ovlp = giao_ovlp_op.is_some();
    // IN-02: carry an explicit per-family `is_rinv_center: bool` on the dispatch
    // tuple (op_kind, rank, is_rinv_center), mirroring the moment path's
    // `is_origj` precedent (moment_dispatch_name below). The nuclear-model choice
    // (type 2 atom-sum -Z vs type 1 single rinv center +1) is now data-driven from
    // this enumerated name table rather than re-derived downstream from the
    // dispatch-table ordinal comparison — adding/reordering a family can no
    // longer silently re-point the nuclear-model branch.
    let giao_nuc_op: Option<(u32, u32, bool)> = match op_name {
        "gnuc" => Some((0, 3, false)), // type 2: atom-sum -Z
        "ignuc" => Some((1, 3, false)),
        "ia01p" => Some((2, 3, true)), // type 1: single rinv center +1
        "a01gp" => Some((3, 9, true)),
        "cg_a11part" => Some((4, 9, true)),
        "giao_a11part" => Some((5, 9, true)),
        _ => None,
    };
    let is_giao_nuc = giao_nuc_op.is_some();

    // Phase 25 HESS-04 (Cluster D): 3rd/4th-order 1e derivative families
    // (deriv3.c rank 27: ipipipnuc/ipipiprinv/ipipnucip/ipiprinvip;
    //  deriv4.c rank 81: ipipipiprinv/ipiprinvipip/ipipiprinvip). Host-routed
    // (FND-02): the bra/ket +2/+3 headroom can elevate the nuclear Rys nroots
    // beyond the device MAX_DEVICE_NROOTS=5 cap.
    let is_deriv34 = crate::kernels::deriv34::is_deriv34(op_name);

    // Phase 24 Cluster A (MOM-01/02/03): overlap-derived position-tensor moment
    // families r/rr/rrr/rrrr/r2/r4/z/zz and their `_origj` variants. Each maps to
    // a (op_mode, moment_order, rank) tuple for the parameterized moment kernel.
    // `_origj` reuses the SAME op_mode/order/rank — only the origin source differs
    // (handled below via drj = rj - origin; for _origj origin = rj so drj = 0).
    // WR-01: map op_name → op_mode only; derive (moment_order, rank) from the shared
    // `moment_params` const fn so the dispatcher, the device sizing, and the device
    // comptime match all share ONE source of truth and cannot drift.
    //
    // IN-01: this exact-name table is also the single source of truth for the
    // origin source. Each arm carries `(op_mode, is_origj)` so the origin-source
    // choice (ket center `rj` vs gauge origin `common_orig`) is data-driven here
    // rather than re-derived downstream from an `op_name.ends_with("_origj")`
    // string suffix — a coincidental future `_origj` spelling cannot silently
    // flip the origin branch.
    let moment_dispatch_name: Option<(u32, bool)> = match op_name {
        "r" => Some((0, false)),
        "r_origj" => Some((0, true)),
        "rr" => Some((1, false)),
        "rr_origj" => Some((1, true)),
        "rrr" => Some((2, false)),
        "rrrr" => Some((3, false)),
        "r2" => Some((4, false)),
        "r2_origj" => Some((4, true)),
        "r4" => Some((5, false)),
        "r4_origj" => Some((5, true)),
        "z" => Some((6, false)),
        "z_origj" => Some((6, true)),
        "zz" => Some((7, false)),
        "zz_origj" => Some((7, true)),
        _ => None,
    };
    // Carries (op_mode, moment_order, rank, is_origj). The origin-source flag rides
    // the dispatch tuple so the numerical origin branch is selected by the same
    // enumerated name table as op_mode, not by string-suffix re-derivation (IN-01).
    let moment_dispatch: Option<(u32, u32, u32, bool)> =
        moment_dispatch_name.map(|(op_mode, is_origj)| {
            let (order, rank) = moment_params(op_mode);
            (op_mode, order, rank, is_origj)
        });
    let is_moment = moment_dispatch.is_some();

    if !is_overlap
        && !is_kinetic
        && !is_nuclear
        && !is_ipovlp
        && !is_ipkin
        && !is_ipnuc
        && !is_iprinv
        && !is_rank9_both
        && !is_rank9_bra
        && !is_moment
        && !is_rinv
        && !is_drinv
        && !is_p4
        && !is_irp
        && !is_ipipr
        && !is_giao_ovlp
        && !is_giao_nuc
        && !is_deriv34
        && !is_sp
        && !is_sigma_family
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
    // Phase 26 GIAO-01 overlap-engine families (govlp/igovlp/cg_irxp/
    // giao_irjxp/igkin) — rank 3, no Rys. Reads the gauge origin (common_orig)
    // for cg_irxp via drj; emits REAL components → host complex re=0/im=value.
    // ─────────────────────────────────────────────────────────────────────────
    if let Some(op_kind) = giao_ovlp_op {
        // D-11: spinor GIAO reps are registered for surface completeness but
        // return UnsupportedApi (no spin block silently computed as spin-free).
        if plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name}"),
            });
        }

        // Internal G-tensor ceiling: nmax = li+lj+3 (max headroom over the
        // overlap-engine families). Fail closed if a corpus shell exceeds the
        // li+lj+3<=8 VRR envelope rather than silently truncating (D-13).
        // IN-03: envelope from the shared `giao_ovlp_nmax` const fn — same source
        // as the host-side device buffer sizing, so the two cannot drift.
        if giao_ovlp_nmax(li as u32, lj as u32) > 8 {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device 1e GIAO kernel supports l_i+l_j+3<=8; got l_i={li}, l_j={lj} \
                     for int1e_{op_name}"
                ),
            });
        }

        // Gauge origin (Phase 22): cg_irxp reads drj = rj - common_orig; the
        // other overlap families ignore drj (govlp/igovlp/igkin use ri/rj only,
        // giao_irjxp uses rj directly inside the kernel).
        let origin: [f64; 3] = plan.operator_env_params.common_orig.unwrap_or([0.0; 3]);
        let drj = [rj[0] - origin[0], rj[1] - origin[1], rj[2] - origin[2]];

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let rank: usize = 3;
        let block_len = nci * ncj;
        let total_len = rank * block_len;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        let mut cart_comp = run_1e_giao_ovlp_on_backend(
            backend,
            op_kind,
            li as u32,
            lj as u32,
            n_prim_i as u32,
            n_prim_j as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            ri,
            rj,
            drj,
            &exps_i,
            &exps_j,
            &coeff_i,
            &coeff_j,
        );

        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        let not0 = write_giao_complex_staging::<F>(
            plan, rank, n_ctr_i, n_ctr_j, nci, ncj, nsi, nsj, li, lj, total_len, block_len,
            &cart_comp, staging,
        )?;

        let staging_bytes = std::mem::size_of_val(staging);
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
    // Phase 26 GIAO-01 nuclear-engine families (gnuc/ignuc/ia01p/a01gp/
    // cg_a11part/giao_a11part) — rank 3/9, Rys atom-sum. cg_a11part reads the
    // gauge origin (drj); emits REAL → host complex re=0/im=value (D-15).
    // ─────────────────────────────────────────────────────────────────────────
    if let Some((op_kind, rank, is_rinv_center)) = giao_nuc_op {
        if plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name}"),
            });
        }

        // 26-05: the 26-04 fail-closed a01gp guard was REMOVED here once the
        // corrected kernel (restored 0.5 common factor, intor1.c:551/572) passed
        // vendor byte-identity at atol=1e-12 (cart+sph) on the non-zero-gauge
        // non-square block. a01gp now rides the normal nuclear-engine path.

        // Rys nroots fail-closed guard (D-13). Internal ceiling nmax = li+lj+5
        // (the a01gp bra+3/ket+2 headroom); nroots = nmax/2 + 1.
        // IN-03: nroots from the shared `giao_nuc_nroots` const fn — same source as
        // the host-side device nuclear buffer sizing, so they cannot drift.
        let nuc_nroots = giao_nuc_nroots(li as u32, lj as u32);
        if nuc_nroots as usize > MAX_DEVICE_NROOTS {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_{op_name} kernel supports nroots<={MAX_DEVICE_NROOTS}; \
                     got nroots={nuc_nroots} for l_i={li}, l_j={lj}"
                ),
            });
        }

        let origin: [f64; 3] = plan.operator_env_params.common_orig.unwrap_or([0.0; 3]);
        let drj = [rj[0] - origin[0], rj[1] - origin[1], rj[2] - origin[2]];

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let block_len = nci * ncj;
        let total_len = (rank as usize) * block_len;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        // Nuclear model (libcint cint1e.c make_g1e_gout int1e_type):
        //   gnuc/ignuc → type 2: sum over ALL nuclei, charge -Z_C, low→high (D-10).
        //   ia01p/a01gp/cg_a11part/giao_a11part → type 1: SINGLE rinv center
        //     (env[PTR_RINV_ORIG]) with charge +1 (CINTg1e_nuc nuc_id=-1).
        // IN-02: `is_rinv_center` is now sourced from the giao_nuc_op dispatch tuple
        // above (per-family bool, mirrors the moment `is_origj` precedent), NOT from
        // the dispatch ordinal (the old `op_kind`-threshold coupling is gone).
        let (origin_coords, origin_charges): (Vec<f64>, Vec<f64>) = if is_rinv_center {
            let rc = plan
                .operator_env_params
                .rinv_orig
                .unwrap_or([0.0, 0.0, 0.0]);
            (vec![rc[0], rc[1], rc[2]], vec![1.0])
        } else {
            let mut oc = Vec::with_capacity(atoms.len() * 3);
            let mut och = Vec::with_capacity(atoms.len());
            for atom in atoms.iter() {
                oc.extend_from_slice(&atom.coord_bohr);
                och.push(-(atom.atomic_number as f64));
            }
            (oc, och)
        };

        let mut cart_comp = run_1e_giao_nuc_on_backend(
            backend,
            op_kind,
            rank,
            nuc_nroots,
            li as u32,
            lj as u32,
            n_prim_i as u32,
            n_prim_j as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            ri,
            rj,
            drj,
            &exps_i,
            &exps_j,
            &coeff_i,
            &coeff_j,
            &origin_coords,
            &origin_charges,
        );

        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        let not0 = write_giao_complex_staging::<F>(
            plan,
            rank as usize,
            n_ctr_i,
            n_ctr_j,
            nci,
            ncj,
            nsi,
            nsj,
            li,
            lj,
            total_len,
            block_len,
            &cart_comp,
            staging,
        )?;

        let staging_bytes = std::mem::size_of_val(staging);
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
    // Phase 24 Cluster A moment path (r/rr/rrr/rrrr/r2/r4/z/zz + _origj)
    // — rank ∈ {1,3,9,27,81} component-leading output
    // ─────────────────────────────────────────────────────────────────────────
    if let Some((op_mode, moment_order, rank, is_origj)) = moment_dispatch {
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
        // IN-01: `is_origj` comes from the enumerated dispatch tuple, not a
        // string-suffix check.
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

        // WR-05: the inert `complex_output` comptime hint was removed from the moment
        // device path. The moment families emit REAL components on-device; the host
        // materializes re=0/im=value for the safe-API complex_values() view. The future
        // GIAO-on-device path (Phase 30 GIAO×σ) will introduce its own output-convention
        // plumbing when it lands rather than carrying a dead arg through the launcher.
        let mut cart_comp = run_1e_moment_on_backend(
            backend,
            op_mode,
            moment_order,
            rank,
            li as u32,
            lj as u32,
            n_prim_i as u32,
            n_prim_j as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            ri,
            rj,
            drj,
            &exps_i,
            &exps_j,
            &coeff_i,
            &coeff_j,
        );

        // Apply the libcint CINTcommon_fac_sp normalization to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        // IN-03: shared component-leading staging writer (cart/sph copy +
        // fail-closed BufferTooSmall + not0 count).
        let not0 = write_component_leading_staging::<F>(
            plan.representation,
            rank as usize,
            n_ctr_i,
            n_ctr_j,
            nci,
            ncj,
            nsi,
            nsj,
            li,
            lj,
            shell_i.kappa,
            shell_j.kappa,
            total_len,
            block_len,
            &cart_comp,
            staging,
        )?;

        let staging_bytes = std::mem::size_of_val(staging);
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
    // Phase 24 Cluster B: plain single-center 1/r potential (`rinv` / `drinv`)
    // — rank 1 (rinv) / rank 3 (drinv) output. Reads the rinv center (env[4..6],
    //   PTR_RINV_ORIG), charge=+1, NO atom-sum (D-04/OQ-1).
    // ─────────────────────────────────────────────────────────────────────────
    if is_rinv || is_drinv {
        // Spinor rinv/drinv reps are registered for surface completeness but not
        // implemented: fail typed, never partial (D-09).
        if plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name}"),
            });
        }

        // Resolve the single rinv origin up front (env[PTR_RINV_ORIG], read in
        // eval_raw). The validator rejects a None origin before kernel entry; we
        // fail typed here too (never read a garbage origin) — defensive gate.
        let rc = plan
            .operator_env_params
            .rinv_orig
            .ok_or(cintxRsError::InvalidEnvParam {
                param: "PTR_RINV_ORIG",
                reason: format!("int1e_{op_name} kernel reached with no rinv origin"),
            })?;

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let rank: usize = if is_rinv { 1 } else { 3 };
        let block_len = nci * ncj;
        let total_len = rank * block_len;

        // Rys nroots fail-closed guard BEFORE any device/Rys call (D-04).
        //   rinv:  nmax = li+lj   → nroots = (li+lj)/2 + 1
        //   drinv: nmax = li+lj+2 → nroots = (li+lj+2)/2 + 1 (the +1 derivative level
        //          raises the VRR ceiling vs the scalar rinv path).
        let nuc_nroots = if is_rinv {
            (li as u32 + lj as u32) / 2 + 1
        } else {
            (li as u32 + lj as u32 + 2) / 2 + 1
        };
        if nuc_nroots as usize > MAX_DEVICE_NROOTS {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_{op_name} kernel supports nroots<={MAX_DEVICE_NROOTS} \
                     (l_i+l_j<=8); got nroots={nuc_nroots} for l_i={li}, l_j={lj}"
                ),
            });
        }

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        // charge = +1 (single-center 1/r, no -Z_C, no atom-sum — g1e.c:226-228).
        let mut cart_comp = if is_rinv {
            run_1e_rinv_on_backend(
                backend,
                nuc_nroots,
                li as u32,
                lj as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                ri,
                rj,
                rc,
                1.0,
                &exps_i,
                &exps_j,
                &coeff_i,
                &coeff_j,
            )
        } else {
            run_1e_drinv_on_backend(
                backend,
                nuc_nroots,
                li as u32,
                lj as u32,
                n_prim_i as u32,
                n_prim_j as u32,
                n_ctr_i as u32,
                n_ctr_j as u32,
                ri,
                rj,
                rc,
                1.0,
                &exps_i,
                &exps_j,
                &coeff_i,
                &coeff_j,
            )
        };

        // Apply the libcint CINTcommon_fac_sp normalization to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        // IN-03: shared component-leading staging writer.
        let not0 = write_component_leading_staging::<F>(
            plan.representation,
            rank,
            n_ctr_i,
            n_ctr_j,
            nci,
            ncj,
            nsi,
            nsj,
            li,
            lj,
            shell_i.kappa,
            shell_j.kappa,
            total_len,
            block_len,
            &cart_comp,
            staging,
        )?;

        let staging_bytes = std::mem::size_of_val(staging);
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
    // Phase 24 Cluster C: `int1e_p4` (∇⁴, rank 1) — Laplacian-of-Laplacian on
    // the overlap G-tensor with BOTH-side +2 headroom (ng={2,2,...}), no Rys.
    // ─────────────────────────────────────────────────────────────────────────
    if is_p4 {
        // Spinor p4 reps are registered for surface completeness but not
        // implemented: fail typed, never partial (D-09).
        if plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name}"),
            });
        }

        // Internal G-tensor ceiling: nmax = li + lj + 4 (BOTH-side +2 headroom).
        // The overlap-derivative engine supports li+lj<=8 (MAX VRR). Fail closed
        // (UnsupportedApi) if a corpus shell would exceed it — NEVER truncate
        // (T-24-04-02). On STO-3G (li,lj<=1) nmax<=6, well within the limit.
        if li as u32 + lj as u32 + 4 > 8 {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_p4 kernel supports l_i+l_j+4<=8 (both-side \
                     headroom); got l_i={li}, l_j={lj}"
                ),
            });
        }

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let block_len = nci * ncj;
        // rank 1: total_len == block_len.
        let total_len = block_len;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        let mut cart_comp = run_1e_p4_on_backend(
            backend,
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
        );

        // Apply the libcint CINTcommon_fac_sp normalization to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        // IN-03: shared component-leading staging writer (p4 rank = 1).
        let not0 = write_component_leading_staging::<F>(
            plan.representation,
            1,
            n_ctr_i,
            n_ctr_j,
            nci,
            ncj,
            nsi,
            nsj,
            li,
            lj,
            shell_i.kappa,
            shell_j.kappa,
            total_len,
            block_len,
            &cart_comp,
            staging,
        )?;

        let staging_bytes = std::mem::size_of_val(staging);
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
    // Phase 24 Cluster D irp path (`int1e_irp` = i·r×∇) — rank-9 (3×3 ∇⊗r)
    // — reads the gauge origin env[PTR_COMMON_ORIG] via drj = rj - common_orig
    // ─────────────────────────────────────────────────────────────────────────
    if is_irp || is_ipipr {
        // Spinor irp reps are registered for surface completeness but not
        // implemented: fail typed, never partial (D-09).
        if is_irp && plan.representation == Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("spinor int1e_{op_name}"),
            });
        }

        // Internal G-tensor ceiling: nmax = li + lj + 2 (ket +2 headroom). The
        // overlap-derivative engine supports li+lj<=8. Fail closed (UnsupportedApi)
        // if a corpus shell would exceed it — NEVER truncate (T-24-05-02). On
        // STO-3G (li,lj<=1) nmax<=4, well within the limit.
        let op_kind = u32::from(is_ipipr);
        let headroom = if is_ipipr { 3 } else { 2 };
        let rank = if is_ipipr { 27 } else { 9 };
        if li as u32 + lj as u32 + headroom > 8 {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_{op_name} kernel supports l_i+l_j+headroom<=8; \
                     got l_i={li}, l_j={lj}, headroom={headroom}"
                ),
            });
        }

        // Gauge origin (D-06): irp reads env[PTR_COMMON_ORIG] via libcint's G1E_RCJ
        // (drj = rj - common_orig). Defaults to [0,0,0] when unset (the gate fixture
        // sets a NON-ZERO origin so this path is genuinely exercised). T-24-05-01:
        // the env slot is finiteness-validated upstream (validator.rs) — no new gap.
        let origin: [f64; 3] = plan.operator_env_params.common_orig.unwrap_or([0.0; 3]);
        let drj = [rj[0] - origin[0], rj[1] - origin[1], rj[2] - origin[2]];

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let block_len = nci * ncj;
        let total_len = rank * block_len;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        let mut cart_9comp = run_1e_irp_on_backend(
            backend,
            li as u32,
            lj as u32,
            n_prim_i as u32,
            n_prim_j as u32,
            n_ctr_i as u32,
            n_ctr_j as u32,
            ri,
            rj,
            drj,
            &exps_i,
            &exps_j,
            &coeff_i,
            &coeff_j,
            op_kind,
        );

        // Apply the libcint CINTcommon_fac_sp normalization to all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart_9comp.iter_mut() {
                *v *= sp_scale;
            }
        }

        // IN-03: shared component-leading staging writer (irp rank = 9).
        let not0 = write_component_leading_staging::<F>(
            plan.representation,
            rank,
            n_ctr_i,
            n_ctr_j,
            nci,
            ncj,
            nsi,
            nsj,
            li,
            lj,
            shell_i.kappa,
            shell_j.kappa,
            total_len,
            block_len,
            &cart_9comp,
            staging,
        )?;

        let staging_bytes = std::mem::size_of_val(staging);
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
    // Both-side rank-9 gradient path (`ipovlpip` / `ipkinip` / `ipnucip` /
    // `iprinvip`)
    // — 9-component output
    // ─────────────────────────────────────────────────────────────────────────
    if is_rank9_both {
        // 27-03 (FND-04): the rank-9 both-side ip-family spinor rep now folds
        // via the centralized derivative wrapper (D-06 transpose owned inside it).

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
        if (is_ipnucip || is_iprinvip) && nuc_nroots_both as usize > MAX_DEVICE_NROOTS {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_{op_name} kernel supports nroots<={MAX_DEVICE_NROOTS}; \
                     got nroots={nuc_nroots_both} for l_i={li}, l_j={lj}"
                ),
            });
        }

        let mut cart_9comp = if is_ipovlpip {
            run_1e_grad_both_on_backend(
                backend,
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
            )
        } else if is_ipkinip {
            run_1e_grad_kin_both_on_backend(
                backend,
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
            )
        } else {
            // ipnucip sums -Z_C over every atom. iprinvip uses one +1/r center
            // from env[PTR_RINV_ORIG], matching CINT1e_drv type=1.
            let (origin_coords, origin_charges) = if is_iprinvip {
                let origin = plan.operator_env_params.rinv_orig.unwrap_or([0.0; 3]);
                (origin.to_vec(), vec![1.0])
            } else {
                let mut coords = Vec::with_capacity(atoms.len() * 3);
                let mut charges = Vec::with_capacity(atoms.len());
                for atom in atoms.iter() {
                    coords.extend_from_slice(&atom.coord_bohr);
                    charges.push(-(atom.atomic_number as f64));
                }
                (coords, charges)
            };
            run_1e_nuc_grad_both_on_backend(
                backend,
                nuc_nroots_both,
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
                &origin_coords,
                &origin_charges,
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
                                    staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
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
                                    staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                }
                            }
                        }
                    }
                }
            }
            // 27-03 (FND-04): fold via the centralized derivative wrapper
            // (ncomp=9, lock component_rank for ipovlpip/ipkinip/ipnucip).
            // The wrapper owns the KET→BRA transpose (D-06).
            Representation::Spinor => {
                cart_to_spinor_sf_derivative_2d::<F>(
                    staging,
                    &cart_9comp,
                    9,
                    li,
                    shell_i.kappa,
                    lj,
                    shell_j.kappa,
                    n_ctr_i,
                    n_ctr_j,
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
    // Phase 25 HESS-04 3rd/4th-order path (deriv3.c rank 27, deriv4.c rank 81):
    // ipipipnuc/ipipiprinv/ipipnucip/ipiprinvip (27),
    // ipipipiprinv/ipiprinvipip/ipipiprinvip (81). HOST-routed (FND-02): the
    // bra/ket +2/+3 headroom can elevate the nuclear Rys nroots beyond the device
    // MAX_DEVICE_NROOTS=5 cap; the host `rys_roots_host` Wheeler path serves 6..12.
    // ─────────────────────────────────────────────────────────────────────────
    if is_deriv34 {
        use crate::kernels::deriv34::{contract_deriv34_block, deriv34_rank, nuclear_origins};

        // 27-03 (FND-04): the rank-27/81 deriv3/deriv4 ip-family spinor rep now
        // folds via the centralized derivative wrapper (ncomp = lock rank).

        let rank = deriv34_rank(op_name);

        // Build the (origin, charge_factor) Coulomb-center list: nuclear families
        // sum over all nuclei (-Z_C); rinv families use the single rinv origin.
        let is_nuc_family = matches!(
            op_name,
            "ipipipnuc" | "ipipnucip" | "ippnucp" | "ippnucpip" | "ipippnucp" | "pnucp"
        );
        let origins: Vec<([f64; 3], f64)> = if is_nuc_family {
            nuclear_origins(atoms)
        } else {
            let origin =
                plan.operator_env_params
                    .rinv_orig
                    .ok_or(cintxRsError::InvalidEnvParam {
                        param: "PTR_RINV_ORIG",
                        reason: format!("{op_name} kernel reached with no rinv origin"),
                    })?;
            vec![(origin, 1.0)]
        };

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        // Fail closed BEFORE any Rys call (FND-02): the host ceiling is nroots<=12.
        // nmax = (li+i_inc) + (lj+j_inc); the deriv3/deriv4 i_inc/j_inc are <=3/<=2.
        let nuc_nroots = (li as u32 + lj as u32 + 5) / 2 + 1;
        if nuc_nroots as usize > HOST_RYS_NROOTS_CEILING_1E {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "host int1e_{op_name} supports nroots<={HOST_RYS_NROOTS_CEILING_1E}; \
                     got nroots={nuc_nroots} for l_i={li}, l_j={lj}"
                ),
            });
        }

        let mut cart = contract_deriv34_block(
            op_name, li, lj, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j, n_ctr_i, n_ctr_j,
            &origins,
        )
        .ok_or_else(|| cintxRsError::UnsupportedApi {
            requested: format!("1e operator '{op_name}' is not a deriv34 family"),
        })?;

        // sp normalization scale on all components.
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in cart.iter_mut() {
                *v *= sp_scale;
            }
        }

        let block_len = nci * ncj;
        let total_len = rank * block_len;

        match plan.representation {
            Representation::Spheric => {
                let ni_sph = n_ctr_i * nsi;
                let nj_sph = n_ctr_j * nsj;
                let sph_block = ni_sph * nj_sph;
                for comp in 0..rank {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let cart_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let mut sph_tmp = vec![0.0_f64; nsi * nsj];
                            cart_to_sph_1e(
                                &cart[cart_base..cart_base + block_len],
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
                                    staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
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
                for comp in 0..rank {
                    for ci in 0..n_ctr_i {
                        for cj in 0..n_ctr_j {
                            let src_base = (ci * n_ctr_j + cj) * total_len + comp * block_len;
                            let block = &cart[src_base..src_base + block_len];
                            let staging_comp_base = comp * cart_block;
                            for jc in 0..ncj {
                                let jj = cj * ncj + jc;
                                for ic in 0..nci {
                                    let ii = ci * nci + ic;
                                    let dst = staging_comp_base + ii + jj * ni_cart;
                                    staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                }
                            }
                        }
                    }
                }
            }
            // 27-03 (FND-04): fold via the centralized derivative wrapper
            // (ncomp = deriv34 lock rank 27/81). Wrapper owns KET→BRA (D-06).
            Representation::Spinor => {
                cart_to_spinor_sf_derivative_2d::<F>(
                    staging,
                    &cart,
                    rank,
                    li,
                    shell_i.kappa,
                    lj,
                    shell_j.kappa,
                    n_ctr_i,
                    n_ctr_j,
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
    // Phase 25 HESS-01 bra-only ∇² rank-9 path (ipipovlp / ipipnuc / ipipkin /
    // ipiprinv) — 9-component output, ng={2,0,...} (ipipkin {2,2,...}).
    // ─────────────────────────────────────────────────────────────────────────
    if is_rank9_bra {
        // 27-03 (FND-04): the rank-9 bra-only Hessian ip-family spinor rep now
        // folds via the centralized derivative wrapper (ncomp=9, D-06 transpose).

        // iprinv: resolve the single rinv origin (validator should have rejected
        // a None origin; fail typed rather than read garbage).
        let ipiprinv_origin: Option<[f64; 3]> = if is_ipiprinv {
            Some(
                plan.operator_env_params
                    .rinv_orig
                    .ok_or(cintxRsError::InvalidEnvParam {
                        param: "PTR_RINV_ORIG",
                        reason: "ipiprinv kernel reached with no rinv origin".to_owned(),
                    })?,
            )
        } else {
            None
        };

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

        // Nuclear/rinv Rys nroots: bra +2 headroom → nmax=li+lj+2, nroots=nmax/2+1.
        // Fail closed before any device/Rys call (FND-02 routes nroots≥6 to host;
        // the device comptime kernel caps at MAX_DEVICE_NROOTS).
        let nuc_nroots = (li as u32 + lj as u32 + 2) / 2 + 1;
        if (is_ipipnuc || is_ipiprinv) && nuc_nroots as usize > MAX_DEVICE_NROOTS {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "device int1e_{op_name} kernel supports nroots<={MAX_DEVICE_NROOTS}; \
                     got nroots={nuc_nroots} for l_i={li}, l_j={lj}"
                ),
            });
        }

        let mut cart_9comp = if is_ipipovlp {
            run_1e_gradgrad_bra_ovlp_on_backend(
                backend,
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
            )
        } else if is_ipipkin {
            run_1e_gradgrad_bra_kin_on_backend(
                backend,
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
            )
        } else if is_ipipnuc {
            // ipipnuc: ∑_C (-Z_C)·∇²-bra over ALL nuclei, low→high (D-10).
            let mut origin_coords = Vec::with_capacity(atoms.len() * 3);
            let mut origin_charges = Vec::with_capacity(atoms.len());
            for atom in atoms.iter() {
                origin_coords.extend_from_slice(&atom.coord_bohr);
                origin_charges.push(-(atom.atomic_number as f64));
            }
            run_1e_nuc_gradgrad_bra_on_backend(
                backend,
                nuc_nroots,
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
                &origin_coords,
                &origin_charges,
            )
        } else {
            // ipiprinv: single rinv origin, factor +1.0, no -Z_C.
            let origin = ipiprinv_origin.expect("ipiprinv origin resolved above");
            run_1e_nuc_gradgrad_bra_on_backend(
                backend,
                nuc_nroots,
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
                &[origin[0], origin[1], origin[2]],
                &[1.0],
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
                                    staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
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
                                    staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                }
                            }
                        }
                    }
                }
            }
            // 27-03 (FND-04): fold via the centralized derivative wrapper
            // (ncomp=9, lock rank for ipipovlp/ipipnuc/ipipkin/ipiprinv).
            Representation::Spinor => {
                cart_to_spinor_sf_derivative_2d::<F>(
                    staging,
                    &cart_9comp,
                    9,
                    li,
                    shell_i.kappa,
                    lj,
                    shell_j.kappa,
                    n_ctr_i,
                    n_ctr_j,
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
    // Group-4 one-electron sigma families, including the gradient/Hessian gap
    // closure families. libcint only exposes these through its spinor driver.
    if is_sigma_family && !is_sp {
        if plan.representation != Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("int1e_{op_name} is Spinor-only"),
            });
        }

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;
        let exps_i = &shell_i.exponents[..n_prim_i];
        let exps_j = &shell_j.exponents[..n_prim_j];
        let coeff_i = &shell_i.coefficients[..n_prim_i * n_ctr_i];
        let coeff_j = &shell_j.coefficients[..n_prim_j * n_ctr_j];

        let is_rinv = matches!(
            op_name,
            "sprinvsp" | "ipsprinvsp" | "ipipsprinvsp" | "ipsprinvspip"
        );
        let (origin_coords, origin_charges) = if is_rinv {
            let origin = plan.operator_env_params.rinv_orig.unwrap_or([0.0; 3]);
            (origin.to_vec(), vec![1.0])
        } else {
            let mut coords = Vec::with_capacity(atoms.len() * 3);
            let mut charges = Vec::with_capacity(atoms.len());
            for atom in atoms.iter() {
                coords.extend_from_slice(&atom.coord_bohr);
                charges.push(-(atom.atomic_number as f64));
            }
            (coords, charges)
        };

        crate::kernels::sigma_1e::launch_int1e_sigma_family_spinor_pair(
            backend,
            op_name,
            li,
            shell_i.kappa,
            lj,
            shell_j.kappa,
            n_prim_i,
            n_prim_j,
            n_ctr_i,
            n_ctr_j,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            &origin_coords,
            &origin_charges,
            staging,
        )?;

        let nonzero_threshold = F::from_f64_lossy(if F::PRECISION == PrecisionKind::F32 {
            1e-12
        } else {
            1e-18
        });
        let not0 = staging
            .iter()
            .filter(|&&value| value.abs() > nonzero_threshold)
            .count() as i32;
        let staging_bytes = std::mem::size_of_val(staging);
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

    // Phase 28 FND-05: int1e_sp Spinor path — σ·p assembler → cart_to_spinor_si_2d
    //
    // The σ·p device assembler (`run_sigma_p_on_backend`, tensor_rank=1) emits the
    // four component-leading gc_x/gc_y/gc_z/gc_1 cart blocks per (ci,cj). The host
    // si_2d transform owns the KET→BRA transpose and reduces them to the spinor
    // block (Pauli-σ bra mix + ordinary ket). nctr>1 is HANDLED: loop the (ci,cj)
    // contraction pairs and scatter each di*dj*2 sub-block into the
    // contraction-major spinor AO grid. int1e_sp is Spinor-only (D-02 vehicle);
    // Spheric/Cart are rejected (no public cart/sph int1e_sp this phase, D-01).
    // ─────────────────────────────────────────────────────────────────────────
    if is_sp {
        if plan.representation != Representation::Spinor {
            return Err(cintxRsError::UnsupportedApi {
                requested: "int1e_sp is Spinor-only (the FND-05 σ·p proof vehicle); \
                            cart/spheric int1e_sp is not registered this phase (D-01)"
                    .to_owned(),
            });
        }

        let n_prim_i = shell_i.nprim as usize;
        let n_prim_j = shell_j.nprim as usize;
        let n_ctr_i = shell_i.nctr as usize;
        let n_ctr_j = shell_j.nctr as usize;
        let block_len = nci * ncj;
        // tensor_rank=1 → 4 gc blocks (gc_x/gc_y/gc_z/gc_1) per (ci,cj).
        let total_len = 4 * block_len;

        let exps_i: Vec<f64> = shell_i.exponents[..n_prim_i].to_vec();
        let exps_j: Vec<f64> = shell_j.exponents[..n_prim_j].to_vec();
        let coeff_i: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
        let coeff_j: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();

        // Device σ·p assembler: 4 component-leading KET-major gc blocks per (ci,cj).
        let mut gc = crate::kernels::sigma_p::run_sigma_p_on_backend(
            backend,
            1, // tensor_rank = 1 (int1e_sp: 3 Pauli + 1 zero scalar slot)
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
        )?;

        // s/p normalization scale (matches the scalar/gradient 1e arms).
        let sp_scale = common_fac_sp(li) * common_fac_sp(lj);
        if (sp_scale - 1.0).abs() > 1e-15 {
            for v in gc.iter_mut() {
                *v *= sp_scale;
            }
        }

        let kappa_i = shell_i.kappa;
        let kappa_j = shell_j.kappa;
        let di = spinor_len(li, kappa_i as i32);
        let dj = spinor_len(lj, kappa_j as i32);
        let ni_sp = n_ctr_i * di;
        let nj_sp = n_ctr_j * dj;

        // Fail-closed staging guard (T-28-01-01, OOM-safe stop contract): refuse
        // before any write if the caller workspace cannot hold the full spinor
        // block, mirroring launch_int1e_sp_spinor_pair. Prevents a panic mid-
        // scatter from leaving a partial write in `staging`.
        let staging_required = ni_sp * nj_sp * 2;
        if staging.len() < staging_required {
            return Err(cintxRsError::BufferTooSmall {
                required: staging_required,
                provided: staging.len(),
            });
        }

        // Fold + scatter per (ci,cj). cart_to_spinor_si_2d OWNS the KET→BRA
        // transpose, so the gc blocks are passed as the assembler emits them
        // (KET-major) — no launcher transpose (do NOT copy the sf_2d single-block
        // launcher transpose, and do NOT copy the sf_2d nctr>1 rejection).
        let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];
        for ci in 0..n_ctr_i {
            for cj in 0..n_ctr_j {
                let base = (ci * n_ctr_j + cj) * total_len;
                let gc_x = &gc[base..base + block_len];
                let gc_y = &gc[base + block_len..base + 2 * block_len];
                let gc_z = &gc[base + 2 * block_len..base + 3 * block_len];
                let gc_1 = &gc[base + 3 * block_len..base + 4 * block_len];

                cart_to_spinor_si_2d::<F>(
                    &mut scratch,
                    gc_x,
                    gc_y,
                    gc_z,
                    gc_1,
                    li,
                    kappa_i,
                    lj,
                    kappa_j,
                )?;

                // scratch is column-major: scratch[(j*di + i)*2 + {re,im}].
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = (j_global * ni_sp + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
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
        let nuc_nroots = (li as u32 + lj as u32).div_ceil(2) + 1;
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
                backend,
                0,
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
            )
        } else if is_ipkin {
            run_1e_grad_bra_on_backend(
                backend,
                1,
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
                backend,
                nuc_nroots,
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
                &origin_coords,
                &origin_charges,
            )
        } else {
            // iprinv: single rinv origin, factor +1.0, no -Z_C (D-08).
            let origin = iprinv_origin.expect("iprinv origin resolved above");
            run_1e_nuc_grad_on_backend(
                backend,
                nuc_nroots,
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
                &[origin[0], origin[1], origin[2]],
                &[1.0],
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
                                    staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
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
                                    staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
                                }
                            }
                        }
                    }
                }
            }
            // 27-03 (FND-04 / D-06 / D-08): the rank-3 ip-gradient spinor fold is
            // now a single call into the centralized derivative wrapper. The
            // wrapper owns the KET→BRA transpose (no launcher-owned transpose
            // remains) AND supports general contraction (nctr>1) contraction-major,
            // so the previous nctr>1 rejection is dropped.
            Representation::Spinor => {
                cart_to_spinor_sf_derivative_2d::<F>(
                    staging,
                    &cart_3comp,
                    3,
                    li,
                    shell_i.kappa,
                    lj,
                    shell_j.kappa,
                    n_ctr_i,
                    n_ctr_j,
                )?;
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

        let staging_bytes = std::mem::size_of_val(staging);
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

    if op_kind == 2 && nroots as usize > 12 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_1e",
            detail: format!(
                "1e nuclear kernel supports nroots<=12; \
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

    // Dispatch the scalar arm onto the resolved backend's device client (f64)
    // or the host loop. Task 33-03: the boundary is the family's ceiling, not a
    // constant — with `int1e` flipped onto the inline extended entry and this
    // backend's FMA probe passing, orders 6..=12 stay on the device.
    let scalar_ceiling = crate::device_rys_ceiling::device_nroots_ceiling(
        backend,
        crate::device_rys_ceiling::RysFamily::Int1e,
    );
    let cart_blocks = if op_kind == 2 && nroots as usize > scalar_ceiling {
        let nci = ncart(li);
        let ncj = ncart(lj);
        let block_len = nci * ncj;
        let mut cart_accum = vec![0.0f64; n_ctr_i * n_ctr_j * block_len];
        for pi in 0..n_prim_i {
            let ai = exps_i[pi];
            for pj in 0..n_prim_j {
                let aj = exps_j[pj];
                let pd = crate::math::pdata::compute_pdata_host(
                    ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0,
                );
                let nuc_prim = contract_nuclear(&pd, ri, rj, li, lj, atoms);
                for ci in 0..n_ctr_i {
                    let ci_coeff = coeff_i[pi * n_ctr_i + ci];
                    for cj in 0..n_ctr_j {
                        let cj_coeff = coeff_j[pj * n_ctr_j + cj];
                        let pair_weight = ci_coeff * cj_coeff;
                        let block_offset = (ci * n_ctr_j + cj) * block_len;
                        for idx in 0..block_len {
                            cart_accum[block_offset + idx] += pair_weight * nuc_prim[idx];
                        }
                    }
                }
            }
        }
        cart_accum
    } else {
        run_1e_scalar_on_backend(
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
        )
    };
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
                            staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
                        }
                    }
                }
            }
        }
        Representation::Spinor => {
            // General-contraction (nctr>1) spin-free 1e cart→spinor (260601-aty).
            // The device scalar kernel already accumulated every (ci,cj) block with
            // ITS OWN per-column coefficients (out_total = nctr_i*nctr_j*block_len),
            // exactly as the Spheric/Cart arms above consume. This arm therefore only
            // transforms + scatters the already-contracted per-column blocks — it does
            // NOT re-apply coefficients. We mirror the Spheric arm's per-(ci,cj) loop
            // and the proven σ·p (si) nctr>1 contraction-major spinor scatter
            // (dst = (j_global*ni_sp + i_global)*2, ni_sp = n_ctr_i*di), the only
            // difference being that cart_to_spinor_sf_2d does NOT own the ket→bra
            // transpose (unlike cart_to_spinor_si_2d), so we keep the per-sub-block
            // ket-major→bra-major transpose the single-block arm carried (260529-jtd/kke).
            //
            // The device scalar kernel emits each (ci,cj) block ket-major / bra-fastest
            // (block[cj*nci + ci]), but cart_to_spinor_sf_2d reads bra-major /
            // ket-fastest (cart[bra*ncj + ket]). For square symmetric blocks (an s side,
            // or an intrinsically transpose-symmetric overlap p×p block) the transpose
            // is a no-op — which is why a NON-SQUARE asymmetric p×d cross block is the
            // configuration that surfaces the orientation, and the nctr>1 fixture is
            // built that way.
            let kappa_i = shell_i.kappa;
            let kappa_j = shell_j.kappa;
            let di = spinor_len(li, kappa_i as i32);
            let dj = spinor_len(lj, kappa_j as i32);
            let ni_sp = n_ctr_i * di; // dense bra spinor dim (contraction-major)
            let nj_sp = n_ctr_j * dj; // dense ket spinor dim

            // Fail-closed staging guard (T-aty-03, OOM-safe stop contract): refuse
            // before any write if the caller workspace cannot hold the full dense
            // interleaved-complex spinor block. Prevents a partial write on nctr>1.
            let staging_required = ni_sp * nj_sp * 2;
            if staging.len() < staging_required {
                return Err(cintxRsError::BufferTooSmall {
                    required: staging_required,
                    provided: staging.len(),
                });
            }

            let mut cart_bra_major = vec![0.0f64; nci * ncj];
            let mut tmp = vec![F::from_f64_lossy(0.0); di * dj * 2];
            for ci in 0..n_ctr_i {
                for cj in 0..n_ctr_j {
                    let base = (ci * n_ctr_j + cj) * block_len;
                    let block = &cart_blocks[base..base + block_len];
                    // ket-major (block[jc*nci + ic]) → bra-major (cart[ic*ncj + jc]).
                    for ic in 0..nci {
                        for jc in 0..ncj {
                            cart_bra_major[ic * ncj + jc] = block[jc * nci + ic];
                        }
                    }
                    cart_to_spinor_sf_2d::<F>(&mut tmp, &cart_bra_major, li, kappa_i, lj, kappa_j)?;
                    // tmp is column-major interleaved: tmp[(j_sp*di + i_sp)*2 + {re,im}].
                    // Scatter contraction-major into the dense spinor AO grid.
                    for j_sp in 0..dj {
                        let j_global = cj * dj + j_sp;
                        for i_sp in 0..di {
                            let i_global = ci * di + i_sp;
                            let src = (j_sp * di + i_sp) * 2;
                            let dst = (j_global * ni_sp + i_global) * 2;
                            staging[dst] = tmp[src];
                            staging[dst + 1] = tmp[src + 1];
                        }
                    }
                }
            }
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
                            staging[dst] = F::from_f64_lossy(block[jc * nci + ic]);
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
    /// A CPU `ResolvedBackend` for the 1e derivative device tests.
    ///
    /// The batched entry points take a backend rather than a raw client because
    /// they upload the flattened basis through the backend's arm.
    fn test_cpu_backend_1e() -> ResolvedBackend {
        ResolvedBackend::from_intent(&cintx_runtime::BackendIntent {
            backend: cintx_runtime::BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend")
    }

    fn cpu_client_1e() -> cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::Runtime;
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// The same client as [`cpu_client_1e`], as a [`ResolvedBackend`].
    ///
    /// The batched entry points take a backend rather than a typed client,
    /// because the five-arm runtime match now lives inside one dispatcher
    /// instead of once per family (Task 35-D).
    #[cfg(feature = "cpu")]
    fn cpu_backend_1e() -> ResolvedBackend {
        ResolvedBackend::Cpu(cpu_client_1e())
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
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nmax = (li + lj) as u32 + 1;
            let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32);
            let host = contract_grad_1e_bra(&g, li, lj, nmax, ai);
            let dev = run_1e_grad_bra_on_backend(
                &test_cpu_backend_1e(),
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
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nmax = (li + lj) as u32 + 3;
            let g = fill_g_tensor_overlap(&pd, ri, rj, nmax, lj as u32 + 2);
            let host = contract_ipkin(&g, li, lj, nmax, ai, aj);
            let dev = run_1e_grad_bra_on_backend(
                &test_cpu_backend_1e(),
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

            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
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

            // Task 35-D wave 3: the per-pair entry point is now a one-pair
            // batch through the same kernel, so this unit test covers the
            // batched path rather than a second implementation of it.
            let dev = run_1e_grad_both_on_backend(
                &cpu_backend_1e(),
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
    fn host_dj_1e(
        src: &[f64],
        gpa: usize,
        dj: usize,
        jmax: usize,
        imax: usize,
        aj2: f64,
    ) -> Vec<f64> {
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
    fn host_di_1e(
        src: &[f64],
        gpa: usize,
        dj: usize,
        jmax: usize,
        imax: usize,
        ai2: f64,
    ) -> Vec<f64> {
        let mut dst = vec![0.0f64; 3 * gpa];
        for axis in 0..3 {
            let off = axis * gpa;
            for jn in 0..=jmax {
                let jbase = jn * dj;
                dst[off + jbase] = ai2 * src[off + jbase + 1];
                for ii in 1..=imax {
                    dst[off + jbase + ii] =
                        (ii as f64) * src[off + jbase + ii - 1] + ai2 * src[off + jbase + ii + 1];
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
            // Task 35-D wave 3: the per-pair entry point is a one-pair batch
            // through the same kernel, so this covers the batched path.
            let dev = run_1e_grad_kin_both_on_backend(
                &cpu_backend_1e(),
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
            // Task 35-D wave 3: the per-pair entry point is a one-pair batch
            // through the same kernel, so this covers the batched path.
            let dev = run_1e_nuc_grad_both_on_backend(
                &cpu_backend_1e(),
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
        let origins: [([f64; 3], f64); 2] = [([0.0, 0.0, 0.0], -8.0), ([0.4, 0.3, 0.9], -1.0)];
        let origin_coords: Vec<f64> = origins
            .iter()
            .flat_map(|(c, _)| c.iter().copied())
            .collect();
        let origin_charges: Vec<f64> = origins.iter().map(|(_, q)| *q).collect();

        // Keep li+lj<=3 so nrys = (li+lj+1)/2+1 <= 2 (matches the scalar nuclear
        // cross-check bound; rys_roots_host covers nrys<=5 but the host reference
        // exercises the device-supported grid).
        for &(li, lj) in &[(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nroots = (li as u32 + lj as u32).div_ceil(2) + 1;
            let host = contract_nuclear_grad(&pd, ri, rj, li, lj, ai, &origins);
            let dev = run_1e_nuc_grad_on_backend(
                &test_cpu_backend_1e(),
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
            let pd = compute_pdata_host(ai, aj, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], 1.0, 1.0);
            let nroots = (li as u32 + lj as u32).div_ceil(2) + 1;
            let host = contract_nuclear_grad(&pd, ri, rj, li, lj, ai, &iprinv_origins);
            let dev = run_1e_nuc_grad_on_backend(
                &test_cpu_backend_1e(),
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
        // Flattened two-shell basis: one primitive, one contraction each.
        let exps = [1.0_f32, 1.0];
        let coeffs = [1.0_f32, 1.0];
        let centers = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 1.4];
        let shell_meta: [u32; 8] = [0, 0, 1, 1, 1, 1, 1, 1];
        // `[si, sj, out_off, class]` — one class, index 0.
        let pairs: [u32; 4] = [0, 1, 0, 0];
        let class_shape: [u32; ONE_E_SHAPE_STRIDE] = [0, 0];
        let coords = [0.0_f32]; // unused for overlap, must be len>0
        let charges = [0.0_f32];
        // overlap s-s: nmax=0, lj_ext=0, g_per_axis=1 → 3 g elements; out_len=1.
        let g_zero = [0.0_f32; 3];
        let out_zero = [0.0_f32; 1];

        let exps_h = client.create_from_slice(f32::as_bytes(&exps));
        let coeffs_h = client.create_from_slice(f32::as_bytes(&coeffs));
        let centers_h = client.create_from_slice(f32::as_bytes(&centers));
        let meta_h = client.create_from_slice(u32::as_bytes(&shell_meta));
        let pairs_h = client.create_from_slice(u32::as_bytes(&pairs));
        let shape_h = client.create_from_slice(u32::as_bytes(&class_shape));
        let coords_h = client.create_from_slice(f32::as_bytes(&coords));
        let charges_h = client.create_from_slice(f32::as_bytes(&charges));
        let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
        let out_h = client.create_from_slice(f32::as_bytes(&out_zero));
        // The extended-Rys tables are an unconditional kernel argument; an
        // overlap smoke launch never reads them.
        let rys_tables = crate::math::rys_wheeler::ext_rys_tables();
        let rys_tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));

        one_electron_scalar_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
            &client,
            crate::plane::single_cube_count(),
            crate::plane::cooperative_cube_dim::<cubecl::cpu::CpuRuntime>(&client, 1),
            unsafe { ArrayArg::from_raw_parts(exps_h, exps.len()) },
            unsafe { ArrayArg::from_raw_parts(coeffs_h, coeffs.len()) },
            unsafe { ArrayArg::from_raw_parts(centers_h, centers.len()) },
            unsafe { ArrayArg::from_raw_parts(meta_h, shell_meta.len()) },
            unsafe { ArrayArg::from_raw_parts(pairs_h, pairs.len()) },
            unsafe { ArrayArg::from_raw_parts(shape_h, class_shape.len()) },
            unsafe { ArrayArg::from_raw_parts(coords_h, 1) },
            unsafe { ArrayArg::from_raw_parts(charges_h, 1) },
            unsafe { ArrayArg::from_raw_parts(rys_tab_h, EXT_TABLES_LEN) },
            unsafe { ArrayArg::from_raw_parts(g_h, 3) },
            unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
            PIE4 as f32,
            0.0_f32, // prim_tol: the f32 smoke launch never screens
            SQRTPI as f32,
            std::f64::consts::PI as f32,
            0,    // natm (unused for overlap)
            1u32, // n_pairs
            1u32, // n_cubes
            3u32, // g_stride (one slab, unpadded)
            0u32, // op_kind = overlap
            1u32, // nroots
            // Cooperative decomposition: one cube, one pair, the shape this
            // test's single 3-element slab is sized for.
            0u32,
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
    // now EVALUATES via the centralized derivative wrapper (27-03 / D-08). The
    // prior UnsupportedApi rejection is gone: nctr>1 composes contraction-major
    // inside cart_to_spinor_sf_derivative_2d.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    #[test]
    fn test_ipovlp_spinor_grad_nctr_gt1_evaluates() {
        // l=0, nctr=2: di=dj=spinor_len(0,0)=2, ni_full=nj_full=2*2=4,
        // spinor_block=4*4*2=32, ncomp=3 → required staging = 96 (256 is ample).
        let result = run_forced_spinor_grad("int1e_ipovlp_sph", 0, 2, None, 256);
        let staging =
            result.expect("spinor ipovlp gradient with nctr>1 should now evaluate (D-08)");
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor ipovlp gradient (nctr>1) staging is all-zero"
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
        let result = run_forced_spinor_grad("int1e_iprinv_sph", 0, 1, Some([0.0, 0.0, 0.0]), 24);
        let staging = result.expect("spinor iprinv gradient (nctr=1) should evaluate");
        assert!(
            staging.iter().any(|v| v.abs() > 1e-14),
            "spinor iprinv gradient staging is all-zero"
        );
    }
}
