//! 4c1e (four-center one-electron overlap) integral kernel.
//!
//! Implements the libcint `g4c1e.c` polynomial 1D recurrence pipeline:
//! 1. Polynomial recurrence fill (NOT Rys quadrature — nroots=1 always).
//! 2. Branch-specific 4D HRR transfer reusing the same 4-branch selection as 2e.
//! 3. Cartesian contraction + optional `cart_to_sph_2e` transform.
//!
//! # Execution model (CubeCL device dispatch)
//!
//! The numeric core — the per-primitive-quartet polynomial G-tensor fill, the
//! 4-branch HRR, and the Cartesian contraction — runs as a real CubeCL
//! `#[cube(launch)]` kernel ([`center_4c1e_kernel`]) **generic over `F: Float`**,
//! dispatched onto the resolved backend's `ComputeClient` (CPU `CpuRuntime`,
//! ROCm `HipRuntime`, …) via [`run_4c1e_device`]. The Cartesian buffer is read
//! back to the host and the `cart_to_sph_2e` transform (whose coefficient tables
//! are host-only) finishes on the host.
//!
//! # Algorithm difference from 2e
//! The 2e kernel uses Rys quadrature roots to fill the G-tensor.
//! The 4c1e kernel uses a polynomial 1D recurrence from g4c1e.c:
//!   `buf[0] = 1`, `buf[1] = -r1r12 * buf[0]`,
//!   `buf[i+1] = 0.5*i/aijkl * buf[i-1] - r1r12 * buf[i]`
//! with `nroots = 1` hardcoded (no quadrature roots needed).
//!
//! ## Precision policy
//!
//! The kernel is genuinely generic over `F: Float`, but the launcher runs it at
//! **f64** on-device for both `PrecisionKind` variants and casts the read-back
//! buffer to `F` at the c2s/output stage via `F::from_f64_lossy`. This preserves
//! the historical "intermediates in f64, output cast to `F`" contract that the
//! f32 parity gate is calibrated against, while moving the real arithmetic onto
//! the device.

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
use crate::specialization::SpecializationKey;
use crate::transform::c2s::{cart_to_sph_2e, ncart, nsph};
use cintx_core::{CintFloat, PrecisionKind, Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::f64::consts::PI;

/// sqrt(pi) constant — matches libcint `SQRTPI`.
const SQRTPI: f64 = 1.7724538509055159_f64;

/// Spherical harmonic normalization prefactor for s and p shells.
fn common_fac_sp(l: u8) -> f64 {
    match l {
        0 => 0.282094791773878143_f64,
        1 => 0.488602511902919921_f64,
        _ => 1.0,
    }
}

/// Enumerate Cartesian component triples (ix, iy, iz) with ix+iy+iz = l.
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

fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("outside Validated4C1E ({reason})"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Launch grouping (Task 35-D, wave 5)
// ─────────────────────────────────────────────────────────────────────────────

/// `u32` shape scalars per class row of the device shape table: `li, lj, lk, ll`.
const FOUR_C1E_SHAPE_STRIDE: usize = 4;

/// `u32` per work row: `si, sj, sk, sl, out_off, class, ci, cj, ck, cl`.
///
/// A row is one *(shell quartet, contraction quartet)*, for the same reason the
/// 3c1e row is: the per-tuple path already launched once per `(ci,cj,ck,cl)`
/// column tuple with that tuple's coefficient columns extracted host-side, so
/// naming the selectors in the row reproduces exactly that arithmetic while
/// collapsing the launches.
const FOUR_C1E_ROW_STRIDE: usize = 10;

/// Per-slot `[gx | gy | gz]` slab stride, padded to a cache line of `f64`.
fn four_c1e_g_slab_stride(max_g_size: usize) -> usize {
    const LINE: usize = 8;
    (3 * max_g_size).div_ceil(LINE) * LINE
}

/// Per-slot polynomial-scratch stride, padded the same way.
fn four_c1e_buf_slab_stride(max_buf: usize) -> usize {
    const LINE: usize = 8;
    max_buf.div_ceil(LINE) * LINE
}

/// One dispatch of the 4c1e family: every work row in it.
///
/// The kernel has no comptime shape parameter — 4c1e is a polynomial recurrence,
/// not a Rys quadrature, so there is no `nroots` to specialize on — and a whole
/// work list is a single dispatch.
#[derive(Clone, Debug, Default)]
pub struct FourC1eLaunchGroup {
    /// [`FOUR_C1E_SHAPE_STRIDE`] `u32` per merged class: `li, lj, lk, ll`.
    pub class_shape: Vec<u32>,
    /// One `common_factor` per class.
    pub class_factor: Vec<f64>,
    /// [`FOUR_C1E_ROW_STRIDE`] `u32` per work row.
    pub rows: Vec<u32>,
    /// Total Cartesian output elements across this group's rows.
    pub out_len: usize,
    /// Widest per-slot `g_size` in the group.
    pub max_g_size: usize,
    /// Widest per-slot polynomial scratch in the group.
    pub max_buf: usize,
}

impl FourC1eLaunchGroup {
    /// Append a class and return the index its rows carry.
    pub fn push_class(&mut self, li: u32, lj: u32, lk: u32, ll: u32, common_factor: f64) -> u32 {
        let index = self.class_factor.len() as u32;
        self.class_shape.extend_from_slice(&[li, lj, lk, ll]);
        self.class_factor.push(common_factor);
        // `shape_sizes_4c1e` is the single host-side mirror of the kernel's
        // inline layout arithmetic; the slab strides derive from it rather than
        // from a second copy that could drift.
        let (g_size, buf_len) =
            shape_sizes_4c1e(li as usize, lj as usize, lk as usize, ll as usize);
        self.max_g_size = self.max_g_size.max(g_size);
        self.max_buf = self.max_buf.max(buf_len);
        index
    }

    /// Append one `(shell quartet, contraction quartet)` row.
    #[allow(clippy::too_many_arguments)]
    pub fn push_row(&mut self, shells: [u32; 4], out_off: u32, class: u32, contraction: [u32; 4]) {
        self.rows.extend_from_slice(&[
            shells[0],
            shells[1],
            shells[2],
            shells[3],
            out_off,
            class,
            contraction[0],
            contraction[1],
            contraction[2],
            contraction[3],
        ]);
    }

    /// Number of work rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len() / FOUR_C1E_ROW_STRIDE
    }

    /// Is this group empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Bytes this group's row and class tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        (self.rows.len() + self.class_shape.len()) * std::mem::size_of::<u32>()
            + self.class_factor.len() * std::mem::size_of::<f64>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device kernel — `#[cube(launch)]`, generic over `F: Float`
// ─────────────────────────────────────────────────────────────────────────────

/// 4c1e polynomial G-tensor fill + 4-branch HRR + Cartesian contraction for one
/// shell quartet, on-device.
///
/// Single work item (`UNIT_POS == 0`) — a faithful, correctness-first port of
/// the host pipeline (`fill_4c1e_g_tensor` + branch-selected `hrr_*_4d` +
/// `contract_4c1e_cart`). All math uses `F::exp` / `F::sqrt` / `F::cast_from` /
/// `F::new` (never methods), statement-form `if`, and `u32` indices/counters
/// with `while` loops — never host `for`/`Vec`/`continue`.
///
/// nroots is always 1 for 4c1e (polynomial recurrence, not Rys quadrature), so
/// every host inner `for r in 0..nroots` loop collapses to a single direct write.
///
/// Layout of `g` (size `3 * g_size`): `[gx | gy | gz]`, each `g_size` long. The
/// stride layout matches `build_4c1e_shape` (di/dk/dl/dj). The 1D polynomial
/// scratch buffer `buf` (size `db * (bigger + 1)`, `db = nmax + mmax + 1`,
/// `bigger = max(nmax, mmax)`) is reused for each of the 3 axes.
///
/// `cart_out` (size `nfi*nfj*nfk*nfl`) is zeroed in-kernel and accumulated over
/// all primitive quartets: `out[i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk]`
/// (i fastest, l slowest). This launch evaluates one (ci,cj,ck,cl) contraction
/// tuple; the host launcher dispatches one launch per tuple.
///
/// Source: libcint-master/src/g4c1e.c, cint4c1e.c `CINT4c1e_loop_nopt`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn center_4c1e_kernel<F: Float + CubeElement>(
    exps: &Array<F>,
    coeffs: &Array<F>,
    centers: &Array<F>,
    shell_meta: &Array<u32>,
    rows: &Array<u32>,
    class_shape: &Array<u32>,
    class_factor: &Array<F>,
    g: &mut Array<F>,
    buf: &mut Array<F>,
    cart_out: &mut Array<F>,
    n_rows: u32,
    n_cubes: u32,
    g_stride: u32,
    buf_stride: u32,
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
        // Two slabs with independent strides: the `[gx|gy|gz]` G-tensor and the
        // 1D/2D polynomial scratch. They are read at unrelated offsets, so
        // unlike the single-stride families they do not share one.
        let gbase = slot * g_stride;
        let bbase = slot * buf_stride;

        #[allow(clippy::manual_div_ceil)]
        let chunk = (n_rows + n_slots - 1u32) / n_slots;
        let qi_start = slot * (chunk * punit + coop);
        let mut qi_stop = (qi_start + chunk) * punit + n_rows * coop;
        if qi_stop > n_rows {
            qi_stop = n_rows;
        }
        let qi_step = n_slots * coop + punit;

        let mut qi = qi_start;
        while qi < qi_stop {
            // A row is one *(shell quartet, contraction quartet)* — see
            // `FourC1eLaunchGroup` for why the contraction is in the row.
            let rrow = qi * comptime!(FOUR_C1E_ROW_STRIDE as u32);
            let si = rows[rrow as usize];
            let sj = rows[(rrow + 1u32) as usize];
            let sk = rows[(rrow + 2u32) as usize];
            let sl = rows[(rrow + 3u32) as usize];
            let out_off = rows[(rrow + 4u32) as usize];
            let cls = rows[(rrow + 5u32) as usize];
            let ci_sel = rows[(rrow + 6u32) as usize];
            let cj_sel = rows[(rrow + 7u32) as usize];
            let ck_sel = rows[(rrow + 8u32) as usize];
            let cl_sel = rows[(rrow + 9u32) as usize];

            let srow = cls * comptime!(FOUR_C1E_SHAPE_STRIDE as u32);
            let li = class_shape[srow as usize];
            let lj = class_shape[(srow + 1u32) as usize];
            let lk = class_shape[(srow + 2u32) as usize];
            let ll = class_shape[(srow + 3u32) as usize];
            let common_factor = class_factor[cls as usize];

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
            let mk = sk * 4u32;
            let eoff_k = shell_meta[mk as usize];
            let coff_k = shell_meta[(mk + 1u32) as usize];
            let nprim_k = shell_meta[(mk + 2u32) as usize];
            let nctr_k = shell_meta[(mk + 3u32) as usize];
            let ml = sl * 4u32;
            let eoff_l = shell_meta[ml as usize];
            let coff_l = shell_meta[(ml + 1u32) as usize];
            let nprim_l = shell_meta[(ml + 2u32) as usize];
            let nctr_l = shell_meta[(ml + 3u32) as usize];

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

            // ── Shape4c1e layout (nroots = 1) ───────────────────────────────
            let nmax = li + lj;
            let mmax = lk + ll;
            // ibase = li > lj, kbase = lk > ll (strict >).
            let mut ibase = false;
            if li > lj {
                ibase = true;
            }
            let mut kbase = false;
            if lk > ll {
                kbase = true;
            }

            // dli/dlj/dlk/dll from the branch selection.
            let mut dli = li + 1u32;
            let mut dlj = li + lj + 1u32;
            if ibase {
                dli = li + lj + 1u32;
                dlj = lj + 1u32;
            }
            let mut dlk = lk + 1u32;
            let mut dll = lk + ll + 1u32;
            if kbase {
                dlk = lk + ll + 1u32;
                dll = ll + 1u32;
            }

            // Strides (nroots = 1).
            let di = 1u32;
            let dk = dli;
            let dl = dli * dlk;
            let dj = dli * dlk * dll;
            let g_size = dli * dlk * dll * dlj;

            // 1D polynomial scratch geometry.
            let db = nmax + mmax + 1u32;
            let mut bigger = nmax;
            if mmax > bigger {
                bigger = mmax;
            }

            // Cartesian output dimensions.
            let nfi = (li + 1u32) * (li + 2u32) / 2u32;
            let nfj = (lj + 1u32) * (lj + 2u32) / 2u32;
            let nfk = (lk + 1u32) * (lk + 2u32) / 2u32;
            let nfl = (ll + 1u32) * (ll + 2u32) / 2u32;
            let out_len = nfi * nfj * nfk * nfl;

            // Zero the accumulation buffer.
            let mut oi = out_off;
            while oi < out_off + out_len {
                cart_out[oi as usize] = F::new(0.0_f32);
                oi += 1u32;
            }

            let total_g = 3u32 * g_size;

            // Primitive quartet loops: pi outer .. pl inner (matches host order).
            let mut pi = 0u32;
            while pi < nprim_i {
                let ai = exps[(eoff_i + pi) as usize];
                let ci_coeff = coeffs[(coff_i + pi * nctr_i + ci_sel) as usize];
                let mut pj = 0u32;
                while pj < nprim_j {
                    let aj = exps[(eoff_j + pj) as usize];
                    let cj_coeff = coeffs[(coff_j + pj * nctr_j + cj_sel) as usize];
                    let aij = ai + aj;

                    // Pair weighted center for ij.
                    let rijx = (ai * rix + aj * rjx) / aij;
                    let rijy = (ai * riy + aj * rjy) / aij;
                    let rijz = (ai * riz + aj * rjz) / aij;

                    // ij overlap prefactor exp(-ai*aj/aij * |ri-rj|^2).
                    let dxij = rix - rjx;
                    let dyij = riy - rjy;
                    let dzij = riz - rjz;
                    let rr_ij = dxij * dxij + dyij * dyij + dzij * dzij;
                    let fac_ij = F::exp(-ai * aj / aij * rr_ij);

                    let mut pk = 0u32;
                    while pk < nprim_k {
                        let ak = exps[(eoff_k + pk) as usize];
                        let ck_coeff = coeffs[(coff_k + pk * nctr_k + ck_sel) as usize];
                        let mut pl = 0u32;
                        while pl < nprim_l {
                            let al = exps[(eoff_l + pl) as usize];
                            let cl_coeff = coeffs[(coff_l + pl * nctr_l + cl_sel) as usize];
                            let akl = ak + al;

                            // Pair weighted center for kl.
                            let rklx = (ak * rkx + al * rlx) / akl;
                            let rkly = (ak * rky + al * rly) / akl;
                            let rklz = (ak * rkz + al * rlz) / akl;

                            // kl overlap prefactor exp(-ak*al/akl * |rk-rl|^2).
                            let dxkl = rkx - rlx;
                            let dykl = rky - rly;
                            let dzkl = rkz - rlz;
                            let rr_kl = dxkl * dxkl + dykl * dykl + dzkl * dzkl;
                            let fac_kl = F::exp(-ak * al / akl * rr_kl);

                            // Cross-pair exponential exp(-a0 * |rij - rkl|^2),
                            // a0 = aij*akl/(aij+akl).
                            let aijkl = aij + akl;
                            let a0 = aij * akl / aijkl;
                            let dxijkl = rijx - rklx;
                            let dyijkl = rijy - rkly;
                            let dzijkl = rijz - rklz;
                            let rr_ijkl = dxijkl * dxijkl + dyijkl * dyijkl + dzijkl * dzijkl;
                            let fac_ijkl = F::exp(-a0 * rr_ijkl);

                            // Quartet prefactor (also folds the contraction coeffs).
                            let weight = ci_coeff * cj_coeff * ck_coeff * cl_coeff;
                            let quartet_fac = common_factor * fac_ij * fac_kl * fac_ijkl * weight;
                            let inv_aijkl = F::new(1.0_f32) / aijkl;
                            let sqrt_aijkl = F::sqrt(aijkl);

                            // ── Fill the polynomial G-tensor per axis ───────────
                            #[unroll]
                            for axis in 0..3u32 {
                                let off = gbase + axis * g_size;

                                // Per-axis coordinates.
                                let mut ri_a = rix;
                                let mut rj_a = rjx;
                                let mut rk_a = rkx;
                                let mut rl_a = rlx;
                                let mut rij_a = rijx;
                                let mut rkl_a = rklx;
                                if axis == 1u32 {
                                    ri_a = riy;
                                    rj_a = rjy;
                                    rk_a = rky;
                                    rl_a = rly;
                                    rij_a = rijy;
                                    rkl_a = rkly;
                                } else if axis == 2u32 {
                                    ri_a = riz;
                                    rj_a = rjz;
                                    rk_a = rkz;
                                    rl_a = rlz;
                                    rij_a = rijz;
                                    rkl_a = rklz;
                                }

                                // Base-center selection (nmax >= mmax branch).
                                let mut r1 = rj_a;
                                let mut r2 = rl_a;
                                if nmax >= mmax {
                                    // ij pair is main, kl pair is auxiliary.
                                    if ibase {
                                        r1 = ri_a;
                                    }
                                    if kbase {
                                        r2 = rk_a;
                                    }
                                } else {
                                    // kl pair is main, ij pair is auxiliary.
                                    r1 = rl_a;
                                    if kbase {
                                        r1 = rk_a;
                                    }
                                    r2 = rj_a;
                                    if ibase {
                                        r2 = ri_a;
                                    }
                                }

                                // Weighted center P = (aij*Rij + akl*Rkl) / aijkl.
                                let rp = (aij * rij_a + akl * rkl_a) * inv_aijkl;
                                let r1r12 = r1 - rp;

                                // Initial value: z-axis gets the prefactor; x/y = 1.
                                let mut buf0 = F::new(1.0_f32);
                                if axis == 2u32 {
                                    buf0 = quartet_fac / (aijkl * sqrt_aijkl);
                                }
                                buf[bbase as usize] = buf0;

                                // Recurrence: buf[i+1] = 0.5*i/aijkl*buf[i-1]
                                //                        - r1r12*buf[i].
                                if nmax + mmax > 0u32 {
                                    buf[(bbase + 1u32) as usize] = -r1r12 * buf[bbase as usize];
                                }
                                let mut ii = 1u32;
                                while ii < nmax + mmax {
                                    let term = F::new(0.5_f32)
                                        * F::cast_from(ii)
                                        * inv_aijkl
                                        * buf[(bbase + ii - 1u32) as usize]
                                        - r1r12 * buf[(bbase + ii) as usize];
                                    buf[(bbase + ii + 1u32) as usize] = term;
                                    ii += 1u32;
                                }

                                // 2D shift fill: buf[j*db + i]
                                //   = buf[(j-1)*db + i+1] + r1r2*buf[(j-1)*db + i].
                                let r1r2 = r1 - r2;
                                let mut jj = 1u32;
                                while jj <= bigger {
                                    // i in 0 .. (db - jj).
                                    let mut i_lim = 0u32;
                                    if db > jj {
                                        i_lim = db - jj;
                                    }
                                    let mut ip = 0u32;
                                    while ip < i_lim {
                                        let prev = (jj - 1u32) * db;
                                        let val = buf[(bbase + prev + ip + 1u32) as usize]
                                            + r1r2 * buf[(bbase + prev + ip) as usize];
                                        buf[(bbase + jj * db + ip) as usize] = val;
                                        ip += 1u32;
                                    }
                                    jj += 1u32;
                                }

                                // Map polynomial buf -> G-tensor.
                                // g2d_ij / g2d_kl pick the ij/kl direction strides.
                                let mut g2d_ij = dj;
                                if ibase {
                                    g2d_ij = di;
                                }
                                let mut g2d_kl = dl;
                                if kbase {
                                    g2d_kl = dk;
                                }
                                if nmax >= mmax {
                                    // n iterates ij (0..=nmax), m iterates kl (0..=mmax).
                                    let mut m = 0u32;
                                    while m <= mmax {
                                        let mut n = 0u32;
                                        while n <= nmax {
                                            let idx = off + n * g2d_ij + m * g2d_kl;
                                            g[idx as usize] = buf[(bbase + m * db + n) as usize];
                                            n += 1u32;
                                        }
                                        m += 1u32;
                                    }
                                } else {
                                    // kl is main, ij is auxiliary — swap roles.
                                    let mut n = 0u32;
                                    while n <= nmax {
                                        let mut m = 0u32;
                                        while m <= mmax {
                                            let idx = off + m * g2d_kl + n * g2d_ij;
                                            g[idx as usize] = buf[(bbase + n * db + m) as usize];
                                            m += 1u32;
                                        }
                                        n += 1u32;
                                    }
                                }
                            }

                            // ── 4-branch HRR (selected by kbase/ibase) ──────────
                            let rirjx = rix - rjx;
                            let rirjy = riy - rjy;
                            let rirjz = riz - rjz;
                            let rkrlx = rkx - rlx;
                            let rkrly = rky - rly;
                            let rkrlz = rkz - rlz;

                            // Inlined per-axis HRR for each branch. nroots = 1, so
                            // the host `for r in 0..nroots` loops become direct
                            // single writes (offset 0).
                            #[unroll]
                            for haxis in 0..3u32 {
                                let off = gbase + haxis * g_size;
                                let mut rirj_d = rirjx;
                                let mut rkrl_d = rkrlx;
                                if haxis == 1u32 {
                                    rirj_d = rirjy;
                                    rkrl_d = rkrly;
                                } else if haxis == 2u32 {
                                    rirj_d = rirjz;
                                    rkrl_d = rkrlz;
                                }

                                if kbase {
                                    if ibase {
                                        // hrr_ik2d_4d: skip if lj==0 && ll==0.
                                        if lj > 0u32 || ll > 0u32 {
                                            // l-shift on kl: l=1..=ll, k=0..=mmax-l,
                                            //   i=0..=nmax. ptr = l*dl + k*dk + i*di.
                                            let mut l = 1u32;
                                            while l <= ll {
                                                let mut k = 0u32;
                                                while k <= mmax - l {
                                                    let mut i = 0u32;
                                                    while i <= nmax {
                                                        let idx = l * dl + k * dk + i * di;
                                                        let v = rkrl_d
                                                            * g[(off + idx - dl) as usize]
                                                            + g[(off + idx - dl + dk) as usize];
                                                        g[(off + idx) as usize] = v;
                                                        i += 1u32;
                                                    }
                                                    k += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            // j-shift on ij: j=1..=lj, l=0..=ll,
                                            //   k=0..=lk, i=0..=nmax-j.
                                            let mut j = 1u32;
                                            while j <= lj {
                                                let mut l = 0u32;
                                                while l <= ll {
                                                    let mut k = 0u32;
                                                    while k <= lk {
                                                        let ptr = j * dj + l * dl + k * dk;
                                                        let mut i = 0u32;
                                                        while i <= nmax - j {
                                                            let base = ptr + i * di;
                                                            let v = rirj_d
                                                                * g[(off + base - dj) as usize]
                                                                + g[(off + base - dj + di)
                                                                    as usize];
                                                            g[(off + base) as usize] = v;
                                                            i += 1u32;
                                                        }
                                                        k += 1u32;
                                                    }
                                                    l += 1u32;
                                                }
                                                j += 1u32;
                                            }
                                        }
                                    } else {
                                        // hrr_kj2d_4d: skip if li==0 && ll==0.
                                        if li > 0u32 || ll > 0u32 {
                                            // i-shift on ij: i=1..=li, j=0..=nmax-i,
                                            //   k=0..=mmax. ptr = j*dj + k*dk + i*di.
                                            let mut i = 1u32;
                                            while i <= li {
                                                let mut j = 0u32;
                                                while j <= nmax - i {
                                                    let mut k = 0u32;
                                                    while k <= mmax {
                                                        let idx = j * dj + k * dk + i * di;
                                                        let v = rirj_d
                                                            * g[(off + idx - di) as usize]
                                                            + g[(off + idx - di + dj) as usize];
                                                        g[(off + idx) as usize] = v;
                                                        k += 1u32;
                                                    }
                                                    j += 1u32;
                                                }
                                                i += 1u32;
                                            }
                                            // l-shift on kl: j=0..=lj, l=1..=ll,
                                            //   k=0..=mmax-l. ptr = j*dj+l*dl+k*dk;
                                            //   inner n in 0..dk.
                                            let mut j = 0u32;
                                            while j <= lj {
                                                let mut l = 1u32;
                                                while l <= ll {
                                                    let mut k = 0u32;
                                                    while k <= mmax - l {
                                                        let ptr = j * dj + l * dl + k * dk;
                                                        let mut n = 0u32;
                                                        while n < dk {
                                                            let idx = ptr + n;
                                                            let v = rkrl_d
                                                                * g[(off + idx - dl) as usize]
                                                                + g[(off + idx - dl + dk) as usize];
                                                            g[(off + idx) as usize] = v;
                                                            n += 1u32;
                                                        }
                                                        k += 1u32;
                                                    }
                                                    l += 1u32;
                                                }
                                                j += 1u32;
                                            }
                                        }
                                    }
                                } else if ibase {
                                    // hrr_il2d_4d: skip if lj==0 && lk==0.
                                    if lj > 0u32 || lk > 0u32 {
                                        // k-shift on kl: k=1..=lk, l=0..=mmax-k,
                                        //   i=0..=nmax. ptr = l*dl + k*dk + i*di.
                                        let mut k = 1u32;
                                        while k <= lk {
                                            let mut l = 0u32;
                                            while l <= mmax - k {
                                                let mut i = 0u32;
                                                while i <= nmax {
                                                    let idx = l * dl + k * dk + i * di;
                                                    let v = rkrl_d * g[(off + idx - dk) as usize]
                                                        + g[(off + idx - dk + dl) as usize];
                                                    g[(off + idx) as usize] = v;
                                                    i += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            k += 1u32;
                                        }
                                        // j-shift on ij: j=1..=lj, l=0..=ll,
                                        //   k=0..=lk, i=0..=nmax-j.
                                        let mut j = 1u32;
                                        while j <= lj {
                                            let mut l = 0u32;
                                            while l <= ll {
                                                let mut k = 0u32;
                                                while k <= lk {
                                                    let ptr = j * dj + l * dl + k * dk;
                                                    let mut i = 0u32;
                                                    while i <= nmax - j {
                                                        let base = ptr + i * di;
                                                        let v = rirj_d
                                                            * g[(off + base - dj) as usize]
                                                            + g[(off + base - dj + di) as usize];
                                                        g[(off + base) as usize] = v;
                                                        i += 1u32;
                                                    }
                                                    k += 1u32;
                                                }
                                                l += 1u32;
                                            }
                                            j += 1u32;
                                        }
                                    }
                                } else {
                                    // hrr_lj2d_4d: skip if li==0 && lk==0.
                                    if li > 0u32 || lk > 0u32 {
                                        // i-shift on ij: i=1..=li, j=0..=nmax-i,
                                        //   l=0..=mmax. ptr = j*dj + l*dl + i*di.
                                        let mut i = 1u32;
                                        while i <= li {
                                            let mut j = 0u32;
                                            while j <= nmax - i {
                                                let mut l = 0u32;
                                                while l <= mmax {
                                                    let idx = j * dj + l * dl + i * di;
                                                    let v = rirj_d * g[(off + idx - di) as usize]
                                                        + g[(off + idx - di + dj) as usize];
                                                    g[(off + idx) as usize] = v;
                                                    l += 1u32;
                                                }
                                                j += 1u32;
                                            }
                                            i += 1u32;
                                        }
                                        // k-shift on kl: j=0..=lj, k=1..=lk,
                                        //   l=0..=mmax-k. ptr = j*dj+l*dl+k*dk;
                                        //   inner n in 0..dk.
                                        let mut j = 0u32;
                                        while j <= lj {
                                            let mut k = 1u32;
                                            while k <= lk {
                                                let mut l = 0u32;
                                                while l <= mmax - k {
                                                    let ptr = j * dj + l * dl + k * dk;
                                                    let mut n = 0u32;
                                                    while n < dk {
                                                        let idx = ptr + n;
                                                        let v = rkrl_d
                                                            * g[(off + idx - dk) as usize]
                                                            + g[(off + idx - dk + dl) as usize];
                                                        g[(off + idx) as usize] = v;
                                                        n += 1u32;
                                                    }
                                                    l += 1u32;
                                                }
                                                k += 1u32;
                                            }
                                            j += 1u32;
                                        }
                                    }
                                }
                            }

                            // ── Contract [gx|gy|gz] into cart_out ───────────────
                            // Reproduce contract_4c1e_cart ordering inline:
                            //   l outer, then k, then j, then i (i fastest, l slowest);
                            //   out_idx = i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk.
                            let gx = gbase;
                            let gy = gbase + g_size;
                            let gz = gbase + 2u32 * g_size;

                            let mut cl_idx = 0u32;
                            let mut la = 0u32;
                            while la <= ll {
                                let lx = ll - la;
                                let ll_minus_lx = ll - lx;
                                let mut lb = 0u32;
                                while lb <= ll_minus_lx {
                                    let ly = ll_minus_lx - lb;
                                    let lz = ll - lx - ly;
                                    let base_lx = lx * dl;
                                    let base_ly = ly * dl;
                                    let base_lz = lz * dl;
                                    let l_out_off = cl_idx * nfi * nfj * nfk;

                                    let mut ck_idx = 0u32;
                                    let mut ka = 0u32;
                                    while ka <= lk {
                                        let kx = lk - ka;
                                        let lk_minus_kx = lk - kx;
                                        let mut kb = 0u32;
                                        while kb <= lk_minus_kx {
                                            let ky = lk_minus_kx - kb;
                                            let kz = lk - kx - ky;
                                            let base_klx = base_lx + kx * dk;
                                            let base_kly = base_ly + ky * dk;
                                            let base_klz = base_lz + kz * dk;
                                            let kl_out_off = l_out_off + ck_idx * nfi * nfj;

                                            let mut cj_idx = 0u32;
                                            let mut ja = 0u32;
                                            while ja <= lj {
                                                let jx = lj - ja;
                                                let lj_minus_jx = lj - jx;
                                                let mut jb = 0u32;
                                                while jb <= lj_minus_jx {
                                                    let jy = lj_minus_jx - jb;
                                                    let jz = lj - jx - jy;
                                                    let base_jklx = base_klx + jx * dj;
                                                    let base_jkly = base_kly + jy * dj;
                                                    let base_jklz = base_klz + jz * dj;
                                                    let jkl_out_off = kl_out_off + cj_idx * nfi;

                                                    let mut ci_idx = 0u32;
                                                    let mut ia = 0u32;
                                                    while ia <= li {
                                                        let ix = li - ia;
                                                        let li_minus_ix = li - ix;
                                                        let mut ib = 0u32;
                                                        while ib <= li_minus_ix {
                                                            let iy = li_minus_ix - ib;
                                                            let iz = li - ix - iy;

                                                            let x_idx = ix * di + base_jklx;
                                                            let y_idx = iy * di + base_jkly;
                                                            let z_idx = iz * di + base_jklz;
                                                            let vx = g[(gx + x_idx) as usize];
                                                            let vy = g[(gy + y_idx) as usize];
                                                            let vz = g[(gz + z_idx) as usize];
                                                            let out_idx = jkl_out_off + ci_idx;
                                                            cart_out
                                                                [(out_off + out_idx) as usize] +=
                                                                vx * vy * vz;

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

                                            ck_idx += 1u32;
                                            kb += 1u32;
                                        }
                                        ka += 1u32;
                                    }

                                    cl_idx += 1u32;
                                    lb += 1u32;
                                }
                                la += 1u32;
                            }

                            pl += 1u32;
                        }
                        pk += 1u32;
                    }
                    pj += 1u32;
                }
                pi += 1u32;
            }

            qi += qi_step;
        }
    }
}

/// Compute the `(g_size, buf_size)` scratch sizes for a 4c1e quartet on the host
/// — mirrors the kernel's inline layout arithmetic so the launcher can allocate
/// the device scratch arrays the kernel never allocates itself.
fn shape_sizes_4c1e(li: usize, lj: usize, lk: usize, ll: usize) -> (usize, usize) {
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
    let g_size = dli * dlk * dll * dlj;
    let db = nmax + mmax + 1;
    let bigger = nmax.max(mmax);
    let buf_size = db * (bigger + 1);
    (g_size, buf_size)
}

fn ensure_validated_4c1e(
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
) -> Result<(), cintxRsError> {
    // D-05: Spinor rejection FIRST — before feature gate or any other check.
    if matches!(plan.representation, Representation::Spinor) {
        return Err(validated_4c1e_error(
            "spinor representation not supported for 4c1e",
        ));
    }

    if specialization.canonical_family() != "4c1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_4c1e",
            detail: format!(
                "canonical_family mismatch for 4c1e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    if !matches!(
        plan.representation,
        Representation::Cart | Representation::Spheric
    ) {
        return Err(validated_4c1e_error("representation must be cart/sph"));
    }
    if !plan.descriptor.entry.component_rank.trim().is_empty()
        && plan.descriptor.entry.component_rank != "scalar"
    {
        return Err(validated_4c1e_error("component rank must be scalar"));
    }
    if plan
        .shells
        .as_slice()
        .iter()
        .any(|shell| shell.ang_momentum > 4)
    {
        return Err(validated_4c1e_error("max(l)>4"));
    }

    Ok(())
}

/// Evaluate every launch group of a batched 4c1e run (Task 35-D, wave 5).
///
/// One dispatch and one readback per group; the flattened basis is the caller's
/// and is uploaded once for the run.
fn run_4c1e_batches<R: Runtime>(
    client: &ComputeClient<R>,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[FourC1eLaunchGroup],
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
        let n_rows = group.len();
        if n_rows == 0 {
            results.push(Vec::new());
            continue;
        }
        let g_stride = four_c1e_g_slab_stride(group.max_g_size);
        let buf_stride = four_c1e_buf_slab_stride(group.max_buf);
        let (n_cubes, cube_dim, n_slots) =
            four_c1e_launch_geometry::<R>(client, n_rows, g_stride + buf_stride);
        let g_len = n_slots * g_stride;
        let buf_len = n_slots * buf_stride;

        let rows_h = client.create_from_slice(u32::as_bytes(&group.rows));
        let shape_h = client.create_from_slice(u32::as_bytes(&group.class_shape));
        let factor_h = client.create_from_slice(f64::as_bytes(&group.class_factor));
        let g_h = client.empty(g_len * std::mem::size_of::<f64>());
        let buf_h = client.empty(buf_len * std::mem::size_of::<f64>());
        let out_h = client.empty(group.out_len * std::mem::size_of::<f64>());
        let per_unit = u32::from(four_c1e_per_unit::<R>(client));

        // SAFETY: every buffer is allocated at the exact length passed to
        // `ArrayArg::from_raw_parts`. In-kernel indices are bounded by
        // `n_rows`, by the class index in each row, by the per-shell
        // `nprim`/`nctr` read from `shell_meta`, and by the per-class extents.
        unsafe {
            center_4c1e_kernel::launch_unchecked::<f64, R>(
                client,
                crate::plane::cube_count_1d(n_cubes),
                cube_dim,
                ArrayArg::from_raw_parts(exps_h.clone(), *exps_len),
                ArrayArg::from_raw_parts(coeffs_h.clone(), *coeffs_len),
                ArrayArg::from_raw_parts(centers_h.clone(), *centers_len),
                ArrayArg::from_raw_parts(meta_h.clone(), *shell_meta_len),
                ArrayArg::from_raw_parts(rows_h.clone(), group.rows.len()),
                ArrayArg::from_raw_parts(shape_h.clone(), group.class_shape.len()),
                ArrayArg::from_raw_parts(factor_h.clone(), group.class_factor.len()),
                ArrayArg::from_raw_parts(g_h.clone(), g_len),
                ArrayArg::from_raw_parts(buf_h.clone(), buf_len),
                ArrayArg::from_raw_parts(out_h.clone(), group.out_len),
                n_rows as u32,
                n_cubes,
                g_stride as u32,
                buf_stride as u32,
                per_unit,
            );
        }

        let raw = client.read_one_unchecked(out_h);
        results.push(f64::from_bytes(&raw)[0..group.out_len].to_vec());
    }
    results
}

/// Does this backend want the one-row-per-unit decomposition?
fn four_c1e_per_unit<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let pinned = *OVERRIDE.get_or_init(|| {
        std::env::var("CINTX_4C1E_PER_UNIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    });
    match pinned {
        Some(value) => value != 0,
        None => !crate::plane::has_planes(client),
    }
}

/// `(n_cubes, cube_dim, n_slots)` for one 4c1e dispatch.
///
/// `slot_words` is the *combined* G-tensor and polynomial-scratch stride, so the
/// memory bound accounts for both slabs rather than only the larger one.
fn four_c1e_launch_geometry<R: Runtime>(
    client: &ComputeClient<R>,
    n_rows: usize,
    slot_words: usize,
) -> (u32, CubeDim, usize) {
    /// Ceiling on the per-launch scratch.
    const MAX_BATCH_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

    let per_slot = slot_words * std::mem::size_of::<f64>();
    let by_memory = (MAX_BATCH_SCRATCH_BYTES / per_slot.max(1)).max(1);

    if four_c1e_per_unit::<R>(client) {
        let units = crate::plane::per_unit_width(
            client,
            n_rows,
            crate::plane::MIN_ITEMS_PER_UNIT_PAIR,
            by_memory,
        );
        return (1, CubeDim::new_1d(units), units as usize);
    }
    let cubes = crate::plane::grid_cube_count(client, n_rows.min(by_memory));
    (cubes, CubeDim::new_1d(1), cubes as usize)
}

/// Backend dispatch for a whole batched 4c1e run.
fn dispatch_4c1e_batches(
    backend: &ResolvedBackend,
    basis: &crate::kernels::two_electron::TwoEBasisHandles,
    groups: &[FourC1eLaunchGroup],
) -> Vec<Vec<f64>> {
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => {
            run_4c1e_batches::<cubecl::cpu::CpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => {
            run_4c1e_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => {
            run_4c1e_batches::<cubecl_cuda::CudaRuntime>(client, basis, groups)
        }
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => {
            run_4c1e_batches::<cubecl_hip::HipRuntime>(client, basis, groups)
        }
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => {
            run_4c1e_batches::<cubecl_wgpu::WgpuRuntime>(client, basis, groups)
        }
    }
}

/// One shell quartet through the batched 4c1e path, returning the Cartesian
/// block of every `(ci, cj, ck, cl)` contraction tuple concatenated in
/// `cl`-slowest order.
///
/// The per-tuple compatibility API evaluates exactly one shell quartet and must
/// keep doing so, but it goes through the *same* kernel as a wide batch — one
/// row per contraction tuple. That is Task 35-D's acceptance bar, and it
/// collapses the `nctr_i * nctr_j * nctr_k * nctr_l` launches the
/// general-contraction case used to cost into one.
#[allow(clippy::too_many_arguments)]
fn run_4c1e_contracted_on_backend(
    backend: &ResolvedBackend,
    l: [u32; 4],
    nprim: [u32; 4],
    nctr: [u32; 4],
    centers: [[f64; 3]; 4],
    common_factor: f64,
    exps: [&[f64]; 4],
    coeffs: [&[f64]; 4],
) -> Vec<f64> {
    let mut basis = crate::kernels::two_electron::TwoEFlatBasis::default();
    for (index, center) in centers.into_iter().enumerate() {
        basis.shell_meta.extend_from_slice(&[
            basis.exps.len() as u32,
            basis.coeffs.len() as u32,
            nprim[index],
            nctr[index],
        ]);
        basis.exps.extend_from_slice(exps[index]);
        basis.coeffs.extend_from_slice(coeffs[index]);
        basis.centers.extend_from_slice(&center);
    }
    let handles = upload_4c1e_basis(backend, &basis);

    let block_len: usize = l.iter().map(|&li| ncart(li as u8)).product();
    let mut group = FourC1eLaunchGroup::default();
    let class = group.push_class(l[0], l[1], l[2], l[3], common_factor);
    // `ci` outermost, `cl` innermost — the order the per-tuple launcher walked
    // its contraction tuples. The caller *sums* these blocks, so the order is
    // load-bearing: summing them in any other sequence would reassociate the
    // additions and move the last bits.
    for ci in 0..nctr[0] {
        for cj in 0..nctr[1] {
            for ck in 0..nctr[2] {
                for cl in 0..nctr[3] {
                    group.push_row([0, 1, 2, 3], group.out_len as u32, class, [ci, cj, ck, cl]);
                    group.out_len += block_len;
                }
            }
        }
    }

    dispatch_4c1e_batches(backend, &handles, std::slice::from_ref(&group))
        .pop()
        .unwrap_or_default()
}

/// Upload a flattened basis through whichever backend arm `backend` is.
fn upload_4c1e_basis(
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

/// Generic inner for the 4c1e launcher.
///
/// Dispatches the [`center_4c1e_kernel`] device kernel (at f64) on `plan`'s
/// resolved backend, reads back the Cartesian buffer, then applies the
/// representation transform with the output cast to `F` (see module precision
/// policy). Preserves the Validated4C1E envelope and polynomial-recurrence
/// G-tensor fill (no Rys quadrature, nroots=1).
fn launch_center_4c1e_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    ensure_validated_4c1e(plan, specialization)?;

    let shells = plan.shells.as_slice();
    if shells.len() < 4 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_4c1e",
            detail: "4c1e kernel requires exactly 4 shells".to_owned(),
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

    // Accumulated Cartesian integral buffer.
    let mut cart_buf = vec![0.0_f64; nfi * nfj * nfk * nfl];

    // Common factor from cint4c1e.c CINT4c1e_loop_nopt:
    //   common_fac = envs->common_factor * SQRTPI * M_PI * sp_factors
    // where envs->common_factor = 1 (set in CINTinit_int4c1e_EnvVars).
    // So the 4c1e common factor is: sqrt(pi) * pi * sp_factors.
    //
    // NOTE: This differs from the 2e formula (pi^3 * 2 / sqrt(pi)) — do NOT use
    // the 2e formula here. The 4c1e integral uses a different normalization.
    let sp_factor = common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk) * common_fac_sp(ll);
    let common_factor = SQRTPI * PI * sp_factor;

    let n_prim_i = shell_i.nprim as usize;
    let n_prim_j = shell_j.nprim as usize;
    let n_prim_k = shell_k.nprim as usize;
    let n_prim_l = shell_l.nprim as usize;

    let n_ctr_i = shell_i.nctr as usize;
    let n_ctr_j = shell_j.nctr as usize;
    let n_ctr_k = shell_k.nctr as usize;
    let n_ctr_l = shell_l.nctr as usize;

    // Primitive exponent arrays (shared across contraction columns).
    let exps_i: Vec<f64> = (0..n_prim_i).map(|p| shell_i.exponents[p]).collect();
    let exps_j: Vec<f64> = (0..n_prim_j).map(|p| shell_j.exponents[p]).collect();
    let exps_k: Vec<f64> = (0..n_prim_k).map(|p| shell_k.exponents[p]).collect();
    let exps_l: Vec<f64> = (0..n_prim_l).map(|p| shell_l.exponents[p]).collect();

    // Task 35-D wave 5: every `(ci, cj, ck, cl)` contraction tuple is one row of
    // a single batched dispatch, rather than one launch each. The arithmetic per
    // row is what the per-tuple launch did — the row names its coefficient
    // columns instead of the host slicing them out first — and the blocks are
    // summed in the same order they were before.
    let coeff_i_all: Vec<f64> = shell_i.coefficients[..n_prim_i * n_ctr_i].to_vec();
    let coeff_j_all: Vec<f64> = shell_j.coefficients[..n_prim_j * n_ctr_j].to_vec();
    let coeff_k_all: Vec<f64> = shell_k.coefficients[..n_prim_k * n_ctr_k].to_vec();
    let coeff_l_all: Vec<f64> = shell_l.coefficients[..n_prim_l * n_ctr_l].to_vec();

    let cart_all = run_4c1e_contracted_on_backend(
        backend,
        [li as u32, lj as u32, lk as u32, ll as u32],
        [
            n_prim_i as u32,
            n_prim_j as u32,
            n_prim_k as u32,
            n_prim_l as u32,
        ],
        [
            n_ctr_i as u32,
            n_ctr_j as u32,
            n_ctr_k as u32,
            n_ctr_l as u32,
        ],
        [ri, rj, rk, rl],
        common_factor,
        [&exps_i, &exps_j, &exps_k, &exps_l],
        [&coeff_i_all, &coeff_j_all, &coeff_k_all, &coeff_l_all],
    );

    for block in cart_all.chunks_exact(cart_buf.len()) {
        for (dst, &src) in cart_buf.iter_mut().zip(block.iter()) {
            *dst += src;
        }
    }

    // Apply cart-to-sph transform or copy Cartesian, casting to F at write.
    match plan.representation {
        Representation::Spheric => {
            let sph = cart_to_sph_2e(&cart_buf, li, lj, lk, ll);
            let sph_size = nsi * nsj * nsk * nsl;
            let copy_len = staging.len().min(sph_size);
            for (dst, &src) in staging[..copy_len].iter_mut().zip(sph[..copy_len].iter()) {
                *dst = F::from_f64_lossy(src);
            }
        }
        _ => {
            let copy_len = staging.len().min(cart_buf.len());
            for (dst, &src) in staging[..copy_len]
                .iter_mut()
                .zip(cart_buf[..copy_len].iter())
            {
                *dst = F::from_f64_lossy(src);
            }
        }
    }

    // Per-symbol nonzero sentinel
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

/// Outer precision dispatcher for the 4c1e kernel.
///
/// Keeps the registered `FamilyLaunchFn` signature and `#[cfg(feature = "with-4c1e")]`
/// gate unchanged. Internally matches on `plan.precision` and delegates to the
/// generic inner `launch_center_4c1e_typed::<F>`, reinterpreting staging via
/// `bytemuck::cast_slice_mut` for the F32 arm (A5 proven sound). The Validated4C1E
/// envelope, polynomial-recurrence G-tensor, and 4-branch HRR are preserved verbatim.
/// CR-01: captures the true output element count BEFORE the bytemuck cast and bounds
/// the typed inner to that count, returning `BufferTooSmall` if the view cannot hold it.
pub fn launch_center_4c1e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    match plan.precision {
        PrecisionKind::F64 => {
            launch_center_4c1e_typed::<f64>(backend, plan, specialization, staging)
        }
        PrecisionKind::F32 => {
            // CR-01: capture the true output element count BEFORE the bytemuck cast.
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
            launch_center_4c1e_typed::<f32>(
                backend,
                plan,
                specialization,
                &mut staging_f32[..out_elems],
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host f64 references (test-only) — exercised by the device-vs-host cross-check
// ─────────────────────────────────────────────────────────────────────────────

/// G-tensor layout metadata for 4c1e (host reference).
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct Shape4c1e {
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
    g_size: usize,
}

/// Build layout metadata for 4c1e with nroots forced to 1 (host reference).
#[cfg(test)]
fn build_4c1e_shape(li: usize, lj: usize, lk: usize, ll: usize) -> Shape4c1e {
    let nroots = 1usize;
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

    Shape4c1e {
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
        g_size,
    }
}

/// Fill G-tensor for one primitive quartet using polynomial 1D recurrence (host
/// reference of the exact device algorithm — g4c1e.c).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fill_4c1e_g_tensor(
    g: &mut [f64],
    shape: &Shape4c1e,
    ri: [f64; 3],
    rj: [f64; 3],
    rk: [f64; 3],
    rl: [f64; 3],
    rij: [f64; 3],
    rkl: [f64; 3],
    aij: f64,
    akl: f64,
    fac: f64,
) {
    let aijkl = aij + akl;
    let nmax = shape.nmax;
    let mmax = shape.mmax;

    for axis in 0..3usize {
        let off = axis * shape.g_size;

        let (r1, r2) = if nmax >= mmax {
            let r1 = if shape.ibase { ri[axis] } else { rj[axis] };
            let r2 = if shape.kbase { rk[axis] } else { rl[axis] };
            (r1, r2)
        } else {
            let r1 = if shape.kbase { rk[axis] } else { rl[axis] };
            let r2 = if shape.ibase { ri[axis] } else { rj[axis] };
            (r1, r2)
        };

        let rp = (aij * rij[axis] + akl * rkl[axis]) / aijkl;
        let r1r12 = r1 - rp;

        let db = nmax + mmax + 1;
        let bigger = nmax.max(mmax);
        let mut buf = vec![0.0f64; db * (bigger + 1)];

        buf[0] = if axis == 2 {
            fac / (aijkl * aijkl.sqrt())
        } else {
            1.0
        };

        if nmax + mmax > 0 {
            buf[1] = -r1r12 * buf[0];
        }
        for i in 1..(nmax + mmax) {
            buf[i + 1] = 0.5 * (i as f64) / aijkl * buf[i - 1] - r1r12 * buf[i];
        }

        let r1r2 = r1 - r2;
        for j in 1..=bigger {
            for i in 0..db.saturating_sub(j) {
                buf[j * db + i] = buf[(j - 1) * db + i + 1] + r1r2 * buf[(j - 1) * db + i];
            }
        }

        if nmax >= mmax {
            for m in 0..=mmax {
                for n in 0..=nmax {
                    let g2d_ij = if shape.ibase { shape.di } else { shape.dj };
                    let g2d_kl = if shape.kbase { shape.dk } else { shape.dl };
                    let idx = off + n * g2d_ij + m * g2d_kl;
                    g[idx] = buf[m * db + n];
                }
            }
        } else {
            for n in 0..=nmax {
                for m in 0..=mmax {
                    let g2d_kl = if shape.kbase { shape.dk } else { shape.dl };
                    let g2d_ij = if shape.ibase { shape.di } else { shape.dj };
                    let idx = off + m * g2d_kl + n * g2d_ij;
                    g[idx] = buf[n * db + m];
                }
            }
        }
    }
}

/// HRR branch for `ibase=false && kbase=false` (host reference).
#[cfg(test)]
fn hrr_lj2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=false && kbase=true` (host reference).
#[cfg(test)]
fn hrr_kj2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=true && kbase=false` (host reference).
#[cfg(test)]
fn hrr_il2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// HRR branch for `ibase=true && kbase=true` (host reference).
#[cfg(test)]
fn hrr_ik2d_4d(g: &mut [f64], shape: &Shape4c1e, rirj: [f64; 3], rkrl: [f64; 3]) {
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

/// Contract `[gx|gy|gz]` into Cartesian 4c1e tensor (host reference).
///
/// Output order: `out[i + j*nfi + k*nfi*nfj + l*nfi*nfj*nfk]` (i fastest, l slowest).
#[cfg(test)]
fn contract_4c1e_cart(g: &[f64], shape: &Shape4c1e, li: u8, lj: u8, lk: u8, ll: u8) -> Vec<f64> {
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

    let irys = 0usize;

    for (l_idx, &(lx, ly, lz)) in cl_comps.iter().enumerate() {
        for (k_idx, &(kx, ky, kz)) in ck_comps.iter().enumerate() {
            for (j_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (i_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
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
                    let val = g[gx_off + x_idx] * g[gy_off + y_idx] * g[gz_off + z_idx];
                    let out_idx = i_idx + j_idx * nfi + k_idx * nfi * nfj + l_idx * nfi * nfj * nfk;
                    out[out_idx] += val;
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-2a: launch_center_4c1e_typed::<f64> byte-identical to baseline.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "with-4c1e")]
    #[test]
    fn test_center_4c1e_parity_f64() {
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;
        use crate::specialization::SpecializationKey;
        use cintx_core::{Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| {
            Arc::new(
                Shell::try_new(
                    atom_idx,
                    0,
                    1,
                    1,
                    0,
                    Representation::Cart,
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let shell_a0 = make_s_shell(0);
        let shell_a1 = make_s_shell(0);
        let shell_b0 = make_s_shell(1);
        let shell_b1 = make_s_shell(1);
        let all_shells = Arc::from(
            vec![
                shell_a0.clone(),
                shell_a1.clone(),
                shell_b0.clone(),
                shell_b1.clone(),
            ]
            .into_boxed_slice(),
        );
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells =
            cintx_core::ShellTuple::try_from_iter([shell_a0, shell_a1, shell_b0, shell_b1])
                .unwrap();

        use cintx_ops::resolver::Resolver;
        let desc = Resolver::descriptor_by_symbol("int4c1e_cart").expect("int4c1e_cart must exist");
        let op_id = desc.id;

        let opts = ExecutionOptions::default();
        let query =
            query_workspace(op_id, Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan =
            ExecutionPlan::new(op_id, Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F64;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging_outer = vec![0.0_f64; 1];
        let mut staging_typed = vec![0.0_f64; 1];

        let result_outer = launch_center_4c1e(&backend, &plan, &spec, &mut staging_outer);
        assert!(
            result_outer.is_ok(),
            "outer f64 4c1e should succeed: {:?}",
            result_outer
        );

        let result_typed =
            launch_center_4c1e_typed::<f64>(&backend, &plan, &spec, &mut staging_typed);
        assert!(
            result_typed.is_ok(),
            "typed f64 4c1e should succeed: {:?}",
            result_typed
        );

        assert_eq!(
            staging_outer[0].to_bits(),
            staging_typed[0].to_bits(),
            "f64 outer and typed 4c1e should be byte-identical: outer={} typed={}",
            staging_outer[0],
            staging_typed[0]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test T05-2b: launch_center_4c1e F32 path runs without panic.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "with-4c1e")]
    #[test]
    fn test_center_4c1e_f32_smoke() {
        use crate::backend::ResolvedBackend;
        use crate::backend::cpu_backend::resolve_cpu_client;
        use crate::specialization::SpecializationKey;
        use cintx_core::{Atom, BasisSet, NuclearModel, PrecisionKind, Representation, Shell};
        use cintx_runtime::{ExecutionOptions, ExecutionPlan, query_workspace};
        use std::sync::Arc;

        let atom_a = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atom_b = Atom::try_new(1, [1.4, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom_a, atom_b].into_boxed_slice());
        let make_s_shell = |atom_idx: u32| {
            Arc::new(
                Shell::try_new(
                    atom_idx,
                    0,
                    1,
                    1,
                    0,
                    Representation::Cart,
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                    Arc::from(vec![1.0_f64].into_boxed_slice()),
                )
                .unwrap(),
            )
        };
        let shell_a0 = make_s_shell(0);
        let shell_a1 = make_s_shell(0);
        let shell_b0 = make_s_shell(1);
        let shell_b1 = make_s_shell(1);
        let all_shells = Arc::from(
            vec![
                shell_a0.clone(),
                shell_a1.clone(),
                shell_b0.clone(),
                shell_b1.clone(),
            ]
            .into_boxed_slice(),
        );
        let basis = BasisSet::try_new(atoms, all_shells).unwrap();
        let shells =
            cintx_core::ShellTuple::try_from_iter([shell_a0, shell_a1, shell_b0, shell_b1])
                .unwrap();

        use cintx_ops::resolver::Resolver;
        let desc = Resolver::descriptor_by_symbol("int4c1e_cart").expect("int4c1e_cart must exist");
        let op_id = desc.id;

        let opts = ExecutionOptions::default();
        let query =
            query_workspace(op_id, Representation::Cart, &basis, shells.clone(), &opts).unwrap();
        let mut plan =
            ExecutionPlan::new(op_id, Representation::Cart, &basis, shells, &query).unwrap();
        plan.precision = PrecisionKind::F32;

        let spec = SpecializationKey::from_plan(&plan);
        let cpu_client = resolve_cpu_client().unwrap();
        let backend = ResolvedBackend::Cpu(cpu_client);

        let mut staging = vec![0.0_f64; 1];
        let result = launch_center_4c1e(&backend, &plan, &spec, &mut staging);
        assert!(result.is_ok(), "F32 4c1e should succeed: {:?}", result);

        let staging_f32 = bytemuck::cast_slice::<f64, f32>(&staging);
        assert!(
            staging_f32[0].is_finite(),
            "F32 4c1e result should be finite: {}",
            staging_f32[0]
        );
    }

    #[test]
    fn test_fill_4c1e_g_tensor_ssss() {
        // s-s-s-s: nmax=0, mmax=0, shape.g_size=1, nroots=1
        let shape = build_4c1e_shape(0, 0, 0, 0);
        assert_eq!(shape.nroots, 1);
        assert_eq!(shape.g_size, 1);

        let mut g = vec![0.0_f64; 3 * shape.g_size];
        let ri = [0.0_f64; 3];
        let rj = [0.0_f64; 3];
        let rk = [1.0_f64, 0.0, 0.0];
        let rl = [1.0_f64, 0.0, 0.0];
        let rij = [0.0_f64; 3];
        let rkl = [1.0_f64, 0.0, 0.0];
        let aij = 2.0;
        let akl = 2.0;
        let fac = 1.5_f64;

        fill_4c1e_g_tensor(&mut g, &shape, ri, rj, rk, rl, rij, rkl, aij, akl, fac);

        let aijkl = aij + akl;
        let expected_gz = fac / (aijkl * aijkl.sqrt());
        assert!(
            (g[0] - 1.0).abs() < 1e-14,
            "gx[0] should be 1.0, got {}",
            g[0]
        );
        assert!(
            (g[1] - 1.0).abs() < 1e-14,
            "gy[0] should be 1.0, got {}",
            g[1]
        );
        assert!(
            (g[2] - expected_gz).abs() < 1e-14,
            "gz[0] should be {expected_gz}, got {}",
            g[2]
        );
    }

    #[test]
    fn test_build_4c1e_shape_nroots_one() {
        for li in 0..=2 {
            for lj in 0..=2 {
                for lk in 0..=2 {
                    for ll in 0..=2 {
                        let shape = build_4c1e_shape(li, lj, lk, ll);
                        assert_eq!(
                            shape.nroots, 1,
                            "nroots must be 1 for 4c1e, got {} for li={li} lj={lj} lk={lk} ll={ll}",
                            shape.nroots
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_spinor_rejected_first() {
        let err = validated_4c1e_error("spinor representation not supported for 4c1e");
        match &err {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("spinor"),
                    "Error message must contain 'spinor', got: {requested}"
                );
            }
            _ => panic!("Expected UnsupportedApi error"),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Device kernel cross-check: the CubeCL kernel (on CpuRuntime, f64) must
    // reproduce the host `fill_4c1e_g_tensor` + branch-selected `hrr_*_4d` +
    // `contract_4c1e_cart` reference for a single primitive quartet.
    // ─────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "cpu")]
    mod device_cross_check {
        use super::super::*;

        fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
            cubecl::cpu::CpuRuntime::client(&Default::default())
        }

        /// Host reference: single-primitive single-contraction shell quartet,
        /// reproducing exactly the device kernel's per-primitive arithmetic.
        #[allow(clippy::too_many_arguments)]
        fn host_cart_4c1e(
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
        ) -> Vec<f64> {
            let shape = build_4c1e_shape(li as usize, lj as usize, lk as usize, ll as usize);

            let aij = ai + aj;
            let akl = ak + al;
            let aijkl = aij + akl;

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

            let dx_ij = ri[0] - rj[0];
            let dy_ij = ri[1] - rj[1];
            let dz_ij = ri[2] - rj[2];
            let rr_ij = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
            let fac_ij = f64::exp(-ai * aj / aij * rr_ij);

            let dx_kl = rk[0] - rl[0];
            let dy_kl = rk[1] - rl[1];
            let dz_kl = rk[2] - rl[2];
            let rr_kl = dx_kl * dx_kl + dy_kl * dy_kl + dz_kl * dz_kl;
            let fac_kl = f64::exp(-ak * al / akl * rr_kl);

            let a0 = aij * akl / aijkl;
            let dx_ijkl = rij[0] - rkl[0];
            let dy_ijkl = rij[1] - rkl[1];
            let dz_ijkl = rij[2] - rkl[2];
            let rr_ijkl = dx_ijkl * dx_ijkl + dy_ijkl * dy_ijkl + dz_ijkl * dz_ijkl;
            let fac_ijkl = f64::exp(-a0 * rr_ijkl);

            let quartet_fac = common_factor * fac_ij * fac_kl * fac_ijkl;

            let mut g = vec![0.0_f64; 3 * shape.g_size];
            fill_4c1e_g_tensor(
                &mut g,
                &shape,
                ri,
                rj,
                rk,
                rl,
                rij,
                rkl,
                aij,
                akl,
                quartet_fac,
            );

            let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            let rkrl = [rk[0] - rl[0], rk[1] - rl[1], rk[2] - rl[2]];

            if shape.kbase {
                if shape.ibase {
                    hrr_ik2d_4d(&mut g, &shape, rirj, rkrl);
                } else {
                    hrr_kj2d_4d(&mut g, &shape, rirj, rkrl);
                }
            } else if shape.ibase {
                hrr_il2d_4d(&mut g, &shape, rirj, rkrl);
            } else {
                hrr_lj2d_4d(&mut g, &shape, rirj, rkrl);
            }

            contract_4c1e_cart(&g, &shape, li, lj, lk, ll)
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
            // Distinct, spread-out centers so the cross-pair exponential does
            // not underflow to all-zeros.
            let ri = [0.0_f64, 0.0, 0.0];
            let rj = [0.7_f64, -0.4, 0.2];
            let rk = [-0.5_f64, 0.9, 0.6];
            let rl = [0.3_f64, 0.5, -0.8];
            let common_factor = SQRTPI
                * std::f64::consts::PI
                * common_fac_sp(li)
                * common_fac_sp(lj)
                * common_fac_sp(lk)
                * common_fac_sp(ll);

            let host = host_cart_4c1e(
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
            );
            // Task 35-D wave 5: the per-quartet entry point is now a one-row
            // batch through the same kernel, so this unit test covers the
            // batched path rather than a second implementation of it.
            let dev = run_4c1e_contracted_on_backend(
                &ResolvedBackend::Cpu(cpu_client()),
                [li as u32, lj as u32, lk as u32, ll as u32],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [ri, rj, rk, rl],
                common_factor,
                [&[ai], &[aj], &[ak], &[al]],
                [&[1.0], &[1.0], &[1.0], &[1.0]],
            );

            assert_eq!(
                host.len(),
                dev.len(),
                "length mismatch for li={li} lj={lj} lk={lk} ll={ll}"
            );
            let mut any_nonzero = false;
            for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
                if h.abs() > 1e-18 {
                    any_nonzero = true;
                }
                let diff = (h - d).abs();
                let thr = 1e-12 + 1e-10 * h.abs();
                assert!(
                    diff <= thr,
                    "device/host mismatch li={li} lj={lj} lk={lk} ll={ll} idx={idx}: \
                     host={h:.15e} dev={d:.15e} diff={diff:.3e}"
                );
            }
            assert!(
                any_nonzero,
                "host reference produced all zeros for li={li} lj={lj} lk={lk} ll={ll} — \
                 exponents/geometry underflowed"
            );
        }

        #[test]
        fn test_device_matches_host_ssss() {
            assert_device_matches_host(0, 0, 0, 0, 1.0, 1.2, 0.8, 0.9);
        }

        #[test]
        fn test_device_matches_host_psss() {
            // li>0: exercises ibase + i-HRR branch (hrr_il2d_4d).
            assert_device_matches_host(1, 0, 0, 0, 0.9, 1.1, 0.7, 1.3);
        }

        #[test]
        fn test_device_matches_host_sssp() {
            // ll>0, kbase=false: exercises hrr_lj2d_4d l-HRR branch.
            assert_device_matches_host(0, 0, 0, 1, 0.8, 1.0, 0.6, 1.2);
        }

        #[test]
        fn test_device_matches_host_spsp() {
            // lj>0 (ibase=false) + ll>0 (kbase=false): hrr_lj2d_4d mixed.
            assert_device_matches_host(0, 1, 0, 1, 1.1, 0.7, 0.9, 0.8);
        }

        #[test]
        fn test_device_matches_host_ppss() {
            // li==lj (ibase=false, nmax=2): ij pair both nonzero.
            assert_device_matches_host(1, 1, 0, 0, 0.6, 0.9, 1.2, 1.0);
        }

        #[test]
        fn test_device_matches_host_pssp() {
            // li>0 (ibase=true) + ll>0 (kbase=false): hrr_il2d_4d.
            assert_device_matches_host(1, 0, 0, 1, 1.0, 0.8, 1.1, 0.7);
        }

        /// Genericity evidence: the kernel compiles and runs for `F = f32`.
        /// Launch a one-row s-s-s-s batch on the CPU runtime and assert the
        /// result is finite.
        #[test]
        fn test_center_4c1e_kernel_generic_f32() {
            let client = cpu_client();
            // s-s-s-s: nmax=0, mmax=0 → dli=dlj=dlk=dll=1, g_size=1, buf db=1
            // bigger=0 → buf_size = 1*(0+1)=1.
            //
            // Four s shells, one primitive and one contraction each, flattened
            // the way `TwoEFlatBasis` lays a basis out.
            let exps = [1.0_f32; 4];
            let coeffs = [1.0_f32; 4];
            let centers = [
                0.0_f32, 0.0, 0.0, 0.7, -0.4, 0.2, -0.5, 0.9, 0.6, 0.3, 0.5, -0.8,
            ];
            let shell_meta: [u32; 16] = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 3, 3, 1, 1];
            // `si, sj, sk, sl, out_off, class, ci, cj, ck, cl`.
            let rows: [u32; 10] = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0];
            let class_shape: [u32; 4] = [0, 0, 0, 0];

            let common_factor = (SQRTPI
                * std::f64::consts::PI
                * common_fac_sp(0)
                * common_fac_sp(0)
                * common_fac_sp(0)
                * common_fac_sp(0)) as f32;
            let class_factor = [common_factor];

            let g_zero = [0.0_f32; 3];
            let buf_zero = [0.0_f32; 1];
            let out_zero = [0.0_f32; 1];

            let exps_h = client.create_from_slice(f32::as_bytes(&exps));
            let coeffs_h = client.create_from_slice(f32::as_bytes(&coeffs));
            let centers_h = client.create_from_slice(f32::as_bytes(&centers));
            let meta_h = client.create_from_slice(u32::as_bytes(&shell_meta));
            let rows_h = client.create_from_slice(u32::as_bytes(&rows));
            let shape_h = client.create_from_slice(u32::as_bytes(&class_shape));
            let factor_h = client.create_from_slice(f32::as_bytes(&class_factor));
            let g_h = client.create_from_slice(f32::as_bytes(&g_zero));
            let buf_h = client.create_from_slice(f32::as_bytes(&buf_zero));
            let out_h = client.create_from_slice(f32::as_bytes(&out_zero));

            center_4c1e_kernel::launch::<f32, cubecl::cpu::CpuRuntime>(
                &client,
                crate::plane::cube_count_1d(1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(exps_h, exps.len()) },
                unsafe { ArrayArg::from_raw_parts(coeffs_h, coeffs.len()) },
                unsafe { ArrayArg::from_raw_parts(centers_h, centers.len()) },
                unsafe { ArrayArg::from_raw_parts(meta_h, shell_meta.len()) },
                unsafe { ArrayArg::from_raw_parts(rows_h, rows.len()) },
                unsafe { ArrayArg::from_raw_parts(shape_h, class_shape.len()) },
                unsafe { ArrayArg::from_raw_parts(factor_h, class_factor.len()) },
                unsafe { ArrayArg::from_raw_parts(g_h, 3) },
                unsafe { ArrayArg::from_raw_parts(buf_h, 1) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), 1) },
                1u32, // n_rows
                1u32, // n_cubes
                3u32, // g_stride (one slab, unpadded)
                1u32, // buf_stride
                1u32, // per_unit
            );

            let raw = client.read_one_unchecked(out_h);
            let out = f32::from_bytes(&raw)[0];
            assert!(
                out.is_finite(),
                "f32 4c1e kernel result must be finite: {out}"
            );
            assert!(
                out > 0.0,
                "s-s-s-s 4c1e f32 result should be positive: {out}"
            );
        }
    }
}
